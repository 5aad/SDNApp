package com.example.sdnapp.deeplab

import android.content.res.AssetManager
import android.graphics.*
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import io.reactivex.BackpressureStrategy
import io.reactivex.Flowable
import io.reactivex.subjects.PublishSubject
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtLoggingLevel
import android.os.SystemClock
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.nio.FloatBuffer
import kotlin.system.measureTimeMillis
import androidx.core.graphics.createBitmap
import java.util.Locale

class DeepSegmentation(
    assets: AssetManager,
    private val focalLenMm: Float,
    private val sensorH_mm: Float
) : ImageAnalysis.Analyzer {

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Constants                                                           */
    /*──────────────────────────────────────────────────────────────────────*/
    companion object {
        private const val IMAGE_MEAN = 128
        private const val IMAGE_STD = 128f
        private const val H = 640        // input height
        private const val W = 800        // input width
        private const val C = 3          // RGB
        private const val NUM_CLASSES = 10
        private const val MODEL_FILENAME = "model.onnx"
        private const val CLASS_COLOR_FILE = "class_colors.csv"
        private const val SIDEWALK_ID = 2 // from CSV (row 2)
        private const val TAG = "DeepSegmentation"

        /* ── target IDs we want distances for ────────────────────────── */
        private const val ID_CYCLE = 6   // "bicycle"
        private const val ID_PERSON = 4   // "person" – example row, adjust to CSV

        /** real‑world heights (mm) per target class */
        private val REAL_HEIGHTS = mapOf(
            ID_CYCLE to 800f,   // adult bike from tyre to handlebar
            ID_PERSON to 1_700f   // average standing adult
        )
        private const val WHITE_DOT_RADIUS = 4f      // px in mask bitmap

        private const val LAT_WIN = 1                   // ★ NEW
    }

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Result DTO                                                          */
    /*──────────────────────────────────────────────────────────────────────*/
    data class SegmentationResults(
        val bitmapMask: Bitmap?,           // overlay with class colours + centre line
        val seenObjects: String,           // text summary of objects detected
        val centerOffsetPx: Int,            // + => user/camera is left of centre‑line, − => right
        val distancesMm: Map<String, Float>,   // label → distance (mm)
        val latencyMs: Float                             // ★ NEW
    )

    /*──────────────────────────────────────────────────────────────────────*/
    /*  ONNX Runtime setup                                                  */
    /*──────────────────────────────────────────────────────────────────────*/
    private val env: OrtEnvironment =
        OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, "Ort")
    private val session: OrtSession

    /*──────────────── rolling-average state ──────────────────────────────*/
    private var accTimeMs = 0L                          // ★ NEW
    private var accFrames = 0                           // ★ NEW
    @Volatile
    var avgLatency = 0f                        // ★ NEW  (readable from UI)

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Buffers & look‑ups                                                  */
    /*──────────────────────────────────────────────────────────────────────*/
    private val imgData = FloatArray(1 * C * H * W)
    private val segmentBits = Array(W) { IntArray(H) }
    private val segmentColors = IntArray(NUM_CLASSES)
    private val labelMap = mutableMapOf<Int, String>()
    private val resultNotifier = PublishSubject.create<SegmentationResults>()

    init {
        /*── load ONNX model ───────────────────────────────────────────────*/
        val bytes = assets.open(MODEL_FILENAME).readBytes()
        val so = OrtSession.SessionOptions().apply {
            addNnapi()
        }
        session = env.createSession(bytes)

        /*── read CSV for colours / labels ─────────────────────────────────*/
        BufferedReader(InputStreamReader(assets.open(CLASS_COLOR_FILE))).use { br ->
            br.readLine() // header
            br.lineSequence().filter { it.isNotBlank() }.forEach { line ->
                val (idStr, rStr, gStr, bStr, name) = line.split(',')
                val id = idStr.toInt()
                if (id in 0 until NUM_CLASSES) {
                    segmentColors[id] = Color.argb(120, rStr.toInt(), gStr.toInt(), bStr.toInt())
                    labelMap[id] = name.trim()
                }
            }
        }
        if (segmentColors[0] == 0) segmentColors[0] = Color.TRANSPARENT // background
    }

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Public stream                                                       */
    /*──────────────────────────────────────────────────────────────────────*/
    fun resultsObserver(): Flowable<SegmentationResults> =
        resultNotifier.toFlowable(BackpressureStrategy.LATEST)

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Analyzer implementation                                             */
    /*──────────────────────────────────────────────────────────────────────*/
    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {

        try {
            val srcW = image.width
            val srcH = image.height

            /*── convert Camera frame (YUV) → Bitmap at model size ─────────*/
            val jpgBytes = NV21toJPEG(DeepUtils.YUV420toNV21(image.image), srcW, srcH)
            val bmpInput = tfResizeBilinear(
                BitmapFactory.decodeByteArray(jpgBytes, 0, jpgBytes.size),
                W, H, image.imageInfo.rotationDegrees
            ) ?: return

            /*── fill imgData[] in NCHW order ───────────────────────────────*/
            val pixels = IntArray(W * H)
            bmpInput.getPixels(pixels, 0, W, 0, 0, W, H)
            bmpInput.recycle()
            var i = 0
            for (y in 0 until H) for (x in 0 until W) {
                val p = pixels[y * W + x]
                imgData[i++] = (((p shr 16) and 0xFF) - IMAGE_MEAN) / IMAGE_STD // R
            }
            for (y in 0 until H) for (x in 0 until W) {
                val p = pixels[y * W + x]
                imgData[i++] = (((p shr 8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD  // G
            }
            for (y in 0 until H) for (x in 0 until W) {
                val p = pixels[y * W + x]
                imgData[i++] = (((p) and 0xFF) - IMAGE_MEAN) / IMAGE_STD        // B
            }

            /*── run inference ─────────────────────────────────────────────*/
            val inputName = session.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(imgData), longArrayOf(1, 3, H.toLong(), W.toLong())
            )
            val infMs = measureTimeMillis {
                val t0 = SystemClock.elapsedRealtimeNanos()      // ★ NEW – start stopwatch
                val out = session.run(mapOf(inputName to inputTensor))
                val logits = (out[0].value as Array<Array<Array<FloatArray>>>)[0] // [C][H][W]

                /*── arg‑max + build colour mask ─────────────────────────*/
                val maskBmp = createBitmap(W, H)
                val canvas = Canvas(maskBmp)
                val pnt = Paint().apply { strokeWidth = 1f }

//                for (y in 0 until H) {
//                    for (x in 0 until W) {
//                        var best = 0; var bestScore = Float.NEGATIVE_INFINITY
//                        for (c in 0 until NUM_CLASSES) {
//                            val s = logits[c][y][x]
//                            if (s > bestScore) { bestScore = s; best = c }
//                        }
//                        segmentBits[x][y] = best
//                        pnt.color = segmentColors[best]
//                        canvas.drawPoint(x.toFloat(), y.toFloat(), pnt)
//                    }
//                }


                // prepare bounding arrays for each target class
                val top = IntArray(NUM_CLASSES) { H }
                val bottom = IntArray(NUM_CLASSES) { -1 }
                val left = IntArray(NUM_CLASSES) { W }
                val right = IntArray(NUM_CLASSES) { -1 }

                for (y in 0 until H) {
                    for (x in 0 until W) {
                        var best = 0;
                        var bestScore = Float.NEGATIVE_INFINITY
                        for (c in 0 until NUM_CLASSES) {
                            val s = logits[c][y][x]
                            if (s > bestScore) {
                                bestScore = s; best = c
                            }
                        }
                        segmentBits[x][y] = best
                        pnt.color = segmentColors[best]
                        canvas.drawPoint(x.toFloat(), y.toFloat(), pnt)

                        if (best in REAL_HEIGHTS.keys) {
                            if (y < top[best]) top[best] = y
                            if (y > bottom[best]) bottom[best] = y
                            if (x < left[best]) left[best] = x
                            if (x > right[best]) right[best] = x
                        }
                    }
                }


                /*── 🚲  Cycle bounding‑rows & distance ───────────────*/
                val distancesMm = mutableMapOf<String, Float>()
                REAL_HEIGHTS.forEach { (id, realH) ->
                    if (bottom[id] >= top[id] && right[id] >= left[id]) {
                        val maskHpx = bottom[id] - top[id] + 1
                        val scale = image.height.toFloat() / H
                        val fullHpx = maskHpx * scale
                        val distMm = (focalLenMm * realH * image.height) /
                                (fullHpx * sensorH_mm)
                        labelMap[id]?.let { distancesMm[it] = distMm }

                        // draw centroid marker in unique colour
                        val cx = (left[id] + right[id]) / 2f
                        val cy = (top[id] + bottom[id]) / 2f
                        Paint().apply {
                            color = Color.WHITE
                            style = Paint.Style.FILL
                        }.also { canvas.drawCircle(cx, cy, WHITE_DOT_RADIUS, it) }
                    }
                }
//                var topCy = H; var bottomCy = -1
//                var leftCy = W; var rightCy = -1
//                for (y in 0 until H) {
//                    for (x in 0 until W) if (segmentBits[x][y] == CYCLE_ID) {
//                        if (y < topCy)    topCy    = y
//                        if (y > bottomCy) bottomCy = y
//                        if (x < leftCy)   leftCy   = x
//                        if (x > rightCy)  rightCy  = x
//                    }
//                }
//                val hasCycle = bottomCy >= topCy && rightCy >= leftCy
//                val cycleDistanceMm: Float? = if (hasCycle) {
//                    val maskH = bottomCy - topCy + 1           // px @ 640×800
//                    val scale = image.height.toFloat() / H
//                    val objFullH = maskH * scale               // px @ full-res
//                    (focalLenMm * REAL_CYCLE_H_MM * image.height) /
//                            (objFullH * sensorH_mm)
//                } else null

                /*── white dot at cycle centroid ───────────────────────*/
//                if (hasCycle) {
//                    val cx = (leftCy + rightCy) / 2f
//                    val cy = (topCy + bottomCy) / 2f
//                    Paint().apply {
//                        color = Color.WHITE
//                        style = Paint.Style.FILL
//                    }.also { canvas.drawCircle(cx, cy, WHITE_DOT_RADIUS, it) }
//                }

                /*── compute centre‑line of sidewalk (bottom ¼ of frame) ─*/
                var acc = 0f;
                var validRows = 0
                for (y in (H * 0.75).toInt() until H) {
                    var sumX = 0;
                    var cnt = 0
                    for (x in 0 until W) if (segmentBits[x][y] == SIDEWALK_ID) {
                        sumX += x; cnt++
                    }
                    if (cnt > 10) { // ignore too‑sparse rows
                        acc += sumX.toFloat() / cnt
                        validRows++
                    }
                }
                val centreX = if (validRows > 0) acc / validRows else W / 2f
                val offsetPx = (W / 2f - centreX).toInt() // + if camera left of line

                /*── draw centre‑line in mask ────────────────────────────*/
                Paint().apply {
                    color = Color.YELLOW; strokeWidth = 3f
                }.also { canvas.drawLine(centreX, 0f, centreX, H.toFloat(), it) }

                /*── build seen‑objects string ───────────────────────────*/
                val seen = mutableMapOf<Int, Int>()
                for (id in 1 until NUM_CLASSES) {
                    var cnt = 0
                    for (y in 0 until H) for (x in 0 until W)
                        if (segmentBits[x][y] == id) cnt++
                    if (cnt > 20) seen[id] = cnt
                }
                val seenStr = seen.entries.sortedByDescending { it.value }
                    .joinToString(", ") { labelMap[it.key] ?: "id=${'$'}{it.key}" }


                /* --------- 4. LATENCY rolling average ---------- */
                val singleMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000f
                accTimeMs += singleMs.toLong()
                accFrames += 1
                if (accFrames == LAT_WIN) {
                    avgLatency = accTimeMs.toFloat() / LAT_WIN
                    Log.i(
                        TAG, String.format(
                            Locale.US,
                            "Avg ONNX latency = %.2f ms", avgLatency
                        )
                    )
                    accTimeMs = 0L
                    accFrames = 0
                }

                /*── emit result ───────────────────────────────────────*/
                resultNotifier.onNext(
                    SegmentationResults(
                        tfResizeBilinear(maskBmp, srcH, srcW, 0),
                        seenStr,
                        offsetPx,
                        distancesMm,
                        avgLatency                                // ★ NEW
                    )
                )
//                }
//                resultNotifier.onNext(
//                    SegmentationResults(
//                        tfResizeBilinear(maskBmp, srcH, srcW, 0),
//                        seenStr, offsetPx, null
//                    )
//                )

                out.close(); inputTensor.close()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Segmentation error", e)
        } finally {
            image.close()
        }
    }

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Helpers                                                              */
    /*──────────────────────────────────────────────────────────────────────*/
    private fun tfResizeBilinear(src: Bitmap?, w: Int, h: Int, rot: Int): Bitmap? {
        if (src == null) return null
        val m = Matrix().apply { postRotate(rot.toFloat()) }
        val tmp = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        val dst = createBitmap(w, h)
        Canvas(dst).drawBitmap(tmp, Rect(0, 0, tmp.width, tmp.height), Rect(0, 0, w, h), null)
        tmp.recycle(); src.recycle()
        return dst
    }

    private fun NV21toJPEG(nv21: ByteArray, w: Int, h: Int): ByteArray {
        val out = ByteArrayOutputStream()
        YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), 100, out)
        return out.toByteArray()
    }
}
