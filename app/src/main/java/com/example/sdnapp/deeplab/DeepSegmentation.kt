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
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.nio.FloatBuffer
import kotlin.system.measureTimeMillis
import androidx.core.graphics.createBitmap

/**
 * Real‑time semantic segmentation using an **ONNX** model (`model.onnx`).
 *
 *  * **Input**: 640×800 RGB (NCHW)
 *  * **Classes**: 10 (see `class_color.csv`)
 *  * **Output**: Mask bitmap + per‑frame centre‑line offset so that the UI can warn the user
 */
class DeepSegmentation(assets: AssetManager) : ImageAnalysis.Analyzer {

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
    }

    /*──────────────────────────────────────────────────────────────────────*/
    /*  Result DTO                                                          */
    /*──────────────────────────────────────────────────────────────────────*/
    data class SegmentationResults(
        val bitmapMask: Bitmap?,           // overlay with class colours + centre line
        val seenObjects: String,           // text summary of objects detected
        val centerOffsetPx: Int            // + => user/camera is left of centre‑line, − => right
    )

    /*──────────────────────────────────────────────────────────────────────*/
    /*  ONNX Runtime setup                                                  */
    /*──────────────────────────────────────────────────────────────────────*/
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

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
                val out = session.run(mapOf(inputName to inputTensor))
                val logits = (out[0].value as Array<Array<Array<FloatArray>>>)[0] // [C][H][W]

                /*── arg‑max + build colour mask ─────────────────────────*/
                val maskBmp = createBitmap(W, H)
                val canvas = Canvas(maskBmp)
                val pnt = Paint().apply { strokeWidth = 1f }

                for (y in 0 until H) {
                    for (x in 0 until W) {
                        var best = 0; var bestScore = Float.NEGATIVE_INFINITY
                        for (c in 0 until NUM_CLASSES) {
                            val s = logits[c][y][x]
                            if (s > bestScore) { bestScore = s; best = c }
                        }
                        segmentBits[x][y] = best
                        pnt.color = segmentColors[best]
                        canvas.drawPoint(x.toFloat(), y.toFloat(), pnt)
                    }
                }

                /*── compute centre‑line of sidewalk (bottom ¼ of frame) ─*/
                var acc = 0f; var validRows = 0
                for (y in (H * 0.75).toInt() until H) {
                    var sumX = 0; var cnt = 0
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

                /*── emit result ───────────────────────────────────────*/
                resultNotifier.onNext(
                    SegmentationResults(
                        tfResizeBilinear(maskBmp, srcH, srcW, 0),
                        seenStr, offsetPx
                    )
                )

                out.close(); inputTensor.close()
            }
            Log.d(TAG, "Frame segmented in ${'$'}infMs ms")
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
