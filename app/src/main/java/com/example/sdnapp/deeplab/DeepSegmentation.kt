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
import java.io.ByteArrayOutputStream
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer
import kotlin.system.measureTimeMillis
import androidx.core.graphics.createBitmap
import androidx.core.graphics.set
const val TAG = "DeeplabAndroid"
/**
 * Segmentation analyzer that runs an **ONNX** model (`model.onnx`) whose output tensor has
 * shape **[1, 10, 640, 800]** (N × C × H × W) and whose colours / labels are defined in
 * **assets/class_color.csv**.
 */
class DeepSegmentation(assets: AssetManager) : ImageAnalysis.Analyzer {

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  Constants                                                               */
    /*───────────────────────────────────────────────────────────────────────────*/
    companion object {
        private const val IMAGE_MEAN = 128
        private const val IMAGE_STD = 128f
        private const val DIM_BATCH = 1
        private const val DIM_CHANNELS = 3
        private const val DIM_HEIGHT = 640   // model input height
        private const val DIM_WIDTH  = 800   // model input width
        private const val OUTPUT_LABELS = 10 // number of semantic classes
        private const val MODEL_FILENAME = "model.onnx"
        private const val CLASS_COLOR_FILE = "class_colors.csv"
    }

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  Data‑holder                                                             */
    /*───────────────────────────────────────────────────────────────────────────*/
    data class SegmentationResults(val bitmapMask: Bitmap?, val seenObjects: String)

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  ONNX Runtime                                                            */
    /*───────────────────────────────────────────────────────────────────────────*/
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  Buffers & lookup tables                                                 */
    /*───────────────────────────────────────────────────────────────────────────*/
    private val imgData = FloatArray(DIM_BATCH * DIM_CHANNELS * DIM_HEIGHT * DIM_WIDTH)
    private val segmentBits = Array(DIM_WIDTH) { IntArray(DIM_HEIGHT) }
    private val segmentColors = IntArray(OUTPUT_LABELS)
    private val labelMap = mutableMapOf<Int, String>()

    private val resultNotifier = PublishSubject.create<SegmentationResults>()

    init {
        /*─────────────────────────────── load ONNX model ───────────────────────*/
        val modelBytes = assets.open(MODEL_FILENAME).readBytes()
        session = env.createSession(modelBytes)

        /*─────────────────────────────── load colour CSV ───────────────────────*/
        BufferedReader(InputStreamReader(assets.open(CLASS_COLOR_FILE))).use { reader ->
            reader.readLine() // skip header
            reader.lineSequence().filter { it.isNotBlank() }.forEach { line ->
                val parts = line.split(',')
                val id = parts[0].toInt()
                val r = parts[1].toInt(); val g = parts[2].toInt(); val b = parts[3].toInt()
                val name = parts[4].trim()
                if (id in 0 until OUTPUT_LABELS) {
                    segmentColors[id] = Color.argb(120, r, g, b) // semi‑transparent
                    labelMap[id] = name
                }
            }
        }
        if (segmentColors[0] == 0) segmentColors[0] = Color.TRANSPARENT // background
    }

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  Public API                                                              */
    /*───────────────────────────────────────────────────────────────────────────*/
    fun resultsObserver(): Flowable<SegmentationResults> =
        resultNotifier.toFlowable(BackpressureStrategy.LATEST)

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  Analyzer implementation                                                 */
    /*───────────────────────────────────────────────────────────────────────────*/
    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        try {
            val srcWidth = image.width
            val srcHeight = image.height

            /*────────── convert Camera frame → Bitmap (resized to model input) ─────────*/
            val jpegBytes = NV21toJPEG(
                DeepUtils.YUV420toNV21(image.image), srcWidth, srcHeight
            )
            val inputBitmap = tfResizeBilinear(
                BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size),
                DIM_WIDTH, DIM_HEIGHT, image.imageInfo.rotationDegrees
            ) ?: return

            /*────────── prepare FloatBuffer in NCHW order ─────────*/
            val pixels = IntArray(DIM_WIDTH * DIM_HEIGHT)
            inputBitmap.getPixels(pixels, 0, DIM_WIDTH, 0, 0, DIM_WIDTH, DIM_HEIGHT)
            inputBitmap.recycle()
            var idx = 0
            // R channel
            for (y in 0 until DIM_HEIGHT) for (x in 0 until DIM_WIDTH) {
                val p = pixels[y * DIM_WIDTH + x]
                imgData[idx++] = (((p shr 16) and 0xFF) - IMAGE_MEAN) / IMAGE_STD
            }
            // G channel
            for (y in 0 until DIM_HEIGHT) for (x in 0 until DIM_WIDTH) {
                val p = pixels[y * DIM_WIDTH + x]
                imgData[idx++] = (((p shr 8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD
            }
            // B channel
            for (y in 0 until DIM_HEIGHT) for (x in 0 until DIM_WIDTH) {
                val p = pixels[y * DIM_WIDTH + x]
                imgData[idx++] = (((p) and 0xFF) - IMAGE_MEAN) / IMAGE_STD
            }

            /*────────── run inference ─────────*/
            val inputName = session.inputNames.iterator().next()
            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(imgData),
                longArrayOf(1, 3, DIM_HEIGHT.toLong(), DIM_WIDTH.toLong())
            )
            val inferenceTime = measureTimeMillis {
                val outputs = session.run(mapOf(inputName to inputTensor))
                val rawOutput = outputs[0].value as Array<Array<Array<FloatArray>>> // [1][10][640][800]
                val probs = rawOutput[0]                                           // strip batch dim → [10][640][800]

                /*────────── arg‑max over channels ─────────*/
                val maskBitmap = createBitmap(DIM_WIDTH, DIM_HEIGHT)
                val seen = mutableMapOf<Int, Int>()

                for (y in 0 until DIM_HEIGHT) {
                    for (x in 0 until DIM_WIDTH) {
                        var bestLabel = 0
                        var bestScore = Float.NEGATIVE_INFINITY
                        for (c in 0 until OUTPUT_LABELS) {
                            val score = probs[c][y][x]  // <-- fixed: Float, not FloatArray
                            if (score > bestScore) {
                                bestScore = score
                                bestLabel = c
                            }
                        }
                        segmentBits[x][y] = bestLabel
                        if (bestLabel != 0) seen[bestLabel] = (seen[bestLabel] ?: 0) + 1
                        maskBitmap[x, y] = segmentColors[bestLabel]
                    }
                }

                /*────────── build seen‑objects summary ─────────*/
                val seenObjects = seen.entries
                    .filter { it.value > 20 }
                    .sortedByDescending { it.value }
                    .joinToString(", ") { labelMap[it.key] ?: "id=${'$'}{it.key}" }

                /*────────── deliver result ─────────*/
                resultNotifier.onNext(
                    SegmentationResults(
                        tfResizeBilinear(maskBitmap, srcHeight, srcWidth, 0),
                        seenObjects
                    )
                )

                outputs.close()
                inputTensor.close()
            }
            Log.i(TAG, "ONNX segmentation in ${'$'}inferenceTime ms")
        } catch (t: Throwable) {
            Log.e(TAG, "Segmentation error", t)
        } finally {
            image.close()
        }
    }

    /*───────────────────────────────────────────────────────────────────────────*/
    /*  Helpers                                                                  */
    /*───────────────────────────────────────────────────────────────────────────*/
    private fun tfResizeBilinear(src: Bitmap?, w: Int, h: Int, rotDeg: Int): Bitmap? {
        if (src == null) return null
        val m = Matrix().apply { postRotate(rotDeg.toFloat()) }
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
