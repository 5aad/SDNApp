// SegmentationProcessor.kt
package com.example.sdnapp.segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.widget.ImageView
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import android.util.Log
import android.widget.Toast
import com.example.sdnapp.utils.ClassDataLoader
import com.example.sdnapp.utils.YuvToRgbConverter
import java.nio.FloatBuffer
import java.util.concurrent.Executors

/**
 * Encapsulates segmentation and depth logic.
 *
 * NOTE: Buffers & bitmaps are now pre-allocated and re-used each frame.
 */
class SegmentationProcessor(
    private val context: Context,
    private val imageView: ImageView
) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val segSession: OrtSession
    private val depthSession: OrtSession
    private val yuvConverter = YuvToRgbConverter(context)
    private lateinit var classColorMap: Map<Int, Int>
    private lateinit var classNameMap: Map<Int, String>

    private val analysisExecutor =
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    private var lastAnalysisTime = 0L
    private val analysisInterval = 1000L

    /* ----------- New reusable members ------------ */
    // For 640×800 segmentation model
    private val segInputBuffer = FloatArray(3 * 640 * 800)
    // For 256×256 depth model
    private val depthInputBuffer = FloatArray(3 * 256 * 256)
    // Camera RGB frame
    private var reusableBitmap: Bitmap? = null
    // Overlay bitmap (mask blended with original)
    private var overlayBitmapCache: Bitmap? = null
    /* -------------------------------------------- */

    init {
        // Load class colors and names
        val (colorMap, nameMap) =
            ClassDataLoader.loadClassData(filename = "class_colors.csv", context = context)
        classColorMap = colorMap
        classNameMap = nameMap

        val modelBytes = context.assets.open("model.onnx").readBytes()
        val soSeg = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
            setInterOpNumThreads(Runtime.getRuntime().availableProcessors())
            addNnapi()
            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
        }
        segSession = ortEnv.createSession(modelBytes, soSeg)

        val depthModelBytes = context.assets.open("midas_small.onnx").readBytes()
        depthSession = ortEnv.createSession(
            depthModelBytes,
            OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                setInterOpNumThreads(Runtime.getRuntime().availableProcessors())
                addNnapi()
            }
        )
    }

    /* --------------- Unchanged public API ------------- */

    fun startCamera(activity: androidx.activity.ComponentActivity) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastAnalysisTime >= analysisInterval) {
                    lastAnalysisTime = currentTime
                    processImageProxy(imageProxy)
                } else {
                    imageProxy.close()
                }
            }

            cameraProvider.bindToLifecycle(
                activity,
                CameraSelector.DEFAULT_BACK_CAMERA,
                imageAnalysis
            )
        }, ContextCompat.getMainExecutor(context))
    }

    /* --------------- Internal helpers ------------- */

    private fun processImageProxy(imageProxy: ImageProxy) {
        // Re-use / resize camera bitmap if needed
        if (reusableBitmap == null ||
            reusableBitmap!!.width != imageProxy.width ||
            reusableBitmap!!.height != imageProxy.height
        ) {
            reusableBitmap =
                Bitmap.createBitmap(imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888)
        }

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        yuvConverter.yuvToRgb(imageProxy, reusableBitmap!!)
        imageProxy.close()

        val processedBitmap =
            if (rotationDegrees == 0) reusableBitmap!!
            else rotateBitmap(reusableBitmap!!, rotationDegrees.toFloat())

        imageView.post { processCameraImage(processedBitmap) }
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(angle) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun processCameraImage(originalBitmap: Bitmap) {
        val mutableOriginal = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val resizedBitmap = Bitmap.createScaledBitmap(mutableOriginal, 640, 800, true)

        val inputData = preprocessImage(resizedBitmap)
        val tensorShape = longArrayOf(1, 3, 640, 800)
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputData), tensorShape)

        val results = segSession.run(mapOf("input" to tensor))
        @Suppress("UNCHECKED_CAST")
        val logits = results[0].value as Array<Array<Array<FloatArray>>>
        val segMap = argmaxSegmentation(logits[0])
        val segMapResized =
            resizeSegmentationMap(segMap, mutableOriginal.width, mutableOriginal.height)

        val overlayBitmap = overlaySegmentation(mutableOriginal, segMapResized)
        imageView.setImageBitmap(overlayBitmap)

        val depthMap = estimateDepth(mutableOriginal)
        val nearestClassId = calculateNearestClass(segMapResized, depthMap)
        val nearestName = classNameMap[nearestClassId] ?: "unknown"
        Toast.makeText(context, "Nearest: $nearestName", Toast.LENGTH_SHORT).show()
    }

    /* ------- Pre-processing now fills reusable arrays ------- */

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Use (and resize if needed) the reusable buffer
        val required = width * height * 3
        val floatArray =
            if (segInputBuffer.size == required) segInputBuffer else FloatArray(required)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in 0 until height) {
            for (j in 0 until width) {
                val index = i * width + j
                val pixel = pixels[index]
                val r = (Color.red(pixel) / 255.0f - mean[0]) / std[0]
                val g = (Color.green(pixel) / 255.0f - mean[1]) / std[1]
                val b = (Color.blue(pixel) / 255.0f - mean[2]) / std[2]
                floatArray[index] = r
                floatArray[index + width * height] = g
                floatArray[index + 2 * width * height] = b
            }
        }
        return floatArray
    }

    private fun preprocessDepthImage(bitmap: Bitmap, targetW: Int, targetH: Int): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)

        // Re-use buffer sized for 3×256×256
        val floatArray = depthInputBuffer
        val pixels = IntArray(targetW * targetH)
        resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        for (i in 0 until targetH) {
            for (j in 0 until targetW) {
                val idx = i * targetW + j
                val pixel = pixels[idx]
                val r = ((Color.red(pixel) / 255.0f) - 0.5f) / 0.5f
                val g = ((Color.green(pixel) / 255.0f) - 0.5f) / 0.5f
                val b = ((Color.blue(pixel) / 255.0f) - 0.5f) / 0.5f
                floatArray[idx] = r
                floatArray[idx + targetW * targetH] = g
                floatArray[idx + 2 * targetW * targetH] = b
            }
        }
        return floatArray
    }

    /* ------- Overlay now re-uses a cached bitmap ------- */

    private fun overlaySegmentation(
        original: Bitmap,
        segMap: Array<IntArray>
    ): Bitmap {
        val width = original.width
        val height = original.height

        if (overlayBitmapCache == null ||
            overlayBitmapCache!!.width != width ||
            overlayBitmapCache!!.height != height
        ) {
            overlayBitmapCache =
                Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        }
        val overlay = overlayBitmapCache!!

        for (y in 0 until height) {
            for (x in 0 until width) {
                val clsId = segMap[y][x]
                val segColor = classColorMap[clsId] ?: Color.BLACK
                val origColor = original.getPixel(x, y)
                overlay.setPixel(x, y, blendPixels(origColor, segColor, 0.4f))
            }
        }
        return overlay
    }

    /* ------- Remaining methods are unchanged ------- */

    private fun argmaxSegmentation(logits: Array<Array<FloatArray>>): Array<IntArray> {
        val numClasses = logits.size
        val height = logits[0].size
        val width = logits[0][0].size
        val segMap = Array(height) { IntArray(width) }
        for (i in 0 until height) {
            for (j in 0 until width) {
                var maxVal = Float.NEGATIVE_INFINITY
                var maxIndex = 0
                for (c in 0 until numClasses) {
                    val value = logits[c][i][j]
                    if (value > maxVal) {
                        maxVal = value
                        maxIndex = c
                    }
                }
                segMap[i][j] = maxIndex
            }
        }
        return segMap
    }

    private fun resizeSegmentationMap(segMap: Array<IntArray>, targetW: Int, targetH: Int): Array<IntArray> {
        val srcH = segMap.size
        val srcW = segMap[0].size
        val resized = Array(targetH) { IntArray(targetW) }
        for (i in 0 until targetH) {
            val srcY = (i * srcH) / targetH
            for (j in 0 until targetW) {
                val srcX = (j * srcW) / targetW
                resized[i][j] = segMap[srcY][srcX]
            }
        }
        return resized
    }

    private fun blendPixels(orig: Int, overlay: Int, alpha: Float): Int {
        val r = ((Color.red(orig) * (1 - alpha)) + (Color.red(overlay) * alpha)).toInt()
        val g = ((Color.green(orig) * (1 - alpha)) + (Color.green(overlay) * alpha)).toInt()
        val b = ((Color.blue(orig) * (1 - alpha)) + (Color.blue(overlay) * alpha)).toInt()
        return Color.rgb(r, g, b)
    }

    private fun estimateDepth(bitmap: Bitmap): Array<FloatArray> {
        val targetW = 256
        val targetH = 256
        val depthInput = preprocessDepthImage(bitmap, targetW, targetH)
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(depthInput), longArrayOf(1, 3, targetH.toLong(), targetW.toLong()))
        val inputName = depthSession.inputNames.first()
        @Suppress("UNCHECKED_CAST")
        val depthOutput = depthSession.run(mapOf(inputName to tensor))[0].value as Array<Array<FloatArray>>
        return resizeDepthMap(depthOutput[0], bitmap.width, bitmap.height)
    }

    private fun resizeDepthMap(depthMap: Array<FloatArray>, targetW: Int, targetH: Int): Array<FloatArray> {
        val srcH = depthMap.size
        val srcW = depthMap[0].size
        val resized = Array(targetH) { FloatArray(targetW) }
        for (i in 0 until targetH) {
            val srcY = (i * srcH) / targetH
            for (j in 0 until targetW) {
                val srcX = (j * srcW) / targetW
                resized[i][j] = depthMap[srcY][srcX]
            }
        }
        return resized
    }

    private fun calculateNearestClass(segMap: Array<IntArray>, depthMap: Array<FloatArray>): Int {
        val minDepthPerClass = mutableMapOf<Int, Float>()
        val height = segMap.size
        val width = segMap[0].size
        for (y in 0 until height) {
            for (x in 0 until width) {
                val cls = segMap[y][x]
                if (cls == 0) continue
                val depthVal = depthMap[y][x]
                if (!minDepthPerClass.containsKey(cls) || depthVal < minDepthPerClass[cls]!!) {
                    minDepthPerClass[cls] = depthVal
                }
            }
        }
        return minDepthPerClass.minByOrNull { it.value }?.key ?: 0
    }

}
