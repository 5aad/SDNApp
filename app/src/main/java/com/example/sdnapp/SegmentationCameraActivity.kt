package com.example.sdnapp

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.speech.tts.TextToSpeech
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.IOException
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors

// Helper class to convert a YUV ImageProxy to a Bitmap using RenderScript.
class YuvToRgbConverter(val context: Context) {
    private val rs: RenderScript = RenderScript.create(context)
    private val scriptYuvToRgb: ScriptIntrinsicYuvToRGB =
        ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    fun yuvToRgb(image: ImageProxy, output: Bitmap) {
        // Convert ImageProxy to NV21 byte array.
        val nv21 = yuv420ToNv21(image)
        val yuvType = android.renderscript.Type.Builder(rs, Element.U8(rs))
            .setX(nv21.size).create()
        val inAllocation = Allocation.createTyped(rs, yuvType)
        inAllocation.copyFrom(nv21)
        val bitmapAllocation = Allocation.createFromBitmap(rs, output)
        scriptYuvToRgb.setInput(inAllocation)
        scriptYuvToRgb.forEach(bitmapAllocation)
        bitmapAllocation.copyTo(output)
    }

    // Basic conversion from YUV_420_888 to NV21. (May need adjustments on some devices.)
    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        // Interleave V and U.
        var index = ySize
        val chromaRowStride = image.planes[1].rowStride
        val chromaPixelStride = image.planes[1].pixelStride
        val uBufferPos = uBuffer.position()
        val vBufferPos = vBuffer.position()
        val chromaWidth = image.width / 2
        val chromaHeight = image.height / 2

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val uIndex = uBufferPos + row * chromaRowStride + col * chromaPixelStride
                val vIndex = vBufferPos + row * chromaRowStride + col * chromaPixelStride
                nv21[index++] = vBuffer.get(vIndex)
                nv21[index++] = uBuffer.get(uIndex)
            }
        }
        return nv21
    }
}

class SegmentationCameraActivity : ComponentActivity(), TextToSpeech.OnInitListener {

    // ONNX Runtime session variable for segmentation.
    private lateinit var ortSession: OrtSession

    // ONNX Runtime session variable for depth estimation.
    private lateinit var depthSession: OrtSession

    // ImageView to display the segmentation overlay.
    private lateinit var imageView: ImageView

    // 1. Map from class label → packed Color int
    private lateinit var classColorMap: Map<Int, Int>
    private lateinit var classNameMap: Map<Int, String>


    // Use a fixed thread pool based on available processors for parallel processing.
    private val analysisExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())

    // Variables to throttle processing (capture one frame every 2.5 seconds).
    private var lastAnalysisTime = 0L
    private val analysisInterval = 2000L

    // YUV to Bitmap converter.
    private lateinit var yuvToRgbConverter: YuvToRgbConverter

    // TextToSpeech instance.
    private lateinit var textToSpeech: TextToSpeech


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // activity_main.xml must include an ImageView with id "imageView".
        setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.imageView)
        yuvToRgbConverter = YuvToRgbConverter(this)

        // 0. Load class colors & names from CSV
        val (colorMap, nameMap) = loadClassData("class_colors.csv")
        classColorMap = colorMap
        classNameMap  = nameMap

        // Initialize TextToSpeech.
        textToSpeech = TextToSpeech(this, this)

        // Request CAMERA permission if not already granted.
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 0)
        } else {
            startCamera()
        }

        try {
            // Initialize segmentation ONNX session.
            val env = OrtEnvironment.getEnvironment()
            val modelBytes = assets.open("model.onnx").readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                setInterOpNumThreads(Runtime.getRuntime().availableProcessors())
                // If your ONNX Runtime build supports NNAPI, you can enable it (uncomment below):
                addNnapi()
            }
            ortSession = env.createSession(modelBytes, sessionOptions)
        } catch (e: IOException) {
            e.printStackTrace()
        }

        try {
            // Initialize depth estimation ONNX session using midas_small.onnx.
            val env = OrtEnvironment.getEnvironment()
            val depthModelBytes = assets.open("midas_small.onnx").readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                setInterOpNumThreads(Runtime.getRuntime().availableProcessors())
                // Uncomment below if NNAPI is supported.
                addNnapi()
            }
            depthSession = env.createSession(depthModelBytes, sessionOptions)

        } catch (e: IOException) {
            e.printStackTrace()
        }

    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech.language = Locale.US
        }
    }

    // Start CameraX and bind the ImageAnalysis use case.
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
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

            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }

    // Convert the ImageProxy to Bitmap, fix orientation, and process.
    private fun processImageProxy(imageProxy: ImageProxy) {
        // Create a bitmap from the imageProxy dimensions.
        var bitmap = Bitmap.createBitmap(
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        yuvToRgbConverter.yuvToRgb(imageProxy, bitmap)
        // Get the rotation degrees from the ImageProxy.
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        // Rotate the bitmap if necessary.
        if (rotationDegrees != 0) {
            bitmap = rotateBitmap(bitmap, rotationDegrees.toFloat())
        }
        imageProxy.close()
        runOnUiThread {
            processCameraImage(bitmap)
        }
    }

    // Helper function to rotate a bitmap by the given angle.
    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(angle) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    // Process the Bitmap: resize, run segmentation & depth estimation, overlay result, and output nearest class.
    private fun processCameraImage(originalBitmap: Bitmap) {
        val mutableOriginal = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        // Resize to model input dimensions for segmentation (800x640).
        val resizedBitmap = Bitmap.createScaledBitmap(mutableOriginal, 800, 640, true)
        val inputData = preprocessImage(resizedBitmap)

        // Create tensor with shape [1, 3, 640, 800].
        val shape = longArrayOf(1, 3, 640, 800)
        val env = OrtEnvironment.getEnvironment()
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)

        // Run the segmentation ONNX inference.
        val results = ortSession.run(mapOf("input" to tensor))
        @Suppress("UNCHECKED_CAST")
        val logits = results[0].value as Array<Array<Array<FloatArray>>>
        val segMap = argmaxSegmentation(logits[0])

        // Resize segmentation map back to original dimensions.
        val segMapResized = resizeSegmentationMap(segMap, mutableOriginal.width, mutableOriginal.height)
        val overlayBitmap = overlaySegmentation(mutableOriginal, segMapResized)
        imageView.setImageBitmap(overlayBitmap)

        // Estimate depth from the original image.
        val depthMap = estimateDepth(mutableOriginal)

        // Compute nearest class (ignoring background) based on depth.
        val nearestClassId = calculateNearestClass(segMapResized, depthMap)

        // Display the result using Toast and TextToSpeech.
        val nearestName = classNameMap[nearestClassId] ?: "unknown"
        Toast.makeText(this, "Nearest: $nearestName", Toast.LENGTH_SHORT).show()
        textToSpeech.speak("Nearest: $nearestName", TextToSpeech.QUEUE_FLUSH, null, null)
    }

    // Preprocess the bitmap to a normalized float array in CHW format for segmentation.
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val channels = 3
        val floatArray = FloatArray(width * height * channels)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
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

    // Perform argmax on the logits to get the segmentation map.
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

    // Resize a segmentation map using nearest neighbor interpolation.
    private fun resizeSegmentationMap(segMap: Array<IntArray>, targetWidth: Int, targetHeight: Int): Array<IntArray> {
        val srcHeight = segMap.size
        val srcWidth = segMap[0].size
        val resized = Array(targetHeight) { IntArray(targetWidth) }
        for (i in 0 until targetHeight) {
            val srcY = (i * srcHeight) / targetHeight
            for (j in 0 until targetWidth) {
                val srcX = (j * srcWidth) / targetWidth
                resized[i][j] = segMap[srcY][srcX]
            }
        }
        return resized
    }

    /**
     * Reads "class_color.csv" and returns:
     *  - Map<label, Color-int>
     *  - Map<label, className>
     */
    private fun loadClassData(filename: String): Pair<Map<Int,Int>, Map<Int,String>> {
        val colors = mutableMapOf<Int, Int>()
        val names  = mutableMapOf<Int, String>()
        assets.open(filename).bufferedReader().useLines { lines ->
            lines.drop(1)
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .forEach { line ->
                    val parts = line.split(",")
                    val label = parts[0].toInt()
                    val r     = parts[1].toInt()
                    val g     = parts[2].toInt()
                    val b     = parts[3].toInt()
                    val clsName = parts[4]
                    colors[label] = Color.rgb(r, g, b)
                    names[label]  = clsName
                }
        }
        return colors to names
    }

    // Overlay segmentation on the original image.
    private fun overlaySegmentation(original: Bitmap, segMap: Array<IntArray>): Bitmap {
        val width  = original.width
        val height = original.height
        val overlay = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val clsId = segMap[y][x]
                // lookup color; default to black if missing
                val segColor = classColorMap[clsId] ?: Color.BLACK
                val origColor = original.getPixel(x, y)
                val blended = blendPixels(origColor, segColor, 0.4f)
                overlay.setPixel(x, y, blended)
            }
        }
        return overlay
    }

    // Blend two colors with the specified alpha value.
    private fun blendPixels(orig: Int, overlay: Int, alpha: Float): Int {
        val r = ((Color.red(orig) * (1 - alpha)) + (Color.red(overlay) * alpha)).toInt()
        val g = ((Color.green(orig) * (1 - alpha)) + (Color.green(overlay) * alpha)).toInt()
        val b = ((Color.blue(orig) * (1 - alpha)) + (Color.blue(overlay) * alpha)).toInt()
        return Color.rgb(r, g, b)
    }

    // Estimate depth from the given bitmap using the midas_small model.
// Estimate depth from the given bitmap using the midas_small model.
    private fun estimateDepth(bitmap: Bitmap): Array<FloatArray> {
        // Define the target resolution for the depth model (adjust if necessary).
        val targetWidth = 256
        val targetHeight = 256
        val depthInput = preprocessDepthImage(bitmap, targetWidth, targetHeight)
        val env = OrtEnvironment.getEnvironment()
        val shape = longArrayOf(1, 3, targetHeight.toLong(), targetWidth.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(depthInput), shape)

        // NOTE: Change the key from "input" to "0" to match the model's expected input name.


        val inputName = depthSession.inputNames.first()
        Log.d("DepthModel", "Using input name: $inputName")

        // Run inference using the retrieved input name.
        val results = depthSession.run(mapOf(inputName to tensor))
        Log.d("DepthModel", "Using input name: ${results[0].value}")
        @Suppress("UNCHECKED_CAST")
        // Assuming the model outputs a tensor of shape [1, 1, targetHeight, targetWidth]
        val depthOutput = results[0].value as Array<Array<FloatArray>>
        val depthArray = depthOutput[0]  // now a 2D array: [targetHeight][targetWidth]
        Log.d("DepthModel", "First depth value: $depthOutput")
        // Resize depth map to original image dimensions.
        return resizeDepthMap(depthArray, bitmap.width, bitmap.height)
    }


    // Preprocess the bitmap for the depth model: convert to CHW normalized float array.
    private fun preprocessDepthImage(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        val width = targetWidth
        val height = targetHeight
        val channels = 3
        val floatArray = FloatArray(width * height * channels)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)
        // Normalize pixels to range [-1, 1]: ((pixel/255) - 0.5)/0.5.
        for (i in 0 until height) {
            for (j in 0 until width) {
                val index = i * width + j
                val pixel = pixels[index]
                val r = ((Color.red(pixel) / 255.0f) - 0.5f) / 0.5f
                val g = ((Color.green(pixel) / 255.0f) - 0.5f) / 0.5f
                val b = ((Color.blue(pixel) / 255.0f) - 0.5f) / 0.5f
                floatArray[index] = r
                floatArray[index + width * height] = g
                floatArray[index + 2 * width * height] = b
            }
        }
        return floatArray
    }

    // Resize a 2D float depth map using nearest-neighbor interpolation.
    private fun resizeDepthMap(depthMap: Array<FloatArray>, targetWidth: Int, targetHeight: Int): Array<FloatArray> {
        val srcHeight = depthMap.size
        val srcWidth = depthMap[0].size
        val resized = Array(targetHeight) { FloatArray(targetWidth) }
        for (i in 0 until targetHeight) {
            val srcY = (i * srcHeight) / targetHeight
            for (j in 0 until targetWidth) {
                val srcX = (j * srcWidth) / targetWidth
                resized[i][j] = depthMap[srcY][srcX]
            }
        }
        return resized
    }

    // Calculate the nearest segmentation class (ignoring background) using the segmentation and depth maps.
    private fun calculateNearestClass(segMap: Array<IntArray>, depthMap: Array<FloatArray>): Int {
        val minDepthPerClass = mutableMapOf<Int, Float>()
        val height = segMap.size
        val width = segMap[0].size
        for (y in 0 until height) {
            for (x in 0 until width) {
                val cls = segMap[y][x]
                if (cls == 0) continue // Skip background.
                val depthVal = depthMap[y][x]
                if (!minDepthPerClass.containsKey(cls) || depthVal < minDepthPerClass[cls]!!) {
                    minDepthPerClass[cls] = depthVal
                }
            }
        }
        var nearestClass = 0
        var minDepth = Float.MAX_VALUE
        for ((cls, depth) in minDepthPerClass) {
            if (depth < minDepth) {
                minDepth = depth
                nearestClass = cls
            }
        }
        return nearestClass
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        textToSpeech.stop()
        textToSpeech.shutdown()
        // Optionally, release ONNX sessions if needed.
    }
}
