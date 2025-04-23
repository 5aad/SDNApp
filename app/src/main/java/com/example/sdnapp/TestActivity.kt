package com.example.sdnapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.widget.ImageView
import androidx.activity.ComponentActivity
import java.io.IOException
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class TestActivity : ComponentActivity() {
    private lateinit var classColorMap: Map<Int, Int>
    // ONNX Runtime session variable
    private lateinit var ortSession: OrtSession

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // activity_main.xml contains an ImageView with id "imageView"
        setContentView(R.layout.activity_test)
        classColorMap = loadClassColorsFromCsv("class_colors.csv")


        try {
            // ------------------------------
            // 1. Load the ONNX model
            // ------------------------------
            val env = OrtEnvironment.getEnvironment()
            // Load model bytes from assets ("model.onnx")
            val modelBytes = assets.open("model.onnx").readBytes()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                setInterOpNumThreads(Runtime.getRuntime().availableProcessors())
                // If your ONNX Runtime build supports NNAPI, you can enable it (uncomment below):
                addNnapi()
            }
            ortSession = env.createSession(modelBytes, sessionOptions)

            // ------------------------------
            // 2. Preprocess the input image
            // ------------------------------
            // Load image from assets ("img2.jpeg")
            val originalBitmap = BitmapFactory.decodeStream(assets.open("img2.jpeg"))
            // Create a mutable copy if needed
            val mutableOriginal = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
            // Resize image to model input dimensions (width=800, height=640)
            val resizedBitmap = Bitmap.createScaledBitmap(mutableOriginal, 800, 640, true)

            // Convert bitmap to a float array with normalization (mean-std)
            val inputData = preprocessImage(resizedBitmap)

            // Create input tensor with shape [1, 3, 640, 800]
            val shape = longArrayOf(1, 3, 640, 800)
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)

            // ------------------------------
            // 3. Run inference
            // ------------------------------
            val results = ortSession.run(mapOf("input" to tensor))
            // Assuming the model outputs logits with shape [1, numClasses, 640, 800]
            val outputTensor = results[0].value
            @Suppress("UNCHECKED_CAST")
            val logits = outputTensor as Array<Array<Array<FloatArray>>>
            // Remove the batch dimension and apply argmax over channels to create segmentation map
            val segMap = argmaxSegmentation(logits[0]) // shape: [height, width]

            // ------------------------------
            // 4. Post-process the segmentation map
            // ------------------------------
            // Resize segmentation map back to original image dimensions using nearest neighbor interpolation
            val segMapResized = resizeSegmentationMap(segMap, mutableOriginal.width, mutableOriginal.height)

            // (Optional) Additional filtering (e.g., median filtering) can be added here if needed.

            // ------------------------------
            // 5. Overlay segmentation on original image
            // ------------------------------
            val overlayBitmap = overlaySegmentation(mutableOriginal, segMapResized)

            // Display the final overlay result in an ImageView
            val imageView: ImageView = findViewById(R.id.imageView)
            imageView.setImageBitmap(overlayBitmap)

            SegmentationEvaluator.evaluateOnValidationSet(
                assets,
                ortSession,
                env,
                classColorMap,
                imageFolder = "validation/images",
                maskFolder  = "validation/masks"
            )

        } catch (e: IOException) {
            e.printStackTrace()
        }


    }

    // Preprocess the bitmap into a float array with shape [3, height, width] and normalize.
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val channels = 3
        val floatArray = FloatArray(width * height * channels)
        // Mean and standard deviation values (RGB order)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Convert from HWC to CHW format and apply normalization
        for (i in 0 until height) {
            for (j in 0 until width) {
                val index = i * width + j
                val pixel = pixels[index]
                // Extract R, G, B components, normalize and assign in CHW order.
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

    // Apply argmax over the channel dimension for a 3D tensor (numClasses, height, width)
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

    // Resize a 2D segmentation map using nearest neighbor interpolation.
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

    // Overlay the segmentation map on the original image by blending colors.
    private fun overlaySegmentation(
        original: Bitmap,
        segMap: Array<IntArray>
    ): Bitmap {
        val width = original.width
        val height = original.height
        val overlay = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val clsId = segMap[y][x]
                // Look up the color; defaults to black if missing
                val segColor = classColorMap[clsId] ?: Color.BLACK
                val origColor = original.getPixel(x, y)
                val blended = blendPixels(origColor, segColor, 0.4f)
                overlay.setPixel(x, y, blended)
            }
        }
        return overlay
    }

    private fun loadClassColorsFromCsv(filename: String): Map<Int, Int> {
        val map = mutableMapOf<Int, Int>()
        assets.open(filename).bufferedReader().useLines { lines ->
            lines.drop(1)  // skip header
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .forEach { line ->
                    // Expect: class_label,red,green,blue,Class_Names
                    val parts = line.split(",")
                    val label = parts[0].toInt()
                    val r = parts[1].toInt()
                    val g = parts[2].toInt()
                    val b = parts[3].toInt()
                    map[label] = Color.rgb(r, g, b)
                }
        }
        return map
    }

    // Blends two colors with a given alpha for the overlay color.
    private fun blendPixels(orig: Int, overlay: Int, alpha: Float): Int {
        val r = ((Color.red(orig) * (1 - alpha)) + (Color.red(overlay) * alpha)).toInt()
        val g = ((Color.green(orig) * (1 - alpha)) + (Color.green(overlay) * alpha)).toInt()
        val b = ((Color.blue(orig) * (1 - alpha)) + (Color.blue(overlay) * alpha)).toInt()
        return Color.rgb(r, g, b)
    }

    companion object
}