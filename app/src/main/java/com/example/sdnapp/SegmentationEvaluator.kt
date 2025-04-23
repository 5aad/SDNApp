package com.example.sdnapp

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import androidx.core.graphics.scale
import androidx.core.graphics.get

object SegmentationEvaluator {

    /** Load an RGB‐encoded mask from assets and return [H][W] class IDs. */
    fun loadMaskAsClassMap(
        assets: AssetManager,
        maskPath: String,
        targetW: Int,
        targetH: Int
    ): Array<IntArray> {
        val bmp = BitmapFactory.decodeStream(assets.open(maskPath))
        val resized = bmp.scale(targetW, targetH, false)
        return Array(targetH) { y ->
            IntArray(targetW) { x ->
                Color.red(resized[x, y])
            }
        }
    }

    /**
     * Compute per‑class IoU between prediction and truth maps.
     * @param pred  [H][W] predicted class IDs
     * @param truth [H][W] ground‑truth class IDs
     * @param numClasses total number of classes
     * @return FloatArray of length numClasses
     */
    fun computeIoU(
        pred: Array<IntArray>,
        truth: Array<IntArray>,
        numClasses: Int
    ): FloatArray {
        val h = pred.size
        val w = pred[0].size
        val iou = FloatArray(numClasses)

        for (c in 0 until numClasses) {
            var inter = 0
            var union = 0
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val p = pred[y][x]
                    val t = truth[y][x]
                    if (p == c && t == c) inter++
                    if (p == c || t == c) union++
                }
            }
            iou[c] = if (union > 0) inter.toFloat() / union else 0f
        }
        return iou
    }

    /** Simple average over all classes. */
    fun meanIoU(iouPerClass: FloatArray): Float =
        iouPerClass.sum() / iouPerClass.size


    /**
     * Evaluate all files in your validation folder.
     * Assumes images and masks share file names.
     */
    fun evaluateOnValidationSet(
        assets: AssetManager,
        session: OrtSession,
        env: OrtEnvironment,
        classColorMap: Map<Int, Int>,
        imageFolder: String = "validation/images",
        maskFolder:  String = "validation/masks"
    ) {
        val fileNames = assets.list(imageFolder)?.take(20) ?: return
        val numClasses = classColorMap.size

        // accumulators
        val sumIoU = FloatArray(numClasses) { 0f }
        var imageCount = 0

        for (name in fileNames) {
            // 1) load & preprocess image
            val bmp0 = BitmapFactory.decodeStream(assets.open("$imageFolder/$name"))
            val bmp = bmp0.scale(800, 640)
            val inputData = preprocessImage(bmp)
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData),
                longArrayOf(1, 3, 640, 800))

            // 2) inference → segMap [H][W]
            val result = session.run(mapOf("input" to tensor))[0].value as
                    Array<Array<Array<FloatArray>>>
            val segMap = argmaxSegmentation(result[0])
            val segResized = resizeSegmentationMap(segMap,
                bmp0.width, bmp0.height)

            // 3) load ground-truth
            val baseName = name.substringBeforeLast('.')        // “foo” from “foo.jpg”
            val maskName = "$baseName.png"
            val truth = loadMaskAsClassMap(assets,
                "$maskFolder/$maskName",
                bmp0.width, bmp0.height)

            // 4) compute IoU
            val iouPerClass = computeIoU(segResized, truth, numClasses)
            for (c in 0 until numClasses) {
                sumIoU[c] += iouPerClass[c]
            }
            imageCount++
        }

        // report average IoU per class and mean IoU
        val avgIoU = sumIoU.map { it / imageCount }.toFloatArray()
        avgIoU.forEachIndexed { c, v ->
            println("Class $c avg IoU = ${"%.3f".format(v)}")
        }
        println("Mean IoU = ${"%.3f".format(meanIoU(avgIoU))}")
    }

    // (Copy your existing preprocessImage / argmaxSegmentation / resizeSegmentationMap here)
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
}
