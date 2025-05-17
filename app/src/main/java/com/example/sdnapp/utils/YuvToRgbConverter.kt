// YuvToRgbConverter.kt
package com.example.sdnapp.utils

import android.content.Context
import android.graphics.Bitmap
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import androidx.camera.core.ImageProxy

/**
 * Helper to convert YUV ImageProxy to RGB Bitmap using RenderScript.
 *
 * NOTE: Re-uses buffers & Allocations to reduce GC pressure.
 */
class YuvToRgbConverter(val context: Context) {

    private val rs: RenderScript = RenderScript.create(context)
    private val scriptYuvToRgb: ScriptIntrinsicYuvToRGB =
        ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    /* ----------- New reusable members ------------ */
    private var yuvBuffer: ByteArray = ByteArray(0)
    private var yuvType: android.renderscript.Type? = null
    private var inAllocation: Allocation? = null
    /* -------------------------------------------- */

    fun yuvToRgb(image: ImageProxy, output: Bitmap) {
        val nv21 = yuv420ToNv21(image)

        // (Re-)create Allocation only if the image size changed
        if (inAllocation == null || yuvType == null || yuvType!!.x != nv21.size) {
            yuvType = android.renderscript.Type.Builder(rs, Element.U8(rs))
                .setX(nv21.size)
                .create()
            inAllocation = Allocation.createTyped(rs, yuvType)
        }

        inAllocation!!.copyFrom(nv21)

        val bitmapAllocation = Allocation.createFromBitmap(rs, output)
        scriptYuvToRgb.setInput(inAllocation)
        scriptYuvToRgb.forEach(bitmapAllocation)
        bitmapAllocation.copyTo(output)
    }

    /** Fills (and re-uses) a shared NV21 buffer instead of allocating a new one each frame. */
    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val totalSize = ySize + uSize + vSize

        // Ensure capacity only when needed
        if (yuvBuffer.size != totalSize) {
            yuvBuffer = ByteArray(totalSize)
        }

        yBuffer.get(yuvBuffer, 0, ySize)
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
                yuvBuffer[index++] = vBuffer.get(vIndex)
                yuvBuffer[index++] = uBuffer.get(uIndex)
            }
        }
        return yuvBuffer
    }
}
