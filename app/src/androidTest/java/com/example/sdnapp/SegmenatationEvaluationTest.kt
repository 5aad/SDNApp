package com.example.sdnapp

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SegmentationEvaluatorTest {
    @Test
    fun evaluateOnValidationSet_printsIoU() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val assetManager = context.assets
        val env = OrtEnvironment.getEnvironment()
        val modelBytes = assetManager.open("model.onnx").readBytes()
        val session = env.createSession(modelBytes, OrtSession.SessionOptions())
        val classColors = TestActivity.loadClassColorsFromCsv("class_colors.csv")

        SegmentationEvaluator.evaluateOnValidationSet(
            assetManager,
            session,
            env,
            classColors,
            imageFolder = "images/val",
            maskFolder  = "masks/val"
        )
        // Check Logcat for output, or adapt evaluateOnValidationSet to return the FloatArray for assertions.
    }
}

