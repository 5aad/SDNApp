package com.example.sdnapp.deeplab
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

class DistanceAnnouncer(private val tts: TextToSpeech) {

    private var isBusy = false
    private var pendingMessage: String? = null

    init {
        tts.language = Locale.US
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String) {
                isBusy = true
            }
            override fun onDone(utteranceId: String) {
                isBusy = false
                // if there's a newer message queued, speak it now
                pendingMessage?.let { msg ->
                    pendingMessage = null
                    speakNow(msg)
                }
            }
            override fun onError(utteranceId: String) {
                isBusy = false
            }
        })
    }

    /**
     * Call this from your segmentation callback with the combined
     * distance + drift message for the current frame.
     */
    fun announce(message: String) {
        if (isBusy) {
            // replace any previous pending message with the latest
            pendingMessage = message
        } else {
            speakNow(message)
        }
    }

    /**
     * Immediately flushes the TTS queue and speaks the provided message.
     */
    private fun speakNow(msg: String) {
        tts.speak(
            msg,
            TextToSpeech.QUEUE_FLUSH,
            null,
            System.currentTimeMillis().toString()
        )
    }
}
