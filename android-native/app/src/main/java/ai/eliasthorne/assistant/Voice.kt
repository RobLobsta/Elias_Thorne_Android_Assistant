package ai.eliasthorne.assistant

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

/**
 * Speech in and out, on the platform's own engines.
 *
 * Android's SpeechRecognizer is usually network-backed, so this is the one part
 * of the app that may not work with the radio off — the model does, the ears may
 * not. [offlinePreferred] asks for on-device recognition where the device has
 * it; text input is always available as the fallback that definitely works.
 *
 * Both engines are main-thread only. Callbacks arrive on the main thread too.
 */
class Voice(private val context: Context) {

    interface Listener {
        /** Interim text, for the caption. */
        fun onPartial(text: String)
        /** A finished utterance. */
        fun onFinal(text: String)
        /** Recognition stopped, with a reason when it failed. */
        fun onListeningStopped(error: String?)
        /** Synthesis finished, so the mic can safely reopen. */
        fun onSpeakingFinished()
    }

    var listener: Listener? = null

    /**
     * TextToSpeech delivers UtteranceProgressListener callbacks on a binder
     * thread, and SpeechRecognizer is main-thread-only. Everything is hopped to
     * the main thread here so a listener can touch views without the exception
     * being thrown inside a framework callback, where it is swallowed and the UI
     * is simply left stuck.
     */
    private val main = Handler(Looper.getMainLooper())

    private fun post(block: () -> Unit) = main.post(block)

    private var recognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    val isRecognitionAvailable: Boolean
        get() = SpeechRecognizer.isRecognitionAvailable(context)

    fun start() {
        tts = TextToSpeech(context) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.language = Locale.getDefault()
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        post { listener?.onSpeakingFinished() }
                    }
                    @Deprecated("platform signature")
                    override fun onError(utteranceId: String?) {
                        post { listener?.onSpeakingFinished() }
                    }
                })
            } else {
                Log.w(TAG, "no speech synthesiser; replies will be text only")
            }
        }
    }

    fun listen() {
        if (!isRecognitionAvailable) {
            listener?.onListeningStopped("speech recognition is unavailable on this device")
            return
        }
        stopListening()

        val r = SpeechRecognizer.createSpeechRecognizer(context)
        recognizer = r
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}

            override fun onPartialResults(partial: Bundle?) {
                partial?.firstResult()?.let { listener?.onPartial(it) }
            }

            override fun onResults(results: Bundle?) {
                val text = results?.firstResult()
                if (text.isNullOrBlank()) {
                    listener?.onListeningStopped(null)
                } else {
                    listener?.onFinal(text)
                }
            }

            override fun onError(code: Int) {
                listener?.onListeningStopped(describe(code))
            }

            override fun onEvent(type: Int, params: Bundle?) {}
        })

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            // Honoured from API 23 where the device has an on-device model; a
            // device without one falls back to the network recogniser.
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, offlinePreferred)
        }
        r.startListening(intent)
    }

    fun stopListening() {
        recognizer?.apply {
            stopListening()
            destroy()
        }
        recognizer = null
    }

    fun speak(text: String) {
        if (!ttsReady || text.isBlank()) {
            post { listener?.onSpeakingFinished() }
            return
        }
        // QUEUE_FLUSH so a barge-in cuts the previous sentence rather than
        // queueing behind it.
        val result = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
        // A refused utterance produces no callback at all — a device with no
        // voice data returns ERROR here and then stays silent forever.
        if (result != TextToSpeech.SUCCESS) {
            Log.w(TAG, "speak() refused (code $result); continuing without audio")
            post { listener?.onSpeakingFinished() }
        }
    }

    fun stopSpeaking() {
        tts?.stop()
    }

    fun shutdown() {
        stopListening()
        tts?.stop()
        tts?.shutdown()
        tts = null
    }

    private fun Bundle.firstResult(): String? =
        getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()

    private fun describe(code: Int): String? = when (code) {
        // Not errors worth showing: the user simply did not say anything.
        SpeechRecognizer.ERROR_NO_MATCH,
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> null
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "microphone permission was refused"
        SpeechRecognizer.ERROR_NETWORK,
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT ->
            "the speech recogniser needs a network; type instead, or install offline speech"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "the recogniser is busy"
        else -> "speech recognition failed ($code)"
    }

    companion object {
        private const val TAG = "elias"
        private const val UTTERANCE_ID = "elias-reply"

        /** Ask for on-device recognition. The platform may ignore it. */
        var offlinePreferred = true
    }
}
