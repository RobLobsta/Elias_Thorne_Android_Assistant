package ai.eliasthorne.assistant

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Elias Thorne — a BitNet assistant that runs entirely on the handset.
 *
 * The loop is deliberately push-to-talk rather than the always-on wake word of
 * the web version: Android's SpeechRecognizer has to be torn down and recreated
 * for every utterance, and running it continuously drains the battery for very
 * little gain when the phone is in your hand anyway.
 */
class MainActivity : AppCompatActivity(), Voice.Listener {

    private val llama = LlamaBridge()
    private lateinit var store: ModelStore
    private lateinit var voice: Voice

    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var talkButton: Button
    private lateinit var modelButton: Button

    private val history = mutableListOf(
        LlamaBridge.Message(
            "system",
            "You are Elias Thorne, an assistant running entirely on this device. " +
                "Answer in one or two short sentences. Never invent a further question.",
        ),
    )

    private var generating: Job? = null
    private var speaking = false

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { importModel(it) }
    }

    private val askForMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startListening() else say("Microphone permission refused — type instead.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        status = findViewById(R.id.status)
        transcript = findViewById(R.id.transcript)
        input = findViewById(R.id.input)
        sendButton = findViewById(R.id.send)
        talkButton = findViewById(R.id.talk)
        modelButton = findViewById(R.id.model)
        transcript.movementMethod = ScrollingMovementMethod()

        store = ModelStore(this)
        voice = Voice(this).also { it.listener = this; it.start() }

        sendButton.setOnClickListener {
            val text = input.text.toString().trim()
            if (text.isNotEmpty()) {
                input.setText("")
                submit(text)
            }
        }
        talkButton.setOnClickListener {
            when {
                generating != null -> cancelGeneration()
                speaking -> { voice.stopSpeaking(); speaking = false; setBusy(false) }
                else -> requestMicThenListen()
            }
        }
        modelButton.setOnClickListener { pickModel.launch(arrayOf("*/*")) }

        if (store.isInstalled) loadModel() else promptForModel()
    }

    override fun onDestroy() {
        voice.shutdown()
        llama.close()
        super.onDestroy()
    }

    /* ------------------------------------------------------------- model */

    private fun promptForModel() {
        setBusy(false)
        status.text = getString(R.string.status_no_model)
        modelButton.visibility = View.VISIBLE
        say(getString(R.string.hint_no_model))
    }

    private fun importModel(uri: Uri) {
        setBusy(true)
        status.text = getString(R.string.status_importing)
        lifecycleScope.launch {
            val error = store.importFrom(uri) { copied, total ->
                val text = if (total > 0) {
                    "Importing… %d%%".format(Locale.getDefault(), (copied * 100 / total))
                } else {
                    "Importing… %s".format(Locale.getDefault(), formatBytes(copied))
                }
                runOnUiThread { status.text = text }
            }
            if (error != null) {
                status.text = getString(R.string.status_import_failed, error)
                setBusy(false)
            } else {
                loadModel()
            }
        }
    }

    private fun loadModel() {
        setBusy(true)
        modelButton.visibility = View.GONE
        status.text = getString(R.string.status_loading, formatBytes(store.installedBytes()))

        lifecycleScope.launch {
            // Leave a core for the UI and the audio pipeline; saturating every
            // core makes the app stutter without decoding meaningfully faster.
            val threads = (Runtime.getRuntime().availableProcessors() - 1).coerceIn(2, 6)
            val error = llama.load(store.modelFile, threads, CONTEXT_TOKENS)
            if (error != null) {
                status.text = getString(R.string.status_load_failed, error)
                modelButton.visibility = View.VISIBLE
                setBusy(false)
            } else {
                status.text = getString(R.string.status_ready, threads)
                setBusy(false)
                say(getString(R.string.hint_ready))
            }
        }
    }

    /* ------------------------------------------------------------- turns */

    private fun submit(text: String) {
        if (!llama.isLoaded) {
            say(getString(R.string.hint_no_model))
            return
        }
        append("You", text)
        history += LlamaBridge.Message("user", text)
        setBusy(true)
        status.text = getString(R.string.status_thinking)

        val start = System.nanoTime()
        var tokens = 0
        val reply = StringBuilder()
        append("Elias", "")

        generating = lifecycleScope.launch {
            val full = llama.generate(history, MAX_TOKENS, TEMPERATURE) { piece ->
                tokens++
                reply.append(piece)
                runOnUiThread { replaceLast("Elias", reply.toString()) }
                // The listener runs on the inference thread; returning false is
                // how the stop button lands mid-generation.
                generating?.isCancelled != true
            }
            generating = null

            val seconds = (System.nanoTime() - start) / 1e9
            val spoken = full.ifBlank { reply.toString() }.trim()
            replaceLast("Elias", spoken.ifBlank { "…" })
            status.text = getString(
                R.string.status_spoke, tokens, seconds, if (seconds > 0) tokens / seconds else 0.0,
            )
            history += LlamaBridge.Message("assistant", spoken)
            trimHistory()

            if (spoken.isNotBlank()) {
                speaking = true
                voice.speak(spoken)
            } else {
                setBusy(false)
            }
        }
    }

    private fun cancelGeneration() {
        llama.stop()
        generating?.cancel()
        generating = null
        setBusy(false)
        status.text = getString(R.string.status_stopped)
    }

    /**
     * Keep the conversation inside the context window.
     *
     * The system turn is never dropped — without it the model forgets who it is
     * and starts answering as the user.
     */
    private fun trimHistory() {
        while (history.size > 1 + MAX_TURNS * 2) history.removeAt(1)
    }

    /* ------------------------------------------------------------- voice */

    private fun requestMicThenListen() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) startListening() else askForMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun startListening() {
        if (!voice.isRecognitionAvailable) {
            say(getString(R.string.hint_no_recognizer))
            return
        }
        status.text = getString(R.string.status_listening)
        talkButton.text = getString(R.string.action_stop)
        voice.listen()
    }

    override fun onPartial(text: String) {
        status.text = getString(R.string.status_heard, text)
    }

    override fun onFinal(text: String) {
        voice.stopListening()
        talkButton.text = getString(R.string.action_talk)
        submit(text)
    }

    override fun onListeningStopped(error: String?) {
        voice.stopListening()
        talkButton.text = getString(R.string.action_talk)
        if (error != null) {
            status.text = error
            say(error)
        } else if (generating == null) {
            status.text = getString(R.string.status_ready_short)
        }
    }

    override fun onSpeakingFinished() {
        speaking = false
        setBusy(false)
    }

    /* ---------------------------------------------------------------- ui */

    private fun setBusy(busy: Boolean) {
        sendButton.isEnabled = !busy && llama.isLoaded
        input.isEnabled = !busy && llama.isLoaded
        talkButton.text = when {
            generating != null || speaking -> getString(R.string.action_stop)
            else -> getString(R.string.action_talk)
        }
        talkButton.isEnabled = llama.isLoaded
    }

    private fun append(who: String, text: String) {
        transcript.append(if (transcript.text.isEmpty()) "" else "\n\n")
        transcript.append("$who: $text")
        scrollToEnd()
    }

    /** Rewrite the last block, so streaming tokens land in one growing bubble. */
    private fun replaceLast(who: String, text: String) {
        val all = transcript.text.toString()
        val at = all.lastIndexOf("\n\n$who: ").let { if (it < 0) all.lastIndexOf("$who: ") else it }
        if (at < 0) return
        val head = all.substring(0, at)
        val separator = if (head.isEmpty()) "" else "\n\n"
        transcript.text = "$head$separator$who: $text"
        scrollToEnd()
    }

    private fun scrollToEnd() {
        val lines = transcript.lineCount
        if (lines == 0) return
        val scroll = transcript.layout?.getLineTop(lines)?.minus(transcript.height) ?: return
        transcript.scrollTo(0, scroll.coerceAtLeast(0))
    }

    private fun say(text: String) {
        status.text = text
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1L shl 30 -> "%.1f GB".format(Locale.getDefault(), bytes / (1L shl 30).toDouble())
        bytes >= 1L shl 20 -> "%.0f MB".format(Locale.getDefault(), bytes / (1L shl 20).toDouble())
        else -> "$bytes B"
    }

    companion object {
        private const val CONTEXT_TOKENS = 2048
        private const val MAX_TOKENS = 160
        private const val TEMPERATURE = 0.7f
        /** User+assistant pairs kept before the oldest is dropped. */
        private const val MAX_TURNS = 6
    }
}
