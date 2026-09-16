package ai.eliasthorne.assistant

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Kotlin side of the native inference core.
 *
 * Everything here is blocking and must be called off the main thread — a token
 * takes tens of milliseconds and a prompt eval takes seconds. [load] and
 * [generate] are suspend functions on [Dispatchers.Default] for that reason.
 */
class LlamaBridge {

    /** A conversation turn. Roles are "system", "user" and "assistant". */
    data class Message(val role: String, val content: String)

    /** Receives generated text as it decodes. Return false to stop. */
    fun interface TokenListener {
        fun onToken(piece: String): Boolean
    }

    @Volatile private var handle: Long = 0L

    val isLoaded: Boolean get() = handle != 0L

    /**
     * Load a GGUF.
     *
     * @return null on success, or a human-readable reason on failure.
     */
    suspend fun load(model: File, threads: Int, contextTokens: Int): String? =
        withContext(Dispatchers.Default) {
            if (handle != 0L) return@withContext null
            if (!model.isFile) return@withContext "no model file at ${model.absolutePath}"

            val error = arrayOfNulls<String>(1)
            val h = nativeLoad(model.absolutePath, threads, contextTokens, error)
            if (h == 0L) {
                return@withContext error[0] ?: "the model could not be loaded"
            }
            handle = h
            Log.i(TAG, "model loaded: ${model.name}")
            null
        }

    /**
     * Generate a reply to [messages], streaming pieces to [listener].
     *
     * The listener is called on the inference thread, not the main thread.
     */
    suspend fun generate(
        messages: List<Message>,
        maxTokens: Int,
        temperature: Float,
        listener: TokenListener,
    ): String = withContext(Dispatchers.Default) {
        val h = handle
        if (h == 0L) return@withContext ""
        nativeGenerate(
            h,
            messages.map { it.role }.toTypedArray(),
            messages.map { it.content }.toTypedArray(),
            maxTokens,
            temperature,
            listener,
        )
    }

    /** Ask an in-flight [generate] to stop. Safe from any thread. */
    fun stop() {
        val h = handle
        if (h != 0L) nativeStop(h)
    }

    fun close() {
        val h = handle
        handle = 0L
        if (h != 0L) nativeFree(h)
    }

    private external fun nativeLoad(
        path: String,
        threads: Int,
        contextTokens: Int,
        errorOut: Array<String?>,
    ): Long

    private external fun nativeGenerate(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        maxTokens: Int,
        temperature: Float,
        listener: TokenListener,
    ): String

    private external fun nativeStop(handle: Long)
    private external fun nativeFree(handle: Long)

    companion object {
        private const val TAG = "elias"

        init {
            System.loadLibrary("elias")
        }
    }
}
