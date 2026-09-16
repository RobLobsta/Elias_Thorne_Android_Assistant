package ai.eliasthorne.assistant

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Where the weights live on the device.
 *
 * The GGUF is ~1.2 GB, so it is never inside the APK: Play caps a delivered
 * app far below that, and a 1.2 GB asset would be copied out of the archive on
 * first run and cost the space twice. Instead the file sits in the app's own
 * files directory, put there either by importing from device storage or by a
 * one-time download the user starts.
 *
 * Keeping it in filesDir matters for more than tidiness: llama.cpp mmaps it, so
 * the pages stay file-backed and evictable rather than counting as dirty
 * anonymous memory. That is what makes a 2B model viable on a 4 GB phone.
 */
class ModelStore(private val context: Context) {

    val modelFile: File get() = File(context.filesDir, MODEL_NAME)

    val isInstalled: Boolean get() = modelFile.length() > MIN_PLAUSIBLE_BYTES

    fun installedBytes(): Long = if (modelFile.isFile) modelFile.length() else 0L

    /**
     * Copy a user-picked GGUF into place.
     *
     * Written to a temporary name and renamed on success, so an interrupted
     * import cannot leave a half-file that looks installed.
     */
    suspend fun importFrom(uri: Uri, onProgress: (copied: Long, total: Long) -> Unit): String? =
        withContext(Dispatchers.IO) {
            val resolver = context.contentResolver
            val total = resolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
            val temp = File(context.filesDir, "$MODEL_NAME.part")
            temp.delete()

            try {
                resolver.openInputStream(uri).use { input ->
                    if (input == null) return@withContext "could not open the selected file"
                    temp.outputStream().use { output ->
                        val buffer = ByteArray(1 shl 20)
                        var copied = 0L
                        while (true) {
                            val n = input.read(buffer)
                            if (n < 0) break
                            output.write(buffer, 0, n)
                            copied += n
                            onProgress(copied, total)
                        }
                    }
                }
            } catch (e: Exception) {
                temp.delete()
                Log.e(TAG, "import failed", e)
                return@withContext e.message ?: "the import failed"
            }

            if (temp.length() <= MIN_PLAUSIBLE_BYTES) {
                temp.delete()
                return@withContext "that file is too small to be a model"
            }
            if (!isGguf(temp)) {
                temp.delete()
                return@withContext "that file is not a GGUF model"
            }

            modelFile.delete()
            if (!temp.renameTo(modelFile)) {
                temp.delete()
                return@withContext "could not move the model into place"
            }
            null
        }

    fun delete() {
        modelFile.delete()
    }

    /** Cheap sanity check so a wrong pick fails here rather than inside llama.cpp. */
    private fun isGguf(file: File): Boolean = try {
        file.inputStream().use { stream ->
            val magic = ByteArray(4)
            stream.read(magic) == 4 && magic.decodeToString() == "GGUF"
        }
    } catch (e: Exception) {
        false
    }

    companion object {
        private const val TAG = "elias"
        const val MODEL_NAME = "model.gguf"

        /** Below this it cannot be a real model, whatever the extension says. */
        private const val MIN_PLAUSIBLE_BYTES = 64L * 1024 * 1024
    }
}
