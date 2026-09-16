// JNI surface for LlamaBridge.kt. All the thinking is in elias_core; this only
// marshals, so the generation path the phone runs is the one the host harness
// exercises.

#include "elias_core.h"

#include <android/log.h>
#include <jni.h>

#include <string>
#include <vector>

#define LOG_TAG "elias-native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::string to_utf8(JNIEnv * env, jstring s) {
    if (!s) return {};
    const char * chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(s, chars);
    return out;
}

/** Holds everything one loaded model needs, including the Java callback glue. */
struct Session {
    elias::Model * model = nullptr;
    elias::Params  params;
};

Session * as_session(jlong handle) { return reinterpret_cast<Session *>(handle); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_eliasthorne_assistant_LlamaBridge_nativeLoad(JNIEnv * env, jobject /*thiz*/, jstring path,
                                                     jint n_threads, jint n_ctx, jobject error_out) {
    const std::string gguf = to_utf8(env, path);

    auto * session = new Session();
    session->params.n_threads = n_threads > 0 ? n_threads : 4;
    session->params.n_ctx     = n_ctx > 0 ? n_ctx : 2048;

    std::string error;
    LOGI("loading %s (threads=%d ctx=%d)", gguf.c_str(), session->params.n_threads,
         session->params.n_ctx);
    try {
        session->model = elias::Model::load(gguf, session->params, error);
    } catch (const std::exception & e) {
        error = e.what();
        session->model = nullptr;
    } catch (...) {
        error = "the model loader threw an unknown exception";
        session->model = nullptr;
    }

    if (!session->model) {
        LOGE("load failed: %s", error.c_str());
        // error_out is a String[1] so the failure reaches Kotlin as a message
        // rather than as a bare null handle.
        if (error_out) {
            auto arr = static_cast<jobjectArray>(error_out);
            env->SetObjectArrayElement(arr, 0, env->NewStringUTF(error.c_str()));
        }
        delete session;
        return 0;
    }
    LOGI("loaded; context %d tokens", session->model->context_size());
    return reinterpret_cast<jlong>(session);
}

JNIEXPORT void JNICALL
Java_ai_eliasthorne_assistant_LlamaBridge_nativeFree(JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    if (auto * s = as_session(handle)) {
        delete s->model;
        delete s;
    }
}

JNIEXPORT void JNICALL
Java_ai_eliasthorne_assistant_LlamaBridge_nativeStop(JNIEnv * /*env*/, jobject /*thiz*/, jlong handle) {
    if (auto * s = as_session(handle)) s->model->request_stop();
}

/**
 * Generate a reply.
 *
 * `roles` and `contents` are parallel String[]s holding the conversation so
 * far. Tokens are handed back through `listener.onToken(String)`; returning
 * false from it stops generation at the next token, which is how the UI's stop
 * button and a barge-in both work.
 */
JNIEXPORT jstring JNICALL
Java_ai_eliasthorne_assistant_LlamaBridge_nativeGenerate(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                                         jobjectArray roles, jobjectArray contents,
                                                         jint max_tokens, jfloat temperature,
                                                         jobject listener) {
    auto * s = as_session(handle);
    if (!s || !roles || !contents) return env->NewStringUTF("");

    std::vector<elias::Message> messages;
    const jsize n = env->GetArrayLength(roles);
    if (n != env->GetArrayLength(contents)) {
        LOGE("roles/contents length mismatch");
        return env->NewStringUTF("");
    }
    messages.reserve(n);
    for (jsize i = 0; i < n; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto text = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
        messages.push_back({to_utf8(env, role), to_utf8(env, text)});
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(text);
    }

    elias::Params params = s->params;
    if (max_tokens > 0)  params.n_predict   = max_tokens;
    if (temperature >= 0) params.temperature = temperature;

    jmethodID on_token = nullptr;
    if (listener) {
        jclass cls = env->GetObjectClass(listener);
        on_token = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)Z");
        env->DeleteLocalRef(cls);
    }

    std::string error;
    std::string reply;
    try {
        reply = s->model->generate(messages, params,
            [&](const std::string & piece) -> bool {
                if (!on_token) return true;
                // elias_core only ever hands over complete UTF-8; were that not
                // so, NewStringUTF would abort the whole process rather than
                // fail, so it is worth not relying on luck.
                jstring jpiece = env->NewStringUTF(piece.c_str());
                if (!jpiece) {
                    env->ExceptionClear();
                    return false;
                }
                const jboolean keep_going = env->CallBooleanMethod(listener, on_token, jpiece);
                env->DeleteLocalRef(jpiece);
                // An exception thrown by the listener must not be swallowed into
                // a half-finished generation.
                if (env->ExceptionCheck()) {
                    env->ExceptionDescribe();
                    env->ExceptionClear();
                    return false;
                }
                return keep_going == JNI_TRUE;
            }, error);
    } catch (const std::exception & e) {
        // A C++ exception unwinding through a JNI frame terminates the process.
        LOGE("generate threw: %s", e.what());
        return env->NewStringUTF("");
    } catch (...) {
        LOGE("generate threw an unknown exception");
        return env->NewStringUTF("");
    }

    if (!error.empty()) LOGE("generate: %s", error.c_str());
    jstring out = env->NewStringUTF(reply.c_str());
    return out ? out : env->NewStringUTF("");
}

}  // extern "C"
