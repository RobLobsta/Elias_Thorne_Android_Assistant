// Inference core for Elias Thorne — BitNet b1.58-2B-4T through bitnet.cpp.
//
// Deliberately free of both JNI and Android: elias_jni.cpp wraps it for the app
// and test_cli.cpp drives the same code on the host, so the generation path
// that ships is the one that was tested.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct llama_model;

namespace elias {

struct Params {
    int  n_ctx        = 2048;
    int  n_threads    = 4;
    int  n_predict    = 256;
    float temperature = 0.7f;
    float top_p       = 0.9f;
    int   top_k       = 40;
    float repeat_penalty = 1.1f;
    int   repeat_last_n  = 128;
    uint32_t seed     = 0xFFFFFFFF;  // LLAMA_DEFAULT_SEED
};

/** A turn of conversation. `role` is "system", "user" or "assistant". */
struct Message {
    std::string role;
    std::string content;
};

/**
 * Render the chat prompt.
 *
 * BitNet 2B4T is NOT Llama-3 despite the token names, and the template baked
 * into Microsoft's own GGUF ("Human: ... BITNETAssistant: ") does not match
 * what the model was trained on — feeding it that produces fluent-looking
 * nonsense that loops. The real template, from the model repo's
 * tokenizer_config.json, is the capitalised role, a colon, the content, and
 * <|eot_id|>, with "Assistant: " as the generation prompt. Rendered here rather
 * than taken from the file so a wrong template in a GGUF cannot break the app.
 */
std::string render_prompt(const std::vector<Message> & messages);

/**
 * Render using the GGUF's own chat template when it declares a usable one, so
 * any standard instruct model can be dropped in; falls back to the BitNet
 * template above otherwise.
 */
std::string render_prompt(const ::llama_model * model, const std::vector<Message> & messages);

/** Called for each decoded token. Return false to stop generation. */
using TokenCallback = std::function<bool(const std::string & piece)>;

class Model {
public:
    ~Model();

    /** Returns nullptr and fills `error` when the model cannot be loaded. */
    static Model * load(const std::string & gguf_path, const Params & params, std::string & error);

    /**
     * Generate a reply to the conversation so far.
     *
     * Blocking; the caller runs it off the UI thread. `cb` receives text as it
     * decodes, and returning false from it stops cleanly at the next token.
     */
    std::string generate(const std::vector<Message> & messages, const Params & params,
                         const TokenCallback & cb, std::string & error);

    /** Ask an in-flight generate() to stop. Safe to call from another thread. */
    void request_stop();

    int  context_size() const;

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

}  // namespace elias
