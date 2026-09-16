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
    // Measured against BitNet 2B4T: at 1.1 it answers correctly and then loops
    // for the whole token budget; at 1.25 over a 256-token window it answers and
    // stops. Higher again starts mangling grammar.
    float repeat_penalty = 1.25f;
    int   repeat_last_n  = 256;
    uint32_t seed     = 0xFFFFFFFF;  // LLAMA_DEFAULT_SEED
    /**
     * Map the weights from the file, or read them into anonymous memory.
     *
     * mmap keeps resident memory low by letting the kernel drop the model's
     * pages under pressure — which is exactly the problem on a device that is
     * always under pressure. Every dropped page is re-read from flash on the
     * next token, and generation touches the whole model per token. Reading the
     * weights in instead costs the full footprint up front, but Android can
     * compress anonymous pages into zram rather than discarding them.
     */
    bool use_mmap = true;
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

/**
 * Timings for the last generate(), split into the two phases.
 *
 * The split is the diagnostic that matters. Prompt evaluation is compute-bound
 * and reuses each weight across many tokens at once; generation re-reads the
 * whole model per token and is bound by memory bandwidth. A build compiled
 * badly makes both slow together. A device that cannot keep the weights
 * resident makes only generation slow, by a lot — so the ratio says which.
 */
struct Stats {
    double prompt_ms   = 0;
    double eval_ms     = 0;
    int    n_prompt    = 0;
    int    n_eval      = 0;
    double prompt_tok_s() const { return n_prompt && prompt_ms > 0 ? n_prompt * 1000.0 / prompt_ms : 0; }
    double eval_tok_s()   const { return n_eval   && eval_ms   > 0 ? n_eval   * 1000.0 / eval_ms   : 0; }
};

/** What the CPU actually reports at runtime, which is not what was compiled. */
std::string cpu_features();

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

    /** Timings from the most recent generate(). */
    Stats last_stats() const;

    /** One line describing the loaded model and how it is being run. */
    std::string describe() const;

    int  context_size() const;

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

}  // namespace elias
