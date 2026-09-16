#include "elias_core.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>

namespace elias {

namespace {

/** Turn markers the model may emit when it decides to keep writing the script. */
const char * const kLeakMarkers[] = {"\nUser:", "\nSystem:", "\nAssistant:", "User:", "System:"};

std::string capitalise(const std::string & role) {
    if (role.empty()) return role;
    std::string out = role;
    out[0] = static_cast<char>(toupper(static_cast<unsigned char>(out[0])));
    return out;
}

/**
 * Length of the longest prefix of `s` that is complete, well-formed UTF-8.
 *
 * Byte-level BPE splits multi-byte characters across tokens — an emoji arrives
 * as a 3-byte piece followed by a 1-byte piece — so a single token's text is
 * often half a character. Handing that to JNI's NewStringUTF is not merely
 * untidy: Android's CheckJNI aborts the process with "input is not valid
 * Modified UTF-8", which reads as a crash with no output at all.
 */
size_t complete_utf8_prefix(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        size_t len;
        if (c < 0x80)            len = 1;
        else if ((c >> 5) == 6)  len = 2;
        else if ((c >> 4) == 14) len = 3;
        else if ((c >> 3) == 30) len = 4;
        else return i;  // a stray continuation byte: stop before it
        if (i + len > s.size()) return i;  // the tail is a partial character
        for (size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return i;
        }
        i += len;
    }
    return i;
}

/** llama.cpp logs every graph reservation at INFO; quiet unless something breaks. */
void quiet_log(ggml_log_level level, const char * text, void * /*user*/) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

}  // namespace

std::string render_prompt(const std::vector<Message> & messages) {
    std::string out;
    for (const auto & m : messages) {
        out += capitalise(m.role);
        out += ": ";
        out += m.content;
        out += "<|eot_id|>";
    }
    out += "Assistant: ";
    return out;
}

std::string render_prompt(const ::llama_model * model, const std::vector<Message> & messages) {
    // Prefer whatever the file declares, so a standard instruct GGUF dropped in
    // here works without a code change. BitNet's own GGUF is the exception: its
    // embedded template ("Human: ... BITNETAssistant: ") is not what the model
    // was trained on, so it is explicitly rejected in favour of the real one.
    const char * tmpl = model ? llama_model_chat_template(model, nullptr) : nullptr;
    const bool usable = tmpl && *tmpl && strstr(tmpl, "BITNETAssistant") == nullptr;
    if (!usable) return render_prompt(messages);

    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const auto & m : messages) chat.push_back({m.role.c_str(), m.content.c_str()});

    std::vector<char> buf(8192);
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(),
                                          (int32_t) buf.size());
    if (n > (int32_t) buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buf.data(),
                                      (int32_t) buf.size());
    }
    if (n < 0) return render_prompt(messages);
    return std::string(buf.data(), n);
}

struct Model::Impl {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    /** Token ids that end the assistant's turn. See resolve_stop_tokens. */
    std::vector<llama_token> stop_ids;
    std::atomic<bool> stop{false};
    std::mutex busy;
};

namespace {

/**
 * Find the ids that end a turn, by name rather than by trusting the file.
 *
 * The chat template ends every turn with <|eot_id|>, but Microsoft's GGUF does
 * not nominate it as an EOS token, so llama_vocab_is_eog never fires on it and
 * the model runs on past its answer — repeating itself, then inventing the next
 * turn. Resolving the literal gives a stop condition that matches the template
 * actually in use.
 */
std::vector<llama_token> resolve_stop_tokens(const llama_vocab * vocab) {
    std::vector<llama_token> ids;
    for (const char * literal : {"<|eot_id|>", "<|end_of_text|>"}) {
        llama_token buf[8];
        const int n = llama_tokenize(vocab, literal, (int32_t) strlen(literal), buf,
                                     (int32_t) (sizeof(buf) / sizeof(buf[0])), false, true);
        // Exactly one token means the vocabulary really has it as a special
        // token; anything else means it was split into bytes and is not one.
        if (n == 1) ids.push_back(buf[0]);
    }
    return ids;
}

}  // namespace

Model::~Model() {
    if (!impl_) return;
    if (impl_->ctx) llama_free(impl_->ctx);
    if (impl_->model) llama_model_free(impl_->model);
    delete impl_;
    impl_ = nullptr;
}

Model * Model::load(const std::string & gguf_path, const Params & params, std::string & error) {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(quiet_log, nullptr);
        llama_backend_init();
    });

    auto mparams = llama_model_default_params();
    // No GPU offload: there is no usable GPU backend for this on a handset, and
    // the whole point of a ternary model is that the CPU is enough.
    mparams.n_gpu_layers = 0;
    // mmap is the reason this fits a 4 GB phone: the ~1.2 GB of weights stay
    // file-backed and evictable instead of becoming dirty anonymous pages.
    mparams.use_mmap  = true;
    mparams.use_mlock = false;

    llama_model * model = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model) {
        error = "could not load " + gguf_path;
        return nullptr;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx     = params.n_ctx;
    cparams.n_batch   = 512;
    cparams.n_threads = params.n_threads;
    cparams.n_threads_batch = params.n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        error = "could not create a context";
        return nullptr;
    }

    auto * m = new Model();
    m->impl_ = new Impl();
    m->impl_->model = model;
    m->impl_->ctx   = ctx;
    m->impl_->vocab = llama_model_get_vocab(model);
    m->impl_->stop_ids = resolve_stop_tokens(m->impl_->vocab);
    return m;
}

int Model::context_size() const {
    return impl_ ? static_cast<int>(llama_n_ctx(impl_->ctx)) : 0;
}

void Model::request_stop() {
    if (impl_) impl_->stop.store(true);
}

std::string Model::generate(const std::vector<Message> & messages, const Params & params,
                            const TokenCallback & cb, std::string & error) {
    if (!impl_) {
        error = "model is not loaded";
        return {};
    }
    // One generation at a time: the context holds the KV cache, and two turns
    // interleaved would corrupt each other's.
    std::lock_guard<std::mutex> lock(impl_->busy);
    impl_->stop.store(false);

    const std::string prompt = render_prompt(impl_->model, messages);

    const int n_prompt = -llama_tokenize(impl_->vocab, prompt.c_str(), (int32_t) prompt.size(),
                                         nullptr, 0, true, true);
    if (n_prompt <= 0) {
        error = "the prompt could not be tokenized";
        return {};
    }
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(impl_->vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(),
                       (int32_t) tokens.size(), true, true) < 0) {
        error = "could not tokenize the prompt";
        return {};
    }

    const int n_ctx = (int) llama_n_ctx(impl_->ctx);
    if (n_prompt >= n_ctx) {
        error = "the conversation no longer fits in the context window";
        return {};
    }

    // A fresh KV cache each turn. The whole conversation is re-sent, which costs
    // prompt-eval time but keeps the state impossible to desynchronise from
    // what the caller thinks was said.
    llama_memory_clear(llama_get_memory(impl_->ctx), true);

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    // Penalties first, then the truncations, then temperature, then the draw —
    // greedy decoding on a 2B model loops on anything with two clauses in it.
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(params.repeat_last_n,
                                                               params.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

    std::string reply;
    // Held across iterations on purpose. llama_batch_get_one stores the pointer
    // it is given rather than copying, so a token declared inside the loop would
    // be read back by the next llama_decode after going out of scope.
    llama_token id = 0;
    // Carries a partial multi-byte character between tokens; see
    // complete_utf8_prefix.
    std::string pending;

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());

    for (int decoded = 0; decoded < params.n_predict; ++decoded) {
        if (impl_->stop.load()) break;

        if (llama_decode(impl_->ctx, batch) != 0) {
            error = "decode failed";
            break;
        }

        id = llama_sampler_sample(smpl, impl_->ctx, -1);
        if (llama_vocab_is_eog(impl_->vocab, id)) break;
        if (std::find(impl_->stop_ids.begin(), impl_->stop_ids.end(), id) != impl_->stop_ids.end()) break;

        char buf[256];
        const int n = llama_token_to_piece(impl_->vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            error = "could not detokenize";
            break;
        }
        const std::string piece(buf, n);
        reply += piece;

        // The model sometimes carries on and writes the user's next line for
        // them. Cut the reply at the first turn marker rather than speaking it.
        bool leaked = false;
        for (const char * marker : kLeakMarkers) {
            const size_t at = reply.find(marker);
            if (at != std::string::npos) {
                reply.erase(at);
                leaked = true;
                break;
            }
        }
        if (leaked) break;

        // Only hand on whole characters; anything half-finished waits for the
        // token that completes it.
        pending += piece;
        const size_t whole = complete_utf8_prefix(pending);
        if (whole > 0) {
            const std::string emit = pending.substr(0, whole);
            pending.erase(0, whole);
            if (cb && !cb(emit)) break;
        }

        batch = llama_batch_get_one(&id, 1);
    }

    llama_sampler_free(smpl);

    // A reply cut short by the token budget can end mid-character; the caller
    // passes this straight to NewStringUTF, so it must be whole.
    reply.resize(complete_utf8_prefix(reply));

    // Trim the trailing whitespace the template's "Assistant: " tends to attract.
    while (!reply.empty() && (reply.back() == '\n' || reply.back() == ' ')) reply.pop_back();
    size_t start = reply.find_first_not_of(" \n");
    if (start != std::string::npos && start > 0) reply.erase(0, start);

    return reply;
}

}  // namespace elias
