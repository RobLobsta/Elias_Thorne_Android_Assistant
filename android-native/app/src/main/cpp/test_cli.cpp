// Host harness for elias_core, so the generation path can be exercised on a
// desktop before it is trusted on a phone. Not part of the Android build.
//
//   elias-test <model.gguf> "first question" ["second question" ...]

#include "elias_core.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

/**
 * Every piece handed to the callback must be complete UTF-8.
 *
 * This is not cosmetic: the JNI layer passes each piece to NewStringUTF, and
 * Android aborts the whole process on malformed Modified UTF-8. Byte-level BPE
 * splits emoji across tokens, so this invariant is what stands between the app
 * and a crash instead of an answer.
 */
static bool valid_utf8(const std::string & s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        size_t len;
        if (c < 0x80)            len = 1;
        else if ((c >> 5) == 6)  len = 2;
        else if ((c >> 4) == 14) len = 3;
        else if ((c >> 3) == 30) len = 4;
        else return false;
        if (i + len > s.size()) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
        }
        i += len;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> \"question\" [\"question\" ...]\n", argv[0]);
        return 2;
    }

    elias::Params params;
    params.n_threads = 4;
    params.n_predict = 96;
    if (const char * v = getenv("ELIAS_MMAP"))        params.use_mmap       = atoi(v) != 0;
    if (const char * v = getenv("ELIAS_THREADS"))     params.n_threads      = atoi(v);
    if (const char * v = getenv("ELIAS_PENALTY"))     params.repeat_penalty = atof(v);
    if (const char * v = getenv("ELIAS_TEMP"))        params.temperature    = atof(v);
    if (const char * v = getenv("ELIAS_PENALTY_LAST")) params.repeat_last_n  = atoi(v);

    std::string error;
    elias::Model * model = elias::Model::load(argv[1], params, error);
    if (!model) {
        fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }

    // ELIAS_SYSTEM lets the system turn be varied from the shell; empty means
    // no system turn at all, which some small models handle better.
    std::vector<elias::Message> chat;
    if (const char * sys = getenv("ELIAS_SYSTEM")) {
        if (*sys) chat.push_back({"system", sys});
    } else {
        chat.push_back({"system", "You are Elias Thorne, a concise, friendly assistant running "
                                  "entirely on this device. Answer in one or two sentences."});
    }

    for (int i = 2; i < argc; ++i) {
        chat.push_back({"user", argv[i]});
        if (i == 2) printf("cpu: %s\n", elias::cpu_features().c_str());
        printf("\nYOU:   %s\nELIAS: ", argv[i]);
        fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        int tokens = 0;
        int bad_pieces = 0;
        const std::string reply = model->generate(chat, params, [&](const std::string & piece) {
            if (!valid_utf8(piece)) {
                ++bad_pieces;
                fprintf(stderr, "\n  !! piece %d is not valid UTF-8 — this would abort on Android\n",
                        tokens);
            }
            fputs(piece.c_str(), stdout);
            fflush(stdout);
            ++tokens;
            return true;
        }, error);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (!error.empty()) {
            fprintf(stderr, "\n  [error: %s]\n", error.c_str());
            error.clear();
        }
        printf("\n       [%d tokens, %.1f s, %.1f tok/s, utf8 %s, reply %s]\n", tokens, secs,
               tokens / secs, bad_pieces == 0 ? "ok" : "BROKEN",
               valid_utf8(reply) ? "ok" : "BROKEN");
        const auto st = model->last_stats();
        printf("       prompt %d tok @ %.1f tok/s | generate %d tok @ %.2f tok/s\n",
               st.n_prompt, st.prompt_tok_s(), st.n_eval, st.eval_tok_s());
        chat.push_back({"assistant", reply});
    }

    delete model;
    return 0;
}
