// Host harness for elias_core, so the generation path can be exercised on a
// desktop before it is trusted on a phone. Not part of the Android build.
//
//   elias-test <model.gguf> "first question" ["second question" ...]

#include "elias_core.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> \"question\" [\"question\" ...]\n", argv[0]);
        return 2;
    }

    elias::Params params;
    params.n_threads = 4;
    params.n_predict = 96;

    std::string error;
    elias::Model * model = elias::Model::load(argv[1], params, error);
    if (!model) {
        fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }

    std::vector<elias::Message> chat = {
        {"system", "You are Elias Thorne, a concise, friendly assistant running "
                   "entirely on this device. Answer in one or two sentences."},
    };

    for (int i = 2; i < argc; ++i) {
        chat.push_back({"user", argv[i]});
        printf("\nYOU:   %s\nELIAS: ", argv[i]);
        fflush(stdout);

        const auto t0 = std::chrono::steady_clock::now();
        int tokens = 0;
        const std::string reply = model->generate(chat, params, [&](const std::string & piece) {
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
        printf("\n       [%d tokens, %.1f s, %.1f tok/s]\n", tokens, secs, tokens / secs);
        chat.push_back({"assistant", reply});
    }

    delete model;
    return 0;
}
