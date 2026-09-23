#include "host.h"
#include "../common/arg_parser.cpp"
#include "utils.cpp"
int metalRounds = 0;

void validateOptions(const ArgParser &parser) {
    auto positive = [&](const char *name, int max) {
        int value = std::stoi(parser.get(name));
        if (value < 1 || value > max) throw std::runtime_error(std::string("unsupported ") + name + ": expected 1.." + std::to_string(max));
    };
    positive("--block-size", 1024);
#if METAL_PROGRAM == 1
    positive("-n1", 16); positive("-n2", 16); positive("-n3", 16);
    int n1 = std::stoi(parser.get("-n1")), n2 = std::stoi(parser.get("-n2")), n3 = std::stoi(parser.get("-n3"));
    if (n1*n2 > 64 || n2*n3 > 64 || n3*n1 > 64)
        throw std::runtime_error("unsupported dimensions: at most 64 elements per factor");
    if (parser.get("--input-path") == "null" && n1*n2*n3 > MAX_RANK)
        throw std::runtime_error("unsupported dimensions: naive rank exceeds 350");
    positive("--schemes", 1048576);
    positive("--max-iterations", 2147483647);
    positive("--plus-iterations", 2147483647);
    for (const char *name : {"--expand-probability", "--reduce-probability", "--sandwiching-probability", "--basis-probability", "--resize-probability"}) {
        double p = std::stod(parser.get(name));
        if (!std::isfinite(p) || p < 0 || !std::isfinite(float(p)) || (p > 0 && float(p) == 0))
            throw std::runtime_error(std::string(name) + " must be a nonnegative finite float");
        if (std::string(name) == "--resize-probability" && p > 1)
            throw std::runtime_error("--resize-probability must be in [0,1]");
    }
#ifndef METAL_F2
    if (std::stod(parser.get("--sandwiching-probability")) != 0)
        throw std::runtime_error("unsupported signed-ternary sandwiching: upstream implementation is empty");
#endif
#elif METAL_PROGRAM == 2
    positive("--schemes", 1048576);
    positive("--max-iterations", 2147483647);
    positive("--max-no-improvements", 2147483647);
#else
    positive("--count", 1048576);
    positive("--schemes-count", 1048576);
    positive("--max-no-improvements", 2147483647);
    if (std::stoi(parser.get("--schemes-count")) > std::stoi(parser.get("--count")))
        throw std::runtime_error("--schemes-count must not exceed --count");
    if (std::stoi(parser.get("--max-flips")) == 2147483647)
        throw std::runtime_error("unsupported --max-flips: maximum is 2147483646");
#endif
}

#if METAL_PROGRAM == 1
#include "flip_graph.cpp"
#include "main_flip_graph.cpp"
#elif METAL_PROGRAM == 2
#include "complexity_minimizer.cpp"
#include "main_complexity_minimizer.cpp"
#else
#include "scheme_additions_reducer.cpp"
#include "main_additions_reducer.cpp"
#endif

int main(int argc, char *argv[]) {
    try {
        return runProgram(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
