// Host-only API driver. No production run entry point is exposed here.
#include "run_config.h"
#include <iostream>
#include <iterator>

int main(int argc, char **argv) {
    try {
        if (argc != 2 && argc != 3) {
            throw std::runtime_error("expected domain and optional helper operation");
        }
        std::string input;
        char byte;
        while (std::cin.get(byte)) {
            if (input.size() == 1048576) {
                throw std::runtime_error("configuration test input limit exceeded");
            }
            input.push_back(byte);
        }
        const auto json = fgm::Parser(input).parse();
        if (argc == 3) {
            auto operand = [&](const char *key) {
                const auto text = json.at(key).str();
                if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
                    throw std::runtime_error("unsigned helper operand required");
                }
                return std::stoull(text);
            };
            const std::string operation = argv[2];
            if (operation == "add") {
                std::cout << fgm::configCheckedAdd(operand("a"), operand("b"));
            } else if (operation == "multiply") {
                std::cout << fgm::configCheckedMultiply(operand("a"), operand("b"));
            } else {
                throw std::runtime_error("unknown helper operation");
            }
            return 0;
        }
        const auto config = fgm::parseControlledConfig(json, argv[1]);
        auto output = fgm::Json::dict();
        output.object["resolved"] = config.resolved();
        output.object["ceiling"] = fgm::Json(int64_t(config.ceiling()));
        output.object["interval_span"] = fgm::Json(int64_t(config.intervalSpan()));
        output.object["below_ceiling_allowed"] = fgm::Json(config.expansionAllowed(config.ceiling() - 1));
        output.object["at_ceiling_allowed"] = fgm::Json(config.expansionAllowed(config.ceiling()));
        output.object["anchor_representation_eligible"] = fgm::Json(config.representationIneligibility(config.anchor).empty());
        std::cout << fgm::dump(output) << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
