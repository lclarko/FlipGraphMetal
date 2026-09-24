// Standard-library engine used to check the independent host RNG model.
#include <cstdint>
#include <iostream>
#include <random>

int main() {
    for (uint32_t seed : {0u, 7u, 4294967295u}) {
        std::mt19937_64 rng(seed);
        for (int i = 0; i < 700; ++i) {
            std::cout << rng() << '\n';
        }
    }
}
