from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from simd import candidate_function


HARNESS = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#define private public
#include "scheme_integer.h"
#undef private
#define thread
using uint = unsigned int;
struct StorageScheme { SchemeInteger *storage; };
'''

CHECK = r'''
bool batched(SchemeInteger &scheme, RandomState &state) {
    int size = scheme.flips[0].size + scheme.flips[1].size + scheme.flips[2].size;
    int indices[MAX_PAIRS * 3];
    randomPermutation(indices, size, state);
    StorageScheme view{&scheme};
    for (int begin = 0; begin < size; begin += 32) {
        int count = std::min(32, size - begin);
        std::array<uint, 32> operations, after;
        RandomState preview = state;
        for (int lane = 0; lane < count; lane++) {
            uint orientation = randomWord(&preview) % 2;
            orientation |= (randomWord(&preview) % 2) << 1;
            after[lane] = preview.value;
            operations[lane] = simdCandidate(view, indices[begin + lane], orientation);
        }
        int winner = 32;
        for (int lane = count - 1; lane >= 0; lane--)
            if (operations[lane] != 0xffffffffu) winner = lane;
        state.value = after[winner < 32 ? winner : count - 1];
        if (winner < 32) {
            uint selected = operations[winner];
            scheme.flip(selected & 3, (selected >> 2) & 3, (selected >> 4) & 3,
                        (selected >> 6) & 511, (selected >> 15) & 511, true);
            return true;
        }
    }
    return false;
}

int main() {
    for (uint seed : {7u, 19u, 41u}) {
        SchemeInteger original;
        original.initializeNaive(3, 3, 3);
        RandomState state{seed};
        for (int step = 0; step < 1000; step++) {
            SchemeInteger candidate = original;
            RandomState compared = state;
            bool flipped = original.tryFlip(state);
            assert(batched(candidate, compared) == flipped);
            assert(state.value == compared.value);
            assert(original.m == candidate.m);
            for (int p = 0; p < 3; p++) {
                assert(original.flips[p].size == candidate.flips[p].size);
                for (size_t q = 0; q < original.flips[p].size; q++)
                    assert(original.flips[p].pairs[q] == candidate.flips[p].pairs[q]);
                for (int r = 0; r < original.m; r++) {
                    assert(original.uvw[p][r] == candidate.uvw[p][r]);
                    assert(original.uvw[p][r].valid == candidate.uvw[p][r].valid);
                    assert(original.uvw[p][r].n == candidate.uvw[p][r].n);
                }
            }
            if (!flipped) original.tryExpand(randint(1, 2, state), state);
            assert(original.validate());
        }
    }
}
'''


class SimdSelection(unittest.TestCase):
    def test_candidate_selection_matches_original_transformations(self):
        original = (ROOT / 'src/metal/scheme_integer.h').read_text()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, binary = directory / 'selection.cpp', directory / 'selection'
            source.write_text(HARNESS + candidate_function(original) + CHECK)
            subprocess.run(['clang++', '-std=c++17', '-O2', '-I' + str(ROOT / 'src/metal'),
                            str(source), '-o', str(binary)], check=True, capture_output=True, timeout=45)
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=45)

    def test_earliest_eligible_lane_with_partial_batches(self):
        for count in [0, 1, 31, 32, 33, 63, 64, 65, 1499, 1500]:
            for success in range(-1, count):
                eligible = [position >= success for position in range(count)] if success >= 0 else [False] * count
                serial = next((p for p, value in enumerate(eligible) if value), None)
                selected = None
                for begin in range(0, count, 32):
                    lanes = [lane if begin + lane < count and eligible[begin + lane] else 32 for lane in range(32)]
                    winner = min(lanes)
                    if winner < 32:
                        selected = begin + winner
                        break
                self.assertEqual(serial, selected)


if __name__ == '__main__':
    unittest.main()
