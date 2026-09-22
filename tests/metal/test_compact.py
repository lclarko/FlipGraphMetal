from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CompactEligibility(unittest.TestCase):
    def test_whole_population_and_option_boundaries(self):
        source = (ROOT / 'src/metal/flip_graph.cpp').read_text()
        start = source.index('static bool canUseCompactScheme(')
        end = source.index('#endif', start)
        functions = source[start:end]
        program = r'''#include <cassert>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#include "scheme_integer.h"
struct FlipGraphProbabilities { float expand, reduce, sandwiching, basis, resize; };
''' + functions + r'''
int main() {
    SchemeInteger current[2], best[2];
    for (int i = 0; i < 2; i++) {
        current[i].initializeNaive(3, 3, 3);
        best[i] = current[i];
    }
    int flips[2] = {};
    FlipGraphProbabilities probabilities{};
    auto eligible = [&] { return canUseCompactWalk(current, best, flips, 2, 32, 1000, 1000000000, probabilities); };
    assert(eligible());
    for (int count : {0, -1})
        assert(!canUseCompactWalk(current, best, flips, count, 32, 1000, 1000000000, probabilities));
    assert(!canUseCompactWalk(current, best, flips, 2, 64, 1000, 1000000000, probabilities));
    assert(!canUseCompactWalk(current, best, flips, 2, 32, 999, 1000000000, probabilities));
    assert(!canUseCompactWalk(current, best, flips, 2, 32, 1000, 999999999, probabilities));
    for (float *value : {&probabilities.expand, &probabilities.reduce, &probabilities.sandwiching,
                        &probabilities.basis, &probabilities.resize}) {
        *value = .01f;
        assert(!eligible());
        *value = 0;
    }
    flips[1] = 999998999;
    assert(eligible());
    flips[1] = 999999000;
    assert(!eligible());
    flips[1] = -1;
    assert(!eligible());
    flips[1] = 0;
    for (SchemeInteger *scheme : {&current[1], &best[1]}) {
        SchemeInteger saved = *scheme;
        for (int field = 0; field < 9; field++) {
            if (field == 0) scheme->n[2] = 4;
            if (field == 1) scheme->nn[2] = 16;
            if (field == 2) scheme->m = 0;
            if (field == 3) scheme->m = 351;
            if (field == 4) scheme->flips[2].size = 501;
            if (field == 5) scheme->uvw[2][scheme->m - 1].n = 16;
            if (field == 6) scheme->uvw[2][scheme->m - 1].values |= T(1) << 9;
            if (field == 7) scheme->uvw[2][scheme->m - 1].signs |= T(1) << 63;
            if (field == 8) scheme->uvw[2][scheme->m - 1].values |= T(1) << 16;
            assert(!eligible());
            *scheme = saved;
        }
        scheme->uvw[0][0].valid = false;
        assert(eligible());
        *scheme = saved;
    }
    for (SchemeInteger *scheme : {&current[1], &best[1]}) {
        scheme->m = 350;
        for (int p = 0; p < 3; p++) {
            scheme->flips[p].size = 500;
            for (int r = 0; r < 350; r++) scheme->uvw[p][r] = Addition(9);
        }
    }
    assert(eligible());
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            code, binary = directory / 'eligibility.cpp', directory / 'eligibility'
            code.write_text(program)
            subprocess.run(['clang++', '-std=c++17', '-O2', '-I' + str(ROOT / 'src/metal'),
                            str(code), '-o', str(binary)], capture_output=True, check=True, timeout=45)
            subprocess.run([str(binary)], capture_output=True, check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
