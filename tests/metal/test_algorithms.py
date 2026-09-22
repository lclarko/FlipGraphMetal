import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('algorithms', ROOT / 'benchmarks/metal/algorithms.py')
algorithms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(algorithms)


class Algorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        fixture = ROOT / 'tests/metal/fixtures/perminov_reduction'
        provenance = json.loads((fixture / 'provenance.json').read_text())
        if provenance['commit'] != algorithms.REDUCTION_COMMIT:
            raise ValueError('reduction fixture commit differs from algorithm experiment')
        for name, expected in provenance['files'].items():
            if hashlib.sha256((fixture / name).read_bytes()).hexdigest() != expected:
                raise ValueError('reduction fixture changed: ' + name)
        upstream = (fixture / 'check_flip_reduce.hpp').read_text()
        for name, lazy, sign in [('original', False, False), ('lazy', True, False),
                                 ('sign', False, True), ('both', True, True)]:
            directory = cls.directory / 'variants' / name
            shutil.copytree(ROOT / 'src/metal', directory)
            scheme = directory / 'scheme_integer.h'
            scheme.write_text(algorithms.transform(scheme.read_text(), lazy, sign, upstream))
            if lazy:
                core = directory / 'core.h'
                core.write_text(core.read_text() + algorithms.CANDIDATE)
            shutil.copy2(fixture / 'LICENSE', directory / 'UPSTREAM_LICENSE')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def compile_run(self, name, text, variant='both'):
        source = self.directory / (name + '.cpp')
        source.write_text(text)
        binary = self.directory / name
        subprocess.run([shutil.which('clang++') or 'c++', '-std=c++17', '-O2',
                        '-I' + str(self.directory / 'variants' / variant), str(source), '-o', str(binary)],
                       check=True, capture_output=True, text=True, timeout=45)
        subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)

    def test_upstream_counterexample(self):
        indices = [0, 1]
        emitted = []
        for p, word in enumerate([0, 0]):
            q = word % (len(indices) - p)
            emitted.append(indices[q])
            indices[p], indices[q] = indices[q], indices[p]
        self.assertEqual(emitted, [0, 0])
        self.assertNotEqual(sorted(emitted), [0, 1])

    def test_generated_suffix_selection(self):
        self.compile_run('selection', r'''#include <cassert>
#include "core.h"
int main() {
    int pair[2] = {0, 1};
    assert(algorithmCandidate(pair, 0, 2, 0) == 0);
    assert(algorithmCandidate(pair, 1, 2, 0) == 1);
    for (uint32_t seed = 1; seed <= 1000; seed++) {
        RandomState state{seed};
        int indices[31];
        bool seen[31] = {};
        for (int p = 0; p < 31; p++) indices[p] = p;
        for (int p = 0; p < 31; p++) {
            int selected = algorithmCandidate(indices, p, 31, randomWord(&state));
            assert(selected >= 0 && selected < 31 && !seen[selected]);
            seen[selected] = true;
        }
    }
}''')

    def test_sign_reductions_preserve_tensor(self):
        self.compile_run('reductions', r'''#include <cassert>
#include <vector>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#define private public
#include "scheme_integer.h"
#undef private
std::vector<int> tensor(const SchemeInteger &s) {
    std::vector<int> result(729);
    for (int i = 0; i < 9; i++) for (int j = 0; j < 9; j++) for (int k = 0; k < 9; k++)
        for (int r = 0; r < s.m; r++) result[(i * 9 + j) * 9 + k] += s.uvw[0][r][i] * s.uvw[1][r][j] * s.uvw[2][r][k];
    return result;
}
int main() {
    for (int signU : {-1, 1}) for (int swap : {0, 1}) {
        SchemeInteger s;
        s.m = 2;
        for (int p = 0; p < 3; p++) {
            s.n[p] = 3; s.nn[p] = 9;
            for (int r = 0; r < 2; r++) s.uvw[p][r] = Addition(9);
        }
        s.uvw[0][0].set(0, 1); s.uvw[0][1].set(0, signU);
        s.uvw[1][0].set(swap, 1); s.uvw[1][1].set(1 - swap, 1);
        s.uvw[2][0].set(2, 1); s.uvw[2][1].set(2, -1);
        s.initFlips();
        auto before = tensor(s);
        assert(s.checkFlipReduce(0, 1, 0, 1, -1));
        assert(s.m == 1 && tensor(s) == before);
        assert(s.uvw[0][0].valid && s.uvw[1][0].valid && s.uvw[2][0].valid);
    }
}''')

    def test_independent_signed_reduction_policy(self):
        self.compile_run('policy', r'''#include <cassert>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#define private public
#include "scheme_integer.h"
#undef private
int compareDense(const int *a, const int *b) {
    if (a[0] == b[0] && a[1] == b[1]) return 1;
    if (a[0] == -b[0] && a[1] == -b[1]) return -1;
    return 0;
}
int firstSign(const int *a) { return a[0] ? a[0] : a[1]; }
int main() {
    int branches[4] = {}, rejected = 0;
    for (int shared : {-1, 1}) for (int a = 0; a < 9; a++) for (int b = 0; b < 9; b++)
    for (int c = 0; c < 9; c++) for (int d = 0; d < 9; d++) {
        if (a == 4 || b == 4 || c == 4 || d == 4) continue;
        int dense[3][2][2] = {{{a % 3 - 1, a / 3 - 1}, {b % 3 - 1, b / 3 - 1}},
                              {{c % 3 - 1, c / 3 - 1}, {d % 3 - 1, d / 3 - 1}}, {{1, 0}, {shared, 0}}};
        int target = -1, retained = 0, result[2] = {}, branch = -1;
        for (int factor = 0; factor < 2 && target == -1; factor++) {
            int cmp = compareDense(dense[factor][0], dense[factor][1]);
            if (!cmp) continue;
            int changed = 1 - factor;
            bool sum = cmp == shared;
            int value[2] = {dense[changed][0][0] + (sum ? 1 : -1) * dense[changed][1][0],
                            dense[changed][0][1] + (sum ? 1 : -1) * dense[changed][1][1]};
            if (value[0] < -1 || value[0] > 1 || value[1] < -1 || value[1] > 1) continue;
            if (sum && firstSign(value) < 0) continue;
            target = changed;
            retained = !sum && firstSign(value) < 0;
            for (int q = 0; q < 2; q++) result[q] = retained ? -value[q] : value[q];
            branch = 2 * factor + !sum;
        }
        SchemeInteger s;
        s.m = 2;
        for (int p = 0; p < 3; p++) {
            s.n[p] = 3; s.nn[p] = 9;
            for (int r = 0; r < 2; r++) {
                s.uvw[p][r] = Addition(9);
                for (int q = 0; q < 2; q++) s.uvw[p][r].set(q, dense[p][r][q]);
            }
        }
        s.initFlips();
        SchemeInteger before = s;
        bool changed = s.checkFlipReduce(0, 1, 0, 1, shared);
        assert(changed == (target != -1));
        if (!changed) {
            rejected++;
            assert(s.m == before.m);
            for (int p = 0; p < 3; p++) {
                assert(s.n[p] == before.n[p] && s.nn[p] == before.nn[p]);
                assert(s.flips[p].size == before.flips[p].size);
                for (size_t pair = 0; pair < s.flips[p].size; pair++) assert(s.flips[p].pairs[pair] == before.flips[p].pairs[pair]);
                for (int r = 0; r < 2; r++) {
                    assert(s.uvw[p][r] == before.uvw[p][r]);
                    assert(s.uvw[p][r].n == before.uvw[p][r].n && s.uvw[p][r].valid == before.uvw[p][r].valid);
                }
            }
            continue;
        }
        branches[branch]++;
        bool zero = result[0] == 0 && result[1] == 0;
        assert(s.m == (zero ? 0 : 1));
        if (!zero) for (int p = 0; p < 3; p++) for (int q = 0; q < 9; q++) {
            int expected = q >= 2 ? 0 : p == target ? result[q] : dense[p][retained][q];
            assert(s.uvw[p][0][q] == expected);
        }
        for (int x = 0; x < 2; x++) for (int y = 0; y < 2; y++) for (int z = 0; z < 2; z++) {
            int tensor = 0;
            for (int r = 0; r < 2; r++) tensor += dense[0][r][x] * dense[1][r][y] * dense[2][r][z];
            int actual = zero ? 0 : s.uvw[0][0][x] * s.uvw[1][0][y] * s.uvw[2][0][z];
            assert(actual == tensor);
        }
    }
    for (int count : branches) assert(count > 0);
    assert(rejected > 0);
}''')

    def test_full_cpu_walks_validate_tensors(self):
        for variant in ['original', 'lazy', 'sign', 'both']:
            self.compile_run('walk-' + variant, r'''#include <cassert>
#include "core.h"
#include "addition.h"
#include "flip_set.h"
#include "scheme_integer.h"
int main() {
    SchemeInteger s;
    s.initializeNaive(3, 3, 3);
    RandomState state{19};
    for (int i = 0; i < 5000; i++) {
        if (!s.tryFlip(state)) s.tryExpand(randint(1, 2, state), state);
        if (i % 100 == 0) assert(s.validate());
    }
    assert(s.validate());
}''', variant)


if __name__ == '__main__':
    unittest.main()
