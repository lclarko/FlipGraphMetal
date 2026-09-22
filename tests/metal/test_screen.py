from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from screen import check_log


def log_for(rounds, duration='1.25'):
    return ''.join('Metal dispatch randomWalkKernel: 32 threads, ' + duration + ' ms GPU\n'
                   + f'ROUND {i} CPU 0.01 METAL_WALL 0.01 MATCH\n' for i in rounds)


class ScreenEvidence(unittest.TestCase):
    def test_exact_round_ids(self):
        seconds, rounds = check_log(log_for(range(6)), 6)
        self.assertEqual(len(seconds), 6)
        self.assertEqual(rounds, [str(i) for i in range(6)])
        for ids in ([0, 1, 2, 3, 4, 4], [0, 1, 3, 2, 4, 5], [0, 1]):
            with self.assertRaisesRegex(RuntimeError, 'comparisons'):
                check_log(log_for(ids), 6)

    def test_positive_gpu_evidence(self):
        with self.assertRaisesRegex(RuntimeError, 'nonpositive'):
            check_log(log_for(range(6), '0'), 6)
        with self.assertRaisesRegex(RuntimeError, 'GPU dispatch count'):
            check_log(log_for(range(6)) + 'Metal dispatch randomWalkKernel: 32 threads, 1 ms GPU\n', 6)
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            check_log('ROUND 0 CPU 1 METAL_WALL 1 MATCH\n', 1)

    def test_other_diagnostics_need_no_round_markers(self):
        self.assertEqual(check_log('Metal dispatch boundaryKernel: 1 threads, 0.5 ms GPU\n'), ([], []))

    def test_storage_rejects_reduced_capacity_before_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'input'
            source.mkdir()
            (source / 'storage.json').write_text('{}')
            result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/metal/build.py'),
                                     '--source', str(source), '--gpu-source', str(source),
                                     '--output', str(directory / 'output'), '--rank-capacity', '32'],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn('storage diagnostics require rank capacity 350', result.stderr)
            self.assertFalse((directory / 'output/matched').exists())


if __name__ == '__main__':
    unittest.main()
