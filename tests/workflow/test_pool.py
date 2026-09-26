"""Native retained-pool contracts and independent parent-selection comparisons."""
import os
from pathlib import Path
import subprocess
import unittest
from host_rng import HostRNG

ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(os.environ.get('FGM_POOL_DRIVER', ROOT / 'build/workflow/test_pool'))


class PoolTests(unittest.TestCase):
    def run_driver(self, *args):
        result = subprocess.run([str(DRIVER), *map(str, args)], text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()

    def test_selection_matches_independent_host_rng(self):
        for mode in ('uniform', 'flips'):
            for seed in (0, 7, 4294967295):
                for weights in ([0], [7], [0, 0, 0], [2, 0, 3], [1 << 63, 1]):
                    with self.subTest(mode=mode, seed=seed, weights=weights):
                        rng = HostRNG(seed)
                        expected = [str(rng.bounded(len(weights)) if mode == 'uniform'
                                        else rng.select(weights)) for _ in range(64)]
                        expected.append('next ' + str(rng.next()))
                        self.assertEqual(self.run_driver('selection', mode, seed, 64, *weights), expected)

    def test_fifo_duplicate_does_not_refresh(self):
        rng = HostRNG(7)
        expected = [('b', 'c')[rng.bounded(2)] for _ in range(32)]
        expected.append('next ' + str(rng.next()))
        self.assertEqual(self.run_driver('fifo'), expected)

    def test_stage_threshold_skip_without_fallback(self):
        self.assertEqual(self.run_driver('stages'), ['PASS stages'])

    def test_stage_reserve_only_refills_empty_roster(self):
        self.assertEqual(self.run_driver('reserves'), ['PASS reserves'])

    def test_memory_and_weight_overflow(self):
        self.assertEqual(self.run_driver('limits'), ['PASS limits'])
