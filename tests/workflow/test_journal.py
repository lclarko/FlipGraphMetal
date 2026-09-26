"""Native journal durability, recovery and bounded historical-index cases."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(os.environ.get('FGM_JOURNAL_DRIVER', ROOT / 'build/workflow/test_journal'))


class JournalTests(unittest.TestCase):
    def test_native_cases(self):
        self.assertTrue(DRIVER.is_file(), 'build native journal test driver first')
        for scenario in ('basic', 'short-writes', 'torn-tail', 'sync-failure',
                         'index-sync-failure', 'recovery-sync-failure', 'head-sync-failure', 'head-required', 'corruption', 'rebuild', 'storage', 'invalid'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as folder:
                result = subprocess.run([str(DRIVER), scenario, str(Path(folder) / 'history')],
                                        text=True, capture_output=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('PASS ' + scenario, result.stdout)


if __name__ == '__main__':
    unittest.main()
