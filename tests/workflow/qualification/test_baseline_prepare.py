"""Source-only qualification of public baseline preparation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('baseline_prepare', ROOT/'benchmarks/workflow/baseline.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class BaselinePreparation(unittest.TestCase):
    def test_exact_public_source_pin_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'baseline'
            receipt = b.freeze_baseline(output)
            self.assertTrue(receipt['complete'])
            self.assertEqual(receipt['source_preparation'], 'complete')
            self.assertEqual(receipt['build_status'], 'NOT RUN')
            self.assertEqual(receipt['measurement_status'], 'NOT RUN')
            self.assertEqual(receipt['commit'], b.BASELINE)
            tree = subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', b.BASELINE+'^{tree}'], text=True).strip()
            self.assertEqual(receipt['tree'], tree)
            expected = subprocess.check_output(
                ['git', '-C', str(ROOT), 'show', b.BASELINE+':src/metal/runtime.mm'])
            self.assertEqual((output/'source/src/metal/runtime.mm').read_bytes(), expected)
            self.assertEqual(receipt['archive_sha256'], b.digest(output/'source.tar'))
            b.assert_source_snapshot(output/'source', receipt)
            self.assertFalse((output/'source/build').exists())
            before = (output/'freeze.json').read_bytes()
            with self.assertRaises(FileExistsError):
                b.freeze_baseline(output)
            self.assertEqual((output/'freeze.json').read_bytes(), before)

    def test_failed_preparation_retains_incomplete_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'failed'
            with mock.patch.object(b.subprocess, 'check_output', side_effect=RuntimeError('missing pin')):
                with self.assertRaises(RuntimeError):
                    b.freeze_baseline(output)
            receipt = json.loads((output/'freeze.json').read_text())
            self.assertFalse(receipt['complete'])
            self.assertEqual(receipt['source_preparation'], 'incomplete')
            self.assertEqual(receipt['measurement_status'], 'NOT RUN')
            self.assertEqual((output/'freeze.sha256').read_text().strip(), b.digest(output/'freeze.json'))


if __name__ == '__main__':
    unittest.main()
