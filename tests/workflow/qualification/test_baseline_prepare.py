"""Source-only qualification of public baseline preparation."""
import importlib.util
import json
import io
import tarfile
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('baseline_prepare', ROOT/'benchmarks/workflow/baseline.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def source_archive(extra=(), omitted=()):
    entries = [('src/metal/runtime.mm', b'runtime'), ('makefile', b'all:\n')]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as stream:
        for name, data in entries + list(extra):
            if name in omitted:
                continue
            if isinstance(data, tarfile.TarInfo):
                stream.addfile(data)
            else:
                info = tarfile.TarInfo(name)
                info.mode = 0o755 if name.endswith('.mm') else 0o644
                info.size = len(data)
                stream.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class ArchiveSafety(unittest.TestCase):
    def test_reject_unsafe_duplicate_special_and_missing_members(self):
        for name in ('../escape', '/absolute', 'src/../escape', './makefile',
                     'build/evidence', '.git/config', 'makefile'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                b.archive_files(source_archive([(name, b'bad')]))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            info = tarfile.TarInfo('unsafe')
            info.type = kind
            info.linkname = 'makefile'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                b.archive_files(source_archive([('unsafe', info)]))
        for missing in ('makefile', 'src/metal/runtime.mm'):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                b.archive_files(source_archive(omitted=[missing]))

    def test_extract_and_compare_complete_source_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = source_archive()
            (folder/'source.tar').write_bytes(data)
            freeze = {'commit': b.BASELINE, 'archive_sha256': b.digest(folder/'source.tar')}
            source = folder/'source'
            b.extract_source(b.archive_files(data), source)
            self.assertEqual((source/'src/metal/runtime.mm').stat().st_mode & 0o777, 0o755)
            b.assert_source_snapshot(source, freeze)
            with self.assertRaises(FileExistsError):
                b.extract_source(b.archive_files(data), source)
            for alteration in ('extra', 'missing', 'changed', 'linked'):
                with self.subTest(alteration=alteration):
                    target = source/'makefile'
                    if alteration == 'extra':
                        target = source/'untracked.txt'
                        target.write_bytes(b'extra')
                    elif alteration == 'missing':
                        target.unlink()
                    elif alteration == 'changed':
                        target.write_bytes(b'changed')
                    else:
                        target.unlink()
                        target.symlink_to(source/'src/metal/runtime.mm')
                    with self.assertRaises(ValueError):
                        b.assert_source_snapshot(source, freeze)
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    (source/'makefile').write_bytes(b'all:\n')


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
