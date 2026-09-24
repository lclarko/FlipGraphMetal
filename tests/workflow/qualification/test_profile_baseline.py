"""Host-only diagnostic snapshot isolation and fail-closed patch tests."""
import importlib.util
import io
import json
from functools import lru_cache
import subprocess
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('profile_baseline', ROOT/'benchmarks/workflow/profile_baseline.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


@lru_cache(maxsize=1)
def source():
    return subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{p.BASELINE}:{p.RUNTIME}'],
        text=True, timeout=10)



def archive(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as stream:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            stream.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def baseline(path):
    path.mkdir()
    data = archive({p.RUNTIME: source().encode(), 'makefile': b'all:\n\ttrue\n', 'README.md': b'baseline\n'})
    (path/'source.tar').write_bytes(data)
    (path/'freeze.json').write_text(json.dumps({'commit':p.BASELINE, 'archive_sha256':p.sha(data)}))


class ProfileTests(unittest.TestCase):
    def test_patch_fails_closed(self):
        for altered in (source().replace('#include <limits>', '#include <cstdint>'),
                        source()+'\n#include <limits>\n', p.instrument(source())):
            with self.assertRaises(ValueError):
                p.instrument(altered)

    def test_new_snapshot_original_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            original, output = folder/'baseline', folder/'profile'
            baseline(original)
            before = {path.name:path.read_bytes() for path in original.iterdir()}
            with self.assertRaises(ValueError):
                p.prepare(original, original/'diagnostic')
            result = p.prepare(original, output)
            self.assertEqual(before, {path.name:path.read_bytes() for path in original.iterdir()})
            self.assertEqual(result['build_status'], 'NOT RUN')
            self.assertEqual(result['behavior_status'], 'NOT VERIFIED')
            changed = [name for name in result['original_files']
                       if result['original_files'][name] != result['diagnostic_files'][name]]
            self.assertEqual(changed, [p.RUNTIME])
            self.assertFalse((output/'source/build').exists())
            self.assertEqual((output/'profile.sha256').read_text().strip(), p.sha((output/'profile.json').read_bytes()))
            self.assertEqual(result['patch_sha256'], p.sha((output/'runtime.patch').read_bytes()))
            with self.assertRaises(ValueError):
                p.prepare(original, output)

    def test_archive_safety(self):
        for path in ('../escape', '/absolute', 'build/evidence', '.git/config'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                p.archive_files(archive({path:b'no'}))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as stream:
            info = tarfile.TarInfo('link')
            info.type = tarfile.SYMTYPE
            info.linkname = '/elsewhere'
            stream.addfile(info)
        with self.assertRaises(ValueError):
            p.archive_files(buffer.getvalue())

    def test_wrong_pin_or_hash_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            original, output = folder/'baseline', folder/'profile'
            baseline(original)
            (original/'source.tar').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                p.prepare(original, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
