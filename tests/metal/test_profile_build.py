"""Check separate reference/candidate snapshots without compiling or dispatching."""
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))


class ProfileBuildIsolation(unittest.TestCase):
    def test_candidate_storage_and_kernel_selection_stay_out_of_reference(self):
        for lanes in (None, 1, 32):
            with self.subTest(lanes=lanes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                reference, candidate = root / 'reference', root / 'candidate'
                for directory, marker in ((reference, 'REFERENCE'), (candidate, 'CANDIDATE')):
                    shutil.copytree(ROOT / 'src/metal', directory)
                    for name in ('core.h', 'runtime.mm', 'kernels.metal'):
                        path = directory / name
                        path.write_text(path.read_text() + '\n// ' + marker + '\n')
                if lanes is not None:
                    (candidate / 'storage.json').write_text(json.dumps({
                        'padded_multiple': 32, 'threads_per_walk': lanes,
                        'buffer_bytes_per_worker': {'13': 6000, '14': 256}}))
                kernel = 'randomWalkCompactKernel' if lanes is None else 'randomWalkKernel'
                output = root / 'generated'
                commands = []

                def fake_compile(command, **kwargs):
                    self.assertEqual(command[:2], ['xcrun', 'clang++'])
                    self.assertTrue(kwargs.get('check'))
                    commands.append(command)
                    Path(command[command.index('-o') + 1]).write_bytes(b'fake compilation output')
                    return subprocess.CompletedProcess(command, 0)

                argv = ['build.py', '--output', str(output), '--source', str(reference),
                        '--gpu-source', str(candidate), '--gpu-kernel', kernel,
                        '--rank-capacity', '350']
                with patch.object(sys, 'argv', argv), patch('subprocess.run', side_effect=fake_compile):
                    runpy.run_path(str(ROOT / 'benchmarks/metal/build.py'), run_name='__main__')
                diagnostic = (output / 'source/runtime.mm').read_text()
                matched = (output / 'source/matched_runtime.mm').read_text()
                self.assertIn('// REFERENCE', diagnostic)
                self.assertNotIn('// CANDIDATE', diagnostic)
                self.assertIn('// CANDIDATE', matched)
                self.assertNotIn('// REFERENCE', matched)
                self.assertNotIn('id<MTLBuffer> storage13;', diagnostic)
                self.assertNotIn('physicalThreads *= 32;', diagnostic)
                if lanes is not None:
                    self.assertIn('id<MTLBuffer> storage13;', matched)
                    self.assertIn('[encoder setBuffer:r.storage14 offset:0 atIndex:14];', matched)
                self.assertEqual('physicalThreads *= 32;' in matched, lanes == 32)
                self.assertEqual(len(commands), 4)
                for command in (commands[0], commands[2]):
                    self.assertIn('-I' + str(output / 'source'), command)
                    self.assertNotIn('-I' + str(output / 'production'), command)
                selector = '-DMETAL_BENCH_WALK_KERNEL="' + kernel + '"'
                self.assertNotIn(selector, commands[0])
                self.assertIn(selector, commands[2])
                self.assertIn('-DMETAL_SOURCE_DIR="' + str(output / 'source') + '"', commands[1])
                self.assertIn('-DMETAL_SOURCE_DIR="' + str(output / 'production') + '"', commands[3])
                self.assertIn(str(output / 'source/matched_runtime.mm'), commands[3])
                self.assertEqual((output / 'production/kernels.metal').read_bytes(),
                                 (candidate / 'kernels.metal').read_bytes())
                self.assertIn('// REFERENCE', (output / 'source/kernels.metal').read_text())
                self.assertNotIn('// CANDIDATE', (output / 'source/kernels.metal').read_text())
                self.assertTrue(json.loads((output / 'build.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
