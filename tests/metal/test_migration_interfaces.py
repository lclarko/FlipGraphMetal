import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from freeze_application import parser_files
from screen import check_log, check_manifest
from application import digest, source_identity


class MigrationInterfaces(unittest.TestCase):
    def test_packed_selector_rejects_general_dispatch_and_nonfinite_time(self):
        log = 'Metal dispatch randomWalkKernel: 32 threads, 1 ms GPU\nROUND 0 CPU 1 METAL_WALL 1 MATCH\n'
        with self.assertRaisesRegex(RuntimeError, 'kernel'):
            check_log(log, 1, 'randomWalkCompactKernel')
        for value in ['nan', 'inf', '-1', '0']:
            with self.assertRaises(RuntimeError):
                check_log(log.replace('1 ms', value + ' ms'), 1)
        self.assertEqual(check_log(log.replace('randomWalkKernel', 'randomWalkCompactKernel'), 1, 'randomWalkCompactKernel')[1], ['0'])

    def test_manifest_kernel_capacity_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'matched').write_bytes(b'executable fixture')
            (path / 'shader.metal').write_text('shader fixture')
            data = dict(complete=True, gpu_kernel='randomWalkCompactKernel', rank_capacity=350,
                        binary_sha256={'matched': digest(path / 'matched')}, snapshot_files=source_identity(path))
            for key, value in [('complete', False), ('gpu_kernel', 'randomWalkKernel'), ('rank_capacity', 32)]:
                (path / 'build.json').write_text(json.dumps({**data, key: value}))
                with self.assertRaises(ValueError):
                    check_manifest(path, 'randomWalkCompactKernel')
            (path / 'build.json').write_text(json.dumps(data))
            check_manifest(path, 'randomWalkCompactKernel')
            (path / 'matched').write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'executable'):
                check_manifest(path, 'randomWalkCompactKernel')
            (path / 'matched').write_bytes(b'executable fixture')
            (path / 'shader.metal').write_text('replaced')
            with self.assertRaisesRegex(ValueError, 'snapshots'):
                check_manifest(path, 'randomWalkCompactKernel')

    def test_profiling_build_rejects_existing_output_without_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / 'keep'
            marker.write_text('retained')
            result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/metal/build.py'), '--output', str(output)], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(), 'retained')
            self.assertFalse((output / 'source').exists())

    def test_snapshot_parser_must_belong_to_selected_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / 'src/metal').mkdir(parents=True)
            (project / 'src/metal/main.cpp').write_text('#include "../common/arg_parser.cpp"')
            with self.assertRaises(ValueError):
                parser_files(project)
            for folder, names in [('common', ('arg_parser.cpp', 'arg_parser.h')), ('entities', ('arg_parser.cu', 'arg_parser.cuh'))]:
                (project / 'src' / folder).mkdir()
                for name in names:
                    (project / 'src' / folder / name).write_text('')
                if folder == 'common':
                    self.assertEqual(len(parser_files(project)), 2)
                else:
                    with self.assertRaisesRegex(ValueError, 'exactly one'):
                        parser_files(project)


if __name__ == '__main__':
    unittest.main()
