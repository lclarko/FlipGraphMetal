import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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

    def test_failed_frozen_shader_build_retains_intent_and_log(self):
        import freeze_application
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project, output = root / 'project', root / 'attempt'
            (project / 'src/metal').mkdir(parents=True)
            (project / 'src/common').mkdir()
            (project / 'scripts').mkdir()
            (project / 'src/metal/main.cpp').write_text('#include "../common/arg_parser.cpp"')
            (project / 'src/metal/runtime.mm').write_text('METAL_LIBRARY_SHA256')
            for name in ('arg_parser.cpp', 'arg_parser.h'):
                (project / 'src/common' / name).write_text('fixture parser')
            for name in ('metal_library.py', 'build_config.py'):
                (project / 'scripts' / name).write_text('fixture helper')
            def failed_compile(command, evidence):
                manifest = json.loads((output / 'build.json').read_text())
                self.assertFalse(manifest['complete'])
                self.assertEqual(manifest['shader_build_command'], command)
                self.assertEqual(manifest['library_mode'], 'metallib')
                evidence.mkdir()
                (evidence / 'run.log').write_text('retained compiler failure')
                (evidence / 'result.json').write_text(json.dumps(dict(argv=command, complete=False)))
                return {'complete': False}
            argv = ['freeze_application.py', '--project-root', str(project), '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(freeze_application, 'run', side_effect=failed_compile) as run:
                with self.assertRaisesRegex(SystemExit, 'shader build incomplete'):
                    freeze_application.main()
                self.assertEqual(run.call_count, 1)
            self.assertEqual((output / 'shader-compile/run.log').read_text(), 'retained compiler failure')
            self.assertFalse((output / 'compile').exists())
            self.assertFalse(json.loads((output / 'build.json').read_text())['complete'])

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
