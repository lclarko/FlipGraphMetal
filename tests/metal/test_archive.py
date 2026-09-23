from pathlib import Path
import json
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/metal'))
import application
from archive import ArchiveResolver
from summarize_application import load_directory
import summarize_application


class ArchiveEvidence(unittest.TestCase):
    def inventory(self, root):
        return dict(files={str(path.relative_to(root)): application.digest(path)
                           for path in root.rglob('*') if path.is_file()},
                    directories=sorted(str(path.relative_to(root)) for path in root.rglob('*') if path.is_dir()))

    def fixture(self, root, backend='candidate'):
        source = root / 'source'
        source.mkdir()
        (source / 'kernels.metal').write_text('kernel void example() {}')
        binary = root / 'program'
        binary.write_bytes(str(source).encode())
        manifest = root / 'build.json'
        application.write_json(manifest, dict(version=1, source_root=str(source),
            source_files=application.source_identity(source), binary=str(binary),
            binary_sha256=application.digest(binary), source_revision='fixture',
            metal_source_dir=str(source), commands=[['clang++', '-DMETAL_SOURCE_DIR="' + str(source) + '"']]))
        identity = application.build_identity(backend, binary, source, manifest)
        run = root / 'run'
        run.mkdir()
        application.write_json(run / 'config.json', dict(identities={backend: identity}, cases=[], rounds=2))
        application.write_json(root / 'archive-map.json', dict(version=1,
            paths={str(binary): 'program', str(source): 'source', str(manifest): 'build.json'},
            campaigns={'run': self.inventory(run)}))
        return identity

    def test_relocated_cpu_and_metal_preserve_identity_and_detect_tamper(self):
        for backend in ('cpu', 'candidate'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                original = root / 'original'
                original.mkdir()
                identity = self.fixture(original, backend)
                moved = root / 'moved'
                original.rename(moved)
                resolver = ArchiveResolver(moved / 'archive-map.json')
                self.assertEqual(load_directory(moved / 'run', resolver)[0]['identities'][backend], identity)
                with self.assertRaises(FileNotFoundError):
                    load_directory(moved / 'run')
                for relative in ('program', 'source/kernels.metal', 'build.json', 'run/config.json'):
                    path = moved / relative
                    before = path.read_bytes()
                    path.write_bytes(before + b' ')
                    with self.subTest(relative=relative), self.assertRaises(ValueError):
                        load_directory(moved / 'run', resolver)
                    path.write_bytes(before)
                (moved / 'source/extra.cpp').write_text('extra')
                with self.assertRaises(ValueError):
                    load_directory(moved / 'run', resolver)

    def test_relocated_records_recheck_seals_artifacts_and_case_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root, 'cpu')
            run = root / 'run'
            case = dict(name='cpu-case', fixture='custom', count=1, seed=7, repeat=0, backend='cpu')
            directory = run / case['name']
            directory.mkdir()
            (directory / 'stdout.log').write_text('retained output')
            record = dict(case, complete=True, wall_seconds=1, reports=[.1, .2], steady_steps_per_second=10,
                          artifacts=application.artifacts(directory))
            application.write_json(directory / 'result.json', record)
            (directory / 'result.sha256').write_text(application.digest(directory / 'result.json'))
            config_path = run / 'config.json'
            config = json.loads(config_path.read_text())
            config['cases'] = [case]

            def reseal_config():
                application.write_json(config_path, config)
                mapping = json.loads((root / 'archive-map.json').read_text())
                mapping['campaigns']['run'] = self.inventory(run)
                application.write_json(root / 'archive-map.json', mapping)
                return ArchiveResolver(root / 'archive-map.json')

            resolver = reseal_config()
            self.assertEqual(len(load_directory(run, resolver)[1]), 1)
            with patch.object(sys, 'argv', ['summary', str(run), '--archive-map', str(root / 'archive-map.json')]), contextlib.redirect_stdout(io.StringIO()) as output:
                summarize_application.main()
            summary = json.loads(output.getvalue())
            self.assertEqual(summary['complete_cases'], 1)
            for key in ('milestone', 'implementation_promotion', 'secondary_nonregression_95'):
                self.assertNotIn(key, summary)
            for relative in ('stdout.log', 'result.json', 'result.sha256'):
                path = directory / relative
                before = path.read_bytes()
                path.write_bytes(before + b'x')
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    load_directory(run, resolver)
                path.write_bytes(before)
            trusted_map = (root / 'archive-map.json').read_bytes()
            original_record = (directory / 'result.json').read_bytes()
            original_seal = (directory / 'result.sha256').read_bytes()
            record['steady_steps_per_second'] = 10000000
            application.write_json(directory / 'result.json', record)
            (directory / 'result.sha256').write_text(application.digest(directory / 'result.json'))
            with self.assertRaisesRegex(ValueError, 'trusted archive inventory'):
                load_directory(run, ArchiveResolver(root / 'archive-map.json'))
            self.assertEqual((root / 'archive-map.json').read_bytes(), trusted_map)
            (directory / 'result.json').write_bytes(original_record)
            (directory / 'result.sha256').write_bytes(original_seal)
            for extra in (run / 'extra.txt', directory / 'extra.txt'):
                extra.write_text('unrecorded')
                with self.assertRaisesRegex(ValueError, 'trusted archive inventory'):
                    load_directory(run, resolver)
                extra.unlink()
            (run / 'extra-case').mkdir()
            with self.assertRaisesRegex(ValueError, 'trusted archive inventory'):
                load_directory(run, resolver)
            (run / 'extra-case').rmdir()
            (directory / 'result.json').unlink()
            with self.assertRaisesRegex(ValueError, 'trusted archive inventory'):
                load_directory(run, resolver)
            (directory / 'result.json').write_bytes(original_record)
            config['cases'][0]['name'] = '../outside'
            with self.assertRaisesRegex(ValueError, 'unsafe case'):
                load_directory(run, reseal_config())

    def test_mapping_cannot_override_recorded_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            (root / 'other').write_text('changed binary')
            path = root / 'archive-map.json'
            mapping = json.loads(path.read_text())
            mapping['paths'][str(root / 'program')] = 'other'
            application.write_json(path, mapping)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_directory(root / 'run', ArchiveResolver(path))

    def test_traversal_absolute_paths_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            path = root / 'archive-map.json'
            original = json.loads(path.read_text())
            for unsafe in ('../outside', '/tmp/outside'):
                mapping = dict(original, paths={'/old/program': unsafe})
                application.write_json(path, mapping)
                with self.assertRaisesRegex(ValueError, 'relative paths'):
                    ArchiveResolver(path)
            application.write_json(path, original)
            for destination in (root / 'program', root.parent):
                link = root / 'source/link'
                link.symlink_to(destination)
                with self.assertRaisesRegex(ValueError, 'symlinks'):
                    ArchiveResolver(path)
                link.unlink()
            resolver = ArchiveResolver(path)
            with self.assertRaisesRegex(ValueError, 'explicit archive mapping'):
                resolver.resolve(Path('/not/mapped'))


class HardwarePrivacy(unittest.TestCase):
    def test_allowlist_excludes_identifiers_and_nested_displays(self):
        raw = {'SPHardwareDataType': [{'machine_model': 'MacBookPro17,1', 'chip_type': 'Apple M1',
                'physical_memory': '16 GB', 'serial_number': 'SECRET', 'platform_UUID': 'SECRET'}],
               'SPDisplaysDataType': [{'sppci_model': 'Apple M1', 'spdisplays_cores': '8',
                'spdisplays_ndrvs': [{'serial': 'SECRET'}], 'device-id': 'SECRET'}]}
        completed = subprocess.CompletedProcess([], 0, json.dumps(raw), 'SECRET stderr')
        with patch.object(application.subprocess, 'run', return_value=completed):
            result = application.hardware_inventory()
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertEqual(result['fields']['SPHardwareDataType'][0]['chip_type'], 'Apple M1')

    def test_failures_never_preserve_raw_output_or_exception_text(self):
        for capture in (subprocess.CompletedProcess([], 1, 'SECRET', 'SECRET'),
                        subprocess.CompletedProcess([], 0, 'SECRET invalid json', 'SECRET'),
                        subprocess.CompletedProcess([], 0, '[]', 'SECRET')):
            with patch.object(application.subprocess, 'run', return_value=capture):
                self.assertNotIn('SECRET', json.dumps(application.hardware_inventory()))
        for error in (OSError('SECRET'), subprocess.TimeoutExpired(['SECRET'], 10, output='SECRET')):
            with patch.object(application.subprocess, 'run', side_effect=error):
                self.assertNotIn('SECRET', json.dumps(application.hardware_inventory()))


if __name__ == '__main__':
    unittest.main()
