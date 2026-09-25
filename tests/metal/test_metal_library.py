"""Host-only tests for shader assembly, receipts, and package bindings."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import metal_library as library


class MetalLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.names = ['core.h', 'addition.h', 'flip_set.h', 'scheme_integer.h',
                      'scheme_z2.h', 'pairs_counter.h', 'additions_reducer.h',
                      'compact.h', 'controlled.h', 'controlled_capture.h', 'controlled_packed.h', 'kernels.metal', 'controlled_kernels.metal', 'reduction_kernels.metal', 'test_kernels.metal']
        for name in self.names:
            (self.source / name).write_text('#pragma once\n' + name)
        self.compiler = self.root / 'fake_compiler.py'
        self.compiler.write_text('#!' + sys.executable + '\n' + '''import pathlib, sys
root = pathlib.Path(__file__).parent
with (root / 'calls').open('a') as stream: stream.write('call\\n')
if (root / 'fail').exists(): sys.exit(2)
source = pathlib.Path(sys.argv[-3]).read_bytes()
pathlib.Path(sys.argv[-1]).write_bytes(b'fake-metallib:' + source)
''')
        self.compiler.chmod(0o755)
        self.compiler_command = shlex.quote(str(self.compiler))
        self.output = self.root / 'bin/shaders/signed.metallib'
        self.header = self.output.with_suffix('.h')

    def build(self, **kwargs):
        return library.build_library(self.source, self.output, self.header,
                                     compiler=self.compiler_command, **kwargs)

    def calls(self):
        return (self.root / 'calls').read_text().splitlines()

    def test_assembly_variants_and_order(self):
        for variant in library.VARIANTS:
            names = self.names[:-1]
            if variant.startswith('f2'):
                names = [name for name in names if name != 'compact.h']
            if variant.endswith('testing'):
                names += ['test_kernels.metal']
            expected = '#define METAL_F2\n' if variant.startswith('f2') else ''
            if variant.endswith('testing'):
                expected += '#define METAL_TESTING\n'
            expected += ''.join('\n' + name + '\n' for name in names)
            self.assertEqual(library.assemble(self.source, variant), expected)
        with self.assertRaises(ValueError):
            library.assemble(self.source, 'other')

    def test_noop_and_header_repair(self):
        result = self.build()
        before = self.output.stat().st_mtime_ns
        self.assertIn('.metal-build-', result['commands'][0][-1])
        receipt = json.loads(self.output.with_name(self.output.name + '.build.json').read_text())
        self.assertEqual(receipt['commands'], result['commands'])
        self.header.write_text('tampered')
        self.assertEqual(result, self.build())
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        self.assertIn(result['sha256'], self.header.read_text())
        self.assertEqual(result['sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_source_flags_and_asset_tamper_rebuild(self):
        self.build()
        (self.source / 'core.h').write_text('changed source')
        self.build()
        self.build(flags='-different')
        self.output.write_bytes(b'tampered')
        self.build(flags='-different')
        self.assertEqual(len(self.calls()), 4)

    def test_failed_compilation_preserves_assets(self):
        self.build()
        paths = [self.output, self.header, self.output.with_suffix('.metal'),
                 self.output.with_name(self.output.name + '.build.json')]
        before = {path: path.read_bytes() for path in paths}
        (self.root / 'fail').touch()
        (self.source / 'core.h').write_text('new source')
        with self.assertRaises(subprocess.CalledProcessError):
            self.build()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_compiler_identity_changes_rebuild(self):
        self.build()
        with self.compiler.open('a') as stream:
            stream.write('\n# changed compiler\n')
        self.build()
        self.assertEqual(len(self.calls()), 2)

    def program_receipt(self, path, variant=None):
        dependencies = {'src/workflow/scheme_tool.cpp': library.digest(b'source')}
        if variant:
            header = path.parent / 'shaders' / (variant + '.h')
            dependencies[str(header)] = library.digest(header.read_bytes())
        stat = path.stat()
        path.with_name(path.name + '.build.json').write_text(json.dumps({
            'inputs': {'command': ['fake-compiler'], 'configuration': {},
                       'dependencies': dependencies},
            'output': {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}))

    def production_inputs(self):
        for variant in ('signed', 'f2'):
            output = self.output.with_name(variant + '.metallib')
            result = library.build_library(self.source, output, output.with_suffix('.h'),
                                           variant=variant, compiler=self.compiler_command)
            for binary, kind in library.PRODUCTION.items():
                if kind == variant:
                    path = self.root / 'bin' / binary
                    path.write_bytes(b'program\0' + result['library'].encode() + b'\0' +
                                     result['sha256'].encode() + b'\0')
                    path.chmod(0o755)
                    self.program_receipt(path, variant)
        for name in library.HOST_PRODUCTION:
            path = self.root / 'bin' / name
            path.write_bytes(b'native host utility')
            path.chmod(0o755)
            self.program_receipt(path)
        (self.root / 'bin/correctness').write_bytes(b'test-only')
        (self.root / 'bin/private-evidence.json').write_text('{}')

    def test_package_selection_and_manifest(self):
        self.production_inputs()
        destination = self.root / 'package'
        manifest = library.package(self.root / 'bin', destination)
        expected = set(library.PRODUCTION) | set(library.HOST_PRODUCTION) | {'shaders/signed.metallib', 'shaders/f2.metallib', 'ATTRIBUTION.md'}
        self.assertEqual(set(manifest['files']), expected)
        self.assertEqual({str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()},
                         expected | {'manifest.json'})
        readme = Path(library.__file__).resolve().parents[1] / 'README.md'
        self.assertIn((destination / 'ATTRIBUTION.md').read_bytes(), readme.read_bytes())
        self.assertTrue((destination / 'ATTRIBUTION.md').read_bytes().startswith(b'## Attribution and rights\n'))
        self.assertNotIn(str(self.root), json.dumps(manifest))
        for name, sha256 in manifest['files'].items():
            self.assertEqual(sha256, hashlib.sha256((destination / name).read_bytes()).hexdigest())
        self.assertTrue((destination / 'flip_graph').stat().st_mode & 0o111)
        with self.assertRaises(ValueError):
            library.package(self.root / 'bin', destination)

    def test_package_requires_authoritative_attribution(self):
        self.production_inputs()
        (self.root / 'README.md').write_text('# Project without attribution\n')
        with patch.object(library, '__file__', str(self.root / 'scripts/metal_library.py')):
            with self.assertRaisesRegex(ValueError, 'Attribution and rights'):
                library.package(self.root / 'bin', self.root / 'package')
        self.assertFalse((self.root / 'package').exists())

    def test_package_rejects_asset_and_program_tamper(self):
        self.production_inputs()
        original = self.output.read_bytes()
        self.output.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'shader/header'):
            library.package(self.root / 'bin', self.root / 'bad-package')
        self.assertFalse((self.root / 'bad-package').exists())
        self.output.write_bytes(original)
        (self.root / 'bin/flip_graph').write_bytes(b'wrong binary')
        with self.assertRaisesRegex(ValueError, 'program/shader'):
            library.package(self.root / 'bin', self.root / 'bad-package')
        self.assertFalse((self.root / 'bad-package').exists())

    def test_package_requires_native_host_tool(self):
        self.production_inputs()
        path = self.root / 'bin/scheme_tool'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'regular program: scheme_tool'):
            library.package(self.root / 'bin', self.root / 'package')
        self.assertFalse((self.root / 'package').exists())

    def test_package_rejects_stale_host_build_receipt(self):
        self.production_inputs()
        (self.root / 'bin/scheme_tool').write_bytes(b'changed utility')
        with self.assertRaisesRegex(ValueError, 'program/build receipt mismatch: scheme_tool'):
            library.package(self.root / 'bin', self.root / 'package')
        self.assertFalse((self.root / 'package').exists())

    def test_package_rejects_wrong_header_build_receipt(self):
        self.production_inputs()
        receipt = self.root / 'bin/flip_graph.build.json'
        data = json.loads(receipt.read_text())
        data['inputs']['dependencies'][str(self.header)] = '0' * 64
        receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'shader header dependency differs'):
            library.package(self.root / 'bin', self.root / 'package')
        self.assertFalse((self.root / 'package').exists())

    def test_library_name_must_be_relative(self):
        for name in ('/absolute', '../outside', 'shaders/../outside', 'shaders/"bad'):
            with self.assertRaises(ValueError):
                self.build(library_name=name)


if __name__ == '__main__':
    unittest.main()
