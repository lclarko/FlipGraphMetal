"""Exercise make dependency decisions with a fake compiler; no Metal workloads."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["flip_graph", "flip_graph_f2", "complexity_minimizer",
           "complexity_minimizer_f2", "additions_reducer", "correctness",
           "f2_correctness", "probe"]


class BuildConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "checkout with spaces and 'quote'"
        self.root.mkdir()
        for name in ("makefile", "scripts/build_config.py", "scripts/metal_library.py"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        for name in ("src/metal/main.cpp", "src/metal/runtime.mm", "src/metal/probe.mm",
                     "src/common/arg_parser.cpp", "src/common/arg_parser.h",
                     "tests/metal/correctness.cpp", "tests/metal/f2_correctness.cpp",
                     "tests/metal/candidate_capacity.h"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.touch()
        for name in (ROOT / "src/metal").iterdir():
            if name.suffix in (".h", ".metal"):
                shutil.copy2(name, self.root / "src/metal" / name.name)
        workflow = self.root / "src/workflow"
        workflow.mkdir(parents=True)
        for name in (ROOT / "src/workflow").iterdir():
            if name.suffix in (".h", ".cpp"):
                shutil.copy2(name, workflow / name.name)
        self.log = self.base / "compiler-log.jsonl"
        self.compiler = self.base / "fake compiler.py"
        self.compiler.write_text("""import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['FAKE_COMPILER_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
pathlib.Path(args[args.index('-o') + 1]).write_text(json.dumps(args))
""")
        self.command = "python3 " + shlex.quote(str(self.compiler))

    def make(self, *settings, success=True):
        result = subprocess.run(
            ["make", *["build/metal/" + name for name in TARGETS],
             "METAL_CXX=" + self.command, "METAL_COMPILER=" + self.command + " --shader", *settings], cwd=self.root,
            env={**os.environ, "FAKE_COMPILER_LOG": str(self.log)},
            text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()
                if "--shader" not in json.loads(line)] if self.log.exists() else []

    def age_outputs(self):
        # Ensure strict timestamp ordering even with older make/filesystem resolution.
        for name in TARGETS:
            os.utime(self.root / "build/metal" / name, (1, 1))

    def test_noop_and_effective_flags(self):
        self.make()
        self.assertEqual(len(self.calls()), 8)
        stamp = self.root / "build/metal/config.json"
        original = stamp.stat().st_mtime_ns
        self.make()
        self.assertEqual(len(self.calls()), 8)
        self.assertEqual(stamp.stat().st_mtime_ns, original)
        self.make("METAL_FLAGS=-O1 -std=c++17")
        self.assertEqual(len(self.calls()), 15)  # Probe flags are separate.
        self.make("METAL_FLAGS=-O1 -std=c++17")
        self.assertEqual(len(self.calls()), 15)
        self.make("METAL_FLAGS=-O1 -std=c++17", "PROBE_FLAGS=-O0")
        self.assertEqual(len(self.calls()), 16)
        self.assertIn("-O0", self.calls()[-1])

    def test_relocated_checkout_rebuilds_without_absolute_runtime_paths(self):
        self.make()
        old = self.root
        self.root = self.base / "relocated checkout 'still quoted'"
        shutil.copytree(old, self.root)
        self.make()
        self.assertTrue(old.exists())
        self.assertEqual(len(self.calls()), 16)
        for args in self.calls()[8:15]:
            self.assertIn('-include', args)
            self.assertTrue(args[args.index('-include') + 1].startswith('build/metal/shaders/'))
            self.assertFalse(any('METAL_SOURCE_DIR' in arg for arg in args))
            self.assertFalse(any(str(old) in arg for arg in args))
        self.make()
        self.assertEqual(len(self.calls()), 16)

    def test_makefile_change_rebuilds(self):
        self.make()
        self.age_outputs()
        with (self.root / "makefile").open("a") as stream:
            stream.write("\n# configuration changed\n")
        self.make()
        self.assertEqual(len(self.calls()), 16)

    def test_compiler_command_change_rebuilds(self):
        self.make()
        self.command += " --extra-compiler-setting"
        self.make()
        self.assertEqual(len(self.calls()), 16)
        self.assertTrue(all("--extra-compiler-setting" in args for args in self.calls()[8:]))

    def test_source_bytes_change_even_with_preserved_timestamp(self):
        self.make()
        source = self.root / "src/metal/main.cpp"
        before = source.stat()
        source.write_text("// changed source\n")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.make()
        self.assertEqual(len(self.calls()), 15)
        self.make()
        self.assertEqual(len(self.calls()), 15)

    def test_workflow_change_rebuilds_only_configured_programs(self):
        self.make()
        source = self.root / "src/workflow/execution.cpp"
        before = source.stat()
        source.write_text(source.read_text() + "\n// changed workflow source\n")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.make()
        calls = self.calls()[8:]
        self.assertEqual({Path(args[args.index('-o') + 1]).name for args in calls},
                         {'flip_graph', 'flip_graph_f2', 'additions_reducer'})
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('src/workflow/scheme_io.cpp' in args and
                            'src/workflow/execution.cpp' in args for args in calls))
        self.make()
        self.assertEqual(len(self.calls()), 11)

    def test_explicit_source_mode_and_return_to_packaged(self):
        self.make()
        self.make("METAL_LIBRARY_MODE=source")
        self.assertEqual(len(self.calls()), 15)
        expected = '-DMETAL_SOURCE_DIR="' + str((self.root / "src/metal").resolve()) + '"'
        for args in self.calls()[8:]:
            self.assertIn(expected, args)
            self.assertNotIn('-include', args)
        self.make("METAL_LIBRARY_MODE=source")
        self.assertEqual(len(self.calls()), 15)
        self.make()
        self.assertEqual(len(self.calls()), 22)
        self.assertTrue(all('-include' in args for args in self.calls()[15:]))

    def test_shader_change_rebinds_programs(self):
        self.make()
        path = self.root / 'src/metal/kernels.metal'
        path.write_text(path.read_text() + '\n// shader change\n')
        self.make()
        self.assertEqual(len(self.calls()), 15)
        self.make()
        self.assertEqual(len(self.calls()), 15)

    def test_missing_compiler_fails_before_compile(self):
        self.make()
        self.command = "missing-compiler-for-config-test"
        self.make(success=False)
        self.assertEqual(len(self.calls()), 8)


if __name__ == "__main__":
    unittest.main()
