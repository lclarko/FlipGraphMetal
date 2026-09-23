"""Host-only checks of smoke supervision and retained evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import smoke


class SmokeEvidence(unittest.TestCase):
    def test_existing_output_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "results.json"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                smoke.main(["--binary-dir", directory, "--signed-fixtures", directory,
                            "--f2-fixtures", directory, "--fixture-receipt", str(marker),
                            "--output", directory])
            self.assertEqual(marker.read_text(), "preserve")

    def test_receipt_detects_original_and_converted_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("input.json", "input.txt", "minimizer.txt"):
                (root / name).write_text("input")
            fixture = {"n": [1, 1, 1], "m": 1, "u": [[1]], "v": [[1]], "w": [[1]]}
            (root / "transform-0.json").write_text(json.dumps(fixture))
            expected = smoke.fixture_hashes(root, root)
            converted = root / "f2.txt"
            converted.write_bytes(smoke.converted_f2(root / "transform-0.json"))
            self.assertEqual(smoke.check_fixtures(root, root, expected, converted), expected)
            converted.write_text("different")
            with self.assertRaisesRegex(ValueError, "converted F2"):
                smoke.check_fixtures(root, root, expected, converted)
            (root / "input.txt").write_text("different")
            with self.assertRaisesRegex(ValueError, "fixture receipt mismatch"):
                smoke.check_fixtures(root, root, expected)

    def test_bad_receipt_prevents_child_launch_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("input.json", "input.txt", "minimizer.txt"):
                (root / name).write_text("input")
            fixture = {"n": [1, 1, 1], "m": 1, "u": [[1]], "v": [[1]], "w": [[1]]}
            (root / "transform-0.json").write_text(json.dumps(fixture))
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(smoke.fixture_hashes(root, root)))
            (root / "minimizer.txt").write_text("changed")
            output = root / "attempt"
            with patch("smoke.subprocess.run") as launch, self.assertRaisesRegex(ValueError, "receipt mismatch"):
                smoke.main(["--binary-dir", directory, "--signed-fixtures", directory,
                            "--f2-fixtures", directory, "--fixture-receipt", str(receipt),
                            "--output", str(output)])
            launch.assert_not_called()
            result = json.loads((output / "results.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["runs"], [])
            self.assertIn("receipt mismatch", result["error"])

    def test_working_directory_is_forwarded_to_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working = root / "unrelated location"
            working.mkdir()
            for name in ("input.json", "transform-0.json", "receipt.json"):
                (root / name).write_text("{}")
            with patch("smoke.check_fixtures"), patch("smoke.verify"), \
                    patch("smoke.converted_f2", return_value=b"fixture"), \
                    patch("smoke.execute", side_effect=RuntimeError("stop before launch")) as execute:
                with self.assertRaisesRegex(RuntimeError, "stop before launch"):
                    smoke.main(["--binary-dir", directory, "--signed-fixtures", directory,
                                "--f2-fixtures", directory, "--fixture-receipt", str(root / "receipt.json"),
                                "--output", str(root / "attempt"), "--working-dir", str(working)])
            self.assertEqual(execute.call_args.args[1], working.resolve())
            summary = json.loads((root / "attempt/results.json").read_text())
            self.assertEqual(summary["working_directory"], str(working.resolve()))

    def test_gpu_evidence_requires_apple_and_positive_finite_timing(self):
        valid = "Metal device: Apple M1\nMetal dispatch example: 4 threads, 0.01 ms GPU\n"
        self.assertEqual(smoke.dispatch_evidence(valid)["dispatches"], [("example", 4, .01)])
        for bad in ("", valid.replace("Apple M1", "Mock"), valid.replace("0.01", "0"),
                    valid.replace("0.01", "nan"), valid.replace("0.01", "inf"),
                    valid.replace("0.01", "-1"), valid.replace("4 threads", "0 threads")):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                smoke.dispatch_evidence(bad)

    def test_execute_inherits_group_and_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = {"complete": False, "runs": []}
            def child(*args, **kwargs):
                self.assertNotIn("start_new_session", kwargs)
                self.assertNotIn("timeout", kwargs)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                kwargs["stdout"].write(b"partial\n")
                self.assertFalse(json.loads((output / "results.json").read_text())["runs"][0]["complete"])
                return subprocess.CompletedProcess(args[0], 1)
            with patch("smoke.subprocess.run", side_effect=child), self.assertRaises(RuntimeError):
                smoke.execute(["example"], output, output, summary, "child.log")
            self.assertEqual((output / "child.log").read_bytes(), b"partial\n")
            self.assertFalse(json.loads((output / "results.json").read_text())["complete"])

    def test_expected_rejection_requires_the_error_and_exit_code(self):
        for code, message, accepted in ((1, "capacity exceeded", True), (0, "capacity exceeded", False),
                                         (1, "different failure", False), (-9, "capacity exceeded", False)):
            with self.subTest(code=code, message=message), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                def child(*args, **kwargs):
                    kwargs["stdout"].write(message.encode())
                    return subprocess.CompletedProcess(args[0], code)
                with patch("smoke.subprocess.run", side_effect=child):
                    arguments = (["example"], output, output, {"runs": []}, "child.log", "capacity exceeded")
                    if accepted:
                        self.assertEqual(smoke.execute(*arguments)["expected_rejection"], "capacity exceeded")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "expected explicit rejection"):
                            smoke.execute(*arguments)

    def test_group_termination_retains_flushed_child_log_and_incomplete_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            module_dir = str(Path(smoke.__file__).parent)
            code = (
                "import sys; from pathlib import Path; "
                "sys.path.insert(0, sys.argv[1]); import smoke; "
                "output=Path(sys.argv[2]); "
                "smoke.execute([sys.executable, '-c', "
                "'import time; print(\"retained partial output\", flush=True); time.sleep(30)'], "
                "output, output, {'complete':False,'runs':[]}, 'child.log')"
            )
            process = subprocess.Popen([sys.executable, "-c", code, module_dir, str(output)],
                                       start_new_session=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                log = output / "child.log"
                while time.monotonic() < deadline:
                    if log.exists() and b"retained partial output" in log.read_bytes():
                        break
                    time.sleep(.02)
                else:
                    self.fail("child failed to flush output")
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2)
                self.assertIn(b"retained partial output", log.read_bytes())
                receipt = json.loads((output / "results.json").read_text())
                self.assertFalse(receipt["complete"])
                self.assertFalse(receipt["runs"][0]["complete"])
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
