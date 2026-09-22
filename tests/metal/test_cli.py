import subprocess
import unittest
from pathlib import Path


class CommandLineTests(unittest.TestCase):
    def reject(self, program, arguments, message):
        path = Path("build/metal") / program
        if not path.exists():
            self.fail("build Metal executables first: missing " + str(path))
        result = subprocess.run([str(path), *arguments], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stdout + result.stderr)

    def test_unknown_option(self):
        self.reject("flip_graph", ["--imaginary", "1"], "unknown argument")

    def test_zero_workers(self):
        self.reject("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "0"], "unsupported --schemes")

    def test_unsupported_dimensions(self):
        self.reject("flip_graph", ["-n1", "16", "-n2", "16", "-n3", "16"], "unsupported dimensions")

    def test_integer_sandwiching(self):
        self.reject("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--sandwiching-probability", "1"], "unsupported signed-ternary sandwiching")

    def test_bad_count_relationship(self):
        self.reject("additions_reducer", ["-i", "unused", "--count", "1", "--schemes-count", "2"], "--schemes-count must not exceed --count")

    def test_bad_number(self):
        self.reject("complexity_minimizer", ["--input-path", "unused", "--max-iterations", "abc"], "is not natural")


if __name__ == "__main__":
    unittest.main()
