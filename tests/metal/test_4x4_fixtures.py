import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from verify import verify

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("make_4x4_fixtures", ROOT / "benchmarks/metal/make_4x4_fixtures.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class Fixtures4x4Tests(unittest.TestCase):
    def test_exact_tensor_and_domain(self):
        for make, rank in ((generator.naive, 64), (generator.strassen, 49)):
            data = make()
            self.assertEqual(data["n"], [4,4,4])
            self.assertEqual(verify(data)["equations"], 4096)
            self.assertEqual(data["m"], rank)
            self.assertFalse(data["z2"])
            self.assertTrue(all(x in (-1,0,1) for key in "uvw" for row in data[key] for x in row))

    def test_checked_in_bytes_and_provenance(self):
        generated = generator.artifacts()
        for name, raw in generated.items():
            self.assertEqual((ROOT / "tests/metal/fixtures" / name).read_bytes(), raw)
            if name.endswith(".provenance.json"):
                metadata = json.loads(raw)
                self.assertEqual(metadata["sha256"], hashlib.sha256(generated[metadata["fixture"]]).hexdigest())
        self.assertEqual(generated, generator.artifacts())

    def test_output_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixtures"
            generator.write_fixtures(output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                generator.write_fixtures(output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_malformed_tensor_rejected(self):
        original = generator.strassen()
        mutations = [lambda d: d["n"].__setitem__(0, 0),
                     lambda d: d["u"][0].__setitem__(0, 2),
                     lambda d: d["v"].pop(),
                     lambda d: d["w"][0].__setitem__(0, -d["w"][0][0])]
        for mutate in mutations:
            data = copy.deepcopy(original)
            mutate(data)
            with self.assertRaises(ValueError):
                generator.encode(data)

    def test_raw_format_has_no_count_prefix(self):
        for make, rank in ((generator.naive, 64), (generator.strassen, 49)):
            tokens = list(map(int, generator.encode(make()).split()))
            self.assertEqual(tokens[:4], [4,4,4,rank])
            self.assertEqual(len(tokens), 4 + 3*rank*16)


if __name__ == "__main__":
    unittest.main()
