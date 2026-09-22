import copy
import unittest
from verify import verify, reconstruct


class VerificationTests(unittest.TestCase):
    def test_integer_and_f2_are_distinct(self):
        data = {"n": [1, 1, 1], "m": 3, "z2": True, "u": [[1]] * 3, "v": [[1]] * 3, "w": [[1]] * 3}
        verify(data)
        data["z2"] = False
        with self.assertRaises(ValueError):
            verify(data)

    def test_signed_circuit(self):
        outputs, count = reconstruct([[{"index": 2, "value": -1}]], [[{"index": 0, "value": 1}, {"index": 1, "value": -1}]], 2, False)
        self.assertEqual(outputs, [[-1, 1]])
        self.assertEqual(count, 1)

    def test_reject_forward_reference(self):
        with self.assertRaises(ValueError):
            reconstruct([], [[{"index": 1, "value": 1}, {"index": 0, "value": 1}]], 1, False)

    def test_reject_wrong_counts_and_tensor(self):
        term = {"index": 0, "value": 1}
        data = {"n": [1, 1, 1], "m": 1, "z2": False, "complexity": {"naive": 0, "reduced": 0}}
        for key in "uvw":
            data[key] = [[term.copy()]]
            data[key + "_fresh"] = []
        verify(data)
        bad = copy.deepcopy(data)
        bad["complexity"]["reduced"] = 1
        with self.assertRaises(ValueError):
            verify(bad)
        bad = copy.deepcopy(data)
        bad["w"][0][0]["value"] = -1
        with self.assertRaises(ValueError):
            verify(bad)


if __name__ == "__main__":
    unittest.main()
