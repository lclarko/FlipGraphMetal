"""Compare the independent host engine to the native standard engine."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from host_rng import HostRNG,U64


class HostRNGTests(unittest.TestCase):
    def test_published_default_sequence(self):
        r=HostRNG(5489)
        self.assertEqual([r.next() for _ in range(3)],
                         [14514284786278117030,4620546740167642908,13109570281517897720])

    def test_native_standard_engine(self):
        binary = Path(os.environ.get('FGM_HOST_RNG_DRIVER',
                                     Path(__file__).resolve().parents[2]/'build/workflow/test_host_rng'))
        actual = list(map(int, subprocess.check_output([str(binary)], text=True, timeout=5).split()))
        expected=[]
        for seed in (0,7,(1<<32)-1):
            r=HostRNG(seed);expected.extend(r.next() for _ in range(700))
        self.assertEqual(actual,expected)

    def test_one_parent_and_all_zero_weights(self):
        r=HostRNG(0)
        self.assertEqual(r.select([1]),0);self.assertEqual(r.draws,1)
        a,b=HostRNG(7),HostRNG(7)
        self.assertEqual(a.select([0]*3),b.bounded(3))
        self.assertEqual(a.words,b.words)

    def test_rejection_boundary_and_exhaustion(self):
        r=HostRNG(0)
        words=iter([0,5])
        r.next=lambda:next(words)
        self.assertEqual(r.bounded(3),2)  # threshold=1; zero rejected
        r.next=lambda:0
        with self.assertRaises(RuntimeError):r.bounded(3)
        words=iter([0])
        r.next=lambda:next(words)
        self.assertEqual(r.bounded(1),0)

    def test_weight_boundary(self):
        r=HostRNG(0)
        r.bounded=lambda bound:2
        self.assertEqual(r.select([2,0,3]),2)
        with self.assertRaises(ValueError):r.select([U64,1])
        with self.assertRaises(ValueError):r.select([])
        with self.assertRaises(ValueError):HostRNG(True)
        with self.assertRaises(ValueError):HostRNG(1<<32)


if __name__=='__main__':unittest.main()
