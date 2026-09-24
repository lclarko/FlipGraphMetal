"""Compare the independent host engine to the native standard engine."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from host_rng import HostRNG,U64


class HostRNGTests(unittest.TestCase):
    def test_published_default_sequence(self):
        r=HostRNG(5489)
        self.assertEqual([r.next() for _ in range(3)],
                         [14514284786278117030,4620546740167642908,13109570281517897720])

    def test_native_standard_engine(self):
        code='''#include <random>
#include <iostream>
int main() { for (unsigned s : {0u,7u,4294967295u}) {
std::mt19937_64 r(s); for (int i=0;i<700;i++) std::cout << r() << "\\n";
} }
'''
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'rng.cpp'; exe=Path(directory)/'rng'
            source.write_text(code)
            subprocess.run(['xcrun','clang++','-std=c++17',str(source),'-o',str(exe)],check=True,capture_output=True)
            actual=list(map(int,subprocess.check_output([str(exe)],text=True).split()))
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
