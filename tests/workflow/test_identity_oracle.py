"""Known-answer encoding tests independent of a future native adapter."""
import copy
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).parent))
import identity_oracle as oracle


class IdentityTests(unittest.TestCase):
    def test_known_answers(self):
        scalar = oracle.schoolbook((1,1,1))
        f2 = oracle.schoolbook((1,1,1),'F2')
        zero = copy.deepcopy(scalar)
        zero['rank']=2
        for key, value in zip('uvw',(0,1,1)): zero[key].append([value])
        repeated=copy.deepcopy(scalar)
        repeated['rank']=3
        for key in 'uvw': repeated[key] += [[1],[1]]
        repeated['u'][2]=[-1]
        cases=[(scalar,'ba7df0b9cf2c12c6511161967cc2606abd4249607645d6d591ff606a26846765'),
               (f2,'efd4dd3be39f198b2b7145c01bcd7f4344184247fd8725d6d00ec5485fd9c0a6'),
               (zero,'42f0021f81254bf76637ad6bbb8563d32ff33ecfb126218cabd59ac6a31026b7'),
               (repeated,'19d6ac0cb279bba477f23b0e39fbd9204e69af4015b0244ca5abf62f5367d468'),
               (oracle.schoolbook((2,3,4)),'0dd72c490cf1f6504bc79171b9b19b1ee4cfd931262bf22f46267cb4ce0ec5d6'),
               (oracle.schoolbook((2,3,4),'F2'),'1724d0275f23a619be93f7317a1bd8b3879fda070b2fc8bfc3b5e23ad81d34c5')]
        for data, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(oracle.identity(data),'fgm-scheme-v1:'+expected)

    def test_magic_and_numeric_gauge(self):
        data=oracle.schoolbook((1,1,1))
        data['u']=[[-1]]; data['w']=[[-1]]
        ordered=oracle.encoded(data,False)
        canonical=oracle.encoded(data,True)
        self.assertEqual(ordered[:11],b'FGMFACTORS\0')
        self.assertEqual(canonical[:10],b'FGMSCHEME\0')
        self.assertEqual(ordered[-3:],bytes((255,1,255)))
        self.assertEqual(canonical[-3:],bytes((1,1,1)))

    def test_order_gauge_and_nonmutation(self):
        data=oracle.schoolbook((2,3,4))
        before=copy.deepcopy(data)
        alias=copy.deepcopy(data)
        for key in 'uvw': alias[key].reverse()
        for key in 'uv': alias[key][0]=[-v for v in alias[key][0]]
        self.assertEqual(oracle.identity(data),oracle.identity(alias))
        self.assertNotEqual(oracle.identity(data,False),oracle.identity(alias,False))
        self.assertEqual(data,before)

    def test_selector_framing(self):
        expected=(b'FGMSELECT\0\x01\x00'+bytes(8)+b'\x0c\0\0\0presentation'
                  +b'\x02\0\0\0ab'+b'\x01\0\0\0c')
        self.assertEqual(oracle.selection_bytes(0,'ab','c'),expected)
        self.assertNotEqual(expected,oracle.selection_bytes(0,'a','bc'))
        self.assertNotEqual(oracle.selection_bytes(1,'x','é'),oracle.selection_bytes(1,'x','e\u0301'))
        ids=['a','b','c']
        self.assertEqual(sorted(ids,key=lambda i:oracle.selection_key(7,'ns',i)),
                         sorted(reversed(ids),key=lambda i:oracle.selection_key(7,'ns',i)))

    def test_invalid_encoding_inputs(self):
        for field, bad in [('rank',True),('rank',0),('domain','Z3'),('orientation','row-major-w')]:
            data=oracle.schoolbook((1,1,1)); data[field]=bad
            with self.subTest(field=field,bad=bad),self.assertRaises(ValueError): oracle.identity(data)
        data=oracle.schoolbook((1,1,1),'F2'); data['u'][0][0]=-1
        with self.assertRaises(ValueError): oracle.identity(data)
        with self.assertRaises(ValueError): oracle.selection_bytes(1<<64,'x','y')
        with self.assertRaises(ValueError): oracle.selection_bytes(True,'x','y')


if __name__=='__main__': unittest.main()
