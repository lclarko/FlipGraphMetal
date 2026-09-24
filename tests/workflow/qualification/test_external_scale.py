"""Explicit external-library scale qualification, separate from routine tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TESTS = Path(__file__).resolve().parents[1]
ROOT = TESTS.parents[1]
sys.path.insert(0, str(TESTS))
import identity_oracle as oracle

BINARY = Path(os.environ.get('FGM_SCHEME_TOOL', ROOT/'build/metal/scheme_tool'))


class ExternalScaleQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise RuntimeError('build native scheme_tool before scale qualification')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_bounded_synthetic_inventory_scale(self):
        data=oracle.schoolbook((1,1,1))
        source=self.root/'shared.json';source.write_text(json.dumps(data))
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        for size in (1000,10000,100000):
            manifest=self.root/f'inventory-{size}.jsonl'
            with manifest.open('w') as stream:
                for i in range(size):
                    row=dict(schema='fgm-collection-v1',namespace='synthetic',id=f'p{i}',path=source.name,
                             sha256=source_hash,format='json',domain='ZT')
                    stream.write(json.dumps(row,separators=(',',':'))+'\n')
            output=self.root/f'selection-{size}.jsonl'
            run=subprocess.run([str(BINARY),'select','--input',str(manifest),'--output',str(output),
                                '--count','3','--seed','7','--selection-memory','67108864'],
                               capture_output=True,text=True,timeout=45)
            self.assertEqual(run.returncode,0,run.stderr)
            records=[json.loads(line) for line in output.read_text().splitlines()]
            expected=sorted((f'p{i}' for i in range(size)),key=lambda id:oracle.selection_key(7,'synthetic',id))[:3]
            self.assertEqual([record['source_binding']['id'] for record in records],expected)
            self.assertEqual(len({record['scheme_id'] for record in records}),1)


if __name__ == '__main__':
    unittest.main()
