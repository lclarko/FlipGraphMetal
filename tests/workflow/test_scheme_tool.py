"""Native scheme-tool acceptance against independent byte and tensor fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import identity_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('FGM_SCHEME_TOOL', ROOT/'build/metal/scheme_tool'))


class SchemeToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise RuntimeError('build native scheme_tool before running its acceptance tests')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def run_tool(self, data=None, command='verify', format='json', domain=None, extra=(), expected=0):
        self.serial += 1
        source, output = self.root/f'in{self.serial}', self.root/f'out{self.serial}'
        if isinstance(data, bytes):
            source.write_bytes(data)
        else:
            source.write_text(json.dumps(data))
        argv = [str(BINARY),command,'--input',str(source),'--output',str(output),'--format',format]
        if domain:
            argv += ['--domain',domain]
        argv += list(extra)
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stderr)
        if expected:
            self.assertFalse(output.exists(), result.stderr)
            return result
        return output

    def test_help(self):
        run = subprocess.run([str(BINARY),'--help'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0)
        self.assertIn('--verification-work',run.stdout)

    def test_known_identities_and_rectangular_roundtrip(self):
        for data in (oracle.schoolbook((1,1,1)),oracle.schoolbook((1,1,1),'F2'),
                     oracle.schoolbook((2,3,4)),oracle.schoolbook((2,3,4),'F2')):
            record = json.loads(self.run_tool(data).read_text())
            self.assertEqual(record['scheme_id'],oracle.identity(data))
            self.assertEqual(record['factors_id'],oracle.identity(data,False))
            raw = self.run_tool(data,'export',extra=('--output-format','cpu-text')).read_bytes()
            back = json.loads(self.run_tool(raw,'import','cpu-text',data['domain']).read_text())
            self.assertEqual(back['factors_id'],record['factors_id'])

    def test_zero_terms_repetitions_and_sign_gauges(self):
        cases=[]
        zero=oracle.schoolbook((1,1,1));zero['rank']=2
        for key,value in zip('uvw',(0,1,1)):
            zero[key].append([value])
        cases.append(zero)
        repeat=oracle.schoolbook((1,1,1));repeat['rank']=3
        repeat['u']=[[1],[1],[-1]];repeat['v']=[[1],[1],[1]];repeat['w']=[[1],[1],[1]]
        cases.append(repeat)
        for data in cases:
            record=json.loads(self.run_tool(data).read_text())
            self.assertEqual(record['scheme_id'],oracle.identity(data))
            self.assertEqual(record['rank'],data['rank'])
        gauge=oracle.schoolbook((1,1,1));gauge['u']=[[-1]];gauge['v']=[[-1]]
        record=json.loads(self.run_tool(gauge).read_text())
        self.assertEqual(record['scheme_id'],oracle.identity(oracle.schoolbook((1,1,1))))
        self.assertEqual(record['u'],[[-1]])
        self.assertNotEqual(record['effective_factors_id'],record['submitted_factors_id'])
        self.assertEqual(record['effective_factors_id'],oracle.identity(oracle.schoolbook((1,1,1)),False))

    def test_domains_and_resource_limits(self):
        data=oracle.schoolbook((1,1,1),'F2');data['rank']=3
        for key in 'uvw':data[key]=[[1],[1],[1]]
        self.run_tool(data)
        data['domain']='ZT'
        self.run_tool(data,expected=1)
        self.run_tool(oracle.schoolbook((2,3,4)),extra=('--verification-work','1'),expected=2)
        self.run_tool(b'1 1 1 1 1 1 1',format='cpu-text',expected=1)
        self.run_tool(oracle.schoolbook((1,1,1)),domain='F2',expected=1)

    def test_strict_json(self):
        base=json.dumps(oracle.schoolbook((1,1,1)))
        for raw in (base.replace('"rank": 1','"rank": true'),
                    base.replace('"rank": 1','"rank": 1.0'),
                    base.replace('"rank": 1','"rank": 1, "rank": 1'),
                    base+' garbage'):
            self.run_tool(raw.encode(),expected=1)
        self.run_tool(base.encode(),extra=('--record-bytes','10'),expected=2)

    def test_array_jsonl_and_text_adapters(self):
        data=oracle.schoolbook((1,2,3))
        for fmt,payload in [('json-array',json.dumps([data,data]).encode()),
                            ('jsonl',(json.dumps(data)+'\n'+json.dumps(data)+'\n').encode())]:
            output=self.run_tool(payload,format=fmt)
            self.assertEqual(len(output.read_text().splitlines()),2)
        for fmt in ('metal-search-text','metal-minimizer-text'):
            raw=self.run_tool(data,'export',extra=('--output-format',fmt)).read_bytes()
            back=json.loads(self.run_tool(raw,format=fmt,domain='ZT').read_text())
            self.assertEqual(back['factors_id'],oracle.identity(data,False))

    def test_circuit_costs_and_binding(self):
        circuit=dict(n=[1,1,1],m=1,z2=False,complexity={'naive':0,'reduced':0})
        for key in 'uvw':
            circuit[key]=[[{'index':0,'value':1}]];circuit[key+'_fresh']=[]
        result=json.loads(self.run_tool(circuit,format='circuit-json').read_text())
        self.assertEqual(result['verified_circuit_additions'],0)
        circuit['complexity']['reduced']=1
        self.run_tool(circuit,format='circuit-json',expected=1)

    def test_no_clobber_and_unknown_options(self):
        output=self.run_tool(oracle.schoolbook((1,1,1)))
        before=output.read_bytes()
        result=subprocess.run([str(BINARY),'verify','--input',str(output),'--output',str(output),'--format','json'],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(output.read_bytes(),before)
        self.run_tool(oracle.schoolbook((1,1,1)),extra=('--unexpected','1'),expected=1)

    def collection(self, ids):
        rows=[]
        for i,id in enumerate(ids):
            data=oracle.schoolbook((1,1,1))
            if i%2:data['u']=[[-1]];data['v']=[[-1]]
            source=self.root/f'source-{i}.json'
            source.write_text(json.dumps(data))
            rows.append(dict(schema='fgm-collection-v1',namespace='public-fixture',id=id,
                             path=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             format='json',domain='ZT',metadata={'group':'example','bound':0.5}))
        manifest=self.root/'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return manifest,rows

    def test_selected_presentations_retain_bindings_and_hash_order(self):
        manifest,rows=self.collection(['a','b','c','d'])
        output=self.root/'selection.jsonl'
        argv=[str(BINARY),'select','--input',str(manifest),'--output',str(output),'--count','3','--seed','7']
        run=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        records=[json.loads(line) for line in output.read_text().splitlines()]
        expected=sorted(['a','b','c','d'],key=lambda id:oracle.selection_key(7,'public-fixture',id))[:3]
        self.assertEqual([r['source_binding']['id'] for r in records],expected)
        self.assertEqual(len({r['scheme_id'] for r in records}),1)
        self.assertEqual(records[0]['source_binding']['metadata']['bound'],.5)
        for r in records:
            self.assertEqual(r['source_binding']['selection_key'],oracle.selection_key(7,'public-fixture',r['source_binding']['id'])[0].hex())

    def test_selection_hash_failure_and_budget(self):
        manifest,rows=self.collection(['a'])
        output=self.root/'selection'
        argv=[str(BINARY),'select','--input',str(manifest),'--output',str(output),'--count','1']
        (self.root/rows[0]['path']).write_text('{}')
        run=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(run.returncode,1)
        self.assertFalse(output.exists())
        run=subprocess.run(argv+['--selection-memory','1'],capture_output=True,text=True)
        self.assertEqual(run.returncode,2)
        self.assertFalse(output.exists())

    def test_legacy_json_and_row_major_orientation(self):
        import sys
        sys.path.insert(0,str(ROOT/'tests/metal'))
        from verify import verify
        data=oracle.schoolbook((1,2,3))
        legacy=json.loads(self.run_tool(data,'export',extra=('--output-format','legacy-json')).read_text())
        self.assertEqual(verify(legacy)['rank'],6)
        rectangular=oracle.schoolbook((2,3,4))
        changed=json.loads(json.dumps(rectangular))
        changed['orientation']='row-major-w'
        for index,row in enumerate(rectangular['w']):
            changed['w'][index]=[row[j*2+i] for i in range(2) for j in range(4)]
        record=json.loads(self.run_tool(changed).read_text())
        self.assertEqual(record['factors_id'],oracle.identity(rectangular,False))
        self.assertEqual(record['source_orientation'],'row-major-w')
        self.assertEqual(record['eligibility']['factor_ranks'],[6,12,8])
        contradiction=oracle.schoolbook((1,1,1));contradiction['z2']=True
        self.run_tool(contradiction,expected=1)

    def test_located_selection_skips_other_domains_and_duplicate_rows(self):
        good=oracle.schoolbook((1,1,1))
        unrelated={'domain':'Z3','unsupported':'not imported'}
        for format,payload,locator in (
            ('jsonl',json.dumps(unrelated)+'\n'+json.dumps(good)+'\n',{'line':2}),
            ('json-array',json.dumps([unrelated,good]),{'index':1})):
            source=self.root/(format+'.source');source.write_text(payload)
            row=dict(schema='fgm-collection-v1',namespace='public',id='a',path=source.name,
                     sha256=hashlib.sha256(source.read_bytes()).hexdigest(),format=format,
                     domain='ZT',locator=locator)
            manifest=self.root/(format+'.manifest');manifest.write_text((json.dumps(row)+'\n')*2)
            output=self.root/(format+'.out')
            run=subprocess.run([str(BINARY),'select','--input',str(manifest),'--output',str(output),'--count','1'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(output.read_text())['scheme_id'],oracle.identity(good))
            row['metadata']={'conflict':True}
            manifest.write_text(manifest.read_text()+json.dumps(row)+'\n')
            rejected=self.root/(format+'.badout')
            run=subprocess.run([str(BINARY),'select','--input',str(manifest),'--output',str(rejected),'--count','1'],capture_output=True,text=True)
            self.assertEqual(run.returncode,1)
            self.assertFalse(rejected.exists())

    def test_normalized_candidate_capacity_is_separate(self):
        data=oracle.schoolbook((1,1,1))
        for _ in range(10):
            for term in [(1,1,1),(-1,1,1),(1,-1,-1),(-1,-1,-1)]:
                for key,value in zip('uvw',term):data[key].append([value])
        data['rank']=41
        result=json.loads(self.run_tool(data).read_text())
        eligibility=result['eligibility']
        self.assertEqual(eligibility['source_candidate_pairs'],[400,400,400])
        self.assertEqual(eligibility['candidate_pairs'][:2],[820,820])
        self.assertFalse(eligibility['search_eligible'])
        self.assertFalse(eligibility['signed_reducer_eligible'])
        self.assertEqual(result['rank'],41)

    def test_circuit_term_work_is_bounded(self):
        data=dict(n=[1,1,1],m=1,z2=False,complexity={'naive':0,'reduced':200})
        for key in 'uvw':
            data[key]=[[{'index':0,'value':1}]];data[key+'_fresh']=[]
        data['u']=[[{'index':0,'value':1}]+[term for _ in range(100) for term in ({'index':0,'value':1},{'index':0,'value':-1})]]
        self.run_tool(data,format='circuit-json',extra=('--verification-work','100'),expected=2)

    def test_modern_metadata_and_claim_validation(self):
        data=oracle.schoolbook((1,1,1));data['metadata']={'groups':{'example:group':['A']}}
        data['source_binding']={'namespace':'example','id':'original'}
        first=json.loads(self.run_tool(data,'import').read_text())
        second=json.loads(self.run_tool(first,'import').read_text())
        self.assertEqual(second['source_binding'],data['source_binding'])
        self.assertEqual(second['metadata'],data['metadata'])
        self.assertEqual(len(second['artifact_bindings']),2)
        second['scheme_id']='fgm-scheme-v1:'+'0'*64
        self.run_tool(second,expected=1)
        data['n']=[2,2,2]
        self.run_tool(data,expected=1)

    def test_filters_before_top_k_and_scan_limit(self):
        manifest,rows=self.collection(['a','b','c'])
        for row in rows:
            row.update(rank=1,dimensions=[1,1,1])
            row['metadata']['groups']={'example:cohort':['keep' if row['id']=='c' else 'other']}
        manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        output=self.root/'filtered'
        argv=[str(BINARY),'select','--input',str(manifest),'--output',str(output),'--count','1','--seed','9',
              '--filter-domain','ZT','--filter-dimensions','1,1,1','--filter-rank','1','--filter-group','example:cohort=keep']
        run=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(output.read_text())['source_binding']['id'],'c')
        self.run_tool(oracle.schoolbook((1,1,1)),extra=('--scan-bytes','1'),expected=2)

    def test_ordered_id_selection(self):
        manifest,_=self.collection(['a','b','c'])
        ids=self.root/'ids.json';ids.write_text('["c","a"]')
        output=self.root/'selected'
        run=subprocess.run([str(BINARY),'select','--input',str(manifest),'--output',str(output),'--count','2','--ids',str(ids)],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        records=[json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r['source_binding']['id'] for r in records],['c','a'])


if __name__=='__main__':
    unittest.main()
