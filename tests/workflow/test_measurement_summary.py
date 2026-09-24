import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[2]/'benchmarks/workflow/summarize_baseline.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i in range(4):
            row={'complete':True,'guard_complete':True,'workload':'test','repeat':i,'exports':[]}
            for key in ('gpu_work_seconds','gpu_all_seconds','process_seconds','workflow_seconds','verification_seconds','peak_process_rss_bytes','peak_system_wired_bytes'):
                row[key]=1+i*.01
            rows.append(row)
        return {'complete':True,'attempts':rows,'protocol':{'mandatory_rows':['test']},'protocol_sha256':'test'}

    def call(self,data):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.json';path.write_text(json.dumps(data))
            return summary.summarize(path)

    def test_diagnostic_not_allowance(self):
        result=self.call(self.fixture())
        self.assertEqual(result['performance_verdict'],'NOT EVALUATED')
        self.assertIn('never inferred',result['practical_regression_allowance'])
        self.assertAlmostEqual(result['workloads']['test']['metrics']['process_seconds']['median'],1.015)

    def test_incomplete_or_missing_rejected(self):
        for mutation in ('incomplete','missing','nonfinite'):
            data=self.fixture()
            if mutation=='incomplete':data['attempts'][0]['complete']=False
            elif mutation=='missing':data['protocol']['mandatory_rows'].append('absent')
            else:data['attempts'][0]['process_seconds']=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.call(data)


if __name__=='__main__':unittest.main()
