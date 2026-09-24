"""Host-only tests for prospective paired performance gates."""
import copy
import importlib.util
import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('performance', Path(__file__).resolve().parents[2] / 'benchmarks/workflow/performance.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def case(candidate=1.01):
    budget = dict(version=1, workload='signed-packed', endpoint='elapsed', units='seconds',
                  direction='lower', estimand='mean paired log ratio', transform='log-ratio',
                  margin=.02, status='approved', approval={'reference': 'test-only', 'version': '1'},
                  rationale='Synthetic test approval only', baseline_source='baseline-sha',
                  candidate_source='candidate-sha', baseline_build='baseline-build-sha',
                  candidate_build='candidate-build-sha', absolute_limit=None)
    protocol = dict(version=1, alpha=.05, looks=[6,12], assumptions=True,
                    protocol_id='synthetic-v1', matched_identity=p.MATCH_KEYS.split())
    identity = {key: 'identical-'+key for key in p.MATCH_KEYS.split()}
    protocol['mandatory_endpoints'] = [{key:budget[key] for key in p.ENDPOINT_KEYS.split()}]
    protocol['prospective_pairs'] = [dict(pair_id=str(i), order='AB' if i%2 == 0 else 'BA', identity=identity.copy()) for i in range(12)]
    protocol['precision_criterion'] = 'Synthetic adequate-resolution criterion for unit tests'
    rows = [dict(pair_id=str(i), order='AB' if i%2 == 0 else 'BA', complete=True,
                 baseline=1., candidate=candidate, baseline_source='baseline-sha',
                 candidate_source='candidate-sha', baseline_build='baseline-build-sha',
                 candidate_build='candidate-build-sha', baseline_identity=identity.copy(),
                 candidate_identity=identity.copy(), resource_violation=False, error=None) for i in range(6)]
    return dict(version=1, budget=budget, budget_sha256=p.digest(budget), protocol=protocol,
                protocol_sha256=p.digest(protocol), observations=rows, look=1, previous=None,
                precision_assessment={'adequate':True,'evidence':'synthetic unit-test evidence'})


def seal(doc):
    doc['protocol']['mandatory_endpoints'][0] = {key:doc['budget'][key] for key in p.ENDPOINT_KEYS.split()}
    doc['budget_sha256'] = p.digest(doc['budget'])
    doc['protocol_sha256'] = p.digest(doc['protocol'])
    return doc


def absolute_case(value=1.0):
    budget = dict(version=1, workload='report', endpoint='workflow_seconds', units='seconds',
                  estimand='arithmetic mean complete workflow elapsed', limit=5.0,
                  status='approved', approval={'reference':'synthetic-only','version':'1'},
                  rationale='synthetic test', candidate_source='source', candidate_build='build',
                  peak_process_rss_limit=64*1024**2)
    identity = {key:'fixed-'+key for key in p.MATCH_KEYS.split()}
    endpoint = {key:budget[key] for key in p.ABSOLUTE_ENDPOINT_KEYS.split()}
    family = [dict(endpoint,workload=name) for name in ('report','roundtrip','select1k','select10k','select100k')]
    protocol = dict(version=2,alpha=.05,looks=[6,12],assumptions=True,protocol_id='synthetic-absolute',
                    mandatory_endpoints=family, precision_criterion='synthetic precision',
                    prospective_trials=[dict(trial_id=str(i),identity=identity.copy()) for i in range(12)])
    rows = [dict(trial_id=str(i),complete=True,value=value,peak_process_rss_bytes=1024,
                 candidate_source='source',candidate_build='build',identity=identity.copy(),
                 evidence_sha256='synthetic-evidence',resource_violation=False,error=None) for i in range(6)]
    return dict(version=2,mode='absolute',budget=budget,budget_sha256=p.digest(budget),protocol=protocol,
                protocol_sha256=p.digest(protocol),observations=rows,look=1,previous=None,
                precision_assessment={'adequate':True,'evidence':'synthetic only'},
                acquisition_complete=True,acquisition_error=None)


class AbsolutePerformanceTests(unittest.TestCase):
    def test_absolute_pass_fail_and_family_correction(self):
        result = p.evaluate(absolute_case())
        self.assertEqual(result['verdict'],'PASS')
        self.assertEqual(result['endpoint_alpha'],.005)
        self.assertNotIn('change',result)
        self.assertNotIn('approved_degradation',result)
        self.assertEqual(p.evaluate(absolute_case(6))['verdict'],'FAIL')

    def test_uncertainty_approval_and_incomplete_not_passes(self):
        doc = absolute_case()
        for index,row in enumerate(doc['observations']):
            row['value']=1 if index%2 else 9
        self.assertEqual(p.evaluate(doc)['verdict'],'INCONCLUSIVE')
        doc = absolute_case()
        doc['budget'].update(status='unapproved',approval=None)
        doc['budget_sha256']=p.digest(doc['budget'])
        self.assertEqual(p.evaluate(doc)['verdict'],'INCONCLUSIVE')
        doc=absolute_case()
        doc['observations'][0].update(complete=False,error='interrupted',value=None)
        self.assertEqual(p.evaluate(doc)['verdict'],'INCONCLUSIVE')
        doc=absolute_case();doc['precision_assessment']['adequate']=False
        self.assertEqual(p.evaluate(doc)['verdict'],'INCONCLUSIVE')

    def test_hard_resources_fail_before_full_sample(self):
        for field,value in (('peak_process_rss_bytes',65*1024**2),('resource_violation',True)):
            doc=absolute_case();doc['observations']=doc['observations'][:1]
            doc['observations'][0][field]=value
            self.assertEqual(p.evaluate(doc)['verdict'],'FAIL')

    def test_second_look_preserves_first_observations_and_assessment(self):
        doc=absolute_case();doc['precision_assessment']['adequate']=False
        first=p.evaluate(doc)
        doc['look']=2
        doc['previous']={key:first[key] for key in ('verdict','budget_sha256','protocol_sha256','observations_sha256','precision_assessment','acquisition_complete','acquisition_error')}
        doc['precision_assessment']={'adequate':True,'evidence':'second-look assessment'}
        doc['observations'] += [dict(row,trial_id=str(index+6)) for index,row in enumerate(doc['observations'])]
        self.assertEqual(p.evaluate(doc)['verdict'],'PASS')
        doc['observations'][0]['value']=2
        with self.assertRaises(ValueError):p.evaluate(doc)

    def test_failed_acquisition_cannot_pass_complete_observations(self):
        doc=absolute_case()
        doc.update(acquisition_complete=False,acquisition_error='source changed after final trial')
        self.assertEqual(p.evaluate(doc)['verdict'],'INCONCLUSIVE')
        doc['budget'].update(status='unapproved',approval=None)
        doc['budget_sha256']=p.digest(doc['budget'])
        doc['observations'][0]['resource_violation']=True
        self.assertEqual(p.evaluate(doc)['verdict'],'FAIL')

    def test_wrong_identity_order_and_retroactive_changes_rejected(self):
        for field,value in (('candidate_build','changed'),('trial_id','8'),('identity',{})):
            doc=absolute_case();doc['observations'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):p.evaluate(doc)
        doc=absolute_case();doc['budget']['limit']=20
        with self.assertRaises(ValueError):p.evaluate(doc)


class PerformanceTests(unittest.TestCase):
    def test_quantiles(self):
        for df, tail, expected in [(1,.025,12.7062047364), (5,.025,2.5705818356),
                                    (11,.025,2.2009851601), (5,.005,4.0321429836)]:
            self.assertAlmostEqual(p.t_critical(tail, df), expected, places=8)

    def test_interval(self):
        low, high = p.interval([1,2,3,4,5,6], .05)
        radius = 2.5705818356*math.sqrt(3.5)/math.sqrt(6)
        self.assertAlmostEqual(low, 3.5-radius, places=8)
        self.assertAlmostEqual(high, 3.5+radius, places=8)

    def test_slowdown_within_tolerance(self):
        result = p.evaluate(case())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['reason'], 'PASS within the approved tolerance; a slowdown was established.')
        self.assertEqual(result['change'], 'slowdown established')
        receipt = result.pop('receipt_sha256')
        self.assertEqual(receipt, p.digest(result))

    def test_regression_and_improvement(self):
        self.assertEqual(p.evaluate(case(1.03))['verdict'], 'FAIL')
        self.assertEqual(p.evaluate(case(.99))['change'], 'improvement established')

    def test_uncertainty(self):
        doc = case()
        for i, row in enumerate(doc['observations']):
            row['candidate'] = .8 if i%2 else 1.2
        self.assertEqual(p.evaluate(doc)['verdict'], 'INCONCLUSIVE')

    def test_unapproved(self):
        doc = case()
        doc['budget'].update(status='unapproved', approval=None)
        result = p.evaluate(seal(doc))
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertFalse(result['budget_readiness'])

    def test_directions(self):
        doc = case(.98)
        doc['budget'].update(direction='higher', margin=.01)
        self.assertEqual(p.evaluate(seal(doc))['verdict'], 'FAIL')
        doc['budget']['direction'] = 'sideways'
        with self.assertRaises(ValueError):
            p.evaluate(seal(doc))

    def test_difference_absolute_and_failed_rows(self):
        doc = case(11)
        doc['budget'].update(transform='difference', margin=11, absolute_limit=10)
        self.assertEqual(p.evaluate(seal(doc))['verdict'], 'FAIL')
        doc['budget']['absolute_limit'] = None
        doc['observations'][0].update(complete=False, error='timeout', candidate=None)
        result = p.evaluate(seal(doc))
        self.assertEqual(result['verdict'], 'INCONCLUSIVE')
        self.assertEqual(result['pairs_retained'], 6)
        doc['observations'][0]['resource_violation'] = True
        self.assertEqual(p.evaluate(doc)['verdict'], 'FAIL')

    def test_mismatches_rejected(self):
        for mutation in ('source', 'work', 'hash', 'exclude', 'duplicate', 'nonfinite'):
            doc = case()
            if mutation == 'source':
                doc['observations'][0]['candidate_source'] = 'other'
            elif mutation == 'work':
                doc['observations'][0]['candidate_identity']['work'] = 'other'
            elif mutation == 'hash':
                doc['budget']['margin'] = .5
            elif mutation == 'exclude':
                doc['observations'][0]['excluded'] = True
            elif mutation == 'duplicate':
                doc['observations'][1]['pair_id'] = '0'
            else:
                doc['observations'][0]['candidate'] = math.nan
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                p.evaluate(doc)

    def test_assumptions_order_and_count(self):
        doc = case()
        doc['protocol']['assumptions'] = False
        self.assertEqual(p.evaluate(seal(doc))['verdict'], 'INCONCLUSIVE')
        doc = case()
        doc['observations'][0]['order'] = 'BA'
        with self.assertRaises(ValueError):
            p.evaluate(doc)
        doc = case()
        doc['observations'].pop()
        self.assertEqual(p.evaluate(doc)['verdict'], 'INCONCLUSIVE')

    def test_second_look_retains_first(self):
        doc = case()
        for i, row in enumerate(doc['observations']):
            row['candidate'] = .99 if i%2 else 1.03
        first = p.evaluate(doc)
        self.assertEqual(first['verdict'], 'INCONCLUSIVE')
        doc['look'] = 2
        doc['previous'] = {key: first[key] for key in ('verdict','budget_sha256','protocol_sha256','observations_sha256','precision_assessment')}
        extra = copy.deepcopy(doc['observations'])
        for i, row in enumerate(extra):
            row['pair_id'] = str(i+6)
        doc['observations'].extend(extra)
        self.assertEqual(p.evaluate(doc)['pairs_retained'], 12)
        doc['observations'][0]['candidate'] = 1
        with self.assertRaises(ValueError):
            p.evaluate(doc)

    def test_cli_receipt_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.json', Path(directory)/'receipt.json'
            source.write_text(json.dumps(case()))
            argv = [sys.executable, str(Path(p.__file__)), str(source), '--output', str(output)]
            run = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(output.read_text())['verdict'], 'PASS')
            self.assertNotEqual(subprocess.run(argv, capture_output=True).returncode, 0)
            source.write_text('{"version":1,"version":1}')
            self.assertNotEqual(subprocess.run(argv, capture_output=True).returncode, 0)

    def test_prospective_work_and_roster_tampering(self):
        doc=case()
        for key in ('baseline_identity','candidate_identity'):
            doc['observations'][0][key]['fixture']='different-but-matched'
        with self.assertRaises(ValueError):
            p.evaluate(doc)
        doc=case()
        doc['protocol']['mandatory_endpoints'][0]['endpoint']='unrelated'
        doc['protocol_sha256']=p.digest(doc['protocol'])
        with self.assertRaises(ValueError):
            p.evaluate(doc)
        doc=case()
        doc['protocol']['prospective_pairs'][7]['pair_id']='0'
        doc['protocol_sha256']=p.digest(doc['protocol'])
        with self.assertRaises(ValueError):
            p.evaluate(doc)

    def test_family_allocation_derived_from_actual_roster(self):
        doc=case()
        extra=copy.deepcopy(doc['protocol']['mandatory_endpoints'][0])
        extra['endpoint']='host-phase'
        doc['protocol']['mandatory_endpoints'].append(extra)
        result=p.evaluate(seal(doc))
        self.assertEqual(result['family_size'],2)
        self.assertEqual(result['endpoint_alpha'],.05/4)
        doc['protocol']['mandatory_endpoints'].append(extra)
        doc['protocol_sha256']=p.digest(doc['protocol'])
        with self.assertRaises(ValueError):
            p.evaluate(doc)

    def test_precision_distinct_from_approval_and_assumptions(self):
        doc=case()
        doc['precision_assessment']={'adequate':False,'evidence':'synthetic coarse timer cannot resolve margin'}
        result=p.evaluate(doc)
        self.assertEqual(result['verdict'],'INCONCLUSIVE')
        self.assertTrue(result['budget_readiness'])
        doc['precision_assessment']['evidence']=''
        with self.assertRaises(ValueError):
            p.evaluate(doc)

    def test_resource_failure_survives_unapproved_margin(self):
        doc=case()
        doc['budget'].update(status='unapproved',approval=None)
        doc['observations'][0].update(complete=False,candidate=None,error='memory cutoff',resource_violation=True)
        doc['precision_assessment']['adequate']=False
        result=p.evaluate(seal(doc))
        self.assertEqual(result['verdict'],'FAIL')
        self.assertFalse(result['budget_readiness'])
        self.assertTrue(result['resource_violation'])
        self.assertEqual(result['pairs_retained'],6)

    def test_no_extra_look(self):
        doc = case()
        doc['look'] = 3
        with self.assertRaises(ValueError):
            p.evaluate(doc)


if __name__ == '__main__':
    unittest.main()
