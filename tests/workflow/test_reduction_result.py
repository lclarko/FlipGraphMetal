"""Native retained best-circuit verification, without a GPU or production search."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


def source(rank=1):
    return dict(n=[1, 1, 1], m=rank, z2=False,
                u=[[1] for _ in range(rank)], v=[[1] for _ in range(rank)],
                w=[[1]]+[[0] for _ in range(rank-1)])


def circuit(sign=1):
    expression = [[dict(index=0, value=sign)]]
    return dict(n=[1, 1, 1], m=1, z2=False, complexity=dict(naive=0, reduced=0),
                u=deepcopy(expression), v=deepcopy(expression), w=[[dict(index=0, value=1)]],
                u_fresh=[], v_fresh=[], w_fresh=[])


class ReductionResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = Path(os.environ.get('FGM_REDUCTION_RESULT_DRIVER', ROOT/'build/workflow/test_reduction_result'))
        if not cls.driver.is_file():
            raise RuntimeError('build native reduction-result driver first')

    def check(self, effective, result, fixed, expected=0, **extra):
        request = dict(effective=effective, circuit=result, fixed=fixed, **extra)
        process = subprocess.run([str(self.driver)], input=json.dumps(request), text=True,
                                 capture_output=True, timeout=10)
        self.assertEqual(process.returncode, expected, process.stderr)
        return json.loads(process.stdout) if not expected else process.stderr

    def test_fixed_exact_factors_and_verified_cost(self):
        result = self.check(source(), circuit(), True)
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['verified_circuit_additions'], 0)

    def test_mathematically_valid_gauge_change_rejected_only_in_fixed_mode(self):
        self.check(source(), circuit(-1), True, expected=1)
        result = self.check(source(), circuit(-1), False)
        self.assertEqual(result['u'], [[-1]])

    def test_rank_changing_best_bound_to_its_own_factors(self):
        result = self.check(source(2), circuit(), False)
        self.assertEqual(result['rank'], 1)
        self.check(source(2), circuit(), True, expected=1)
        incorrect = circuit(); incorrect['m'] = 2
        self.check(source(2), incorrect, False, expected=1)

    def test_invalid_tensor_and_false_operation_claim_fail(self):
        incorrect = circuit(); incorrect['w'][0][0]['value'] = -1
        self.check(source(), incorrect, False, expected=1)
        incorrect = circuit(); incorrect['complexity']['reduced'] = 1
        self.check(source(), incorrect, False, expected=1)

    def test_domain_and_verification_budget_fail_closed(self):
        incorrect = circuit(); incorrect['z2'] = True
        self.check(source(), incorrect, False, expected=1)
        self.check(source(), circuit(), False, expected=2, work=1)

    def test_bounded_native_serialization(self):
        for value,limit,status in [('12345',5,0),('12345',4,2),('',0,0),('x',0,2)]:
            process=subprocess.run([str(self.driver)],input=json.dumps(dict(serialize=value,limit=limit)),
                                   text=True,capture_output=True,timeout=10)
            self.assertEqual(process.returncode,status,process.stderr)
            if not status:self.assertEqual(json.loads(process.stdout)['bytes'],value)
