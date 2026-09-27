"""Independent finite-family oracle and exact transposition checks.

The oracle derives signed operations from vectors and explores available sets;
it never reads the native operation table. GPU qualification reuses these cases
through the existing correctness executable's --constructor-input option.
"""
from collections import deque
from itertools import combinations, product
import json
import os
from pathlib import Path
import random
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


def direction(vector):
    sign = next((1 if x > 0 else -1 for x in vector if x), 1)
    return tuple(sign*x for x in vector)


def family(targets, inputs):
    basis = [tuple(int(i == j) for j in range(inputs)) for i in range(inputs)]
    forms = basis.copy()
    for row in targets:
        row = direction(row)
        if any(row) and row not in forms:
            forms.append(row)
    slots = []
    for left, right in combinations(range(len(forms)), 2):
        for sign in (1, -1):
            aux = direction(tuple(a+sign*b for a, b in zip(forms[left], forms[right])))
            slots.append((left, right, aux))
    return forms, slots


def oracle(targets, inputs, candidate=-1):
    forms, slots = family(targets, inputs)
    required = frozenset(forms)
    creation = None
    if candidate >= 0:
        left, right, aux = slots[candidate]
        if not any(aux) or aux in forms:
            return 2
        creation = (forms[left], forms[right], aux)
        forms = forms+[aux]
    allowed = set(forms)
    initial = frozenset(forms[:inputs])
    pending, seen = deque([initial]), {initial}
    while pending:
        available = pending.popleft()
        if required <= available:
            return 1
        additions = set()
        for a, b in combinations(available, 2):
            for sign in (-1, 1):
                result = direction(tuple(x+sign*y for x, y in zip(a, b)))
                if result in allowed and result not in available:
                    if creation and result == creation[2]:
                        continue
                    additions.add(result)
        if creation and creation[0] in available and creation[1] in available:
            additions.add(creation[2])
        for new in additions-available:
            state = available | {new}
            if state not in seen:
                seen.add(state)
                pending.append(state)
    return 0


def capacity_case():
    targets = []
    for i in range(2, 9):
        for sign in (1, -1):
            row = [1, 1]+[0]*7
            row[i] = sign
            targets.append(row)
    for a, b in list(combinations(range(2, 9), 2))[:9]:
        row = [0]*9
        row[a] = row[b] = 1
        targets.append(row)
    return dict(inputs=9, targets=targets)


def small_cases():
    cases = [dict(inputs=3, targets=[[1, 1, 1], [1, 1, -1]]),
             dict(inputs=2, targets=[[0, 0], [1, 1], [-1, -1], [0, -1]]),
             dict(inputs=1, targets=[[2]]),
             dict(inputs=6, targets=[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]),
             dict(inputs=2, targets=[[1, 1], [1, -1], [2, 1]])]
    rng = random.Random(314159)
    for _ in range(12):
        cases.append(dict(inputs=3, targets=[[rng.randrange(-1, 2) for _ in range(3)] for _ in range(4)]))
    return cases


def check_witness(case, result, candidate):
    forms, slots = family(case['targets'], case['inputs'])
    if candidate >= 0:
        forms.append(slots[candidate][2])
    available = set(range(case['inputs']))
    for out, left, right, ls, rs in result['gates']:
        assert left != right and left in available and right in available
        assert out not in available and ls in (-1, 1) and rs in (-1, 1)
        assert tuple(ls*x+rs*y for x, y in zip(forms[left], forms[right])) == forms[out]
        if out == len(forms)-1 and candidate >= 0:
            assert (left, right) == slots[candidate][:2]
        available.add(out)
    assert result['count'] == len(result['gates']) <= 24
    assert result['available'] == sum(1 << i for i in available)
    if result['status'] == 1:
        assert set(range(len(forms)-(candidate >= 0))) <= available
        wires=[tuple(int(i==j) for j in range(case['inputs'])) for i in range(case['inputs'])]
        def expression(terms):
            return [sum(t['value']*wires[t['index']][i] for t in terms) for i in range(case['inputs'])]
        for gate in result['fresh']:wires.append(expression(gate))
        assert [expression(output) for output in result['outputs']] == case['targets']


def check_oracle(case, response):
    forms, slots = family(case['targets'], case['inputs'])
    assert response['directions'] == len(forms)
    assert response['candidate_slots'] == len(slots)
    for candidate, result in enumerate(response['results'], -1):
        assert result['status'] == oracle(case['targets'], case['inputs'], candidate), (case, candidate, result)
        check_witness(case, result, candidate)


class ConstructorTests(unittest.TestCase):
    def driver(self, request, expected=0):
        driver = Path(os.environ.get('FGM_REDUCTION_RESULT_DRIVER', ROOT/'build/workflow/test_reduction_result'))
        process = subprocess.run([str(driver)], input=json.dumps(request), text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, expected, process.stderr)
        return json.loads(process.stdout) if expected == 0 else process.stderr

    def test_independent_availability_oracle(self):
        for case in small_cases():
            with self.subTest(case=case):
                check_oracle(case, self.driver(case))

    def test_unreachable_cycles_and_excluded_families(self):
        cases = small_cases()
        self.assertEqual(oracle(**cases[0]), 0)
        self.assertTrue(any(r['status'] == 1 for r in self.driver(cases[0])['results']))
        for case in (cases[2], cases[3]):
            self.assertFalse(any(r['status'] == 1 for r in self.driver(case)['results']))

    def test_capacity_bit32_and_24_gates(self):
        case = capacity_case()
        response = self.driver(case)
        self.assertEqual(response['directions'], 32)
        self.assertEqual(response['candidate_slots'], 992)
        witness = response['results'][1]  # auxiliary e0+e1, slot 0
        self.assertEqual(witness['status'], 1)
        self.assertEqual(witness['count'], 24)
        self.assertTrue(witness['available'] & (1 << 32))
        for candidate, result in enumerate(response['results'], -1):
            check_witness(case, result, candidate)
        self.driver(dict(inputs=9, targets=case['targets']+[[0]*9]), expected=1)
        self.driver(dict(inputs=10, targets=[[0]*10]), expected=1)
        self.driver(dict(inputs=2, targets=[[1]]), expected=1)
        self.driver(dict(inputs=1, targets=[[1 << 30]]), expected=1)

    def test_invalid_witnesses_fail_instead_of_retaining_incumbent(self):
        case=small_cases()[0]
        for kind in ('dependency','sign','capacity','availability','auxiliary'):
            self.driver(dict(case,tamper=kind),expected=1)

    def test_nonternary_auxiliary_is_replayed(self):
        case=small_cases()[4]
        forms,slots=family(case['targets'],case['inputs'])
        response=self.driver(case)
        candidates=[i for i,(_,_,aux) in enumerate(slots)
                    if max(map(abs,aux))>1 and response['results'][i+1]['status']==1
                    and any(g[0]==len(forms) for g in response['results'][i+1]['gates'])]
        self.assertTrue(candidates)
        for candidate in candidates:check_witness(case,response['results'][candidate+1],candidate)

    def test_public_covered_components(self):
        fixtures = ROOT/'benchmarks/workflow/fixtures/fgm1/factors'
        for name, key, gates in [('cn122', 'u', 13), ('cn122', 'v', 14), ('sun', 'u', 13)]:
            data = json.loads((fixtures/f'{name}.json').read_text())
            response = self.driver(dict(inputs=9, targets=data[key]))
            successes = [r['count'] for r in response['results'] if r['status'] == 1]
            self.assertTrue(successes, (name, key))
            self.assertLessEqual(min(successes), gates)

    def test_transposition_signed_aliases_zero_and_inactive(self):
        term = lambda i, v=1: dict(index=i, value=v)
        response = self.driver(dict(transpose=True, inputs=3,
            fresh=[[term(0), term(1, -1)], [term(3), term(0)]],
            outputs=[[term(4)], [term(4, -1)], [], [term(1)], [term(3), term(1)]]))
        self.assertEqual(response['outputs'][2], [])  # inactive original input
        self.assertEqual(response['count'], len(response['fresh']))


if __name__ == '__main__':
    unittest.main()
