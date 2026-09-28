"""Vector-derived oracle for the bounded two-auxiliary constructor.

The expected family and reachability graph are reconstructed here from integer
vectors. This module does not import native relation or operation tables.
"""

from collections import deque
from copy import deepcopy
from functools import lru_cache
from itertools import combinations, product
import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


def canonical(vector):
    vector = tuple(vector)
    first = next((x for x in vector if x), 0)
    return tuple((-1 if first < 0 else 1) * x for x in vector)


def add(left, right, sign):
    return canonical(a + sign * b for a, b in zip(left, right))


def directions(inputs, targets):
    basis = [tuple(int(i == j) for j in range(inputs)) for i in range(inputs)]
    for target in targets:
        vector = canonical(target)
        if any(vector) and vector not in basis:
            basis.append(vector)
    return tuple(basis)


@lru_cache(maxsize=None)
def routes(vectors):
    return tuple((a, b, sign, add(vectors[a], vectors[b], sign))
                 for a, b in combinations(range(len(vectors)), 2)
                 for sign in (1, -1))


@lru_cache(maxsize=None)
def slot(vectors, raw):
    n = len(vectors)
    first_routes = routes(vectors)
    first_id, second_id = divmod(raw, n * (n + 1))
    a, b, sign, helper1 = first_routes[first_id]
    second_routes = routes(vectors + (helper1,))
    c, d, second_sign, helper2 = second_routes[second_id]
    return (a, b, sign, helper1), (c, d, second_sign, helper2)


@lru_cache(maxsize=None)
def classify(vectors, raw):
    n = len(vectors)
    first, second = slot(vectors, raw)
    if not any(first[3]) or first[3] in vectors:
        return 'first_invalid'
    if not any(second[3]) or second[3] in vectors or second[3] == first[3]:
        return 'second_invalid'
    if second[1] < n:
        # Both helpers have independent original-D routes. Keep one order.
        second_id = routes(vectors).index(second)
        first_id = routes(vectors).index(first)
        if second_id <= first_id:
            return 'symmetry_filtered'
    return 'prepared'


def reachable(vectors, inputs, first, second=None):
    """Explore available-vector sets, using only arithmetic on their vectors."""
    n = len(vectors)
    values = vectors + (first[3],) + ((second[3],) if second else ())
    required = (1 << n) - 1
    initial = (1 << inputs) - 1
    edges = []
    for left, right in combinations(range(len(values)), 2):
        for sign in (1, -1):
            result = add(values[left], values[right], sign)
            for out in range(inputs, n):
                if result == values[out]:
                    edges.append((1 << out, (1 << left) | (1 << right)))
    edges.append((1 << n, (1 << first[0]) | (1 << first[1])))
    if second:
        edges.append((1 << (n + 1), (1 << second[0]) | (1 << second[1])))
    queue, seen = deque((initial,)), {initial}
    while queue:
        available = queue.popleft()
        if available & required == required:
            return True
        for output, prereqs in edges:
            if available & output or available & prereqs != prereqs:
                continue
            updated = available | output
            if updated not in seen:
                seen.add(updated)
                queue.append(updated)
    return False


def expected_stats(vectors, prefix):
    result = {'raw_scanned': prefix, 'first_invalid': 0,
              'second_invalid': 0, 'symmetry_filtered': 0, 'prepared': 0}
    for raw in range(prefix):
        result[classify(vectors, raw)] += 1
    return result


def raw_for(vectors, first_pair, second_pair):
    total = len(routes(vectors)) * len(vectors) * (len(vectors) + 1)
    return next(raw for raw in range(total)
                if slot(vectors, raw)[0][:3] == first_pair
                and slot(vectors, raw)[1][:3] == second_pair)


def creation_gate(values, out, route):
    a, b, sign, _ = route
    raw = tuple(x + sign * y for x, y in zip(values[a], values[b]))
    normalize = -1 if next(x for x in raw if x) < 0 else 1
    return [out, a, b, normalize, normalize * sign]


def check_expression(case, result):
    wires = [tuple(int(i == j) for j in range(case['inputs']))
             for i in range(case['inputs'])]

    def expression(terms):
        return tuple(sum(term['value'] * wires[term['index']][i]
                         for term in terms) for i in range(case['inputs']))

    for fresh in result['fresh']:
        wires.append(expression(fresh))
    assert [expression(output) for output in result['outputs']] == [
        tuple(row) for row in case['targets']]


def check_witness(case, vectors, raw, result):
    n = len(vectors)
    first, second = slot(vectors, raw)
    values = vectors + (first[3], second[3])
    assert result['helpers'] == [list(first[3]), list(second[3])]
    assert result['creations'] == [creation_gate(values, n, first),
                                   creation_gate(values, n + 1, second)]
    available = set(range(case['inputs']))
    for out, left, right, left_sign, right_sign in result['gates']:
        assert out not in available and 0 <= out < n + 2
        assert left != right and left in available and right in available
        assert left_sign in (-1, 1) and right_sign in (-1, 1)
        calculated = tuple(left_sign * x + right_sign * y
                           for x, y in zip(values[left], values[right]))
        assert calculated == values[out], (case, raw, result)
        if out in (n, n + 1):
            route = first if out == n else second
            assert (left, right) == route[:2]
        available.add(out)
    assert result['count'] == len(result['gates']) <= 25
    assert result['available'] == sum(1 << index for index in available)
    if result['status'] == 1:
        assert set(range(n)) <= available
        check_expression(case, result)


def finite_panels():
    for inputs, alphabet, max_targets in ((2, range(-2, 3), 3),
                                          (3, range(-1, 2), 2)):
        all_vectors = {canonical(v) for v in product(alphabet, repeat=inputs)}
        basis = {tuple(int(i == j) for j in range(inputs)) for i in range(inputs)}
        choices = sorted(all_vectors - basis - {tuple(0 for _ in range(inputs))})
        for count in range(max_targets + 1):
            for targets in combinations(choices, count):
                yield dict(inputs=inputs, targets=[list(v) for v in targets])


def full_capacity_case():
    targets = []
    for i in range(2, 9):
        for sign in (1, -1):
            row = [1, 1] + [0] * 7
            row[i] = sign
            targets.append(row)
    for i in range(4, 9):
        for sign in (1, -1):
            row = [0] * 9
            row[2] = row[3] = 1
            row[i] = sign
            targets.append(row)
            if len(targets) == 23:
                return dict(inputs=9, targets=targets)
    raise AssertionError('capacity fixture has too few directions')


class TwoAuxConstructorTests(unittest.TestCase):
    def driver(self, request, expected=0, timeout=30):
        driver = Path(os.environ.get(
            'FGM_REDUCTION_RESULT_DRIVER', ROOT / 'build/workflow/test_reduction_result'))
        process = subprocess.run([str(driver)], input=json.dumps(dict(request, two_aux=True)),
                                 text=True, capture_output=True, timeout=timeout)
        self.assertEqual(process.returncode, expected, process.stderr)
        return json.loads(process.stdout) if expected == 0 else process.stderr

    def test_finite_vector_panels(self):
        for case in finite_panels():
            with self.subTest(case=case):
                vectors = directions(case['inputs'], case['targets'])
                response = self.driver(case)
                raw_total = len(routes(vectors)) * len(vectors) * (len(vectors) + 1)
                self.assertEqual(response['directions'], len(vectors))
                self.assertEqual(response['raw_pair_slots'], raw_total)
                self.assertEqual(response['stats'], expected_stats(vectors, raw_total))
                expected_prepared = [raw for raw in range(raw_total)
                                     if classify(vectors, raw) == 'prepared']
                self.assertEqual([entry['raw_slot'] for entry in response['results']],
                                 expected_prepared)
                for entry in response['results']:
                    raw = entry['raw_slot']
                    first, second = slot(vectors, raw)
                    self.assertEqual(entry['helpers'], [list(first[3]), list(second[3])])
                    forward = reachable(vectors, case['inputs'], first, second)
                    self.assertEqual(entry['status'], int(forward), (case, raw))
                    if second[1] < len(vectors):
                        self.assertEqual(reachable(vectors, case['inputs'], second, first),
                                         forward, (case, raw))
                    check_witness(case, vectors, raw, entry)

    def test_raw_slot_selection_and_prefix(self):
        case = dict(inputs=3, targets=[[1, 1, 1], [1, 1, -1]])
        vectors = directions(case['inputs'], case['targets'])
        total = len(routes(vectors)) * len(vectors) * (len(vectors) + 1)
        prepared = [raw for raw in range(total) if classify(vectors, raw) == 'prepared']
        self.assertTrue(prepared)
        selected = [prepared[0], prepared[len(prepared) // 2], prepared[-1]]
        response = self.driver(dict(case, raw_slots=selected, max_pair_slots=1))
        self.assertEqual([r['raw_slot'] for r in response['results']], selected)
        for entry in response['results']:
            check_witness(case, vectors, entry['raw_slot'], entry)
        for prefix in (1, 63, 64, 65, total - 1):
            response = self.driver(dict(case, max_pair_slots=prefix, summary_only=True))
            self.assertEqual(response['stats'], expected_stats(vectors, prefix))
            self.assertEqual(response['results'], [])

        boundary = dict(inputs=2, targets=[[3, 1]])
        vectors = directions(boundary['inputs'], boundary['targets'])
        first_success = next(raw for raw in range(len(routes(vectors)) *
                           len(vectors) * (len(vectors) + 1))
                           if classify(vectors, raw) == 'prepared' and
                           reachable(vectors, boundary['inputs'], *slot(vectors, raw)))
        short = self.driver(dict(boundary, max_pair_slots=first_success,
                                 summary_only=True))
        exact = self.driver(dict(boundary, max_pair_slots=first_success + 1,
                                 summary_only=True))
        self.assertEqual(short['successes'], 0)
        self.assertNotIn('first_success', short)
        self.assertEqual(exact['successes'], 1)
        self.assertEqual(exact['first_success']['raw_slot'], first_success)
        self.assertEqual(exact['results'], [])

    def test_structure_and_route_identity(self):
        # Independent original-D routes have one canonical helper order.
        separate = dict(inputs=6, targets=[[1, 1, 1, 0, 0, 0],
                                           [0, 0, 0, 1, 1, 1]])
        vectors = directions(separate['inputs'], separate['targets'])
        forward = raw_for(vectors, (0, 1, 1), (3, 4, 1))
        reverse = raw_for(vectors, (3, 4, 1), (0, 1, 1))
        self.assertEqual(classify(vectors, forward), 'prepared')
        self.assertEqual(classify(vectors, reverse), 'symmetry_filtered')
        selected = self.driver(dict(separate, raw_slots=[reverse, forward]))
        self.assertEqual([r['raw_slot'] for r in selected['results']], [forward])
        self.assertEqual(selected['results'][0]['status'], 1)
        check_witness(separate, vectors, forward, selected['results'][0])

        # The same second vector has distinct dependent and independent routes.
        dependent = dict(inputs=2, targets=[[3, 1]])
        vectors = directions(dependent['inputs'], dependent['targets'])
        independent_raw = raw_for(vectors, (0, 1, 1), (0, 2, -1))
        dependent_raw = raw_for(vectors, (0, 1, 1), (0, len(vectors), 1))
        entries = self.driver(dict(dependent, raw_slots=[independent_raw, dependent_raw]))['results']
        self.assertEqual([e['raw_slot'] for e in entries],
                         [independent_raw, dependent_raw])
        self.assertEqual(entries[0]['helpers'], entries[1]['helpers'])
        self.assertNotEqual(entries[0]['creations'][1], entries[1]['creations'][1])
        self.assertEqual([e['status'] for e in entries], [0, 1])
        self.assertTrue(any(g[0] == len(vectors) + 1 for g in entries[1]['gates']))
        for entry in entries:
            check_witness(dependent, vectors, entry['raw_slot'], entry)

        # A valid slot may complete the targets without emitting either helper.
        unused = dict(inputs=2, targets=[[1, 1]])
        vectors = directions(unused['inputs'], unused['targets'])
        raw = raw_for(vectors, (0, 2, 1), (1, 2, 1))
        result = self.driver(dict(unused, raw_slots=[raw]))['results'][0]
        self.assertEqual(result['status'], 1)
        self.assertEqual(result['used_helpers'], 0)
        self.assertEqual(result['live_count'], 1)
        check_witness(unused, vectors, raw, result)

        # Both helper vectors are needed as operands of the final target gate.
        both = dict(inputs=4, targets=[[1, 1, 1, 1]])
        vectors = directions(both['inputs'], both['targets'])
        raw = raw_for(vectors, (0, 1, 1), (2, 3, 1))
        result = self.driver(dict(both, raw_slots=[raw]))['results'][0]
        self.assertEqual((result['status'], result['used_helpers'], result['live_count']),
                         (1, 2, 3))
        check_witness(both, vectors, raw, result)

        # The two signed targets share a dependent intermediate. Neither one
        # helper nor two independently created helpers can complete them.
        dependent4 = dict(inputs=4, targets=[[1, 1, 1, 1], [1, 1, 1, -1]])
        vectors = directions(dependent4['inputs'], dependent4['targets'])
        raw = raw_for(vectors, (0, 1, 1), (2, len(vectors), 1))
        result = self.driver(dict(dependent4, raw_slots=[raw]))['results'][0]
        self.assertEqual((result['status'], result['used_helpers'], result['live_count']),
                         (1, 2, 4))
        check_witness(dependent4, vectors, raw, result)
        self.assertFalse(any(reachable(vectors, dependent4['inputs'], first)
                             for first in routes(vectors)
                             if any(first[3]) and first[3] not in vectors))
        total = len(routes(vectors)) * len(vectors) * (len(vectors) + 1)
        self.assertFalse(any(reachable(vectors, dependent4['inputs'], *slot(vectors, r))
                             for r in range(total)
                             if classify(vectors, r) == 'prepared'
                             and slot(vectors, r)[1][1] < len(vectors)))

    def test_34_directions_and_25_gates(self):
        case = full_capacity_case()
        vectors = directions(case['inputs'], case['targets'])
        self.assertEqual(len(vectors), 32)
        raw = raw_for(vectors, (0, 1, 1), (2, 3, 1))
        result = self.driver(dict(case, raw_slots=[raw]))['results'][0]
        self.assertEqual(result['status'], 1)
        self.assertEqual(result['count'], 25)
        self.assertTrue(result['available'] & (1 << 33))
        check_witness(case, vectors, raw, result)

    def test_aliases_cycles_and_coefficient_bounds(self):
        aliased = dict(inputs=2, targets=[[0, 0], [1, 1], [-1, -1], [0, -1]])
        vectors = directions(aliased['inputs'], aliased['targets'])
        raw = next(raw for raw in range(len(routes(vectors)) * len(vectors) * (len(vectors) + 1))
                   if classify(vectors, raw) == 'prepared')
        result = self.driver(dict(aliased, raw_slots=[raw]))['results'][0]
        check_witness(aliased, vectors, raw, result)

        ordered = dict(inputs=2, targets=[[2, 1], [1, 1], [-2, -1], [2, 1]])
        vectors = directions(ordered['inputs'], ordered['targets'])
        self.assertEqual(vectors[2:], ((2, 1), (1, 1)))
        raw = next(raw for raw in range(len(routes(vectors)) * len(vectors) *
                                        (len(vectors) + 1))
                   if classify(vectors, raw) == 'prepared')
        result = self.driver(dict(ordered, raw_slots=[raw]))['results'][0]
        check_witness(ordered, vectors, raw, result)

        cyclic = dict(inputs=3, targets=[[1, 1, 1], [1, 1, -1]])
        vectors = directions(cyclic['inputs'], cyclic['targets'])
        raw = next(raw for raw in range(len(routes(vectors)) * len(vectors) * (len(vectors) + 1))
                   if classify(vectors, raw) == 'prepared'
                   and not reachable(vectors, cyclic['inputs'], *slot(vectors, raw)))
        result = self.driver(dict(cyclic, raw_slots=[raw]))['results'][0]
        self.assertEqual(result['status'], 0)
        check_witness(cyclic, vectors, raw, result)

        for case in (dict(inputs=1, targets=[[1 << 30]]),
                     dict(inputs=10, targets=[]),
                     dict(inputs=2, targets=[[1]])):
            self.driver(case, expected=1)
        maximum_input_coefficient = (1 << 31) // 4 - 1
        high = dict(inputs=1, targets=[[maximum_input_coefficient]])
        vectors = directions(high['inputs'], high['targets'])
        result = self.driver(dict(high, raw_slots=[2]))['results'][0]
        check_witness(high, vectors, 2, result)
        self.driver(dict(inputs=1, targets=[[maximum_input_coefficient + 1]]),
                    expected=1)

        near_bound = dict(inputs=2, targets=[
            [maximum_input_coefficient, 0],
            [maximum_input_coefficient - 1, 1],
            [maximum_input_coefficient - 2, -1]])
        vectors = directions(near_bound['inputs'], near_bound['targets'])
        raw = raw_for(vectors, (2, 3, 1), (4, len(vectors), 1))
        first, second = slot(vectors, raw)
        self.assertEqual(first[3], (2 * maximum_input_coefficient - 1, 1))
        self.assertEqual(second[3], (3 * maximum_input_coefficient - 3, 0))
        self.assertGreater(first[3][0] + second[3][0], (1 << 31) - 1)
        result = self.driver(dict(near_bound, raw_slots=[raw]))['results'][0]
        check_witness(near_bound, vectors, raw, result)

    def test_explicit_witness_validation(self):
        case = dict(inputs=2, targets=[[3, 1]])
        vectors = directions(case['inputs'], case['targets'])
        raw = raw_for(vectors, (0, 1, 1), (0, len(vectors), 1))
        result = self.driver(dict(case, raw_slots=[raw]))['results'][0]
        self.assertEqual(result['status'], 1)
        base = {key: deepcopy(result[key])
                for key in ('raw_slot', 'status', 'count', 'available', 'gates')}
        replay = self.driver(dict(case, witnesses=[base]))
        self.assertEqual(replay['results'][0]['status'], 1)
        check_witness(case, vectors, raw, replay['results'][0])

        bad = {}

        def altered(name, change):
            witness = deepcopy(base)
            change(witness)
            bad[name] = witness

        altered('complete_claimed_incomplete', lambda w: w.update(status=0))
        altered('invalid_status', lambda w: w.update(status=2))
        altered('wrong_availability', lambda w: w.update(available=w['available'] ^ (1 << 4)))
        altered('missing_output_gate',
                lambda w: (w['gates'].pop(), w.update(count=w['count'] - 1,
                                                      available=w['available'] ^ (1 << 2))))
        altered('unavailable_dependency', lambda w: w['gates'][0].__setitem__(1, 3))
        altered('invalid_sign', lambda w: w['gates'][0].__setitem__(3, 0))
        altered('changed_creation_route',
                lambda w: w['gates'][0].__setitem__(slice(1, 3), [1, 0]))
        altered('duplicate_gate_output', lambda w: w['gates'][1].__setitem__(0, 3))
        altered('wrong_gate_arithmetic', lambda w: w['gates'][0].__setitem__(4, -1))
        altered('too_few_rule_checks', lambda w: w.update(rule_checks=w['count'] - 1))
        altered('too_many_sweeps', lambda w: w.update(sweeps=27))
        altered('too_many_gates',
                lambda w: (w['gates'].extend([w['gates'][-1]] * 23),
                           w.update(count=26)))
        altered('mask_above_direction_limit',
                lambda w: w.update(available=w['available'] | (1 << 34)))
        altered('gate_output_above_direction_limit',
                lambda w: w['gates'][0].__setitem__(0, 34))
        altered('gate_operand_above_direction_limit',
                lambda w: w['gates'][0].__setitem__(1, 34))
        altered('minimum_integer_sign',
                lambda w: w['gates'][0].__setitem__(3, -(1 << 31)))
        altered('filtered_raw_slot', lambda w: w.update(raw_slot=0))
        altered('out_of_range_raw_slot',
                lambda w: w.update(raw_slot=len(routes(vectors)) * len(vectors) *
                                                (len(vectors) + 1)))
        altered('unclosed_negative',
                lambda w: (w['gates'].pop(), w.update(count=w['count'] - 1,
                                                      available=w['available'] ^ (1 << 2),
                                                      status=0)))
        for name, witness in bad.items():
            with self.subTest(corruption=name):
                self.driver(dict(case, witnesses=[witness]), expected=1)
        # A valid first witness must not hide an invalid tail in the same tile.
        self.driver(dict(case, witnesses=[base, bad['invalid_sign']]), expected=1)

    def test_negative_witnesses_require_a_fixed_point(self):
        def prepared(case, first_pair, second_pair):
            vectors = directions(case['inputs'], case['targets'])
            raw = raw_for(vectors, first_pair, second_pair)
            result = self.driver(dict(case, raw_slots=[raw]))['results'][0]
            return raw, result

        def partial(case, raw, gates):
            available = (1 << case['inputs']) - 1
            for gate in gates:
                available |= 1 << gate[0]
            return dict(raw_slot=raw, status=0, count=len(gates),
                        available=available, gates=deepcopy(gates))

        direct = dict(inputs=2, targets=[[1, 1]])
        direct_raw, direct_success = prepared(direct, (0, 2, 1), (1, 2, 1))
        self.assertEqual(direct_success['status'], 1)
        target_only = dict(raw_slot=direct_raw, status=1, count=1,
                           available=(1 << 3) - 1,
                           gates=deepcopy(direct_success['gates'][:1]))
        target_only_result = self.driver(dict(direct, witnesses=[target_only]))['results'][0]
        self.assertEqual((target_only_result['status'],
                          target_only_result['live_count'],
                          target_only_result['used_helpers']), (1, 1, 0))
        two_base = dict(inputs=2, targets=[[1, 1], [2, 1]])
        two_base_raw, two_base_success = prepared(two_base, (0, 1, -1), (0, 3, 1))
        self.assertEqual(two_base_success['status'], 1)

        dependent = dict(inputs=2, targets=[[3, 1]])
        vectors = directions(dependent['inputs'], dependent['targets'])
        dependent_raw, dependent_success = prepared(
            dependent, (0, 1, 1), (0, len(vectors), 1))
        self.assertEqual(dependent_success['status'], 1)

        both = dict(inputs=4, targets=[[1, 1, 1, 1]])
        both_raw, both_success = prepared(both, (0, 1, 1), (2, 3, 1))
        self.assertEqual(both_success['status'], 1)
        dependent4 = dict(inputs=4, targets=[[1, 1, 1, 1], [1, 1, 1, -1]])
        vectors = directions(dependent4['inputs'], dependent4['targets'])
        dependent4_raw, dependent4_success = prepared(
            dependent4, (0, 1, 1), (2, len(vectors), 1))
        self.assertEqual(dependent4_success['status'], 1)

        cases = (
            ('enabled_base_from_empty', direct, direct_raw, []),
            ('enabled_base_after_first_target', two_base, two_base_raw,
             two_base_success['gates'][:1]),
            ('omitted_first_helper', dependent, dependent_raw, []),
            ('omitted_dependent_second_helper', dependent, dependent_raw,
             dependent_success['gates'][:1]),
            ('omitted_target_after_dependent_helpers', dependent, dependent_raw,
             dependent_success['gates'][:2]),
            ('both_independent_helpers_enabled', both, both_raw, []),
            ('independent_second_helper_enabled', both, both_raw,
             both_success['gates'][:1]),
            ('target_enabled_by_both_helpers', both, both_raw,
             both_success['gates'][:2]),
            ('second_signed_target_enabled', dependent4, dependent4_raw,
             dependent4_success['gates'][:3]),
        )
        for name, case, raw, gates in cases:
            with self.subTest(omission=name):
                witness = partial(case, raw, gates)
                error = self.driver(dict(case, witnesses=[witness]), expected=1)
                self.assertIn('not a fixed point', error)

        cyclic = dict(inputs=3, targets=[[1, 1, 1]])
        raw, result = prepared(cyclic, (0, 3, -1), (1, 3, -1))
        self.assertEqual(result['status'], 0)
        fixed = self.driver(dict(cyclic, witnesses=[partial(cyclic, raw, [])]))
        self.assertEqual(fixed['results'][0]['status'], 0)
        self.assertGreater(fixed['results'][0]['negative_rule_checks'], 0)

        complete = {key: deepcopy(dependent_success[key])
                    for key in ('raw_slot', 'status', 'count', 'available', 'gates')}
        tail = partial(dependent, dependent_raw, dependent_success['gates'][:2])
        error = self.driver(dict(dependent, witnesses=[complete, tail]), expected=1)
        self.assertIn('not a fixed point', error)

    def test_sun_v_raw_budget_and_full_filter_census(self):
        fixture = ROOT / 'benchmarks/workflow/fixtures/fgm1/factors/sun.json'
        case = dict(inputs=9, targets=json.loads(fixture.read_text())['v'])
        vectors = directions(case['inputs'], case['targets'])
        self.assertEqual(len(vectors), 20)
        total = len(routes(vectors)) * len(vectors) * (len(vectors) + 1)
        self.assertEqual(total, 159600)
        raw = 32877
        self.assertEqual(classify(vectors, raw), 'prepared')
        selected = self.driver(dict(case, raw_slots=[raw]))['results'][0]
        self.assertEqual((selected['status'], selected['count']), (1, 13))
        check_witness(case, vectors, raw, selected)
        prefix = self.driver(dict(case, max_pair_slots=raw + 1, prepare_only=True,
                                  summary_only=True))
        self.assertEqual(prefix['stats'], expected_stats(vectors, raw + 1))
        self.assertEqual(prefix['results'], [])
        full = self.driver(dict(case, max_pair_slots=total, prepare_only=True,
                                summary_only=True), timeout=60)
        self.assertEqual(full['stats']['raw_scanned'], total)
        self.assertEqual(full['stats']['prepared'], 75342)
        self.assertEqual(full['successes'], 0)

    def test_transposed_two_aux_success(self):
        case = dict(inputs=2, targets=[[3, 1]])
        vectors = directions(case['inputs'], case['targets'])
        raw = raw_for(vectors, (0, 1, 1), (0, len(vectors), 1))
        result = self.driver(dict(case, raw_slots=[raw],
                                  transpose_two_aux=True))['results'][0]
        self.assertEqual(result['status'], 1)
        wires = [(1,)]

        def expression(terms):
            return tuple(sum(term['value'] * wires[term['index']][i]
                             for term in terms) for i in range(1))

        for fresh in result['transpose_fresh']:
            wires.append(expression(fresh))
        self.assertEqual([expression(out) for out in result['transpose_outputs']],
                         [(3,), (1,)])
        self.assertEqual(result['transpose_count'], len(result['transpose_fresh']))
        self.assertEqual(result['transpose_count'], 2)
        self.assertEqual([term['index'] for term in result['transpose_fresh'][0]],
                         [0, 0])

    def test_forward_doubling_is_excluded(self):
        case = dict(inputs=1, targets=[[2]])
        vectors = directions(case['inputs'], case['targets'])
        response = self.driver(case)
        self.assertGreater(response['stats']['prepared'], 0)
        self.assertEqual(response['successes'], 0)
        for result in response['results']:
            self.assertEqual(result['status'], 0)
            self.assertFalse(reachable(vectors, case['inputs'],
                                       *slot(vectors, result['raw_slot'])))
            check_witness(case, vectors, result['raw_slot'], result)


if __name__ == '__main__':
    unittest.main()
