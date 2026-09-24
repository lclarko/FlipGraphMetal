"""Host-only native integration tests; explicit built adapter is required.

The reviewed tiny fixtures cover their named cases, not full GPU acceptance.
No GPU is initialized. Set FGM_SCALAR_REFERENCE to a completed receipted build.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tests/metal'))
from verify import verify
from reference_policy import Config, Policy, U32, WorkerRNG
from scalar_bridge import NativeArithmetic, check_build, instrument, operation, validate_rng


def naive(dimensions, f2=False):
    a, b, c = dimensions
    result = {'n': list(dimensions), 'm': a*b*c, 'z2': f2, 'u': [], 'v': [], 'w': []}
    for i in range(a):
        for j in range(c):
            for k in range(b):
                for key, width, index in (('u', a*b, i*b+k), ('v', b*c, k*c+j), ('w', c*a, j*a+i)):
                    row = [0]*width
                    row[index] = 1
                    result[key].append(row)
    return result


def fixture(f2=False, size=3):
    if size not in (3,4):
        raise ValueError('only public 3x3 and 4x4 fixtures')
    name = 'strassen_3x3_f2.txt' if f2 and size == 3 else f'strassen_{size}x{size}.txt'
    values = list(map(int, (ROOT / 'tests/metal/fixtures' / name).read_text().split()))
    a, b, c, m = values[:4]
    out = {'n': [a,b,c], 'm': m, 'z2': f2}
    offset = 4
    for key, width in zip('uvw', (a*b,b*c,c*a)):
        out[key] = [values[offset+r*width:offset+(r+1)*width] for r in range(m)]
        offset += m*width
    assert offset == len(values)
    if f2:
        for key in 'uvw':
            out[key] = [[value % 2 for value in row] for row in out[key]]
    return out


def wide_f2():
    # Four independent 3x3 products along shared dimension k: a public rank104
    # scheme for 3x12 by 12x3, with 36-entry U/V factors. W has 428 pairs.
    base = fixture(True)
    out = {'n': [3,12,3], 'm': 4*base['m'], 'z2': True, 'u': [], 'v': [], 'w': []}
    for block in range(4):
        for r in range(base['m']):
            u, v = [0]*36, [0]*36
            for i in range(3):
                for k in range(3):
                    u[i*12+block*3+k] = base['u'][r][i*3+k]
            for k in range(3):
                for j in range(3):
                    v[(block*3+k)*3+j] = base['v'][r][k*3+j]
            out['u'].append(u); out['v'].append(v); out['w'].append(base['w'][r].copy())
    return out


def f2_boundary_fixture(width):
    """Public F2 block sums with U/V width33 or64 and expansion headroom."""
    if width == 33:
        blocks = [fixture(True) for _ in range(3)] + [naive((3,2,3), True)]
    elif width == 64:
        block = fixture(True, 4)
        blocks = [block for _ in range(4)]
    else:
        raise ValueError('only declared boundary widths33 and64')
    a,_,c = blocks[0]['n']
    shared = sum(block['n'][1] for block in blocks)
    out = {'n':[a,shared,c], 'm':sum(block['m'] for block in blocks),
           'z2':True, 'u':[], 'v':[], 'w':[]}
    offset = 0
    for block in blocks:
        if block['n'][0] != a or block['n'][2] != c:
            raise ValueError('incompatible block dimensions')
        inner = block['n'][1]
        for r in range(block['m']):
            u,v = [0]*(a*shared),[0]*(shared*c)
            for i in range(a):
                for k in range(inner):
                    u[i*shared+offset+k] = block['u'][r][i*inner+k]
            for k in range(inner):
                for j in range(c):
                    v[(offset+k)*c+j] = block['v'][r][k*c+j]
            out['u'].append(u);out['v'].append(v);out['w'].append(block['w'][r].copy())
        offset += inner
    return out


def overflow_fixture():
    """Add cancelling F2 pairs to public rank196 input; no tensor change."""
    data = f2_boundary_fixture(64)
    used = {k:set(map(tuple,data[k])) for k in 'uvw'}
    def fresh(key, start):
        width = len(data[key][0])
        for value in range(start, start+10000):
            row = [(value >> i) & 1 for i in range(width)]
            if tuple(row) not in used[key]:
                used[key].add(tuple(row))
                return row
        raise ValueError('finite fixture construction exhausted')
    common = fresh('w',40000)
    a = fresh('w',20000)
    b = [x^y for x,y in zip(a,common)]
    if tuple(b) in used['w'] or not any(b):
        raise ValueError('fixture factors overlap')
    def pair(u,v,w):
        for _ in range(2):
            for key,row in zip('uvw',(u,v,w)):
                data[key].append(row.copy())
            data['m'] += 1
    for i in range(10):
        pair(fresh('u',1000+i*100), fresh('v',3000+i*100), common)
    u = fresh('u',8000)
    pair(u,fresh('v',9000),a)
    pair(u,fresh('v',10000),b)
    return data


class TransformationTests(unittest.TestCase):
    def test_native_rng_records_are_independently_checked(self):
        correct = {'words':[270369,67634689], 'draws':2, 'rng':67634689}
        validate_rng(1, correct)
        for changes in ({'words':[270369,0]}, {'rng':3}, {'draws':3},
                        {'words':[True,67634689]}):
            with self.assertRaises(ValueError):
                validate_rng(1, {**correct, **changes})

    def test_checked_instrumentation_rejects_unpinned_shape(self):
        with self.assertRaises(ValueError):
            instrument('core.h', b'wrong header')
        with self.assertRaises(ValueError):
            instrument('scheme_integer.h', b'private:\nprivate:\n')
        self.assertEqual(instrument('addition.h', b'unchanged'), (b'unchanged', []))


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('FGM_SCALAR_REFERENCE')
        if not value:
            raise unittest.SkipTest('explicit isolated scalar build required; native reference NOT VALIDATED')
        cls.binary = check_build(value)

    def execute(self, data, op='inspect', rng=1, ceiling=350, candidates=None):
        result = operation(self.binary, data, op, rng, ceiling, candidates)
        verify(result['scheme'])  # independent dense exact tensor equations
        self.assertEqual(result['draws'], len(result['words']))
        return result

    def test_direct_decode_preserves_signs_zeros_and_reducibility(self):
        data = naive((1,1,1))
        data.update(m=3, u=[[1],[1],[-1]], v=[[1],[1],[1]], w=[[1],[1],[1]])
        # 1 + 1 - 1 = 1; no eager reduction or sign normalization on import.
        result = self.execute(data)
        self.assertEqual(result['scheme'], data)
        reduced = self.execute(data, 'reduce')
        self.assertEqual(reduced['outcome'], 'applied')
        self.assertLess(reduced['scheme']['m'], 3)
        self.assertEqual(reduced['draws'], 0)
        self.assertEqual(reduced['removed_terms'], 3-reduced['scheme']['m'])
        self.assertIsNone(reduced['reduction_operations'])

    def test_domain_and_rectangular_exact_equations(self):
        for f2 in (False, True):
            for shape in ((1,1,1),(2,3,2)):
                with self.subTest(f2=f2, shape=shape):
                    data = naive(shape, f2)
                    inspected = self.execute(data)
                    self.assertEqual(inspected['scheme'], data)
                    flipped = self.execute(data, 'flip', candidates=inspected['candidates'])
                    self.assertIn(flipped['outcome'], ('applied','unsuccessful'))
        modular_only = {'n':[1,1,1], 'm':3, 'z2':True, 'u':[[1]]*3,'v':[[1]]*3,'w':[[1]]*3}
        self.execute(modular_only)
        modular_only['z2'] = False
        with self.assertRaisesRegex(ValueError, 'tensor invalid'):
            operation(self.binary, modular_only, 'inspect', 1, 350)

    def test_rank_blocked_no_draws(self):
        for f2 in (False, True):
            for op in ('plus','random','existing'):
                result = self.execute(naive((2,2,2), f2), op)
                self.assertEqual((result['outcome'], result['rng'], result['draws']), ('rank_blocked',1,0))

    def test_zero_rng_rejected_and_no_candidates_flip_no_draws(self):
        data = naive((1,1,1), True)
        with self.assertRaises(ValueError):
            operation(self.binary, data, 'flip', 0, 350)
        result = self.execute(data, 'flip')
        self.assertEqual((result['outcome'],result['draws']), ('unsuccessful',0))

    def test_ordered_candidates_survive_roundtrip_and_are_validated(self):
        data = naive((2,2,2))
        inspected = self.execute(data)
        lists = copy.deepcopy(inspected['candidates'])
        for group in lists:
            group['pairs'].reverse()
            if group['pairs']:
                group['pairs'][0].reverse()  # pair orientation also preserved
        result = self.execute(data, candidates=lists)
        self.assertEqual(result['candidates'], lists)
        bad = copy.deepcopy(lists)
        bad[0]['pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            operation(self.binary, data, 'inspect', 1, 350, bad)

    def test_initial_overflow_is_explicit(self):
        result = self.execute(naive((6,6,6)))
        self.assertEqual(result['outcome'], 'capacity_error')
        self.assertEqual(result['draws'], 0)
        self.assertTrue(any(x['overflow'] for x in result['candidates']))

    def test_proposals_do_not_mutate_on_rejection(self):
        seen = set()
        for f2 in (False, True):
            data = fixture(f2)
            initial = self.execute(data)
            for op in ('plus','random','existing'):
                for seed in (1,7,19):
                    result = self.execute(data, op, seed, 27, initial['candidates'])
                    seen.add(result['outcome'])
                    if result['outcome'] in ('tuple_rejection','coefficient_rejection'):
                        self.assertEqual(result['scheme'], data)
                        self.assertEqual(result['candidates'], initial['candidates'])
                    else:
                        self.assertEqual(result['outcome'], 'applied')
                        self.assertLessEqual(result['scheme']['m'], data['m']+1)
        self.assertIn('applied', seen)
        self.assertTrue({'tuple_rejection','coefficient_rejection'} & seen)

    def test_random_split_draws_and_wide_f2(self):
        for data in (fixture(False), wide_f2()):
            result = self.execute(data, 'random', 1, 350)
            # index, permutation(2), then signed sign/value OR F2 low/high.
            self.assertEqual(result['words'], [270369,67634689,2647435461,307599695,2398689233])
            if data['z2']:
                self.assertEqual(result['outcome'], 'applied')
                self.assertEqual(result['scheme']['m'], data['m']+1)
                self.assertTrue(any(result['scheme']['u'][270369 % data['m']][32:]))

    def test_f2_width33_and64_actual_random_proposals(self):
        for width, expected_rank, w_pairs in ((33,96,366),(64,196,294)):
            with self.subTest(width=width):
                data = f2_boundary_fixture(width)
                self.assertEqual(data['m'], expected_rank)
                self.assertEqual(len(data['u'][0]), width)
                initial = self.execute(data)
                self.assertEqual(initial['outcome'], 'inspected')
                self.assertEqual(len(initial['candidates'][2]['pairs']), w_pairs)
                self.assertTrue(all(not group['overflow'] for group in initial['candidates']))
                result = self.execute(data, 'random', 1, 350, initial['candidates'])
                self.assertEqual(result['outcome'], 'applied')
                self.assertEqual(result['scheme']['m'], expected_rank+1)
                self.assertEqual(result['words'], [270369,67634689,2647435461,307599695,2398689233])
                mask = (307599695 | (2398689233 << 32)) & ((1 << width)-1)
                expected = [(mask >> bit) & 1 for bit in range(width)]
                self.assertEqual(result['scheme']['u'][270369 % expected_rank], expected)

    def test_existing_split_domain_draw_order(self):
        for f2, expected in ((False,[270369,67634689,2647435461]),
                             (True,[270369,67634689,2647435461,307599695])):
            result = self.execute(fixture(f2),'existing')
            self.assertEqual(result['words'], expected)

    def test_reviewed_small_arithmetic_and_target(self):
        golden = json.loads((HERE / 'golden/scalar_small_cases_v1.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'docs/specifications/FGM-CONTRACT-v1.md').read_bytes()).hexdigest(),
                         golden['specification_sha256'])
        for case in golden['cases']:
            with self.subTest(case=case['name']):
                result = self.execute(case['input'], case['operation'], case['rng'], case['ceiling'])
                self.assertEqual(result, case['expected'])
        case = golden['controller_case']
        policy = Policy(Config(**case['config']), case['input']['m'])
        bridge = NativeArithmetic(policy, case['input'], self.binary)
        policy.step(bridge)
        self.assertEqual(policy.summary(), case['expected'])
        verify(bridge.scheme)

    def test_reviewed_arithmetic_restart_trace(self):
        golden = json.loads((HERE / 'golden/scalar_restart_v1.json').read_text())
        spec = ROOT / 'docs/specifications/FGM-CONTRACT-v1.md'
        self.assertEqual(hashlib.sha256(spec.read_bytes()).hexdigest(), golden['specification_sha256'])
        receipt = json.loads((self.binary.parent / 'receipt.json').read_text())
        self.assertEqual({name: row['original_sha256'] for name, row in receipt['headers'].items()},
                         golden['arithmetic_headers'])
        policy = Policy(Config(**golden['config']), golden['parent']['m'])
        initial = self.execute(golden['parent'], rng=policy.rng.state, ceiling=policy.config.ceiling())
        self.assertEqual(initial, golden['initial'])
        bridge = NativeArithmetic(policy, golden['parent'], self.binary)
        for key in ('accepted_step','pending_restart'):
            policy.step(bridge)
            verify(bridge.scheme)
            self.assertEqual(policy.summary(), golden[key]['policy'])
            self.assertEqual(bridge.trace[-1]['result'], golden[key]['native'])
        policy.commit_observations(verified=True, durable=True)
        bridge.install_restart(golden['parent'])
        installed = self.execute(golden['parent'], rng=policy.rng.state, ceiling=policy.config.ceiling())
        self.assertEqual(installed, golden['installed_restart']['native'])
        self.assertEqual(policy.summary(), golden['installed_restart']['policy'])
        verify(golden['mandatory_scheme'])
        self.assertEqual(policy.mandatory['rank'], golden['mandatory_scheme']['m'])

    def test_reviewed_boundary_traces(self):
        # Replay checks reviewed expectations; it is not their independent derivation.
        golden = json.loads((HERE / 'golden/scalar_boundaries_v1.json').read_text())
        self.assertEqual(golden['review']['status'], 'reviewed')
        payload = {key:value for key,value in golden.items() if key != 'review'}
        self.assertEqual(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                                   allow_nan=False).encode()).hexdigest(),
                         golden['review']['payload_sha256'])
        spec = ROOT / 'docs/specifications/FGM-CONTRACT-v1.md'
        self.assertEqual(hashlib.sha256(spec.read_bytes()).hexdigest(), golden['specification_sha256'])
        receipt = json.loads((self.binary.parent / 'receipt.json').read_text())
        self.assertEqual(receipt['arithmetic_reference'], golden['arithmetic_reference'])
        self.assertEqual({name: row['original_sha256'] for name, row in receipt['headers'].items()},
                         golden['arithmetic_headers'])

        def expanded(value):
            if isinstance(value, list):
                return [expanded(item) for item in value]
            if isinstance(value, dict):
                for name in ('schemes', 'candidates'):
                    if set(value) == {'$' + name}:
                        return copy.deepcopy(golden['value_tables'][name][value['$' + name]])
                return {key: expanded(item) for key, item in value.items()}
            return value

        for case in golden['cases']:
            with self.subTest(case=case['name']):
                declared = case['fixture']
                if declared['kind'] == 'wide-f2':
                    data = f2_boundary_fixture(declared['width'])
                elif declared['kind'] == 'public':
                    data = fixture(declared['f2'], declared['size'])
                else:
                    data = naive((1, 1, 1), declared['f2'])
                    if declared['terms'] == 3:
                        data.update(m=3, u=[[1], [1], [1 if declared['f2'] else -1]],
                                    v=[[1], [1], [1]], w=[[1], [1], [1]])
                payload = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), case['fixture_sha256'])
                self.assertEqual('F2' if data['z2'] else 'ZT', case['domain'])
                for name, expected in case['source_files'].items():
                    self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
                verify(data)
                if case['kind'] == 'primitive':
                    initial = self.execute(data)
                    result = self.execute(data, case['invocation']['operation'],
                                          case['invocation']['rng'], case['invocation']['ceiling'],
                                          initial['candidates'])
                    actual = {'initial': initial, 'result': result}
                else:
                    policy = Policy(Config(**case['config']), data['m'])
                    bridge = NativeArithmetic(policy, data, self.binary,
                                              proposal_limit=case['proposal_limit'])
                    for step in range(case['steps']):
                        policy.step(bridge)
                        verify(bridge.scheme)
                        if case['split_after'] == step + 1:
                            policy.commit_observations(verified=True, durable=True)
                            policy.batch_boundary()
                    for action in case.get('actions', []):
                        if action == 'step':
                            policy.step(bridge)
                            verify(bridge.scheme)
                        elif action == 'commit':
                            policy.commit_observations(verified=True, durable=True)
                        elif action == 'request_restart':
                            policy.pending_restart = True
                        elif action == 'install_restart':
                            bridge.install_restart(expanded(case['restart_scheme']))
                        else:
                            self.fail('unknown trace action: ' + action)
                    if case.get('reject_commit'):
                        with self.assertRaises(ValueError):
                            policy.commit_observations(verified=True, durable=True)
                    actual = policy.comparison_record()
                self.assertEqual(actual, expanded(case['expected']))

    def test_restart_installs_different_same_rank_parent_and_retains_snapshots(self):
        for f2 in (False, True):
            data = naive((2,2,2), f2)
            policy = Policy(Config(seed=7, anchor=8, dimensions=(2,2,2),
                                   interval_min=100, interval_max=100), 8)
            bridge = NativeArithmetic(policy, data, self.binary)
            policy.step(bridge)
            self.assertIsNotNone(policy.mandatory)
            original_capture = copy.deepcopy(policy.mandatory)
            original_optional = copy.deepcopy(policy.optional)
            original_best = copy.deepcopy(policy.best_state)
            verify(original_capture['state']['scheme'])
            verify(original_best['scheme'])
            self.assertEqual(original_best['scheme']['m'], policy.best)
            parent = copy.deepcopy(bridge.scheme)
            for key in 'uvw':
                parent[key] = list(reversed(parent[key]))
            self.assertNotEqual(parent, bridge.scheme)
            verify(parent)
            # Explicit host restart request isolates installation from trigger policy.
            policy.pending_restart = True
            policy.commit_observations(verified=True, durable=True)
            with self.assertRaises(ValueError):
                policy.install_restart(parent['m'])
            before = policy.rng.state
            expected_initial = operation(self.binary, parent, 'inspect', before, policy.config.ceiling())
            bridge.install_restart(parent)
            self.assertEqual(bridge.scheme, parent)
            self.assertEqual(bridge.candidates, expected_initial['candidates'])
            self.assertEqual(policy.state, bridge.snapshot())
            self.assertEqual(policy.best_state, bridge.snapshot())
            self.assertEqual(policy.mandatory, original_capture)
            self.assertEqual(policy.optional, original_optional)
            expected_flip = operation(self.binary, parent, 'flip', policy.rng.state,
                                      policy.config.ceiling(), bridge.candidates)
            trace_index = len(bridge.trace)
            policy.step(bridge)
            self.assertEqual(bridge.trace[trace_index]['result'], expected_flip)
            self.assertEqual(policy.mandatory, original_capture)
            verify(bridge.scheme)
            summary = policy.summary()
            summary['state']['scheme']['u'][0][0] = 999
            self.assertNotEqual(summary['state'], policy.state)

    def test_existing_target_binds_verified_state_without_rng_or_installation(self):
        target = naive((1,1,1), True)
        initial = Policy(Config(anchor=1, dimensions=(1,1,1), target_rank=1), 1)
        bridge = NativeArithmetic(initial, target, self.binary)
        self.assertIsNone(initial.rng)
        self.assertEqual(initial.pending_target_state, bridge.snapshot())
        initial.commit_observations(verified=True, durable=True)
        self.assertEqual(initial.terminal, 'target_met')
        parent = copy.deepcopy(target)
        parent['m'] = 3
        for key in 'uvw':
            parent[key] *= 3
        policy = Policy(Config(anchor=3, dimensions=(1,1,1), target_rank=1), 3)
        bridge = NativeArithmetic(policy, parent, self.binary)
        policy.pending_restart = True
        rng, controls = policy.rng.state, policy.controls
        bridge.install_restart(target)
        self.assertEqual(policy.terminal, 'existing_target_pending')
        self.assertEqual(policy.pending_target_state['scheme'], target)
        self.assertEqual(policy.state['scheme'], parent)
        self.assertEqual(bridge.scheme, parent)
        self.assertEqual((policy.rng.state, policy.controls), (rng, controls))
        with self.assertRaises(ValueError):
            policy.commit_observations(verified=True, durable=False)
        policy.commit_observations(verified=True, durable=True)
        self.assertEqual(policy.terminal, 'target_met')

    def test_initial_parent_dimensions_must_match_policy(self):
        policy = Policy(Config(anchor=8, dimensions=(1,2,4)), 8)
        with self.assertRaises(ValueError):
            NativeArithmetic(policy, naive((2,2,2), True), self.binary)

    def test_restart_rejects_invalid_tensor_without_state_change(self):
        data = naive((2,2,2), True)
        policy = Policy(Config(anchor=8, dimensions=(2,2,2)), 8)
        bridge = NativeArithmetic(policy, data, self.binary)
        policy.pending_restart = True
        before = copy.deepcopy(policy.summary())
        malformed = copy.deepcopy(data)
        malformed['u'][0] = [0]*4
        with self.assertRaises(ValueError):
            bridge.install_restart(malformed)
        self.assertEqual(policy.summary(), before)
        self.assertEqual(bridge.scheme, data)

    def test_controlled_q_endpoints_and_split_batch_equivalence(self):
        for f2 in (False, True):
            for q in (0, U32):
                with self.subTest(f2=f2, q=q):
                    data = fixture(f2)
                    config = Config(seed=7, anchor=data['m'], reduction_q=q,
                                    flip_budget=4, interval_min=100, interval_max=103,
                                    optional_quota=1)
                    walks = []
                    for split in (False, True):
                        policy = Policy(config, data['m'])
                        bridge = NativeArithmetic(policy, data, self.binary)
                        # Direct xorshift32 KAT for seed7: first word1360045822.
                        self.assertEqual((policy.rng.state,policy.rng.draws,policy.countdown),
                                         (1360045822,1,102))
                        for step in range(4):
                            policy.step(bridge)
                            verify(bridge.scheme)
                            if split and step == 1:
                                before = (policy.rng.state, policy.rng.draws, policy.countdown)
                                policy.commit_observations(verified=True, durable=True)
                                policy.batch_boundary()
                                self.assertEqual(before, (policy.rng.state, policy.rng.draws, policy.countdown))
                        record = policy.comparison_record()
                        decisions = [e for e in record['events'] if e['event'] == 'reduction_decision']
                        self.assertTrue(decisions)
                        reductions = [e for e in record['events'] if e['event'] == 'reduction']
                        self.assertEqual(len(reductions), 0 if q == 0 else len(decisions))
                        self.assertEqual(record['attempted']['reduction'], len(reductions))
                        self.assertEqual(record['attempted']['flip'], 4)
                        self.assertEqual(record['remaining']['flips'], 0)
                        self.assertEqual(record['summary']['terminal'], 'budget_exhausted')
                        replay = WorkerRNG(config.seed, config.worker)
                        self.assertEqual(record['worker_rng_words'], [replay.next() for _ in range(policy.rng.draws)])
                        self.assertEqual([w for e in record['events'] for w in e['words']], record['worker_rng_words'])
                        native = [n for e in record['events'] for n in e.get('native_observations', [])]
                        self.assertEqual(native, bridge.trace)
                        self.assertEqual(record['initial_state']['scheme'], data)
                        self.assertEqual(record['native_call_count'], len(native))
                        self.assertEqual(sum(record['native_outcome_counts'].values()), len(native))
                        captures = record['completed_capture_batches'] + [{
                            'optional': record['summary']['optional'],
                            'encounters': record['summary']['optional_encounters'],
                            'drops': record['summary']['optional_drops']}]
                        eligible = sum(e['event'] == 'observed' and e['rank'] == config.anchor
                                       for e in record['events'])
                        self.assertEqual(sum(c['encounters'] for c in captures), eligible)
                        self.assertEqual(sum(len(c['optional'])+c['drops'] for c in captures), eligible)
                        self.assertEqual(sum(n['result']['removed_terms'] or 0 for n in native),
                                         record['summary']['removed_terms']['total'])
                        walks.append(record)
                    whole, split = walks
                    for key in ('worker_rng_words', 'timers', 'remaining', 'attempted', 'applied'):
                        self.assertEqual(whole[key], split[key])
                    for key in ('state', 'best_state', 'rank', 'best', 'removed_terms', 'terminal'):
                        self.assertEqual(whole['summary'][key], split['summary'][key])
                    # Capture counters reset at a committed batch boundary. All
                    # arithmetic observations and their ordered states must agree.
                    def walk_events(record):
                        return [{k:v for k,v in e.items() if k not in ('optional_encounters','optional_drops')}
                                for e in record['events']]
                    self.assertEqual(walk_events(whole), walk_events(split))
                    self.assertEqual(len(split['completed_capture_batches']), 1)

    def test_controlled_single_proposal_exhaustion_records_exact_draws(self):
        # Fixed public fixtures and seed witnesses. Expected proposal words are
        # independently xorshift-replayed below; additional seeds supplement
        # the frozen full-state traces.
        for f2, seed, op, outcome, words in (
                (False, 1, 'random', 'coefficient_rejection',
                 [2129252540,3677366947,316708785,46153978,1363130165]),
                (True, 28, 'plus', 'tuple_rejection', [2017205331,2373333128])):
            with self.subTest(f2=f2):
                data = fixture(f2)
                policy = Policy(Config(seed=seed, anchor=data['m'], flip_budget=3), data['m'])
                bridge = NativeArithmetic(policy, data, self.binary, proposal_limit=1)
                policy.step(bridge)
                verify(bridge.scheme)
                record = policy.comparison_record()
                expansion = [e for e in record['events'] if e['event'] == 'expansion']
                self.assertEqual(len(expansion), 1)
                event = expansion[0]
                self.assertEqual(event['outcome'], 'proposal_exhausted')
                self.assertEqual(len(event['native_observations']), 1)
                proposal = event['native_observations'][0]
                self.assertEqual((proposal['operation'], proposal['result']['outcome']), (op, outcome))
                self.assertEqual(proposal['result']['words'], words)
                replay = WorkerRNG(0, 0)
                replay.state = proposal['rng_before']
                self.assertEqual(words, [replay.next() for _ in words])
                # Failed proposal does not commit its temporary factor state.
                self.assertEqual(proposal['result']['scheme'], bridge.trace[0]['result']['scheme'])
                self.assertEqual(record['applied']['expansion'], 0)
                self.assertEqual(record['attempted']['expansion'], 1)
                self.assertFalse(record['summary']['pending_restart'])
                self.assertEqual(record['events'][-1]['event'], 'event_countdown')
                self.assertEqual(len(record['events'][-1]['words']), 1)

    def test_native_bounded_invocation_matches_individual_proposals(self):
        for f2, seed in ((False, 1), (True, 28)):
            data = fixture(f2)
            policy = Policy(Config(seed=seed, anchor=data['m'], flip_budget=3), data['m'])
            bridge = NativeArithmetic(policy, data, self.binary, proposal_limit=1)
            policy.step(bridge)
            start = bridge.trace[-1]
            before = bridge.trace[0]['result']
            bounded = operation(self.binary, before['scheme'], start['operation'],
                                start['rng_before'], policy.config.ceiling(),
                                before['candidates'], proposal_limit=3)
            verify(bounded['scheme'])
            self.assertGreater(len(bounded['observations']), 1)
            rng, scheme, candidates, expected_words = start['rng_before'], before['scheme'], before['candidates'], []
            for entry in bounded['observations']:
                one = operation(self.binary, scheme, start['operation'], rng,
                                policy.config.ceiling(), candidates)
                self.assertEqual(entry, {'operation': start['operation'], 'rng_before': rng, 'result': one})
                expected_words.extend(one['words'])
                rng, scheme, candidates = one['rng'], one['scheme'], one['candidates']
            self.assertEqual(bounded['words'], expected_words)
            self.assertEqual(bounded['scheme'], scheme)
            self.assertEqual(bounded['candidates'], candidates)
            last = bounded['observations'][-1]['result']['outcome']
            expected = 'proposal_exhausted' if last in ('tuple_rejection','coefficient_rejection') else last
            self.assertEqual(bounded['outcome'], expected)
            if expected == 'proposal_exhausted':
                self.assertEqual(len(bounded['observations']), 3)

    def test_actual_one_two_recoveries_and_second_primitive_ceiling(self):
        cases = ((0,2,'alternatives',1), (0x9e3779b9,2,'rank-reduction',2),
                 (0x9e3779b9,1,'alternatives',2), (0x9e3779b9,1,'rank-reduction',2))
        for f2 in (False,True):
            for seed,excursion,mode,requested in cases:
                with self.subTest(f2=f2,seed=seed,excursion=excursion,mode=mode):
                    data = fixture(f2,size=4)
                    verify(data)
                    policy = Policy(Config(seed=seed,anchor=49,dimensions=(4,4,4),
                                           mode=mode,excursion=excursion),49)
                    bridge = NativeArithmetic(policy,data,self.binary,proposal_limit=8)
                    # Every factor is distinct, so the flip has no candidates
                    # and consumes no arithmetic words before recovery selection.
                    for key in 'uvw':
                        self.assertEqual(len(set(map(tuple,data[key]))),49)
                    self.assertTrue(all(not group['pairs'] for group in bridge.candidates))
                    policy.step(bridge)
                    verify(bridge.scheme)
                    record = policy.comparison_record()
                    events = record['events']
                    recovery = next(e for e in events if e['event']=='expansion_event')
                    self.assertTrue(recovery['recovery'])
                    self.assertEqual(recovery['count'],requested)
                    self.assertEqual(len(recovery['words']),1)
                    self.assertEqual(1+recovery['words'][0]%2,requested)
                    self.assertEqual(bridge.trace[0]['result']['words'],[])
                    self.assertEqual(bridge.trace[0]['result']['outcome'],'unsuccessful')
                    applied = min(requested,excursion)
                    expansions = [e for e in events if e['event']=='expansion']
                    self.assertEqual(len(expansions),applied)
                    self.assertEqual(record['applied'],{'flip':0,'reduction':0,'expansion':applied})
                    self.assertEqual(policy.rank,49+applied)
                    expected = ['initial_countdown','flip','expansion_event']
                    for index,event in enumerate(expansions):
                        expected += ['expansion','observed']
                        self.assertEqual(event['outcome'],'applied')
                        self.assertEqual(event['rank'],50+index)
                        proposals = event['native_observations']
                        # One operator word precedes all native proposal words.
                        self.assertEqual(event['operator'],event['words'][0]%3)
                        self.assertEqual(event['words'][1:],
                                         [w for attempt in proposals for w in attempt['result']['words']])
                        self.assertEqual(proposals[-1]['result']['outcome'],'applied')
                        verify(proposals[-1]['result']['scheme'])
                    if requested>excursion:
                        expected.append('rank_blocked')
                        blocked = next(e for e in events if e['event']=='rank_blocked')
                        self.assertEqual(blocked['operation'],'primitive')
                        self.assertEqual(blocked['words'],[])
                        self.assertEqual(blocked['rng_before'],blocked['rng'])
                        self.assertEqual(blocked['attempted']['expansion'],1)
                    expected.append('event_countdown')
                    self.assertEqual([e['event'] for e in events],expected)
                    self.assertEqual(len(events[-1]['words']),1)
                    self.assertFalse(policy.pending_restart)
                    self.assertEqual(policy.mandatory['rank'],50)
                    self.assertEqual(policy.mandatory['state'],
                                     {'scheme':expansions[0]['native_observations'][-1]['result']['scheme'],
                                      'candidates':expansions[0]['native_observations'][-1]['result']['candidates']})

    def test_actual_midwalk_overflow_preserves_prior_committed_evidence(self):
        data = overflow_fixture()
        verify(data)
        initial = self.execute(data)
        def pairs(scheme):
            return [[[i,j] for i in range(scheme['m']) for j in range(i+1,scheme['m'])
                     if scheme[key][i] == scheme[key][j]] for key in 'uvw']
        expected = pairs(data)
        self.assertEqual([len(p) for p in expected], [16,12,486])
        for actual, full in zip(initial['candidates'],expected):
            self.assertFalse(actual['overflow'])
            self.assertEqual({tuple(sorted(p)) for p in actual['pairs']},set(map(tuple,full)))
        result = self.execute(data,'flip',2859722289,350,initial['candidates'])
        self.assertEqual(result['words'],[11,2974059,738900491])
        self.assertEqual(result['outcome'],'capacity_error')
        self.assertEqual(len(pairs(result['scheme'])[2]),505)
        self.assertEqual(result['candidates'][2]['overflow'],1)
        self.assertEqual(len(result['candidates'][2]['pairs']),500)
        # Seed obtained by inverting six xorshift steps before the known state:
        # init, three F2 flip words, q=0 decision, restart countdown.
        parent = f2_boundary_fixture(64)
        config = Config(seed=2812048845, anchor=220, dimensions=(4,16,4),
                        interval_min=100, interval_max=100)
        policy = Policy(config,parent['m'])
        bridge = NativeArithmetic(policy,parent,self.binary)
        policy.step(bridge)
        verify(bridge.scheme)
        policy.commit_observations(verified=True,durable=True)
        committed = copy.deepcopy(policy.mandatory)
        self.assertIsNotNone(committed)
        policy.pending_restart = True
        bridge.install_restart(data)
        self.assertEqual(policy.rng.state,2859722289)
        policy.step(bridge)
        self.assertEqual(policy.terminal,'capacity_error')
        self.assertFalse(policy.dispatch_complete)
        self.assertEqual(policy.mandatory,committed)
        with self.assertRaises(ValueError):
            policy.commit_observations(verified=True,durable=True)
        event = policy.comparison_record()['events'][-1]
        self.assertEqual(event['event'],'capacity_error')
        self.assertEqual(event['native_observations'][0]['result'],result)
        verify(committed['state']['scheme'])

    def test_bounded_transport_rejects_malformed_native_observations(self):
        data = fixture(True)
        policy = Policy(Config(seed=28,anchor=data['m'],flip_budget=3),data['m'])
        bridge = NativeArithmetic(policy,data,self.binary,proposal_limit=1)
        policy.step(bridge)
        start, before = bridge.trace[-1], bridge.trace[0]['result']
        args = (self.binary,before['scheme'],start['operation'],start['rng_before'],
                policy.config.ceiling(),before['candidates'])
        rejected = operation(*args,proposal_limit=1)
        accepted = operation(*args,proposal_limit=3)
        self.assertEqual(rejected['outcome'],'proposal_exhausted')
        self.assertEqual(accepted['outcome'],'applied')
        cases = []
        # Real one-attempt exhaustion cannot satisfy a declared quota of two.
        cases.append(('early exhaustion',copy.deepcopy(rejected),2))
        extra = copy.deepcopy(accepted)
        terminal = copy.deepcopy(extra['observations'][-1])
        terminal['rng_before'] = terminal['result']['rng']
        terminal['result'].update(words=[],draws=0)
        extra['observations'].append(terminal)
        cases.append(('after terminal',extra,len(extra['observations'])))
        for field in ('scheme','candidates'):
            changed = copy.deepcopy(rejected)
            state = changed['observations'][0]['result'][field]
            if field == 'scheme':
                state['u'][0][0] ^= 1
            else:
                group = next(g for g in state if g['pairs'])
                group['pairs'][0].reverse()
            # Aggregate and final observation agree; the violation is specifically
            # mutation by a rejected proposal, not a mismatched outer record.
            changed[field] = copy.deepcopy(state)
            cases.append(('rejected '+field,changed,1))
        for field in ('scheme','candidates'):
            changed = copy.deepcopy(accepted)
            if field == 'scheme':
                changed[field]['u'][0][0] ^= 1
            else:
                group = next(g for g in changed[field] if g['pairs'])
                group['pairs'][0].reverse()
            cases.append(('aggregate '+field,changed,3))
        for label,malformed,quota in cases:
            with self.subTest(case=label):
                # Build/header identity checks remain real; only the process reply
                # is replaced. All RNG records remain independently consistent.
                reply = SimpleNamespace(returncode=0,stdout=json.dumps(malformed),stderr='')
                with patch('scalar_bridge.subprocess.run',return_value=reply) as process:
                    with self.assertRaises(ValueError):
                        operation(*args,proposal_limit=quota)
                    process.assert_called_once()

    def test_real_arithmetic_drives_independent_policy(self):
        for f2 in (False,True):
            data=fixture(f2)
            policy=Policy(Config(seed=7,anchor=data['m'],flip_budget=3,interval_min=2,interval_max=2),data['m'])
            bridge=NativeArithmetic(policy,data,self.binary,proposal_limit=4)
            for _ in range(3):
                policy.step(bridge)
                verify(bridge.scheme)
                self.assertEqual(policy.rank,bridge.scheme['m'])
                if policy.pending_restart:
                    break
            self.assertTrue(bridge.trace)
            self.assertEqual(policy.rng.state, bridge.trace[-1]['result']['rng']) if policy.terminal else None
            # Replay identical input/config through separate state; this is a
            # determinism check, not an independently frozen golden trace.
            other=Policy(policy.config,data['m'])
            other_bridge=NativeArithmetic(other,data,self.binary,proposal_limit=4)
            for _ in range(policy.flips):
                other.step(other_bridge)
            self.assertEqual(other.summary(),policy.summary())
            self.assertEqual(other_bridge.trace,bridge.trace)


if __name__ == '__main__':
    unittest.main()
