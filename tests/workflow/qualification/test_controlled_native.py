"""Production controlled arithmetic/controller against independent pinned references.

Host-only focused verification, using drivers built by the existing Make workflow.
It does not establish GPU or packed equivalence or production persistence.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import re
import os
from pathlib import Path
import subprocess
import sys
import unittest

from reference_policy import Config, Policy
from scalar_bridge import NativeArithmetic, check_build, operation
from test_scalar_reference import fixture, naive, f2_boundary_fixture, verify

ROOT = Path(__file__).resolve().parents[3]


def text_scheme(scheme):
    rows = [' '.join(map(str, scheme['n'] + [scheme['m']]))]
    rows += [' '.join(map(str, row)) for key in 'uvw' for row in scheme[key]]
    return '\n'.join(rows) + '\n'


def normalized(source):
    result = deepcopy(source)
    if not result['z2']:
        for r in range(result['m']):
            for key in 'uv':
                first = next((x for x in result[key][r] if x), 0)
                if first < 0:
                    result[key][r] = [-x for x in result[key][r]]
                    result['w'][r] = [-x for x in result['w'][r]]
    return result


def settings(source, **changes):
    values = dict(seed=0, mode='alternatives', anchor=source['m'],
                  dimensions=tuple(source['n']), excursion=3,
                  interval_min=2, interval_max=4, reduction_q=0,
                  flip_budget=10, control_budget=30, stagnation_limit=100,
                  optional_quota=3)
    values.update(changes)
    return Config(**values)


def config_json(config, f2, limit):
    return dict(schema='fgm-controlled-config-v1', policy='controlled-v1',
                mode=config.mode, domain='F2' if f2 else 'ZT', seed=config.seed,
                dimensions=list(config.dimensions),
                **{'collection_rank' if config.mode == 'alternatives' else 'stage_rank': config.anchor},
                excursion=config.excursion, interval_min=config.interval_min,
                interval_max=config.interval_max, reduction_q=config.reduction_q,
                stagnation_limit=config.stagnation_limit, flip_budget=config.flip_budget,
                control_budget=config.control_budget, optional_quota=config.optional_quota,
                proposal_limit=limit, target_rank=config.target_rank)


class ControlledNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = Path(os.environ.get('FGM_CONTROLLED_DRIVER', ROOT/'build/workflow/test_controlled'))
        reference = os.environ.get('FGM_SCALAR_REFERENCE')
        if not cls.driver.is_file() or not reference:
            raise RuntimeError('build native controlled driver and supply FGM_SCALAR_REFERENCE')
        cls.reference = Path(reference)
        check_build(cls.reference)

    def native(self, source, config, commands, limit=7, rng=None, expected=0):
        request = dict(config=config_json(config, source['z2'], limit),
                       parent_text=text_scheme(source), commands=commands, worker=config.worker)
        if rng is not None:
            request['rng'] = rng
        result = subprocess.run([str(self.driver), 'F2' if source['z2'] else 'ZT'],
                                input=json.dumps(request), text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stderr)
        return json.loads(result.stdout) if not expected else result

    def assert_snapshot(self, actual, policy):
        expected = dict(state=policy.state, best_state=policy.best_state,
                        rng=None if policy.rng is None else policy.rng.state,
                        countdown=policy.countdown, stagnation=policy.stagnation,
                        flips=policy.flips, controls=policy.controls,
                        attempted=policy.attempted, applied=policy.applied,
                        mandatory=policy.mandatory, optional=policy.optional,
                        optional_encounters=policy.optional_encounters,
                        optional_drops=policy.optional_drops, terminal=policy.terminal,
                        pending_restart=policy.pending_restart,
                        mandatory_committed=policy.mandatory_committed,
                        removed_terms=policy.removed_terms)
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)

    def compare_policy(self, source, config, commands, actual=None):
        if actual is None:
            actual = self.native(source, config, commands)['snapshots']
        parent = normalized(source)
        policy = Policy(config, parent['m'])
        arithmetic = NativeArithmetic(policy, parent, self.reference, proposal_limit=7)
        self.assert_snapshot(actual[0], policy)
        for command, snapshot in zip(commands, actual[1:]):
            before = 0 if policy.rng is None else len(policy.rng.words)
            if command == 'step':
                policy.step(arithmetic)
            elif command == 'commit':
                policy.commit_observations(verified=True, durable=True)
            elif command == 'boundary':
                policy.batch_boundary()
            elif command == 'restart':
                arithmetic.install_restart(parent)
            elif command == 'stage':
                policy.charge_stage_credit()
            else:
                raise AssertionError('unknown comparison command')
            self.assert_snapshot(snapshot, policy)
            self.assertEqual(snapshot['words'], [] if policy.rng is None else policy.rng.words[before:])
        return actual

    def test_bounded_proposals_complete_state(self):
        for f2 in (False, True):
            source = fixture(f2)
            config = settings(source)
            for rng in (1, 2, 7, 17, 99, 123456, 4294967295):
                for op in ('plus', 'random', 'existing'):
                    with self.subTest(f2=f2, rng=rng, op=op):
                        actual = self.native(source, config, [op], rng=rng)['snapshots']
                        initial = actual[0]['state']
                        expected = operation(self.reference, initial['scheme'], op, rng,
                                             config.ceiling(), candidates=initial['candidates'], proposal_limit=7)
                        self.assertEqual(actual[1]['state'], {key: expected[key] for key in ('scheme', 'candidates')})
                        for key in ('rng', 'outcome', 'words'):
                            self.assertEqual(actual[1][key], expected[key], key)

    def test_individual_rejected_proposal_trace(self):
        observed = set()
        for f2 in (False, True):
            source = fixture(f2)
            config = settings(source)
            for rng in (1, 2, 7, 17, 99, 123456, 4294967295):
                for op in ('plus', 'random', 'existing'):
                    with self.subTest(f2=f2, rng=rng, op=op):
                        actual = self.native(source, config, [op], limit=1, rng=rng)['snapshots']
                        initial = actual[0]['state']
                        expected = operation(self.reference, initial['scheme'], op, rng,
                                             config.ceiling(), candidates=initial['candidates'], proposal_limit=1)
                        proposal = expected['observations'][0]['result']
                        observed.add(proposal['outcome'])
                        self.assertEqual(actual[1]['state'], {key: expected[key] for key in ('scheme', 'candidates')})
                        for key in ('rng', 'words', 'outcome'):
                            self.assertEqual(actual[1][key], expected[key], key)
                        for result, counter in (('tuple_rejection', 'tuple_rejections'),
                                                ('coefficient_rejection', 'coefficient_rejections')):
                            self.assertEqual(actual[1][counter], int(proposal['outcome'] == result))
                        if proposal['outcome'] in ('tuple_rejection', 'coefficient_rejection'):
                            self.assertEqual(actual[1]['state'], initial)
        self.assertTrue({'tuple_rejection', 'coefficient_rejection', 'applied'} <= observed)

    def test_controller_modes_thresholds_and_capture_quotas(self):
        for f2 in (False, True):
            source = fixture(f2)
            for mode in ('alternatives', 'rank-reduction'):
                for q in (0, 4294967295):
                    for quota in (0, 3):
                        with self.subTest(f2=f2, mode=mode, q=q, quota=quota):
                            self.compare_policy(source, settings(source, mode=mode, reduction_q=q,
                                                               optional_quota=quota), ['step']*10)

    def test_commit_boundary_and_restart_preserve_rng_and_lifetime(self):
        for f2 in (False, True):
            source = naive((1, 1, 1), f2)
            config = settings(source, interval_min=1, interval_max=1, optional_quota=0)
            self.compare_policy(source, config, ['step', 'commit', 'boundary', 'restart', 'stage', 'step'])
            source = fixture(f2)
            self.compare_policy(source, settings(source, stagnation_limit=1),
                                ['step', 'commit', 'boundary', 'restart', 'step'])

    def test_uncommitted_mandatory_capture_blocks_boundary(self):
        for f2 in (False, True):
            source = fixture(f2)
            self.native(source, settings(source), ['step', 'boundary'], expected=1)
            self.native(source, settings(source, stagnation_limit=1), ['step', 'restart'], expected=1)

    def test_existing_target_and_zero_budget(self):
        for f2 in (False, True):
            source = naive((1, 1, 1), f2)
            self.compare_policy(source, settings(source, target_rank=1, optional_quota=0),
                                ['step', 'commit', 'boundary'])
            self.compare_policy(source, settings(source, flip_budget=0), ['step'])

    def test_target_on_last_flip_precedes_reduction_and_budget(self):
        for f2 in (False, True):
            source = naive((1, 1, 1), f2)
            # Valid rank-two presentation above the naive ceiling, with a zero W term.
            source['m'] = 2
            source['u'].append([1]); source['v'].append([1]); source['w'].append([0])
            config = settings(source, target_rank=1, flip_budget=1, reduction_q=4294967295)
            actual = self.compare_policy(source, config, ['step', 'commit'])
            self.assertEqual(actual[1]['terminal'], 'target_pending')
            self.assertEqual(actual[1]['attempted']['reduction'], 0)
            self.assertEqual(actual[1]['attempted']['expansion'], 0)
            self.assertEqual(actual[2]['terminal'], 'target_met')

    def test_f2_width_boundaries_random_proposals(self):
        for width in (33, 64):
            source = f2_boundary_fixture(width)
            config = settings(source)
            for rng in (17, 123456):
                with self.subTest(width=width, rng=rng):
                    actual = self.native(source, config, ['random'], rng=rng)['snapshots']
                    initial = actual[0]['state']
                    expected = operation(self.reference, initial['scheme'], 'random', rng,
                                         config.ceiling(), candidates=initial['candidates'], proposal_limit=7)
                    self.assertEqual(actual[1]['state'], {key: expected[key] for key in ('scheme', 'candidates')})
                    for key in ('rng', 'outcome', 'words'):
                        self.assertEqual(actual[1][key], expected[key], key)

    # GPU cases are added only on explicit request, never by the ordinary host suite.
    if os.environ.get('FGM_CONTROLLED_GPU_EVIDENCE'):
        def test_requested_gpu_complete_state(self):
            sys.path.insert(0, str(ROOT/'benchmarks/metal'))
            from guard import run as guarded_run
            evidence = Path(os.environ['FGM_CONTROLLED_GPU_EVIDENCE']).resolve()
            evidence.mkdir(parents=True, exist_ok=False)
            for f2 in (False, True):
                domain = 'F2' if f2 else 'ZT'
                variable = 'FGM_CONTROLLED_GPU_F2' if f2 else 'FGM_CONTROLLED_GPU_ZT'
                binary = Path(os.environ[variable]).resolve(strict=True)
                cases = []
                source = fixture(f2)
                for mode in ('alternatives', 'rank-reduction'):
                    config = settings(source, mode=mode, optional_quota=1,
                                      interval_min=1, interval_max=2,
                                      seed=0x9e3779b9 if mode == 'alternatives' else 0,
                                      reduction_q=4294967295 if mode == 'rank-reduction' else 0)
                    cases.append((mode, source, config, ['step', 'step', 'commit', 'boundary', 'step'], 7, None))
                tiny = naive((1, 1, 1), f2)
                cases.append(('existing-target', tiny, settings(tiny, target_rank=1, optional_quota=0),
                              ['step', 'commit', 'boundary'], 7, None))
                cases.append(('zero-budget', tiny, settings(tiny, flip_budget=0), ['step'], 7, None))
                cases.append(('restart', source, settings(source, stagnation_limit=1),
                              ['step', 'commit', 'boundary', 'restart', 'step'], 7, None))
                padded = deepcopy(tiny)
                padded['m'] = 2
                padded['u'].append([1]); padded['v'].append([1]); padded['w'].append([0])
                cases.append(('last-attempt-target', padded,
                              settings(padded, target_rank=1, flip_budget=1, reduction_q=4294967295),
                              ['step', 'commit'], 7, None))
                cases.append(('proposal-exhaustion', source, settings(source, optional_quota=0), ['plus'], 1, 7))
                if f2:
                    for width in (33, 64):
                        wide = f2_boundary_fixture(width)
                        cases.append(('width'+str(width), wide, settings(wide), ['random'], 7, 17))
                requests = []
                for name, parent, config, commands, limit, rng in cases:
                    item = dict(config=config_json(config, f2, limit), parent_text=text_scheme(parent), commands=commands)
                    if rng is not None:
                        item['rng'] = rng
                    requests.append(item)
                request_path = evidence/(domain+'.json')
                request_path.write_text(json.dumps({'cases': requests}, sort_keys=True)+'\n')
                output = evidence/domain
                record = guarded_run([str(binary), domain, '--gpu', '--request', str(request_path)], output)
                self.assertTrue(record['complete'], record)
                lines = (output/'run.log').read_text().splitlines()
                devices = [line.removeprefix('Metal device: ').strip() for line in lines if line.startswith('Metal device: ')]
                self.assertEqual(len(devices), 1)
                self.assertTrue(devices[0])
                libraries = [line for line in lines if line.startswith('Metal library: ')]
                self.assertEqual(len(libraries), 1)
                self.assertRegex(libraries[0], r'^Metal library: .+ SHA256 [0-9a-f]{64}$')
                dispatches = [line for line in lines if line.startswith('Metal dispatch ')]
                expected_dispatches = sum(command in ('step', 'plus', 'random', 'existing')
                                          for item in requests for command in item['commands'])
                self.assertEqual(len(dispatches), expected_dispatches)
                timings = []
                for line in dispatches:
                    match = re.fullmatch(r'Metal dispatch (controlledGeneralKernel|controlledTestKernel): 1 threads, (\S+) ms GPU', line)
                    self.assertIsNotNone(match, line)
                    elapsed = float(match.group(2))
                    self.assertTrue(math.isfinite(elapsed) and elapsed > 0, line)
                    timings.append(elapsed)
                actuals = [json.loads(line) for line in lines if line.startswith('{')]
                self.assertEqual(len(actuals), len(cases))
                verified = set()
                for (name, parent, config, commands, limit, rng), actual in zip(cases, actuals):
                    with self.subTest(domain=domain, case=name):
                        self.assertEqual(actual, self.native(parent, config, commands, limit=limit, rng=rng))
                        for snapshot in actual['snapshots']:
                            states = [snapshot['state'], snapshot['best_state']]
                            if snapshot['mandatory'] is not None:
                                states.append(snapshot['mandatory']['state'])
                            states.extend(item['state'] for item in snapshot['optional'])
                            for state in states:
                                scheme = state['scheme']
                                self.assertEqual(scheme['z2'], f2)
                                key = json.dumps(scheme, sort_keys=True)
                                if key not in verified:
                                    verify(scheme)
                                    verified.add(key)
                        if rng is None:
                            self.compare_policy(parent, config, commands, actual=actual['snapshots'])
                        else:
                            initial = actual['snapshots'][0]['state']
                            expected = operation(self.reference, initial['scheme'], commands[0], rng,
                                                 config.ceiling(), candidates=initial['candidates'], proposal_limit=limit)
                            final = actual['snapshots'][-1]
                            self.assertEqual(final['state'], {key: expected[key] for key in ('scheme', 'candidates')})
                            for key in ('outcome', 'rng', 'words'):
                                self.assertEqual(final[key], expected[key])
                            if name == 'proposal-exhaustion':
                                self.assertEqual(final['outcome'], 'proposal_exhausted')
                (evidence/(domain+'-verification.json')).write_text(json.dumps(dict(
                    device=devices[0], library=libraries[0], dispatch_gpu_ms=timings,
                    cases=[item[0] for item in cases], exact_verified_schemes=len(verified),
                    domain=domain, complete=True), sort_keys=True)+'\n')

    if os.environ.get('FGM_CONTROLLED_GPU_PACKED'):
        def test_requested_packed_complete_state(self):
            sys.path.insert(0, str(ROOT/'benchmarks/metal'))
            from guard import run as guarded_run
            evidence = Path(os.environ['FGM_CONTROLLED_GPU_EVIDENCE']+'-packed').resolve()
            replay = os.environ.get('FGM_CONTROLLED_GPU_REPLAY') == '1'
            if not replay:
                evidence.mkdir(parents=True, exist_ok=False)
            binary = Path(os.environ['FGM_CONTROLLED_GPU_PACKED']).resolve(strict=True)
            source = fixture(False)
            cases = []
            for mode in ('alternatives', 'rank-reduction'):
                config = settings(source, mode=mode, interval_min=1, interval_max=2,
                                  reduction_q=4294967295 if mode == 'rank-reduction' else 0)
                cases.append((mode, config, ['step', 'step', 'commit', 'boundary', 'step'], 1, None, 7))
            for op in ('plus', 'random', 'existing'):
                cases.append((op, settings(source), [op], 1, 7, 1))
            for workers in (31, 32, 33):
                mode = 'rank-reduction' if workers == 32 else 'alternatives'
                cases.append(('lanes'+str(workers), settings(source, mode=mode), ['step', 'step'], workers, None, 7))
            requests = []
            for name, config, commands, workers, rng, limit in cases:
                item = dict(config=config_json(config, False, limit), parent_text=text_scheme(source),
                            commands=commands, workers=workers)
                if rng is not None:
                    item['rng'] = rng
                requests.append(item)
            request_path = evidence/'requests.json'
            request_bytes = json.dumps({'cases': requests}, sort_keys=True)+'\n'
            if replay:
                self.assertEqual(request_path.read_text(), request_bytes)
            else:
                request_path.write_text(request_bytes)
            runs = {}
            for backend in ('general', 'packed'):
                argv = [str(binary), 'ZT', '--gpu', '--request', str(request_path)]
                if backend == 'packed':
                    argv.append('--packed')
                output = evidence/backend
                if replay:
                    result_bytes = (output/'result.json').read_bytes()
                    self.assertEqual(hashlib.sha256(result_bytes).hexdigest(), (output/'result.sha256').read_text().strip())
                    result = json.loads(result_bytes)
                    self.assertEqual(result['argv'], argv)
                    self.assertEqual(result['artifacts']['run.log'], hashlib.sha256((output/'run.log').read_bytes()).hexdigest())
                else:
                    result = guarded_run(argv, output)
                self.assertTrue(result['complete'], result)
                lines = (output/'run.log').read_text().splitlines()
                devices = [line for line in lines if line.startswith('Metal device: ')]
                libraries = [line for line in lines if line.startswith('Metal library: ')]
                self.assertEqual(len(devices), 1); self.assertTrue(devices[0].removeprefix('Metal device: ').strip())
                self.assertEqual(len(libraries), 1)
                self.assertRegex(libraries[0], r'^Metal library: .+ SHA256 [0-9a-f]{64}$')
                expected = [(case[3], command) for case in cases for command in case[2]
                            if command in ('step', 'plus', 'random', 'existing')]
                dispatches = [line for line in lines if line.startswith('Metal dispatch ')]
                self.assertEqual(len(dispatches), len(expected))
                timings = []
                for line, (workers, command) in zip(dispatches, expected):
                    match = re.fullmatch(r'Metal dispatch (\w+): (\d+) threads, (\S+) ms GPU', line)
                    self.assertIsNotNone(match, line)
                    self.assertEqual(int(match.group(2)), workers)
                    permitted = {'controlledPackedReductionKernel', 'controlledPackedAlternativesKernel'} if backend == 'packed' else {'controlledGeneralKernel'}
                    if command != 'step':
                        permitted = {'controlledPackedTestKernel' if backend == 'packed' else 'controlledTestKernel'}
                    self.assertIn(match.group(1), permitted)
                    elapsed = float(match.group(3))
                    self.assertTrue(math.isfinite(elapsed) and elapsed > 0, line)
                    timings.append(elapsed)
                records = [json.loads(line) for line in lines if line.startswith('{')]
                self.assertEqual(len(records), len(cases))
                runs[backend] = dict(records=records, device=devices[0], library=libraries[0], dispatch_gpu_ms=timings)
            self.assertEqual(runs['general']['device'], runs['packed']['device'])
            self.assertEqual(runs['general']['library'], runs['packed']['library'])
            verified = set()
            for case, general, packed in zip(cases, runs['general']['records'], runs['packed']['records']):
                name, config, commands, workers, rng, limit = case
                with self.subTest(case=name):
                    self.assertEqual(packed.pop('packed_buffer_bytes'), 26400*((workers+31)//32*32))
                    self.assertEqual(packed, general)
                    for lane in range(workers):
                        local_config = replace(config, worker=lane)
                        native = self.native(source, local_config, commands, rng=rng, limit=limit)
                        actual = [entry['lanes'][lane] if workers > 1 else entry for entry in packed['snapshots']]
                        self.assertEqual(actual, native['snapshots'])
                        if rng is None:
                            self.compare_policy(source, local_config, commands, actual=actual)
                        for snapshot in actual:
                            states = [snapshot['state'], snapshot['best_state']]
                            if snapshot['mandatory'] is not None:
                                states.append(snapshot['mandatory']['state'])
                            states.extend(item['state'] for item in snapshot['optional'])
                            for state in states:
                                self.assertIs(state['scheme']['z2'], False)
                                key = json.dumps(state['scheme'], sort_keys=True)
                                if key not in verified:
                                    verify(state['scheme']); verified.add(key)
            (evidence/'verification.json').write_text(json.dumps(dict(
                complete=True, cases=[case[0] for case in cases], exact_verified_schemes=len(verified),
                device=runs['packed']['device'], library=runs['packed']['library'],
                general_gpu_ms=runs['general']['dispatch_gpu_ms'], packed_gpu_ms=runs['packed']['dispatch_gpu_ms']), sort_keys=True)+'\n')
