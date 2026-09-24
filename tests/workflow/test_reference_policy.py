"""Host-only tests of scripted policy ordering, not arithmetic correctness."""
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reference_policy import (ARITHMETIC_REFERENCE, MODEL_KIND, ArithmeticEvent as E,
                              Config, Policy, Scripts, U32, U64, WorkerRNG, add)

ROOT = HERE.parents[1]
REPAIR_SEED = 0x9e3779b9


def host_boundary_record(case):
    """Script host acknowledgments around the independent worker, not a journal."""
    from host_rng import HostRNG
    from identity_oracle import identity
    sys.path.insert(0, str(ROOT / 'tests/metal'))
    from verify import verify

    def scheme_id(state):
        data = state['scheme']
        verify(data)
        return identity(dict(domain='F2' if data['z2'] else 'ZT', orientation='cyclic-w',
                             dimensions=data['n'], rank=data['m'],
                             **{key:data[key] for key in 'uvw'}))

    states = case['states']
    identities = {name:scheme_id(state) for name,state in states.items()}
    known = set(identities[name] for name in case['known'])
    policy = Policy(Config(**case['config']), states['initial']['scheme']['m'])
    policy.bind_state(states['initial'])
    host = None
    record = {'initial':policy.comparison_record(), 'actions':[]}
    for action in case['actions']:
        event = {'action':action['op']}
        try:
            if action['op'] == 'step':
                def events(key):
                    result = []
                    for item in action.get(key, []):
                        value = dict(item)
                        if 'state' in value:
                            value['state'] = states[value['state']]
                        result.append(E(**value))
                    return result
                script = Scripts(events('flips'), events('reductions'), events('expansions'))
                policy.step(script)
                if any(script.remaining().values()):
                    raise AssertionError('unused scripted arithmetic')
            elif action['op'] == 'request_restart':
                # Isolated host request tests installation preflight, not triggering.
                policy.pending_restart = True
            elif action['op'] == 'commit':
                candidates = ([policy.pending_target_state] if policy.pending_target_state is not None
                              else ([policy.mandatory['state']] if policy.mandatory else [])
                              + [sample['state'] for sample in policy.optional])
                ids = [scheme_id(state) for state in candidates]
                policy.commit_observations(verified=action['verified'], durable=action['durable'])
                event['new_identities'] = []
                for key in ids:
                    if key not in known:
                        event['new_identities'].append(key)
                        known.add(key)
                event['committed_identities'] = list(dict.fromkeys(ids))
            elif action['op'] == 'restart':
                if not policy.dispatch_complete or not policy.mandatory_committed:
                    raise ValueError('prior mandatory commitment required before host selection')
                if host is None:
                    host = HostRNG(policy.config.seed)
                event['host_before'] = {'state':host.words.copy(), 'index':host.index, 'draws':host.draws}
                words = []
                original = host.next
                def traced_next():
                    word = original()
                    words.append(word)
                    return word
                host.next = traced_next
                try:
                    selected = host.select(action['weights'])
                finally:
                    host.next = original
                name = action['parents'][selected]
                event.update(host_words=words, selected=name,
                             host_after={'state':host.words.copy(), 'index':host.index, 'draws':host.draws})
                policy.install_restart(states[name]['scheme']['m'], parent_state=states[name])
            elif action['op'] == 'batch_boundary':
                policy.batch_boundary()
            else:
                raise AssertionError('unknown fixture action')
            event['outcome'] = 'accepted'
        except ValueError:
            event['outcome'] = 'rejected'
        event['known_identities'] = sorted(known)
        event['worker'] = policy.comparison_record()
        record['actions'].append(event)
    return record


class RNGTests(unittest.TestCase):
    def test_repaired_state_known_answers(self):
        rng = WorkerRNG(REPAIR_SEED, 0)
        self.assertEqual(rng.state, 1)
        # Public xorshift32 state-one vector; independently hand-checked shifts.
        self.assertEqual([rng.next() for _ in range(5)],
                         [270369, 67634689, 2647435461, 307599695, 2398689233])

    def test_zero_seed_and_worker_wrapping(self):
        self.assertEqual(WorkerRNG(0, 0).state, 0x9e3779b9)
        self.assertEqual(WorkerRNG(0, U32).state, 1)
        self.assertEqual(WorkerRNG(7, 1).state, 7 ^ ((0x9e3779b9 * 2) & U32))
        for seed in (-1, U32 + 1, True):
            with self.assertRaises(ValueError):
                WorkerRNG(seed, 0)

    def test_singleton_draw_and_full_width_interval(self):
        rng = WorkerRNG(REPAIR_SEED, 0)
        self.assertEqual(rng.bounded(1), 0)
        self.assertEqual(rng.draws, 1)
        self.assertEqual(rng.interval(0, U32), 67634689)
        self.assertEqual(rng.interval(9, 9), 9)
        self.assertEqual(rng.draws, 3)
        for bounds in ((2, 1), (0, U64)):
            with self.assertRaises(ValueError):
                rng.interval(*bounds)
        with self.assertRaises(ValueError):
            rng.bounded(0)

    def test_permutation_draw_order(self):
        rng = WorkerRNG(REPAIR_SEED, 0)
        self.assertEqual(rng.permutation(0), [])
        self.assertEqual(rng.permutation(1), [0])
        self.assertEqual(rng.draws, 0)
        # 270369 % 3 == 0, 67634689 % 2 == 1.
        self.assertEqual(rng.permutation(3), [2, 1, 0])
        self.assertEqual(rng.draws, 2)


class GoldenTests(unittest.TestCase):
    def test_hand_written_boundary_traces(self):
        path = HERE / 'golden/controller_boundaries_v1.json'
        golden = json.loads(path.read_text())
        spec = ROOT / 'docs/specifications/FGM-CONTRACT-v1.md'
        self.assertEqual(hashlib.sha256(spec.read_bytes()).hexdigest(), golden['specification_sha256'])
        self.assertEqual(golden['model'], MODEL_KIND)
        self.assertEqual(golden['arithmetic_reference'], ARITHMETIC_REFERENCE)
        for case in golden['cases']:
            with self.subTest(case=case['name']):
                policy = Policy(Config(**case['config']), case['parent_rank'])
                script = Scripts(flips=[E(**event) for event in case['flips']])
                for _ in range(case['steps']):
                    policy.step(script)
                observed = policy.summary()
                for key, expected in case['expected'].items():
                    self.assertEqual(observed[key], expected, key)
                self.assertEqual([[event['event'], event['rng']] for event in policy.events],
                                 case['event_rng'])
                self.assertFalse(any(script.remaining().values()))


class HostBoundaryGoldenTests(unittest.TestCase):
    def test_frozen_host_boundary_and_checked_ceiling_traces(self):
        golden = json.loads((HERE/'golden/host_boundaries_v1.json').read_text())
        self.assertEqual(golden['review']['status'], 'reviewed')
        payload = {key:value for key,value in golden.items() if key != 'review'}
        self.assertEqual(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                                   allow_nan=False).encode()).hexdigest(),
                         golden['review']['payload_sha256'])
        self.assertEqual(golden['specification_sha256'], hashlib.sha256(
            (ROOT/'docs/specifications/FGM-CONTRACT-v1.md').read_bytes()).hexdigest())
        self.assertEqual(golden['arithmetic_reference'], ARITHMETIC_REFERENCE)
        for case in golden['cases']:
            with self.subTest(case=case['name']):
                submitted = {key:value for key,value in case.items() if key not in ('expected','input_sha256')}
                payload = json.dumps(submitted,sort_keys=True,separators=(',',':')).encode()
                self.assertEqual(hashlib.sha256(payload).hexdigest(),case['input_sha256'])
                if case['kind'] == 'host-boundary':
                    actual = host_boundary_record(case)
                else:
                    try:
                        value = (add(*case['arguments']) if case['kind']=='rank-increment'
                                 else Config(**case['config']).ceiling())
                        actual = {'outcome':'accepted','value':value}
                    except ValueError:
                        actual = {'outcome':'eligibility_error'}
                self.assertEqual(json.loads(json.dumps(actual)),case['expected'])


class ControllerTests(unittest.TestCase):
    def policy(self, rank=5, **config):
        return Policy(Config(**{'seed': REPAIR_SEED, 'anchor': 5, **config}), rank)

    def test_configuration_checks_and_ceiling(self):
        for config in (Config(interval_min=2, interval_max=1), Config(interval_min=0),
                       Config(anchor=U64, excursion=1), Config(dimensions=(U64, 2, 2))):
            with self.assertRaises(ValueError):
                Policy(config, 1)
        self.assertEqual(Config(anchor=23, excursion=100).ceiling(), 27)
        self.assertEqual(Config(anchor=23, excursion=1).ceiling(), 24)
        self.assertEqual(Config(anchor=23, excursion=100, rank_capacity=25).ceiling(), 25)
        self.assertEqual(self.policy(rank=9, dimensions=(2, 2, 2), anchor=9).rank, 9)
        with self.assertRaises(ValueError):
            add(U64, 1)

    def test_parent_preflight_no_rng_and_commit_required(self):
        policy = self.policy(target_rank=5)
        self.assertIsNone(policy.rng)
        for verified, durable in ((False, True), (True, False)):
            with self.assertRaises(ValueError):
                policy.commit_observations(verified=verified, durable=durable)
        policy.commit_observations(verified=True, durable=True)
        self.assertEqual(policy.terminal, 'target_met')
        with self.assertRaises(ValueError):
            Policy(Config(), 23, parent_verified=False)

    def test_q_zero_draw_and_batch_continuity(self):
        policy = self.policy(interval_min=5, interval_max=5)
        script = Scripts(flips=[E('applied', 5), E('applied', 5)])
        policy.step(script)
        self.assertEqual((policy.rng.draws, policy.countdown), (2, 4))
        with self.assertRaises(ValueError):
            policy.batch_boundary()
        policy.commit_observations(verified=True, durable=True)
        state = policy.rng.state
        policy.batch_boundary()
        self.assertEqual(policy.rng.state, state)
        policy.step(script)
        self.assertEqual((policy.rng.draws, policy.countdown), (3, 3))

    def test_reduction_target_stops_before_expansion(self):
        policy = self.policy(reduction_q=U32, target_rank=4)
        script = Scripts(flips=[E('applied', 5)], reductions=[E('applied', 4)])
        policy.step(script)
        self.assertEqual(policy.terminal, 'target_pending')
        self.assertEqual(policy.rng.draws, 2)
        self.assertEqual(policy.applied, {'flip': 1, 'reduction': 1, 'expansion': 0})
        self.assertEqual(policy.mandatory['operation'], 'reduction')

    def test_reduction_threshold_inclusive(self):
        for q, expected in ((67634688, 0), (67634689, 1)):
            policy = self.policy(reduction_q=q, interval_min=5, interval_max=5)
            policy.step(Scripts(flips=[E('applied', 5)], reductions=[E('applied', 4)]))
            self.assertEqual(policy.applied['reduction'], expected)

    def test_blocked_scheduled_still_redraws(self):
        policy = self.policy(rank=8, anchor=8, dimensions=(2, 2, 2), excursion=0)
        policy.step(Scripts(flips=[E('applied', 8)]))
        self.assertEqual(policy.rng.draws, 3)  # init, q=0 decision, interval
        self.assertFalse(policy.pending_restart)
        self.assertEqual(policy.countdown, 1)

    def test_two_recovery_primitives_recheck_cap(self):
        policy = self.policy(excursion=1)
        # Recovery count uses second word (odd), requesting two. Operator is
        # third word % 3 == 0. First append reaches ceiling; second is blocked.
        policy.step(Scripts(flips=[E('unsuccessful')],
                            expansions=[E('applied', 6, expected_operator=0)]))
        self.assertEqual(policy.rank, 6)
        self.assertEqual(policy.rng.draws, 4)
        self.assertEqual(policy.applied['expansion'], 1)
        self.assertFalse(policy.pending_restart)
        self.assertEqual([e['event'] for e in policy.events].count('rank_blocked'), 1)

    def test_recovery_target_prevents_second_primitive_and_redraw(self):
        policy = self.policy(target_rank=4)
        policy.step(Scripts(flips=[E('unsuccessful')], expansions=[E('applied', 4)]))
        self.assertEqual((policy.terminal, policy.rng.draws), ('target_pending', 3))

    def test_exhausted_recovery_redraws_requests_restart(self):
        policy = self.policy()
        policy.step(Scripts(flips=[E('unsuccessful')],
                            expansions=[E('proposal_exhausted'), E('coefficient_rejection')]))
        self.assertTrue(policy.pending_restart)
        self.assertEqual(policy.rng.draws, 5)
        self.assertIsNone(policy.mandatory)
        before = policy.summary()
        policy.step(Scripts())
        self.assertEqual(before, policy.summary())

    def test_applied_first_recovery_retained_when_second_rejects(self):
        policy = self.policy()
        policy.step(Scripts(flips=[E('unsuccessful')],
                            expansions=[E('applied', 6), E('proposal_exhausted')]))
        self.assertEqual(policy.rank, 6)
        self.assertFalse(policy.pending_restart)
        self.assertEqual(policy.mandatory['rank'], 6)

    def test_optional_quota_and_earliest_mandatory(self):
        policy = self.policy(optional_quota=1, interval_min=9, interval_max=9)
        policy.step(Scripts(flips=[E('applied', 5)]))
        first = policy.mandatory.copy()
        policy.step(Scripts(flips=[E('applied', 5)]))
        self.assertEqual(policy.mandatory, first)
        self.assertEqual((len(policy.optional), policy.optional_encounters, policy.optional_drops), (1, 2, 1))
        other = self.policy(mode='rank-reduction', optional_quota=0, interval_min=9, interval_max=9)
        other.step(Scripts(flips=[E('applied', 4)]))
        self.assertEqual(other.optional_drops, 1)
        self.assertEqual(other.mandatory['rank'], 4)

    def test_capacity_failure_quarantines_observations(self):
        policy = self.policy(interval_min=9, interval_max=9)
        policy.step(Scripts(flips=[E('applied', 5)]))
        policy.step(Scripts(flips=[E('capacity_error')]))
        self.assertFalse(policy.dispatch_complete)
        self.assertEqual(policy.terminal, 'capacity_error')
        self.assertIsNotNone(policy.mandatory)
        with self.assertRaises(ValueError):
            policy.commit_observations(verified=True, durable=True)

    def test_restart_preserves_mandatory_rng_and_charges_credit(self):
        policy = self.policy(interval_min=9, interval_max=9, stagnation_limit=1, control_budget=2)
        policy.step(Scripts(flips=[E('applied', 5)]))
        retained = policy.mandatory.copy()
        with self.assertRaises(ValueError):
            policy.install_restart(5)
        policy.commit_observations(verified=True, durable=True)
        policy.install_restart(5)
        self.assertEqual(policy.mandatory, retained)
        self.assertEqual((policy.rng.draws, policy.controls, policy.countdown), (3, 2, 9))
        policy.step(Scripts())
        self.assertEqual(policy.terminal, 'budget_exhausted')
        self.assertEqual(policy.flips, 1)

    def test_restart_target_no_installation_draw_or_credit(self):
        policy = self.policy(target_rank=4, interval_min=9, interval_max=9, stagnation_limit=1)
        policy.step(Scripts(flips=[E('applied', 5)]))
        policy.commit_observations(verified=True, durable=True)
        state = policy.rng.state
        policy.install_restart(4)
        self.assertEqual(policy.terminal, 'existing_target_pending')
        self.assertEqual(policy.controls, 1)
        self.assertEqual(policy.rng.state, state)
        policy.commit_observations(verified=True, durable=True)
        self.assertEqual(policy.terminal, 'target_met')

    def test_lifetime_collision_no_post_flip_draw(self):
        for limits in ({'flip_budget': 1}, {'control_budget': 1}):
            policy = self.policy(**limits)
            policy.step(Scripts(flips=[E('unsuccessful')]))
            self.assertEqual(policy.terminal, 'budget_exhausted')
            self.assertEqual(policy.rng.draws, 1)
            self.assertFalse(policy.pending_restart)
        empty = self.policy(control_budget=0)
        empty.step(Scripts())
        self.assertEqual((empty.controls, empty.flips, empty.rng.draws), (0, 0, 1))

    def test_strict_improvement_resets_stagnation(self):
        policy = self.policy(interval_min=9, interval_max=9, stagnation_limit=1)
        policy.step(Scripts(flips=[E('applied', 4)]))
        self.assertEqual(policy.stagnation, 0)
        self.assertFalse(policy.pending_restart)

    def test_stage_credit_is_not_free(self):
        policy = self.policy(control_budget=1)
        policy.charge_stage_credit()
        with self.assertRaises(ValueError):
            policy.charge_stage_credit()
        policy.step(Scripts())
        self.assertEqual((policy.flips, policy.terminal), (0, 'budget_exhausted'))

    def test_inclusive_initial_and_event_countdowns(self):
        policy = self.policy(rank=8, anchor=8, dimensions=(2, 2, 2),
                             interval_min=2, interval_max=4)
        # 270369 % 3 == 0, so initial countdown is the inclusive minimum.
        self.assertEqual(policy.countdown, 2)
        policy.step(Scripts(flips=[E('applied', 8)]))
        self.assertEqual(policy.countdown, 1)
        self.assertEqual(policy.rng.draws, 2)
        policy.step(Scripts(flips=[E('applied', 8)]))
        # Second successful-flip decision uses word3; blocked scheduled event
        # draws word4=307599695, remainder2, reaching inclusive maximum4.
        self.assertEqual(policy.countdown, 4)
        self.assertEqual(policy.rng.draws, 4)

    def test_scheduled_exhaustion_redraw_without_restart(self):
        policy = self.policy()
        policy.step(Scripts(flips=[E('applied', 5)],
                            expansions=[E('proposal_exhausted')]))
        self.assertEqual(policy.rng.draws, 4)
        self.assertEqual(policy.countdown, 1)
        self.assertFalse(policy.pending_restart)
        self.assertEqual(policy.applied['expansion'], 0)

    def test_quota_does_not_change_walk_rng_or_policy(self):
        policies = [self.policy(optional_quota=q, interval_min=9, interval_max=9)
                    for q in (0, 2)]
        for policy in policies:
            policy.step(Scripts(flips=[E('applied', 5)]))
        self.assertEqual(policies[0].rng.state, policies[1].rng.state)
        self.assertEqual(policies[0].mandatory, policies[1].mandatory)
        self.assertEqual(policies[0].countdown, policies[1].countdown)
        self.assertEqual(policies[0].optional_drops, 1)
        self.assertEqual(len(policies[1].optional), 1)

    def test_script_draws_are_explicit_not_arithmetic_proof(self):
        policy = self.policy(flip_budget=1)
        policy.step(Scripts(flips=[E('applied', 5, draws=2)]))
        self.assertEqual(policy.rng.draws, 3)
        self.assertEqual(policy.rng.state, 2647435461)
        with self.assertRaises(AssertionError):
            self.policy().step(Scripts())


if __name__ == '__main__':
    unittest.main()
