"""Host protocol tests for opt-in two-auxiliary additive evaluations.

The explicit result file is a journal fixture, not a constructor result or GPU
qualification. Its one raw slot is filtered before dispatch.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_run_config import ROOT, settings


SOURCE = ROOT/'benchmarks/workflow/fixtures/fgm1/factors/cn122.json'
FAMILY = 'signed-two-aux-distinct-v1'


def fixture_result(source, factors_id, budget=1):
    costs = {}
    circuit = dict(n=source['n'], m=source['m'], z2=False)
    for key in 'uvw':
        rows = source[key] if key != 'w' else [
            [source['w'][r][i] for r in range(23)] for i in range(9)]
        outputs = [[dict(index=i, value=value) for i, value in enumerate(row) if value]
                   for row in rows]
        circuit[key] = outputs
        circuit[key+'_fresh'] = []
        costs[key] = sum(max(len(row)-1, 0) for row in outputs)
    total_cost = sum(costs.values())
    circuit['complexity'] = dict(naive=total_cost, reduced=total_cost)
    stages = {}
    for stage, output in (('u', 'u'), ('v', 'v'), ('wt', 'w')):
        basis = {tuple(int(i == j) for i in range(9)) for j in range(9)}
        targets = []
        for row in source[output]:
            sign = next((1 if value > 0 else -1 for value in row if value), 0)
            if sign:
                targets.append(tuple(sign*value for value in row))
        directions = basis | set(targets)
        n = len(directions)
        m = n-9
        q = len(targets)
        raw_total = n*(n-1)*n*(n+1)
        # Raw slot zero chooses e0+e1 twice. CN122 has no such target, so the
        # first helper is new and the second is filtered as a duplicate.
        assert (1, 1, 0, 0, 0, 0, 0, 0, 0) not in directions
        report = dict(schema='fgm-two-aux-report-v1', family=FAMILY,
                      implementation='metal-two-aux-v1',
                      enumeration_order='lexicographic-pairs-plus-first-v1',
                      stage=stage, requested_auxiliaries=2, max_pair_slots=budget,
                      target_directions=m, nonzero_output_occurrences=q,
                      improvement_floor=m+2+(q-9 if stage == 'wt' else 0),
                      raw_family_size=raw_total, prefix_limit=min(raw_total, budget),
                      baseline_cost=costs[output], incumbent_cost=costs[output],
                      final_cost=costs[output], stop_reason='budget-exhausted',
                      coverage='partial', selected_witness=None,
                      raw_scanned=budget, first_invalid=0, second_invalid=budget,
                      symmetry_filtered=0, prepared=0, dispatched=0,
                      completed=0, validated=0, gpu_rule_checks=0, gpu_sweeps=0,
                      negative_validation_rule_checks=0, logical_prefix=budget,
                      batch_tail_candidates=0, used_auxiliaries=0,
                      unvisited_raw_slots=raw_total-budget,
                      microseconds={key: 0 for key in (
                          'allocation', 'preparation', 'dispatch', 'gpu',
                          'witness_validation', 'transposition', 'verification')})
        assert costs[output] > report['improvement_floor']
        stages[stage] = report
    return dict(circuit=circuit, verified_circuit_additions=total_cost,
                verified_circuit_additions_by_stage=costs,
                baseline_additions=total_cost, baseline_additions_by_stage=costs,
                pre_two_aux_additions_by_stage=costs,
                phase_microseconds={},
                construction={key: {'status': 'family-exhausted'} for key in stages},
                stage_sources={key: 'baseline' for key in 'uvw'},
                two_auxiliary=dict(effective_factors_id=factors_id, stages=stages),
                rounds_completed=0, flip_attempts=0, flips_applied=0)


class TwoAuxSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = Path(os.environ.get('FGM_EXECUTION_DRIVER', ROOT/'build/workflow/test_execution'))
        self.journal = Path(os.environ.get('FGM_JOURNAL_DRIVER', ROOT/'build/workflow/test_journal'))
        self.source = json.loads(SOURCE.read_text())
        self.source_path = self.root/'source.json'
        self.source_path.write_text(json.dumps(self.source))
        self.fixture_path = self.root/'evaluation.json'
        self.history = self.root/'history'
        self.serial = 0
        self.config = dict(schema='fgm-run-v1', operation='search', workflow='additive-search',
            policy=dict(settings(), seed=7),
            evaluation=dict(domain='ZT', strategy='combined', seed=7, reducers=2,
                            rounds=2, no_improvements=2, schemes=1, max_flips=0,
                            target_additions=0,
                            constructor=dict(family=FAMILY, max_pair_slots=1)),
            input=dict(kind='files', files=[dict(path=str(self.source_path), format='json')]),
            execution=dict(workers=1, batch_steps=3, max_batches=1, block_size=32,
                           backend='general', memory_bytes=536870912),
            pool=dict(capacity_per_rank=16, reserve_per_rank=16, elite_capacity=8,
                      memory_bytes=8388608, stage_threshold=1, selector='cost-diverse'),
            history=dict(path=str(self.history), transaction_bytes=8388608,
                         storage_bytes=536870912, index_memory_bytes=1048576))

    def run_native(self, *, validate=False, failure='', expected=0, dynamic=False):
        self.serial += 1
        output = self.root/f'receipt-{self.serial}.json'
        config_path = self.root/f'config-{self.serial}.json'
        config_path.write_text(json.dumps(dict(self.config, output=str(output))))
        command = [str(self.binary), '--run-config', str(config_path)]
        if validate:
            command.append('--validate-only')
        process = subprocess.run(command, capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PATH':'', 'FGM_TEST_SEARCH_FAILURE':failure,
                 'FGM_TEST_EVALUATION_RESULT':str(self.fixture_path),
                 'FGM_TEST_DYNAMIC_EVALUATION':'1' if dynamic else ''})
        self.assertEqual(process.returncode, expected, process.stderr)
        receipt = json.loads(output.read_text()) if output.exists() else None
        return process, receipt

    def write_fixture(self, source=None):
        _, admission = self.run_native(validate=True)
        value = fixture_result(source or self.source,
                               admission['presentations'][0]['effective_factors_id'])
        self.fixture_path.write_text(json.dumps(value))
        return value, admission

    def test_initial_resume_and_walk_seed_change(self):
        _, admission = self.write_fixture()
        enabled_host = admission['reserved_host_bytes']
        constructor = self.config['evaluation'].pop('constructor')
        _, disabled = self.run_native(validate=True)
        self.config['evaluation']['constructor'] = constructor
        self.assertEqual(enabled_host-disabled['reserved_host_bytes'], 256*1024)
        _, first = self.run_native()
        self.assertEqual(first['status'], 'complete')
        self.assertEqual(first['reserved_host_bytes'], enabled_host)
        self.assertEqual(first['planned_buffer_bytes'], admission['planned_buffer_bytes'])
        self.assertEqual(first['evaluated_historical'], 1)
        self.assertEqual(first['evaluated_current_run'], 1)
        self.assertEqual(first['best_evaluation']['two_auxiliary']['effective_factors_id'],
                         admission['presentations'][0]['effective_factors_id'])
        self.config['input'] = dict(kind='resume', journal=str(self.history))
        self.config['policy']['seed'] = 19
        _, resumed = self.run_native()
        self.assertEqual(resumed['status'], 'complete')
        self.assertEqual(resumed['evaluated_historical'], 1)
        self.assertEqual(resumed['evaluated_current_run'], 0)
        self.assertEqual(resumed['best_evaluation'], first['best_evaluation'])

    def test_first_effective_presentation_and_contract_changes(self):
        alias = copy.deepcopy(self.source)
        for key in 'uvw':
            alias[key] = list(reversed(alias[key]))
        self.source_path.write_text(json.dumps(alias)+'\n'+json.dumps(self.source)+'\n')
        self.config['input']['files'][0]['format'] = 'jsonl'
        self.write_fixture(alias)
        _, first = self.run_native()
        self.assertEqual(first['seed_duplicates'], 1)
        self.assertEqual(first['evaluated_historical'], 1)
        self.assertEqual(first['best_evaluation']['factors_id'],
                         first['presentations'][0]['effective_factors_id'])
        self.config['input'] = dict(kind='resume', journal=str(self.history))
        for field, value in (('constructor', None), ('constructor', dict(family=FAMILY, max_pair_slots=2)),
                             ('constructor', dict(family='other', max_pair_slots=1))):
            with self.subTest(field=field, value=value):
                old = self.config['evaluation'].get(field)
                if value is None:
                    del self.config['evaluation'][field]
                else:
                    self.config['evaluation'][field] = value
                self.run_native(validate=True, expected=1)
                self.config['evaluation'][field] = old

    def test_disabled_history_rejects_enabled_resume(self):
        self.config['evaluation'].pop('constructor')
        _, first = self.run_native()
        self.assertEqual(first['evaluated_historical'], 1)
        self.config['input'] = dict(kind='resume', journal=str(self.history))
        self.config['evaluation']['constructor'] = dict(family=FAMILY, max_pair_slots=1)
        self.run_native(validate=True, expected=1)

    def test_failure_before_and_after_committed_batch(self):
        self.write_fixture()
        _, failed = self.run_native(failure='evaluation', expected=1)
        self.assertEqual(failed['journal_sequence'], 0)
        self.assertIsNone(failed.get('best_evaluation'))
        self.assertNotIn('circuit_artifact', failed)
        self.history = self.root/'batch-history'
        self.config['history']['path'] = str(self.history)
        self.config['policy'].update(flip_budget=32, control_budget=32)
        self.config['execution']['batch_steps'] = 1
        _, first = self.run_native(dynamic=True)
        self.assertEqual(first['completed_batches'], 1)
        self.assertGreater(first['evaluated_historical'], 1)
        self.config['input'] = dict(kind='resume', journal=str(self.history))
        _, after_batch = self.run_native(dynamic=True, failure='end', expected=1)
        self.assertEqual(after_batch['status'], 'failed')
        self.assertEqual(after_batch['completed_batches'], 1)
        self.assertGreater(after_batch['journal_sequence'], first['journal_sequence'])
        self.assertEqual(after_batch['evaluated_historical'], first['evaluated_historical'])
        self.assertEqual(after_batch['best_evaluation'], first['best_evaluation'])
        self.assertNotIn('circuit_artifact', after_batch)
        _, resumed = self.run_native(dynamic=True)
        self.assertEqual(resumed['status'], 'complete')
        self.assertEqual(resumed['evaluated_historical'], first['evaluated_historical'])

    def test_producer_tamper_rejected_on_replay(self):
        self.write_fixture()
        self.run_native()
        inspected = subprocess.run([str(self.journal), 'inspect', str(self.history)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        transactions = [json.loads(line) for line in inspected.stdout.splitlines()]
        transactions[0]['evaluations'][0]['producer']['executable_sha256'] = '0'*64
        forged = self.root/'forged-history'
        fixture = subprocess.run([str(self.journal), 'fixture-large', str(forged)],
            input=json.dumps(transactions), capture_output=True, text=True, timeout=30)
        self.assertEqual(fixture.returncode, 0, fixture.stderr)
        self.config['input'] = dict(kind='resume', journal=str(forged))
        self.config['history']['path'] = str(forged)
        self.run_native(validate=True, expected=1)

    def test_invalid_report_rejected_before_initial_commit(self):
        valid, _ = self.write_fixture()
        mutations = (
            lambda r: r['two_auxiliary']['stages']['u'].update(implementation='wrong'),
            lambda r: r['two_auxiliary']['stages']['u'].update(enumeration_order='wrong'),
            lambda r: r['two_auxiliary'].update(effective_factors_id='0'*64),
            lambda r: r['two_auxiliary']['stages']['u'].update(final_cost=0),
            lambda r: r['two_auxiliary']['stages']['u'].update(second_invalid=0),
            lambda r: r['two_auxiliary']['stages']['u'].update(gpu_rule_checks=1),
            lambda r: r['two_auxiliary']['stages']['u'].update(negative_validation_rule_checks=1),
            lambda r: r['stage_sources'].update(u='cancellation'),
            lambda r: r['construction']['u'].update(status='one-auxiliary'),
            lambda r: r.pop('two_auxiliary'),
            lambda r: r.pop('pre_two_aux_additions_by_stage'),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                broken = copy.deepcopy(valid)
                mutate(broken)
                self.fixture_path.write_text(json.dumps(broken))
                _, receipt = self.run_native(expected=1)
                self.assertEqual(receipt['journal_sequence'], 0)
                self.assertIsNone(receipt.get('best_evaluation'))
                self.assertNotIn('circuit_artifact', receipt)

    def test_report_envelope_keeps_existing_record_limit(self):
        valid, _ = self.write_fixture()
        stage = valid['two_auxiliary']['stages']['u']
        stage['test_padding'] = ''
        encode = lambda: json.dumps(valid, separators=(',', ':'), sort_keys=True)
        stage['test_padding'] = 'x'*(65500-len(encode()))
        self.fixture_path.write_text(encode())
        process, receipt = self.run_native(expected=2)
        self.assertIn('evaluation exceeds reserved record capacity', process.stderr)
        self.assertEqual(receipt['journal_sequence'], 0)
        self.assertIsNone(receipt.get('best_evaluation'))
        self.assertNotIn('circuit_artifact', receipt)


if __name__ == '__main__':
    unittest.main()
