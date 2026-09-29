"""Host-only schedule checks for the proposed M4 manifest."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/workflow'))
import baseline
import two_aux_protocol as m4


def fixtures():
    cohorts = {'CAL2': ['sun', 'cn122'], 'SKIP2': ['original', 'laderman'],
               'E4': ['smirnov', 'eligible-2', 'eligible-3', 'eligible-4'],
               'U8': [f'candidate-{i}' for i in range(8)]}
    factors = []
    for i, (cohort, id_) in enumerate((cohort, id_) for cohort, ids in cohorts.items() for id_ in ids):
        factors.append({'id': id_, 'cohort': cohort, 'path': f'factors/{id_}.json',
                        'sha256': f'{i+1:064x}', 'canonical_id': f'fgm-scheme-v1:{i+101:064x}',
                        'effective_factors_id': f'fgm-factors-v1:{i+201:064x}'})
    # Production identities are hashes; these distinct synthetic test digests
    # exercise format and uniqueness without asserting content verification.
    panel = {'schema': 'fgm-two-aux-panel-v1', 'cohorts': cohorts, 'factors': factors,
             'freeze_bindings': {'bank_sha256': f'{301:064x}',
                                 'selection_sha256': f'{302:064x}',
                                 'screening_sha256': f'{303:064x}'}}
    builds = {}
    for i, role in enumerate(('baseline', 'candidate')):
        builds[role] = {'source_commit': f'{i+401:040x}',
                        'library_sha256': f'{i+501:064x}',
                        'executables': {name: {'path': f'{role}/bin/{name}', 'sha256': f'{i*10+j+601:064x}'}
                                        for j, name in enumerate(sorted(m4.EXECUTABLES))}}
    return panel, builds


class TwoAuxProtocolTests(unittest.TestCase):
    def setUp(self):
        self.panel, self.builds = fixtures()
        self.manifest = m4.plan(self.panel, self.builds, 'example-001')

    def test_fixed_counts_conditions_and_seed_order(self):
        experiments = self.manifest['experiments']
        for name in ('A', 'B'):
            attempts = experiments[name]['attempts']
            self.assertEqual(len(attempts), 80)
            self.assertEqual(sum(len(a['cells']) for a in attempts), 480)
            for factor_index in range(16):
                for repeat in range(1, 6):
                    cells = attempts[factor_index*5 + repeat-1]['cells']
                    expected_seeds = [m4.SEEDS[(factor_index+repeat+offset)%3] for offset in range(3)]
                    self.assertEqual([cells[i]['trial_seed'] for i in (0, 2, 4)], expected_seeds)
                    for seed_index in range(3):
                        pair = cells[seed_index*2:seed_index*2+2]
                        self.assertEqual([c['evaluation_settings'] for c in pair],
                                         [self.manifest['evaluator_quantum']]*2)
                        self.assertEqual([c['reduction_seed'] for c in pair], [expected_seeds[seed_index]]*2)
                        self.assertEqual({c['treatment']['id'] for c in pair},
                                         {'baseline-omitted', 'candidate-omitted'} if name == 'A' else
                                         {'candidate-omitted', 'candidate-enabled'})
                        for cell in pair:
                            self.assertEqual('constructor' in cell['treatment'],
                                             cell['treatment']['id'] == 'candidate-enabled')
                        actual = pair[0]['treatment']['id']
                        first = ('baseline-omitted' if name == 'A' else 'candidate-omitted')
                        parity = (factor_index + m4.SEEDS.index(expected_seeds[seed_index]) + repeat) % 2
                        self.assertEqual(actual == first, parity == 0)
                    self.assertEqual(attempts[factor_index*5 + repeat-1]['condition']['kind'], 'separate-factor')

    def test_timed_attempts_and_pinned_seeds(self):
        experiments = self.manifest['experiments']
        cs = experiments['C-S']['attempts']
        self.assertEqual(len(cs), 6)
        self.assertEqual([len(a['arms']) for a in cs], [12]*6)
        self.assertEqual([[a['seconds'] for a in attempt['arms']].count(20) for attempt in cs], [12]*6)
        self.assertEqual({tuple(a['endpoints']) for attempt in cs for a in attempt['arms']}, {(10, 20)})
        self.assertEqual([id_ for attempt in cs for id_ in attempt['factor_ids']],
                         self.panel['cohorts']['E4'] + self.panel['cohorts']['U8'])
        cn = experiments['C-N']['attempts']
        self.assertEqual(len(cn), 2)
        self.assertEqual([len(a['arms']) for a in cn], [6, 6])
        self.assertEqual([a['factor_ids'] for a in cn],
                         [['original', 'laderman', 'smirnov'], self.panel['cohorts']['U8']])
        self.assertEqual({(a['seconds'], tuple(a['endpoints'])) for attempt in cn for a in attempt['arms']},
                         {(60, (30, 60))})
        for name in ('C-S', 'C-N'):
            for attempt in experiments[name]['attempts']:
                self.assertEqual(attempt['condition']['kind'],
                                 'shared-roster' if name == 'C-N' else 'separate-factor')
                for arm in attempt['arms']:
                    if name == 'C-S':
                        self.assertEqual(arm['evaluation_settings'], self.manifest['evaluator_quantum'])
                        self.assertEqual(arm['reduction_seed_rule']['namespace'], arm['factor_ids'][0])
                        self.assertEqual(arm['first_reduction_seed'], baseline.effectiveness_seed(
                            arm['factor_ids'][0], arm['trial_seed'], 'reduce', 0))
                        self.assertTrue({'native_search_settings', 'generation_seed_rule',
                                         'first_generation_seed', 'evaluation_base_seed'}.isdisjoint(arm))
                    else:
                        self.assertEqual(arm['native_search_settings'], experiments[name]['native_search'])
                        self.assertEqual(arm['first_generation_seed'], baseline.fgm3_generation_seed(arm['trial_seed'], 0))
                        self.assertEqual(arm['evaluation_base_seed'], arm['trial_seed'])
                        self.assertTrue({'evaluation_settings', 'reduction_seed_rule',
                                         'first_reduction_seed'}.isdisjoint(arm))
                    self.assertEqual(arm['condition']['factor_ids'], arm['factor_ids'])
        self.assertNotIn('native_search', experiments['C-S'])
        self.assertEqual(cn[0]['arms'][0]['treatment']['id'], 'candidate-omitted')
        self.assertEqual(cn[1]['arms'][0]['treatment']['id'], 'candidate-enabled')

    def test_budgets_and_config(self):
        for experiment in self.manifest['experiments'].values():
            self.assertEqual(sum(experiment['phases_seconds_per_attempt'].values()), 900)
            self.assertEqual(experiment['attempt_budget_seconds'], 900)
        self.assertEqual(self.manifest['resources']['wired_limit_bytes'], 4294967296)
        self.assertEqual(self.manifest['resources']['prelaunch_bytes'] +
                         self.manifest['resources']['launch_reserve_bytes'], 4294967296)
        self.assertEqual(self.manifest['evaluator_quantum']['reducers'], 128)
        search = self.manifest['experiments']['C-N']['native_search']
        self.assertEqual((search['execution']['workers'], search['execution']['batch_steps'],
                          search['execution']['max_batches'], search['execution']['block_size']), (1, 32, 1, 32))
        self.assertEqual(search['pool']['selector'], 'cost-diverse')
        self.assertEqual(search['pool']['capacity_per_rank'], 16)
        self.assertEqual(search['pool']['elite_capacity'], 8)
        self.assertEqual(search['circuit_target'], 54)

    def test_accounting_and_report_boundaries(self):
        accounting = self.manifest['accounting']
        self.assertIn('independently exact-verified', accounting['endpoint_credit'])
        self.assertIn('never credit them retroactively', accounting['late_results'])
        self.assertIn('arm clock', accounting['time'])
        self.assertIn('planned denominators', accounting['missing_results'])
        self.assertIn('blocks measurement', accounting['qualification_failure'])
        self.assertIsNone(self.manifest['gates']['numeric_performance_allowance'])
        self.assertIn('numeric allowance', accounting['performance_verdict'])
        self.assertEqual(set(accounting['metrics']), {'A', 'B', 'C-S', 'C-N'})
        for cohort in ('CAL2', 'SKIP2', 'E4', 'U8', 'C-N G3'):
            self.assertIn(cohort, accounting['cohort_reporting'])
        self.assertIn('exclude', accounting['cohort_reporting']['CAL2'])
        self.assertIn('exclude', accounting['cohort_reporting']['SKIP2'])
        self.assertIn('separately', accounting['cohort_reporting']['E4'])
        self.assertIn('including failures and ineligible', accounting['cohort_reporting']['U8'])

    def test_explicit_qualification_schedule(self):
        q = self.manifest['qualification']
        fixed = q['fixed_factor_schedule']
        native = q['native_schedule']
        self.assertEqual(len(fixed), 16)
        self.assertEqual(sum(len(a['cells']) for a in fixed), 96)
        self.assertEqual(len(native), 4)
        self.assertEqual(sum(len(a['units']) for a in native), 24)
        for a in fixed:
            self.assertEqual({(c['trial_seed'], c['treatment']['id']) for c in a['cells']},
                             {(s, t) for s in (7, 19, 41) for t in ('candidate-omitted', 'candidate-enabled')})
            self.assertEqual(a['calibrated_allowance_ceiling_seconds'], 20)
            self.assertEqual(sum(a['phases_seconds'].values()), 900)
        for a in native:
            self.assertEqual(sum(a['phases_seconds'].values()), 900)
            self.assertEqual(a['calibrated_allowance_ceiling_seconds'], 60)
            self.assertEqual([u['trial_seed'] for u in a['units']], [7, 7, 19, 19, 41, 41])
            for i in range(0, 6, 2):
                init, resume = a['units'][i:i+2]
                self.assertEqual((init['operation'], resume['operation']), ('initialize', 'resume'))
                self.assertEqual(init['history_id'], resume['history_id'])
                for chunk, unit in enumerate((init, resume)):
                    self.assertEqual(unit['chunk'], chunk)
                    self.assertEqual(unit['generation_seed'], baseline.fgm3_generation_seed(unit['trial_seed'], chunk))
                    self.assertEqual(unit['evaluation_base_seed'], unit['trial_seed'])

    def test_input_rejection(self):
        bad = []
        item = copy.deepcopy(self.panel); item['cohorts']['U8'][0] = 'sun'; bad.append((item, self.builds))
        item = copy.deepcopy(self.panel); item['factors'][1]['canonical_id'] = item['factors'][0]['canonical_id']; bad.append((item, self.builds))
        item = copy.deepcopy(self.panel); item['factors'][1]['effective_factors_id'] = item['factors'][0]['effective_factors_id']; bad.append((item, self.builds))
        item = copy.deepcopy(self.panel); item['factors'][0]['path'] = '../escaped'; bad.append((item, self.builds))
        item = copy.deepcopy(self.panel); item['factors'][0]['sha256'] = '0'*64; bad.append((item, self.builds))
        item = copy.deepcopy(self.panel); item['cohorts']['E4'][0] = 'other'; bad.append((item, self.builds))
        item = copy.deepcopy(self.builds); item['candidate']['executables']['flip_graph']['path'] = '/absolute'; bad.append((self.panel, item))
        item = copy.deepcopy(self.builds); item['candidate']['source_commit'] = 'a'*40; bad.append((self.panel, item))
        for panel, builds in bad:
            with self.subTest(panel=panel, builds=builds), self.assertRaises(ValueError):
                m4.plan(panel, builds, 'example-001')

    def test_validator_rejects_schedule_and_budget_mutations(self):
        m4.validate_manifest(self.manifest)
        for mutate in (
            lambda v: v['experiments']['A']['attempts'][0]['cells'][0].update(reduction_seed=42),
            lambda v: v['experiments']['B']['attempts'][1]['cells'][0]['treatment'].update(build='baseline'),
            lambda v: v['experiments']['C-S']['attempts'][0]['arms'][0].update(seconds=21),
            lambda v: v['experiments']['C-N']['attempts'][1]['arms'].reverse(),
            lambda v: v['experiments']['C-N']['phases_seconds_per_attempt'].update(arms=361),
            lambda v: v['resources'].update(wired_limit_bytes=1),
            lambda v: v['gates'].update(measurement_approval='approved'),
            lambda v: v['accounting'].update(endpoint_credit='all results'),
            lambda v: v['experiments']['C-S']['attempts'][0]['arms'][0].update(first_generation_seed=1),
            lambda v: v['experiments']['C-N']['attempts'][0]['arms'][0].update(first_reduction_seed=1),
            lambda v: v['panel']['factors'][0].update(sha256=f'{999:064x}'),
            lambda v: v['builds']['candidate'].update(source_commit=f'{999:040x}'),
            lambda v: v['input_bindings'].update(panel_sha256=f'{999:064x}'),
        ):
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                m4.validate_manifest(changed)

    def test_cli_creates_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel, builds, output = root/'panel.json', root/'builds.json', root/'manifest.json'
            panel.write_text(json.dumps(self.panel))
            builds.write_text(json.dumps(self.builds))
            command = [sys.executable, str(ROOT/'benchmarks/workflow/two_aux_protocol.py'),
                       '--panel', str(panel), '--builds', str(builds),
                       '--experiment-id', 'example-001', '--output', str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(json.loads(output.read_text()), self.manifest)
            before = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
