"""Host-only checks of M4 reporting denominators and evidence gates."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/workflow'))
import two_aux_report as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SummaryTest(unittest.TestCase):
    def test_stage_credit_uses_incumbent_and_smaller_family_gate(self):
        skipped_floor = dict(baseline_cost=20, incumbent_cost=14, final_cost=14,
                             improvement_floor=16, stop_reason='proved-no-improvement',
                             raw_scanned=0)
        skipped_smaller = dict(baseline_cost=20, incumbent_cost=19, final_cost=19,
                               improvement_floor=16, stop_reason='smaller-family-succeeded',
                               raw_scanned=0)
        eligible = dict(baseline_cost=20, incumbent_cost=19, final_cost=18,
                        improvement_floor=16, stop_reason='bound-attained',
                        raw_scanned=12)
        self.assertEqual(r.stage_outcomes({'stages': {'u': skipped_floor,
                                                       'v': skipped_smaller}}),
                         {'eligible': 0, 'escalated': 0, 'improved': 0})
        self.assertEqual(r.stage_outcomes({'stages': {'w': eligible}}),
                         {'eligible': 1, 'escalated': 1, 'improved': 1})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'runs'
        self.manifest = self.base / 'manifest.json'
        self.unit = lambda ident, treatment: dict(id=ident, factor_id='f', trial_seed=7,
                                                   repeat=1, treatment={'id': treatment})
        self.attempt = dict(id='pair', factor_id='f', cells=[self.unit('a', 'baseline-omitted'),
                                                             self.unit('b', 'candidate-omitted')])
        manifest = dict(schema='fgm-two-aux-experiment-manifest-v1',
                        panel={'factors': [{'id': 'f', 'cohort': 'E4'}],
                               'cohorts': {'CAL2': [], 'SKIP2': [], 'E4': [], 'U8': []}},
                        qualification={'fixed_factor_schedule': [], 'native_schedule': []},
                        experiments={'A': {'attempts': [self.attempt]}, 'B': {'attempts': []},
                                     'C-S': {'attempts': []}, 'C-N': {'attempts': []}},
                        resources={'wired_limit_bytes': 4})
        self.hash = write(self.manifest, manifest)
        self.patch = patch.object(r, 'MANIFEST_SHA', self.hash)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def fixed(self, ident, additions, elapsed=2):
        directory = self.root / 'measurement/A/pair'
        row = dict(id=ident, status='verified', additions=additions,
                   additions_by_stage={'u': additions - 2, 'v': 1, 'w': 1},
                   elapsed_seconds=elapsed, durable_seconds=elapsed)
        receipt = directory / ident / 'receipt.json'
        artifact = directory / ident / 'artifact.jsonl'
        native_result = {'circuit': {'uvw': ident},
                         'phase_microseconds': {'construction_u': 7},
                         'dispatch_microseconds': 11, 'verification_microseconds': 2}
        row['receipt_path'] = str(receipt.relative_to(directory))
        row['receipt_sha256'] = write(receipt, {'results': [native_result],
            'planned_buffer_bytes': 191876132, 'reserved_host_bytes': 134217728,
            'admission_content_bytes': 451288, 'memory_accounting': 'accounted scope',
            'dispatch_microseconds': 12, 'verification_microseconds': 3})
        row['gpu'] = {'gpu_all_seconds': 0.004, 'dispatches': [['kernel', 1, 4]]}
        row['artifact_path'] = str(artifact.relative_to(directory))
        row['artifact_sha256'] = write(artifact, {})
        evidence = directory / ident / 'verification.json'
        value = dict(schema='fgm-two-aux-m4-fixed-verification-v1',
                     receipt_sha256=row['receipt_sha256'], artifact_sha256=row['artifact_sha256'],
                     verified=dict(factors_id='factor-f', additions=additions,
                                   additions_by_stage=row['additions_by_stage']),
                     result=native_result)
        row['evidence_path'] = str(evidence.relative_to(directory))
        row['verification_sha256'] = write(evidence, value)
        return row

    def result(self, rows):
        directory = self.root / 'measurement/A/pair'
        attempt_sha = write(directory / 'attempt-manifest.json',
                            dict(frozen_manifest_sha256=self.hash, phase='measurement',
                                 experiment='A', attempt=self.attempt))
        value = dict(schema='fgm-two-aux-m4-attempt-v1', phase='measurement', experiment='A',
                     attempt_id='pair', manifest_sha256=self.hash, status='complete', rows=rows,
                     review_sha256='0'*64, authorization_sha256='1'*64,
                     attempt_manifest_sha256=attempt_sha)
        checksum = write(directory / 'result.json', value)
        (directory / 'result.sha256').write_text(checksum + '\n')

    def test_missing_and_paired_arithmetic(self):
        data = r.report(self.manifest, self.root)
        self.assertEqual(data['attempts'][0]['status'], 'missing')
        self.assertEqual(data['fixed']['A']['E4']['planned'], 1)
        a, b = self.fixed('a', 10), self.fixed('b', 8, 3)
        guard = self.root / 'measurement/A/pair/a/native-guard'
        result_sha = write(guard / 'result.json', {'memory': [{'wired_bytes': 99}],
                                                    'wall_seconds': 2})
        (guard / 'run.log').write_text(' 42 maximum resident set size\n')
        a['native_guard'] = {'result_sha256': result_sha,
                             'log_sha256': r.sha(guard / 'run.log')}
        self.result([a, b])
        data = r.report(self.manifest, self.root)
        cohort = data['fixed']['A']['E4']
        self.assertEqual(cohort['paired'], 1)
        self.assertEqual(cohort['improvements']['total']['samples'], [2])
        self.assertEqual(cohort['paired_latency_delta_seconds']['median'], -1)
        self.assertEqual(cohort['latency_seconds']['baseline-omitted']['median'], 2)
        self.assertEqual(cohort['latency_seconds']['candidate-omitted']['median'], 3)
        item = data['attempts'][0]['rows'][0]
        self.assertEqual(item['receipt_accounted_memory']['planned_buffer_bytes'], 191876132)
        self.assertEqual(item['receipt_accounted_memory']['reserved_host_bytes'], 134217728)
        self.assertEqual(item['timing']['receipt_microseconds']['dispatch_microseconds'], 12)
        self.assertEqual(item['timing']['result_phase_microseconds']['construction_u'], 7)
        self.assertEqual(item['timing']['gpu_seconds'], 0.004)
        observed = data['memory']['observed'][0]
        self.assertEqual((observed['phase'], observed['experiment']), ('measurement', 'A'))

    def test_native_feedback_marks_late_verified_evaluations(self):
        rows = [dict(status='verified', durable_seconds=20,
                     chunks=[dict(index=0, new_evaluations=[{'scheme_id': 'seed'}],
                                  new_installations=[{'installed': True}])]),
                dict(status='verified', durable_seconds=65,
                     chunks=[dict(index=1, new_evaluations=[{'scheme_id': 'descendant'}],
                                  new_installations=[{'installed': False}])])]
        value = r.native_feedback(rows, ['seed'], 60)
        self.assertEqual(value['scope'], 'all verified units including late work')
        self.assertEqual(value['late_evaluations'], 1)
        self.assertEqual(value['descendant_candidates'], 1)
        self.assertEqual(value['feedback_installations'], 1)

    def test_planned_pairs_count_all_three_seeds(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['experiments']['A']['attempts'][0]['cells'] = [
            self.unit(f'{seed}-{treatment}', treatment)
            for seed in (7, 19, 41)
            for treatment in ('baseline-omitted', 'candidate-omitted')]
        self.hash = write(self.manifest, manifest)
        r.MANIFEST_SHA = self.hash
        data = r.report(self.manifest, self.root)
        self.assertEqual(data['fixed']['A']['E4']['planned'], 3)

    def test_changed_evidence_cannot_form_pair(self):
        rows = [self.fixed('a', 10), self.fixed('b', 8)]
        self.result(rows)
        path = self.root / 'measurement/A/pair/b/artifact.jsonl'
        path.write_text('changed')
        data = r.report(self.manifest, self.root)
        self.assertEqual(data['attempts'][0]['rows'][1]['status'], 'evidence-invalid')
        self.assertEqual(data['fixed']['A']['E4']['paired'], 0)

    def test_changed_result_sidecar_preserves_planned_denominator(self):
        self.result([self.fixed('a', 10), self.fixed('b', 8)])
        (self.root / 'measurement/A/pair/result.json').write_text('{}')
        data = r.report(self.manifest, self.root)
        self.assertEqual(data['attempts'][0]['status'], 'result-invalid')
        self.assertEqual(data['fixed']['A']['E4']['planned'], 1)

    def test_same_cost_different_circuit_is_semantic_finding(self):
        a, b = self.fixed('a', 10), self.fixed('b', 10)
        (self.root / 'measurement/A/pair/b/artifact.jsonl').write_text('{"different":true}\n')
        b['artifact_sha256'] = r.sha(self.root / 'measurement/A/pair/b/artifact.jsonl')
        verification = self.root / 'measurement/A/pair/b/verification.json'
        doc = json.loads(verification.read_text())
        doc['artifact_sha256'] = b['artifact_sha256']
        b['verification_sha256'] = write(verification, doc)
        self.result([a, b])
        data = r.report(self.manifest, self.root)
        self.assertEqual(data['fixed']['A']['E4']['disabled_semantic_mismatches'], 1)

    def test_native_export_guard_names_and_full_stage_report(self):
        directory = self.root / 'native'
        export = directory / 'native-context/h/chunks/000/evaluations.jsonl'
        stage = {'baseline_cost': 12, 'improvement_floor': 10, 'raw_scanned': 4,
                 'microseconds': {'gpu': 3}, 'gpu_sweeps': 2}
        export.parent.mkdir(parents=True)
        previous = {'schema': 'fgm-journal-evaluation-v1',
                    'evaluation': {'scheme_id': 'previous',
                                   'two_auxiliary': {'stages': {'u': {'raw_scanned': 999}}}}}
        current = {'schema': 'fgm-journal-evaluation-v1',
                   'evaluation': {'scheme_id': 'new',
                                  'two_auxiliary': {'stages': {'u': stage}}}}
        export.write_text(json.dumps(previous) + '\n' + json.dumps(current) + '\n')
        guard = {}
        for name in ('evaluation_export', 'observation_export'):
            sub = export.parent / (name.replace('_', '-') + '-guard')
            result_sha = write(sub / 'result.json', {'memory': [{'wired_bytes': 99}], 'wall_seconds': 2})
            (sub / 'run.log').write_text(' 42 maximum resident set size\n')
            guard[name + '_guard'] = {'result_sha256': result_sha,
                                      'log_sha256': r.sha(sub / 'run.log')}
        row = {'id': 'child', 'evidence_path': 'native-context/h/chunks/000/verification.json',
               'chunks': [dict(evaluation_export=str(export.relative_to(directory)),
                               evaluation_export_sha256=r.sha(export),
                               new_evaluations=[{'scheme_id': 'new'}], **guard)]}
        stages, memory = r.details(directory, row, {'schema': 'fgm3-chunk-verification-v1'})
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]['stages']['u']['microseconds']['gpu'], 3)
        self.assertEqual({m['scope'] for m in memory},
                         {'evaluation-export-guard', 'observation-export-guard'})

    def test_native_new_evaluations_must_match_verification(self):
        directory = self.root / 'native'
        chunk = {'new_evaluations': [{'scheme_id': 'changed', 'verified_seconds': 5}]}
        for stem in ('receipt', 'evaluation_export', 'observation_export'):
            path = directory / (stem + '.json')
            chunk[stem + '_sha256'] = write(path, {})
            chunk[stem + '_path' if stem == 'receipt' else stem] = path.name
        doc = dict(schema='fgm3-chunk-verification-v1', best_additions=54,
                   new_evaluations=[{'scheme_id': 'original'}],
                   **{stem + '_sha256': chunk[stem + '_sha256'] for stem in
                      ('receipt', 'evaluation_export', 'observation_export')})
        row = dict(status='verified', best_additions=54, chunks=[chunk],
                   evidence_path='verification.json')
        row['verification_sha256'] = write(directory / row['evidence_path'], doc)
        with self.assertRaisesRegex(ValueError, 'new evaluations differ'):
            r.evidence(directory, row)

    def test_native_receipt_accounting_and_timing_scope(self):
        directory = self.root / 'native'
        receipt = directory / 'native-context/h/chunks/000/receipt.json'
        checksum = write(receipt, {'planned_buffer_bytes': 9, 'reserved_host_bytes': 8,
                                   'admission_content_bytes': 7,
                                   'memory_accounting': 'planned allocations',
                                   'dispatch_microseconds': 31,
                                   'best_evaluation': {'phase_microseconds': {'construction_u': 4}}})
        row = dict(elapsed_seconds=6, headroom_wait_seconds=1,
                   chunks=[dict(receipt_path=str(receipt.relative_to(directory)),
                                receipt_sha256=checksum, host_phases_microseconds={'setup': 3},
                                new_evaluations=[{'phase_microseconds': {'baseline': 2}}],
                                gpu={'gpu_all_seconds': .03, 'dispatches': [1, 2]})])
        accounted, timing = r.receipt_metrics(directory, row,
                                             {'schema': 'fgm3-chunk-verification-v1'})
        self.assertEqual(accounted['admission_content_bytes'], 7)
        self.assertEqual(timing['receipt_microseconds']['dispatch_microseconds'], 31)
        self.assertEqual(timing['native_host_phases_microseconds']['setup'], 3)
        self.assertEqual(timing['new_evaluation_phase_microseconds'][0]['baseline'], 2)
        self.assertEqual(timing['gpu_dispatch_count'], 2)

    def test_late_verified_child_is_retained_without_earlier_credit(self):
        manifest = json.loads(self.manifest.read_text())
        arm = dict(id='arm', factor_ids=['f'], treatment={'id': 'candidate-enabled'},
                   endpoints=[10, 20])
        attempt = dict(id='search', arms=[arm])
        manifest['experiments']['C-S']['attempts'] = [attempt]
        self.hash = write(self.manifest, manifest)
        r.MANIFEST_SHA = self.hash
        child = self.fixed('a', 53)
        child['durable_seconds'] = 15
        child['headroom_wait_seconds'] = 1.5
        shutil.copytree(self.root / 'measurement/A/pair/a',
                        self.root / 'measurement/C-S/search/a')
        directory = self.root / 'measurement/C-S/search'
        result = dict(schema='fgm-two-aux-m4-attempt-v1', phase='measurement',
                      experiment='C-S', attempt_id='search', manifest_sha256=self.hash,
                      review_sha256='0'*64, authorization_sha256='1'*64,
                      status='complete', rows=[dict(id='arm', status='complete', started_seconds=0,
                                                    headroom_wait_seconds=99,
                                                    blocks=[child])])
        result['attempt_manifest_sha256'] = write(directory / 'attempt-manifest.json',
            dict(frozen_manifest_sha256=self.hash, phase='measurement',
                 experiment='C-S', attempt=attempt))
        (directory / 'result.sha256').write_text(write(directory / 'result.json', result) + '\n')
        data = r.report(self.manifest, self.root)
        points = data['search']['C-S']['arms'][0]['endpoints']
        self.assertEqual(points['10']['verified_evaluations'], 0)
        self.assertEqual(points['20']['verified_evaluations'], 1)
        self.assertEqual(data['attempts'][-1]['rows'][0]['verified_children'][0]['late_for_endpoints'], [10])
        self.assertEqual(data['attempts'][-1]['headroom_wait_seconds'], 1.5)


if __name__ == '__main__':
    unittest.main()
