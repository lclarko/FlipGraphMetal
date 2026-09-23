from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/metal'))
import application
from application import case_matrix
from summarize_application import combine_directories, load_directory


class FixtureBlocks(unittest.TestCase):
    def matrix(self, directory, fixtures, incomplete=None, runner='first'):
        cases = case_matrix(['cpu', 'baseline', 'candidate'], fixtures, [2048], [7, 19, 41, 73, 101], 6, True)
        config = {'version': 1, 'cases': cases, 'identities': {'same': 'build'}, 'rounds': 33,
                  'counterbalance': {'enabled': True, 'seed_order': [7, 19, 41, 73, 101], 'panels_per_seed': 3},
                  'fixture_sha256': 'fixture', 'platform': 'macOS', 'developer_dir': '/Xcode',
                  'iterations_per_round': 1000, 'time_limit': 45, 'wired_limit': 3 * 1024**3,
                  'runner_sha256': runner}
        accepted, missing = {}, []
        for case in cases:
            key = (case['fixture'], case['count'], case['seed'], case['repeat'], case['backend'])
            if case['fixture'] == incomplete and case['repeat'] == 5:
                missing.append({'name': case['name'], 'reason': 'cleanup failed', 'started': True})
            else:
                accepted[key] = {**case, 'complete': True, 'steady_steps_per_second': 1.2 if case['backend'] == 'candidate' else 1}
        return Path(directory), config, accepted, missing

    def combine(self, matrices, reason=None):
        with patch('summarize_application.digest', return_value='config-hash'):
            return combine_directories(matrices, reason)

    def test_complete_fixture_blocks_preserve_failed_input(self):
        first = self.matrix('/first', ['naive', 'rank26'], incomplete='rank26')
        second = self.matrix('/second', ['rank26'], runner='second')
        config, records, missing, provenance = self.combine([first, second], 'cleanup after timed work changed')
        self.assertEqual(len(records), 180)
        self.assertFalse(missing)
        self.assertEqual(config['cases'], first[1]['cases'][:90] + second[1]['cases'])
        self.assertFalse(provenance['input_matrices'][0]['original_matrix_complete'])
        self.assertEqual(provenance['input_matrices'][0]['excluded_cases'], first[3])
        self.assertEqual([row['directory'] for row in provenance['selected_blocks']], ['/first', '/second'])
        for key in second[2]:
            self.assertIs(records[key], second[2][key])

    def test_duplicate_complete_fixture_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.combine([self.matrix('/first', ['naive']), self.matrix('/second', ['naive'])])

    def test_no_partial_panel_or_case_splicing(self):
        first = self.matrix('/first', ['rank26'], incomplete='rank26')
        second = self.matrix('/second', ['rank26'], incomplete='rank26')
        config, records, missing, provenance = self.combine([first, second])
        self.assertFalse(records)
        self.assertFalse(config['cases'])
        self.assertEqual(len(missing), 1)
        self.assertFalse(provenance['selected_blocks'])

    def test_incompatible_source_machine_fixture_or_design_rejected(self):
        for field in ('identities', 'rounds', 'counterbalance', 'fixture_sha256', 'platform', 'developer_dir', 'iterations_per_round'):
            first = self.matrix('/first', ['naive'])
            second = self.matrix('/second', ['rank26'])
            second[1][field] = 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'incompatible'):
                self.combine([first, second])

    def test_kernel_selector_is_part_of_combined_protocol(self):
        for left, right in [(None, 'randomWalkKernel'),
                            ('randomWalkKernel', None),
                            ('randomWalkKernel', 'randomWalkCompactKernel')]:
            first = self.matrix('/first', ['naive'])
            second = self.matrix('/second', ['rank26'])
            if left:
                first[1]['expected_kernel'] = left
            if right:
                second[1]['expected_kernel'] = right
            with self.subTest(left=left, right=right), self.assertRaisesRegex(ValueError, 'incompatible'):
                self.combine([first, second])

    def test_summary_rechecks_declared_kernel_device_and_logs(self):
        case = dict(name='gpu-case', fixture='custom', count=2048, seed=7, repeat=0, backend='candidate')
        identity = dict(binary='/unused/program', source='/unused/source', build_manifest='/unused/build.json')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / case['name']
            directory.mkdir()
            stdout = 'Metal device: Apple M1\n'
            stderr = 'Metal dispatch randomWalkKernel: 2048 threads, 12.5 ms GPU\n' * 2
            (directory / 'stdout.log').write_text(stdout)
            (directory / 'stderr.log').write_text(stderr)
            config = dict(rounds=2, cases=[case], identities={'candidate': identity},
                          expected_kernel='randomWalkKernel')
            application.write_json(root / 'config.json', config)
            record = dict(case, complete=True, wall_seconds=1, reports=[.4, .6],
                          steady_steps_per_second=100, expected_kernel='randomWalkKernel')
            record.update(application.gpu_evidence(stdout, stderr, 2, 'randomWalkKernel'))

            def seal():
                record['artifacts'] = application.artifacts(directory)
                application.write_json(directory / 'result.json', record)
                (directory / 'result.sha256').write_text(application.digest(directory / 'result.json') + '\n')

            seal()
            with patch('summarize_application.build_identity', return_value=identity):
                self.assertEqual(len(load_directory(root)[1]), 1)
                record['expected_kernel'] = 'randomWalkCompactKernel'
                seal()
                with self.assertRaisesRegex(ValueError, 'expected kernel'):
                    load_directory(root)
                record['expected_kernel'] = 'randomWalkKernel'
                record['gpu_seconds'] = [.02, .02]
                seal()
                with self.assertRaisesRegex(ValueError, 'retained logs'):
                    load_directory(root)
                record['gpu_seconds'] = [.0125, .0125]
                (directory / 'stderr.log').write_text(stderr.replace('randomWalkKernel', 'randomWalkCompactKernel'))
                seal()
                with self.assertRaisesRegex(ValueError, 'search kernel'):
                    load_directory(root)

    def test_runner_change_requires_explicit_reason(self):
        with self.assertRaisesRegex(ValueError, 'runner-change-reason'):
            self.combine([self.matrix('/first', ['naive']), self.matrix('/second', ['rank26'], runner='second')])


if __name__ == '__main__':
    unittest.main()
