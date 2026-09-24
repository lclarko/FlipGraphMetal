"""Host-only checks for retained production selection and diagnostic observations."""
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/workflow'))
SPEC = importlib.util.spec_from_file_location(
    'qualify_profile', ROOT / 'benchmarks/workflow/baseline.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def observation(kernel='walk', **changes):
    fields = dict(kernel=kernel, setup_seconds='0.001', commit_wait_seconds='0.1',
                  validation_report_seconds='0.002', tracked_shared_live_bytes='4096',
                  tracked_shared_peak_bytes='8192')
    fields.update(changes)
    return 'FGM_PROFILE_V1 ' + ' '.join(f'{key}={value}' for key, value in fields.items())


class ProfileObservations(unittest.TestCase):
    def test_phase_values_and_dispatch_order(self):
        log = '\n'.join(['ordinary log', observation(), observation('reduce')])
        rows = p.parse_profile_observations(log, ['walk', 'reduce'])
        self.assertEqual([row['kernel'] for row in rows], ['walk', 'reduce'])
        self.assertEqual(rows[0]['setup_seconds'], 0.001)
        self.assertEqual(rows[0]['tracked_shared_live_bytes'], 4096)
        for expected in (['reduce', 'walk'], ['walk'], ['walk', 'reduce', 'walk'], []):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                p.parse_profile_observations(log, expected)
        with self.assertRaises(ValueError):
            p.parse_profile_observations('no observations', ['walk'])

    def test_phases_require_positive_finite_values(self):
        for field in ('setup_seconds', 'commit_wait_seconds', 'validation_report_seconds'):
            for value in ('nan', 'inf', '-inf', '0', '-0.01', 'bad'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    p.parse_profile_observations(observation(**{field: value}), ['walk'])

    def test_memory_requires_positive_integer_lengths_and_valid_peak(self):
        for field in ('tracked_shared_live_bytes', 'tracked_shared_peak_bytes'):
            for value in ('0', '-1', '1.5', 'nan'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    p.parse_profile_observations(observation(**{field: value}), ['walk'])
        with self.assertRaises(ValueError):
            p.parse_profile_observations(observation(tracked_shared_peak_bytes='2048'), ['walk'])

    def test_schema_rejects_missing_extra_and_duplicate_fields(self):
        line = observation()
        invalid = [line + ' kernel=walk', line + ' unrecognized=1',
                   line.replace('setup_seconds=0.001 ', ''), line + ' broken',
                   line.replace('kernel=walk', 'kernel='),
                   line.replace('FGM_PROFILE_V1', 'FGM_PROFILE_V1x')]
        for log in invalid:
            with self.subTest(log=log), self.assertRaises(ValueError):
                p.parse_profile_observations(log, ['walk'])

    def test_selects_earliest_complete_repeat_from_explicit_run(self):
        prior = {'attempts': [
            {'workload': 'search', 'repeat': 9, 'complete': True},
            {'workload': 'search', 'repeat': 0, 'complete': False},
            {'workload': 'search', 'repeat': 3, 'complete': True}]}
        row, path = p.production_attempt(prior, Path('/retained/run-name'), 'search')
        self.assertEqual(row['repeat'], 3)
        self.assertEqual(path, Path('/retained/run-name/03-search'))
        with self.assertRaises(ValueError):
            p.production_attempt(prior, Path('/retained/run-name'), 'absent')

    def test_ambiguous_or_invalid_repeat_fails_closed(self):
        for repeat in (-1, True, '3', None):
            prior = {'attempts': [{'workload': 'search', 'repeat': repeat, 'complete': True}]}
            with self.subTest(repeat=repeat), self.assertRaises(ValueError):
                p.production_attempt(prior, Path('/run'), 'search')
        row = {'workload': 'search', 'repeat': 0, 'complete': True}
        with self.assertRaises(ValueError):
            p.production_attempt({'attempts': [row, row]}, Path('/run'), 'search')


class RetainedProfileIdentity(unittest.TestCase):
    def test_diagnostic_source_rejects_links_and_changed_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder/'source'
            source.mkdir()
            original = source/'runtime.mm'
            original.write_bytes(b'runtime')
            declaration = {'diagnostic_files': {'runtime.mm': p.digest(original)}}
            p.assert_diagnostic(source, declaration)
            linked_root = folder/'linked'
            linked_root.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                p.assert_diagnostic(linked_root, declaration)
            copy = folder/'runtime.mm'
            copy.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(copy)
            with self.assertRaises(ValueError):
                p.assert_diagnostic(source, declaration)
            original.unlink()
            original.write_bytes(copy.read_bytes())
            nested = source/'nested'
            nested.symlink_to(folder, target_is_directory=True)
            with self.assertRaises(ValueError):
                p.assert_diagnostic(source, declaration)
            nested.unlink()
            original.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                p.assert_diagnostic(source, declaration)

    def test_profile_archive_mismatch_rejects_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            production, profile = folder/'production', folder/'profile'
            production.mkdir()
            profile.mkdir()
            (production/'qualification.json').write_text(json.dumps({
                'schema': 'fgm-baseline-qualification-v1', 'complete': True,
                'protocol': {}, 'attempts': [],
                'baseline': {'commit': p.BASELINE, 'archive_sha256': 'original'}}))
            for number, archive_hash in enumerate(('different', None)):
                declaration = {'baseline_commit': p.BASELINE}
                if archive_hash is not None:
                    declaration['baseline_archive_sha256'] = archive_hash
                (profile/'profile.json').write_text(json.dumps(declaration))
                with self.subTest(archive_hash=archive_hash), \
                        mock.patch.object(p, 'assert_source_snapshot') as snapshot, \
                        mock.patch.object(p, 'guarded_run') as guard:
                    with self.assertRaises(ValueError):
                        p.execute(folder/'source', folder/f'output-{number}', None, [], 1,
                                  profile=profile, production_run=production)
                    snapshot.assert_not_called()
                    guard.assert_not_called()

    def test_retained_versions_preserve_mode_and_unavailable_timings(self):
        documents = [
            {'schema': 'fgm-baseline-qualification-v1',
             'attempts': [{'workflow_seconds': 3.0}]},
            {'schema': 'fgm-profile-qualification-v1',
             'rows': [{'metrics': {'gpu_work_seconds': 2.0}, 'exports': []}]},
            {'schema': 'fgm-baseline-qualification-v2', 'mode': 'production',
             'attempts': [{'mode': 'production', 'workflow_seconds': 4.0}]},
            {'schema': 'fgm-baseline-qualification-v2', 'mode': 'profile',
             'attempts': [{'mode': 'profile', 'workflow_seconds': 5.0}]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'qualification.json'
            for index, document in enumerate(documents):
                with self.subTest(schema=document['schema'], index=index):
                    raw = json.dumps(document).encode()
                    path.write_bytes(raw)
                    result = p.read_qualification(path)
                    self.assertEqual(path.read_bytes(), raw)
                    expected_mode = 'profile' if index in (1, 3) else 'production'
                    self.assertEqual(result['mode'], expected_mode)
                    self.assertEqual(result['attempts'][0]['mode'], expected_mode)
                    if index == 1:
                        self.assertEqual(result['attempts'][0]['gpu_work_seconds'], 2.0)
                        self.assertIsNone(result['attempts'][0]['workflow_seconds'])
                        self.assertIsNone(result['attempts'][0]['verification_seconds'])
                    else:
                        self.assertEqual(result['attempts'][0]['workflow_seconds'],
                                         document['attempts'][0]['workflow_seconds'])
                    if expected_mode == 'profile':
                        with self.assertRaises(ValueError):
                            p.summarize(path)
            for document in (
                    {'schema': 'unknown', 'attempts': []},
                    {'schema': 'fgm-baseline-qualification-v2', 'mode': 'production',
                     'attempts': [{'mode': 'profile'}]}):
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    p.read_qualification(path)


if __name__ == '__main__':
    unittest.main()
