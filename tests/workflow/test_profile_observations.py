"""Host-only checks for retained production selection and diagnostic observations."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/workflow'))
SPEC = importlib.util.spec_from_file_location(
    'qualify_profile', ROOT / 'benchmarks/workflow/qualify_profile.py')
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


if __name__ == '__main__':
    unittest.main()
