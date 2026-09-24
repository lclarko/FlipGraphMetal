"""Completed-workload execution and profile comparison without GPU dispatch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_measurement_adapters import b, circuit, log, scalar


class MeasurementExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.serial = 0

    def prepare(self, name='fixed-reducer', mode='production', exports=None,
                complete=True, include_profile=True, matched_exports=None):
        self.serial += 1
        attempt = self.root / str(self.serial)
        attempt.mkdir()
        row = next(row for row in b.ROWS if row[0] == name)
        config = b.protocol({row[0]: 2 for row in b.ROWS})
        (attempt / 'input.txt').write_bytes(b.adapter_bytes(scalar(), 'reducer'))
        (attempt / 'effective-input.json').write_text(json.dumps(scalar()))
        record = dict(workload=name, mode=mode, complete=False, exports=[])
        receipt = attempt / 'record.json'
        clock = [100.0]
        observed = []
        payloads = [circuit()] if exports is None else exports
        production = None
        if mode == 'profile':
            prior_directory = self.root / f'production-{self.serial}'
            prior_directory.mkdir()
            prior = {'exports': []}
            for i, payload in enumerate(payloads if matched_exports is None else matched_exports):
                path = prior_directory / f'{i}.json'
                path.write_text(json.dumps(payload))
                prior['exports'].append({'file': path.name, 'sha256': b.digest(path)})
            production = (prior, prior_directory)
        output = log(row[4], mutation=name == 'mutation-reducer')
        if mode == 'profile' and include_profile:
            for kernel, _, _ in b.dispatch_evidence(output)['dispatches']:
                output += (f'FGM_PROFILE_V1 kernel={kernel} setup_seconds=0.001 '
                           'commit_wait_seconds=0.002 validation_report_seconds=0.003 '
                           'tracked_shared_live_bytes=32 tracked_shared_peak_bytes=64\n')

        def save():
            receipt.write_text(json.dumps(record))

        def guarded(command, directory):
            observed.append(command)
            self.assertFalse(json.loads(receipt.read_text())['complete'])
            directory.mkdir()
            with (directory / 'run.log').open('x') as stream:
                stream.write(output[:len(output)//2])
                stream.flush()
                stream.write(output[len(output)//2:])
            export_directory = attempt / 'exports'
            export_directory.mkdir()
            for i, payload in enumerate(payloads):
                (export_directory / f'{i}.json').write_text(json.dumps(payload))
            clock[0] += 3
            return dict(complete=complete, memory=[dict(wired_bytes=1234)])

        exact_verify = b.verify

        def verify(*args):
            result = exact_verify(*args)
            clock[0] += 4
            return result

        def execute():
            with mock.patch.object(b, 'guarded_run', side_effect=guarded), \
                    mock.patch.object(b.time, 'monotonic', side_effect=lambda: clock[0]), \
                    mock.patch.object(b, 'verify', side_effect=verify):
                return b.run_attempt(row, config, self.root / 'source', attempt, record, save,
                                     production=production)

        return execute, record, attempt, observed

    def test_same_command_and_verification_timing_in_both_modes(self):
        for mode in ('production', 'profile'):
            with self.subTest(mode=mode):
                execute, record, attempt, observed = self.prepare(mode=mode)
                execute()
                expected = ['/usr/bin/time', '-l', '-p',
                            str(self.root / 'source/build/metal/additions_reducer'),
                            '--seed', '7', '--block-size', '32', '--rounds', '2',
                            '-i', str(attempt / 'input.txt'), '-o', str(attempt / 'exports'),
                            '--count', '32', '--schemes-count', '1', '--max-flips', '0',
                            '--max-no-improvements', '2']
                self.assertEqual(observed, [expected])
                self.assertTrue(record['complete'])
                self.assertEqual(record['process_seconds'], 1.25)
                self.assertEqual(record['verification_seconds'], 4)
                self.assertEqual(record['workflow_seconds'], 7)
                self.assertEqual(record['peak_system_wired_bytes'], 1234)
                self.assertEqual(len(record['exports']), 1)
                self.assertEqual(json.loads((attempt / 'record.json').read_text()),
                                 json.loads(json.dumps(record)))
                if mode == 'profile':
                    self.assertEqual([p['kernel'] for p in record['profile']],
                                     [p[0] for p in record['dispatches']])

    def test_incomplete_attempt_preserves_record_and_partial_evidence(self):
        for mode in ('production', 'profile'):
            with self.subTest(mode=mode):
                execute, record, attempt, _ = self.prepare(mode=mode, complete=False)
                with self.assertRaises(RuntimeError):
                    execute()
                retained = json.loads((attempt / 'record.json').read_text())
                self.assertFalse(retained['complete'])
                self.assertFalse(retained['guard_complete'])
                self.assertIn('error', retained)
                self.assertTrue((attempt / 'guard/run.log').read_text())
                self.assertTrue((attempt / 'exports/0.json').is_file())
                self.assertEqual(record['exports'], [])

    def test_reducer_requires_verified_output_in_both_modes(self):
        for mode in ('production', 'profile'):
            with self.subTest(mode=mode):
                execute, record, attempt, _ = self.prepare(mode=mode, exports=[])
                with self.assertRaises(ValueError):
                    execute()
                self.assertFalse(record['complete'])
                self.assertIn('error', json.loads((attempt / 'record.json').read_text()))

    def test_fixed_mode_rejects_changed_factors_of_valid_tensor(self):
        changed = circuit()
        changed['u'][0][0]['value'] = changed['v'][0][0]['value'] = -1
        b.verify(changed)
        for mode in ('production', 'profile'):
            with self.subTest(mode=mode):
                execute, record, _, _ = self.prepare(mode=mode, exports=[changed])
                with self.assertRaisesRegex(ValueError, 'differ from reference'):
                    execute()
                self.assertFalse(record['complete'])
                self.assertEqual(record['exports'], [])

    def test_mutation_dispatch_is_not_applied_mutation_evidence(self):
        changed = circuit()
        changed['u'][0][0]['value'] = changed['v'][0][0]['value'] = -1
        for mode in ('production', 'profile'):
            for payload, expected in ((circuit(), 'NOT VERIFIED'),
                                      (changed, 'changed verified circuit factors')):
                with self.subTest(mode=mode, expected=expected):
                    execute, record, _, _ = self.prepare('mutation-reducer', mode, [payload])
                    execute()
                    self.assertTrue(record['complete'])
                    self.assertEqual(record['applied_mutation_evidence'], expected)

    def test_profile_requires_observations_in_completed_attempt(self):
        execute, record, attempt, _ = self.prepare(mode='profile', include_profile=False)
        with self.assertRaises(ValueError):
            execute()
        self.assertFalse(record['complete'])
        self.assertIn('error', json.loads((attempt / 'record.json').read_text()))

    def test_profile_comparison_checks_export_bytes_and_retained_production(self):
        execute, original, source_attempt, _ = self.prepare()
        execute()
        execute, diagnostic, _, _ = self.prepare(mode='profile')
        execute()
        b.compare_profile(diagnostic, original, source_attempt)
        changed = copy.deepcopy(diagnostic)
        changed['exports'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            b.compare_profile(changed, original, source_attempt)
        changed_circuit = circuit()
        changed_circuit['u'][0][0]['value'] = changed_circuit['v'][0][0]['value'] = -1
        execute, record, attempt, _ = self.prepare('mutation-reducer', 'profile',
                                                   [changed_circuit], matched_exports=[circuit()])
        with self.assertRaises(ValueError):
            execute()
        self.assertFalse(record['complete'])
        self.assertFalse(json.loads((attempt / 'record.json').read_text())['complete'])
        path = source_attempt / original['exports'][0]['file']
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            b.compare_profile(diagnostic, original, source_attempt)


if __name__ == '__main__':
    unittest.main()
