"""Host-only clock and supervision boundaries for M4 attempts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/workflow'))
import two_aux_execute as x


class Clock:
    def __init__(self):
        self.value = 0.
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


class IsolatedBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet = self.root / 'packet'
        self.packet.mkdir()
        (self.packet / 'manifest.json').write_text('{}\n')
        self.review = self.root / 'review.json'
        self.review.write_text('{}\n')
        self.auth = self.root / 'authorization.json'
        self.auth.write_text('{}\n')
        self.clock = Clock()

    def attempt(self, sample=None, guard=None):
        return x.Attempt(self.root / 'attempt',
                         dict(resources=dict(outer_seconds=900), builds={}, panel=dict(factors=[])),
                         x.digest(self.packet / 'manifest.json'), x.digest(self.review),
                         'qualification', 'fixed', dict(id='one'),
                         [dict(id='first', status='unrun'), dict(id='second', status='unrun')],
                         {}, clock=self.clock.now, sleep=self.clock.sleep,
                         sample=sample or (lambda **kwargs: 0),
                         guard=guard or (lambda *args, **kwargs: self.fail('child launched')),
                         started=0, packet=self.packet, authorization=self.auth,
                         review=self.review)

    def test_exact_allowance_boundaries(self):
        self.assertEqual(x.b.fgm3_chunk_allowance([14.4]), 20)
        self.assertEqual(x.b.fgm3_chunk_allowance([14.41]), 21)
        self.assertEqual(x.b.fgm3_chunk_allowance([46.4]), 60)
        self.assertEqual(x.b.fgm3_chunk_allowance([46.41]), 61)
        self.assertEqual(x.b.fgm3_qualification_work_seconds(
            dict(elapsed_seconds=20, headroom_wait_seconds=5.6)), 14.4)

    def test_prelaunch_sample_or_save_can_close_arm_without_guard(self):
        def sample(*, timeout):
            self.clock.sleep(2)
            return 0
        run = self.attempt(sample=sample)
        with self.assertRaises(x.ArmClosed):
            run.headroom(dict(id='first'), 20, arm_deadline=21)
        self.assertEqual(self.clock.now(), 2)
        # The second case crosses the boundary in the durable prelaunch save.
        run.sample = lambda **kwargs: 0
        run.check_launch = lambda: None
        run.launch_builds = {'candidate': {}}
        run.builds = {'candidate': {}}
        original_save = run.save
        def delayed_save():
            original_save()
            self.clock.sleep(2)
        run.save = delayed_save
        with mock.patch.object(x, 'build_inventory', return_value={}):
            with self.assertRaises(x.ArmClosed):
                run.child('native', ['mock'], 'candidate', dict(id='first'),
                          arm_deadline=24, allowance=20)

    def test_failed_guard_keeps_sidecars_and_unrun_tail(self):
        def guard(argv, destination, *, absolute_deadline, wired_limit_bytes):
            destination.mkdir(parents=True)
            x.durable_json(destination / 'result.json', dict(complete=False))
            (destination / 'run.log').write_text('failed\n')
            self.assertEqual(absolute_deadline, 850)
            return dict(complete=False, wired_limit=x.WIRED_LIMIT,
                        forced_termination=True, cleanup_failure=False,
                        wall_seconds=1)
        run = self.attempt(guard=guard)
        run.check_launch = lambda: None
        run.launch_builds = {'candidate': {}}
        run.builds = {'candidate': {}}
        row = run.data['rows'][0]
        with mock.patch.object(x, 'build_inventory', return_value={}):
            with self.assertRaisesRegex(ValueError, 'child failed'):
                run.child('native', ['mock'], 'candidate', row)
        self.assertIsNotNone(row['native_guard']['result_sha256'])
        run.finish(RuntimeError('guard failed'))
        self.assertEqual(run.data['status'], 'failed')
        self.assertEqual(run.data['rows'][1]['status'], 'unrun')

    def test_late_verified_work_has_no_earlier_endpoint_credit(self):
        fixed = [dict(status='verified', durable_seconds=10.01, additions=56),
                 dict(status='verified', durable_seconds=20.01, additions=54)]
        self.assertEqual(x.endpoint(fixed, 0, 10)['verified_evaluations'], 0)
        self.assertEqual(x.endpoint(fixed, 0, 20)['verified_evaluations'], 1)
        self.assertFalse(x.endpoint(fixed, 0, 20)['target_54_attained'])

    def test_qualification_seal_and_evidence_corruption_block_reuse(self):
        unit = dict(id='cell', treatment=dict(id='candidate-enabled'))
        planned = dict(id='attempt', factor_id='fixture', cells=[unit])
        manifest = dict(qualification=dict(fixed_factor_schedule=[planned],
                                           native_schedule=[]))
        directory = self.root / 'qualifications' / 'fixed' / 'attempt'
        directory.mkdir(parents=True)
        evidence = directory / 'verification.json'
        receipt = directory / 'receipt.json'
        artifact = directory / 'artifact.jsonl'
        for path in (evidence, receipt, artifact):
            path.write_text('{}\n')
        x.durable_json(directory / 'attempt-manifest.json', dict(attempt=planned))
        row = dict(id='cell', status='verified', elapsed_seconds=10,
                   headroom_wait_seconds=1,
                   evidence_path='verification.json', verification_sha256=x.digest(evidence),
                   receipt_path='receipt.json', receipt_sha256=x.digest(receipt),
                   artifact_path='artifact.jsonl', artifact_sha256=x.digest(artifact))
        for name in ('native', 'verifier'):
            destination = directory / 'cell' / (name + '-guard')
            destination.mkdir(parents=True)
            x.durable_json(destination / 'result.json', dict(
                complete=True, forced_termination=False, cleanup_failure=False,
                time_limit=x.CHILD_SECONDS, wired_limit=x.WIRED_LIMIT))
            (destination / 'run.log').write_text('verified\n')
            row[name + '_guard'] = dict(result_sha256=x.digest(destination / 'result.json'),
                                        log_sha256=x.digest(destination / 'run.log'))
        result = dict(schema='fgm-two-aux-m4-attempt-v1', phase='qualification',
                      experiment='fixed', attempt_id='attempt', manifest_sha256='manifest',
                      review_sha256='review', authorization_sha256='authorization',
                      status='complete', rows=[row],
                      attempt_manifest_sha256=x.digest(directory / 'attempt-manifest.json'))
        x.durable_json(directory / 'result.json', result)
        (directory / 'result.sha256').write_text(x.digest(directory / 'result.json') + '\n')
        allowance = x.qualification_allowances(self.root / 'qualifications', manifest,
                                                'manifest', 'review', 'authorization')
        self.assertEqual(allowance[('fixed', ('fixture', 'candidate-enabled'))], 14)
        guard_log = directory / 'cell/verifier-guard/run.log'
        guard_log.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            x.qualification_allowances(self.root / 'qualifications', manifest,
                                       'manifest', 'review', 'authorization')
        guard_log.write_text('verified\n')
        evidence.write_text('{"changed": true}\n')
        with self.assertRaisesRegex(ValueError, 'evidence changed|verification evidence changed'):
            x.qualification_allowances(self.root / 'qualifications', manifest,
                                       'manifest', 'review', 'authorization')

    def test_native_host_verifier_reuses_bound_checker(self):
        files = {}
        for name in ('receipt', 'config', 'evaluations', 'observations', 'population'):
            files[name] = self.root / (name + '.json')
            files[name].write_text('{}\n')
        request = self.root / 'request.json'
        response = self.root / 'response.json'
        payload = dict(schema='fgm-two-aux-m4-native-verifier-input-v1',
                       output=str(self.root), population=str(self.root), protocol={},
                       selector='cost-diverse', chunk=0, state={}, record={}, started=0,
                       expected_population_count=3,
                       paths={name: str(path) for name, path in files.items()},
                       hashes={name: x.digest(path) for name, path in files.items()})
        x.durable_json(request, payload)
        def binder(*args):
            args[9]['status'] = 'verified'
            args[5]['evaluations'] = []
        with mock.patch.object(x.b, 'fgm3_verify_chunk', side_effect=binder) as called:
            x.verify_native(request, response)
        self.assertEqual(called.call_count, 1)
        self.assertEqual(json.loads(response.read_text())['input_sha256'], x.digest(request))
        response.unlink()
        files['receipt'].write_text('{"tampered": true}\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            x.verify_native(request, response)

    def test_native_hashing_can_close_arm_before_any_guard(self):
        # Run the real chunk preparation and admission path with fake binaries.
        # This exercises scheduling only; no factor verification is mocked as
        # successful evidence and no child is launched.
        run = self.attempt()
        binaries = self.root / 'bin'
        binaries.mkdir()
        (binaries / 'flip_graph').write_text('not executable\n')
        inventory = {'flip_graph': x.digest(binaries / 'flip_graph')}
        protocol = json.loads(x.b.FGM3_PROTOCOL_PATH.read_text())
        protocol.update(native_admission_seconds=240, wired_limit_bytes=x.WIRED_LIMIT,
                        launch_wired_reserve_bytes=2348810240)
        entries = [dict(path='factor.json', sha256=x.digest(self.auth), canonical_id='fixture')]
        def population(*args):
            directory = run.output / 'populations' / 'first'
            directory.mkdir(parents=True)
            (directory / 'factor.json').write_bytes(self.auth.read_bytes())
            x.durable_json(directory / 'population.json', dict(entries=entries))
            return directory, entries
        run.check_launch = lambda: self.clock.sleep(20)
        unit = dict(trial_seed=7, treatment=dict(build='candidate'), operation='initialize')
        row = run.data['rows'][0]
        with mock.patch.object(x, 'population', side_effect=population), \
                mock.patch.object(x, 'native_protocol', return_value=protocol), \
                mock.patch.object(x, 'build_inventory', return_value=inventory):
            x.native_unit(run, self.packet, {'candidate': {'flip_graph': binaries / 'flip_graph'}},
                          unit, row, ['fixture'], allowance=5, arm_deadline=10)
        self.assertEqual(self.clock.now(), 20)
        self.assertEqual(row['status'], 'unrun')
        self.assertEqual(row['stop_reason'], 'arm admission allowance')
        self.assertNotIn('native_guard', row['chunks'][0])

    def test_native_phase_budget_does_not_replace_outer_cleanup_reserve(self):
        launched = []
        def guard(argv, destination, *, absolute_deadline, wired_limit_bytes):
            launched.append((self.clock.now(), absolute_deadline))
            destination.mkdir(parents=True)
            result = dict(complete=False, wired_limit=wired_limit_bytes,
                          forced_termination=False, cleanup_failure=False,
                          wall_seconds=0, error='simulated child failure')
            x.durable_json(destination / 'result.json', result)
            (destination / 'run.log').write_text('')
            return result
        run = self.attempt(guard=guard)
        run.work_deadline = 360
        self.clock.sleep(175)  # Five 35-second units leave 185 phase seconds.
        binaries = self.root / 'bin'
        binaries.mkdir()
        (binaries / 'flip_graph').write_text('not executable\n')
        inventory = {'flip_graph': x.digest(binaries / 'flip_graph')}
        protocol = json.loads(x.b.FGM3_PROTOCOL_PATH.read_text())
        protocol.update(native_admission_seconds=240, wired_limit_bytes=x.WIRED_LIMIT,
                        launch_wired_reserve_bytes=2348810240)
        entries = [dict(path='factor.json', sha256=x.digest(self.auth), canonical_id='fixture')]
        def population(*args):
            directory = run.output / 'populations' / 'first'
            directory.mkdir(parents=True)
            (directory / 'factor.json').write_bytes(self.auth.read_bytes())
            x.durable_json(directory / 'population.json', dict(entries=entries))
            return directory, entries
        run.check_launch = lambda: None
        unit = dict(trial_seed=7, treatment=dict(build='candidate'), operation='initialize')
        with mock.patch.object(x, 'population', side_effect=population), \
                mock.patch.object(x, 'native_protocol', return_value=protocol), \
                mock.patch.object(x, 'build_inventory', return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, 'simulated child failure'):
                x.native_unit(run, self.packet,
                              {'candidate': {'flip_graph': binaries / 'flip_graph'}},
                              unit, run.data['rows'][0], ['fixture'], allowance=60)
        self.assertEqual(launched, [(175, 360)])
        # An export may wait briefly inside the phase while its conservative
        # remaining-child reserve is still available in the outer envelope.
        run.sample = mock.Mock(side_effect=[x.PRELAUNCH_CEILING + 1, 0])
        self.clock.value = 230
        run.headroom(dict(id='export'), 0, global_work_seconds=145)
        self.assertEqual(run.sample.call_count, 2)
        self.assertAlmostEqual(self.clock.now(), 230.025)

    def test_slow_native_verifier_failure_keeps_deadline_and_unrun_tail(self):
        def guard(argv, destination, *, absolute_deadline, wired_limit_bytes):
            self.assertIn('--verify-native', argv)
            self.assertEqual(absolute_deadline, 850)
            self.assertEqual(wired_limit_bytes, x.WIRED_LIMIT)
            self.clock.sleep(45.01)
            destination.mkdir(parents=True)
            result = dict(complete=False, wired_limit=wired_limit_bytes,
                          forced_termination=True, cleanup_failure=False,
                          wall_seconds=45.01, error='time limit')
            x.durable_json(destination / 'result.json', result)
            (destination / 'run.log').write_text('')
            return result
        run = self.attempt(guard=guard)
        run.check_launch = lambda: None
        run.launch_builds = run.builds = {'candidate': {}}
        row = run.data['rows'][0]
        with mock.patch.object(x, 'build_inventory', return_value={}):
            with self.assertRaisesRegex(ValueError, 'native-verifier child failed') as failed:
                run.child('native-verifier', ['python', '--verify-native', 'in', 'out'],
                          'candidate', row, work_admission=False)
        run.finish(failed.exception)
        self.assertEqual(run.data['status'], 'failed')
        self.assertTrue(row['native-verifier_guard']['forced_termination'])
        self.assertEqual(run.data['rows'][1]['status'], 'unrun')

    def test_fixed_verifier_can_preserve_its_strict_stdout_contract(self):
        def guard(argv, destination, **kwargs):
            self.assertEqual(argv, ['python', 'verify.py', 'circuit'])
            destination.mkdir(parents=True)
            result = dict(complete=True, wired_limit=x.WIRED_LIMIT,
                          forced_termination=False, cleanup_failure=False, wall_seconds=1)
            x.durable_json(destination / 'result.json', result)
            (destination / 'run.log').write_text('circuit PASS {}\n')
            return result
        run = self.attempt(guard=guard)
        run.check_launch = lambda: None
        run.launch_builds = run.builds = {'candidate': {}}
        with mock.patch.object(x, 'build_inventory', return_value={}):
            log = run.child('verifier', ['python', 'verify.py', 'circuit'], 'candidate',
                            run.data['rows'][0], work_admission=False, timed=False)
        self.assertEqual(log.read_text().splitlines(), ['circuit PASS {}'])

    def test_phase_ceilings_count_waits_and_preserve_final_write_failure(self):
        run = self.attempt()
        run.phase_limits = dict(preparation=75, native=270, independent_verification=270,
                                headroom_wait=100, audit=135, finalization=50)
        run.data['preparation_seconds'] = 1
        run.data['rows'][0]['headroom_wait_seconds'] = 100.1
        self.clock.sleep(101.1)
        with self.assertRaisesRegex(ValueError, 'headroom_wait phase ceiling'):
            x.check_phase_ceilings(run)
        run.headroom_seconds = 100
        with self.assertRaisesRegex(ValueError, 'headroom phase ceiling'):
            run.headroom(dict(id='first'), 0)
        # The successful row set is not enough if sealing crosses the deadline.
        for row in run.data['rows']:
            row['status'] = 'verified'
        original = x.durable_text
        seals = []
        def slow_second_seal(path, value):
            original(path, value)
            seals.append(path)
            if len(seals) == 2:
                self.clock.sleep(901)
        with mock.patch.object(x, 'durable_text', side_effect=slow_second_seal):
            run.finish()
        saved = json.loads((run.output / 'result.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual((run.output / 'result.sha256').read_text(),
                         x.digest(run.output / 'result.json') + '\n')


if __name__ == '__main__':
    unittest.main()
