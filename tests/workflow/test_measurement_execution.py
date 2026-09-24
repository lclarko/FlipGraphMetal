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


class NativeHostMeasurementTests(unittest.TestCase):
    def test_host_repetition_limits_precede_execution(self):
        for repeats in (0,13,True):
            with self.subTest(repeats=repeats),mock.patch.object(b,'guarded_run') as guard:
                with self.assertRaises(ValueError):
                    b.execute_host(Path('/unused/native'),Path('/unused/output'),repeats)
                guard.assert_not_called()

    def test_host_measurement_freezes_work_and_retains_identity_failures(self):
        for failure in (None,'initial-source','fixture-change','final-source'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);binary=root/'native';binary.write_bytes(b'not executable')
                build=binary.with_name(binary.name+'.build.json');build.write_text('{}')
                candidate={'candidate_source':'source','candidate_build':b.content_hash(
                    {'binary_sha256':b.digest(binary),'receipt_sha256':b.digest(build)})}
                machine={'hardware':{},'os_version':'synthetic'}
                source=root/'fixed.txt';source.write_bytes(b'fixture')
                fixture_hash=b.content_hash({source.name:b.digest(source)})
                work={name:{'fixture_sha256':fixture_hash,'input_files':[source]} for name in b.HOST_WORKLOADS}
                calls=[0]
                def identity(_):
                    calls[0]+=1
                    changed=(failure=='initial-source' and calls[0]==2 or
                             failure=='final-source' and calls[0]==32)
                    return dict(candidate,candidate_source='changed') if changed else candidate
                def attempt(binary,name,work,directory,record,save):
                    protocol=json.loads((root/'out/protocol.json').read_text())
                    self.assertEqual(protocol['workloads'],list(b.HOST_WORKLOADS))
                    self.assertEqual(protocol['repetitions'],6)
                    retained=json.loads((root/'out/measurement.json').read_text())
                    self.assertFalse(retained['complete'])
                    self.assertFalse(retained['attempts'][-1]['complete'])
                    directory.mkdir()
                    record.update(complete=True,workflow_seconds=1.,peak_process_rss_bytes=1024,
                                  resource_violation=False,error=None)
                    if failure=='fixture-change':source.write_bytes(b'changed')
                    save()
                with mock.patch.object(b,'host_candidate_identity',side_effect=identity), \
                        mock.patch.object(b,'host_machine_identity',return_value=machine), \
                        mock.patch.object(b,'prepare_host_inputs',return_value=work), \
                        mock.patch.object(b,'run_host_attempt',side_effect=attempt) as run:
                    if failure:
                        with self.assertRaises(ValueError):b.execute_host(binary,root/'out')
                    else:
                        b.execute_host(binary,root/'out')
                retained=json.loads((root/'out/measurement.json').read_text())
                self.assertEqual(retained['complete'],failure is None)
                self.assertEqual(run.call_count,{'initial-source':0,'fixture-change':1}.get(failure,30))
                self.assertEqual((root/'out/native-build.json').read_bytes(),build.read_bytes())
                self.assertEqual(retained['protocol_sha256'],b.content_hash(retained['protocol']))
                self.assertEqual((root/'out/measurement.sha256').read_text().strip(),b.digest(root/'out/measurement.json'))
                for row in retained['attempts']:
                    path=root/'out'/f"{row['repeat']:02d}-{row['workload']}"/'trial.json'
                    self.assertEqual(row['evidence_sha256'],b.digest(path))
                if failure:self.assertTrue(retained['error'])
                if failure=='final-source':
                    self.assertTrue(all(row['complete'] for row in retained['attempts']))

    def test_integer_verification_does_not_accept_only_mod_two_identity(self):
        oracle=b.host_oracle()
        scheme=oracle.schoolbook((1,1,1),'F2')
        scheme.update(rank=3,u=[[1],[1],[1]],v=[[1],[1],[1]],w=[[1],[1],[1]])
        def report(data):
            return dict(data,scheme_id=oracle.identity(data),factors_id=oracle.identity(data,False))
        self.assertEqual(b.verify_host_record(report(scheme),scheme)['domain'],'F2')
        integer=dict(scheme,domain='ZT')
        with self.assertRaises(ValueError):b.verify_host_record(report(integer),integer)

    def test_report_and_roundtrip_include_independent_checks(self):
        oracle=b.host_oracle()
        scheme=oracle.schoolbook((1,1,1))
        record_output=dict(scheme,scheme_id=oracle.identity(scheme),factors_id=oracle.identity(scheme,False))
        for name,count in (('report-100',1),('cpu-roundtrip-5',10)):
            with self.subTest(workload=name),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);source=root/'source.json';source.write_text(json.dumps(scheme))
                work={'input':source,'inputs':[source]*5,'expected':[scheme]*(5 if count==10 else 1)}
                record={'error':None};clock=[0.0];commands=[]
                def guard(argv,directory):
                    commands.append(argv);directory.mkdir()
                    (directory/'run.log').write_text('real 0.25\n1024 maximum resident set size\n')
                    output=Path(argv[argv.index('--output')+1])
                    if 'export' in argv:output.write_text('1 1 1 1 1 1 1')
                    else:output.write_text(json.dumps(record_output)+'\n')
                    clock[0]+=1
                    return {'complete':True,'memory':[{'wired_bytes':4096}]}
                verify=b.verify_host_record
                def checked(*args):
                    result=verify(*args);clock[0]+=2;return result
                with mock.patch.object(b,'guarded_run',side_effect=guard), \
                        mock.patch.object(b.time,'monotonic',side_effect=lambda:clock[0]), \
                        mock.patch.object(b,'verify_host_record',side_effect=checked):
                    b.run_host_attempt(root/'native',name,work,root/'attempt',record,lambda:None)
                self.assertEqual(len(commands),count)
                self.assertEqual(record['native_process_seconds'],count*.25)
                self.assertEqual(record['independent_verification_seconds'],len(work['expected'])*2)
                self.assertEqual(record['workflow_seconds'],count+len(work['expected'])*2)
                self.assertEqual(record['peak_process_rss_bytes'],1024)
                self.assertTrue(record['complete'])
                for argv in commands:
                    for flag,value in b.HOST_LIMITS.items():
                        self.assertEqual(argv[argv.index(flag)+1],str(value))

    def test_native_failure_retains_partial_evidence(self):
        cases=(('time limit',-15,'partial output',True),
               ('wired memory exceeded 3 GiB',-15,'partial output',True),
               ('exit 2',2,'partial output\nresource_limit: scan budget exhausted\n',True),
               ('exit 2',2,'error: other failure',False),
               ('exit 1',1,'resource_limit: unexpected exit code',False),
               ('exit 2',2,'error: resource_limit: embedded text',False))
        for reason,exit_code,log,resource in cases:
            with self.subTest(reason=reason,exit_code=exit_code,log=log),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);source=root/'input';source.write_text('input')
                record={'error':None};saves=[]
                def guard(argv,directory):
                    directory.mkdir();(directory/'run.log').write_text(log)
                    return {'complete':False,'error':reason,'exit_code':exit_code,'memory':[]}
                with mock.patch.object(b,'guarded_run',side_effect=guard),self.assertRaises(RuntimeError):
                    b.run_host_attempt(root/'native','report-100',{'input':source,'expected':[]},
                                       root/'attempt',record,lambda:saves.append(dict(record)))
                self.assertFalse(record['complete'])
                self.assertEqual(record['resource_violation'],resource)
                self.assertEqual((root/'attempt/guard-0/run.log').read_text(),log)
                self.assertTrue(saves[-1]['error'])

    def test_independent_host_factors_and_identity_checks(self):
        oracle=b.host_oracle();scheme=oracle.schoolbook((1,1,1))
        row=dict(scheme,scheme_id=oracle.identity(scheme),factors_id=oracle.identity(scheme,False))
        self.assertEqual(b.verify_host_record(row,scheme)['domain'],'ZT')
        for key,value in (('scheme_id','wrong'),('factors_id','wrong'),('u',[[0]])):
            with self.subTest(key=key),self.assertRaises(ValueError):
                b.verify_host_record(dict(row,**{key:value}),scheme)


if __name__ == '__main__':
    unittest.main()
