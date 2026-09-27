"""Completed-workload execution and profile comparison without GPU dispatch."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_measurement_adapters import b, circuit, log, scalar
import guard as metal_guard


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


class FGM2DeadlineTests(unittest.TestCase):
    def test_supplied_public_witnesses_bind_raw_factors_without_synthesis(self):
        private=b.ROOT/'benchmarks/workflow/fixtures/fgm1/factors/original.json'
        _,sources=b.fgm2_inputs(private)
        coverage=b.fgm2_reference_coverage(sources)
        self.assertEqual({row['id'] for row in coverage},{'sun-56','cn122-55','cn122-58'})
        self.assertNotEqual(sources['sun']['raw_factors_id'],sources['sun']['factors_id'])
        sun=next(row for row in coverage if row['id']=='sun-56')
        self.assertEqual(sun['factors_id'],sources['sun']['raw_factors_id'])
        self.assertEqual(sun['construction_family_witness'],'unassessed')

    def test_preliminary_binding_checks_ordered_factors_counts_and_hash(self):
        private=b.ROOT/'benchmarks/workflow/fixtures/fgm1/factors/original.json'
        _,sources=b.fgm2_inputs(private)
        source=sources['cn122']
        supplied=b.ROOT/'benchmarks/workflow/fixtures/fgm1/circuits/cn122-55.json'
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'circuit.jsonl'
            binding=dict(input_presentation_index=0,submitted_factors_id=source['raw_factors_id'],
                effective_input_factors_id=source['factors_id'],source_sha256=source['sha256'])
            circuit_data=json.loads(supplied.read_text());circuit_data['source_binding']=binding
            path.write_text(json.dumps(circuit_data,separators=(',',':'))+'\n')
            result=dict(circuit_record_index=0,result_rank=23,
                result_factors_id=source['factors_id'],effective_input_factors_id=source['factors_id'],
                verified_circuit_additions=55,verified_circuit_additions_by_stage=dict(u=13,v=14,w=28))
            result.update(binding)
            receipt=dict(circuit_artifact=dict(path=str(path),format='jsonl',records=1,sha256=b.digest(path)),results=[result])
            preliminary=b.fgm2_preliminary_circuit(path,receipt,source)
            self.assertEqual(preliminary['claimed_additions'],55)
            normalized=source['reference']
            reference=Path(temporary)/'reference.json'
            reference.write_text(json.dumps(dict(n=normalized['dimensions'],m=normalized['rank'],z2=False,
                **{key:normalized[key] for key in 'uvw'})))
            verified=subprocess.run([sys.executable,str(b.ROOT/'tests/metal/verify.py'),str(path),
                '--reference',str(reference)],capture_output=True,text=True,check=True)
            verifier_log=Path(temporary)/'verifier.log';verifier_log.write_text(verified.stdout)
            self.assertEqual(b.fgm2_bind_verifier(verifier_log,path,reference,preliminary,receipt,source)['additions'],55)
            result['verified_circuit_additions_by_stage']['w']=29
            with self.assertRaises(ValueError):b.fgm2_bind_verifier(verifier_log,path,reference,preliminary,receipt,source)
            result['verified_circuit_additions_by_stage']['w']=28
            receipt['circuit_artifact']['sha256']='0'*64
            with self.assertRaises(ValueError):b.fgm2_preliminary_circuit(path,receipt,source)
            receipt['circuit_artifact']['sha256']=b.digest(path)
            receipt['circuit_artifact']['format']='circuit-json'
            with self.assertRaises(ValueError):b.fgm2_preliminary_circuit(path,receipt,source)
            receipt['circuit_artifact']['format']='jsonl'
            circuit_data['factors_id']='wrong'
            path.write_text(json.dumps(circuit_data,separators=(',',':'))+'\n')
            receipt['circuit_artifact']['sha256']=b.digest(path)
            with self.assertRaises(ValueError):b.fgm2_preliminary_circuit(path,receipt,source)

    def test_frozen_schedule_has_paired_seeds_and_rotated_order(self):
        protocol=json.loads(b.FGM2_PROTOCOL_PATH.read_text())
        rows=b.fgm2_schedule(protocol)
        self.assertEqual(len(rows),72)
        for name in [*protocol['public_inputs'],'private']:
            for seed in protocol['seeds']:
                self.assertEqual({r['strategy'] for r in rows if r['input']==name and r['seed']==seed},
                                 set(protocol['strategies']))
        self.assertEqual([r['strategy'] for r in rows[:4]],protocol['strategies'])
        self.assertEqual([r['strategy'] for r in rows[4:8]],protocol['strategies'][1:]+protocol['strategies'][:1])

    def synthetic_run(self, native_seconds=1, verifier_seconds=1, preliminary_seconds=0,
                      forced=False, jump_after_first=0, binding_error=False,
                      native_guard_raises=False, verifier_guard_raises=False,
                      verifier_cleanup_failure=False, verifier_timeout=False, missing_guard_evidence=False,
                      verifier_late_exit=False, headroom_seconds=0, headroom_sample_seconds=0,
                      preparation_seconds=0, final_binding_seconds=0,
                      schedule_count=2, finalization_seconds=0, preparation_error=False,
                      headroom_sample_error=None, closing_write_seconds=0):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        binary=root/'bin';binary.mkdir()
        for name in ('additions_reducer','scheme_tool'):(binary/name).write_text('synthetic')
        shader=binary/'shaders';shader.mkdir()
        (shader/'signed.metallib').write_text('synthetic library')
        (shader/'signed.metallib.build.json').write_text(json.dumps({'sha256':b.digest(shader/'signed.metallib')}))
        stat=(binary/'additions_reducer').stat()
        (binary/'additions_reducer.build.json').write_text(json.dumps(dict(
            output=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),
            inputs=dict(dependencies={}))))
        source=root/'factor.json';source.write_text('{}')
        protocol=json.loads(b.FGM2_PROTOCOL_PATH.read_text())
        factors=dict(dimensions=[3,3,3],rank=23,domain='ZT',orientation='cyclic-w',u=[],v=[],w=[])
        inputs={'original':dict(source=source,sha256=b.digest(source),reference=factors,factors_id='id')}
        schedule=[dict(id=f'original-7-{index:02d}',input='original',seed=7,
                       strategy='baseline' if index%2==0 else 'transpose')
                  for index in range(schedule_count)]
        tick=[0.]
        guards=[]
        original_write_json=b.write_json
        original_report=b.fgm2_report
        sampled=[False]
        wait_remaining=[headroom_seconds]
        closing_started=[False]
        closing_write_recorded=[False]
        def clock():return tick[0]
        def sample(timeout):
            if not sampled[0]:
                sampled[0]=True
                tick[0]+=headroom_sample_seconds
                if headroom_sample_error=='timeout':
                    raise subprocess.TimeoutExpired(cmd='wired memory sample',timeout=timeout)
                if headroom_sample_error=='oserror':
                    raise OSError('wired memory sample failed')
            return 3221225472 if wait_remaining[0] else 0
        def sleep(seconds):
            tick[0]+=wait_remaining[0]
            wait_remaining[0]=0
        def write_json(path,value):
            if Path(path).name=='config.json':
                tick[0]+=preparation_seconds
                if preparation_error:raise OSError('config persistence failed')
            if (closing_started[0] and not closing_write_recorded[0]
                    and Path(path).name=='measurement.json'):
                closing_write_recorded[0]=True
                tick[0]+=closing_write_seconds
            return original_write_json(path,value)
        def report(data,summary):
            closing_started[0]=True
            tick[0]+=finalization_seconds
            return original_report(data,summary)
        def guarded(argv,directory,*,absolute_deadline):
            self.assertLess(tick[0],absolute_deadline)
            directory.mkdir()
            if not missing_guard_evidence or (argv[0]=='/usr/bin/time' and not forced):
                (directory/'result.json').write_text('{}')
            (directory/'run.log').write_text('synthetic')
            guards.append((Path(argv[3] if argv[0]=='/usr/bin/time' else argv[0]).name,absolute_deadline))
            if argv[0]=='/usr/bin/time':
                tick[0]+=native_seconds
                if native_guard_raises:raise OSError('native guard artifact write failed')
                config=json.loads(Path(argv[-1]).read_text())
                receipt_path=Path(config['output'])
                result=dict(strategy=config['reduction']['strategy'],baseline_additions=60,
                            baseline_additions_by_stage=dict(u=20,v=20,w=20),
                            stage_sources=dict(u='baseline',v='baseline',w='baseline'),
                            construction={},phase_microseconds={},
                            verified_circuit_additions=60,
                            verified_circuit_additions_by_stage=dict(u=20,v=20,w=20))
                receipt=dict(status='complete',execution_started=True,
                             configuration_sha256=b.digest(Path(argv[-1])),
                             executable_sha256=b.digest(root/'out/binaries/additions_reducer'),
                             library_mode='metallib',
                             library_sha256=b.digest(root/'out/binaries/shaders/signed.metallib'),
                             presentations=[dict(source_sha256=b.digest(root/'out/inputs/original.json'),
                                                 effective_factors_id='id')],
                             results=[result])
                receipt_path.write_text(json.dumps(receipt))
                Path(str(receipt_path)+'.circuits.jsonl').write_text('{}\n')
                if jump_after_first:tick[0]+=jump_after_first
                return dict(complete=not forced,forced_termination=forced,cleanup_failure=False)
            tick[0]+=verifier_seconds
            if verifier_guard_raises:raise OSError('verifier guard artifact write failed')
            if verifier_cleanup_failure:
                return dict(complete=False,forced_termination=False,cleanup_failure=True)
            if verifier_late_exit:
                return dict(complete=False,forced_termination=False,cleanup_failure=False,
                            exit_code=0,error='time limit at completion')
            if verifier_timeout:
                return dict(complete=False,forced_termination=True,cleanup_failure=False,
                            exit_code=-15,error='time limit')
            return dict(complete=True,forced_termination=False,cleanup_failure=False)
        def preliminary(*_):
            tick[0]+=preliminary_seconds
            return dict(factors_id='id',claimed_additions=60,circuit_sha256='hash')
        def bind(*_):
            tick[0]+=final_binding_seconds
            if binding_error:raise ValueError('independent factor mismatch')
            return dict(factors_id='id',additions=60,
                        additions_by_stage=dict(u=20,v=20,w=20),circuit_sha256='hash')
        with mock.patch.object(b,'fgm2_inputs',return_value=(protocol,inputs)), \
             mock.patch.object(b,'fgm2_reference_coverage',return_value=[]), \
             mock.patch.object(b,'fgm2_schedule',return_value=schedule), \
             mock.patch.object(b,'host_machine_identity',return_value={'hardware':{}}), \
             mock.patch.object(b,'source_identity',return_value={}), \
             mock.patch.object(b,'wired_memory',side_effect=sample), \
             mock.patch.object(b,'write_json',side_effect=write_json), \
             mock.patch.object(b,'fgm2_report',side_effect=report), \
             mock.patch.object(b,'native_dispatch_evidence',return_value={'status':'GPU_EXECUTED',
                'dispatches':[('transposePairKernel',32,1),('constructorClosureKernel',32,1)]}), \
             mock.patch.object(b,'fgm2_preliminary_circuit',side_effect=preliminary), \
             mock.patch.object(b,'fgm2_bind_verifier',side_effect=bind):
            data=b.execute_fgm2(source,binary,root/'out',clock=clock,sleeper=sleep,guard=guarded)
        return data,guards,root

    def test_native_and_verifier_receive_guard_start_deadlines(self):
        data,guards,root=self.synthetic_run()
        self.assertEqual([row['status'] for row in data['trials']],['verified','verified'])
        self.assertEqual([deadline for _,deadline in guards],[45,46,47,48])
        self.assertFalse(data['complete']) # A two-row injected schedule cannot complete 72 trials.
        self.assertTrue((root/'out/binaries/additions_reducer').is_file())
        self.assertTrue((root/'out/harness/tests/metal/verify.py').is_file())
        self.assertIn('original | 7 | baseline', (root/'out/report.md').read_text())
        self.assertEqual(data['trials'][0]['native_command'][3],str(root/'out/binaries/additions_reducer'))
        self.assertEqual(data['trials'][0]['verification_command'][1],
                         str(root/'out/harness/tests/metal/verify.py'))

    def test_work_beyond_old_trial_cutoff_can_verify(self):
        data,guards,_=self.synthetic_run(native_seconds=7,preliminary_seconds=4,
                                          verifier_seconds=11)
        self.assertEqual(data['trials'][0]['status'],'verified')
        self.assertEqual(guards[0][1],45)
        self.assertEqual(guards[1][1],56)
        self.assertFalse(data['complete'])

    def test_headroom_wait_does_not_shorten_native_guard(self):
        data,guards,_=self.synthetic_run(headroom_seconds=9)
        self.assertEqual(data['trials'][0]['status'],'verified')
        self.assertEqual(guards[0][1],54)
        self.assertEqual(guards[1][1],55)

    def test_no_native_launch_after_preparation_or_headroom_exhausts_admission(self):
        for option in (dict(preparation_seconds=751),dict(headroom_sample_seconds=751),
                       dict(headroom_seconds=751)):
            with self.subTest(option=option):
                data,guards,_=self.synthetic_run(**option)
                self.assertEqual(guards,[])
                self.assertEqual(data['trials'][0]['status'],'unrun')
                self.assertIn('budget',data['trials'][0]['unrun_reason'])

    def test_verifier_requires_full_global_admission(self):
        data,guards,_=self.synthetic_run(preliminary_seconds=801)
        self.assertEqual(len(guards),1)
        self.assertEqual(data['trials'][0]['status'],'artifact-awaiting-verification')
        self.assertEqual(data['trials'][1]['status'],'unrun')
        data,guards,_=self.synthetic_run(preliminary_seconds=799)
        self.assertEqual(len(guards),2)
        self.assertEqual(guards[1][1],845)

    def test_child_guard_completion_precedes_binding_cost(self):
        data,guards,_=self.synthetic_run(native_seconds=45.1,preliminary_seconds=.2,
                                          verifier_seconds=45.1,final_binding_seconds=.2)
        self.assertEqual(data['trials'][0]['status'],'verified')
        self.assertAlmostEqual(guards[0][1],45)
        self.assertAlmostEqual(guards[1][1],90.3)
        self.assertGreater(data['trials'][0]['native_receipt_binding_seconds'],0)
        self.assertGreater(data['trials'][0]['final_binding_seconds'],0)

    def test_row_snapshot_matches_measurement_after_persistence(self):
        data,_,root=self.synthetic_run()
        for index,row in enumerate(data['trials']):
            trial=root/'out/trials'/f'{index:02d}-{row["id"]}'/'trial.json'
            snapshot=json.loads(trial.read_text())
            self.assertEqual(snapshot,json.loads(json.dumps(
                {key:value for key,value in row.items() if key!='trial_sha256'})))
            self.assertEqual(row['trial_sha256'],b.digest(trial))

    def test_failed_phase_cost_and_preparation_status_are_retained(self):
        data,_,_=self.synthetic_run(preparation_error=True,preparation_seconds=2)
        self.assertEqual(data['trials'][0]['status'],'failed')
        self.assertEqual(data['trials'][0]['preparation_seconds'],2)
        data,_,_=self.synthetic_run(native_guard_raises=True,native_seconds=3)
        self.assertEqual(data['trials'][0]['status'],'terminated')
        self.assertEqual(data['trials'][0]['native_supervised_seconds'],3)
        data,_,_=self.synthetic_run(verifier_guard_raises=True,verifier_seconds=4)
        self.assertEqual(data['trials'][0]['status'],'artifact-awaiting-verification')
        self.assertEqual(data['trials'][0]['verifier_supervised_seconds'],4)

    def test_binding_past_global_deadline_has_no_completion_credit(self):
        data,guards,_=self.synthetic_run(final_binding_seconds=899)
        self.assertEqual(len(guards),2)
        self.assertEqual(data['trials'][0]['status'],'late-verified')
        self.assertEqual(data['status'],'budget-exceeded')

    def test_complete_coverage_and_finalization_overrun(self):
        data,_,_=self.synthetic_run(schedule_count=72)
        self.assertTrue(data['complete'])
        self.assertEqual(data['status'],'complete')
        self.assertEqual(b.fgm2_summary(data)['verified'],72)
        data,_,_=self.synthetic_run(schedule_count=72,finalization_seconds=901)
        self.assertFalse(data['complete'])
        self.assertEqual(data['status'],'budget-exceeded')

    def test_finalization_report_matches_retained_measurement(self):
        data,_,root=self.synthetic_run(schedule_count=72,finalization_seconds=5)
        output=root/'out'
        retained=json.loads((output/'measurement.json').read_text())
        summary=json.loads((output/'summary.json').read_text())
        report=(output/'report.md').read_text()
        self.assertEqual(data['status'],'complete')
        self.assertTrue(data['complete'])
        self.assertEqual(retained['finalization_seconds'],data['finalization_seconds'])
        self.assertIn(f"finalization: {retained['finalization_seconds']:.3f} s",report)
        self.assertEqual(retained['timing_scope'],data['timing_scope'])
        self.assertIn(retained['timing_scope'],report)
        self.assertEqual((summary['status'],summary['complete'],summary['verified']),
                         ('complete',True,72))
        self.assertIn('Status: complete; verified: 72/72.',report)
        self.assertEqual((output/'measurement.sha256').read_text().strip(),
                         b.digest(output/'measurement.json'))

    def test_budget_overrun_during_closing_write_updates_all_outputs(self):
        data,_,root=self.synthetic_run(schedule_count=72,closing_write_seconds=757)
        output=root/'out'
        retained=json.loads((output/'measurement.json').read_text())
        summary=json.loads((output/'summary.json').read_text())
        report=(output/'report.md').read_text()
        self.assertEqual((data['status'],retained['status'],summary['status']),
                         ('budget-exceeded',)*3)
        self.assertFalse(data['complete'])
        self.assertFalse(retained['complete'])
        self.assertFalse(summary['complete'])
        self.assertGreater(retained['budget_check_elapsed_seconds'],900)
        self.assertEqual(retained['budget_check_elapsed_seconds'],
                         data['budget_check_elapsed_seconds'])
        self.assertIn(f"Budget check after closing writes: {retained['budget_check_elapsed_seconds']:.3f} s.",
                      report)
        self.assertIn('Status: budget-exceeded; verified: 72/72.',report)
        self.assertEqual((output/'measurement.sha256').read_text().strip(),
                         b.digest(output/'measurement.json'))

    def test_headroom_sample_timeout_classifies_budget_boundary(self):
        data,guards,_=self.synthetic_run(preparation_seconds=749.9,
            headroom_sample_seconds=.2,headroom_sample_error='timeout',schedule_count=1)
        self.assertEqual(guards,[])
        self.assertEqual(data['trials'][0]['status'],'unrun')
        self.assertIn('budget',data['trials'][0]['unrun_reason'])
        self.assertEqual(data['stop_reason'],'global admission cutoff')

    def test_headroom_sample_timeout_with_budget_is_failure(self):
        data,guards,_=self.synthetic_run(headroom_sample_seconds=.2,
            headroom_sample_error='timeout',schedule_count=1)
        self.assertEqual(guards,[])
        self.assertEqual(data['trials'][0]['status'],'failed')
        self.assertNotIn('unrun_reason',data['trials'][0])
        self.assertIn('wired memory sample',data['trials'][0]['error'])

    def test_forced_gpu_termination_stops_future_trials(self):
        data,guards,_=self.synthetic_run(forced=True)
        self.assertEqual(data['trials'][0]['status'],'terminated')
        self.assertEqual(data['trials'][1]['status'],'unrun')
        self.assertEqual(len(guards),1)

    def test_independent_binding_failure_stops_future_gpu_trials(self):
        data,guards,_=self.synthetic_run(binding_error=True)
        self.assertEqual(data['trials'][0]['status'],'artifact-awaiting-verification')
        self.assertEqual(data['trials'][1]['status'],'unrun')
        self.assertEqual(data['stop_reason'],'correctness or identity failure')
        self.assertEqual(len(guards),2)

    def test_guard_exceptions_and_verifier_cleanup_stop_future_gpu_trials(self):
        for option, expected_status in (('native_guard_raises','terminated'),
                                        ('verifier_guard_raises','artifact-awaiting-verification'),
                                        ('verifier_cleanup_failure','artifact-awaiting-verification')):
            with self.subTest(option=option):
                data,guards,_=self.synthetic_run(**{option:True})
                self.assertEqual(data['trials'][0]['status'],expected_status)
                self.assertEqual(data['trials'][1]['status'],'unrun')
                self.assertEqual(data['stop_reason'],'unknown process cleanup after guard failure')
                self.assertEqual(len(guards),1 if option=='native_guard_raises' else 2)

    def test_cleanup_flags_survive_missing_guard_evidence(self):
        for options in (dict(forced=True),dict(verifier_cleanup_failure=True)):
            data,guards,_=self.synthetic_run(missing_guard_evidence=True,**options)
            self.assertEqual(data['trials'][1]['status'],'unrun')
            self.assertEqual(len(guards),1 if options.get('forced') else 2)
            self.assertIn(data['stop_reason'],('forced GPU termination','unknown process cleanup after guard failure'))

    def test_verifier_timeout_is_incomplete_not_correctness_rejection(self):
        data,guards,_=self.synthetic_run(verifier_timeout=True)
        self.assertEqual([row['status'] for row in data['trials']],
                         ['artifact-awaiting-verification']*2)
        self.assertEqual(len(guards),4)
        self.assertNotIn('stop_reason',data)
        self.assertTrue(all('error' not in row for row in data['trials']))

    def test_successful_late_verifier_exit_keeps_late_evidence(self):
        data,guards,_=self.synthetic_run(verifier_late_exit=True)
        self.assertEqual([row['status'] for row in data['trials']],['late-verified']*2)
        self.assertFalse(data['complete'])
        self.assertEqual(len(guards),4)

    def test_global_admission_cutoff_retains_unrun_coverage(self):
        data,guards,_=self.synthetic_run(jump_after_first=845)
        self.assertEqual(data['trials'][1]['status'],'unrun')
        self.assertEqual(data['stop_reason'],'global admission cutoff')
        self.assertEqual(len(guards),1)

    def test_guard_rechecks_after_bounded_initial_memory_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            tick=[100.]
            def sample(timeout):
                self.assertEqual(timeout,1)
                tick[0]=101.
                return 0
            with mock.patch.object(metal_guard.time,'monotonic',side_effect=lambda:tick[0]), \
                 mock.patch.object(metal_guard,'wired_memory',side_effect=sample), \
                 mock.patch.object(metal_guard.subprocess,'Popen') as popen:
                result=metal_guard.run(['synthetic'],Path(temporary)/'guard',absolute_deadline=101.)
            popen.assert_not_called()
            self.assertFalse(result['complete'])
            self.assertIn('time limit before launch',result['error'])

    def test_guard_default_limit_and_positive_byte_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for index,value in enumerate((0,-1,True,1.5,'4 GiB',None)):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'positive integer'):
                    metal_guard.run(['synthetic'],root/str(index),wired_limit_bytes=value)
                self.assertFalse((root/str(index)).exists())
            with mock.patch.object(metal_guard,'wired_memory',return_value=metal_guard.LIMIT+1), \
                 mock.patch.object(metal_guard.subprocess,'Popen') as popen:
                result=metal_guard.run(['synthetic'],root/'default')
            popen.assert_not_called()
            self.assertEqual(result['wired_limit'],3*1024**3)
            self.assertEqual(result['error'],'wired memory exceeded 3 GiB before launch')
            self.assertFalse(result['forced_termination'])

    def test_guard_four_gib_allows_samples_below_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            process=mock.Mock()
            process.poll.side_effect=(None,0,0)
            process.returncode=0
            limit=4*1024**3
            with mock.patch.object(metal_guard,'wired_memory',side_effect=(3*1024**3+1,limit-1)), \
                 mock.patch.object(metal_guard.subprocess,'Popen',return_value=process), \
                 mock.patch.object(metal_guard,'terminate') as terminate:
                result=metal_guard.run(['synthetic'],Path(temporary)/'guard',wired_limit_bytes=limit)
            self.assertTrue(result['complete'])
            self.assertEqual(result['wired_limit'],limit)
            self.assertEqual([row['wired_bytes'] for row in result['memory']],[3*1024**3+1,limit-1])
            self.assertEqual(result['time_limit'],45)
            self.assertFalse(result['forced_termination'])
            terminate.assert_called_once_with(process)

    def test_guard_four_gib_terminates_on_runtime_excess(self):
        with tempfile.TemporaryDirectory() as temporary:
            process=mock.Mock()
            process.poll.side_effect=(None,None)
            process.returncode=None
            limit=4*1024**3
            def terminated(child):
                self.assertIs(child,process)
                child.returncode=-15
            with mock.patch.object(metal_guard,'wired_memory',side_effect=(limit-1,limit+1)), \
                 mock.patch.object(metal_guard.subprocess,'Popen',return_value=process), \
                 mock.patch.object(metal_guard,'terminate',side_effect=terminated) as terminate:
                result=metal_guard.run(['synthetic'],Path(temporary)/'guard',wired_limit_bytes=limit)
            self.assertFalse(result['complete'])
            self.assertEqual(result['wired_limit'],limit)
            self.assertEqual(result['error'],'wired memory exceeded 4 GiB')
            self.assertTrue(result['forced_termination'])
            self.assertFalse(result['cleanup_failure'])
            self.assertEqual(result['exit_code'],-15)
            terminate.assert_called_once_with(process)


if __name__ == '__main__':
    unittest.main()
