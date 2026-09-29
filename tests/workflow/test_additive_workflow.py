"""Focused independent bindings and clock guards for the FGM-3 harness."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/workflow'))
import baseline as b


class Clock:
    def __init__(self,value=0):self.value=value
    def now(self):return self.value
    def sleep(self,seconds):self.value+=seconds


class AdditiveWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads(b.FGM3_PROTOCOL_PATH.read_text())

    def evaluation(self):
        factors=json.loads((ROOT/'benchmarks/workflow/fixtures/fgm1/factors/cn122.json').read_text())
        circuit=json.loads((ROOT/'benchmarks/workflow/fixtures/fgm1/circuits/cn122-58.json').read_text())
        source=dict(dimensions=[3,3,3],rank=23,domain='ZT',orientation='cyclic-w',
                    **{key:factors[key] for key in 'uvw'})
        oracle=b.host_oracle()
        settings=dict(self.protocol['evaluation'],seed=7)
        producer=dict(executable_sha256='a'*64,library_mode='metallib',library_sha256='b'*64)
        checked=b.verify(circuit,factors)
        evaluation=dict(schema='fgm-additive-evaluation-v1',scheme_id=oracle.identity(source),
            factors_id=oracle.identity(source,False),source_factors_id=oracle.identity(source,False),
            seed=b.fgm3_evaluation_seed(7,oracle.identity(source,False)),settings=settings,
            producer=producer,admission_order=0,additions=checked['additions'],
            additions_by_stage=checked['additions_by_stage'],circuit=circuit,
            phase_microseconds={},construction={},stage_sources={},baseline_additions=58,
            baseline_additions_by_stage=checked['additions_by_stage'])
        row=dict(schema='fgm-journal-evaluation-v1',source_scheme=source,evaluation=evaluation)
        return row,settings,producer

    def test_exact_factor_cost_seed_and_build_bindings(self):
        row,settings,producer=self.evaluation()
        self.assertEqual(b.fgm3_checked_evaluation(row,settings,producer)['additions'],58)
        mutations=[('source',lambda r:r['source_scheme']['u'][0].__setitem__(0,1-r['source_scheme']['u'][0][0])),
                   ('count',lambda r:r['evaluation'].__setitem__('additions',57)),
                   ('stage',lambda r:r['evaluation']['additions_by_stage'].__setitem__('u',0)),
                   ('seed',lambda r:r['evaluation'].__setitem__('seed',1)),
                   ('producer',lambda r:r['evaluation']['producer'].__setitem__('executable_sha256','c'*64))]
        for name,change in mutations:
            with self.subTest(name=name):
                altered=copy.deepcopy(row);change(altered)
                with self.assertRaises((ValueError,AssertionError)):
                    b.fgm3_checked_evaluation(altered,settings,producer)

    def test_canonical_alias_does_not_re_evaluate_or_replace_prefix(self):
        row,settings,producer=self.evaluation()
        first=b.fgm3_bind_export([], [row],settings,producer,1)
        self.assertEqual(len(first),1)
        self.assertEqual(b.fgm3_bind_export([row],[row],settings,producer,2),[])
        duplicate=copy.deepcopy(row)
        duplicate['evaluation']['admission_order']=1
        with self.assertRaisesRegex(ValueError,'novelty'):
            b.fgm3_bind_export([row],[row,duplicate],settings,producer,2)
        changed=copy.deepcopy(row)
        changed['evaluation']['additions']=57
        with self.assertRaisesRegex(ValueError,'prefix'):
            b.fgm3_bind_export([row],[changed],settings,producer,2)

    def test_parent_installation_requires_verified_score_and_group(self):
        row,settings,producer=self.evaluation()
        known=b.fgm3_bind_export([], [row],settings,producer,1)
        event=dict(schema='fgm-parent-installation-v1',selected_parent_id=known[0]['scheme_id'],
            selected_factors_id=known[0]['factors_id'],selected_additions=58,
            selection_group='elite',installed=True,kind='initial',worker=0)
        self.assertEqual(len(b.fgm3_bind_installations([], [event],known,'cost-diverse')),1)
        altered=dict(event,selected_additions=57)
        with self.assertRaises(ValueError):b.fgm3_bind_installations([], [altered],known,'cost-diverse')
        with self.assertRaises(ValueError):b.fgm3_bind_installations([], [event],known,'uniform')

    def test_initial_submission_and_normalized_evaluation_bind_separately(self):
        entries=[dict(sha256=str(i),source_factors_id=f'raw-{i}',
                      effective_factors_id=f'effective-{i}') for i in range(16)]
        presentations=[dict(source_sha256=str(i),submitted_factors_id=f'raw-{i}',
                            effective_factors_id=f'effective-{i}') for i in range(16)]
        evaluations=[dict(source_factors_id=f'effective-{i}') for i in range(16)]
        b.fgm3_bind_initial_population(entries,presentations,evaluations)
        altered=copy.deepcopy(evaluations)
        altered[1]['source_factors_id']='raw-1'
        with self.assertRaisesRegex(ValueError,'normalized'):
            b.fgm3_bind_initial_population(entries,presentations,altered)

    def test_explicit_m4_initial_population_count_preserves_fgm3_default(self):
        for count in (3, 8):
            entries=[dict(sha256=str(i),source_factors_id=f'raw-{i}',
                          effective_factors_id=f'effective-{i}') for i in range(count)]
            presentations=[dict(source_sha256=str(i),submitted_factors_id=f'raw-{i}',
                                effective_factors_id=f'effective-{i}') for i in range(count)]
            evaluations=[dict(source_factors_id=f'effective-{i}') for i in range(count)]
            b.fgm3_bind_initial_population(entries,presentations,evaluations,
                                           expected_population_count=count)
            with self.assertRaisesRegex(ValueError,'population'):
                b.fgm3_bind_initial_population(entries,presentations,evaluations)
            with self.assertRaisesRegex(ValueError,'population'):
                b.fgm3_bind_initial_population(entries,presentations[:-1],evaluations,
                                               expected_population_count=count)

    def test_valid_factor_substitution_cannot_relabel_frozen_population(self):
        original=ROOT/'benchmarks/workflow/fixtures/fgm1/factors/original.json'
        replacement=ROOT/'benchmarks/workflow/fixtures/fgm1/factors/cn122.json'
        with tempfile.TemporaryDirectory() as temp:
            copied=Path(temp)/'input.json'
            copied.write_bytes(original.read_bytes())
            checked=b.fgm2_source(copied)
            entry=dict(label='original',source_path='build/fgm2/comparison-02/inputs/original.json',
                       sha256=b.digest(copied),source_sha256=b.digest(original),
                       source_factors_id=checked['raw_factors_id'],effective_factors_id=checked['factors_id'])
            pinned=dict(path='inputs/original.json',sha256=entry['sha256'],
                        raw_factors_id=entry['source_factors_id'],factors_id=entry['effective_factors_id'],
                        dimensions=[3,3,3],rank=23,domain='ZT')
            b.fgm3_bind_population_entry(entry,copied,original,pinned,name='original')
            copied.write_bytes(replacement.read_bytes())
            alternate=b.fgm2_source(copied)
            forged=dict(entry,sha256=b.digest(copied),source_sha256=b.digest(copied),
                        source_factors_id=alternate['raw_factors_id'],
                        effective_factors_id=alternate['factors_id'])
            with self.assertRaisesRegex(ValueError,'pinned measurement'):
                b.fgm3_bind_population_entry(forged,copied,replacement,pinned,name='original')

            selected_source=dict(forged,label='retained-001',source_path='recorded/retained.json',
                                 canonical_id=b.host_oracle().identity(alternate['raw_reference']))
            retained=dict(source_path='recorded/retained.json',source_sha256=entry['sha256'],
                          input_sha256=entry['sha256'],canonical_id=b.host_oracle().identity(
                              checked['raw_reference']),source_factors_id=entry['source_factors_id'],
                          effective_factors_id=entry['effective_factors_id'])
            collection=dict(id='retained-001',sha256=entry['sha256'],
                            path='selection-inputs/001.json',namespace='test')
            binding=dict(collection,manifest_sha256='m')
            selected=dict(source_binding=binding,source_bindings=[binding],source_sha256=forged['sha256'],
                          scheme_id=selected_source['canonical_id'],
                          submitted_factors_id=forged['source_factors_id'],
                          effective_factors_id=forged['effective_factors_id'])
            with self.assertRaisesRegex(ValueError,'retained native binding'):
                b.fgm3_bind_population_entry(selected_source,copied,replacement,retained,
                    collection=collection,selected=selected,selection_sha256='m')

    def test_allowance_endpoints_headroom_and_finalization(self):
        self.assertEqual(b.fgm3_chunk_allowance([18,24,20,21]),32)
        p=self.protocol
        self.assertEqual(b.fgm3_arm_order(p),[(7,'uniform'),(7,'cost-diverse'),
            (19,'cost-diverse'),(19,'uniform'),(41,'uniform'),(41,'cost-diverse')])
        self.assertIsNone(b.fgm3_admission(0,90,900,32,p))
        self.assertEqual(b.fgm3_admission(59,90,900,32,p),'arm admission allowance')
        self.assertEqual(b.fgm3_admission(0,90,100,32,p),'global admission reserve')
        self.assertEqual(b.fgm3_admission(0,90,190,32,p),'global admission reserve')
        clock=Clock()
        samples=[]
        def high_then_available(timeout):
            samples.append(timeout)
            clock.sleep(10)
            return p['wired_limit_bytes'] if len(samples)<3 else 0
        reason,wait=b.fgm3_wait_headroom(clock.now,clock.sleep,900,90,32,p,high_then_available)
        self.assertIsNone(reason)
        self.assertGreaterEqual(wait,30)
        clock=Clock(55)
        reason,_=b.fgm3_wait_headroom(clock.now,clock.sleep,900,90,32,p,
                                      lambda timeout:p['wired_limit_bytes'])
        self.assertEqual(reason,'arm admission allowance')
        clock=Clock(55)
        def timed_out_sample(timeout):
            clock.sleep(4)
            raise subprocess.TimeoutExpired('vm_stat',timeout)
        reason,_=b.fgm3_wait_headroom(clock.now,clock.sleep,900,90,32,p,timed_out_sample)
        self.assertEqual(reason,'arm admission allowance')

    def test_late_verification_has_no_earlier_endpoint_credit(self):
        state=dict(evaluations=[dict(scheme_id='new',additions=55,additions_by_stage=dict(u=13,v=14,w=28),
                                     verified_seconds=31)],
            observations=[dict(scheme_id='new',rank=23,observed_at_seconds=31)],
            installations=[dict(parent_id='new',group='elite',installed=True,kind='restart',
                                verified_seconds=62)],initial_ids=set())
        self.assertIsNone(b.fgm3_endpoint(state,0,30)['best_additions'])
        self.assertEqual(b.fgm3_endpoint(state,0,60)['best_additions'],55)
        self.assertEqual(b.fgm3_endpoint(state,0,90)['feedback_installations'],1)
        state['installations']=[dict(parent_id='new',group='elite',installed=True,kind='initial',
                                     verified_seconds=62)]
        self.assertEqual(b.fgm3_endpoint(state,0,90)['feedback_installations'],1)
        self.assertEqual(b.fgm3_endpoint(state,0,90)['restart_feedback_installations'],0)
        state['initial_ids']={'new'}
        self.assertEqual(b.fgm3_endpoint(state,0,90)['canonical_discoveries'],0)

    def test_feedback_verdict_requires_timely_cost_diverse_installation(self):
        arms=[dict(selector='uniform',endpoints={'90':dict(feedback_installations=1)}),
              dict(selector='cost-diverse',endpoints={'90':dict(feedback_installations=0)},
                   feedback_installations=1)]
        self.assertEqual(b.fgm3_feedback_finding(arms,True),
                         'inconclusive: no timely cost-diverse feedback installation')
        arms[1]['endpoints']['90']['feedback_installations']=1
        self.assertIsNone(b.fgm3_feedback_finding(arms,True))
        self.assertEqual(b.fgm3_feedback_finding(arms,False),'inconclusive: comparison incomplete')

    def test_export_children_keep_finalization_reserve(self):
        p=self.protocol
        self.assertTrue(b.fgm3_export_budget(0,150,2,p))
        self.assertFalse(b.fgm3_export_budget(1,150,2,p))
        self.assertTrue(b.fgm3_export_budget(45,150,1,p))
        self.assertFalse(b.fgm3_export_budget(46,150,1,p))

    def test_optional_scan_limit_reaches_native_config_and_journal_export(self):
        for scan in (None,1024**3):
            with self.subTest(scan=scan),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);population=root/'population';population.mkdir()
                binaries=root/'binaries';binaries.mkdir()
                inventory={}
                for name in ('flip_graph','scheme_tool','shaders/signed.metallib'):
                    path=binaries/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(name);inventory[name]=b.digest(path)
                protocol=copy.deepcopy(self.protocol)
                if scan is not None:protocol['limits']=dict(scan_bytes=scan)
                state=dict(build_inventory=inventory,evaluations=[],installations=[],observations=[])
                observed=[]
                guard_limits=[]
                def guarded(argv,destination,guard,**kwargs):
                    observed.append(argv)
                    guard_limits.append(kwargs['wired_limit_bytes'])
                    if len(observed)==1:
                        config_path=Path(argv[argv.index('--run-config')+1])
                        config=json.loads(config_path.read_text())
                        receipt=Path(config['output'])
                        b.write_json(receipt,dict(status='complete',execution_started=True,
                            configuration_sha256=b.digest(config_path),
                            executable_sha256=inventory['flip_graph'],library_mode='metallib',
                            library_sha256=inventory['shaders/signed.metallib'],run_id='test',
                            counters={},seed_duplicates=0,completed_batches=1,pool_evictions=0))
                        destination.mkdir();(destination/'run.log').write_text('')
                        return dict(wall_seconds=0)
                    if len(observed)==2:
                        Path(argv[argv.index('--output')+1]).write_text('')
                        return dict(wall_seconds=0)
                    raise RuntimeError('captured journal export arguments')
                with mock.patch.object(b,'fgm3_guarded',side_effect=guarded), \
                     mock.patch.object(b,'native_dispatch_evidence',return_value={}):
                    with self.assertRaisesRegex(RuntimeError,'captured journal export'):
                        b.fgm3_run_chunk(root,population,binaries,protocol,'uniform',7,1,
                            'arm',1,state,900,90,20,clock=lambda:0,sleeper=lambda _:None,
                            guard=lambda *args:None,memory_sample=lambda timeout:0)
                config=json.loads((root/'arm/chunks/001/config.json').read_text())
                if scan is None:self.assertNotIn('limits',config)
                else:self.assertEqual(config['limits'],dict(scan_bytes=scan))
                self.assertEqual(len(observed),3)
                self.assertEqual(guard_limits,[protocol['wired_limit_bytes']]*3)
                for exported,mode in zip(observed[1:],('--evaluations','--observations')):
                    self.assertIn(mode,exported)
                    self.assertEqual(exported[exported.index('--scan-bytes')+1],
                                     str(scan if scan is not None else protocol['history']['storage_bytes']))
                    self.assertEqual(exported[exported.index('--record-bytes')+1],
                                     str(protocol['history']['transaction_bytes']))
        for invalid in (0,-1,True,1.5,'1073741824',None):
            with self.subTest(invalid=invalid):
                protocol=copy.deepcopy(self.protocol);protocol['limits']=dict(scan_bytes=invalid)
                with self.assertRaisesRegex(ValueError,'positive integer'):
                    b.fgm3_config(protocol,'uniform',7,1,1,Path('/unused'),
                                  Path('/unused/history'),Path('/unused/receipt'))

    def test_guard_policy_is_passed_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/'guard'
            calls=[]
            def guarded(argv,output,*,wired_limit_bytes):
                calls.append(wired_limit_bytes)
                output.mkdir()
                (output/'run.log').write_text('completed')
                b.write_json(output/'result.json',dict(wired_limit=wired_limit_bytes))
                return dict(complete=True,wall_seconds=1,memory=[dict(wired_bytes=3*1024**3+1)],
                            wired_limit=wired_limit_bytes)
            record={}
            result=b.fgm3_guarded(['true'],destination,guarded,record=record,field='native_guard',
                                  wired_limit_bytes=4*1024**3)
            self.assertEqual(calls,[4*1024**3])
            self.assertEqual(result['wired_limit_bytes'],4*1024**3)
            self.assertEqual(record['native_guard']['wired_limit_bytes'],4*1024**3)
            self.assertEqual(json.loads((destination/'result.json').read_text())['wired_limit'],4*1024**3)
            mismatched={}
            with self.assertRaisesRegex(ValueError,'differs from protocol'):
                b.fgm3_guarded(['true'],Path(temp)/'other',
                    lambda *args,**kwargs:dict(complete=True,wall_seconds=1,
                                               wired_limit=3*1024**3),
                    record=mismatched,field='native_guard',wired_limit_bytes=4*1024**3)
            self.assertEqual(mismatched['native_guard']['status'],'policy-mismatch')

    def test_postwrite_change_stops_before_native_and_forced_cleanup_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            population=root/'population';population.mkdir()
            (population/'population.json').write_text(json.dumps(dict(entries=[])))
            binaries=root/'binaries';binaries.mkdir()
            (binaries/'flip_graph').write_text('binary')
            state=dict(build_inventory={'flip_graph':b.digest(binaries/'flip_graph')},
                       evaluations=[],installations=[],observations=[])
            def mutate(timeout):
                path=root/'arm/chunks/000/config.json'
                path.write_text(path.read_text()+' ')
                return 0
            def no_guard(*args,**kwargs):
                self.fail('native guard ran after config changed')
            with self.assertRaisesRegex(ValueError,'changed before native'):
                b.fgm3_run_chunk(root,population,binaries,self.protocol,'uniform',7,2,
                    'arm',0,state,900,90,20,clock=lambda:0,sleeper=lambda _:None,
                    guard=no_guard,memory_sample=mutate)
        with self.assertRaisesRegex(RuntimeError,'forcibly terminated'):
            b.fgm3_guarded(['true'],Path('/unused'),lambda *args,**kwargs:
                dict(complete=False,forced_termination=True,cleanup_failure=False),expected='native')
        with self.assertRaisesRegex(RuntimeError,'cleanup is uncertain'):
            b.fgm3_guarded(['true'],Path('/unused'),lambda *args,**kwargs:
                dict(complete=False,forced_termination=False,cleanup_failure=True),expected='native')
        with self.assertRaisesRegex(RuntimeError,'unknown cleanup'):
            b.fgm3_guarded(['true'],Path('/unused'),lambda *args,**kwargs:
                (_ for _ in ()).throw(RuntimeError('guard crashed')),expected='native')
        with tempfile.TemporaryDirectory() as temp:
            record={}
            with self.assertRaisesRegex(ValueError,'sidecars'):
                b.fgm3_guarded(['true'],Path(temp),lambda *args,**kwargs:
                    dict(complete=True,wall_seconds=1,memory=[]),record=record,field='native_guard')
            self.assertEqual(record['native_guard']['status'],'missing-sidecar')
            with self.assertRaisesRegex(RuntimeError,'unknown cleanup'):
                b.fgm3_guarded(['true'],Path(temp),lambda *args,**kwargs:
                    (_ for _ in ()).throw(RuntimeError('no sidecars')),
                    record=record,field='evaluation_export_guard')
            self.assertEqual(record['evaluation_export_guard']['status'],'unknown-cleanup')
            self.assertIsNone(record['evaluation_export_guard']['result_sha256'])

    def test_failed_later_chunk_retains_earlier_endpoint_and_elapsed_attempt(self):
        for failed_stage in ('native','export'):
            with self.subTest(failed_stage=failed_stage),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);population=root/'population';population.mkdir()
                b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
                binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
                clock=Clock()
                launches=[]
                def chunk(output,population,binaries,protocol,selector,seed,workers,arm,index,state,
                          global_deadline,arm_deadline,allowance,**kwargs):
                    launches.append((arm,index))
                    if arm.startswith('qualification/'):
                        clock.sleep(1)
                        if index==0:state['evaluations']=[{} for _ in range(16)]
                        return dict(status='verified',headroom_wait_seconds=0,best_additions=55,
                                    new_installations=[],new_observations=[])
                    if index==0:
                        clock.sleep(10)
                        state['evaluations']=[dict(scheme_id=str(i),additions=55 if i==0 else 58,
                            additions_by_stage=dict(u=10,v=10,w=35 if i==0 else 38),
                            phase_microseconds={},verified_seconds=clock.now()) for i in range(16)]
                        return dict(status='verified',headroom_wait_seconds=0,best_additions=55,
                                    new_installations=[],new_observations=[],verified_seconds=clock.now(),
                                    host_phases_microseconds={},counters={},seed_duplicates=0,pool_evictions=0,
                                    native_guard=dict(wall_seconds=10),
                                    evaluation_export_guard=dict(wall_seconds=0),
                                    observation_export_guard=dict(wall_seconds=0),gpu=dict(gpu_all_seconds=0))
                    record=kwargs['record']
                    record['status']='native-started' if failed_stage=='native' else 'evaluation-export-started'
                    field='native_guard' if failed_stage=='native' else 'evaluation_export_guard'
                    if failed_stage=='export':record['native_guard']=dict(wall_seconds=0)
                    def guarded_failure(argv,destination,*,wired_limit_bytes):
                        clock.sleep(45)
                        return dict(complete=False,forced_termination=failed_stage=='native',
                                    cleanup_failure=False,wall_seconds=45,memory=[],error='injected failure')
                    b.fgm3_guarded(['injected'],Path(output)/arm/'chunks'/f'{index:03d}'/field,
                        guarded_failure,expected=failed_stage,record=record,field=field)
                    self.fail('failed guard returned')
                with (mock.patch.object(b,'fgm3_population',return_value={}),
                      mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                      mock.patch.object(b,'fgm3_run_chunk',side_effect=chunk),
                      mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                    result=b.execute_fgm3(population,binaries,root/'out',clock=clock.now,
                        sleeper=clock.sleep,guard=lambda *args:None,memory_sample=lambda timeout:0)
                self.assertEqual(result['status'],'stopped')
                self.assertFalse(result['complete'])
                self.assertEqual(len(result['arms']),1)
                arm=result['arms'][0]
                self.assertEqual(arm['status'],'incomplete')
                self.assertEqual(arm['elapsed_seconds'],55)
                self.assertEqual(arm['endpoints']['30']['best_additions'],55)
                self.assertNotIn('60',arm['endpoints'])
                self.assertNotIn('90',arm['endpoints'])
                self.assertEqual([row['status'] for row in arm['chunks']],['verified','failed'])
                self.assertEqual(arm['chunks'][1]['elapsed_seconds'],45)
                self.assertEqual(arm['chunks'][1]['failed_stage'],
                                 'native-started' if failed_stage=='native' else 'evaluation-export-started')
                field='native_guard' if failed_stage=='native' else 'evaluation_export_guard'
                self.assertEqual(arm['chunks'][1][field]['wall_seconds'],45)
                self.assertEqual(arm['native_wall_seconds'],55 if failed_stage=='native' else 10)
                self.assertEqual(arm['export_wall_seconds'],45 if failed_stage=='export' else 0)
                self.assertEqual(len(launches),6)
                self.assertEqual(len(result['qualification']['attempts'][0]['chunks']),4)
                persisted=json.loads((root/'out/measurement.json').read_text())
                self.assertEqual(persisted['arms'][0]['chunks'][1]['status'],'failed')

    def test_finalization_detects_time_spent_on_last_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            clock=Clock(8.5)
            data=dict(status='complete',complete=True,qualification={},arms=[])
            def save():
                b.write_json(out/'measurement.json',data)
                clock.sleep(1)
            b.fgm3_finalize(data,out,0,10,clock.now,save)
            self.assertEqual(data['status'],'budget-exceeded')
            self.assertFalse(data['complete'])
            self.assertEqual((out/'measurement.sha256').read_text().strip(),b.digest(out/'measurement.json'))
            self.assertIn('budget-exceeded',(out/'report.md').read_text())

    def test_qualification_time_precedes_measured_900_second_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
            binaries=root/'binaries';binaries.mkdir()
            (binaries/'flip_graph').write_text('binary')
            clock=Clock()
            def checked_population(path):
                clock.sleep(5)
                return {'entries':[]}
            def chunk(output,population,binaries,protocol,selector,seed,workers,arm,index,state,
                      global_deadline,arm_deadline,allowance,**kwargs):
                duration=20 if arm.startswith('qualification/') else 10
                clock.sleep(duration)
                if index==0:
                    # The native chunk owns circuit validation; this mock exercises scheduling only.
                    state['evaluations']=[dict(scheme_id=str(i),additions=58,
                        verified_seconds=clock.now()) for i in range(16)]
                return dict(status='verified',elapsed_seconds=duration,headroom_wait_seconds=0,
                            best_additions=55,new_installations=[],new_observations=[],
                            verified_seconds=clock.now())
            with (mock.patch.object(b,'fgm3_population',side_effect=checked_population),
                  mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                  mock.patch.object(b,'fgm3_run_chunk',side_effect=chunk),
                  mock.patch.object(b,'fgm3_summarize_arm',side_effect=lambda arm,*args:
                      arm.update(endpoints={'90':dict(feedback_installations=0)})),
                  mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                result=b.execute_fgm3(population,binaries,root/'out',clock=clock.now,
                                      sleeper=clock.sleep,guard=lambda *args:None,
                                      memory_sample=lambda timeout:0)
            self.assertTrue(result['complete'])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['arms']),6)
            self.assertGreaterEqual(result['preparation_qualification_elapsed_seconds'],80)
            self.assertLess(result['final_elapsed_seconds'],600)
            self.assertGreater(result['total_elapsed_seconds_before_finalization'],
                               result['final_elapsed_seconds'])

    def test_qualification_allowance_excludes_headroom_but_charges_elapsed_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
            binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
            clock=Clock()
            def chunk(output,population,binaries,protocol,selector,seed,workers,arm,index,state,
                      global_deadline,arm_deadline,allowance,**kwargs):
                if arm.startswith('qualification/'):
                    duration,wait=(45,30) if selector=='uniform' and index==0 else (10,0)
                    clock.sleep(duration)
                    if index==0:state['evaluations']=[{} for _ in range(16)]
                    return dict(status='verified',headroom_wait_seconds=wait,best_additions=55,
                                new_installations=[],new_observations=[])
                if index:
                    return dict(status='unrun',stop_reason='injected no further work',headroom_wait_seconds=0)
                clock.sleep(15)
                state['evaluations']=[{} for _ in range(16)]
                return dict(status='verified',headroom_wait_seconds=5,best_additions=55,
                            new_installations=[],new_observations=[],verified_seconds=clock.now())
            with (mock.patch.object(b,'fgm3_population',return_value={}),
                  mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                  mock.patch.object(b,'fgm3_run_chunk',side_effect=chunk),
                  mock.patch.object(b,'fgm3_summarize_arm',side_effect=lambda arm,*args:
                      arm.update(endpoints={'90':dict(feedback_installations=0)})),
                  mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                result=b.execute_fgm3(population,binaries,root/'out',clock=clock.now,
                    sleeper=clock.sleep,guard=lambda *args:None,memory_sample=lambda timeout:0)
            pilot=result['qualification']
            self.assertEqual(pilot['status'],'qualified')
            self.assertEqual(pilot['chunk_allowance_seconds'],21)
            self.assertEqual(pilot['slower_observed_seconds'],45)
            self.assertEqual(pilot['slower_work_seconds'],15)
            self.assertEqual(pilot['slower_work_total_seconds'],45)
            self.assertEqual(pilot['slower_work_headroom_wait_seconds'],30)
            self.assertEqual(pilot['attempts'][0]['chunks'][0]['elapsed_seconds'],45)
            self.assertEqual(pilot['elapsed_seconds'],75)
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['arms']),6)
            self.assertTrue(all(arm['elapsed_seconds']>=90 and arm['chunks'][0]['headroom_wait_seconds']==5
                                and arm['chunks'][0]['elapsed_seconds']==15 for arm in result['arms']))
            self.assertGreaterEqual(result['total_elapsed_seconds_before_finalization'],75+6*90)
            self.assertIn('45.000 s elapsed minus 30.000 s headroom wait',(root/'out/report.md').read_text())

    def test_qualification_rejects_impossible_elapsed_headroom_pairs(self):
        for elapsed,wait in ((0,0),(45,45),(45,46),(45,-1),(True,0),(45,True),
                             (float('nan'),0),(45,float('inf')),('45',0)):
            with self.subTest(elapsed=elapsed,wait=wait),self.assertRaisesRegex(ValueError,'timing is invalid'):
                b.fgm3_qualification_work_seconds(dict(elapsed_seconds=elapsed,headroom_wait_seconds=wait))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
            binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
            clock=Clock()
            def impossible(*args,**kwargs):
                state=args[9]
                state['evaluations']=[{} for _ in range(16)]
                clock.sleep(45)
                return dict(status='verified',headroom_wait_seconds=45,best_additions=55)
            with (mock.patch.object(b,'fgm3_population',return_value={}),
                  mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                  mock.patch.object(b,'fgm3_run_chunk',side_effect=impossible),
                  mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                result=b.execute_fgm3(population,binaries,root/'out',clock=clock.now,
                    sleeper=clock.sleep,guard=lambda *args:None,memory_sample=lambda timeout:0)
            self.assertEqual(result['qualification']['status'],'failed')
            self.assertIn('timing is invalid',result['error'])
            self.assertEqual(result['arms'],[])

    def test_orchestration_does_not_complete_zero_initialization_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
            binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
            clock=Clock()
            def chunk(output,population,binaries,protocol,selector,seed,workers,arm,index,state,
                      global_deadline,arm_deadline,allowance,**kwargs):
                if arm.startswith('qualification/'):
                    clock.sleep(20)
                    if index==0:state['evaluations']=[{} for _ in range(16)]
                    return dict(status='verified',elapsed_seconds=20,headroom_wait_seconds=0,
                                best_additions=55,new_installations=[],new_observations=[])
                return dict(status='unrun',stop_reason='headroom unavailable')
            with (mock.patch.object(b,'fgm3_population',return_value={}),
                  mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                  mock.patch.object(b,'fgm3_run_chunk',side_effect=chunk),
                  mock.patch.object(b,'fgm3_summarize_arm',side_effect=lambda arm,*args:arm.update(endpoints={})),
                  mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                result=b.execute_fgm3(population,binaries,root/'out',clock=clock.now,
                    sleeper=clock.sleep,guard=lambda *args:None,memory_sample=lambda timeout:0)
            self.assertFalse(result['complete'])
            self.assertEqual(result['finding'],'inconclusive: comparison incomplete')
            self.assertEqual(len(result['arms']),6)
            self.assertTrue(all(arm['status']=='incomplete' and not arm['initial_verified']
                                for arm in result['arms']))

    def test_unusable_allowance_and_exception_statuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
            binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
            clock=Clock()
            def chunk(output,population,binaries,protocol,selector,seed,workers,arm,index,state,
                      global_deadline,arm_deadline,allowance,**kwargs):
                clock.sleep(20)
                if index==0:state['evaluations']=[{} for _ in range(16)]
                return dict(status='verified',elapsed_seconds=20,headroom_wait_seconds=0,
                            best_additions=55,new_installations=[],new_observations=[])
            common=(mock.patch.object(b,'fgm3_population',return_value={}),
                    mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                    mock.patch.object(b,'fgm3_run_chunk',side_effect=chunk),
                    mock.patch.object(b,'host_machine_identity',return_value={'test':True}))
            with ExitStack() as stack:
                for patcher in common:stack.enter_context(patcher)
                stack.enter_context(mock.patch.object(b,'fgm3_chunk_allowance',return_value=90))
                result=b.execute_fgm3(population,binaries,root/'unsuitable',clock=clock.now,
                    sleeper=clock.sleep,guard=lambda *args:None,memory_sample=lambda timeout:0)
            self.assertEqual(result['qualification']['status'],'unsuitable')
            self.assertEqual([a['status'] for a in result['qualification']['attempts']],
                             ['unsuitable','unsuitable'])
            self.assertEqual(result['arms'],[])
            self.assertIn('measured window not started',(root/'unsuitable/report.md').read_text())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[]))
            binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
            with (mock.patch.object(b,'fgm3_population',return_value={}),
                  mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                  mock.patch.object(b,'fgm3_run_chunk',side_effect=RuntimeError('injected')),
                  mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                result=b.execute_fgm3(population,binaries,root/'failed',clock=lambda:0,
                    sleeper=lambda _:None,guard=lambda *args:None,memory_sample=lambda timeout:0)
            self.assertEqual(result['qualification']['status'],'failed')
            self.assertEqual(result['qualification']['attempts'][0]['status'],'failed')
            failed=result['qualification']['attempts'][0]['chunks'][0]
            self.assertEqual((failed['status'],failed['failed_stage']),('failed','scheduled'))
            self.assertEqual(failed['elapsed_seconds'],0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);population=root/'population';population.mkdir()
            b.write_json(population/'population.json',dict(entries=[dict(canonical_id=str(i)) for i in range(16)]))
            binaries=root/'binaries';binaries.mkdir();(binaries/'flip_graph').write_text('binary')
            clock=Clock()
            def interrupted_chunk(output,population,binaries,protocol,selector,seed,workers,arm,index,state,
                                  global_deadline,arm_deadline,allowance,**kwargs):
                if not arm.startswith('qualification/'):raise RuntimeError('interrupted arm')
                clock.sleep(20)
                if index==0:state['evaluations']=[{} for _ in range(16)]
                return dict(status='verified',elapsed_seconds=20,headroom_wait_seconds=0,
                            best_additions=55,new_installations=[],new_observations=[])
            with (mock.patch.object(b,'fgm3_population',return_value={}),
                  mock.patch.object(b,'fgm3_build_files',return_value={'flip_graph':b.digest(binaries/'flip_graph')}),
                  mock.patch.object(b,'fgm3_run_chunk',side_effect=interrupted_chunk),
                  mock.patch.object(b,'host_machine_identity',return_value={'test':True})):
                result=b.execute_fgm3(population,binaries,root/'interrupted',clock=clock.now,
                    sleeper=clock.sleep,guard=lambda *args:None,memory_sample=lambda timeout:0)
            self.assertEqual(result['status'],'stopped')
            self.assertEqual(result['qualification']['status'],'qualified')
            self.assertEqual(result['arms'][0]['status'],'incomplete')
            self.assertIn('interrupted arm',result['arms'][0]['stop_reason'])


if __name__=='__main__':unittest.main()
