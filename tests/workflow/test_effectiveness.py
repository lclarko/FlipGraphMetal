"""Host-only checks for the retained effectiveness panel and bounded schedule."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/workflow'))
import baseline
from verify import verify


PANEL = ROOT / 'benchmarks/workflow/fixtures/fgm1/panel.json'


def measurement_with_cell(root, active):
    shutil.copytree(PANEL.parent, root/'panel')
    panel, protocol = baseline.load_effectiveness_panel(root/'panel/panel.json')
    cells = []
    for entry in panel['entries']:
        for seed in protocol['seeds']:
            for arm in protocol['arms']:
                cell = dict(id=f"{entry['id']}-{seed}-{arm}",panel=entry['id'],seed=seed,
                            arm=arm,status='unrun',reference_costs=[r['claimed_additions'] for r in entry['references']])
                if (entry['id'],seed,arm)==('original',7,'generate'):
                    cell.update(copy.deepcopy(active))
                cells.append(cell)
    measurement = dict(schema='fgm-effectiveness-measurement-v1',complete=False,
                       bindings=dict(panel_sha256=baseline.digest(root/'panel/panel.json')),
                       protocol=protocol,reference_verification=[],cells=cells)
    baseline.write_json(root/'measurement.json',measurement)
    return next(e for e in panel['entries'] if e['id']=='original'), measurement


class EffectivenessTests(unittest.TestCase):
    def test_retained_sources_reconstruct_exact_reference_factors(self):
        panel, protocol = baseline.load_effectiveness_panel(PANEL)
        self.assertEqual([entry['id'] for entry in panel['entries']], list(baseline.FGM1_PANEL))
        self.assertEqual(protocol['endpoints'], [10, 20])
        self.assertEqual(protocol['reducers'], 128)
        self.assertEqual(sum(len(entry['references']) for entry in panel['entries']), 3)
        for entry in panel['entries']:
            factor = json.loads((PANEL.parent/entry['path']).read_text())
            self.assertEqual(verify(factor)['rank'], 23)
            for reference in entry['references']:
                with self.subTest(reference=reference['id']):
                    converted = baseline.reference_circuit(PANEL.parent/reference['source'], reference['id'])
                    self.assertEqual(converted, json.loads((PANEL.parent/reference['path']).read_text()))
                    result = verify(converted, factor)
                    self.assertEqual(result['additions'], reference['claimed_additions'])
                    self.assertEqual(result['additions_by_stage'], reference['claimed_additions_by_stage'])

    def test_reference_rejects_corrupt_output_and_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = json.loads((PANEL.parent/'sources/cn122-58.json').read_text())
            raw['w'][0][0]['value'] *= -1
            path = root/'wrong-w.json'
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                baseline.reference_circuit(path, 'cn122-58')
            gates = json.loads((PANEL.parent/'sources/cn122-55.json').read_text())
            gates['circuits']['U_input_9_to_23']['gates'][0]['slot'] += 1
            path = root/'wrong-gate.json'
            path.write_text(json.dumps(gates))
            with self.assertRaises(ValueError):
                baseline.reference_circuit(path, 'cn122-55')

    def test_panel_rejects_modified_retained_source(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)/'bundle'
            shutil.copytree(PANEL.parent, bundle)
            source = bundle/'sources/sun-56.py'
            source.write_bytes(source.read_bytes()+b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                baseline.load_effectiveness_panel(bundle/'panel.json')

    def test_panel_rejects_false_stage_claim_even_with_intact_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)/'bundle'
            shutil.copytree(PANEL.parent, bundle)
            manifest = json.loads((bundle/'panel.json').read_text())
            reference = manifest['entries'][3]['references'][0]
            reference['claimed_additions_by_stage']['w'] += 1
            (bundle/'panel.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                baseline.load_effectiveness_panel(bundle/'panel.json')

    def test_panel_rejects_missing_reference_roster(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)/'bundle'
            shutil.copytree(PANEL.parent, bundle)
            manifest = json.loads((bundle/'panel.json').read_text())
            manifest['entries'][3]['references'] = []
            (bundle/'panel.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                baseline.load_effectiveness_panel(bundle/'panel.json')

    def test_seed_is_stable_and_changes_with_trial_or_block(self):
        seed = baseline.effectiveness_seed('sun', 19, 'generate', 3)
        expected = int.from_bytes(hashlib.sha256(b'["sun",19,"generate",3]').digest()[:4], 'little') or 1
        self.assertEqual(seed, expected)
        self.assertEqual(seed, baseline.effectiveness_seed('sun', 19, 'generate', 3))
        self.assertNotEqual(seed, baseline.effectiveness_seed('sun', 19, 'generate', 4))
        self.assertNotEqual(seed, baseline.effectiveness_seed('sun', 41, 'generate', 3))

    def test_native_configs_keep_finite_settings_and_owned_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = PANEL.parent/'factors/original.json'
            history = root/'run.journal'
            entry = json.loads(PANEL.read_text())['entries'][0]
            calibration = dict(rounds=37, workers=12)
            reduced = baseline.effectiveness_config(entry, 7, calibration, 'reduce', source, history)
            self.assertEqual(reduced['reduction']['rounds'], 37)
            self.assertEqual(reduced['reduction']['reducers'], baseline.FGM1_PROTOCOL['reducers'])
            self.assertEqual(reduced['reduction']['schemes'], 1)
            self.assertEqual(reduced['reduction']['max_flips'], 0)
            self.assertEqual(reduced['input']['kind'], 'files')
            generated = baseline.effectiveness_config(entry, 7, calibration, 'search', source, history)
            self.assertEqual(generated['policy']['mode'], 'alternatives')
            self.assertEqual(generated['policy']['collection_rank'], 23)
            self.assertEqual(generated['execution']['workers'], 12)
            self.assertEqual(generated['execution']['backend'], 'packed')
            self.assertEqual(generated['history']['path'], str(history))
            self.assertEqual(generated['input']['kind'], 'files')
            history.mkdir()
            resumed = baseline.effectiveness_config(entry, 7, calibration, 'search', source, history)
            self.assertEqual(resumed['input'], dict(kind='resume', journal=str(history)))
            self.assertEqual(resumed['history']['path'], generated['history']['path'])

    def test_capture_accounting_separates_duplicates_redundancy_and_rank(self):
        def row(sid, rank, mandatory, operation, factors='f'):
            return dict(scheme_id=sid, rank=rank, mandatory=mandatory, run_id='r', batch=1,
                        worker=0, control=4, operation=operation, factors_id=factors)
        observations = [row('seed',23,True,0), row('seed',23,False,0),
                        row('new',23,False,1), row('new',23,False,1),
                        row('off-rank',22,False,2)]
        counts = baseline.effectiveness_accounting(observations, 'seed', capture_drops=2)
        self.assertEqual(counts['distinct_discoveries'], 1)
        self.assertEqual(counts['optional_rank23_captures'], 3)
        self.assertEqual(counts['optional_duplicates'], 2)
        self.assertEqual(counts['optional_seed_rediscoveries'], 1)
        self.assertEqual(counts['mandatory_optional_redundancy'], 1)
        self.assertEqual(counts['capture_drops'], 2)
        self.assertAlmostEqual(counts['encounter_drop_rate'], 2/5)

    def test_endpoint_uses_verified_time_and_censors_late_work(self):
        def evaluation(sid, cost, when):
            return dict(scheme_id=sid, additions=cost, additions_by_stage=dict(u=cost,v=0,w=0),
                        verified_seconds=when)
        cell = dict(reference_costs=[55,56], evaluations=[evaluation('a',57,4), evaluation('a',54,12),
                        evaluation('b',53,21)], discoveries=[dict(scheme_id='a',verified_seconds=3),
                        dict(scheme_id='b',verified_seconds=8), dict(scheme_id='late',verified_seconds=21)])
        result = baseline.effectiveness_endpoints(cell)
        self.assertEqual(result['10']['best_additions'], 57)
        self.assertEqual(result['10']['evaluated_candidates'], 1)
        self.assertEqual(result['10']['unevaluated_discoveries'], 1)
        self.assertFalse(result['10']['reference_attainment']['56'])
        self.assertEqual(result['20']['best_additions'], 54)
        self.assertEqual(result['20']['cost_histogram'], {'54':1})
        self.assertTrue(result['20']['reference_attainment']['55'])
        self.assertEqual(result['20']['distinct_discoveries'], 2)

    def test_fake_clock_enforces_global_reserve_and_arm_deadline(self):
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            run = baseline.EffectivenessRun(Path(temp), Path(temp), clock=lambda: now[0])
            run.started = 0.
            run.deadline = 900.
            now[0] = 849.9
            run.check_launch(stop=850.)
            now[0] = 850.
            with self.assertRaises(baseline.EffectivenessStop):
                run.check_launch(stop=850.)
            now[0] = 849.
            with self.assertRaises(baseline.EffectivenessStop):
                run.check_launch(stop=20.)
            now[0] = 851.
            with self.assertRaisesRegex(baseline.EffectivenessStop, 'cleanup reserve'):
                run.check_launch()

    def test_calibration_reuse_rejects_changed_bindings_and_slow_pilots(self):
        bindings = dict(panel='p', executable='e',
                        protocol_sha256=baseline.content_hash(baseline.FGM1_PROTOCOL))
        rows = [dict(kind=kind, setting=16, panel=panel, complete=True, seconds=limit,
                     steps=[dict(path=f'{kind}-{panel}.json',sha256='pending')])
                for kind, limit in (('reduce', 1), ('search', 5)) for panel in baseline.FGM1_PANEL]
        calibration = dict(schema='fgm-effectiveness-calibration-v1', complete=True,
                           bindings=bindings, settings=dict(rounds=16, workers=16), pilots=rows)
        self.assertTrue(baseline.calibration_valid(calibration, bindings))
        self.assertFalse(baseline.calibration_valid(calibration, dict(panel='different', executable='e')))
        previous = copy.deepcopy(baseline.FGM1_PROTOCOL)
        previous['reducers'] *= 2
        self.assertNotEqual(baseline.content_hash(previous), bindings['protocol_sha256'])
        changed_bindings = dict(bindings, protocol_sha256=baseline.content_hash(previous))
        self.assertFalse(baseline.calibration_valid(calibration, changed_bindings))
        changed = copy.deepcopy(calibration)
        changed['settings']['workers'] = 12
        self.assertFalse(baseline.calibration_valid(changed, bindings))
        changed = copy.deepcopy(calibration)
        changed['pilots'][-1]['seconds'] = 5.01
        self.assertFalse(baseline.calibration_valid(changed, bindings))
        changed = copy.deepcopy(calibration)
        changed['pilots'][-1]['steps'] = []
        self.assertFalse(baseline.calibration_valid(changed, bindings))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for row in rows:
                step = row['steps'][0]
                path = root/step['path']
                path.write_text(json.dumps(dict(complete=True)))
                step['sha256'] = baseline.digest(path)
            # A complete flag alone is not a retained native verification proof.
            self.assertFalse(baseline.calibration_valid(calibration, bindings, root))
            (root/rows[0]['steps'][0]['path']).write_text(json.dumps(dict(complete=False)))
            self.assertFalse(baseline.calibration_valid(calibration, bindings, root))

    def test_calibration_halves_only_failing_quantum(self):
        panel, _ = baseline.load_effectiveness_panel(PANEL)
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            run = baseline.EffectivenessRun(Path(temp), Path(temp), clock=lambda: now[0])
            run.started = 0.
            run.deadline = 10000.
            calls = []
            def invoke(config, label, stop):
                run.check_launch(stop)
                kind = config['operation']
                setting = config['reduction']['rounds'] if kind == 'reduce' else config['execution']['workers']
                now[0] += (1.2 if setting == 32 else .8) if kind == 'reduce' else (6. if setting == 32 else 4.)
                calls.append((kind, setting, label))
                return dict(evidence_path='unused', evidence_sha256='unused', observations=[])
            run.invoke = invoke
            saved = []
            calibration = baseline.calibrate_effectiveness(run, panel['entries'], PANEL.parent,
                                                            {'test':'binding'}, lambda value:saved.append(copy.deepcopy(value)))
            self.assertEqual(calibration['settings'], dict(rounds=16, workers=16))
            self.assertTrue(calibration['complete'])
            self.assertTrue(baseline.calibration_valid(calibration, {'test':'binding'}))
            self.assertEqual(len([item for item in calls if item[:2] == ('reduce', 32)]), 5)
            self.assertEqual(len([item for item in calls if item[:2] == ('search', 32)]), 5)
            self.assertGreater(len(saved), len(calls))

    def test_legacy_reduction_evidence_uses_retained_reducer_population(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root/'step'
            attempt.mkdir()
            source = PANEL.parent/'factors/sun.json'
            circuit_source = PANEL.parent/'circuits/sun-56.json'
            shutil.copyfile(source,attempt/'input.json')
            shutil.copyfile(circuit_source,attempt/'receipt.json.circuits.jsonl')
            circuit = json.loads(circuit_source.read_text())
            result = verify(circuit)
            factors = dict(dimensions=circuit['n'],rank=circuit['m'],domain='ZT',orientation='cyclic-w',
                           **dict(zip('uvw',baseline.circuit_factors(circuit))))
            counts = {k:baseline.reconstruct(circuit[k],circuit[k+'_fresh'],23 if k=='w' else 9,False)[1]
                      for k in 'uvw'}
            evaluation = dict(scheme_id=baseline.host_oracle().identity(factors),
                              factors_id=baseline.host_oracle().identity(factors,False),
                              additions=result['additions'],additions_by_stage=counts,
                              circuit_sha256=baseline.digest(attempt/'receipt.json.circuits.jsonl'))
            config = dict(input=dict(kind='files'),reduction=dict(rounds=16,reducers=256,
                          max_flips=0,schemes=1,target_additions=0,no_improvements=16))
            baseline.write_json(attempt/'config.json',config)
            receipt = dict(status='complete',configuration=dict(operation='reduce'),
                           configuration_sha256=baseline.digest(attempt/'config.json'),
                           presentations=[dict(source_sha256=baseline.digest(attempt/'input.json'))],
                           counters={},results=[dict(verified_circuit_additions_by_stage=counts)])
            baseline.write_json(attempt/'receipt.json',receipt)
            guard = attempt/'guard-00'
            guard.mkdir()
            command = ['/test/additions_reducer']
            baseline.write_json(guard/'result.json',dict(complete=True,argv=['/usr/bin/time','-l','-p',*command]))
            (guard/'run.log').write_text('real 0.01\n123 maximum resident set size\n')
            step = dict(complete=True,operation='reduce',workflow_seconds=1.,started_seconds=0.,
                        verified_at_seconds=.5,native_receipt_sha256=baseline.digest(attempt/'receipt.json'),
                        commands=[command],guards=[dict(path='guard-00',result_sha256=baseline.digest(guard/'result.json'),
                            log_sha256=baseline.digest(guard/'run.log'))],gpu={},counters={},host_phases={},
                        evaluation=evaluation)
            baseline.write_json(attempt/'step.json',step)
            reference = dict(path='step/step.json',sha256=baseline.digest(attempt/'step.json'))
            # This fixture circuit retains the source gauge; isolate the reducer-population check.
            with mock.patch.object(baseline,'effective_factor_id',return_value=evaluation['factors_id']), \
                 mock.patch.object(baseline,'native_dispatch_evidence',return_value={}), \
                 mock.patch.object(baseline,'native_verify_circuits'):
                accepted = baseline.checked_effectiveness_step(root,reference,expected_reducers=256)
                self.assertEqual(accepted['evaluation'],evaluation)
                with self.assertRaisesRegex(ValueError,'reduction quantum'):
                    baseline.checked_effectiveness_step(root,reference)

    def test_legacy_v1_v2_summaries_pass_retained_reducer_count(self):
        for protocol in (baseline.FGM1_PREVIOUS_PROTOCOL,baseline.FGM1_HEADROOM_PROTOCOL):
            with self.subTest(version=protocol['version']), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                entry, measurement = measurement_with_cell(root,{})
                baseline.write_json(root/'panel/protocol.json',protocol)
                panel = json.loads((root/'panel/panel.json').read_text())
                panel['artifacts']['protocol.json'] = baseline.digest(root/'panel/protocol.json')
                baseline.write_json(root/'panel/panel.json',panel)
                source = json.loads((root/'panel'/entry['path']).read_text())
                evaluation = dict(scheme_id=entry['scheme_id'],factors_id=baseline.effective_factor_id(source),
                                  additions=60,additions_by_stage=dict(u=20,v=20,w=20),verified_seconds=.5)
                cell = next(item for item in measurement['cells'] if item['id']=='original-7-generate')
                cell.update(status='incomplete',started_seconds=0.,workflow_seconds=1.,
                            evaluations=[evaluation],discoveries=[],observations=[],counters={},pending_scheme_ids=[],
                            steps=[dict(path='legacy-reduction',sha256='checked-by-mock')])
                cell['endpoints'] = baseline.effectiveness_endpoints(cell)
                cell['capture_accounting'] = baseline.effectiveness_accounting([],entry['scheme_id'])
                measurement['protocol'] = protocol
                measurement['bindings']['panel_sha256'] = baseline.digest(root/'panel/panel.json')
                baseline.write_json(root/'measurement.json',measurement)
                reduced = dict(complete=True,operation='reduce',started_seconds=0.,workflow_seconds=1.,
                               verified_at_seconds=.5,evaluation={k:v for k,v in evaluation.items() if k!='verified_seconds'})
                with mock.patch.object(baseline,'checked_effectiveness_step',return_value=reduced) as checked:
                    summary = baseline.summarize_effectiveness(root/'measurement.json')
                self.assertEqual(summary['protocol_version'],protocol['version'])
                self.assertFalse(summary['complete'])
                checked.assert_called_once_with(root.resolve(),cell['steps'][0],measurement['bindings'],
                                                expected_reducers=protocol['reducers'])

    def test_generation_queue_is_cost_blind_and_cumulative_captures_count_once(self):
        panel, _ = baseline.load_effectiveness_panel(PANEL)
        entries = {entry['id']:entry for entry in panel['entries']}
        original, laderman, smirnov = (entries[name] for name in ('original','laderman','smirnov'))
        def observation(entry, slot):
            factors = json.loads((PANEL.parent/entry['path']).read_text())
            scheme = dict(schema='fgm-scheme-v1', dimensions=factors['n'], rank=factors['m'],
                          domain='ZT', orientation='cyclic-w', **{k:factors[k] for k in 'uvw'})
            return dict(run_id='r', sequence=2, batch=1, worker=0, slot=slot, mandatory=slot==0,
                        control=4, operation=slot, scheme_id=entry['scheme_id'],
                        factors_id=baseline.host_oracle().identity(scheme, False), rank=23,
                        domain='ZT', scheme=scheme)
        cumulative = [observation(laderman, 0), observation(smirnov, 1)]
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            run = baseline.EffectivenessRun(Path(temp), Path(temp), clock=lambda: now[0])
            run.started = 0.
            run.deadline = 1000.
            calls = []
            def invoke(config, label, stop):
                run.check_launch(stop)
                now[0] += 3.
                calls.append((config['operation'], config['input'], config.get('history', {}).get('path')))
                if config['operation'] == 'search':
                    return dict(observations=copy.deepcopy(cumulative), counters=dict(capture_drops=1),
                                evidence_path='unused', evidence_sha256='unused',
                                verified_at_seconds=now[0])
                source = Path(config['input']['files'][0]['path'])
                data = json.loads(source.read_text())
                sid = baseline.host_oracle().identity(dict(dimensions=data['n'], rank=data['m'],
                    domain='ZT', orientation='cyclic-w', **{k:data[k] for k in 'uvw'}))
                costs = {original['scheme_id']:70, laderman['scheme_id']:80, smirnov['scheme_id']:50}
                return dict(evaluation=dict(scheme_id=sid, additions=costs[sid],
                            additions_by_stage=dict(u=costs[sid],v=0,w=0)),
                            evidence_path='unused', evidence_sha256='unused',
                            verified_at_seconds=now[0])
            run.invoke = invoke
            cell = dict(id='original-7-generate', panel='original', seed=7, arm='generate', status='planned',
                        reference_costs=[55,56])
            baseline.run_effectiveness_cell(run, cell, original, PANEL.parent,
                                            dict(rounds=16,workers=16), lambda:None)
            self.assertEqual(cell['status'], 'complete')
            self.assertEqual([item['scheme_id'] for item in cell['evaluations'][:3]],
                             [original['scheme_id'], laderman['scheme_id'], smirnov['scheme_id']])
            self.assertEqual(len(cell['discoveries']), 2)
            self.assertEqual(len(cell['observations']), 2)
            self.assertGreaterEqual(sum(operation=='search' for operation,_,_ in calls), 2)
            search_histories = {history for operation,_,history in calls if operation=='search'}
            self.assertEqual(len(search_histories), 1)
            self.assertEqual(cell['capture_accounting']['capture_drops'],
                             sum(operation=='search' for operation,_,_ in calls))

    def test_empty_discoveries_keep_searching_and_late_results_are_censored(self):
        panel, _ = baseline.load_effectiveness_panel(PANEL)
        entry = panel['entries'][0]
        late_entry = panel['entries'][1]
        late_factors = json.loads((PANEL.parent/late_entry['path']).read_text())
        late_scheme = dict(schema='fgm-scheme-v1', dimensions=late_factors['n'], rank=23,
                           domain='ZT', orientation='cyclic-w', **{k:late_factors[k] for k in 'uvw'})
        late_observation = dict(run_id='r', sequence=2, batch=1, worker=0, slot=0,
                                mandatory=True, control=1, operation=0, rank=23, domain='ZT',
                                scheme_id=late_entry['scheme_id'],
                                factors_id=baseline.host_oracle().identity(late_scheme,False),
                                scheme=late_scheme)
        for duration, late in ((2.,False),(11.,True)):
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as temp:
                now = [0.]
                run = baseline.EffectivenessRun(Path(temp), Path(temp), clock=lambda: now[0])
                run.started = 0.
                run.deadline = 1000.
                calls = []
                def invoke(config, label, stop):
                    run.check_launch(stop)
                    now[0] += duration
                    calls.append(config['operation'])
                    if config['operation']=='reduce':
                        return dict(evaluation=dict(scheme_id=entry['scheme_id'],additions=60,
                                    additions_by_stage=dict(u=20,v=20,w=20)),
                                    evidence_path='unused',evidence_sha256='unused',
                                    verified_at_seconds=now[0])
                    return dict(observations=[late_observation] if late else [],counters={},
                                evidence_path='unused',evidence_sha256='unused',
                                verified_at_seconds=now[0])
                run.invoke = invoke
                cell = dict(id='empty-generate', panel=entry['id'], seed=7, arm='generate',
                            status='planned', reference_costs=[55,56])
                baseline.run_effectiveness_cell(run,cell,entry,PANEL.parent,
                                                dict(rounds=16,workers=16),lambda:None)
                self.assertEqual(cell['status'],'complete')
                self.assertGreaterEqual(calls.count('search'),1 if late else 2)
                if late:
                    self.assertEqual(len(cell['discoveries']),1)
                    self.assertEqual(cell['endpoints']['20']['distinct_discoveries'],0)
                    self.assertEqual(cell['pending_scheme_ids'],[late_entry['scheme_id']])
                    self.assertIsNone(cell['endpoints']['10']['best_additions'])
                    self.assertEqual(cell['endpoints']['20']['best_additions'],60)
                else:
                    self.assertEqual(cell['discoveries'],[])

    def test_global_reserve_stops_midchain_without_completion_credit(self):
        panel, _ = baseline.load_effectiveness_panel(PANEL)
        entry = panel['entries'][0]
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            run = baseline.EffectivenessRun(Path(temp),Path(temp),clock=lambda:now[0])
            run.started = 0.
            run.deadline = 56.
            calls = []
            def invoke(config,label,stop):
                run.check_launch(stop)
                now[0] += 4.
                calls.append(config['operation'])
                if config['operation']=='reduce':
                    return dict(evaluation=dict(scheme_id=entry['scheme_id'],additions=60,
                                additions_by_stage=dict(u=20,v=20,w=20)),evidence_path='unused',
                                evidence_sha256='unused',verified_at_seconds=now[0])
                return dict(observations=[],counters={},evidence_path='unused',
                            evidence_sha256='unused',verified_at_seconds=now[0])
            run.invoke = invoke
            cell = dict(id='global-stop',panel=entry['id'],seed=7,arm='generate',
                        status='planned',reference_costs=[55,56])
            baseline.run_effectiveness_cell(run,cell,entry,PANEL.parent,dict(rounds=16,workers=16),lambda:None)
            self.assertEqual(calls,['reduce','search'])
            self.assertEqual(cell['status'],'incomplete')
            self.assertIn('cleanup reserve',cell['stop_reason'])

    def test_endpoint_uses_recorded_verification_time_before_host_return(self):
        panel, _ = baseline.load_effectiveness_panel(PANEL)
        entry = panel['entries'][0]
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            run = baseline.EffectivenessRun(Path(temp),Path(temp),clock=lambda:now[0])
            run.started = 0.
            run.deadline = 1000.
            def invoke(config,label,stop):
                run.check_launch(stop)
                now[0] += 12.
                return dict(evaluation=dict(scheme_id=entry['scheme_id'],additions=54,
                            additions_by_stage=dict(u=18,v=18,w=18)),
                            verified_at_seconds=9. if now[0]==12. else now[0],
                            evidence_path='unused',evidence_sha256='unused')
            run.invoke = invoke
            cell = dict(id='timing-fixed',panel=entry['id'],seed=7,arm='fixed',
                        status='planned',reference_costs=[55,56])
            baseline.run_effectiveness_cell(run,cell,entry,PANEL.parent,dict(rounds=16,workers=16),lambda:None)
            self.assertEqual(cell['evaluations'][0]['verified_seconds'],9.)
            self.assertEqual(cell['endpoints']['10']['best_additions'],54)

    def test_failed_search_export_retains_counters_and_summary_replays_partial_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'blocks').mkdir()
            binaries = root/'binaries'
            binaries.mkdir()
            (binaries/'flip_graph').write_bytes(b'fake native executable')
            now = [0.]
            run = baseline.EffectivenessRun(root,binaries,clock=lambda:now[0])
            run.started = 0.
            run.deadline = 1000.
            config = dict(operation='search',input=dict(kind='resume'),
                          history=dict(path=str(root/'history'),transaction_bytes=8388608))
            calls = [0]
            def command(argv, attempt, record, stop=None):
                calls[0] += 1
                if calls[0] == 1:
                    receipt = dict(status='complete',execution_started=True,
                        configuration_sha256=baseline.digest(attempt/'config.json'),
                        executable_sha256=baseline.digest(binaries/'flip_graph'),
                        configuration=dict(operation='search'),
                        counters=dict(discoveries_current_run=3))
                    baseline.write_json(attempt/'receipt.json',receipt)
                    return ''
                raise baseline.EffectivenessStop('block or arm deadline during export')
            run.command = command
            with self.assertRaises(baseline.EffectivenessStop) as caught:
                run.invoke(config,'failed-export',stop=20.)
            partial = caught.exception.effectiveness_record
            self.assertEqual(partial['counters']['discoveries_current_run'],3)
            self.assertTrue((root/partial['evidence_path']).is_file())
            checked = baseline.checked_effectiveness_step(root,dict(path=partial['evidence_path'],
                                                                   sha256=partial['evidence_sha256']))
            self.assertFalse(checked['complete'])
            self.assertEqual(checked['counters']['discoveries_current_run'],3)

            entry, _ = measurement_with_cell(root, {})
            source = json.loads((root/'panel'/entry['path']).read_text())
            evaluation = dict(scheme_id=entry['scheme_id'],factors_id=baseline.effective_factor_id(source),additions=60,
                              additions_by_stage=dict(u=20,v=20,w=20))
            reduction = dict(operation='reduce',complete=True,started_seconds=0.,workflow_seconds=1.,
                             verified_at_seconds=1.,evaluation=evaluation)
            steps = [0]
            def invoke(config,label,stop):
                steps[0] += 1
                if steps[0] == 1:
                    now[0] = 1.
                    return dict(evaluation=copy.deepcopy(evaluation),verified_at_seconds=1.,
                                evidence_path='mock-reduce',evidence_sha256='mock')
                raise caught.exception
            run.invoke = invoke
            cell = dict(id='original-7-generate',panel='original',seed=7,arm='generate',
                        status='unrun',reference_costs=[])
            baseline.run_effectiveness_cell(run,cell,entry,root/'panel',dict(rounds=16,workers=16),lambda:None)
            self.assertEqual(cell['status'],'incomplete')
            self.assertEqual(cell['unexported_native_discoveries'],3)
            self.assertEqual(cell['steps'][-1]['path'],partial['evidence_path'])
            measurement_with_cell_data = json.loads((root/'measurement.json').read_text())
            for item in measurement_with_cell_data['cells']:
                if item['id']==cell['id']:
                    item.update(cell)
            baseline.write_json(root/'measurement.json',measurement_with_cell_data)
            original = baseline.checked_effectiveness_step
            def checked(base, reference, bindings=None, expected_reducers=None):
                return (copy.deepcopy(reduction) if reference['path']=='mock-reduce' else
                        original(base,reference,bindings,expected_reducers))
            with mock.patch.object(baseline,'checked_effectiveness_step',side_effect=checked):
                summary = baseline.summarize_effectiveness(root/'measurement.json')
            self.assertFalse(summary['complete'])
            self.assertEqual(next(item for item in summary['cells'] if item['id']==cell['id'])
                             ['unexported_native_discoveries'],3)

    def test_summary_rejects_tampered_stage_time_and_discoveries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry, _ = measurement_with_cell(root,{})
            source = json.loads((root/'panel'/entry['path']).read_text())
            evaluation = dict(scheme_id=entry['scheme_id'],factors_id=baseline.effective_factor_id(source),additions=60,
                              additions_by_stage=dict(u=20,v=20,w=20),verified_seconds=1.)
            cell = dict(status='incomplete',started_seconds=0.,workflow_seconds=1.,
                        evaluations=[evaluation],discoveries=[],observations=[],
                        steps=[dict(path='mock-reduce',sha256='mock')],counters={},pending_scheme_ids=[])
            cell['endpoints'] = baseline.effectiveness_endpoints(dict(cell,reference_costs=[]))
            cell['capture_accounting'] = baseline.effectiveness_accounting([],entry['scheme_id'])
            document = json.loads((root/'measurement.json').read_text())
            active = next(item for item in document['cells'] if item['id']=='original-7-generate')
            active.update(cell)
            baseline.write_json(root/'measurement.json',document)
            step = dict(operation='reduce',complete=True,started_seconds=0.,workflow_seconds=1.,
                        verified_at_seconds=1.,evaluation={k:v for k,v in evaluation.items() if k!='verified_seconds'})
            with mock.patch.object(baseline,'checked_effectiveness_step',return_value=step):
                self.assertFalse(baseline.summarize_effectiveness(root/'measurement.json')['complete'])
                for field, value in (('additions_by_stage',dict(u=21,v=20,w=20)),
                                     ('verified_seconds',2.)):
                    changed = copy.deepcopy(document)
                    target = next(item for item in changed['cells'] if item['id']=='original-7-generate')
                    target['evaluations'][0][field] = value
                    baseline.write_json(root/'measurement.json',changed)
                    with self.assertRaises(ValueError):
                        baseline.summarize_effectiveness(root/'measurement.json')
                changed = copy.deepcopy(document)
                target = next(item for item in changed['cells'] if item['id']=='original-7-generate')
                target['discoveries'] = [dict(scheme_id='invented',factors_id='invented',verified_seconds=1.)]
                baseline.write_json(root/'measurement.json',changed)
                with self.assertRaises(ValueError):
                    baseline.summarize_effectiveness(root/'measurement.json')

                for status in ('complete', 'incomplete', 'unrun'):
                    with self.subTest(missing_evaluations_status=status):
                        changed = copy.deepcopy(document)
                        target = next(item for item in changed['cells'] if item['id']=='original-7-generate')
                        target['status'] = status
                        del target['evaluations']
                        baseline.write_json(root/'measurement.json', changed)
                        with self.assertRaisesRegex(ValueError, 'missing evaluations'):
                            baseline.summarize_effectiveness(root/'measurement.json')

    def test_summary_accepts_censored_step_without_command_after_arm_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry, _ = measurement_with_cell(root,{})
            source = json.loads((root/'panel'/entry['path']).read_text())
            evaluation = dict(scheme_id=entry['scheme_id'],factors_id=baseline.effective_factor_id(source),
                              additions=60,additions_by_stage=dict(u=20,v=20,w=20),verified_seconds=1.)
            no_launch = root/'blocks/no-launch'
            no_launch.mkdir(parents=True)
            baseline.write_json(no_launch/'step.json',dict(complete=False,operation='search',
                started_seconds=20.01,workflow_seconds=0.,commands=[],error='block or arm deadline'))
            cell = dict(status='incomplete',started_seconds=0.,workflow_seconds=20.01,
                        evaluations=[evaluation],discoveries=[],observations=[],counters={},
                        pending_scheme_ids=[],steps=[dict(path='mock-reduce',sha256='mock'),
                            dict(path='blocks/no-launch/step.json',sha256=baseline.digest(no_launch/'step.json'))])
            cell['endpoints'] = baseline.effectiveness_endpoints(dict(cell,reference_costs=[]))
            cell['capture_accounting'] = baseline.effectiveness_accounting([],entry['scheme_id'])
            document = json.loads((root/'measurement.json').read_text())
            next(item for item in document['cells'] if item['id']=='original-7-generate').update(cell)
            baseline.write_json(root/'measurement.json',document)
            reduction = dict(operation='reduce',complete=True,started_seconds=0.,workflow_seconds=1.,
                             verified_at_seconds=1.,evaluation={k:v for k,v in evaluation.items() if k!='verified_seconds'})
            original = baseline.checked_effectiveness_step
            def checked(base,reference,bindings=None,expected_reducers=None):
                return (copy.deepcopy(reduction) if reference['path']=='mock-reduce' else
                        original(base,reference,bindings,expected_reducers))
            with mock.patch.object(baseline,'checked_effectiveness_step',side_effect=checked):
                summary = baseline.summarize_effectiveness(root/'measurement.json')
            self.assertFalse(summary['complete'])
            self.assertEqual(next(row for row in summary['cells'] if row['id']=='original-7-generate')
                             ['endpoints']['10']['best_additions'],60)

    def test_wired_headroom_wait_charges_time_and_launches_after_release(self):
        threshold = baseline.LIMIT-baseline.FGM1_PROTOCOL['launch_wired_reserve_bytes']
        now = [0.]
        sleeps = []
        def sleeper(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root/'attempt'
            attempt.mkdir()
            run = baseline.EffectivenessRun(root,root,clock=lambda:now[0],sleeper=sleeper)
            run.started = 0.
            run.deadline = 900.
            record = dict(commands=[])
            def guarded(argv, folder):
                folder.mkdir()
                baseline.write_json(folder/'result.json',dict(complete=True))
                (folder/'run.log').write_text('real 0.01\n123 maximum resident set size\n')
                return dict(complete=True,memory=[])
            command = [str(root/'flip_graph')]
            with mock.patch.object(baseline,'wired_memory',side_effect=[threshold+1,threshold+1,threshold]), \
                 mock.patch.object(baseline,'guarded_run',side_effect=guarded):
                log = run.command(command,attempt,record,stop=1.)
            self.assertIn('real 0.01',log)
            self.assertEqual(record['commands'],[command])
            self.assertAlmostEqual(record['headroom_wait_seconds'],.05)
            self.assertEqual(sleeps,[.025,.025])

    def test_host_command_skips_wired_headroom_wait(self):
        now = [0.]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root/'attempt'
            attempt.mkdir()
            run = baseline.EffectivenessRun(root,root,clock=lambda:now[0],
                sleeper=lambda seconds:now.__setitem__(0,now[0]+seconds))
            run.started = 0.
            run.deadline = 900.
            record = dict(commands=[])
            command = [str(root/'scheme_tool')]
            def guarded(argv, folder):
                folder.mkdir()
                baseline.write_json(folder/'result.json',dict(complete=True))
                (folder/'run.log').write_text('real 0.01\n123 maximum resident set size\n')
                return dict(complete=True,memory=[])
            with mock.patch.object(baseline,'wired_memory',side_effect=AssertionError('host command polled GPU memory')), \
                 mock.patch.object(baseline,'guarded_run',side_effect=guarded) as launch:
                log = run.command(command,attempt,record,stop=1.)
                launch.assert_called_once()
            self.assertIn('real 0.01',log)
            self.assertEqual(record['commands'],[command])
            self.assertEqual(record['headroom_wait_seconds'],0.)
            self.assertEqual(now[0],0.)

    def test_headroom_wait_stops_before_arm_or_global_deadline_without_launch(self):
        threshold = baseline.LIMIT-baseline.FGM1_PROTOCOL['launch_wired_reserve_bytes']
        for deadline, stop, message in ((900.,.05,'block or arm deadline'),
                                        (50.04,None,'cleanup reserve')):
            with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as temp:
                now = [0.]
                run = baseline.EffectivenessRun(Path(temp),Path(temp),clock=lambda:now[0],
                    sleeper=lambda seconds:now.__setitem__(0,now[0]+seconds))
                run.started = 0.
                run.deadline = deadline
                attempt = Path(temp)/'attempt'
                attempt.mkdir()
                record = dict(commands=[])
                with mock.patch.object(baseline,'wired_memory',return_value=threshold+1), \
                     mock.patch.object(baseline,'guarded_run') as guarded:
                    with self.assertRaisesRegex(baseline.EffectivenessStop,message):
                        run.command([str(Path(temp)/'additions_reducer')],attempt,record,stop=stop)
                    guarded.assert_not_called()
                self.assertEqual(record['commands'],[])
                self.assertGreaterEqual(record['headroom_wait_seconds'],.05)


if __name__ == '__main__':
    unittest.main()
