"""Host-only native policy-settings contracts; no search dispatch is available."""
import os
import json
from pathlib import Path
import subprocess
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def settings():
    return dict(schema='fgm-controlled-config-v1', policy='controlled-v1',
                mode='alternatives', domain='ZT', seed=0, dimensions=[3, 3, 3],
                collection_rank=23, excursion=2, interval_min=4, interval_max=4,
                reduction_q=0, stagnation_limit=10, flip_budget=0,
                control_budget=0, optional_quota=0, target_rank=None)


class RunConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ.get('FGM_RUN_CONFIG_DRIVER',
                                         ROOT/'build/workflow/test_run_config'))
        if not cls.binary.is_file():
            raise RuntimeError('build native test_run_config before running its tests')

    def invoke(self, value, domain='ZT', helper=None):
        command = [str(self.binary), domain] + ([] if helper is None else [helper])
        return subprocess.run(command, input=value if isinstance(value, str) else json.dumps(value),
                              capture_output=True, text=True, timeout=5)

    def accepted(self, value, domain='ZT'):
        result = self.invoke(value, domain)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_resolved_roundtrip_and_explicit_endpoints(self):
        config = settings()
        result = self.accepted(config)
        self.assertEqual(result['resolved'], dict(config, proposal_limit=64))
        self.assertEqual(result['interval_span'], 1)
        self.assertEqual(result['ceiling'], 25)
        self.assertTrue(result['below_ceiling_allowed'])
        self.assertFalse(result['at_ceiling_allowed'])
        self.assertEqual(self.accepted(result['resolved']), result)
        config.update(seed=(1 << 32)-1, reduction_q=(1 << 32)-1, domain='F2', proposal_limit=1, target_rank=0)
        self.assertEqual(self.accepted(config, 'F2')['resolved'], config)

    def test_anchors_and_ceiling_components(self):
        for dimensions, anchor, excursion, expected, eligible in (
                ([3, 3, 3], 23, 1, 24, True),
                ([3, 3, 3], 28, 20, 27, True),
                ([8, 8, 8], 343, 100, 350, True),
                ([1, 32, 1], 32, 0, 32, False),
                ([3, 3, 3], 351, 0, 27, False)):
            config = settings()
            config.update(mode='rank-reduction', dimensions=dimensions, stage_rank=anchor, excursion=excursion)
            del config['collection_rank']
            result = self.accepted(config)
            self.assertEqual(result['ceiling'], expected)
            self.assertEqual(result['anchor_representation_eligible'], eligible)

    def test_rejects_unknown_conflicting_missing_and_malformed_fields(self):
        changes = [dict(policy='legacy-v1'), dict(schema='unknown'), dict(domain='Z3'),
                   dict(mode='meta'), dict(extra=1), dict(stage_rank=23),
                   dict(seed=True), dict(seed=1.0), dict(seed=1 << 32), dict(reduction_q=-1),
                   dict(reduction_q=1 << 32), dict(proposal_limit=0), dict(proposal_limit=1 << 32),
                   dict(target_rank=-1), dict(target_rank=True), dict(interval_min=0),
                   dict(interval_max=3), dict(stagnation_limit=0), dict(collection_rank=0),
                   dict(dimensions=[3, 3]), dict(dimensions=[0, 3, 3]),
                   dict(flip_budget=1 << 63)]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(self.invoke(dict(settings(), **change)).returncode, 0)
        for field in settings():
            config = settings()
            del config[field]
            with self.subTest(missing=field):
                self.assertNotEqual(self.invoke(config).returncode, 0)
        self.assertNotEqual(self.invoke(settings(), 'F2').returncode, 0)
        duplicate = json.dumps(settings())[:-1]+', "seed": 1}'
        self.assertNotEqual(self.invoke(duplicate).returncode, 0)

    def test_checked_arithmetic_and_large_parse_boundaries(self):
        maximum = (1 << 64)-1
        for helper, a, b, expected in [('add', maximum-1, 1, maximum),
                                      ('multiply', maximum, 1, maximum),
                                      ('multiply', maximum, 0, 0)]:
            result = self.invoke({'a': str(a), 'b': str(b)}, helper=helper)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(int(result.stdout), expected)
        for helper, a, b in [('add', maximum, 1), ('multiply', maximum, 2)]:
            self.assertNotEqual(self.invoke({'a': str(a), 'b': str(b)}, helper=helper).returncode, 0)
        config = dict(settings(), dimensions=[(1 << 63)-1, 3, 3])
        self.assertNotEqual(self.invoke(config).returncode, 0)
        config = dict(settings(), interval_min=1, interval_max=(1 << 63)-1)
        self.assertEqual(self.accepted(config)['interval_span'], (1 << 63)-1)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = Path(os.environ.get('FGM_EXECUTION_DRIVER', ROOT/'build/workflow/test_execution'))
        self.scheme = dict(dimensions=[1,1,1], rank=1, domain='ZT', orientation='cyclic-w',
                           u=[[1]], v=[[1]], w=[[1]])
        (self.root/'scheme.json').write_text(json.dumps(self.scheme))
        self.config = dict(schema='fgm-run-v1', operation='search',
            policy=dict(settings(), dimensions=[1,1,1], collection_rank=1),
            input=dict(kind='files', files=[dict(path='scheme.json', format='json')]),
            execution=dict(workers=2, batch_steps=3, block_size=32, backend='general', memory_bytes=16*1024*1024),
            output='receipt.json')

    def invoke(self, config=None, extra=()):
        (self.root/'run.json').write_text(json.dumps(self.config if config is None else config))
        return subprocess.run([str(self.binary), '--run-config', str(self.root/'run.json'),
                               '--validate-only', *extra], capture_output=True, text=True, timeout=10,
                              env={**os.environ, 'PATH':''}, cwd='/')

    def test_native_preflight_without_python_or_metal(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root/'receipt.json').read_text())
        self.assertEqual(record['status'], 'validated')
        self.assertFalse(record['execution_started'])
        self.assertIsNone(record['actual_backend'])
        self.assertEqual(record['admitted_inputs'], 1)
        self.assertEqual(record['configuration']['policy']['proposal_limit'], 64)
        self.assertEqual(record['presentations'][0]['verification'], 'exact-Z')
        self.assertEqual(len(record['executable_sha256']), 64)
        self.assertGreater(record['planned_buffer_bytes'], 0)
        self.assertTrue(all(value == 0 for value in record['counters'].values()))

    def test_seed_aliases_retain_presentations(self):
        alternate = dict(self.scheme, u=[[-1]], v=[[-1]])
        (self.root/'scheme.jsonl').write_text(json.dumps(self.scheme)+'\n'+json.dumps(alternate)+'\n')
        self.config['input']['files'] = [dict(path='scheme.jsonl', format='jsonl')]
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root/'receipt.json').read_text())
        self.assertEqual((record['admitted_inputs'], record['seed_duplicates']), (1, 1))
        self.assertEqual(len(record['presentations']), 2)
        self.assertNotEqual(record['presentations'][0]['factors_id'], record['presentations'][1]['factors_id'])

    def test_above_ceiling_rank_is_admitted_without_cleanup(self):
        (self.root/'scheme.json').write_text(json.dumps(dict(self.scheme, rank=3,
            u=[[1],[1],[-1]], v=[[1],[1],[1]], w=[[1],[1],[1]])))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root/'receipt.json').read_text())['presentations'][0]['rank'], 3)

    def test_invalid_domain_tensor_and_config_do_not_publish(self):
        for field, value in [('domain','F2'), ('u',[[0]])]:
            (self.root/'scheme.json').write_text(json.dumps(dict(self.scheme, **{field:value})))
            self.assertEqual(self.invoke().returncode, 1)
            self.assertFalse((self.root/'receipt.json').exists())
        (self.root/'scheme.json').write_text(json.dumps(self.scheme))
        for change in (dict(extra=True), dict(operation='other'), dict(policy=dict(settings(), extra=1))):
            self.assertEqual(self.invoke(dict(self.config, **change)).returncode, 1)
            self.assertFalse((self.root/'receipt.json').exists())
        self.assertEqual(self.invoke(extra=('--seed','7')).returncode, 1)

    def test_memory_and_scan_exhaustion_are_resource_results(self):
        for config in (dict(self.config, execution=dict(self.config['execution'], memory_bytes=1)),
                       dict(self.config, limits=dict(scan_bytes=1))):
            result = self.invoke(config)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('resource_limit:', result.stderr)
            self.assertFalse((self.root/'receipt.json').exists())

    def test_numeric_metadata_is_charged_as_owned_storage(self):
        value = dict(self.scheme, metadata=dict(values=[0]*5000))
        (self.root/'scheme.json').write_text(json.dumps(value))
        self.config['execution']['memory_bytes'] = 600000
        self.assertEqual(self.invoke().returncode, 2)
        self.assertFalse((self.root/'receipt.json').exists())

    def test_execution_rejects_unrepresentable_anchor_without_truncation(self):
        for rank in (351, (1 << 32)+1):
            self.config['policy']['collection_rank']=rank
            result=self.invoke()
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('execution representation', result.stderr)
            self.assertFalse((self.root/'receipt.json').exists())

    def test_existing_output_preserved(self):
        (self.root/'receipt.json').write_text('preserve me')
        self.assertEqual(self.invoke().returncode, 1)
        self.assertEqual((self.root/'receipt.json').read_text(), 'preserve me')

    def test_reduction_reserves_verification_workspace_before_execution(self):
        config = dict(self.config, operation='reduce', reduction=dict(domain='ZT', seed=7,
            rounds=2, reducers=2, schemes=2, max_flips=1, no_improvements=2, target_additions=0))
        del config['policy']
        config['execution'] = dict(config['execution'], memory_bytes=64*1024*1024)
        result = self.invoke(config)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('memory budget', result.stderr)
        self.assertFalse((self.root/'receipt.json').exists())
        self.assertFalse((self.root/'receipt.json.circuits.jsonl').exists())
        config['execution']['memory_bytes'] = 256*1024*1024
        result = self.invoke(config)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root/'receipt.json').read_text())
        self.assertGreater(record['reserved_host_bytes'], 64*1024*1024)
        self.assertFalse(record['execution_started'])

    def test_selection_is_not_resume(self):
        import hashlib
        checksum = hashlib.sha256((self.root/'scheme.json').read_bytes()).hexdigest()
        row = dict(schema='fgm-collection-v1', namespace='public', id='one', path='scheme.json',
                   sha256=checksum, format='json', domain='ZT', dimensions=[1,1,1])
        (self.root/'manifest.jsonl').write_text(json.dumps(row)+'\n')
        self.config['input'] = dict(kind='selection', manifest='manifest.jsonl', count=1, seed=7,
                                    filters=dict(dimensions=[1,1,1]))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root/'receipt.json').read_text())
        self.assertEqual(record['configuration']['input']['kind'], 'selection')
        self.assertEqual(record['presentations'][0]['source_binding']['id'], 'one')


if __name__ == '__main__':
    unittest.main()
