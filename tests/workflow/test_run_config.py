"""Host-only native policy-settings contracts; no search dispatch is available."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

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
        parent = ROOT/'build/parity'
        parent.mkdir(parents=True, exist_ok=True)
        cls.directory = Path(tempfile.mkdtemp(prefix='run-config-', dir=parent))
        cls.binary = cls.directory/'run_config'
        sources = [ROOT/'src/workflow/json.h', ROOT/'src/workflow/run_config.h',
                   ROOT/'tests/workflow/run_config.cpp']
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {'complete': False, 'gpu_execution': 'NOT RUN',
                   'sources': {str(path.relative_to(ROOT)): digest(path) for path in sources}}
        command = ['xcrun', 'clang++', '-std=c++17', '-O2',
                   '-I'+str(ROOT/'src/workflow'), str(sources[-1]), '-o', str(cls.binary)]
        receipt['command'] = command
        receipt['compiler'] = subprocess.check_output(['xcrun', 'clang++', '--version'], text=True)
        try:
            with (cls.directory/'compile.log').open('xb') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=45)
            if any(digest(path) != receipt['sources'][str(path.relative_to(ROOT))] for path in sources):
                raise RuntimeError('configuration sources changed during compilation')
            receipt['executable_sha256'] = digest(cls.binary)
            receipt['complete'] = True
        finally:
            (cls.directory/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')

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


if __name__ == '__main__':
    unittest.main()
