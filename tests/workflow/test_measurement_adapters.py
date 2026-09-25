"""Host-only baseline adapter, evidence and binding checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('parity_baseline', ROOT/'benchmarks/workflow/baseline.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def scalar():
    return dict(n=[1,1,1], m=1, z2=False, u=[[1]], v=[[1]], w=[[1]])


def circuit():
    result = dict(n=[1,1,1], m=1, z2=False, complexity={'naive':0, 'reduced':0})
    for key in 'uvw':
        result[key] = [[{'index':0, 'value':1}]]
        result[key+'_fresh'] = []
    return result


def log(kernel='minimizeKernel', rounds=2, mutation=False):
    lines = ['Metal device: Apple Synthetic Test', 'Metal dispatch initializeKernel: 32 threads, 1 ms GPU']
    for i in range(rounds):
        if mutation and i:
            lines.append('Metal dispatch flipSchemesKernel: 32 threads, 2 ms GPU')
        lines.append(f'Metal dispatch {kernel}: 32 threads, 3 ms GPU')
    return '\n'.join(lines+['real 1.25', '   123456 maximum resident set size'])+'\n'


class BaselineTests(unittest.TestCase):
    def test_adapter_headers(self):
        data = scalar()
        self.assertEqual(b.adapter_bytes(data, 'search').split(), [b'1', b'1',b'1',b'1',b'1', b'1',b'1',b'1'])
        self.assertEqual(b.adapter_bytes(data, 'minimizer').split(), [b'1']*8)
        self.assertEqual(b.adapter_bytes(data, 'reducer').split(), [b'1']*7)
        self.assertEqual(b.adapter_bytes(data, 'search').splitlines()[0], b'1')
        self.assertEqual(b.adapter_bytes(data, 'minimizer').splitlines()[0], b'1 1 1 1 1')
        with self.assertRaises(ValueError):
            b.adapter_bytes(data, 'unknown')

    def test_read_raw_domains(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'fixture.txt'
            path.write_text('1 1 1 3\n'+'1\n'*9)
            self.assertTrue(b.read_raw(path, True)['z2'])
            with self.assertRaises(ValueError):
                b.read_raw(path, False)
            path.write_text('1 1 1 1\n1 1')
            with self.assertRaises(ValueError):
                b.read_raw(path, False)
            path.write_text('1 1 1 1\n2 1 1')
            with self.assertRaises(ValueError):
                b.read_raw(path, True)

    def test_normalization_does_not_modify_source(self):
        data = scalar()
        data['u'][0][0] = data['v'][0][0] = -1
        original = copy.deepcopy(data)
        self.assertEqual(b.normalized_input(data), scalar())
        self.assertEqual(data, original)

    def test_rectangular_adapters_preserve_factor_order(self):
        data = dict(n=[1,2,3], m=6, z2=False, u=[], v=[], w=[])
        for k in range(2):
            for j in range(3):
                data['u'].append([int(x == k) for x in range(2)])
                data['v'].append([int(x == k*3+j) for x in range(6)])
                data['w'].append([int(x == j) for x in range(3)])
        b.verify(data)
        raw = [*data['n'], data['m'], *[x for key in 'uvw' for row in data[key] for x in row]]
        self.assertEqual(list(map(int, b.adapter_bytes(data, 'reducer').split())), raw)
        self.assertEqual(list(map(int, b.adapter_bytes(data, 'search').split())), [1,*raw])
        self.assertEqual(list(map(int, b.adapter_bytes(data, 'minimizer').split())), raw[:4]+[1]+raw[4:])

    def test_metrics(self):
        metrics = b.parse_metrics(log(), 'minimizeKernel', 2)
        self.assertEqual(metrics['completed_rounds'], 2)
        self.assertAlmostEqual(metrics['gpu_work_seconds'], .006)
        self.assertAlmostEqual(metrics['gpu_all_seconds'], .007)
        self.assertEqual(metrics['process_seconds'], 1.25)
        self.assertEqual(metrics['peak_process_rss_bytes'], 123456)
        self.assertIsNone(metrics['host_phases'])
        self.assertIn('legacy_applied_operation_counters', metrics['unavailable'])

    def test_reject_incomplete_wrong_gpu_and_kernel(self):
        for altered in (log().replace('Apple Synthetic Test', 'Other GPU'),
                        log().replace('3 ms GPU', '0 ms GPU'),
                        log().replace('3 ms GPU', 'nan ms GPU'),
                        log().replace('real 1.25', ''),
                        log().replace('minimizeKernel', 'randomWalkKernel')):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                b.parse_metrics(altered, 'minimizeKernel', 2)
        with self.assertRaises(ValueError):
            b.parse_metrics(log(), 'minimizeKernel', 3)

    def test_mutation_dispatch_required_but_not_applied_proof(self):
        b.parse_metrics(log('runReducersKernel', mutation=True), 'runReducersKernel', 2, True)
        with self.assertRaises(ValueError):
            b.parse_metrics(log('runReducersKernel'), 'runReducersKernel', 2, True)
        # Factor reconstruction, not a dispatch alone, supports changed-export evidence.
        original = circuit()
        self.assertEqual(b.circuit_factors(original), [scalar()[key] for key in 'uvw'])
        changed = copy.deepcopy(original)
        changed['u'][0][0]['value'] = changed['v'][0][0]['value'] = -1
        b.verify(changed)
        self.assertNotEqual(b.circuit_factors(changed), b.circuit_factors(original))

    def test_fixed_binding_rejects_different_valid_tensor(self):
        export = circuit()
        b.verify(export, scalar())
        export['u'][0][0]['value'] = export['v'][0][0]['value'] = -1
        b.verify(export)
        with self.assertRaisesRegex(ValueError, 'differ from reference'):
            b.verify(export, scalar())

    def test_inventory_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            build = source/'build/metal'
            (build/'shaders').mkdir(parents=True)
            for name in b.PROGRAMS:
                (build/name).write_bytes(b'executable')
            for name in ('signed.metallib','f2.metallib'):
                (build/'shaders'/name).write_bytes(b'library')
            (build/'config.json').write_text('{}')
            identity = b.inventory(source)
            b.assert_inventory(source, identity)
            (build/'flip_graph').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                b.assert_inventory(source, identity)
            (build/'shaders/f2.metallib').unlink()
            with self.assertRaises(ValueError):
                b.inventory(source)

    def test_missing_program_rejected_on_initial_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            build = source/'build/metal'
            (build/'shaders').mkdir(parents=True)
            for name in b.PROGRAMS[1:]:
                (build/name).write_bytes(b'executable')
            for name in ('signed.metallib','f2.metallib'):
                (build/'shaders'/name).write_bytes(b'library')
            (build/'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                b.inventory(source)

    def test_wrong_pin_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder/'source'
            source.mkdir()
            (folder/'freeze.json').write_text(json.dumps({'commit':'wrong'}))
            with mock.patch.object(b, 'guarded_run') as guard, self.assertRaises(ValueError):
                b.execute(source, folder/'attempt', {}, [], 1)
            guard.assert_not_called()

    def test_command_rounds_prevent_early_stagnation_stop(self):
        config = b.protocol({row[0]:6 for row in b.ROWS})
        for row in b.ROWS:
            argv = b.command(row, config, Path('/binary'), Path('/input'), Path('/output'))
            self.assertEqual(argv[argv.index('--rounds')+1], '6')
            if '--max-no-improvements' in argv:
                self.assertEqual(argv[argv.index('--max-no-improvements')+1], '6')
            if row[0] == 'mutation-reducer':
                self.assertEqual(argv[argv.index('--schemes-count')+1], '2')
                self.assertEqual(argv[argv.index('--max-flips')+1], '10')


if __name__ == '__main__':
    unittest.main()
