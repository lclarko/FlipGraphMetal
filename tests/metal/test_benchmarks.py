import json
import math
from pathlib import Path
import sys
import subprocess
import time
import tempfile
import unittest
from unittest.mock import Mock, patch

BENCHMARKS = Path(__file__).resolve().parents[2] / 'benchmarks/metal'
sys.path.insert(0, str(BENCHMARKS))
import application
from summarize_application import implementation_panels, implementation_summary, paired_interval, panel_interval


class BenchmarkEvidence(unittest.TestCase):
    def test_manifest_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'source'
            source.mkdir()
            code = source / 'main.cpp'
            code.write_text('int main() { return 0; }\n')
            binary = root / 'program'
            binary.write_bytes(b'fixture executable')
            manifest = root / 'build.json'
            manifest.write_text(json.dumps({
                'version': 1, 'source_root': str(source),
                'source_files': application.source_identity(source),
                'binary': str(binary), 'binary_sha256': application.digest(binary),
                'commands': [['clang++', str(code), '-o', str(binary)]],
                'source_revision': 'fixture',
            }))
            application.build_identity('cpu', binary, source, manifest)
            code.write_text('int main() { return 1; }\n')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                application.build_identity('cpu', binary, source, manifest)

    def test_metal_manifest_requires_compiled_shader_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'source'
            source.mkdir()
            (source / 'kernels.metal').write_text('kernel void example() {}\n')
            binary = root / 'program'
            binary.write_bytes(b'wrong source location')
            manifest = root / 'build.json'
            manifest.write_text(json.dumps({
                'version': 1, 'source_root': str(source),
                'source_files': application.source_identity(source),
                'binary': str(binary), 'binary_sha256': application.digest(binary),
                'commands': [['clang++', '-DMETAL_SOURCE_DIR="' + str(source) + '"']],
                'source_revision': 'fixture', 'metal_source_dir': str(source),
            }))
            with self.assertRaisesRegex(ValueError, 'not present'):
                application.build_identity('baseline', binary, source, manifest)

    def test_packaged_library_binding_tampering_and_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'source'
            source.mkdir()
            (source / 'kernels.metal').write_text('kernel fixture')
            asset = root / 'shaders/signed.metallib'
            asset.parent.mkdir()
            asset.write_bytes(b'compiled library fixture')
            checksum = application.digest(asset)
            header = source / 'library.h'
            header.write_text('#define METAL_LIBRARY_NAME "shaders/signed.metallib"\n'
                              '#define METAL_LIBRARY_SHA256 "' + checksum + '"\n')
            binary = root / 'program'
            binary.write_bytes(('shaders/signed.metallib ' + checksum).encode())
            manifest = root / 'build.json'
            data = dict(version=1, source_root=str(source), source_files=application.source_identity(source),
                        binary=str(binary), binary_sha256=application.digest(binary), source_revision='fixture',
                        commands=[['clang++', '-include', str(header), '-o', str(binary)]],
                        metal_library=dict(mode='metallib', library='shaders/signed.metallib',
                                           sha256=checksum, header=str(header)))
            manifest.write_text(json.dumps(data))
            application.build_identity('candidate', binary, source, manifest)
            asset.write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'digest'):
                application.build_identity('candidate', binary, source, manifest)
            asset.write_bytes(b'compiled library fixture')
            for field, value in [('mode', 'source'), ('library', '../escape.metallib'), ('sha256', '0'*64)]:
                changed = dict(data, metal_library={**data['metal_library'], field: value})
                manifest.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    application.build_identity('candidate', binary, source, manifest)
            changed = dict(data, commands=[['clang++', '-DMETAL_SOURCE_DIR="' + str(source) + '"',
                                           '-include', str(header), '-o', str(binary)]])
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'substitute'):
                application.build_identity('candidate', binary, source, manifest)
            manifest.write_text(json.dumps(data))
            import shutil
            from archive import ArchiveResolver
            relocated = root / 'bundle'
            relocated.mkdir()
            shutil.copytree(source, relocated / 'source')
            shutil.copytree(asset.parent, relocated / 'shaders')
            shutil.copy2(binary, relocated / 'program')
            shutil.copy2(manifest, relocated / 'build.json')
            mapping = relocated / 'archive-map.json'
            mapping.write_text(json.dumps(dict(version=1, campaigns={}, paths={
                str(source): 'source', str(header): 'source/library.h',
                str(binary): 'program', str(manifest): 'build.json'})))
            asset.unlink()
            application.build_identity('candidate', binary, source, manifest,
                                       resolver=ArchiveResolver(mapping))
            (relocated / 'shaders/signed.metallib').write_bytes(b'corrupted relocated asset')
            with self.assertRaisesRegex(ValueError, 'digest'):
                application.build_identity('candidate', binary, source, manifest,
                                           resolver=ArchiveResolver(mapping))

    def test_artifact_inventory_detects_additions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'stdout.log').write_text('original\n')
            original = application.artifacts(directory)
            (directory / 'result.json').write_text('{}')
            (directory / 'result.sha256').write_text('seal')
            self.assertEqual(original, application.artifacts(directory))
            (directory / 'extra.json').write_text('{}')
            self.assertNotEqual(original, application.artifacts(directory))

    def test_cleanup_reaps_exited_process_before_signaling(self):
        process = subprocess.Popen(['/usr/bin/true'], start_new_session=True)
        try:
            time.sleep(.1)
            application.terminate(process)
            self.assertEqual(process.returncode, 0)
        finally:
            process.wait(timeout=2)

    def test_bootstrap_uses_seed_and_run_units(self):
        pairs = [(seed, repeat, 1.25) for seed in [7, 19] for repeat in range(3)]
        result = paired_interval(pairs, iterations=100)
        self.assertEqual(result['independent_seeds'], 2)
        self.assertEqual(result['pairs'], 6)
        self.assertEqual(result['median_ratio'], 1.25)
        self.assertEqual(result['confidence_95'], [1.25, 1.25])
        self.assertEqual(result, paired_interval(pairs, iterations=100))

    def test_cleanup_permission_error_requires_absent_group(self):
        process = Mock(pid=123, poll=Mock(return_value=0))
        snapshot = subprocess.CompletedProcess([], 0, '1 1\n456 456\n', '')
        with patch.object(application.os, 'killpg', side_effect=PermissionError), \
                patch.object(application.subprocess, 'run', return_value=snapshot) as run:
            application.terminate(process)
        self.assertEqual(run.call_args.args[0], ['/bin/ps', '-A', '-o', 'pid=,pgid='])
        self.assertGreater(run.call_args.kwargs['timeout'], 0)
        self.assertLessEqual(run.call_args.kwargs['timeout'], 2)
        process.wait.assert_called_once_with(timeout=2)

    def test_cleanup_permission_error_rejects_members_or_failed_snapshot(self):
        for output in ['', 'pid pgid\n']:
            process = Mock(pid=123, poll=Mock(return_value=0))
            snapshot = subprocess.CompletedProcess([], 0, output, '')
            with self.subTest(output=output), \
                    patch.object(application.os, 'killpg', side_effect=PermissionError), \
                    patch.object(application.subprocess, 'run', return_value=snapshot):
                with self.assertRaises((PermissionError, RuntimeError)):
                    application.terminate(process)
        for error in [subprocess.CalledProcessError(1, 'ps'), subprocess.TimeoutExpired('ps', 2)]:
            process = Mock(pid=123, poll=Mock(return_value=0))
            with self.subTest(error=error), \
                    patch.object(application.os, 'killpg', side_effect=PermissionError), \
                    patch.object(application.subprocess, 'run', side_effect=error):
                with self.assertRaises(type(error)):
                    application.terminate(process)

    def test_cleanup_permission_error_does_not_accept_live_leader(self):
        process = Mock(pid=123, poll=Mock(return_value=None))
        with patch.object(application.os, 'killpg', side_effect=PermissionError) as kill, \
                patch.object(application.time, 'monotonic', side_effect=[0, 1, 2, 2]), \
                patch.object(application.time, 'sleep'), \
                patch.object(application.subprocess, 'run') as run:
            with self.assertRaisesRegex(PermissionError, 'KILL after TERM grace; leader=None'):
                application.terminate(process)
        run.assert_not_called()
        self.assertEqual(kill.call_args.args, (123, application.signal.SIGKILL))

    def test_cleanup_permission_error_can_finish_during_grace(self):
        for lingering in [False, True]:
            process = Mock(pid=123, poll=Mock(side_effect=[0, None if not lingering else 0, 0, 0]))
            absent = subprocess.CompletedProcess([], 0, '1 1\n', '')
            present = subprocess.CompletedProcess([], 0, '1 1\n456 123\n', '')
            snapshots = [present, absent] if lingering else [absent]
            with self.subTest(lingering=lingering), \
                    patch.object(application.os, 'killpg', side_effect=PermissionError), \
                    patch.object(application.subprocess, 'run', side_effect=snapshots), \
                    patch.object(application.time, 'monotonic', return_value=0):
                application.terminate(process)
            process.wait.assert_called_once_with(timeout=2)

    def test_cleanup_permission_error_retains_member_evidence_at_deadline(self):
        process = Mock(pid=123, poll=Mock(return_value=0))
        present = subprocess.CompletedProcess([], 0, '1 1\n456 123\n', '')
        with patch.object(application.os, 'killpg', side_effect=PermissionError) as kill, \
                patch.object(application.subprocess, 'run', return_value=present), \
                patch.object(application.time, 'monotonic', side_effect=[0, 0, 2, 2]):
            with self.assertRaisesRegex(PermissionError, r'KILL after TERM grace; leader=0; members=\[\[456, 123\]\]'):
                application.terminate(process)
        self.assertEqual(kill.call_args.args, (123, application.signal.SIGKILL))


class CounterbalancedPanels(unittest.TestCase):
    def fixture(self, seeds=(7, 19), repeats=2, factor=1.5, fixtures=('naive',)):
        cases = application.case_matrix(['cpu', 'baseline', 'candidate'], list(fixtures), [2048], list(seeds), repeats, True)
        config = {'cases': cases, 'rounds': 33, 'counterbalance': {'enabled': True, 'seed_order': list(seeds)}}
        accepted = {}
        for index, case in enumerate(cases):
            rate = 100 * (2 if index % 2 else 1)
            if case['backend'] == 'candidate':
                rate *= factor
            accepted[(case['fixture'], case['count'], case['seed'], case['repeat'], case['backend'])] = {
                **case, 'complete': True, 'steady_steps_per_second': rate}
        return config, accepted

    def test_panel_cancels_period_two_multiplier(self):
        config, accepted = self.fixture()
        panels, excluded = implementation_panels(config, accepted)
        self.assertFalse(excluded)
        self.assertEqual(len(panels), 2)
        for panel in panels:
            self.assertAlmostEqual(panel['geometric_ratio'], 1.5)
            self.assertEqual(sorted(row['ratio'] for row in panel['raw_pairs']), [.75, 3.0])
        result = panel_interval(panels, iterations=100)
        self.assertAlmostEqual(result['geometric_mean_ratio'], 1.5)
        for bound in result['confidence_95']:
            self.assertAlmostEqual(bound, 1.5)
        self.assertEqual(result, panel_interval(panels, iterations=100))

    def test_panel_weights_seeds_equally(self):
        panels = [{'seed': 7, 'log_ratio': math.log(4)},
                  {'seed': 19, 'log_ratio': math.log(1)}, {'seed': 19, 'log_ratio': math.log(1)}]
        result = panel_interval(panels, iterations=100)
        self.assertAlmostEqual(result['geometric_mean_ratio'], 2)

    def test_incomplete_panel_is_excluded(self):
        config, accepted = self.fixture(seeds=(7,))
        del accepted[('naive', 2048, 7, 1, 'candidate')]
        panels, excluded = implementation_panels(config, accepted)
        self.assertFalse(panels)
        self.assertEqual(len(excluded), 1)
        self.assertIn('incomplete', excluded[0]['reason'])

    def test_wrong_recorded_order_is_rejected(self):
        config, accepted = self.fixture(seeds=(7,))
        config['cases'][3], config['cases'][4] = config['cases'][4], config['cases'][3]
        panels, excluded = implementation_panels(config, accepted)
        self.assertFalse(panels)
        self.assertEqual(len(excluded), 1)

    def test_screening_summary_has_no_promotion_thresholds(self):
        config, accepted = self.fixture()
        summary = implementation_summary(config, accepted)
        self.assertNotIn('performance_gate', summary)
        self.assertNotIn('final_design_complete', summary)
        self.assertTrue(summary['comparisons'])

    def test_complete_prospective_design_reports_all_panels(self):
        config, accepted = self.fixture(seeds=(7, 19, 41, 73, 101), repeats=6, fixtures=('naive', 'rank26'))
        summary = implementation_summary(config, accepted)
        self.assertEqual(len(summary['panels']), 30)
        self.assertFalse(summary['excluded_panels'])
        self.assertNotIn('performance_gate', summary)

    def test_three_panel_orientation_balance(self):
        config, _ = self.fixture(seeds=(7, 19, 41, 73, 101), repeats=6)
        starts = [case for case in config['cases'] if case['panel_position'] == 0]
        self.assertEqual(sum(case['orientation'] == 'ABBA' for case in starts), 8)
        self.assertEqual(sum(case['orientation'] == 'BAAB' for case in starts), 7)
        for index in range(0, len(config['cases']), 6):
            panel = config['cases'][index:index + 6]
            self.assertEqual(panel[0]['backend'], 'cpu')
            self.assertEqual(panel[-1]['backend'], 'cpu')
            self.assertEqual([case['backend'] for case in panel], [case['backend'] for case in panel[::-1]])

    def test_counterbalance_rejects_incompatible_matrix(self):
        with self.assertRaises(ValueError):
            application.case_matrix(['baseline', 'candidate'], ['naive'], [2048], [7], 6, True)
        with self.assertRaises(ValueError):
            application.case_matrix(['cpu', 'baseline', 'candidate'], ['naive'], [2048], [7], 3, True)

    @staticmethod
    def naive_raw(n):
        a, b, c = n
        factors = [[], [], []]
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    for rows, width, index in zip(factors, [a*b, b*c, c*a],
                                                  [i*b+j, j*c+k, k*a+i]):
                        rows.append([int(t == index) for t in range(width)])
        values = [*n, a*b*c, *(v for rows in factors for row in rows for v in row)]
        return ' '.join(map(str, values)) + '\n'

    def test_custom_signed_dimensions_and_backend_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for n in [[3, 3, 3], [4, 4, 4], [2, 3, 4]]:
                fixture = root / ('fixture-' + '-'.join(map(str, n)))
                raw = self.naive_raw(n)
                fixture.write_text(raw)
                metadata, text = application.custom_fixture(fixture)
                self.assertEqual(metadata['n'], n)
                self.assertEqual(metadata['rank'], n[0]*n[1]*n[2])
                self.assertEqual(text, raw)
                self.assertEqual(metadata['sha256'], application.digest(fixture))
                for backend in ['cpu', 'baseline', 'candidate']:
                    directory = root / (fixture.name + backend)
                    directory.mkdir()
                    argv = application.command('/unused/program', backend, 2048, 7, 6,
                                               'custom', directory, fixture_path=fixture)
                    self.assertEqual((directory / 'input.txt').read_text(),
                                     ('' if backend == 'cpu' else '1\n') + raw)
                    if backend == 'cpu':
                        self.assertNotIn('-n1', argv)
                        self.assertEqual(argv[argv.index('--ring') + 1], 'ZT')
                    else:
                        for index, dimension in enumerate(n, 1):
                            self.assertEqual(argv[argv.index(f'-n{index}') + 1], str(dimension))

    def test_custom_rejects_invalid_domain_count_and_tensor(self):
        good = self.naive_raw([2, 3, 4]).split()
        variants = []
        for field, value in [(0, '0'), (0, '17'), (3, '0'), (3, '351'), (4, '2')]:
            changed = good.copy()
            changed[field] = value
            variants.append(' '.join(changed))
        variants += [' '.join(good[:-1]), ' '.join(good + ['0']),
                     '16 16 1 1 ' + '0 ' * 288, 'bad 3 3 1',
                     '2 3 4 1 ' + '0 ' * 26]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.txt'
            for text in variants:
                with self.subTest(text=text[:40]):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        application.custom_fixture(path)

    def test_gpu_evidence_requires_apple_expected_kernel_and_finite_timings(self):
        line = 'Metal dispatch randomWalkKernel: 2048 threads, 12.5 ms GPU\n'
        stdout = 'Metal device: Apple M1\n'
        evidence = application.gpu_evidence(stdout, line * 2, 2, 'randomWalkKernel')
        self.assertEqual(evidence['gpu_device'], 'Apple M1')
        self.assertEqual(evidence['gpu_seconds'], [.0125, .0125])
        variants = [(stdout, line, 'randomWalkKernel'),
                    (stdout, line * 3, 'randomWalkKernel'),
                    (stdout, line * 2, 'randomWalkCompactKernel'),
                    ('', line * 2, 'randomWalkKernel'),
                    ('Metal device: Intel GPU\n', line * 2, 'randomWalkKernel'),
                    (stdout * 2, line * 2, 'randomWalkKernel')]
        for bad in ['0', 'nan', 'inf', '-1', '1.2.3', '9' * 400]:
            variants.append((stdout, line.replace('12.5', bad) * 2, 'randomWalkKernel'))
        for out, err, kernel in variants:
            with self.subTest(stderr=err[:90], stdout=out), self.assertRaises(ValueError):
                application.gpu_evidence(out, err, 2, kernel)

    def test_gpu_evidence_accepts_only_search_kernels(self):
        for name in ['randomWalkKernel', 'randomWalkCompactKernel']:
            match = application.GPU.search(f'Metal dispatch {name}: 2048 threads, 12.5 ms GPU')
            self.assertEqual(match['kernel'], name)
            self.assertEqual(float(match['ms']), 12.5)
        self.assertIsNone(application.GPU.search('Metal dispatch initializeNaiveKernel: 2048 threads, 12.5 ms GPU'))


if __name__ == '__main__':
    unittest.main()
