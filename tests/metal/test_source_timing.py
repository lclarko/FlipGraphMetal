"""Source-clock parsing and burst-delivery tests, with no Metal execution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/metal'))
import application
import summarize_application


ENDING = '- iteration time (last / min / max / mean): 0.10 / 0.10 / 0.10 / 0.10\n'


def cpu_report(iteration, elapsed, count=2048):
    # Labels/field order copied from the pinned CPU FlipGraph::report output.
    return (f'| threads: 8                  flip iters: 1.00K              iteration: {iteration} |\n'
            f'| count: {count}               reset iters: 1.0B              elapsed: {elapsed} |\n'
            '| ring: ZT                plus diff: 4               improvements: 1 / 10 |\n'
            '+--------+------+------+------------+------------+---------+-----------+------------+\n'
            + ENDING)


def metal_report(iteration, elapsed, count=2048):
    # The ordinal/time row follows this header directly in the frozen executable.
    return ('+--------------------------------------------------------------+\n'
            '| Schemes            Iteration            Elapsed time         |\n'
            f'| {count:7}            {iteration:9}            {elapsed:>12}         |\n'
            '+--------+--------+--------+-------+------+------+-------------+\n'
            + ENDING.replace('0.10', '0.100'))


class SourceTiming(unittest.TestCase):
    def test_seventeen_reports_measure_sixteen_intervals(self):
        for backend, factory, precision in [('cpu', cpu_report, 2),
                                            ('candidate', metal_report, 3)]:
            with self.subTest(backend=backend):
                text = ''.join(factory(i, f'{i / 10:.{precision}f}') for i in range(1, 18))
                result = application.source_timing(text, backend, 2048, 17)
                self.assertEqual(result['measured_report_indices'], [1, 17])
                self.assertEqual(len(result['source_reports']), 17)
                self.assertAlmostEqual(result['steady_seconds'], 1.6)
                self.assertAlmostEqual(result['steady_steps_per_second'], 32768000 / 1.6)
                self.assertEqual(result['source_completed_overshoot'], 0)
                with self.assertRaises(ValueError):
                    application.source_timing(text, backend, 2048, 18)
        device = 'Metal device: Apple M1\n'
        dispatch = 'Metal dispatch randomWalkKernel: 2048 threads, 12.5 ms GPU\n'
        evidence = application.gpu_evidence(device, dispatch * 17, 17, 'randomWalkKernel')
        self.assertEqual(len(evidence['gpu_seconds']), 17)
        for count in [16, 18]:
            with self.subTest(dispatches=count), self.assertRaises(ValueError):
                application.gpu_evidence(device, dispatch * count, 17, 'randomWalkKernel')

    def test_cpu_cumulative_window_and_precision(self):
        result = application.source_timing(''.join(cpu_report(i, f'{i / 10:.2f}')
                                                  for i in range(1, 4)), 'cpu', 2048, 3)
        self.assertEqual(result['measured_report_indices'], [1, 3])
        self.assertAlmostEqual(result['steady_seconds'], .2)
        self.assertEqual(result['steady_seconds_uncertainty'], .011)
        self.assertEqual(result['source_completed_overshoot'], 0)
        self.assertEqual(result['source_partial_report'], {})
        work = 2 * 2048 * 1000
        self.assertAlmostEqual(result['steady_steps_per_second'], work / .2)
        self.assertAlmostEqual(result['steady_steps_per_second_bounds'][0], work / .211)
        self.assertAlmostEqual(result['steady_steps_per_second_bounds'][1], work / .189)

    def test_metal_source_window_ignores_gpu_and_report_durations(self):
        text = ''.join('Metal dispatch randomWalkCompactKernel: 2048 threads, 12.0 ms GPU\n'
                       + metal_report(i, f'{i / 10:.3f}') for i in range(1, 4))
        result = application.source_timing(text, 'candidate', 2048, 3)
        self.assertAlmostEqual(result['steady_seconds'], .2)
        self.assertEqual(result['steady_seconds_uncertainty'], .002)
        self.assertAlmostEqual(result['steady_steps_per_second_bounds'][0], 4096000 / .202)
        self.assertAlmostEqual(result['steady_steps_per_second_bounds'][1], 4096000 / .198)
        self.assertEqual(result, application.source_timing(text, 'baseline', 2048, 3))

    def test_cpu_completed_and_partial_overshoot_are_retained(self):
        text = ''.join(cpu_report(i, f'{i / 10:.2f}') for i in range(1, 5))
        text += '| threads: 8                  flip iters: 1.00K              iteration: 5 |\n'
        result = application.source_timing(text, 'cpu', 2048, 3)
        self.assertEqual(result['source_completed_overshoot'], 1)
        self.assertEqual(result['source_partial_report'], {'iteration': 5})
        self.assertEqual(len(result['source_reports']), 4)
        self.assertAlmostEqual(result['steady_seconds'], .2)
        self.assertIn('No equal-budget quality claim', result['quality_scope'])

    def test_only_unterminated_cpu_trailer_after_budget_is_retained(self):
        text = ''.join(cpu_report(i, f'{i / 10:.2f}') for i in range(1, 4))
        for fragment in ['| threads: 8  flip it', '| count: 2048  reset iters: 1.0B elapsed: 0.',
                         '- iteration time (last / min']:
            with self.subTest(fragment=fragment):
                result = application.source_timing(text + fragment, 'cpu', 2048, 3)
                self.assertEqual(result['source_trailing_fragment'], fragment)
                self.assertEqual(result['source_completed_overshoot'], 0)
                self.assertAlmostEqual(result['steady_seconds'], .2)
        with self.assertRaises(ValueError):
            application.source_timing(text + '| threads: broken\n', 'cpu', 2048, 3)
        with self.assertRaises(ValueError):
            application.source_timing(cpu_report(1, '0.10') + '| threads: truncated', 'cpu', 2048, 3)

    def test_missing_skipped_and_duplicate_completed_reports_reject(self):
        for factory, backend, precision in [(cpu_report, 'cpu', 2), (metal_report, 'candidate', 3)]:
            for ordinals in [(1, 3, 4), (1, 2, 2), (2, 3, 4), (1, 2)]:
                with self.subTest(backend=backend, ordinals=ordinals), self.assertRaises(ValueError):
                    text = ''.join(factory(i, f'{position / 10:.{precision}f}')
                                   for position, i in enumerate(ordinals, 1))
                    application.source_timing(text, backend, 2048, 3)

    def test_wrong_count_and_bad_clocks_reject(self):
        for factory, backend, precision in [(cpu_report, 'cpu', 2), (metal_report, 'candidate', 3)]:
            for middle in ['nan', 'inf', '-0.20', '2e-1', f'{.1:.{precision}f}',
                           f'{.09:.{precision}f}', '0.2.0']:
                with self.subTest(backend=backend, clock=middle), self.assertRaises(ValueError):
                    text = factory(1, f'{.1:.{precision}f}') + factory(2, middle) + factory(3, f'{.3:.{precision}f}')
                    application.source_timing(text, backend, 2048, 3)
            with self.subTest(backend=backend, wrong_count=True), self.assertRaises(ValueError):
                text = ''.join(factory(i, f'{i / 10:.{precision}f}', 2047 if i == 2 else 2048)
                               for i in range(1, 4))
                application.source_timing(text, backend, 2048, 3)

    def test_malformed_or_overlapping_headers_and_endings_reject(self):
        for factory, backend, precision in [(cpu_report, 'cpu', 2), (metal_report, 'candidate', 3)]:
            first = factory(1, f'{.1:.{precision}f}')
            rest = ''.join(factory(i, f'{i / 10:.{precision}f}') for i in [2, 3])
            footer = ENDING if backend == 'cpu' else ENDING.replace('0.10', '0.100')
            wrong_header = first.replace('| threads:', '| threadz:') if backend == 'cpu' else first.replace('Elapsed time', 'Elapsed wrong')
            cases = [ENDING + first + rest, wrong_header + rest,
                     first.replace(footer, '') + rest, first + footer + rest,
                     first + rest.removesuffix(footer)]
            for index, text in enumerate(cases):
                with self.subTest(backend=backend, variant=index), self.assertRaises(ValueError):
                    application.source_timing(text, backend, 2048, 3)

    def test_cpu_contract_fields_and_precision_are_checked(self):
        text = ''.join(cpu_report(i, f'{i / 10:.2f}') for i in range(1, 4))
        for before, after in [('threads: 8', 'threads: 4'), ('1.00K', '2.00K'),
                              ('1.0B', '1.0M'), ('elapsed: 0.20', 'elapsed: 0.200')]:
            with self.subTest(field=before), self.assertRaises(ValueError):
                application.source_timing(text.replace(before, after, 1), 'cpu', 2048, 3)

    def test_metal_requires_exact_cap_with_no_partial_next_report(self):
        text = ''.join(metal_report(i, f'{i / 10:.3f}') for i in range(1, 4))
        for extra in [metal_report(4, '.400'), metal_report(4, '0.400'),
                      '| Schemes            Iteration            Elapsed time         |\n',
                      metal_report(4, '0.400').split(ENDING.split(':')[0])[0]]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                application.source_timing(text + extra, 'candidate', 2048, 3)

    def test_timing_window_must_exceed_precision_bound(self):
        for backend, text in [('cpu', cpu_report(1, '0.10') + cpu_report(2, '0.11')),
                              ('candidate', metal_report(1, '0.100') + metal_report(2, '0.101'))]:
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                application.source_timing(text, backend, 2048, 2)

    def test_coalesced_cpu_process_keeps_source_timing_and_legacy_rejects(self):
        # One small atomic write delivers complete reports together. The process
        # runs no search: this checks transport/termination, not GPU performance.
        text = ''.join(cpu_report(i, f'{i / 10:.2f}') for i in range(1, 5))
        expected = application.source_timing(text, 'cpu', 2048, 3)
        code = 'import os,time; time.sleep(.45); os.write(1, ' + repr(text.encode()) + ')'
        with tempfile.TemporaryDirectory() as temporary:
            for timing in ['source-elapsed', 'arrival']:
                directory = Path(temporary) / timing
                (directory / 'schemes').mkdir(parents=True)
                with patch.object(application, 'wired_memory', return_value=1024):
                    record = application.run_case({'backend': 'cpu', 'count': 2048},
                                                  [sys.executable, '-c', code], directory, 3, timing)
                self.assertTrue(record['coalesced_reports'])
                if timing == 'source-elapsed':
                    self.assertTrue(record['complete'], record.get('error'))
                    for key, value in expected.items():
                        self.assertEqual(record[key], value)
                    self.assertIsNone(record['arrival_steps_per_second'])
                    self.assertEqual(record['verified_exports'], 0)
                    sealed = (directory / 'result.sha256').read_text().strip()
                    self.assertEqual(sealed, application.digest(directory / 'result.json'))
                    self.assertEqual(json.loads((directory / 'result.json').read_text()), record)
                else:
                    self.assertFalse(record['complete'])
                    self.assertIn('multiple reports', record['error'])

    def test_summary_recomputes_source_clock_even_with_valid_result_seal(self):
        text = ''.join(cpu_report(i, f'{i / 10:.2f}') for i in range(1, 4))
        case = dict(name='clock-case', fixture='custom', count=2048, seed=7, repeat=0, backend='cpu')
        # Build provenance has its own tests. This fixture isolates the new
        # protocol/retained-log checks while using real artifact/result hashes.
        identity = dict(binary='/unused/program', source='/unused/source', build_manifest='/unused/build.json')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / case['name']
            directory.mkdir()
            (directory / 'stdout.log').write_text(text)
            config = dict(rounds=3, cases=[case], identities={'cpu': identity},
                          timing_method=application.SOURCE_TIMING)
            application.write_json(root / 'config.json', config)
            record = dict(case, complete=True, wall_seconds=1, reports=[.4, .4, .4])
            record.update(application.source_timing(text, 'cpu', 2048, 3))
            record['artifacts'] = application.artifacts(directory)

            def seal():
                application.write_json(directory / 'result.json', record)
                (directory / 'result.sha256').write_text(application.digest(directory / 'result.json') + '\n')

            seal()
            with patch.object(summarize_application, 'build_identity', return_value=identity):
                _, accepted, missing = summarize_application.load_directory(root)
                self.assertEqual(len(accepted), 1)
                self.assertFalse(missing)
                record['steady_steps_per_second'] *= 1.1
                seal()
                with self.assertRaisesRegex(ValueError, 'retained stdout'):
                    summarize_application.load_directory(root)
                record.update(application.source_timing(text, 'cpu', 2048, 3))
                seal()
                config.pop('timing_method')
                application.write_json(root / 'config.json', config)
                with self.assertRaisesRegex(ValueError, 'timing method'):
                    summarize_application.load_directory(root)

    def test_summary_rejects_mixing_arrival_and_source_protocols(self):
        common = dict(version=1, identities={}, rounds=33, counterbalance={}, fixture_sha256='same',
                      platform='same', developer_dir=None, iterations_per_round=1000,
                      time_limit=45, wired_limit=application.LIMIT, cases=[], runner_sha256='same')
        changed = dict(common, timing_method=application.SOURCE_TIMING)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            summarize_application.combine_directories([
                (Path('/unused/arrival'), common, {}, []),
                (Path('/unused/source'), changed, {}, []),
            ])


if __name__ == '__main__':
    unittest.main()
