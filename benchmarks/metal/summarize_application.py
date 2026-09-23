import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from application import SOURCE_TIMING, artifacts, build_identity, digest, gpu_evidence, source_timing


def distribution(values):
    return {'count': len(values), 'minimum': min(values), 'median': statistics.median(values),
            'maximum': max(values), 'samples': values}


def process_timing(record, rounds):
    wall = record['wall_seconds']
    rate = record['steady_steps_per_second']
    if any(not math.isfinite(value) or value <= 0 for value in [wall, rate]):
        raise ValueError('process timing and throughput must be positive and finite')
    source = record.get('timing_method') == SOURCE_TIMING
    count_ok = (len(record['reports']) >= rounds if source and record['backend'] == 'cpu'
                else len(record['reports']) == rounds)
    if rounds < 2 or not count_ok:
        raise ValueError('process report count does not match configured rounds')
    result = {'process_wall_seconds': wall}
    if 'process_seconds' in record:
        observed = record['process_seconds']
        if not math.isfinite(observed) or observed <= 0 or observed > wall:
            raise ValueError('observed process duration must be positive and no greater than runner duration')
        result['process_seconds'] = observed
    if record['backend'] != 'cpu':
        gpu = record['gpu_seconds']
        if len(gpu) != rounds or any(not math.isfinite(value) or value <= 0 for value in gpu):
            raise ValueError('GPU timings must contain one positive finite value per round')
        result['mean_gpu_dispatch_seconds'] = statistics.mean(gpu[1:])
    if source:
        result['source_interval_seconds'] = record['steady_seconds']
        result['source_interval_uncertainty'] = record['steady_seconds_uncertainty']
        result['completed_cpu_overshoot'] = record['source_completed_overshoot']
    return result


def paired_interval(pairs, iterations=10000):
    by_seed = defaultdict(list)
    for seed, repeat, ratio in pairs:
        by_seed[seed].append(ratio)
    seeds = sorted(by_seed)
    rng = random.Random(190741)
    bootstrap = []
    for _ in range(iterations):
        values = []
        for seed in rng.choices(seeds, k=len(seeds)):
            values.extend(rng.choices(by_seed[seed], k=len(by_seed[seed])))
        bootstrap.append(statistics.median(values))
    bootstrap.sort()
    return {'median_ratio': statistics.median(p[2] for p in pairs),
            'confidence_95': [bootstrap[int(iterations * .025)], bootstrap[int(iterations * .975)]],
            'pairs': len(pairs), 'independent_seeds': len(seeds), 'bootstrap_seed': 190741,
            'bootstrap_iterations': iterations}


def implementation_panels(config, accepted):
    grouped = defaultdict(list)
    for position, case in enumerate(config['cases']):
        key = (case['fixture'], case['count'], case['seed'], case['repeat'] // 2)
        grouped[key].append((position, case))
    panels, excluded = [], []
    design = config.get('counterbalance', {})
    prospective = design.get('enabled', False)
    for (fixture, count, seed, panel), entries in sorted(grouped.items()):
        identity = {'fixture': fixture, 'count': count, 'seed': seed, 'panel': panel}
        gpu_entries = [(position, case) for position, case in entries if case['backend'] in ('baseline', 'candidate')]
        ordered = [case['backend'] for _, case in gpu_entries]
        repeats = [case['repeat'] for _, case in gpu_entries]
        orientation = 'ABBA' if ordered == ['baseline', 'candidate', 'candidate', 'baseline'] else 'BAAB' if ordered == ['candidate', 'baseline', 'baseline', 'candidate'] else None
        if not orientation or repeats != [2 * panel, 2 * panel, 2 * panel + 1, 2 * panel + 1]:
            excluded.append({**identity, 'reason': 'missing or non-counterbalanced recorded case order'})
            continue
        if prospective:
            seed_order = design.get('seed_order', [])
            expected = 'ABBA' if seed in seed_order and (seed_order.index(seed) + panel) % 2 == 0 else 'BAAB'
            sequence = ['cpu', *ordered, 'cpu']
            if (seed not in seed_order or orientation != expected or len(entries) != 6
                    or [case['backend'] for _, case in entries] != sequence
                    or [position for position, _ in entries] != list(range(entries[0][0], entries[0][0] + 6))
                    or any(case.get('panel') != panel or case.get('orientation') != orientation
                           or case.get('panel_position') != index for index, (_, case) in enumerate(entries))):
                excluded.append({**identity, 'reason': 'prospective panel metadata or CPU-bracketed order mismatch'})
                continue
        keys = [(fixture, count, seed, repeat, backend) for repeat in [2 * panel, 2 * panel + 1] for backend in ['baseline', 'candidate']]
        if prospective:
            keys += [(case['fixture'], case['count'], case['seed'], case['repeat'], case['backend']) for _, case in entries if case['backend'] == 'cpu']
        if any(key not in accepted or not accepted[key].get('complete', False) for key in keys):
            excluded.append({**identity, 'reason': 'incomplete baseline/candidate panel'})
            continue
        raw = []
        for repeat in [2 * panel, 2 * panel + 1]:
            baseline = accepted[(fixture, count, seed, repeat, 'baseline')]['steady_steps_per_second']
            candidate = accepted[(fixture, count, seed, repeat, 'candidate')]['steady_steps_per_second']
            if not math.isfinite(baseline) or not math.isfinite(candidate) or min(baseline, candidate) <= 0:
                raise ValueError('panel throughput must be positive and finite')
            raw.append({'repeat': repeat, 'baseline': baseline, 'candidate': candidate, 'ratio': candidate / baseline})
        contrast = sum(math.log(row['ratio']) for row in raw) / 2
        panels.append({**identity, 'orientation': orientation, 'raw_pairs': raw, 'log_ratio': contrast,
                       'geometric_ratio': math.exp(contrast)})
    return panels, excluded


def panel_interval(panels, iterations=10000):
    by_seed = defaultdict(list)
    for panel in panels:
        by_seed[panel['seed']].append(panel['log_ratio'])
    if not by_seed:
        raise ValueError('no complete panels')
    seeds = sorted(by_seed)
    mean = statistics.mean(statistics.mean(by_seed[seed]) for seed in seeds)
    rng = random.Random(190741)
    samples = []
    for _ in range(iterations):
        sampled_seeds = rng.choices(seeds, k=len(seeds))
        means = [statistics.mean(rng.choices(by_seed[seed], k=len(by_seed[seed]))) for seed in sampled_seeds]
        samples.append(math.exp(statistics.mean(means)))
    samples.sort()
    return {'geometric_mean_ratio': math.exp(mean),
            'confidence_95': [samples[int(iterations * .025)], samples[int(iterations * .975)]],
            'independent_seeds': len(seeds), 'panels': len(panels), 'bootstrap_seed': 190741,
            'bootstrap_iterations': iterations,
            'seed_effects': [{'seed': seed, 'panels': len(by_seed[seed]),
                             'geometric_mean_ratio': math.exp(statistics.mean(by_seed[seed]))} for seed in seeds]}


def implementation_summary(config, accepted):
    panels, excluded = implementation_panels(config, accepted)
    groups = defaultdict(list)
    for panel in panels:
        groups[(panel['fixture'], panel['count'])].append(panel)
    comparisons = [{'fixture': fixture, 'count': count, **panel_interval(values)}
                   for (fixture, count), values in sorted(groups.items())]
    required = {(fixture, 2048, seed, panel) for fixture in ['naive', 'rank26']
                for seed in [7, 19, 41, 73, 101] for panel in range(3)}
    completed = {(p['fixture'], p['count'], p['seed'], p['panel']) for p in panels}
    ready = config.get('counterbalance', {}).get('enabled', False) and config['rounds'] == 33 and required <= completed and not excluded
    primary = next((x for x in comparisons if (x['fixture'], x['count']) == ('naive', 2048)), None)
    secondary = next((x for x in comparisons if (x['fixture'], x['count']) == ('rank26', 2048)), None)
    gate = 'NOT VERIFIED'
    secondary_gate = 'NOT VERIFIED'
    if ready and primary and secondary:
        if secondary['geometric_mean_ratio'] < .95:
            secondary_gate = 'FAIL'
        elif secondary['confidence_95'][0] >= .95:
            secondary_gate = 'PASS'
        if primary['geometric_mean_ratio'] < 1.15 or secondary_gate == 'FAIL':
            gate = 'FAIL'
        elif primary['confidence_95'][0] > 1 and secondary_gate == 'PASS':
            gate = 'PASS'
    return {'performance_gate': gate, 'secondary_nonregression_95': secondary_gate,
            'final_design_complete': bool(ready), 'panels': panels, 'excluded_panels': excluded, 'comparisons': comparisons,
            'method': 'Each panel averages two opposite-order log candidate/baseline ratios. Effect exponentiates mean seed-level mean panel contrasts. Bootstrap resamples seeds and whole panels; individual panel positions and dispatches are never resampled.',
            'scope': 'Additional implementation performance criterion only: primary2048 geometric gain at least15% and95% interval above1; rank26 interval at least0.95. Does not replace the raw CPU median milestone or correctness review.',
            'limitations': 'Counterbalancing cancels linear log-time drift and multiplicative period-two effects; it does not establish a frequency cause or remove arbitrary carryover. Intervals describe the tested seeds and panels.'}


def load_directory(directory):
    directory = directory.resolve(strict=True)
    config = json.loads((directory / 'config.json').read_text())
    for backend, identity in config['identities'].items():
        current = build_identity(backend, Path(identity['binary']), Path(identity['source']), Path(identity['build_manifest']))
        if current != identity:
            raise ValueError('source or build identity changed: ' + str(directory))
    accepted, missing = {}, []
    names = [case['name'] for case in config['cases']]
    if len(names) != len(set(names)):
        raise ValueError('duplicate configured cases: ' + str(directory))
    if any(case['backend'] not in config['identities'] for case in config['cases']):
        raise ValueError('missing backend build identity: ' + str(directory))
    for case in config['cases']:
        path = directory / case['name'] / 'result.json'
        record = json.loads(path.read_text()) if path.is_file() else {}
        seal = path.with_name('result.sha256')
        if record and (not seal.is_file() or seal.read_text().strip() != digest(path)):
            raise ValueError(f'altered or unsealed record: {path}')
        if record and any(record.get(k) != v for k, v in case.items()):
            raise ValueError(f'case identity mismatch: {path}')
        if record and artifacts(path.parent) != record.get('artifacts'):
            raise ValueError(f'altered artifact inventory: {path.parent}')
        if not record.get('complete'):
            missing.append({'name': case['name'], 'reason': record.get('error', record.get('cleanup_error', 'missing')),
                            'record': str(path), 'started': bool(record),
                            'cleanup_error': record.get('cleanup_error')})
        else:
            if record.get('timing_method', 'arrival') != config.get('timing_method', 'arrival'):
                raise ValueError('record timing method differs from configured protocol')
            if record.get('timing_method') == SOURCE_TIMING:
                expected = source_timing((path.parent / 'stdout.log').read_text(),
                                         case['backend'], case['count'], config['rounds'])
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError('source timing metadata differs from retained stdout')
            if config.get('expected_kernel') is not None:
                if record.get('expected_kernel') != config['expected_kernel']:
                    raise ValueError('record expected kernel differs from configured protocol')
                if case['backend'] != 'cpu':
                    evidence = gpu_evidence((path.parent / 'stdout.log').read_text(),
                                            (path.parent / 'stderr.log').read_text(),
                                            config['rounds'], config['expected_kernel'])
                    if any(record.get(key) != value for key, value in evidence.items()):
                        raise ValueError('GPU evidence differs from retained logs')
            process_timing(record, config['rounds'])
            accepted[(case['fixture'], case['count'], case['seed'], case['repeat'], case['backend'])] = record
    return config, accepted, missing


def combine_directories(inputs, runner_change_reason=None):
    first = inputs[0][1]
    common = ('version', 'identities', 'rounds', 'counterbalance', 'fixture_sha256', 'platform',
              'developer_dir', 'iterations_per_round', 'time_limit', 'wired_limit')
    for directory, config, accepted, missing in inputs:
        if (any(key not in config or key not in first or config[key] != first[key] for key in common)
                or config.get('expected_kernel') != first.get('expected_kernel')
                or config.get('custom_fixture') != first.get('custom_fixture')
                or config.get('timing_method', 'arrival') != first.get('timing_method', 'arrival')):
            raise ValueError('incompatible fixture-block inputs: ' + str(directory))
    hashes = {config['runner_sha256'] for _, config, _, _ in inputs}
    if len(hashes) > 1 and not runner_change_reason:
        raise ValueError('different runners require --runner-change-reason')
    merged = dict(first, cases=[])
    accepted, missing, selected = {}, [], []
    histories = []
    candidates = defaultdict(list)
    for directory, config, records, failures in inputs:
        histories.append({'directory': str(directory), 'config_sha256': digest(directory / 'config.json'),
                          'runner_sha256': config['runner_sha256'], 'platform': config['platform'],
                          'developer_dir': config['developer_dir'], 'planned_cases': len(config['cases']),
                          'complete_cases': len(records), 'original_matrix_complete': not failures,
                          'excluded_cases': failures})
        machine = directory / 'machine.json'
        if machine.is_file():
            histories[-1]['machine'] = {'path': str(machine), 'sha256': digest(machine)}
        for fixture in {case['fixture'] for case in config['cases']}:
            cases = [case for case in config['cases'] if case['fixture'] == fixture]
            expected = {(fixture, 2048, seed, repeat, backend) for seed in (7, 19, 41, 73, 101)
                        for repeat in range(6) for backend in ('cpu', 'baseline', 'candidate')}
            keys = [(case['fixture'], case['count'], case['seed'], case['repeat'], case['backend']) for case in cases]
            panels, excluded = implementation_panels(dict(config, cases=cases), records)
            if (config['rounds'] == 33 and config.get('counterbalance', {}).get('enabled') and
                    len(keys) == len(expected) and set(keys) == expected and expected <= records.keys()
                    and not excluded and len(panels) == 15):
                candidates[fixture].append((directory, cases, records))
    for fixture in sorted({case['fixture'] for _, config, _, _ in inputs for case in config['cases']}):
        blocks = candidates[fixture]
        if len(blocks) > 1:
            raise ValueError('ambiguous completed fixture blocks: ' + fixture)
        if not blocks:
            missing.append({'fixture': fixture, 'reason': 'no whole completed final fixture block'})
            continue
        directory, cases, records = blocks[0]
        merged['cases'].extend(cases)
        for case in cases:
            key = (case['fixture'], case['count'], case['seed'], case['repeat'], case['backend'])
            accepted[key] = records[key]
        selected.append({'fixture': fixture, 'directory': str(directory), 'cases': len(cases),
                         'reason': 'unique whole completed 33-report, five-seed, three-panel fixture block'})
    return merged, accepted, missing, {'selected_blocks': selected, 'input_matrices': histories,
        'runner_change_reason': runner_change_reason,
        'scope': 'Whole fixture blocks only; no panel or case splicing. Original incomplete matrices remain incomplete. Historical failed and not-started cases are retained in input_matrices and never accepted.'}


def main():
    parser = argparse.ArgumentParser(description='Summarize independent application benchmark runs')
    parser.add_argument('directory', type=Path, nargs='+')
    parser.add_argument('--runner-change-reason')
    args = parser.parse_args()
    directories = [path.resolve(strict=True) for path in args.directory]
    if len(set(directories)) != len(directories):
        raise ValueError('duplicate input directory')
    inputs = [(directory, *load_directory(directory)) for directory in directories]
    block_selection = None
    if len(inputs) == 1:
        _, config, accepted, missing = inputs[0]
    else:
        config, accepted, missing, block_selection = combine_directories(inputs, args.runner_change_reason)
    groups = defaultdict(list)
    for (fixture, count, seed, repeat, backend), record in accepted.items():
        groups[(fixture, count, backend)].append(record)
    output = {'planned_cases': len(config['cases']), 'complete_cases': len(accepted),
              'excluded_cases': missing, 'rounds': config['rounds'], 'distributions': [], 'comparisons': []}
    if config.get('timing_method') == SOURCE_TIMING:
        output['timing_method'] = SOURCE_TIMING
    if block_selection is not None:
        output['fixture_block_selection'] = block_selection
    for (fixture, count, backend), records in sorted(groups.items()):
        timings = [process_timing(record, config['rounds']) for record in records]
        entry = {'fixture': fixture, 'count': count, 'backend': backend,
                 'steps_per_second': distribution([record['steady_steps_per_second'] for record in records]),
                 'process_wall_seconds': distribution([timing['process_wall_seconds'] for timing in timings])}
        if backend != 'cpu':
            entry['mean_gpu_dispatch_seconds'] = distribution([timing['mean_gpu_dispatch_seconds'] for timing in timings])
        observed = [(record, timing) for record, timing in zip(records, timings) if 'process_seconds' in timing]
        if observed:
            entry['process_seconds'] = distribution([timing['process_seconds'] for _, timing in observed])
            entry['process_seconds_order'] = [{'seed': record['seed'], 'repeat': record['repeat']} for record, _ in observed]
        entry['process_seconds_missing'] = len(records) - len(observed)
        entry['process_order'] = [{'seed': record['seed'], 'repeat': record['repeat']} for record in records]
        if config.get('timing_method') == SOURCE_TIMING:
            entry['source_interval_seconds'] = distribution([timing['source_interval_seconds'] for timing in timings])
            entry['source_interval_uncertainty'] = timings[0]['source_interval_uncertainty']
            entry['completed_cpu_overshoot'] = [timing['completed_cpu_overshoot'] for timing in timings]
            entry['coalesced_processes'] = sum(bool(record['coalesced_reports']) for record in records)
        output['distributions'].append(entry)
    for fixture, count in sorted({key[:2] for key in accepted}):
        for reference in ['baseline', 'cpu']:
            pairs = []
            for key, candidate in accepted.items():
                f, n, seed, repeat, backend = key
                other = accepted.get((f, n, seed, repeat, reference))
                if (f, n, backend) == (fixture, count, 'candidate') and other:
                    pairs.append((seed, repeat, candidate['steady_steps_per_second'] / other['steady_steps_per_second']))
            if pairs:
                output['comparisons'].append({'fixture': fixture, 'count': count, 'reference': reference,
                                               **paired_interval(pairs)})
                if config.get('timing_method') == SOURCE_TIMING:
                    bounded = []
                    for key, candidate in accepted.items():
                        f, n, seed, repeat, backend = key
                        other = accepted.get((f, n, seed, repeat, reference))
                        if (f, n, backend) == (fixture, count, 'candidate') and other:
                            bounded.append((seed, repeat, candidate['steady_steps_per_second_bounds'][0] /
                                            other['steady_steps_per_second_bounds'][1]))
                    output['comparisons'][-1]['conservative_rounding_lower_ratios'] = paired_interval(bounded)
    required = {(fixture, 2048, seed, repeat, backend) for fixture in ['naive', 'rank26']
                for seed in [7, 19, 41, 73, 101] for repeat in range(3) for backend in ['cpu', 'baseline', 'candidate']}
    complete = not missing and config['rounds'] == 33 and required <= accepted.keys()
    primary = next((x for x in output['comparisons'] if (x['fixture'], x['count'], x['reference']) == ('naive', 2048, 'cpu')), None)
    secondary = next((x for x in output['comparisons'] if (x['fixture'], x['count'], x['reference']) == ('rank26', 2048, 'baseline')), None)
    output['milestone'] = 'NOT VERIFIED'
    if complete and primary and secondary:
        output['milestone'] = 'PASS' if primary['median_ratio'] >= 1.2 and primary['confidence_95'][0] > 1 and secondary['median_ratio'] >= .95 else 'FAIL'
    output['milestone_scope'] = 'CPU throughput target and median secondary regression check only; implementation promotion and correctness acceptance require separate review.'
    output['implementation_promotion'] = 'NOT VERIFIED'
    output['counterbalanced_implementation'] = implementation_summary(config, accepted)
    output['correctness_suite'] = 'NOT VERIFIED'
    output['secondary_nonregression_95'] = ('PASS' if secondary['confidence_95'][0] >= .95 else 'NOT VERIFIED') if complete and secondary else 'NOT VERIFIED'
    output['method'] = 'Paired process ratios; hierarchical resampling of seeds and runs; first report discarded; dispatches are not independent samples.'
    output['timing_scope'] = 'Each distribution sample represents one process. GPU samples average command-buffer GPU seconds over rounds after the first. Application throughput uses inter-report elapsed time. Optional process_seconds measures immediately before process launch through observed child reaping before output verification; it includes pipe draining and any cleanup before reaping, not an exact OS exit timestamp. process_wall_seconds is total recorded runner time including output verification and final cleanup. Missing process_seconds is explicit. CPU runs stop intentionally at the report limit.'
    if config.get('timing_method') == SOURCE_TIMING:
        output['timing_scope'] = ('Each sample represents one process. Throughput uses cumulative source report-entry clocks, '
            'report1 through the requested final report, independent of stdout delivery batching. Conservative interval '
            'uncertainty is 11ms for CPU and 2ms for Metal. Conservative rounding ratios use candidate lower/reference '
            'upper throughput bounds. GPU means use command-buffer durations after the first dispatch. Observed process '
            'time includes startup, pipe draining and termination; runner time also includes verification. CPU stop is '
            'asynchronous and completed report overshoot is explicit; exports may include work beyond the measurement '
            'window. These results do not establish equal-budget search quality. Legacy arrival measurements are separate.')
    output['summary_script_sha256'] = digest(Path(__file__))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
