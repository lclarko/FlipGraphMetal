"""Qualify Metal two-auxiliary closure against the independent vector oracle.

Prepare requests with --prepare-only, or run each bounded child under the
existing Metal guard. No production evaluator or campaign is invoked here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/workflow'))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from test_two_aux_constructor import (add, canonical, check_witness, classify,
                                      directions, expected_stats, finite_panels,
                                      full_capacity_case, raw_for, reachable,
                                      routes, slot)
from application import wired_memory, write_json
from guard import (DEFAULT_WIRED_LIMIT_BYTES, LAUNCH_WIRED_CEILING_BYTES,
                   LAUNCH_WIRED_RESERVE_BYTES, launch_headroom_available, run)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def total_slots(vectors):
    n = len(vectors)
    return n * (n - 1) * n * (n + 1)


def expected_trace(vectors, inputs, first, second):
    """Recreate the fixed scan from coefficient vectors, without native rules."""
    n = len(vectors)
    values = vectors + (first[3], second[3])
    rules = []
    for left in range(n + 2):
        for right in range(left + 1, n + 2):
            if right >= n or (left < n and right < n):
                for sign in (1, -1):
                    raw = tuple(a + sign * b for a, b in zip(values[left], values[right]))
                    if not any(raw):
                        continue
                    normalize = -1 if next(x for x in raw if x) < 0 else 1
                    value = canonical(raw)
                    out = next((i for i in range(inputs, n) if vectors[i] == value), None)
                    if out is not None:
                        rules.append((0 if right < n else 1,
                                      [out, left, right, normalize, normalize * sign]))

    def creation(out, route):
        left, right, sign, _ = route
        raw = tuple(a + sign * b for a, b in zip(values[left], values[right]))
        normalize = -1 if next(x for x in raw if x) < 0 else 1
        return [out, left, right, normalize, normalize * sign]

    ordered = [gate for group in (0, 1) for kind, gate in rules if kind == group]
    ordered += [creation(n, first), creation(n + 1, second)]
    available = (1 << inputs) - 1
    required = (1 << n) - 1
    gates = []
    checks = sweeps = 0
    changed = True
    while changed and available & required != required:
        changed = False
        sweeps += 1
        for gate in ordered:
            checks += 1
            out, left, right = gate[:3]
            if available & (1 << out) or not available & (1 << left) or not available & (1 << right):
                continue
            gates.append(gate)
            available |= 1 << out
            changed = True
    return dict(status=int(available & required == required), count=len(gates),
                available=available, rule_checks=checks, sweeps=sweeps, gates=gates)


def panel_request(case):
    vectors = directions(case['inputs'], case['targets'])
    return dict(case, two_aux=True, max_pair_slots=total_slots(vectors))


def special_requests():
    cases = []
    capacity = full_capacity_case()
    vectors = directions(capacity['inputs'], capacity['targets'])
    cases.append(dict(capacity, two_aux=True,
                      raw_slots=[raw_for(vectors, (0, 1, 1), (2, 3, 1))]))
    for case, first, second in [
        (dict(inputs=2, targets=[[3, 1]]), (0, 1, 1), (0, 3, 1)),
        (dict(inputs=4, targets=[[1, 1, 1, 1]]), (0, 1, 1), (2, 3, 1)),
        (dict(inputs=4, targets=[[1, 1, 1, 1], [1, 1, 1, -1]]),
         (0, 1, 1), (2, 6, 1)),
    ]:
        vectors = directions(case['inputs'], case['targets'])
        cases.append(dict(case, two_aux=True, raw_slots=[raw_for(vectors, first, second)]))
    maximum = (1 << 31) // 4 - 1
    numeric = dict(inputs=2, targets=[[maximum, 0], [maximum - 1, 1],
                                     [maximum - 2, -1]])
    vectors = directions(numeric['inputs'], numeric['targets'])
    cases.append(dict(numeric, two_aux=True,
                      raw_slots=[raw_for(vectors, (2, 3, 1), (4, len(vectors), 1))]))
    for case in (dict(inputs=2, targets=[[0, 0], [1, 1], [-1, -1], [0, -1]]),
                 dict(inputs=3, targets=[[1, 1, 1], [1, 1, -1]])):
        vectors = directions(case['inputs'], case['targets'])
        raw = next(i for i in range(total_slots(vectors))
                   if classify(vectors, i) == 'prepared'
                   and (case['inputs'] == 2 or not reachable(vectors, case['inputs'], *slot(vectors, i))))
        cases.append(dict(case, two_aux=True, raw_slots=[raw]))
    return cases


def stage_requests():
    case = dict(inputs=4, targets=[[1, 1, 1, 1]])
    vectors = directions(case['inputs'], case['targets'])
    return [dict(case, two_aux=True, gpu_stage=True, transpose_two_aux=transpose,
                 tile_size=tile, max_pair_slots=total_slots(vectors))
            for transpose in (False, True) for tile in (1, 32, 128)]


def required_stage_requests():
    case = dict(inputs=6, targets=[[1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1]])
    vectors = directions(case['inputs'], case['targets'])
    first_success = next(raw for raw in range(total_slots(vectors))
                         if classify(vectors, raw) == 'prepared'
                         and reachable(vectors, case['inputs'], *slot(vectors, raw)))
    requests = []
    for transpose in (False, True):
        for tile in (1, 32, 128):
            for budget in (first_success, first_success + 1, first_success + 1):
                requests.append(dict(case, two_aux=True, gpu_stage=True,
                                     require_two_live=True,
                                     transpose_two_aux=transpose, tile_size=tile,
                                     max_pair_slots=budget))
    return requests, first_success



def tail_requests():
    case = dict(inputs=6, targets=[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    return [dict(case, two_aux=True, gpu_stage=True, require_two_live=True,
                 transpose_two_aux=transpose, tile_size=tile, max_pair_slots=256)
            for transpose in (False, True) for tile in (1, 32, 128)]


def qualify_tails(requests, responses):
    for request, response in zip(requests, responses):
        vectors = directions(request['inputs'], request['targets'])
        prepared = [raw for raw in range(request['max_pair_slots'])
                    if classify(vectors, raw) == 'prepared']
        position = prepared.index(42)
        tile = request['tile_size']
        dispatched = min(((position // tile) + 1) * tile, len(prepared))
        raw_end = (prepared[dispatched-1] + 1 if dispatched % tile == 0
                   else request['max_pair_slots'])
        report = response['report']
        for key, value in expected_stats(vectors, raw_end).items():
            require(report[key] == value, 'physical tail enumeration: ' + key)
        require(report['dispatched'] == report['completed'] == report['validated'] == dispatched,
                'physical tail lane counts')
        require(report['logical_prefix'] == 43 and report['selected_witness']['raw_slot'] == 42,
                'tail changed logical selection')
        require(report['batch_tail_candidates'] == dispatched-position-1, 'tail count')
        require(report['unvisited_raw_slots'] == total_slots(vectors)-raw_end, 'unvisited slots')
        require(report['stop_reason'] == 'bound-attained' and report['coverage'] == 'partial',
                'tail coverage claim')
        qualify_stage_vectors(request, response)
    require(responses[-1]['report']['batch_tail_candidates'] > 0, 'tail fixture needs later lanes')


def nonfixedpoint_request():
    case = dict(inputs=2, targets=[[1, 1]])
    vectors = directions(case['inputs'], case['targets'])
    raw = raw_for(vectors, (0, 2, 1), (1, 2, 1))
    trace = expected_trace(vectors, case['inputs'], *slot(vectors, raw))
    gate = trace['gates'][0]
    witness = dict(raw_slot=raw, status=1, count=1, available=7,
                   gates=[gate], rule_checks=1, sweeps=0)
    return dict(case, two_aux=True, witnesses=[witness])


def qualification_identities(binary):
    paths = [binary, binary.with_name(binary.name + '.build.json')]
    paths += [ROOT / path for path in (
        'tests/metal/two_aux.py', 'tests/metal/correctness.cpp',
        'tests/workflow/two_aux_constructor.h', 'tests/workflow/test_two_aux_constructor.py',
        'src/workflow/two_aux_execution.h', 'src/workflow/two_aux_validation.h',
        'src/metal/two_aux_constructor.h', 'src/metal/circuit_constructor.h',
        'src/metal/reduction_kernels.metal', 'benchmarks/metal/guard.py')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def omission_requests():
    direct = dict(inputs=2, targets=[[1, 1]])
    two_base = dict(inputs=2, targets=[[1, 1], [2, 1]])
    dependent = dict(inputs=2, targets=[[3, 1]])
    both = dict(inputs=4, targets=[[1, 1, 1, 1]])
    dependent4 = dict(inputs=4, targets=[[1, 1, 1, 1], [1, 1, 1, -1]])
    fixtures = [
        ('enabled_base_from_empty', direct, (0, 2, 1), (1, 2, 1), 0),
        ('enabled_base_after_first_target', two_base, (0, 1, -1), (0, 3, 1), 1),
        ('omitted_first_helper', dependent, (0, 1, 1), (0, 3, 1), 0),
        ('omitted_dependent_second_helper', dependent, (0, 1, 1), (0, 3, 1), 1),
        ('omitted_target_after_dependent_helpers', dependent, (0, 1, 1), (0, 3, 1), 2),
        ('both_independent_helpers_enabled', both, (0, 1, 1), (2, 3, 1), 0),
        ('independent_second_helper_enabled', both, (0, 1, 1), (2, 3, 1), 1),
        ('target_enabled_by_both_helpers', both, (0, 1, 1), (2, 3, 1), 2),
        ('second_signed_target_enabled', dependent4, (0, 1, 1), (2, 6, 1), 3),
    ]
    result = []
    for name, case, first_pair, second_pair, count in fixtures:
        vectors = directions(case['inputs'], case['targets'])
        raw = raw_for(vectors, first_pair, second_pair)
        first, second = slot(vectors, raw)
        trace = expected_trace(vectors, case['inputs'], first, second)
        require(trace['status'] == 1 and count < trace['count'], 'omission source trace')
        gates = trace['gates'][:count]
        available = (1 << case['inputs']) - 1
        for gate in gates:
            available |= 1 << gate[0]
        witness = dict(raw_slot=raw, status=0, count=count, available=available,
                       gates=gates, rule_checks=count, sweeps=0)
        result.append((name, [dict(case, two_aux=True, witnesses=[witness])]))
    return result


def corruption_request():
    case = dict(inputs=2, targets=[[3, 1]])
    vectors = directions(case['inputs'], case['targets'])
    successes, failures = [], []
    for raw in range(total_slots(vectors)):
        if classify(vectors, raw) != 'prepared':
            continue
        (successes if reachable(vectors, case['inputs'], *slot(vectors, raw)) else failures).append(raw)
    require(bool(successes and failures), 'corruption fixture lacks both outcomes')
    return dict(case, two_aux=True, raw_slots=[successes[0], failures[0]],
                tamper_lane=1, tamper_kind='premature_fixed_point')


def qualify_trace(request, response):
    vectors = directions(request['inputs'], request['targets'])
    require(response['directions'] == len(vectors), 'direction count')
    require(response['raw_pair_slots'] == total_slots(vectors), 'raw family size')
    if 'raw_slots' in request:
        expected = [raw for raw in request['raw_slots'] if classify(vectors, raw) == 'prepared']
    else:
        expected = [raw for raw in range(request['max_pair_slots'])
                    if classify(vectors, raw) == 'prepared']
        require(response['stats'] == expected_stats(vectors, request['max_pair_slots']), 'filter counts')
    require([item['raw_slot'] for item in response['results']] == expected, 'prepared slot order')
    require(response['batch_validated'] == len(expected), 'whole-batch validation count')
    for item in response['results']:
        raw = item['raw_slot']
        first, second = slot(vectors, raw)
        oracle = expected_trace(vectors, request['inputs'], first, second)
        for key, value in oracle.items():
            require(item[key] == value, f'{key} at raw slot {raw}')
        require(item['status'] == reachable(vectors, request['inputs'], first, second),
                f'reachability at raw slot {raw}')
        check_witness(request, vectors, raw, item)
        require(item['count'] <= 25 and item['sweeps'] <= 26 and
                item['rule_checks'] <= 29224 and item['available'] < (1 << 34),
                f'bounded witness at raw slot {raw}')
        if len(vectors) == 32:
            require(item['status'] == 1 and item['count'] == 25 and
                    item['available'] & (1 << 33), '34-direction, 25-gate capacity')


def qualify_stage(requests, responses):
    for request, response in zip(requests, responses):
        report = response['report']
        require(report['validated'] == report['completed'] == report['dispatched'], 'stage batch counts')
        require(report['raw_scanned'] <= request['max_pair_slots'], 'stage budget')
        require(report['logical_prefix'] <= report['raw_scanned'], 'stage logical prefix')
        require(report['coverage'] in ('full', 'partial'), 'stage coverage')
        require(response['status'] == 1, 'stage has no result')
        require(report['selected_witness']['raw_slot'] < request['max_pair_slots'], 'selected route')
        qualify_stage_vectors(request, response)
    for offset in (0, 3):
        group = responses[offset:offset + 3]
        require(len({json.dumps((r['count'], r['fresh'], r['outputs'],
                                r['report']['selected_witness']['raw_slot'])) for r in group}) == 1,
                'tile-dependent stage output')
    require(responses[3]['count'] == 0, 'transposed four-input singleton cost')


def qualify_stage_vectors(request, response):
    targets = request['targets']
    transpose = request['transpose_two_aux']
    width = len(targets) if transpose else request['inputs']
    expected = ([list(row) for row in zip(*targets)] if transpose else targets)
    wires = [tuple(int(i == j) for j in range(width)) for i in range(width)]

    def evaluate(expression):
        return tuple(sum(term['value'] * wires[term['index']][c] for term in expression)
                     for c in range(width))

    for fresh in response['fresh']:
        wires.append(evaluate(fresh))
    require([list(evaluate(row)) for row in response['outputs']] == expected,
            'stage vector reconstruction')
    additions = len(response['fresh']) + sum(max(0, len(row) - 1) for row in response['outputs'])
    require(response['count'] == additions, 'stage addition count')


def qualify_required_stage(requests, responses, first_success):
    winners = {False: [], True: []}
    for index, (request, response) in enumerate(zip(requests, responses)):
        report = response['report']
        require(report['raw_scanned'] == request['max_pair_slots'], 'required stage exact prefix')
        require(report['validated'] == report['completed'] == report['dispatched'], 'required batch counts')
        if index % 3 == 0:
            require(response['status'] == 0 and report['stop_reason'] == 'budget-exhausted',
                    'one-slot-short prefix must fail')
            require(report['logical_prefix'] == first_success, 'short logical prefix')
        else:
            require(response['status'] == 1 and report['stop_reason'] == 'bound-attained',
                    'exact prefix must succeed')
            require(report['logical_prefix'] == first_success + 1, 'success logical prefix')
            require(report['selected_witness']['raw_slot'] == first_success, 'first successful route')
            require(report['used_auxiliaries'] == 2, 'both helpers live')
            require(response['count'] == (0 if request['transpose_two_aux'] else 4),
                    'required stage cost')
            qualify_stage_vectors(request, response)
            winners[request['transpose_two_aux']].append((response['count'], response['fresh'],
                response['outputs'], report['selected_witness'], report['logical_prefix']))
    for transpose, group in winners.items():
        require(len(group) == 6 and all(item == group[0] for item in group),
                f'tile/repeat determinism for transpose={transpose}')


def wait_for_headroom(max_seconds):
    start = time.monotonic()
    samples = []
    while True:
        wired = wired_memory(timeout=2)
        samples.append(dict(seconds=time.monotonic() - start, wired_bytes=wired))
        if launch_headroom_available(wired):
            return dict(seconds=time.monotonic() - start, samples=samples, admitted=True)
        if time.monotonic() - start >= max_seconds:
            return dict(seconds=time.monotonic() - start, samples=samples, admitted=False)
        time.sleep(min(.5, max_seconds - (time.monotonic() - start)))


def run_group(name, requests, output, binary, wait_seconds, expected_error=None):
    input_path = output / (name + '-input.json')
    payload = (json.dumps(requests, separators=(',', ':'), sort_keys=True) + '\n').encode()
    require(len(payload) <= 1048576, 'constructor input bound')
    require(input_path.read_bytes() == payload, 'frozen request changed: ' + name)
    headroom = wait_for_headroom(wait_seconds)
    write_json(output / (name + '-headroom.json'), headroom)
    require(headroom['admitted'], 'prelaunch wired-memory ceiling: ' + name)
    result = run([str(binary), '--constructor-input', str(input_path),
                  '--output-dir', str(output / (name + '-exports'))],
                 output / (name + '-guard'), wired_limit_bytes=DEFAULT_WIRED_LIMIT_BYTES)
    log = (output / (name + '-guard/run.log')).read_text()
    if expected_error:
        require(not result['complete'] and result.get('exit_code') == 1 and
                expected_error in log,
                'invalid witness was accepted: ' + name)
        return None
    require(result['complete'], 'guarded child incomplete: ' + name)
    response = json.loads((output / (name + '-exports/constructor.json')).read_text())
    require(len(response) == len(requests), 'response count: ' + name)
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/metal/correctness')
    parser.add_argument('--panel-chunk', type=int, default=8)
    parser.add_argument('--headroom-wait-seconds', type=float, default=45.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    require(1 <= args.panel_chunk <= 16 and 0 <= args.headroom_wait_seconds <= 60,
            'qualification chunk or wait bound')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    panels = [panel_request(case) for case in finite_panels()]
    groups = [(f'panels-{start:03d}', panels[start:start + args.panel_chunk])
              for start in range(0, len(panels), args.panel_chunk)]
    required_stage, first_success = required_stage_requests()
    groups += [('structure-numeric', special_requests()), ('stage-tiles', stage_requests()),
               ('required-stage-tiles', required_stage),
               ('batch-tails', tail_requests()),
               ('successful-nonfixedpoint', [nonfixedpoint_request()]),
               ('corrupt-tail', [corruption_request()])]
    groups += [('omission-' + name, requests) for name, requests in omission_requests()]
    manifest = []
    for name, requests in groups:
        payload = (json.dumps(requests, separators=(',', ':'), sort_keys=True) + '\n').encode()
        require(len(payload) <= 1048576, 'constructor input bound')
        path = output / (name + '-input.json')
        path.write_bytes(payload)
        manifest.append(dict(name=name, requests=len(requests), input_sha256=hashlib.sha256(payload).hexdigest()))
    identities = {} if args.prepare_only else qualification_identities(args.binary.resolve())
    write_json(output / 'requests.json', dict(identities=identities, schema='fgm-two-aux-qualification-v1',
                                               panels=len(panels), groups=manifest,
                                               guard_seconds=45,
                                               prelaunch_ceiling_bytes=LAUNCH_WIRED_CEILING_BYTES,
                                               launch_wired_reserve_bytes=LAUNCH_WIRED_RESERVE_BYTES,
                                               headroom_wait_seconds=args.headroom_wait_seconds,
                                               wired_limit_bytes=DEFAULT_WIRED_LIMIT_BYTES))
    write_json(output / 'result.json', dict(complete=False, panels=len(panels),
                                            groups=len(groups), prepare_only=args.prepare_only))
    if args.prepare_only:
        print('Prepared', len(panels), 'panels in', output)
        return
    try:
        require(args.binary.is_file(), 'correctness binary missing')
        for name, requests in groups:
            expected_error = 'not a fixed point' if name == 'corrupt-tail' or name.startswith('omission-') else None
            response = run_group(name, requests, output, args.binary.resolve(),
                                 args.headroom_wait_seconds, expected_error)
            if name.startswith('panels') or name == 'structure-numeric':
                for request, item in zip(requests, response):
                    qualify_trace(request, item)
            elif name == 'stage-tiles':
                qualify_stage(requests, response)
            elif name == 'batch-tails':
                qualify_tails(requests, response)
            elif name == 'successful-nonfixedpoint':
                require(response[0]['batch_validated'] == 1 and
                        response[0]['results'][0]['status'] == 1 and
                        response[0]['results'][0]['count'] == 1, 'successful nonfixedpoint rejected')
            elif name == 'required-stage-tiles':
                qualify_required_stage(requests, response, first_success)
            require(qualification_identities(args.binary.resolve()) == identities, 'qualification inputs changed')
            write_json(output / (name + '-verified.json'), dict(complete=True, requests=len(requests)))
    except Exception as error:
        write_json(output / 'result.json', dict(complete=False, panels=len(panels),
                                                groups=len(groups), error=str(error)))
        raise
    write_json(output / 'result.json', dict(complete=True, panels=len(panels), groups=len(groups)))
    print('Two-aux GPU qualification passed:', output)


if __name__ == '__main__':
    main()
