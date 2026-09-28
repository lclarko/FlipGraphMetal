"""Bounded native additive-constructor qualification on public rank-23 factors.

Each native child uses the ordinary Metal guard. The output directory is new and
keeps configurations, receipts, exports, guard logs, and a partial result on error.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/workflow'))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from application import digest, write_json
from guard import (DEFAULT_WIRED_LIMIT_BYTES, LAUNCH_WIRED_CEILING_BYTES,
                   LAUNCH_WIRED_RESERVE_BYTES, run)
from two_aux import wait_for_headroom
import baseline as b

FIXTURES = ROOT / 'benchmarks/workflow/fixtures/fgm1/factors'
PROTOCOL = ROOT / 'benchmarks/workflow/fixtures/fgm3/protocol.json'
EXPECTED = {'sun': dict(u=13, v=13, w=30),
            'cn122': dict(u=13, v=14, w=28)}
CONSTRUCTOR = dict(family='signed-two-aux-distinct-v1', max_pair_slots=65536)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rows(path):
    result = []
    with path.open() as stream:
        for line in stream:
            require(len(line.encode()) <= 8388608 and line.endswith('\n'),
                    'oversized or unterminated JSONL record')
            result.append(json.loads(line))
    return result


def journal_hashes(path):
    return {item.name: digest(item) for item in sorted(path.iterdir()) if item.is_file()}


def untimed(value):
    if isinstance(value, dict):
        return {key: untimed(item) for key, item in value.items()
                if key != 'source_binding' and 'microseconds' not in key}
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def make_config(protocol, inputs, history, receipt, *, resume=False,
                constructor=True, slot_budget=65536):
    policy = deepcopy(protocol['policy'])
    policy['seed'] = 7
    evaluation = deepcopy(protocol['evaluation'])
    evaluation['seed'] = 7
    if constructor:
        evaluation['constructor'] = dict(CONSTRUCTOR, max_pair_slots=slot_budget)
    pool = deepcopy(protocol['pool'])
    pool['selector'] = 'cost-diverse'
    execution = deepcopy(protocol['execution'])
    execution.update(workers=1, batch_steps=32, max_batches=1, block_size=32,
                     backend='packed')
    input_spec = (dict(kind='resume', journal=str(history)) if resume else
                  dict(kind='files', files=[dict(path=str(path), format='json', domain='ZT')
                                             for path in inputs]))
    return dict(schema='fgm-run-v1', operation='search', workflow='additive-search',
                policy=policy, evaluation=evaluation, pool=pool, execution=execution,
                input=input_spec, output=str(receipt),
                history=dict(path=str(history), **protocol['history']))


def run_child(name, argv, output, inventory, deadline, *, expected_rejection=False):
    remaining = deadline - time.monotonic() - 50 - 45
    require(remaining > 0, name + ': outer deadline leaves no child allowance')
    headroom = wait_for_headroom(min(45., remaining))
    write_json(output / (name + '-headroom.json'),
               dict(headroom, ceiling_bytes=LAUNCH_WIRED_CEILING_BYTES,
                    reserve_bytes=LAUNCH_WIRED_RESERVE_BYTES))
    require(headroom['admitted'], name + ': prelaunch headroom unavailable')
    require(all(digest(path) == value for path, value in inventory.items()),
            name + ': build or qualification input changed')
    require(deadline - time.monotonic() >= 95,
            name + ': headroom wait consumed child allowance')
    child = run(['/usr/bin/env', 'PATH=', *[str(arg) for arg in argv]], output / (name + '-guard'),
                wired_limit_bytes=DEFAULT_WIRED_LIMIT_BYTES)
    if expected_rejection:
        log = (output / (name + '-guard/run.log')).read_text()
        require(not child['complete'] and child.get('exit_code') == 1 and
                not child['forced_termination'] and not child['cleanup_failure'] and
                child.get('error') == 'exit 1' and child['wall_seconds'] < 45 and
                'additive resume policy or evaluation settings changed' in log and
                not any(marker in log for marker in
                        ('Metal device:', 'Metal library:', 'Metal dispatch ')),
                name + ': rejection did not happen before GPU initialization')
    else:
        require(child['complete'] and not child['forced_termination'] and
                not child['cleanup_failure'], name + ': guarded child incomplete')
    return child


def validate_receipt(path, config_path, build, *, expected_batches):
    receipt = json.loads(path.read_text())
    require(receipt['status'] == 'complete' and receipt['execution_started'] and
            receipt['configuration_sha256'] == digest(config_path) and
            receipt['executable_sha256'] == build['flip_graph'] and
            receipt['library_mode'] == 'metallib' and
            receipt['library_sha256'] == build['shaders/signed.metallib'],
            'native receipt, configuration, or producer mismatch')
    require(receipt['configuration']['evaluation']['constructor'] == CONSTRUCTOR and
            receipt['configuration']['execution']['max_batches'] == 1 and
            receipt['completed_batches'] == expected_batches and
            receipt['terminal_reason'] == 'batch_limit',
            'native settings or bounded batch mismatch')
    require(receipt['two_aux_preparation_bytes'] == 262144 and
            receipt['reserved_host_bytes'] == (2*receipt['configuration']['pool']['memory_bytes'] +
                4*receipt['configuration']['history']['transaction_bytes'] +
                receipt['configuration']['history']['index_memory_bytes'] +
                128*receipt['configuration']['limits']['record_bytes'] + 262144) and
            receipt['planned_buffer_bytes'] + receipt['admission_content_bytes'] +
            receipt['reserved_host_bytes'] <= receipt['configuration']['execution']['memory_bytes'],
            'two-aux host preparation or resource plan missing')
    return receipt


def export(name, output, binaries, history, receipt, inventory, deadline):
    destination = output / (name + '-evaluations.jsonl')
    argv = [binaries / 'scheme_tool', 'analyze', '--format', 'journal',
            '--evaluations', '--input', history, '--receipt', receipt,
            '--record-bytes', '8388608', '--scan-bytes', '536870912',
            '--output', destination]
    run_child(name + '-export', argv, output, inventory, deadline)
    evaluations, installations = b.fgm3_export_rows(destination)
    return destination, evaluations, installations


def check_evaluations(previous, current, receipt, config, producer, initial):
    checked = b.fgm3_bind_export(previous, current, config['evaluation'], producer, 0)
    require(receipt['evaluated_historical'] == len(current) and
            receipt['evaluated_current_run'] == len(checked),
            'receipt evaluation counts disagree with acknowledged export')
    require(len(current) >= initial and len({r['evaluation']['scheme_id'] for r in current}) == len(current),
            'initial scores or once-per-history canonical identity missing')
    for row in current:
        evaluation = row['evaluation']
        construction = evaluation['construction']
        require(evaluation['settings']['constructor'] == CONSTRUCTOR and
                evaluation['two_auxiliary']['effective_factors_id'] == evaluation['factors_id'] and
                evaluation['pre_two_aux_additions_by_stage'] and
                evaluation['two_auxiliary']['stages'] and
                construction and evaluation['stage_sources'],
                'production evaluation dropped constructor provenance')
        require(sum(evaluation['additions_by_stage'].values()) == evaluation['additions'],
                'stage additions do not sum to total')
    best = receipt['best_evaluation']
    require(any(row['evaluation'] == best for row in current),
            'best evaluation absent from acknowledged prefix')
    artifact = receipt['circuit_artifact']
    artifact_path = Path(artifact['path'])
    require(artifact_path == Path(str(receipt['configuration']['output']) + '.circuits.jsonl') and
            artifact['records'] == 1 and digest(artifact_path) == artifact['sha256'] and
            rows(artifact_path) == [best['circuit']],
            'published best circuit does not match acknowledged evaluation')
    return checked


def compare_standalone(name, row, output, binaries, producer, inventory, deadline):
    evaluation = row['evaluation']
    source = row['source_scheme']
    raw = dict(n=[3, 3, 3], m=23, z2=False, **{key: source[key] for key in 'uvw'})
    effective = b.normalized_input(raw)
    factor_path = output / (name + '-effective.json')
    write_json(factor_path, effective)
    config = dict(schema='fgm-run-v1', operation='reduce',
                  reduction=dict(evaluation['settings'], seed=evaluation['seed']),
                  input=dict(kind='files', files=[dict(path=str(factor_path), format='json', domain='ZT')]),
                  execution=dict(workers=1, batch_steps=1, block_size=32,
                                 backend='general', memory_bytes=536870912),
                  output=str(output / (name + '-reduction-receipt.json')))
    config_path = output / (name + '-reduction-config.json')
    write_json(config_path, config)
    run_child(name + '-reduction', [binaries / 'additions_reducer', '--run-config', config_path],
              output, inventory, deadline)
    receipt = json.loads(Path(config['output']).read_text())
    require(receipt['status'] == 'complete' and
            receipt['configuration_sha256'] == digest(config_path) and
            receipt['executable_sha256'] == producer['reducer_sha256'] and
            receipt['library_mode'] == 'metallib' and
            receipt['library_sha256'] == producer['library_sha256'],
            name + ': standalone reducer producer mismatch')
    require(len(receipt['presentations']) == 1 and
            receipt['presentations'][0]['effective_factors_id'] == evaluation['factors_id'],
            name + ': standalone reducer effective factors mismatch')
    result, = receipt['results']
    artifact_path = Path(receipt['circuit_artifact']['path'])
    require(receipt['circuit_artifact']['records'] == 1 and
            digest(artifact_path) == receipt['circuit_artifact']['sha256'],
            name + ': standalone reducer artifact mismatch')
    circuit, = rows(artifact_path)
    checked = b.verify(circuit, effective)
    require(checked['additions'] == evaluation['additions'] and
            checked['additions_by_stage'] == evaluation['additions_by_stage'],
            name + ': standalone circuit count mismatch')
    for key, value in [('verified_circuit_additions', 'additions'),
                       ('verified_circuit_additions_by_stage', 'additions_by_stage'),
                       ('baseline_additions', 'baseline_additions'),
                       ('baseline_additions_by_stage', 'baseline_additions_by_stage'),
                       ('stage_sources', 'stage_sources'),
                       ('construction', 'construction'),
                       ('two_auxiliary', 'two_auxiliary'),
                       ('pre_two_aux_additions_by_stage', 'pre_two_aux_additions_by_stage'),
                       ('rounds_completed', 'rounds_completed'),
                       ('flip_attempts', 'flip_attempts'),
                       ('flips_applied', 'flips_applied')]:
        require(untimed(result[key]) == untimed(evaluation[value]),
                name + ': standalone ' + key + ' mismatch')
    for stage in 'uvw':
        require(circuit[stage] == evaluation['circuit'][stage] and
                circuit[stage + '_fresh'] == evaluation['circuit'][stage + '_fresh'],
                name + ': standalone circuit stage mismatch')
    require(result['two_auxiliary']['effective_factors_id'] == evaluation['factors_id'],
            name + ': standalone effective factors mismatch')
    return dict(receipt_sha256=digest(Path(config['output'])),
                circuit_sha256=digest(artifact_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'build/metal')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + 900
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='fgm-two-aux-search-qualification-v1', complete=False,
                  steps=[], guard_seconds=45, outer_seconds=900, finalization_reserve_seconds=50,
                  wired_limit_bytes=DEFAULT_WIRED_LIMIT_BYTES,
                  prelaunch_ceiling_bytes=LAUNCH_WIRED_CEILING_BYTES)
    write_json(output / 'result.json', result)
    try:
        binaries = args.binary_dir.resolve(strict=True)
        build = b.fgm3_build_files(binaries)
        extra = {name: digest(binaries / name) for name in
                 ('additions_reducer', 'additions_reducer.build.json')}
        reducer_build = json.loads((binaries / 'additions_reducer.build.json').read_text())
        stat = (binaries / 'additions_reducer').stat()
        require(reducer_build['output'] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                'standalone reducer build receipt mismatch')
        for dependency, checksum in reducer_build['inputs']['dependencies'].items():
            path = Path(dependency)
            require(digest(path if path.is_absolute() else ROOT / path) == checksum,
                    'standalone reducer build dependency changed: ' + dependency)
        inventory = {binaries / key: value for key, value in dict(build, **extra).items()}
        source_files = [Path(__file__).resolve(), ROOT / 'tests/metal/two_aux.py',
                        ROOT / 'tests/metal/verify.py', ROOT / 'benchmarks/workflow/baseline.py',
                        ROOT / 'benchmarks/metal/guard.py', ROOT / 'benchmarks/metal/application.py',
                        ROOT / 'tests/workflow/identity_oracle.py', PROTOCOL,
                        *(FIXTURES / (name + '.json') for name in EXPECTED)]
        inventory.update({path: digest(path) for path in source_files})
        protocol = json.loads(PROTOCOL.read_text())
        inputs = []
        for name in EXPECTED:
            source = FIXTURES / (name + '.json')
            target = output / (name + '-input.json')
            shutil.copyfile(source, target)
            b.verify(json.loads(target.read_text()))
            inputs.append(target)
            inventory[target] = digest(target)
        write_json(output / 'manifest.json', dict(
            schema='fgm-two-aux-search-inputs-v1',
            plan=dict(grouping='shared-roster', inputs=['sun', 'cn122'],
                      generation_seed=7, evaluation_base_seed=7,
                      evaluation=dict(protocol['evaluation'], seed=7, constructor=CONSTRUCTOR),
                      standalone_seed='evaluationSeed(7, effective ordered-factor identity)',
                      native=make_config(protocol, inputs, output/'history', output/'initial-receipt.json'),
                      steps=['initial', 'initial-export', 'sun-reduction', 'cn122-reduction',
                             'resume', 'resume-export', 'changed-budget', 'disabled-constructor',
                             'fresh-disabled', 'fresh-disabled-export'],
                      outer_seconds=900, finalization_reserve_seconds=50, child_seconds=45,
                      headroom_wait_seconds=45, wired_limit_bytes=DEFAULT_WIRED_LIMIT_BYTES,
                      production_path='',
                      prelaunch_ceiling_bytes=LAUNCH_WIRED_CEILING_BYTES),
            sha256={str(path): value for path, value in inventory.items()}))
        history = output / 'history'
        producer = dict(executable_sha256=build['flip_graph'], library_mode='metallib',
                        library_sha256=build['shaders/signed.metallib'])
        comparison_producer = dict(producer, reducer_sha256=extra['additions_reducer'])
        previous = []
        previous_installations = []
        for name, resume in [('initial', False), ('resume', True)]:
            receipt_path = output / (name + '-receipt.json')
            config = make_config(protocol, inputs, history, receipt_path, resume=resume)
            config_path = output / (name + '-config.json')
            write_json(config_path, config)
            run_child(name, [binaries / 'flip_graph', '--run-config', config_path], output, inventory, deadline)
            receipt = validate_receipt(receipt_path, config_path, build, expected_batches=1)
            export_path, current, installations = export(name, output, binaries, history,
                                                         receipt_path, inventory, deadline)
            new = check_evaluations(previous, current, receipt, config, producer, len(inputs))
            known = [b.fgm3_checked_evaluation(row, config['evaluation'], producer)
                     for row in current]
            b.fgm3_bind_installations(previous_installations, installations, known,
                                      'cost-diverse')
            if not resume:
                require(len(current) >= 2 and receipt['seed_duplicates'] == 0,
                        'public initial population was not scored twice')
                for name_fixture, row in zip(EXPECTED, current[:2]):
                    raw_source = json.loads((FIXTURES / (name_fixture + '.json')).read_text())
                    effective = b.normalized_input(raw_source)
                    source = dict(dimensions=[3, 3, 3], rank=23, domain='ZT', orientation='cyclic-w',
                                  **{key: effective[key] for key in 'uvw'})
                    require(row['evaluation']['factors_id'] == b.host_oracle().identity(source, False),
                            name_fixture + ': initial factor presentation mismatch')
                    require(row['evaluation']['additions_by_stage'] == EXPECTED[name_fixture],
                            name_fixture + ': public calibration stage cost mismatch')
                    compare_standalone(name_fixture, row, output, binaries,
                                       comparison_producer, inventory, deadline)
            else:
                require(current[:len(previous)] == previous and
                        receipt['evaluated_historical'] == len(previous) + len(new) and
                        receipt['counters']['flip_attempts'] > 0,
                        'resume lost evaluated prefix or did no controlled work')
            result['steps'].append(dict(name=name, receipt_sha256=digest(receipt_path),
                                        export_sha256=digest(export_path), evaluations=len(current),
                                        new_evaluations=len(new), completed_batches=1))
            write_json(output / 'result.json', result)
            previous = current
            previous_installations = installations
        # Rejected resumes must leave the acknowledged journal and exported prefix intact.
        before = journal_hashes(history)
        for name, constructor, slots in [('changed-budget', True, 65535),
                                         ('disabled-constructor', False, 65536)]:
            receipt_path = output / (name + '-receipt.json')
            config = make_config(protocol, inputs, history, receipt_path,
                                 resume=True, constructor=constructor, slot_budget=slots)
            config_path = output / (name + '-config.json')
            write_json(config_path, config)
            run_child(name, [binaries / 'flip_graph', '--run-config', config_path], output,
                      inventory, deadline, expected_rejection=True)
            require(not receipt_path.exists() and journal_hashes(history) == before,
                    name + ': rejected resume wrote receipt or changed history')
            result['steps'].append(dict(name=name, rejected_before_gpu=True))
            write_json(output / 'result.json', result)
        # A changed constructor choice is accepted only with its own fresh history.
        fresh = output / 'fresh-disabled-history'
        receipt_path = output / 'fresh-disabled-receipt.json'
        config = make_config(protocol, inputs, fresh, receipt_path, constructor=False)
        config_path = output / 'fresh-disabled-config.json'
        write_json(config_path, config)
        run_child('fresh-disabled', [binaries / 'flip_graph', '--run-config', config_path],
                  output, inventory, deadline)
        fresh_receipt = json.loads(receipt_path.read_text())
        require(fresh_receipt['status'] == 'complete' and fresh_receipt['completed_batches'] == 1 and
                fresh_receipt['configuration_sha256'] == digest(config_path) and
                journal_hashes(history) == before, 'fresh-history opt-in failed or changed original history')
        _, fresh_rows, _ = export('fresh-disabled', output, binaries, fresh,
                                  receipt_path, inventory, deadline)
        require(fresh_receipt['evaluated_historical'] == len(fresh_rows) and
                fresh_receipt['evaluated_current_run'] == len(fresh_rows) and
                len(fresh_rows) >= 2 and
                all(row['evaluation']['settings'] == config['evaluation'] and
                    'two_auxiliary' not in row['evaluation'] for row in fresh_rows),
                'fresh disabled history evaluation mismatch')
        b.fgm3_bind_export([], fresh_rows, config['evaluation'], producer, 0)
        result['steps'].append(dict(name='fresh-disabled', receipt_sha256=digest(receipt_path)))
        resident = []
        for row in previous[2:]:
            evaluation = row['evaluation']
            for stage, report in evaluation['two_auxiliary']['stages'].items():
                if (report['dispatched'] > 0 and
                        report['validated'] == report['dispatched'] and
                        report['gpu_rule_checks'] > 0):
                    resident.append(dict(factors_id=evaluation['factors_id'], stage=stage,
                                         dispatched=report['dispatched'],
                                         validated=report['validated'],
                                         gpu_rule_checks=report['gpu_rule_checks']))
        result['resident_constructor_dispatches'] = resident
        require(resident,
                'unmet gate: no descendant dispatched two-auxiliary closure with resident search buffers')
        require(all(digest(path) == value for path, value in inventory.items()),
                'qualification inputs changed')
        require(time.monotonic() < deadline, 'outer deadline exceeded during verification')
        result['complete'] = True
    except Exception as error:
        result['wall_seconds'] = time.monotonic() - started
        result['error'] = str(error)
        write_json(output / 'result.json', result)
        raise
    result['wall_seconds'] = time.monotonic() - started
    write_json(output / 'result.json', result)
    if time.monotonic() >= deadline:
        result.update(complete=False, error='outer deadline exceeded during finalization',
                      wall_seconds=time.monotonic()-started)
        write_json(output / 'result.json', result)
        raise ValueError(result['error'])
    print('Native additive constructor qualification passed:', output)


if __name__ == '__main__':
    main()
