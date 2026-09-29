"""Freeze the proposed M4 two-auxiliary experiments without running them."""

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from baseline import effectiveness_seed, fgm3_generation_seed


ROOT = Path(__file__).resolve().parents[2]
FGM3_PATH = ROOT / 'benchmarks/workflow/fixtures/fgm3/protocol.json'
FGM3_SHA256 = 'e274e3080a2feed16324ecd3d51a7d1194faaf33906a528f1c0662d15d41173b'
SEEDS = [7, 19, 41]
COHORTS = {'CAL2': 2, 'SKIP2': 2, 'E4': 4, 'U8': 8}
EXECUTABLES = {'additions_reducer', 'flip_graph', 'scheme_tool'}
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def _object(value, keys, label):
    if type(value) is not dict or not keys <= value.keys():
        raise ValueError(f'{label} must contain {sorted(keys)}')


def _string(value, label):
    if type(value) is not str or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f'invalid {label}')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError(f'invalid UTF-8 {label}') from exc


def _hex(value, length, label):
    if type(value) is not str or not (HEX40 if length == 40 else HEX64).fullmatch(value):
        raise ValueError(f'invalid {label}')
    if len(set(value)) == 1:
        raise ValueError(f'placeholder {label}')


def _path(value, label):
    _string(value, label)
    path = PurePosixPath(value)
    if (path.is_absolute() or value.startswith('\\') or '\\' in value or
            any(part in ('', '.', '..') for part in value.split('/')) or ':' in path.parts[0]):
        raise ValueError(f'unsafe {label}')


def _path_fields(value, label):
    if type(value) is dict:
        for key, item in value.items():
            if key == 'path' or key.endswith('_path'):
                _path(item, f'{label}.{key}')
            else:
                _path_fields(item, f'{label}.{key}')
    elif type(value) is list:
        for index, item in enumerate(value):
            _path_fields(item, f'{label}[{index}]')


def _panel(value):
    _object(value, {'schema', 'cohorts', 'factors', 'freeze_bindings'}, 'panel')
    if value['schema'] != 'fgm-two-aux-panel-v1' or type(value['cohorts']) is not dict or set(value['cohorts']) != set(COHORTS):
        raise ValueError('panel schema or cohorts')
    cohorts = value['cohorts']
    for name, count in COHORTS.items():
        ids = cohorts[name]
        if type(ids) is not list or len(ids) != count:
            raise ValueError(f'{name} must have {count} distinct IDs')
        for id_ in ids:
            _string(id_, f'{name} ID')
        if len(set(ids)) != count:
            raise ValueError(f'{name} has duplicate IDs')
    if cohorts['CAL2'] != ['sun', 'cn122'] or cohorts['SKIP2'] != ['original', 'laderman'] or cohorts['E4'][0] != 'smirnov':
        raise ValueError('required calibration, skip, or E4 anchor order')
    ordered = sum((cohorts[name] for name in COHORTS), [])
    if len(set(ordered)) != 16:
        raise ValueError('cohort IDs overlap')
    if type(value['factors']) is not list or len(value['factors']) != 16:
        raise ValueError('panel needs 16 factors')
    by_id = {}
    canonical = set()
    effective = set()
    for row in value['factors']:
        _object(row, {'id', 'cohort', 'path', 'sha256', 'canonical_id', 'effective_factors_id'}, 'factor')
        _string(row['id'], 'factor ID')
        _string(row['cohort'], 'cohort')
        if row['id'] in by_id or row['cohort'] not in COHORTS or row['id'] not in cohorts[row['cohort']]:
            raise ValueError('factor ID or cohort mismatch')
        _path(row['path'], 'factor path')
        _hex(row['sha256'], 64, 'factor sha256')
        for key, prefix, seen in (('canonical_id', 'fgm-scheme-v1:', canonical),
                                  ('effective_factors_id', 'fgm-factors-v1:', effective)):
            identity = row[key]
            if type(identity) is not str or not identity.startswith(prefix):
                raise ValueError(f'invalid {key}')
            _hex(identity[len(prefix):], 64, key)
            if identity in seen:
                raise ValueError(f'duplicate {key}')
            seen.add(identity)
        _path_fields(row, 'factor')
        by_id[row['id']] = row
    if set(by_id) != set(ordered):
        raise ValueError('factor inventory differs from cohorts')
    bindings = value['freeze_bindings']
    _object(bindings, {'bank_sha256', 'selection_sha256', 'screening_sha256'}, 'freeze bindings')
    for key in ('bank_sha256', 'selection_sha256', 'screening_sha256'):
        _hex(bindings[key], 64, key)
    return by_id


def _builds(value):
    _object(value, {'baseline', 'candidate'}, 'builds')
    for role in ('baseline', 'candidate'):
        build = value[role]
        _object(build, {'source_commit', 'executables', 'library_sha256'}, f'{role} build')
        _hex(build['source_commit'], 40, 'source commit')
        _hex(build['library_sha256'], 64, 'library sha256')
        exes = build['executables']
        if type(exes) is not dict or set(exes) != EXECUTABLES:
            raise ValueError(f'{role} executable inventory')
        for name, executable in exes.items():
            _object(executable, {'path', 'sha256'}, f'{role}.{name}')
            _path(executable['path'], f'{role}.{name} path')
            _hex(executable['sha256'], 64, f'{role}.{name} sha256')
        _path_fields(build, role)


def _fgm3():
    raw = FGM3_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FGM3_SHA256:
        raise ValueError('pinned FGM-3 protocol changed')
    return json.loads(raw)


def _digest_json(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError('inputs must be finite UTF-8 JSON values') from exc
    return hashlib.sha256(raw).hexdigest()


def _reduction_seed(factor_id, trial_seed, block):
    return effectiveness_seed(factor_id, trial_seed, 'reduce', block)


def _native_seed(trial_seed, chunk):
    return fgm3_generation_seed(trial_seed, chunk)


def _condition(factor_ids, shared):
    return {'kind': 'shared-roster' if shared else 'separate-factor',
            'factor_ids': list(factor_ids), 'reuse_population': bool(shared)}


def _treatment(kind):
    treatment = {'id': kind, 'build': 'baseline' if kind == 'baseline-omitted' else 'candidate'}
    if kind == 'candidate-enabled':
        treatment['constructor'] = {'family': 'signed-two-aux-distinct-v1', 'max_pair_slots': 65536}
    return treatment


def _attempts_fixed(factors, experiment, quantum):
    attempts = []
    for factor_index, factor in enumerate(factors):
        for repeat in range(1, 6):
            cells = []
            for seed_index in ((factor_index + repeat + offset) % 3 for offset in range(3)):
                seed = SEEDS[seed_index]
                pair = ('baseline-omitted', 'candidate-omitted') if experiment == 'A' else ('candidate-omitted', 'candidate-enabled')
                for enabled in (False, True) if (factor_index + seed_index + repeat) % 2 == 0 else (True, False):
                    cells.append({'id': f'f{factor_index:02d}-r{repeat}-s{seed_index}-t{int(enabled)}',
                                  'factor_id': factor['id'], 'repeat': repeat, 'trial_seed': seed,
                                  'reduction_seed': seed, 'treatment': _treatment(pair[int(enabled)]),
                                  'evaluation_settings': copy.deepcopy(quantum),
                                  'condition': _condition([factor['id']], False)})
            attempts.append({'id': f'f{factor_index:02d}-r{repeat}', 'factor_id': factor['id'],
                             'repeat': repeat, 'cells': cells, 'condition': _condition([factor['id']], False)})
    return attempts


def _attempts_timed(factors, shared, seconds, endpoints, quantum, search):
    attempts = []
    groups = [factors] if shared else [factors[i:i+2] for i in range(0, len(factors), 2)]
    if shared:
        groups = [factors[:3], factors[3:]]
    for group_index, group in enumerate(groups):
        roster = [factor['id'] for factor in group]
        arms = []
        for seed_index, seed in enumerate(SEEDS):
            for roster_index, factor in enumerate(group if not shared else [group[0]]):
                factor_index = (group_index * 2 + roster_index) if not shared else group_index
                for enabled in (False, True) if (factor_index + seed_index) % 2 == 0 else (True, False):
                    arm_id = f'{group_index:02d}-s{seed_index}-f{roster_index}-t{int(enabled)}'
                    arm = {'id': arm_id, 'trial_seed': seed,
                           'factor_ids': roster if shared else [factor['id']],
                           'condition': _condition(roster if shared else [factor['id']], shared),
                           'treatment': _treatment(('candidate-omitted', 'candidate-enabled')[int(enabled)]),
                           'seconds': seconds, 'endpoints': endpoints[:]}
                    if shared:
                        arm.update(native_search_settings=copy.deepcopy(search), evaluation_base_seed=seed,
                                   generation_seed_rule={'namespace': 'fgm3', 'trial': seed,
                                                         'stage': 'generation', 'chunk_start': 0, 'chunk_step': 1},
                                   first_generation_seed=_native_seed(seed, 0))
                    else:
                        arm.update(evaluation_settings=copy.deepcopy(quantum),
                                   reduction_seed_rule={'namespace': factor['id'], 'trial': seed,
                                                        'stage': 'reduce', 'block_start': 0, 'block_step': 1},
                                   first_reduction_seed=_reduction_seed(factor['id'], seed, 0))
                    arms.append(arm)
        attempts.append({'id': f'{"cn" if shared else "cs"}-{group_index:02d}', 'factor_ids': roster,
                         'condition': _condition(roster, shared), 'arms': arms})
    return attempts


def plan(panel, builds, experiment_id):
    """Return a deterministic proposal bound to supplied identities and inputs."""
    _string(experiment_id, 'experiment ID')
    by_id = _panel(panel)
    _builds(builds)
    pinned = _fgm3()
    ordered = [by_id[id_] for name in COHORTS for id_ in panel['cohorts'][name]]
    search = {'execution': dict(pinned['execution'], workers=1, max_batches=1),
              'policy': copy.deepcopy(pinned['policy']), 'pool': dict(pinned['pool'], selector='cost-diverse'),
              'evaluation': dict(pinned['evaluation']), 'history': dict(pinned['history']),
              'circuit_target': 54,
              'target_rule': 'record target 54; never shorten an arm on attainment'}
    quantum = {'domain': 'ZT', 'strategy': 'combined',
               'reducers': 128, 'rounds': 16, 'no_improvements': 16, 'schemes': 1,
               'max_flips': 0, 'target_additions': 0}
    common = {'seeds': SEEDS[:], 'evaluator_quantum': quantum,
              'resources': {'child_timeout_seconds': 45, 'outer_seconds': 900,
                            'finalization_reserve_seconds': 50, 'wired_limit_bytes': 4294967296,
                            'prelaunch_bytes': 1946157056, 'launch_reserve_bytes': 2348810240,
                            'execution_memory_bytes': 536870912},
              'qualification': {'fixed_factor_attempts': 16, 'native_roster_treatment_attempts': 4,
                                'intent': 'calibrate frozen work units before measurement',
                                'allowance_rule': 'ceil(1.25 * max(elapsed_seconds - headroom_wait_seconds)) + 2',
                                'excluded_from_work_seconds': 'explicit prelaunch headroom wait only',
                                'headroom_wait_charged_to': ['arm', 'global'],
                                'admission': 'each work unit must fit the remaining arm; initial and resume are checked separately; neither guarantees both fit',
                                'result_binding': 'qualification output must name the SHA-256 of this frozen manifest',
                                'result_use': 'fill calibrated allowances only; do not change settings or schedules'},
              'accounting': {
                  'endpoint_credit': 'credit only independently exact-verified, durable results completed by each endpoint',
                  'late_results': 'retain later verified results as evidence; never credit them retroactively',
                  'time': 'charge headroom waits, child work, verification, and audit to attempt global time; charge waits and work within an arm to its arm clock',
                  'missing_results': 'retain incomplete, failed, and unrun cells with planned denominators; do not replace or reschedule them',
                  'qualification_failure': 'blocks measurement; retain the failed qualification attempt',
                  'performance_verdict': 'descriptive only until an independent numeric allowance is approved',
                  'metrics': {
                      'A': 'paired verified fixed-factor cost and runtime for baseline omitted versus candidate omitted',
                      'B': 'paired verified fixed-factor cost and runtime for candidate omitted versus enabled',
                      'C-S': 'per-factor best verified additions by 10 and 20 seconds, target-54 attainment, and verified evaluation count',
                      'C-N': 'per-roster best verified additions by 30 and 60 seconds, target-54 attainment, and verified evaluation count'},
                  'cohort_reporting': {
                      'CAL2': 'calibration only; exclude from effectiveness estimates',
                      'SKIP2': 'skip controls only; exclude from eligible-benefit estimates',
                      'E4': 'report conditional eligible benefit separately',
                      'U8': 'report unconditional frozen-candidate workload including failures and ineligible cases',
                      'C-N G3': 'report shared Original/Laderman/Smirnov roster separately from U8',
                      'pooling': 'do not pool separate-factor and shared-roster conditions'}},
              'gates': {'phase': 'protocol-proposal', 'protocol_approval': 'pending',
                        'heldout_qualification_approval': 'pending', 'measurement_approval': 'pending',
                        'fixture_publication_approval': 'pending', 'publication_approval': 'pending',
                        'numeric_performance_allowance': None,
                        'performance_allowance': 'descriptive only'}}
    experiments = {}
    for name in ('A', 'B'):
        experiments[name] = {'kind': 'fixed-factor', 'attempt_budget_seconds': 900, 'phases_seconds_per_attempt': {
            'preparation': 75, 'native': 270, 'independent_verification': 270,
            'headroom_wait': 100, 'audit': 135, 'finalization': 50},
            'attempts': _attempts_fixed(ordered, name, quantum),
            'seed_rule': 'literal trial seed 7, 19, or 41 in every repeat; each treatment pair shares it'}
    cs = [by_id[id_] for name in ('E4', 'U8') for id_ in panel['cohorts'][name]]
    experiments['C-S'] = {'kind': 'timed-separate-factor', 'attempt_budget_seconds': 900, 'phases_seconds_per_attempt': {
        'preparation': 75, 'arms': 240, 'audit': 200, 'drain': 335, 'finalization': 50},
        'attempts': _attempts_timed(cs, False, 20, [10, 20], quantum, search)}
    cn = [by_id[id_] for id_ in ['original', 'laderman', 'smirnov'] + panel['cohorts']['U8']]
    experiments['C-N'] = {'kind': 'timed-shared-roster', 'attempt_budget_seconds': 900, 'phases_seconds_per_attempt': {
        'preparation': 75, 'arms': 360, 'audit': 200, 'drain': 215, 'finalization': 50},
        'attempts': _attempts_timed(cn, True, 60, [30, 60], quantum, search), 'native_search': copy.deepcopy(search)}
    qualification = common['qualification']
    qualification['fixed_factor_schedule'] = []
    for attempt in experiments['B']['attempts'][::5]:
        qualification['fixed_factor_schedule'].append({
            'id': 'qualify-' + attempt['id'], 'factor_id': attempt['factor_id'],
            'cells': copy.deepcopy(attempt['cells']), 'calibrated_allowance_ceiling_seconds': 20,
            'phases_seconds': copy.deepcopy(experiments['B']['phases_seconds_per_attempt'])})
    qualification['native_schedule'] = []
    for attempt in experiments['C-N']['attempts']:
        for kind in ('candidate-omitted', 'candidate-enabled'):
            units = []
            for arm in attempt['arms']:
                if arm['treatment']['id'] != kind:
                    continue
                for operation in ('initialize', 'resume'):
                    units.append({'id': arm['id'] + '-' + operation,
                                  'trial_seed': arm['trial_seed'], 'operation': operation,
                                  'chunk': 0 if operation == 'initialize' else 1,
                                  'generation_seed': _native_seed(arm['trial_seed'], 0 if operation == 'initialize' else 1),
                                  'evaluation_base_seed': arm['trial_seed'],
                                  'history_id': arm['id'], 'treatment': copy.deepcopy(arm['treatment']),
                                  'native_search_settings': copy.deepcopy(search)})
            phases = copy.deepcopy(experiments['C-N']['phases_seconds_per_attempt'])
            phases['work_units'] = phases.pop('arms')
            qualification['native_schedule'].append({
                'id': 'qualify-' + attempt['id'] + '-' + kind,
                'factor_ids': attempt['factor_ids'][:], 'units': units,
                'history_rule': 'fresh for each seed and treatment; resume only its matching initialization',
                'calibrated_allowance_ceiling_seconds': 60, 'phases_seconds': phases})
    return {'schema': 'fgm-two-aux-experiment-manifest-v1', 'experiment_id': experiment_id,
            'panel': copy.deepcopy(panel), 'builds': copy.deepcopy(builds),
            'input_bindings': {'panel_sha256': _digest_json(panel), 'builds_sha256': _digest_json(builds)},
            'fgm3_protocol': {'path': 'benchmarks/workflow/fixtures/fgm3/protocol.json', 'sha256': FGM3_SHA256},
            'protocol': {'family': 'signed-two-aux-distinct-v1', 'max_pair_slots': 65536,
                         'version': 'fgm-two-aux-m4-v1'}, **common, 'experiments': experiments}


def validate_manifest(value):
    """Reject any changed schedule, binding, resource, or gate field."""
    _object(value, {'panel', 'builds', 'experiment_id'}, 'manifest')
    bindings = value.get('input_bindings')
    if bindings != {'panel_sha256': _digest_json(value['panel']),
                    'builds_sha256': _digest_json(value['builds'])}:
        raise ValueError('embedded input bindings changed')
    if value != plan(value['panel'], value['builds'], value['experiment_id']):
        raise ValueError('manifest differs from the frozen proposal')
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    panel = json.loads(args.panel.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs)
    builds = json.loads(args.builds.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs)
    manifest = plan(panel, builds, args.experiment_id)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
