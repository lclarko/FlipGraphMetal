"""Qualify pinned production programs using public fixtures and retained guards."""
import argparse
import ast
import copy
from collections import defaultdict
import hashlib
import heapq
import io
import json
import math
import platform
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import shutil
import sys
import tempfile
import time
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
sys.path.insert(0, str(ROOT / 'tests/metal'))
from application import digest, hardware_inventory, write_json, source_identity
from guard import run as guarded_run
from application import wired_memory, LIMIT
from smoke import dispatch_evidence
from verify import verify, reconstruct

BASELINE = '2f91a882dab71cd94de1897f397fe96271920797'
PROGRAMS = ('flip_graph', 'flip_graph_f2', 'complexity_minimizer',
            'complexity_minimizer_f2', 'additions_reducer')
ROWS = (
    ('signed-packed', 'flip_graph', 'rank23_3x3.txt', False, 'randomWalkCompactKernel'),
    ('signed-naive4', 'flip_graph', 'naive_4x4.txt', False, 'randomWalkKernel'),
    ('signed-strassen4', 'flip_graph', 'strassen_4x4.txt', False, 'randomWalkKernel'),
    ('f2-search', 'flip_graph_f2', 'strassen_3x3_f2.txt', True, 'randomWalkKernel'),
    ('signed-minimizer', 'complexity_minimizer', 'rank23_3x3.txt', False, 'minimizeKernel'),
    ('f2-minimizer', 'complexity_minimizer_f2', 'strassen_3x3_f2.txt', True, 'minimizeKernel'),
    ('fixed-reducer', 'additions_reducer', 'rank23_3x3.txt', False, 'runReducersKernel'),
    ('mutation-reducer', 'additions_reducer', 'rank23_3x3.txt', False, 'runReducersKernel'),
)


def archive_files(data):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or str(path) != member.name.rstrip('/')
                    or not path.parts or path.parts[0] in ('build', '.git')):
                raise ValueError('unsafe or non-source archive path')
            if member.isdir():
                continue
            if not member.isfile() or member.name in result:
                raise ValueError('archive contains links, special files or duplicate entries')
            stream = archive.extractfile(member)
            result[member.name] = (stream.read(), member.mode & 0o777)
    if 'src/metal/runtime.mm' not in result or 'makefile' not in result:
        raise ValueError('missing required baseline sources')
    return result


def extract_source(files, source):
    """Write a validated archive inventory into a new source directory."""
    source.mkdir()
    for name, (payload, mode) in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
        path.chmod(mode)


def freeze_baseline(output):
    """Archive the public pinned commit into a new, unbuilt source snapshot."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        'schema': 'fgm-baseline-freeze-v1',
        'commit': BASELINE,
        'behavioral_reference': '9ea5bfc144b528184c32c178cae0b49bc60e8e3d',
        'source': str(output / 'source'),
        'created_utc': time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()),
        'complete': False,
        'source_preparation': 'incomplete',
        'build_status': 'NOT RUN',
        'measurement_status': 'NOT RUN',
    }
    write_json(output / 'freeze.json', receipt)
    try:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(ROOT), *args])

        commit = git('rev-parse', '--verify', BASELINE + '^{commit}').decode().strip()
        if commit != BASELINE:
            raise ValueError('baseline commit identity mismatch')
        receipt['tree'] = git('rev-parse', '--verify', BASELINE + '^{tree}').decode().strip()
        archive = git('archive', '--format=tar', BASELINE)
        (output / 'source.tar').write_bytes(archive)
        receipt['archive_sha256'] = hashlib.sha256(archive).hexdigest()
        files = archive_files(archive)
        source = output / 'source'
        extract_source(files, source)
        assert_source_snapshot(source, receipt)
        receipt['source_preparation'] = 'complete'
        receipt['complete'] = True
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        write_json(output / 'freeze.json', receipt)
        (output / 'freeze.sha256').write_text(digest(output / 'freeze.json') + '\n')
    return receipt


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_raw(path, f2):
    values = [int(x) for x in path.read_text().split()]
    if len(values) < 4:
        raise ValueError('short fixture')
    a, b, c, m = values[:4]
    lengths = (a*b, b*c, c*a)
    if min(a, b, c, m) < 1 or len(values) != 4 + m*sum(lengths):
        raise ValueError('fixture shape mismatch')
    result = dict(n=[a,b,c], m=m, z2=f2)
    offset = 4
    for key, length in zip('uvw', lengths):
        result[key] = [values[offset+r*length:offset+(r+1)*length] for r in range(m)]
        offset += m*length
    verify(result)
    return result


def normalized_input(data):
    result = copy.deepcopy(data)
    for r in range(result['m']):
        for key in 'uv':
            first = next((v for v in result[key][r] if v), 0)
            if first < 0:
                result[key][r] = [-v for v in result[key][r]]
                result['w'][r] = [-v for v in result['w'][r]]
    verify(result)
    return result


def adapter_bytes(data, kind):
    header = [*data['n'], data['m']]
    if kind == 'minimizer':
        header += [1]
    lines = [' '.join(map(str, header))]
    lines += [' '.join(map(str, row)) for key in 'uvw' for row in data[key]]
    text = '\n'.join(lines) + '\n'
    if kind == 'search':
        text = '1\n' + text
    elif kind not in ('minimizer', 'reducer'):
        raise ValueError('unknown adapter')
    return text.encode()


def inventory(source):
    build = source / 'build/metal'
    paths = [build / program for program in PROGRAMS]
    paths += sorted(build.glob('*.build.json'))
    paths += [build / 'config.json']
    paths += sorted((build / 'shaders').glob('*'))
    for required in ('signed.metallib', 'f2.metallib'):
        if not (build / 'shaders' / required).is_file():
            raise ValueError('compiled shader library missing')
    for required in [*(build / program for program in PROGRAMS), build / 'config.json']:
        if not required.is_file() or required.is_symlink():
            raise ValueError('missing or linked baseline executable/configuration: ' + str(required))
    return {str(p.relative_to(source)): digest(p) for p in paths if p.is_file()}


def assert_source_snapshot(source, freeze):
    source = Path(source)
    archive_path = source.parent / 'source.tar'
    if freeze.get('commit') != BASELINE or digest(archive_path) != freeze.get('archive_sha256'):
        raise ValueError('baseline source archive identity mismatch')
    expected = archive_files(archive_path.read_bytes())
    for name, (payload, _) in expected.items():
        live = source / name
        if (any(path.is_symlink() for path in (live, *live.parents)
                if path == source or path.is_relative_to(source))
                or not live.is_file() or live.read_bytes() != payload):
            raise ValueError('live source differs from baseline archive: ' + name)
    actual = set()
    for live in source.rglob('*'):
        relative = live.relative_to(source)
        if relative.parts[0] == 'build' or '__pycache__' in relative.parts:
            continue
        if live.is_symlink():
            raise ValueError('linked baseline source')
        if live.is_file():
            actual.add(relative.as_posix())
    if actual != set(expected):
        raise ValueError('extra or missing baseline source files')


def assert_inventory(source, expected):
    actual = inventory(source)
    if actual != expected:
        raise ValueError('frozen build identity changed')


def protocol(rounds):
    return {'schema': 'fgm-baseline-protocol-v1', 'baseline': BASELINE,
            'seed': 7, 'block_size': 32, 'search_schemes': 512,
            'minimizer_schemes': 32, 'reducer_count': 32,
            'max_iterations': 1000, 'rounds': rounds,
            'shader_library_mode': 'metallib',
            'mandatory_rows': [r[0] for r in ROWS],
            'role': 'baseline-only qualification; no candidate acceptance',
            'required_timing_seconds': [5, 10],
            'feature_budgets': 'unapproved'}


def command(row, config, binary, input_path, exports):
    name, program, _, _, _ = row
    rounds = config['rounds'][name]
    argv = [str(binary / program), '--seed', str(config['seed']),
            '--block-size', str(config['block_size']), '--rounds', str(rounds)]
    if program.startswith('flip_graph'):
        dims = [4,4,4] if name in ('signed-naive4', 'signed-strassen4') else [3,3,3]
        for i, n in enumerate(dims, 1):
            argv += [f'-n{i}', str(n)]
        argv += ['--input-path', str(input_path), '--path', str(exports),
                 '--schemes', str(config['search_schemes']), '--max-iterations', str(config['max_iterations']),
                 '--plus-iterations', '1000000000', '--resize-probability', '0',
                 '--expand-probability', '0', '--reduce-probability', '0',
                 '--basis-probability', '0', '--sandwiching-probability', '0']
    elif program.startswith('complexity_minimizer'):
        argv += ['--input-path', str(input_path), '--path', str(exports),
                 '--schemes', str(config['minimizer_schemes']), '--max-iterations', str(config['max_iterations']),
                 '--target-complexity', '0', '--max-no-improvements', str(rounds)]
    else:
        argv += ['-i', str(input_path), '-o', str(exports), '--count', str(config['reducer_count']),
                 '--schemes-count', '2' if name == 'mutation-reducer' else '1',
                 '--max-flips', '10' if name == 'mutation-reducer' else '0',
                 '--max-no-improvements', str(rounds)]
    return argv


def parse_metrics(log, kernel, rounds, mutation=False):
    evidence = dispatch_evidence(log)
    selected = [d for d in evidence['dispatches'] if d[0] == kernel]
    if len(selected) != rounds:
        raise ValueError(f'expected {rounds} {kernel} dispatches; observed {len(selected)}')
    if mutation and sum(d[0] == 'flipSchemesKernel' for d in evidence['dispatches']) != rounds-1:
        raise ValueError('missing reducer mutation dispatches')
    elapsed = re.findall(r'^real\s+([0-9.]+)$', log, re.M)
    rss = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', log, re.M)
    if len(elapsed) != 1 or len(rss) != 1 or float(elapsed[0]) <= 0:
        raise ValueError('missing process timing or RSS')
    return {**evidence, 'gpu_work_seconds': sum(d[2] for d in selected)/1000,
            'gpu_all_seconds': sum(d[2] for d in evidence['dispatches'])/1000,
            'process_seconds': float(elapsed[0]), 'peak_process_rss_bytes': int(rss[0]),
            'completed_rounds': len(selected), 'host_phases': None, 'metal_allocation_bytes': None,
            'unavailable': ['host_phases', 'metal_allocation_bytes', 'legacy_applied_operation_counters']}


def circuit_factors(data):
    factors = []
    a,b,c = data['n']
    for key, width in zip('uvw', (a*b,b*c,data['m'])):
        forms, _ = reconstruct(data[key], data[key+'_fresh'], width, False)
        factors.append(forms if key != 'w' else [list(row) for row in zip(*forms)])
    return factors


def read_qualification(path):
    """Read retained v1 records or the common v2 structure without rewriting evidence."""
    data = json.loads(Path(path).read_text())
    schema = data.get('schema')
    if schema == 'fgm-baseline-qualification-v1':
        data['mode'] = 'production'
        for row in data['attempts']:
            row['mode'] = 'production'
    elif schema == 'fgm-profile-qualification-v1':
        data['mode'] = 'profile'
        data['attempts'] = []
        for old in data['rows']:
            row = dict(old)
            row.update(row.pop('metrics', {}))
            row['mode'] = 'profile'
            # v1 did not measure workflow/verification time for diagnostics.
            row['workflow_seconds'] = row['verification_seconds'] = None
            data['attempts'].append(row)
    elif schema != 'fgm-baseline-qualification-v2' or data.get('mode') not in ('production', 'profile'):
        raise ValueError('unknown qualification schema or mode')
    if any(row.get('mode') != data['mode'] for row in data['attempts']):
        raise ValueError('mixed production and diagnostic attempts')
    return data


def parse_profile_observations(log, expected_kernels):
    """Parse one positive, finite host/allocation observation per GPU dispatch."""
    phases = {'setup_seconds', 'commit_wait_seconds', 'validation_report_seconds'}
    memory = {'tracked_shared_live_bytes', 'tracked_shared_peak_bytes'}
    expected = phases | memory | {'kernel'}
    observations = []
    for line in log.splitlines():
        if not line.startswith('FGM_PROFILE_V1'):
            continue
        parts = line.split()
        if parts[0] != 'FGM_PROFILE_V1':
            raise ValueError('invalid profiling marker')
        values = {}
        for part in parts[1:]:
            if part.count('=') != 1:
                raise ValueError('malformed profiling field')
            key, value = part.split('=', 1)
            if key in values:
                raise ValueError('duplicate profiling field')
            values[key] = value
        if set(values) != expected or not values['kernel']:
            raise ValueError('unexpected profiling fields')
        for key in phases:
            value = float(values[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('profiling phases must be positive and finite')
            values[key] = value
        for key in memory:
            if not re.fullmatch(r'[0-9]+', values[key]):
                raise ValueError('tracked memory must be integer bytes')
            values[key] = int(values[key])
            if values[key] <= 0:
                raise ValueError('tracked memory must be positive')
        if values['tracked_shared_live_bytes'] > values['tracked_shared_peak_bytes']:
            raise ValueError('live tracked memory exceeds peak')
        observations.append(values)
    if not expected_kernels or [r['kernel'] for r in observations] != list(expected_kernels):
        raise ValueError('profile dispatch observations missing or reordered')
    return observations


def production_attempt(prior, production_run, workload):
    """Choose the earliest complete repeat without assuming a directory label."""
    rows = [row for row in prior['attempts'] if row['workload'] == workload]
    seen = set()
    for row in rows:
        repeat = row.get('repeat')
        if type(repeat) is not int or repeat < 0 or repeat in seen:
            raise ValueError('invalid or ambiguous production repeat')
        seen.add(repeat)
    complete = [row for row in rows if row.get('complete') is True]
    if not complete:
        raise ValueError('missing complete production workload')
    selected = min(complete, key=lambda row: row['repeat'])
    return selected, production_run / f"{selected['repeat']:02d}-{workload}"


def run_attempt(row, config, source, attempt, record, save, *, production=None):
    """One guarded process and exact export checks, timed through verification."""
    name, _, _, _, kernel = row
    reference = json.loads((attempt/'effective-input.json').read_text())
    start = time.monotonic()
    argv = command(row, config, source/'build/metal', attempt/'input.txt', attempt/'exports')
    record['argv'] = argv
    save()
    try:
        guard = guarded_run(['/usr/bin/time', '-l', '-p', *argv], attempt/'guard')
        record['guard_complete'] = guard['complete']
        if not guard['complete']:
            raise RuntimeError(f'{name}: incomplete guarded attempt')
        log = (attempt/'guard/run.log').read_text()
        record.update(parse_metrics(log, kernel, config['rounds'][name], name=='mutation-reducer'))
        verify_start = time.monotonic()
        mutation = False
        for path in sorted((attempt/'exports').rglob('*.json')):
            export = json.loads(path.read_text())
            verdict = verify(export, reference if name=='fixed-reducer' else None)
            if name=='mutation-reducer' and 'u_fresh' in export:
                mutation |= circuit_factors(export) != [reference[k] for k in 'uvw']
            record['exports'].append({'file': str(path.relative_to(attempt)), 'sha256': digest(path), **verdict})
        record['verification_seconds'] = time.monotonic()-verify_start
        record['workflow_seconds'] = time.monotonic()-start
        record['peak_system_wired_bytes'] = max(r['wired_bytes'] for r in guard['memory'])
        if name.endswith('reducer') and not record['exports']:
            raise ValueError('reducer exported no independently verified circuit')
        record['applied_mutation_evidence'] = ('changed verified circuit factors' if mutation else 'NOT VERIFIED') if name=='mutation-reducer' else None
        if record['mode'] == 'profile':
            record['profile'] = parse_profile_observations(log, [r[0] for r in record['dispatches']])
            if production is None:
                raise ValueError('profile attempt requires matched production evidence')
            compare_profile(record, *production)
        record['complete'] = True
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        save()
    return record


def compare_profile(record, original, source_attempt):
    for entry in original['exports']:
        if digest(source_attempt/entry['file']) != entry['sha256']:
            raise ValueError('production export changed')
    if sorted(r['sha256'] for r in record['exports']) != sorted(r['sha256'] for r in original['exports']):
        raise ValueError('diagnostic export bytes differ from production')


def assert_diagnostic(source, declaration):
    expected = declaration['diagnostic_files']
    if source.is_symlink() or not source.is_dir():
        raise ValueError('diagnostic source must be an ordinary directory')
    actual = {}
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if relative.parts[0] == 'build' or '__pycache__' in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError('linked diagnostic source')
        if path.is_file():
            actual[relative.as_posix()] = digest(path)
    if actual != expected:
        raise ValueError('diagnostic source changed')


def execute(source, output, config, names, repetitions, *, profile=None, production_run=None):
    output.mkdir(parents=True, exist_ok=False)
    mode = 'profile' if profile is not None else 'production'
    prior = None
    if mode == 'profile':
        prior = read_qualification(production_run/'qualification.json')
        declaration = json.loads((profile/'profile.json').read_text())
        if (prior['mode'] != 'production' or prior.get('complete') is not True
                or prior.get('error') or declaration['baseline_commit'] != BASELINE):
            raise ValueError('incomplete or wrong baseline/profile')
        config = prior['protocol']
        freeze = prior['baseline']
        if declaration.get('baseline_archive_sha256') != freeze['archive_sha256']:
            raise ValueError('diagnostic baseline archive identity mismatch')
        assert_source_snapshot(source, freeze)
        assert_inventory(source, prior['build_inventory'])
        diagnostic = profile/'source'
        assert_diagnostic(diagnostic, declaration)
        identity = inventory(diagnostic)
        for variant in ('signed', 'f2'):
            key = f'build/metal/shaders/{variant}.metallib'
            if identity[key] != prior['build_inventory'][key]:
                raise ValueError('profiling changed compiled shader bytes')
        execution_source = diagnostic
    else:
        freeze = json.loads((source.parent/'freeze.json').read_text())
        if freeze.get('commit') != BASELINE:
            raise ValueError('wrong baseline snapshot')
        assert_source_snapshot(source, freeze)
        identity = inventory(source)
        execution_source = source
    write_json(output/'protocol.json', config)
    (output/'protocol.sha256').write_text(digest(output/'protocol.json')+'\n')
    harness = {str(p.relative_to(ROOT)): digest(p) for p in
               [Path(__file__), ROOT/'benchmarks/metal/guard.py', ROOT/'benchmarks/metal/application.py',
                ROOT/'tests/metal/smoke.py', ROOT/'tests/metal/verify.py']}
    for name, expected in harness.items():
        destination = output/'harness'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (ROOT/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('harness changed while freezing')
        destination.write_bytes(payload)
    receipt = {'schema': 'fgm-baseline-qualification-v2', 'mode': mode, 'complete': False,
               'baseline': freeze, 'build_inventory': identity, 'hardware': hardware_inventory(),
               'os': subprocess.check_output(['sw_vers'], text=True).strip(),
               'compiler': subprocess.check_output(['xcrun', 'clang++', '--version'], text=True).strip(),
               'protocol': config, 'protocol_sha256': content_hash(config), 'harness': harness,
               'attempts': [], 'performance_acceptance': 'NOT EVALUATED'}
    if prior is not None:
        receipt.update(production_evidence_sha256=digest(production_run/'qualification.json'),
                       profile_declaration_sha256=digest(profile/'profile.json'),
                       qualification_scope='byte-identical shaders, matched inputs, independently verified identical exports; not full-walk equivalence',
                       performance_acceptance='NOT EVALUATED; diagnostic observations only')

    def save():
        write_json(output/'qualification.json', receipt)

    save()
    try:
        for repeat in range(repetitions):
            for row in ROWS:
                name, program, fixture, f2, kernel = row
                if name not in names:
                    continue
                assert_inventory(execution_source, identity)
                fixture_path = source/'tests/metal/fixtures'/fixture
                data = read_raw(fixture_path, f2)
                kind = 'search' if program.startswith('flip_graph') else 'minimizer' if program.startswith('complexity') else 'reducer'
                attempt = output/f'{repeat:02d}-{name}'
                attempt.mkdir()
                adapted = attempt/'input.txt'
                adapted.write_bytes(adapter_bytes(data, kind))
                reference = normalized_input(data) if not f2 else data
                write_json(attempt/'effective-input.json', reference)
                record = {'workload': name, 'mode': mode, 'repeat': repeat, 'complete': False,
                          'fixture_sha256': digest(fixture_path), 'input_sha256': digest(adapted),
                          'expected_kernel': kernel, 'exports': [], 'verification_seconds': None}
                receipt['attempts'].append(record)
                save()
                if prior is not None:
                    original, source_attempt = production_attempt(prior, production_run, name)
                    if original['fixture_sha256'] != record['fixture_sha256'] or original['expected_kernel'] != kernel:
                        raise ValueError('baseline fixture or kernel does not match workload')
                    if digest(source_attempt/'input.txt') != original['input_sha256'] or record['input_sha256'] != original['input_sha256']:
                        raise ValueError('baseline input changed')
                    if json.loads((source_attempt/'effective-input.json').read_text()) != reference:
                        raise ValueError('baseline effective factors changed')
                    record['production_repeat'] = original['repeat']
                run_attempt(row, config, execution_source, attempt, record, save,
                            production=(original, source_attempt) if prior is not None else None)
                save()
                print(name, f"process={record['process_seconds']:.2f}s GPU={record['gpu_work_seconds']:.3f}s exports={len(record['exports'])}", flush=True)
        assert_inventory(execution_source, identity)
        assert_source_snapshot(source, freeze)
        if prior is not None:
            assert_inventory(source, prior['build_inventory'])
            assert_diagnostic(diagnostic, declaration)
        receipt['complete'] = True
        receipt['measurement_acquisition'] = 'complete'
        receipt['qualification_gaps'] = ['practical budgets unapproved']
        if prior is None:
            receipt['qualification_gaps'][:0] = ['host phases need a behavior-qualified diagnostic', 'Metal allocation profile not yet qualified']
        if any(row['workload'] == 'mutation-reducer' and row['applied_mutation_evidence'] == 'NOT VERIFIED' for row in receipt['attempts']):
            receipt['qualification_gaps'].append('actual reducer mutation not established for every attempted case')
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        save()
        (output/'qualification.sha256').write_text(digest(output/'qualification.json')+'\n')
    return receipt


def summarize(path):
    path=Path(path)
    raw=path.read_bytes()
    data=read_qualification(path)
    if data['mode'] != 'production':
        raise ValueError('diagnostic observations are not production repeatability')
    if not data.get('complete') or data.get('error'):
        raise ValueError('incomplete qualification evidence')
    rows=defaultdict(list)
    for entry in data['attempts']:
        if not entry.get('complete') or not entry.get('guard_complete'):
            raise ValueError('incomplete retained attempt')
        rows[entry['workload']].append(entry)
    expected=set(data['protocol']['mandatory_rows'])
    if set(rows)!=expected:
        raise ValueError('missing mandatory workload')
    result={'schema':'fgm-baseline-variability-v1',
            'evidence_sha256':hashlib.sha256(raw).hexdigest(),
            'protocol_sha256':data['protocol_sha256'],
            'performance_verdict':'NOT EVALUATED',
            'practical_regression_allowance':'unapproved; never inferred from repeatability',
            'workloads':{}}
    for name,entries in sorted(rows.items()):
        entries.sort(key=lambda r:r['repeat'])
        if len(entries)<4 or len(entries)%2 or [r['repeat'] for r in entries]!=list(range(len(entries))):
            raise ValueError('need complete sequential baseline pairs')
        metrics={}
        for key in ('gpu_work_seconds','gpu_all_seconds','process_seconds','workflow_seconds',
                    'verification_seconds','peak_process_rss_bytes','peak_system_wired_bytes'):
            values=[r[key] for r in entries]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
                raise ValueError('invalid baseline metric')
            mean=statistics.mean(values)
            metrics[key]={'median':statistics.median(values),'mean':mean,
                          'min':min(values),'max':max(values),'stdev':statistics.stdev(values),
                          'coefficient_of_variation':statistics.stdev(values)/mean if mean else None}
            if min(values)>0:
                pairs=[math.log(values[i+1]/values[i]) for i in range(0,len(values),2)]
                metrics[key]['paired_log_differences']=pairs
                metrics[key]['paired_log_stdev']=statistics.stdev(pairs)
        exports=[[(e['sha256'],e['domain'],e['rank']) for e in row['exports']] for row in entries]
        result['workloads'][name]={'repeats':len(entries),'metrics':metrics,
                                  'identical_verified_exports':all(x==exports[0] for x in exports),
                                  'actual_mutation_verified':all(r['applied_mutation_evidence']=='changed verified circuit factors' for r in entries) if name=='mutation-reducer' else None}
    return result



HOST_WORKLOADS = ('report-100', 'cpu-roundtrip-5', 'select-1000', 'select-10000', 'select-100000')
HOST_LIMITS = {'--record-bytes': 1048576, '--verification-work': 10000000,
               '--selection-memory': 67108864, '--scan-bytes': 268435456}


def host_oracle():
    sys.path.insert(0, str(ROOT/'tests/workflow'))
    import identity_oracle
    return identity_oracle


def host_machine_identity():
    return {'hardware': hardware_inventory(), 'os_version': platform.mac_ver()[0]}


def host_candidate_identity(binary):
    binary = Path(binary).resolve(strict=True)
    receipt_path = binary.with_name(binary.name+'.build.json')
    receipt = json.loads(receipt_path.read_text())
    stat = binary.stat()
    configuration=json.loads((ROOT/'build/workflow/config.json').read_text())
    if receipt['inputs']['configuration']!=configuration:
        raise ValueError('host build configuration changed')
    if Path(configuration['source_dir']).resolve()!=ROOT/'src/workflow':
        raise ValueError('host build source directory mismatch')
    if receipt['output'] != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
        raise ValueError('host executable differs from build receipt')
    for name, checksum in receipt['inputs']['dependencies'].items():
        path = Path(name)
        if digest(path if path.is_absolute() else ROOT/path) != checksum:
            raise ValueError('host build dependency changed: '+name)
    return {'candidate_source': content_hash(source_identity(ROOT)),
            'candidate_build': content_hash({'binary_sha256':digest(binary),
                                             'receipt_sha256':digest(receipt_path)})}


def prepare_host_inputs(source, output):
    """Generate public workload inputs and independent expected identities before timing."""
    oracle = host_oracle()
    output.mkdir(parents=True, exist_ok=False)
    schemes=[]
    for name, f2 in (('rank23_3x3.txt',False),('strassen_4x4.txt',False),('strassen_3x3_f2.txt',True)):
        raw=read_raw(source/'tests/metal/fixtures'/name,f2)
        schemes.append(dict(dimensions=raw['n'],rank=raw['m'],domain='F2' if f2 else 'ZT',
                            orientation='cyclic-w',**{key:raw[key] for key in 'uvw'}))
    schemes.extend(oracle.schoolbook((2,3,4),domain) for domain in ('ZT','F2'))
    paths=[]
    for index,scheme in enumerate(schemes):
        path=output/f'scheme-{index}.json';write_json(path,scheme);paths.append(path)
    report=output/'report.jsonl'
    report.write_text(''.join(json.dumps(schemes[i%5],separators=(',',':'))+'\n' for i in range(100)))
    scalar=output/'scalar.json';write_json(scalar,oracle.schoolbook((1,1,1)))
    scalar_hash=digest(scalar)
    workloads={'report-100':{'input':report,'expected':schemes*20},
               'cpu-roundtrip-5':{'inputs':paths,'expected':schemes}}
    for size in (1000,10000,100000):
        manifest=output/f'inventory-{size}.jsonl'
        with manifest.open('x') as stream:
            for index in range(size):
                row=dict(schema='fgm-collection-v1',namespace='synthetic',id=f'p{index}',
                         path=scalar.name,sha256=scalar_hash,format='json',domain='ZT')
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
        ids=heapq.nsmallest(3,(f'p{i}' for i in range(size)),
                           key=lambda value:oracle.selection_key(7,'synthetic',value))
        workloads[f'select-{size}']={'input':manifest,'expected':[oracle.schoolbook((1,1,1))]*3,
                                   'selected_ids':ids}
    for name,work in workloads.items():
        inputs=work.get('inputs',[work.get('input')])
        if name.startswith('select-'):inputs=inputs+[scalar]
        work['input_files']=inputs
        work['fixture_sha256']=content_hash({path.name:digest(path) for path in inputs})
    return workloads


def verify_host_record(record, expected):
    oracle=host_oracle()
    if any(record.get(key)!=expected[key] for key in ('dimensions','rank','domain','orientation','u','v','w')):
        raise ValueError('host output changed submitted scheme')
    for field,canonical in (('scheme_id',True),('factors_id',False)):
        if record.get(field)!=oracle.identity(expected,canonical):
            raise ValueError('host identity mismatch')
    return verify(dict(n=record['dimensions'],m=record['rank'],z2=record['domain']=='F2',
                       **{key:record[key] for key in 'uvw'}))


def run_host_attempt(binary, name, work, attempt, record, save):
    """One independent native workflow trial, including exact output checks."""
    attempt.mkdir()
    record.update(complete=False,commands=[],outputs=[],native_process_seconds=0.,
                  independent_verification_seconds=0.,peak_process_rss_bytes=0,resource_violation=False)
    limits=[item for key,value in HOST_LIMITS.items() for item in (key,str(value))]
    start=time.monotonic()
    try:
        output_paths=[]
        if name=='cpu-roundtrip-5':
            commands=[]
            for index,(source,scheme) in enumerate(zip(work['inputs'],work['expected'])):
                text=attempt/f'cpu-{index}.txt';result=attempt/f'restored-{index}.jsonl'
                commands.extend([
                    ['export','--input',str(source),'--output',str(text),'--format','json','--output-format','cpu-text'],
                    ['import','--input',str(text),'--output',str(result),'--format','cpu-text','--domain',scheme['domain']]])
                output_paths.append(result)
        else:
            result=attempt/'result.jsonl';output_paths.append(result)
            command=['select' if name.startswith('select-') else 'verify','--input',str(work['input']),
                     '--output',str(result)]
            command += ['--count','3','--seed','7'] if name.startswith('select-') else ['--format','jsonl']
            commands=[command]
        for index,args in enumerate(commands):
            argv=[str(binary),*args,*limits]
            record['commands'].append(argv);save()
            guard_path=attempt/f'guard-{index}'
            guard=guarded_run(['/usr/bin/time','-l','-p',*argv],guard_path)
            if not guard['complete']:
                reason=str(guard.get('error'))
                log_path=guard_path/'run.log'
                native_resource=(guard.get('exit_code')==2 and log_path.is_file()
                                 and any(line.startswith('resource_limit:')
                                         for line in log_path.read_text().splitlines()))
                record['resource_violation']=('time limit' in reason or
                                              'wired memory exceeded' in reason or native_resource)
                raise RuntimeError('incomplete guarded host process: '+reason)
            log=(guard_path/'run.log').read_text()
            elapsed=re.findall(r'^real\s+([0-9.]+)$',log,re.M)
            rss=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',log,re.M)
            if (len(elapsed)!=1 or len(rss)!=1 or not math.isfinite(float(elapsed[0]))
                    or float(elapsed[0])<0 or int(rss[0])<=0):
                raise ValueError('missing native process elapsed/RSS')
            record['native_process_seconds']+=float(elapsed[0])
            record['peak_process_rss_bytes']=max(record['peak_process_rss_bytes'],int(rss[0]))
            record.setdefault('peak_system_wired_bytes',0)
            record['peak_system_wired_bytes']=max(record['peak_system_wired_bytes'],
                                                 max(sample['wired_bytes'] for sample in guard['memory']))
        verify_start=time.monotonic()
        records=[]
        for path in output_paths:
            records.extend(json.loads(line) for line in path.read_text().splitlines())
            record['outputs'].append({'file':path.name,'sha256':digest(path)})
        if len(records)!=len(work['expected']):raise ValueError('host result count mismatch')
        for result,expected in zip(records,work['expected']):verify_host_record(result,expected)
        if name.startswith('select-'):
            ids=[row['source_binding']['id'] for row in records]
            if ids!=work['selected_ids']:raise ValueError('independent selection order mismatch')
            oracle=host_oracle()
            for row,source_id in zip(records,ids):
                if row['source_binding']['selection_key']!=oracle.selection_key(7,'synthetic',source_id)[0].hex():
                    raise ValueError('independent selection digest mismatch')
        record['independent_verification_seconds']=time.monotonic()-verify_start
        record['workflow_seconds']=time.monotonic()-start
        record['complete']=True
    except Exception as error:
        record['error']=str(error)
        raise
    finally:
        save()
    return record


def execute_host(binary, output, repetitions=6):
    """Record repeatable native workflows with exact checks, without performance verdicts."""
    if type(repetitions) is not int or not 1 <= repetitions <= 12:
        raise ValueError('host repetitions must be 1..12')
    binary=binary.resolve(strict=True)
    candidate=host_candidate_identity(binary)
    hardware=host_machine_identity()
    output.mkdir(parents=True,exist_ok=False)
    build_receipt=output/'native-build.json'
    build_receipt.write_bytes(binary.with_name(binary.name+'.build.json').read_bytes())
    binary_hash=digest(binary)
    if content_hash({'binary_sha256':binary_hash,'receipt_sha256':digest(build_receipt)})!=candidate['candidate_build']:
        raise ValueError('host build changed while retaining receipt')
    harness={str(path.relative_to(ROOT)):digest(path) for path in
             (Path(__file__),ROOT/'benchmarks/metal/guard.py',ROOT/'benchmarks/metal/application.py',
              ROOT/'tests/metal/verify.py',ROOT/'tests/workflow/identity_oracle.py')}
    receipt=dict(schema='fgm-host-measurement-v1',complete=False,candidate=candidate,
                 hardware=hardware,harness=harness,source_files=source_identity(ROOT),
                 binary_sha256=binary_hash,build_receipt='native-build.json',
                 build_receipt_sha256=digest(build_receipt),
                 revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                 attempts=[],measurement_scope='descriptive host baseline; no statistical performance verdict')
    def save():write_json(output/'measurement.json',receipt)
    save()
    try:
        work=prepare_host_inputs(ROOT,output/'inputs')
        clock=time.get_clock_info('monotonic')
        protocol=dict(schema='fgm-host-measurement-protocol-v1',workloads=list(HOST_WORKLOADS),
                      repetitions=repetitions,ordering='repeat then listed workload',seed=7,
                      native_resource_limits=HOST_LIMITS,fixtures={name:item['fixture_sha256'] for name,item in work.items()},
                      clock={'implementation':clock.implementation,'resolution':clock.resolution,'monotonic':clock.monotonic},
                      timings={'native_process_seconds':'sum of child process elapsed times reported by time -p',
                               'independent_verification_seconds':'output reading, exact tensor and identity checks',
                               'workflow_seconds':'monotonic elapsed through all native commands and independent checks'},
                      candidate=candidate,hardware=hardware,harness=harness)
        write_json(output/'protocol.json',protocol)
        receipt.update(protocol=protocol,protocol_sha256=content_hash(protocol))
        save()
        for index in range(repetitions):
            for name in HOST_WORKLOADS:
                if host_candidate_identity(binary)!=candidate:raise ValueError('host source/build changed')
                if content_hash({path.name:digest(path) for path in work[name]['input_files']})!=work[name]['fixture_sha256']:
                    raise ValueError('host fixture changed')
                record={'workload':name,'repeat':index,'complete':False,'error':None}
                receipt['attempts'].append(record);save()
                attempt=output/f'{index:02d}-{name}'
                try:run_host_attempt(binary,name,work[name],attempt,record,save)
                finally:
                    if attempt.is_dir():
                        write_json(attempt/'trial.json',record)
                        record['evidence_sha256']=digest(attempt/'trial.json')
                    save()
                if content_hash({path.name:digest(path) for path in work[name]['input_files']})!=work[name]['fixture_sha256']:
                    raise ValueError('host fixture changed')
        if host_candidate_identity(binary)!=candidate:raise ValueError('host source/build changed')
        receipt['complete']=True
    except Exception as error:
        receipt['error']=str(error)
        raise
    finally:
        save();(output/'measurement.sha256').write_text(digest(output/'measurement.json')+'\n')
    return receipt


def native_attempt_config(config, base, attempt):
    """Isolate writable outputs; never resume or write the supplied history."""
    import shutil
    result = copy.deepcopy(config)
    def absolute(value):
        path = Path(value)
        return (path if path.is_absolute() else base/path).resolve(strict=True)
    route = result['input']
    if route['kind'] == 'files':
        for item in route['files']:
            item['path'] = str(absolute(item['path']))
    elif route['kind'] == 'selection':
        route['manifest'] = str(absolute(route['manifest']))
        if 'ids' in route:
            route['ids'] = str(absolute(route['ids']))
    elif route['kind'] == 'resume':
        source = absolute(route['journal'])
        files = list(source.rglob('*'))
        if any(p.is_symlink() or (not p.is_dir() and not p.is_file()) for p in files):
            raise ValueError('resume history contains links or special files')
        budget = result.get('history', {}).get('storage_bytes', 268435456)
        if sum(p.stat().st_size for p in files if p.is_file()) > budget:
            raise ValueError('resume copy exceeds history storage budget')
        shutil.copytree(source, attempt/'history')
        route['journal'] = str(attempt/'history')
    else:
        raise ValueError('unsupported native input route')
    result['output'] = str(attempt/'receipt.json')
    if 'history' in result:
        result['history']['path'] = str(attempt/'history')
    return result


def native_verify_record(data):
    """Independent tensor and identity checks for journal export records."""
    expected = {key:data[key] for key in ('dimensions','rank','domain','orientation','u','v','w')}
    return verify_host_record(data, expected)


def native_verify_circuits(path, receipt, record_limit=1048576):
    artifact = receipt['circuit_artifact']
    if digest(path) != artifact['sha256']:
        raise ValueError('circuit artifact hash mismatch')
    count = 0
    with path.open() as stream:
        while line := stream.readline(record_limit+1):
            if len(line.encode()) > record_limit or not line.endswith('\n'):
                raise ValueError('circuit artifact record exceeds bound')
            data = json.loads(line)
            verify(data)
            result = receipt['results'][count]
            factors = dict(dimensions=data['n'],rank=data['m'],domain='ZT',orientation='cyclic-w',
                           **dict(zip('uvw',circuit_factors(data))))
            identity = host_oracle().identity(factors, False)
            if (result['circuit_record_index'] != count or result['result_factors_id'] != identity
                    or result['result_rank'] != data['m']
                    or result['verified_circuit_additions'] != data['complexity']['reduced']):
                raise ValueError('retained best circuit binding mismatch')
            if result['mode'] == 'fixed':
                source=receipt['presentations'][result['input_presentation_index']]
                native_verify_record(source)
                reference=dict(n=source['dimensions'],m=source['rank'],z2=False,
                               **{key:source['effective_factors'][key] for key in 'uvw'})
                verify(data,reference)
                if identity != result['effective_input_factors_id']:
                    raise ValueError('fixed circuit differs from effective input factors')
            count += 1
    if count != artifact['records'] or count != len(receipt['results']):
        raise ValueError('circuit artifact record count mismatch')
    return count


def native_dispatch_evidence(log, receipt):
    required=receipt['counters'].get('flip_attempts',0)>0 or bool(receipt.get('results'))
    if not re.search(r'^Metal dispatch ',log,re.M):
        if required: raise ValueError('completed native work lacks GPU dispatch evidence')
        return {'status':'GPU_NOT_RUN','gpu_all_seconds':0.,'dispatches':[]}
    evidence=dispatch_evidence(log)
    libraries=re.findall(r'^Metal library: (\S+) SHA256 ([0-9a-f]{64})$',log,re.M)
    if len(libraries)!=1 or libraries[0][1]!=receipt.get('library_sha256'):
        raise ValueError('GPU library identity differs from native receipt')
    names={entry[0] for entry in evidence['dispatches']}
    config=receipt['configuration']
    if config['operation']=='search':
        expected='controlledGeneralKernel'
        if receipt['actual_backend']=='packed':
            expected=('controlledPackedReductionKernel' if config['policy']['mode']=='rank-reduction'
                      else 'controlledPackedAlternativesKernel')
        if expected not in names: raise ValueError('missing expected controlled kernel')
    else:
        if 'initializeReducersKernel' not in names: raise ValueError('missing reducer initialization')
        if any(result['rounds_completed'] for result in receipt['results']):
            expected='runDirectReducersKernel' if config['reduction']['max_flips'] else 'runReducersKernel'
            if expected not in names: raise ValueError('missing expected reducer kernel')
    return dict(evidence,status='GPU_EXECUTED',library=libraries[0],
                gpu_all_seconds=sum(item[2] for item in evidence['dispatches'])/1000)


def execute_native(config_path, binary, output, repetitions=1):
    """Bounded descriptive workflow adapter. No regression budget is implied."""
    if not 1 <= repetitions <= 12:
        raise ValueError('repetitions must be 1..12')
    config_path, binary = config_path.resolve(strict=True), binary.resolve(strict=True)
    tool = binary.with_name('scheme_tool')
    if not tool.is_file():
        raise ValueError('native binary directory requires scheme_tool')
    if config_path.stat().st_size > 1048576:
        raise ValueError('native configuration exceeds 1 MiB')
    config = json.loads(config_path.read_text())
    output = output.absolute()
    output.mkdir(parents=True,exist_ok=False)
    identities = {str(path):digest(path) for path in
                  [binary,tool,*sorted(binary.parent.glob('*.build.json')),
                   *sorted((binary.parent/'shaders').glob('*.metallib'))]}
    evidence = dict(schema='fgm-native-workflow-measurement-v1',complete=False,
                    verdict='NOT EVALUATED',regression_budget=None,
                    config_sha256=digest(config_path),hardware=host_machine_identity(),
                    build_inventory=identities,attempts=[])
    def save():
        write_json(output/'native-workflow.json',evidence)
    save()
    try:
        for index in range(repetitions):
            attempt=output/f'attempt-{index:02d}'
            attempt.mkdir()
            resolved=native_attempt_config(config,config_path.parent,attempt)
            write_json(attempt/'config.json',resolved)
            record=dict(complete=False,commands=[],native_process_seconds=0.,
                        peak_process_rss_bytes=0,peak_system_wired_bytes=0)
            evidence['attempts'].append(record)
            started=time.monotonic()
            def command(argv):
                guard_path=attempt/f'guard-{len(record["commands"])}'
                record['commands'].append(argv);save()
                guard=guarded_run(['/usr/bin/time','-l','-p',*argv],guard_path)
                if not guard['complete']:
                    raise RuntimeError('native guarded process incomplete: '+str(guard.get('error')))
                log=(guard_path/'run.log').read_text()
                elapsed=re.findall(r'^real\s+([0-9.]+)$',log,re.M)
                rss=re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$',log,re.M)
                if len(elapsed)!=1 or len(rss)!=1 or not math.isfinite(float(elapsed[0])) or int(rss[0])<=0:
                    raise ValueError('missing process elapsed/RSS evidence')
                record['native_process_seconds']+=float(elapsed[0])
                record['peak_process_rss_bytes']=max(record['peak_process_rss_bytes'],int(rss[0]))
                record['peak_system_wired_bytes']=max(record['peak_system_wired_bytes'],
                    max(sample['wired_bytes'] for sample in guard['memory']))
                return log
            workflow_log=command([str(binary),'--run-config',str(attempt/'config.json')])
            receipt=json.loads((attempt/'receipt.json').read_text())
            if receipt['status']!='complete' or not receipt['execution_started']:
                raise ValueError('native workflow did not complete')
            if receipt['configuration_sha256']!=digest(attempt/'config.json') or receipt['executable_sha256']!=digest(binary):
                raise ValueError('native receipt configuration/executable identity mismatch')
            record['receipt_sha256']=digest(attempt/'receipt.json')
            record['runtime_identity']={key:receipt[key] for key in
                ('executable_sha256','library_sha256','library_mode','specification_sha256') if key in receipt}
            record['counters']=receipt['counters']
            record['gpu']=native_dispatch_evidence(workflow_log,receipt)
            record['host_phases']={k:v for k,v in receipt.items() if k.endswith('_microseconds')}
            record['requested_backend']=receipt['requested_backend']
            record['actual_backend']=receipt['actual_backend']
            record['presentations']=receipt['presentations']
            for source in receipt['presentations']:
                if all(key in source for key in 'uvw'):
                    native_verify_record(source)
            if resolved['input']['kind']=='files':
                input_hashes={digest(Path(item['path'])) for item in resolved['input']['files']}
                if any(p['source_sha256'] not in input_hashes for p in receipt['presentations']):
                    raise ValueError('admitted source hash differs from input files')
                record['source_sha256']=sorted(input_hashes)
            verification_start=time.monotonic()
            if resolved['operation']=='reduce':
                path=Path(str(attempt/'receipt.json')+'.circuits.jsonl')
                if Path(receipt['circuit_artifact']['path'])!=path:
                    raise ValueError('unexpected circuit artifact path')
                record['verified_records']=native_verify_circuits(path,receipt,
                    resolved.get('limits',{}).get('record_bytes',1048576))
            else:
                history=Path(resolved.get('history',{}).get('path',str(attempt/'receipt.json')+'.journal'))
                if resolved['input']['kind']=='resume':
                    history=Path(resolved['input']['journal'])
                exported=attempt/'committed.jsonl'
                limits=receipt['configuration']['limits']
                transaction_bytes=receipt['configuration']['history']['transaction_bytes']
                command([str(tool),'verify','--format','journal','--input',str(history),
                         '--output',str(exported),'--record-bytes',str(transaction_bytes),
                         '--scan-bytes',str(limits['scan_bytes']),
                         '--verification-work',str(limits['verification_work']),
                         '--selection-memory',str(limits['selection_memory'])])
                count=0
                with exported.open() as stream:
                    while line := stream.readline(1048577):
                        if len(line.encode())>1048576 or not line.endswith('\n'):
                            raise ValueError('journal export record exceeds adapter bound')
                        native_verify_record(json.loads(line));count+=1
                record['verified_records']=count
                record['export_sha256']=digest(exported)
            record['verification_seconds']=time.monotonic()-verification_start
            record['verification_timing_scope']='native journal export where required plus independent exact checks'
            record['workflow_seconds']=time.monotonic()-started
            record['complete']=True
            save()
        if any(digest(Path(path))!=checksum for path,checksum in identities.items()):
            raise ValueError('native build identity changed during measurement')
        evidence['complete']=True
    except Exception as error:
        evidence['error']=str(error)
        raise
    finally:
        save()
    return evidence


FGM1_PANEL = ('original', 'laderman', 'smirnov', 'sun', 'cn122')
FGM1_PROTOCOL = {
    'schema': 'fgm-effectiveness-protocol-v1', 'version': 1,
    'panel': list(FGM1_PANEL), 'seeds': [7, 19, 41],
    'arms': ['fixed', 'generate'], 'arm_seconds': 20, 'endpoints': [10, 20],
    'total_seconds': 900, 'child_reserve_seconds': 50,
    'reducers': 256, 'initial_rounds': 32, 'initial_workers': 32,
    'reduction_pilot_seconds': 1, 'generation_pilot_seconds': 5,
    'search': {'batch_steps': 64, 'flip_budget': 64, 'control_budget': 64,
               'optional_quota': 2, 'excursion': 2, 'interval_min': 4,
               'interval_max': 8, 'stagnation_limit': 100, 'proposal_limit': 64,
               'reduction_q': 0},
    'seed_encoding': 'SHA256 compact ASCII JSON [panel,trial,stage,block], first 4 bytes little endian; zero becomes one',
}
FGM1_PREVIOUS_PROTOCOL = copy.deepcopy(FGM1_PROTOCOL)
FGM1_PROTOCOL.update(version=2, launch_wired_reserve_bytes=1275068416, headroom_poll_seconds=0.025)
FGM1_HEADROOM_PROTOCOL = copy.deepcopy(FGM1_PROTOCOL)
FGM1_PROTOCOL.update(version=3, reducers=128)


def reference_circuit(source, kind):
    """Convert retained author data only; never execute an upstream program."""
    result = dict(n=[3, 3, 3], m=23, z2=False)
    if kind == 'cn122-58':
        result = json.loads(source.read_text())
    elif kind == 'cn122-55':
        certificate = json.loads(source.read_text())
        for key, name in zip('uvw', ('U_input_9_to_23', 'V_input_9_to_23', 'W_output_raw_23_to_9')):
            side = certificate['circuits'][name]
            fresh = []
            for index, gate in enumerate(side['gates'], side['input_count']):
                if gate['slot'] != index:
                    raise ValueError('certificate gate slots are not consecutive')
                fresh.append([dict(index=gate[term], value=gate[term+'_sign'])
                              for term in ('left', 'right')])
            result[key+'_fresh'] = fresh
            result[key] = [[dict(index=item['slot'], value=item['sign'])] for item in side['outputs']]
    elif kind == 'sun-56':
        assignments = [node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SIDES'
                                                               for t in node.targets)]
        if len(assignments) != 1:
            raise ValueError('expected one literal SIDES assignment')
        sides = ast.literal_eval(assignments[0].value)
        for key in 'uvw':
            side = sides[key.upper()]
            result[key+'_fresh'] = [[dict(index=a, value=1), dict(index=b, value=sign)]
                                    for a, sign, b in side['inter']]
            result[key] = [[dict(index=index, value=sign) for index, sign in expression]
                           for expression in side['final']]
        # The source output is row-major; FGM W uses j*3+i.
        result['w'] = [result['w'][i] for i in (0, 3, 6, 1, 4, 7, 2, 5, 8)]
    else:
        raise ValueError('unknown reference circuit')
    factors = dict(n=[3, 3, 3], m=23, z2=False, **dict(zip('uvw', circuit_factors(result))))
    counts = {key: reconstruct(result[key], result[key+'_fresh'], 23 if key == 'w' else 9, False)[1]
              for key in 'uvw'}
    result['complexity'] = dict(naive=verify(factors)['additions'], reduced=sum(counts.values()))
    verify(result, factors)
    return result


def prepare_effectiveness_panel(corpus, output):
    """Freeze five factors and literal reference sources from the retained corpus."""
    corpus = Path(corpus).resolve(strict=True)
    manifest = json.loads((corpus/'manifest.json').read_text())
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('factors', 'circuits', 'sources'):
        (output/name).mkdir()
    def retained(relative, destination):
        source = corpus/relative
        if digest(source) != manifest['artifacts'][relative]:
            raise ValueError('retained source hash mismatch: '+relative)
        shutil.copyfile(source, output/destination)
        return output/destination
    entries = []
    provenance = []
    for name in FGM1_PANEL:
        relative = f'fixtures/{name}.json'
        path = retained(relative, f'factors/{name}.json')
        data = json.loads(path.read_text())
        if set(data) != {'n', 'm', 'z2', 'u', 'v', 'w'} or data['n'] != [3, 3, 3] or data['m'] != 23 or data['z2']:
            raise ValueError('panel input must contain rank-23 signed factors only')
        verify(data)
        entries.append(dict(id=name, path=f'factors/{name}.json', sha256=digest(path),
                            format='json', domain='ZT', references=[]))
        item = next(row for row in manifest['inputs'] if row['id'] == name)
        provenance.append(dict(id=name, source_sha256=digest(path), lineage=item['lineage'],
                               source_metadata=item.get('source_metadata', {})))
    cn = 'sources/data/schemes/source/cn122_add55/34949f9ce50a89a5ad6b47a17f5834ad5a87a2fb/'
    sun = 'sources/data/schemes/source/sun2026_56/2917e6dedb624340a7a75fbb0214627ed545ea84/'
    cases = [
        ('cn122-55', 'cn122', cn+'search_runs/cn122_add55/certificate.json', 'sources/cn122-55.json',
         {'u':13, 'v':14, 'w':28}, 'https://github.com/trylogical/cn122_add55', '34949f9ce50a89a5ad6b47a17f5834ad5a87a2fb'),
        ('cn122-58', 'cn122', cn+'external/FastMatrixMultiplication/schemes/results/addition_reduced_ZT/3x3x3_m23_cr58_cn122_ZT_reduced.json',
         'sources/cn122-58.json', {'u':14, 'v':15, 'w':29}, 'https://github.com/dronperminov/FastMatrixMultiplication', '98ba522db92b74f1f8c561a78038ff3091356d73'),
        ('sun-56', 'sun', sun+'verify.py', 'sources/sun-56.py', {'u':13, 'v':13, 'w':30},
         'https://github.com/sunyinqi0508/3by3r23-56a', '2917e6dedb624340a7a75fbb0214627ed545ea84'),
    ]
    for label, name, relative, dest, counts, repo, commit in cases:
        source = retained(relative, dest)
        circuit = reference_circuit(source, label)
        factor = json.loads((output/f'factors/{name}.json').read_text())
        verify(circuit, factor)
        path = output/f'circuits/{label}.json'
        write_json(path, circuit)
        next(row for row in entries if row['id'] == name)['references'].append(dict(
            id=label, path=f'circuits/{label}.json', sha256=digest(path), source=dest,
            source_sha256=digest(source), repository=repo, commit=commit,
            claimed_additions=sum(counts.values()), claimed_additions_by_stage=counts))
    for relative, name in ((cn+'LICENSE', 'LICENSE-cn122.txt'),
                           (cn+'THIRD_PARTY_LICENSES/Perminov-MIT.txt', 'LICENSE-perminov.txt'),
                           (sun+'LICENSE', 'LICENSE-sun.txt')):
        retained(relative, 'sources/'+name)
    write_json(output/'provenance.json', dict(schema='fgm-panel-provenance-v1', factors=provenance,
        conversion='CN122 uses cyclic-W raw output; Sun row-major outputs permuted to cyclic-W; term order and signs preserved'))
    write_json(output/'protocol.json', FGM1_PROTOCOL)
    panel = dict(schema='fgm-effectiveness-panel-v1', entries=entries, protocol='protocol.json',
                 artifacts={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output/'panel.json', panel)
    return panel


def effectiveness_seed(panel, trial, stage, block):
    encoded = json.dumps([panel, trial, stage, block], separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], 'little') or 1


def effective_factor_id(data):
    factors = normalized_input(data)
    return host_oracle().identity(dict(dimensions=factors['n'], rank=factors['m'], domain='ZT',
                                      orientation='cyclic-w', **{k:factors[k] for k in 'uvw'}), False)


def panel_artifact(root, name, checksum):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('panel paths must stay within the bundle')
    path = root/relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve()):
        raise ValueError('panel artifact escapes bundle')
    if path.stat().st_size > 4*1048576 or digest(path) != checksum:
        raise ValueError('panel artifact hash/size mismatch: '+name)
    return path


def load_effectiveness_panel(path, expected_protocol=None):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 1048576:
        raise ValueError('panel manifest exceeds limit')
    panel = json.loads(path.read_text())
    if panel['schema'] != 'fgm-effectiveness-panel-v1' or [e['id'] for e in panel['entries']] != list(FGM1_PANEL):
        raise ValueError('unexpected effectiveness panel roster')
    for name, checksum in panel['artifacts'].items():
        panel_artifact(path.parent, name, checksum)
    protocol = json.loads((path.parent/panel['protocol']).read_text())
    expected_protocol = expected_protocol or FGM1_PROTOCOL
    if expected_protocol not in (FGM1_PROTOCOL, FGM1_HEADROOM_PROTOCOL, FGM1_PREVIOUS_PROTOCOL) or protocol != expected_protocol:
        raise ValueError('unsupported effectiveness protocol; version changes require implementation and calibration')
    canonical = set()
    expected_references = {'sun': {'sun-56': {'u':13,'v':13,'w':30}},
                           'cn122': {'cn122-55': {'u':13,'v':14,'w':28}, 'cn122-58': {'u':14,'v':15,'w':29}}}
    for entry in panel['entries']:
        expected = expected_references.get(entry['id'], {})
        if len(entry['references']) != len(expected) or {r['id'] for r in entry['references']} != set(expected):
            raise ValueError('reference roster differs from fixed panel')
        factor_path = panel_artifact(path.parent, entry['path'], entry['sha256'])
        if panel['artifacts'].get(entry['path']) != entry['sha256']:
            raise ValueError('panel factor hash declarations disagree')
        data = json.loads(factor_path.read_text())
        if set(data) != {'n', 'm', 'z2', 'u', 'v', 'w'} or data['n'] != [3, 3, 3] or data['m'] != 23 or data['z2'] is not False:
            raise ValueError('panel factors must be signed rank-23 3x3 with no circuit metadata')
        verify(data)
        sid = host_oracle().identity(dict(dimensions=data['n'], rank=data['m'], domain='ZT',
                                          orientation='cyclic-w', **{k:data[k] for k in 'uvw'}))
        if sid in canonical:
            raise ValueError('panel contains canonical aliases')
        canonical.add(sid)
        entry['scheme_id'] = sid
        for reference in entry['references']:
            circuit_path = panel_artifact(path.parent, reference['path'], reference['sha256'])
            source = panel_artifact(path.parent, reference['source'], reference['source_sha256'])
            if reference_circuit(source, reference['id']) != json.loads(circuit_path.read_text()):
                raise ValueError('reference circuit differs from retained source conversion')
            circuit_data = json.loads(circuit_path.read_text())
            result = verify(circuit_data, data)
            if result['additions'] != reference['claimed_additions']:
                raise ValueError('reference source count mismatch')
            counts = {k:reconstruct(circuit_data[k], circuit_data[k+'_fresh'],
                                    23 if k == 'w' else 9, False)[1] for k in 'uvw'}
            if counts != reference['claimed_additions_by_stage']:
                raise ValueError('reference source stage count mismatch')
            if counts != expected[reference['id']]:
                raise ValueError('reference stage counts differ from panel protocol')
    return panel, protocol


def effectiveness_config(entry, seed, calibration, operation, source, history, protocol=None, resume=None):
    protocol = protocol or FGM1_PROTOCOL
    config = dict(schema='fgm-run-v1', operation=operation, output='receipt.json',
                  input={'kind':'files', 'files':[{'path':str(source), 'format':'json', 'domain':'ZT'}]},
                  execution=dict(workers=1, batch_steps=64, block_size=32, backend='general', memory_bytes=536870912))
    if operation == 'reduce':
        rounds = calibration['rounds']
        config['reduction'] = dict(domain='ZT', seed=seed, rounds=rounds, reducers=protocol['reducers'],
                                   schemes=1, max_flips=0, no_improvements=rounds, target_additions=0)
    else:
        policy = copy.deepcopy(protocol['search'])
        policy.pop('batch_steps')
        policy.update(schema='fgm-controlled-config-v1', policy='controlled-v1', mode='alternatives',
                      domain='ZT', dimensions=[3, 3, 3], seed=seed, collection_rank=23, target_rank=None)
        config['policy'] = policy
        config['execution'].update(workers=calibration['workers'], backend='packed')
        config['pool'] = dict(capacity_per_rank=16, reserve_per_rank=16, memory_bytes=1048576,
                              stage_threshold=1, selector='uniform')
        # Reserve bounded evidence for 32 workers and three capture slots each.
        config['history'] = dict(path=str(history), storage_bytes=268435456,
                                 transaction_bytes=8388608, index_memory_bytes=1048576)
        if resume is None:
            resume = history.exists()
        if resume:
            config['input'] = dict(kind='resume', journal=str(history))
    return config


def effectiveness_accounting(observations, seed_id, capture_drops=0):
    """Descriptive capture accounting, independent of circuit scores."""
    optional_seen = {seed_id}
    canonical = {seed_id}
    mandatory_events = set()
    optional_events = set()
    optional = duplicates = seed_hits = evictions = 0
    active = {23:[seed_id]}
    for row in observations:
        sid, rank = row['scheme_id'], row['rank']
        roster = active.setdefault(rank, [])
        if sid not in roster:
            if len(roster) == 16:
                roster.pop(0)
                evictions += 1
            roster.append(sid)
        if rank != 23:
            continue
        canonical.add(sid)
        event = (row['run_id'], row['batch'], row['worker'], row['control'], row['operation'], row['factors_id'])
        if row['mandatory']:
            mandatory_events.add(event)
        else:
            optional_events.add(event)
            optional += 1
            duplicates += sid in optional_seen
            seed_hits += sid == seed_id
            optional_seen.add(sid)
    return dict(distinct_discoveries=len(canonical)-1, optional_rank23_captures=optional,
                optional_duplicates=duplicates, optional_duplicate_rate=duplicates/optional if optional else None,
                optional_seed_rediscoveries=seed_hits,
                mandatory_optional_redundancy=len(mandatory_events & optional_events),
                capture_drops=capture_drops, encounter_drop_rate=capture_drops/(optional+capture_drops) if optional+capture_drops else None,
                active_pool_evictions_derived=evictions,
                eviction_scope='FIFO replay of captured IDs with one rank-23 seed and capacity 16 per rank')


def effectiveness_endpoints(cell, endpoints=(10, 20)):
    result = {}
    for endpoint in endpoints:
        best = {}
        for item in cell['evaluations']:
            if item['verified_seconds'] <= endpoint:
                sid = item['scheme_id']
                if sid not in best or item['additions'] < best[sid]['additions']:
                    best[sid] = item
        counts = defaultdict(int)
        for item in best.values():
            counts[str(item['additions'])] += 1
        discoveries = {item['scheme_id'] for item in cell['discoveries'] if item['verified_seconds'] <= endpoint}
        winner = min(best.values(), key=lambda item:item['additions']) if best else None
        result[str(endpoint)] = dict(best_additions=winner['additions'] if winner else None,
            best_additions_by_stage=winner['additions_by_stage'] if winner else None,
            cost_histogram=dict(sorted(counts.items(), key=lambda kv:int(kv[0]))),
            distinct_discoveries=len(discoveries), evaluated_candidates=len(best),
            unevaluated_discoveries=len(discoveries-set(best)),
            reference_attainment={str(cost):bool(winner and winner['additions'] <= cost)
                                  for cost in cell['reference_costs']})
    return result


class EffectivenessStop(Exception):
    """An orderly scheduling stop, never a successful incomplete measurement."""


class EffectivenessRun:
    def __init__(self, output, binary_dir, clock=None, sleeper=None):
        self.output = Path(output)
        self.binary_dir = Path(binary_dir)
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.started = None
        self.deadline = None
        self.serial = 0

    def check_launch(self, stop=None):
        now = self.clock()
        if self.deadline is not None and self.deadline-now < FGM1_PROTOCOL['child_reserve_seconds']:
            raise EffectivenessStop('overall envelope cleanup reserve')
        if stop is not None and now >= stop:
            raise EffectivenessStop('block or arm deadline')

    def command(self, argv, attempt, record, stop=None):
        wait_started = self.clock()
        try:
            if Path(argv[0]).name in ('flip_graph', 'additions_reducer'):
                self.await_headroom(stop)
            else:
                self.check_launch(stop)
        finally:
            record['headroom_wait_seconds'] = record.get('headroom_wait_seconds', 0.)+self.clock()-wait_started
        guard_dir = attempt/f'guard-{len(record["commands"]):02d}'
        record['commands'].append(argv)
        write_json(attempt/'step.json', record)
        guard = guarded_run(['/usr/bin/time', '-l', '-p', *argv], guard_dir)
        record.setdefault('guards', []).append(dict(path=guard_dir.name,
            result_sha256=digest(guard_dir/'result.json'),
            log_sha256=digest(guard_dir/'run.log') if (guard_dir/'run.log').exists() else None))
        record['peak_system_wired_bytes'] = max(record.get('peak_system_wired_bytes', 0),
                                                max((r['wired_bytes'] for r in guard.get('memory', [])), default=0))
        if not guard['complete']:
            raise RuntimeError('guarded process incomplete: '+str(guard.get('error')))
        log = (guard_dir/'run.log').read_text()
        elapsed = re.findall(r'^real\s+([0-9.]+)$', log, re.M)
        rss = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', log, re.M)
        if len(elapsed) != 1 or len(rss) != 1:
            raise ValueError('missing process time/RSS')
        record['native_process_seconds'] = record.get('native_process_seconds', 0.)+float(elapsed[0])
        record['peak_process_rss_bytes'] = max(record.get('peak_process_rss_bytes', 0), int(rss[0]))
        return log

    def await_headroom(self, stop=None):
        """Wait within the scored budget for observed GPU allocation headroom."""
        started = self.clock()
        while True:
            self.check_launch(stop)
            wired = wired_memory()
            self.check_launch(stop)
            if wired <= LIMIT-FGM1_PROTOCOL['launch_wired_reserve_bytes']:
                return self.clock()-started
            self.sleeper(FGM1_PROTOCOL['headroom_poll_seconds'])

    def invoke(self, config, label, stop=None):
        self.check_launch(stop)
        start = self.clock()
        self.serial += 1
        attempt = self.output/'blocks'/f'{self.serial:05d}-{label}'
        attempt.mkdir(parents=True)
        config = copy.deepcopy(config)
        config['output'] = str(attempt/'receipt.json')
        if config['input']['kind'] == 'files':
            shutil.copyfile(config['input']['files'][0]['path'], attempt/'input.json')
            config['input']['files'][0]['path'] = str(attempt/'input.json')
        write_json(attempt/'config.json', config)
        record = dict(complete=False, operation=config['operation'], commands=[], label=label,
                      started_seconds=start-self.started if self.started is not None else None)
        try:
            binary = self.binary_dir/('flip_graph' if config['operation'] == 'search' else 'additions_reducer')
            log = self.command([str(binary), '--run-config', str(attempt/'config.json')], attempt, record, stop)
            receipt = json.loads((attempt/'receipt.json').read_text())
            record['counters'] = receipt.get('counters', {})
            record['native_receipt_sha256'] = digest(attempt/'receipt.json')
            if receipt['status'] != 'complete' or not receipt['execution_started']:
                raise ValueError('native workflow incomplete')
            if receipt['configuration_sha256'] != digest(attempt/'config.json') or receipt['executable_sha256'] != digest(binary):
                raise ValueError('native workflow identity mismatch')
            if config['input']['kind'] == 'files':
                hashes = {digest(Path(item['path'])) for item in config['input']['files']}
                if any(p['source_sha256'] not in hashes for p in receipt['presentations']):
                    raise ValueError('native admitted source differs from supplied factors')
            record['gpu'] = native_dispatch_evidence(log, receipt)
            record['host_phases'] = {k:v for k,v in receipt.items() if k.endswith('_microseconds')}
            record['native_results'] = receipt.get('results', [])
            verify_start = self.clock()
            if config['operation'] == 'reduce':
                artifact = attempt/'receipt.json.circuits.jsonl'
                native_verify_circuits(artifact, receipt)
                if len(receipt['results']) != 1:
                    raise ValueError('effectiveness reduction requires one presentation')
                circuit = json.loads(artifact.read_text())
                independent = verify(circuit)
                counts = {k:reconstruct(circuit[k], circuit[k+'_fresh'], 23 if k == 'w' else 9, False)[1] for k in 'uvw'}
                if receipt['results'][0]['verified_circuit_additions_by_stage'] != counts:
                    raise ValueError('native/independent stage count mismatch')
                data = dict(dimensions=circuit['n'], rank=circuit['m'], domain='ZT', orientation='cyclic-w',
                            **dict(zip('uvw', circuit_factors(circuit))))
                record['evaluation'] = dict(scheme_id=host_oracle().identity(data),
                    factors_id=host_oracle().identity(data, False), additions=independent['additions'],
                    additions_by_stage=counts, circuit_path=str(artifact.relative_to(self.output)),
                    circuit_sha256=digest(artifact))
            else:
                exported = attempt/'observations.jsonl'
                export_start = self.clock()
                self.command([str(self.binary_dir/'scheme_tool'), 'analyze', '--format', 'journal', '--observations',
                              '--input', config['history']['path'], '--output', str(exported),
                              '--record-bytes', str(config['history']['transaction_bytes']), '--scan-bytes', '268435456'],
                             attempt, record, stop)
                record['observation_export_seconds'] = self.clock()-export_start
                verify_start = self.clock()
                observations = checked_observations(exported)
                record['observations_path'] = str(exported.relative_to(self.output))
                record['observations_sha256'] = digest(exported)
                record['observations'] = observations
            record['verification_seconds'] = self.clock()-verify_start
            record['verified_at_seconds'] = self.clock()-self.started if self.started is not None else None
            record['complete'] = True
        except Exception as error:
            record['error'] = str(error)
            error.effectiveness_record = record
            raise
        finally:
            record['workflow_seconds'] = self.clock()-start
            disk = {k:v for k,v in record.items() if k != 'observations'}
            write_json(attempt/'step.json', disk)
            record['evidence_path'] = str((attempt/'step.json').relative_to(self.output))
            record['evidence_sha256'] = digest(attempt/'step.json')
        return record


def effectiveness_identity(binary_dir, panel_path):
    """Bind calibration to actual build inputs, executables, fixtures and hardware."""
    binary_dir = Path(binary_dir).resolve(strict=True)
    inventory = {}
    for name in ('scheme_tool', 'flip_graph', 'additions_reducer'):
        binary = binary_dir/name
        path = binary.with_name(name+'.build.json')
        build = json.loads(path.read_text())
        stat = binary.stat()
        if build['output'] != {'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns}:
            raise ValueError('build receipt does not match executable: '+name)
        for dependency, checksum in build['inputs']['dependencies'].items():
            source = Path(dependency)
            if digest(source if source.is_absolute() else ROOT/source) != checksum:
                raise ValueError('build dependency changed: '+dependency)
        inventory[name] = digest(binary)
        inventory[path.name] = digest(path)
    for library in sorted((binary_dir/'shaders').glob('*.metallib')):
        inventory[str(library.relative_to(binary_dir))] = digest(library)
    if not inventory.get('shaders/signed.metallib'):
        raise ValueError('effectiveness measurements require compiled signed Metal library')
    source_hashes = source_identity(ROOT)
    source_hashes['docs/specifications/FGM-CONTRACT-v1.md'] = digest(ROOT/'docs/specifications/FGM-CONTRACT-v1.md')
    return dict(panel_sha256=digest(panel_path), protocol_sha256=content_hash(FGM1_PROTOCOL),
                build_inventory=inventory, source_inventory=source_hashes, hardware=host_machine_identity())


def checked_observations(path):
    rows = []
    with path.open() as stream:
        while line := stream.readline(1048577):
            if len(line.encode()) > 1048576 or not line.endswith('\n'):
                raise ValueError('observation record exceeds limit')
            row = json.loads(line)
            if row['schema'] != 'fgm-journal-observation-v1':
                raise ValueError('unexpected observation schema')
            data = row['scheme']
            verify(dict(n=data['dimensions'],m=data['rank'],z2=data['domain']=='F2', **{k:data[k] for k in 'uvw'}))
            if row['scheme_id'] != host_oracle().identity(data) or row['factors_id'] != host_oracle().identity(data, False):
                raise ValueError('observation identity mismatch')
            if row['rank'] != data['rank'] or row['domain'] != data['domain']:
                raise ValueError('observation rank/domain mismatch')
            rows.append(row)
    return rows


def retained_effectiveness_layout(base, reference, config):
    """Resolve relocated evidence without following paths recorded by a past run."""
    parts = PurePosixPath(reference['path']).parts
    if len(parts) < 3 or parts[-3] != 'blocks' or parts[-1] != 'step.json' or any(
            part != 'calibration-evidence' for part in parts[:-3]):
        raise ValueError('unexpected effectiveness step location')
    output = Path(config['output'])
    if not output.is_absolute() or output.parts[-3:] != ('blocks', parts[-2], 'receipt.json'):
        raise ValueError('configuration output differs from retained step')
    return Path(base).joinpath(*parts[:-3]), output.parents[2]


def checked_effectiveness_config(base, reference, step, entry, seed, settings, operation,
                                 history, protocol, resume=False):
    config = step['config']
    _, recorded_root = retained_effectiveness_layout(base, reference, config)
    output = Path(config['output'])
    expected = effectiveness_config(entry, seed, settings, operation, output.with_name('input.json'),
                                    recorded_root/history, protocol=protocol, resume=resume)
    expected['output'] = str(output)
    if config != expected or step['operation'] != operation:
        raise ValueError('step configuration differs from calibrated deterministic protocol')
    if 'receipt' in step:
        resolved = copy.deepcopy(expected)
        resolved.setdefault('history', dict(path=str(output)+'.journal', storage_bytes=536870912,
            transaction_bytes=1048576, index_memory_bytes=1048576))
        resolved.update(discovery_target=None, limits=dict(record_bytes=1048576, scan_bytes=268435456,
            selection_memory=67108864, verification_work=10000000))
        if step['receipt']['configuration'] != resolved:
            raise ValueError('receipt configuration differs from calibrated deterministic protocol')
    if step.get('commands'):
        binary = 'additions_reducer' if operation == 'reduce' else 'flip_graph'
        expected_command = [str(recorded_root/'binaries'/binary), '--run-config', str(output.with_name('config.json'))]
        if step['commands'][0] != expected_command:
            raise ValueError('step command differs from retained configuration')


def retained_effectiveness_history(base, reference, config):
    retained_root, recorded_root = retained_effectiveness_layout(base, reference, config)
    relative = Path(config['history']['path']).relative_to(recorded_root)
    if not relative.parts or '..' in relative.parts or relative.parts[0] not in ('cells', 'pilots'):
        raise ValueError('journal is not owned by this effectiveness run')
    history = retained_root/relative
    if not history.resolve(strict=True).is_relative_to(Path(base).resolve()):
        raise ValueError('journal escapes retained evidence')
    return history


def checked_effectiveness_journal(base, reference, config, receipt_path, observations):
    history = retained_effectiveness_history(base, reference, config)
    with tempfile.TemporaryDirectory(prefix='fgm-journal-replay-') as temporary:
        exported = Path(temporary)/'observations.jsonl'
        result = subprocess.run([str(ROOT/'build/metal/scheme_tool'), 'analyze', '--format', 'journal',
            '--observations', '--receipt', str(receipt_path), '--input', str(history), '--output', str(exported),
            '--record-bytes', str(config['history']['transaction_bytes']), '--scan-bytes', '268435456'],
            capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise ValueError('journal receipt replay failed: '+result.stderr.strip())
        if observations is not None and checked_observations(exported) != observations:
            raise ValueError('observation export differs from acknowledged journal prefix')


def checked_effectiveness_step(base, reference, bindings=None, expected_reducers=None):
    path = panel_artifact(Path(base), reference['path'], reference['sha256'])
    step = json.loads(path.read_text())
    for field in ('workflow_seconds', 'started_seconds'):
        if type(step[field]) not in (int, float) or not math.isfinite(step[field]) or step[field] < 0:
            raise ValueError('invalid retained step timing')
    config_path = path.parent/'config.json'
    if config_path.exists():
        step['config'] = json.loads(config_path.read_text())
    if not step['complete']:
        if 'native_receipt_sha256' in step:
            receipt_path = path.parent/'receipt.json'
            if digest(receipt_path) != step['native_receipt_sha256']:
                raise ValueError('partial step receipt hash mismatch')
            receipt = json.loads(receipt_path.read_text())
            if digest(config_path) != receipt['configuration_sha256'] or step['counters'] != receipt['counters']:
                raise ValueError('partial step counters differ from receipt')
            step['receipt'] = receipt
            if step['operation'] == 'search' and receipt.get('run_id'):
                checked_effectiveness_journal(base, reference, step['config'], receipt_path, None)
        return step
    receipt_path = path.parent/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if digest(receipt_path) != step['native_receipt_sha256'] or digest(config_path) != receipt['configuration_sha256']:
        raise ValueError('step receipt/configuration hash mismatch')
    config = json.loads(config_path.read_text())
    step['receipt'] = receipt
    if receipt['status'] != 'complete' or receipt['configuration']['operation'] != step['operation']:
        raise ValueError('step operation or completion mismatch')
    if not step.get('guards') or len(step['guards']) != len(step['commands']):
        raise ValueError('missing guarded command evidence')
    for guard, command in zip(step['guards'], step['commands']):
        folder = path.parent/guard['path']
        if digest(folder/'result.json') != guard['result_sha256'] or digest(folder/'run.log') != guard['log_sha256']:
            raise ValueError('guard artifact hash mismatch')
        guarded = json.loads((folder/'result.json').read_text())
        if guarded['complete'] is not True or guarded['argv'] != ['/usr/bin/time','-l','-p',*command]:
            raise ValueError('completed step has incomplete guard')
    if bindings is not None:
        binary = 'additions_reducer' if step['operation']=='reduce' else 'flip_graph'
        if receipt['executable_sha256'] != bindings['build_inventory'][binary]:
            raise ValueError('native executable differs from frozen build')
    if config['input']['kind'] == 'files':
        source = path.parent/'input.json'
        source_data = json.loads(source.read_text())
        verify(source_data)
        if not receipt['presentations'] or any(p['source_sha256'] != digest(source) for p in receipt['presentations']):
            raise ValueError('native input differs from retained source')
        step['input_factors_id'] = effective_factor_id(source_data)
    gpu = json.loads(json.dumps(native_dispatch_evidence((path.parent/step['guards'][0]['path']/'run.log').read_text(), receipt)))
    if step['gpu'] != gpu or step['counters'] != receipt['counters']:
        raise ValueError('step GPU/counter evidence differs from native artifacts')
    if step['host_phases'] != {k:v for k,v in receipt.items() if k.endswith('_microseconds')}:
        raise ValueError('step phase timings differ from native receipt')
    timestamp = step['verified_at_seconds']
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or not step['started_seconds'] <= timestamp <= step['started_seconds']+step['workflow_seconds']:
        raise ValueError('step verification time outside retained interval')
    if step['operation'] == 'reduce':
        reduction = config['reduction']
        expected_reducers = FGM1_PROTOCOL['reducers'] if expected_reducers is None else expected_reducers
        if any(reduction[k] != value for k,value in dict(max_flips=0,schemes=1,target_additions=0,
                no_improvements=reduction['rounds'],reducers=expected_reducers).items()):
            raise ValueError('reduction quantum differs from fixed-factor protocol')
        circuit_path = path.parent/'receipt.json.circuits.jsonl'
        native_verify_circuits(circuit_path, receipt)
        circuit = json.loads(circuit_path.read_text())
        result = verify(circuit)
        factors = dict(dimensions=circuit['n'],rank=circuit['m'],domain='ZT',orientation='cyclic-w',
                       **dict(zip('uvw', circuit_factors(circuit))))
        counts = result['additions_by_stage']
        expected = dict(scheme_id=host_oracle().identity(factors),factors_id=host_oracle().identity(factors,False),
                        additions=result['additions'],additions_by_stage=counts,circuit_sha256=digest(circuit_path))
        if any(step['evaluation'][k] != v for k,v in expected.items()) or receipt['results'][0]['verified_circuit_additions_by_stage'] != counts:
            raise ValueError('retained evaluation differs from independently verified circuit')
        if step['evaluation']['factors_id'] != step['input_factors_id']:
            raise ValueError('reduction did not preserve retained input presentation')
    else:
        observations = path.parent/'observations.jsonl'
        if digest(observations) != step['observations_sha256']:
            raise ValueError('observation artifact hash mismatch')
        step['observations'] = checked_observations(observations)
        checked_effectiveness_journal(base, reference, config, receipt_path, step['observations'])
    step['config'] = config
    return step


def calibration_valid(calibration, bindings, base=None):
    if calibration.get('schema') != 'fgm-effectiveness-calibration-v1' or calibration.get('complete') is not True:
        return False
    if calibration.get('bindings') != bindings:
        return False
    settings = calibration.get('settings', {})
    if any(type(settings.get(k)) is not int or settings[k] not in (1, 2, 4, 8, 16, 32) for k in ('rounds', 'workers')):
        return False
    for kind, setting, limit in (('reduce', settings['rounds'], 1), ('search', settings['workers'], 5)):
        rows = [row for row in calibration.get('pilots', []) if row['kind'] == kind and row['setting'] == setting]
        if [row['panel'] for row in rows] != list(FGM1_PANEL):
            return False
        if any(row.get('complete') is not True or type(row['seconds']) not in (int,float)
               or not math.isfinite(row['seconds']) or not 0 <= row['seconds'] <= limit
               or not row.get('steps') for row in rows):
            return False
    if base is not None:
        protocol = next((p for p in (FGM1_PROTOCOL, FGM1_HEADROOM_PROTOCOL, FGM1_PREVIOUS_PROTOCOL)
                         if content_hash(p)==bindings.get('protocol_sha256')), None)
        if protocol is None:
            return False
        for row in calibration['pilots']:
            checked_steps = []
            for index, step in enumerate(row.get('steps', [])):
                try:
                    checked = checked_effectiveness_step(base, step, bindings, expected_reducers=protocol['reducers'])
                    if not checked.get('complete'):
                        return False
                    operation = row['kind'] if index == 0 else 'reduce'
                    stage = 'pilot-'+row['kind'] if index == 0 else 'pilot-evaluate'
                    seed = effectiveness_seed(row['panel'], 7, stage, row['setting'] if index == 0 else index+1)
                    quantum = dict(settings, rounds=row['setting'] if row['kind']=='reduce' else settings['rounds'],
                                   workers=row['setting'] if row['kind']=='search' else settings['workers'])
                    checked_effectiveness_config(base, step, checked, None, seed, quantum, operation,
                        Path('pilots')/f'{row["kind"]}-{row["setting"]}-{row["panel"]}'/'history', protocol)
                    checked_steps.append(checked)
                except (KeyError, ValueError, OSError):
                    return False
            if sum(step['workflow_seconds'] for step in checked_steps) > row['seconds']:
                return False
            if checked_steps and checked_steps[-1]['started_seconds']+checked_steps[-1]['workflow_seconds']-checked_steps[0]['started_seconds'] > row['seconds']:
                return False
            if row['complete'] and row['seconds'] <= (1 if row['kind']=='reduce' else 5):
                initial = json.loads((Path(base)/'panel'/f'factors/{row["panel"]}.json').read_text())
                if not checked_steps or checked_steps[0].get('input_factors_id') != effective_factor_id(initial):
                    return False
                if row['kind'] == 'reduce':
                    if len(checked_steps) != 1 or checked_steps[0]['operation'] != 'reduce':
                        return False
                else:
                    if not checked_steps or checked_steps[0]['operation'] != 'search':
                        return False
                    initial = json.loads((Path(base)/'panel'/f'factors/{row["panel"]}.json').read_text())
                    seed_id = host_oracle().identity(dict(dimensions=initial['n'],rank=23,domain='ZT',orientation='cyclic-w',
                                                         **{k:initial[k] for k in 'uvw'}))
                    pending = {}
                    for observation in checked_steps[0]['observations']:
                        sid = observation['scheme_id']
                        if observation['rank']==23 and sid != seed_id and sid not in pending:
                            scheme = observation['scheme']
                            pending[sid] = effective_factor_id(dict(n=scheme['dimensions'], m=23, z2=False,
                                                                  **{k:scheme[k] for k in 'uvw'}))
                    if [(s['evaluation']['scheme_id'],s['evaluation']['factors_id']) for s in checked_steps[1:]] != list(pending.items()):
                        return False
    return True


def checked_effectiveness_calibration(base, data):
    calibration_path = Path(base)/'calibration.json'
    if digest(calibration_path) != data.get('calibration_sha256'):
        raise ValueError('measurement calibration hash mismatch')
    calibration = json.loads(calibration_path.read_text())
    bindings = data['bindings']
    if bindings['protocol_sha256'] != content_hash(data['protocol']):
        raise ValueError('measurement protocol binding mismatch')
    for folder, key in (('binaries', 'build_inventory'), ('source', 'source_inventory')):
        for name, checksum in bindings[key].items():
            panel_artifact(Path(base)/folder, name, checksum)
    if not calibration_valid(calibration, bindings, base):
        raise ValueError('retained calibration or bindings are invalid')
    return calibration


def candidate_file(run, row):
    data = row['scheme']
    path = run.output/'candidates'/(row['factors_id'].split(':')[-1]+'.json')
    factor = dict(n=data['dimensions'], m=data['rank'], z2=data['domain']=='F2', **{k:data[k] for k in 'uvw'})
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != factor:
            raise ValueError('ordered candidate identity collision')
    else:
        write_json(path, factor)
    return path


def calibrate_effectiveness(run, entries, panel_root, bindings, save):
    calibration = dict(schema='fgm-effectiveness-calibration-v1', complete=False,
                       bindings=bindings, settings={'rounds':32, 'workers':32}, pilots=[])
    for kind, setting_name, limit in (('reduce', 'rounds', 1), ('search', 'workers', 5)):
        while True:
            setting = calibration['settings'][setting_name]
            passed = True
            for entry in entries:
                run.check_launch()
                started = run.clock()
                stop = started+limit
                row = dict(kind=kind, setting=setting, panel=entry['id'], complete=False, steps=[])
                calibration['pilots'].append(row)
                save(calibration)
                source = panel_root/entry['path']
                history = run.output/'pilots'/f'{kind}-{setting}-{entry["id"]}'/'history'
                history.parent.mkdir(parents=True, exist_ok=True)
                try:
                    seed = effectiveness_seed(entry['id'], 7, 'pilot-'+kind, setting)
                    config = effectiveness_config(entry, seed, calibration['settings'], kind, source, history)
                    record = run.invoke(config, 'pilot-'+kind, stop)
                    row['steps'].append({'path':record['evidence_path'], 'sha256':record['evidence_sha256']})
                    if kind == 'search':
                        seen = {entry['scheme_id']}
                        for observation in record['observations']:
                            if observation['rank'] != 23 or observation['scheme_id'] in seen:
                                continue
                            seen.add(observation['scheme_id'])
                            path = candidate_file(run, observation)
                            config = effectiveness_config(entry, effectiveness_seed(entry['id'], 7, 'pilot-evaluate', len(seen)),
                                                          calibration['settings'], 'reduce', path, history)
                            reduced = run.invoke(config, 'pilot-evaluate', stop)
                            row['steps'].append({'path':reduced['evidence_path'], 'sha256':reduced['evidence_sha256']})
                    row['complete'] = True
                except EffectivenessStop as error:
                    row['error'] = str(error)
                    if run.deadline-run.clock() < FGM1_PROTOCOL['child_reserve_seconds']:
                        raise
                finally:
                    row['seconds'] = run.clock()-started
                    row['passed'] = row['complete'] and row['seconds'] <= limit
                    passed &= row['passed']
                    save(calibration)
            if passed:
                break
            if setting == 1:
                raise RuntimeError(f'calibration cannot fit {kind} at minimum quantum')
            calibration['settings'][setting_name] //= 2
    calibration['complete'] = True
    save(calibration)
    return calibration


def run_effectiveness_cell(run, cell, entry, panel_root, settings, save):
    started = run.clock()
    stop = started+FGM1_PROTOCOL['arm_seconds']
    cell.update(status='running', evaluations=[], discoveries=[], observations=[], steps=[], counters={},
                started_seconds=started-run.started)
    history = run.output/'cells'/cell['id']/'history'
    history.parent.mkdir(parents=True, exist_ok=True)
    queue = [(entry['scheme_id'], panel_root/entry['path'])]
    seen = {entry['scheme_id']}
    observed = set()
    block = reduction = 0
    save()
    try:
        while run.clock() < stop:
            run.check_launch(stop)
            if queue or cell['arm'] == 'fixed':
                sid, source = queue[0] if queue else (entry['scheme_id'], panel_root/entry['path'])
                config = effectiveness_config(entry, effectiveness_seed(entry['id'], cell['seed'], 'reduce', reduction),
                                              settings, 'reduce', source, history)
                record = run.invoke(config, cell['id']+'-reduce', stop)
                evaluated = record['evaluation']
                if evaluated['scheme_id'] != sid:
                    raise ValueError('evaluation canonical identity differs from queued factors')
                evaluated['verified_seconds'] = record['verified_at_seconds']-cell['started_seconds']
                cell['evaluations'].append(evaluated)
                if queue:
                    queue.pop(0)
                reduction += 1
            else:
                config = effectiveness_config(entry, effectiveness_seed(entry['id'], cell['seed'], 'search', block),
                                              settings, 'search', panel_root/entry['path'], history)
                record = run.invoke(config, cell['id']+'-search', stop)
                captured_at = record['verified_at_seconds']-cell['started_seconds']
                for row in record['observations']:
                    key = (row['sequence'], row['worker'], row['slot'])
                    if key in observed:
                        continue
                    observed.add(key)
                    cell['observations'].append({k:v for k,v in row.items() if k != 'scheme'})
                    if row['rank'] == 23 and row['scheme_id'] not in seen:
                        seen.add(row['scheme_id'])
                        queue.append((row['scheme_id'], candidate_file(run, row)))
                        cell['discoveries'].append(dict(scheme_id=row['scheme_id'], factors_id=row['factors_id'],
                                                       verified_seconds=captured_at))
                for key, value in record['counters'].items():
                    if key != 'discoveries_historical':
                        cell['counters'][key] = cell['counters'].get(key, 0)+value
                block += 1
            cell['steps'].append(dict(path=record['evidence_path'], sha256=record['evidence_sha256']))
        cell['status'] = 'complete'
    except EffectivenessStop as error:
        record = getattr(error, 'effectiveness_record', None)
        if record:
            cell['steps'].append(dict(path=record['evidence_path'], sha256=record['evidence_sha256']))
            cell['unexported_native_discoveries'] = record.get('counters', {}).get('discoveries_current_run', 0)
            if record['operation'] == 'search':
                for key, value in record.get('counters', {}).items():
                    if key != 'discoveries_historical':
                        cell['counters'][key] = cell['counters'].get(key, 0)+value
        cell['status'] = 'complete' if run.clock() >= stop else 'incomplete'
        cell['stop_reason'] = str(error)
    except Exception as error:
        record = getattr(error, 'effectiveness_record', None)
        if record:
            cell['steps'].append(dict(path=record['evidence_path'], sha256=record['evidence_sha256']))
            if record['operation'] == 'search':
                for key, value in record.get('counters', {}).items():
                    if key != 'discoveries_historical':
                        cell['counters'][key] = cell['counters'].get(key, 0)+value
        cell['status'] = 'failed'
        cell['error'] = str(error)
        raise
    finally:
        cell['workflow_seconds'] = run.clock()-started
        cell['pending_scheme_ids'] = [sid for sid, _ in queue]
        cell['endpoints'] = effectiveness_endpoints(cell)
        cell['capture_accounting'] = effectiveness_accounting(cell['observations'], entry['scheme_id'],
                                                              cell['counters'].get('capture_drops', 0))
        write_json(history.parent/'cell.json', cell)
        save()


def validate_effectiveness_timeline(data, checked_cells):
    """Check that retained arm and step intervals fit the measurement clock."""
    tolerance = 1e-8

    def seconds(record, field):
        if field not in record:
            raise ValueError(f'missing {field} timing')
        value = record[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {field} timing')
        return value

    elapsed = seconds(data, 'elapsed_seconds')
    finalization = seconds(data, 'finalization_seconds') if 'finalization_seconds' in data else 0
    total = seconds(data['protocol'], 'total_seconds')
    if data['complete'] and elapsed > total + tolerance:
        raise ValueError('completed measurement exceeds protocol time')

    previous_arm_end = 0
    for cell, checked_steps in checked_cells:
        arm_start = seconds(cell, 'started_seconds')
        arm_duration = seconds(cell, 'workflow_seconds')
        arm_end = arm_start + arm_duration
        if not math.isfinite(arm_end):
            raise ValueError('invalid arm end timing')
        if arm_start < previous_arm_end - tolerance:
            raise ValueError('measurement arms overlap or are out of order')
        previous_step_end = arm_start
        durations = []
        for step in checked_steps:
            step_start = seconds(step, 'started_seconds')
            step_duration = seconds(step, 'workflow_seconds')
            step_end = step_start + step_duration
            if not math.isfinite(step_end):
                raise ValueError('invalid step end timing')
            if step_start < previous_step_end - tolerance:
                raise ValueError('measurement steps overlap or are out of order')
            if step_end > arm_end + tolerance:
                raise ValueError('measurement step extends beyond its arm')
            previous_step_end = step_end
            durations.append(step_duration)
        if math.fsum(durations) > arm_duration + tolerance:
            raise ValueError('measurement steps exceed arm elapsed time')
        previous_arm_end = arm_end

    if previous_arm_end + finalization > elapsed + tolerance:
        raise ValueError('measurement arms and finalization exceed elapsed time')


def summarize_effectiveness(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    if data['schema'] != 'fgm-effectiveness-measurement-v1':
        raise ValueError('not an effectiveness measurement')
    panel, protocol = load_effectiveness_panel(path.parent/'panel/panel.json', data['protocol'])
    if data['protocol'] != protocol or digest(path.parent/'panel/panel.json') != data['bindings']['panel_sha256']:
        raise ValueError('measurement panel/protocol binding mismatch')
    entries = {entry['id']:entry for entry in panel['entries']}
    calibration = checked_effectiveness_calibration(path.parent, data) if any(
        'evaluations' in cell for cell in data['cells']) else None
    checked_cells = []
    rows = []
    for cell in data['cells']:
        if 'evaluations' not in cell and (cell['status'] != 'unrun' or
                set(cell) - {'id', 'panel', 'seed', 'arm', 'status', 'reference_costs'}):
            raise ValueError('missing evaluations: only an unrun cell without results may omit evidence')
        entry = entries[cell['panel']]
        if cell['id'] != f'{cell["panel"]}-{cell["seed"]}-{cell["arm"]}':
            raise ValueError('cell identity differs from panel/seed/arm')
        if cell['reference_costs'] != [r['claimed_additions'] for r in entry['references']]:
            raise ValueError('cell reference thresholds differ from panel')
        costs = dict(generation_seconds=0., reduction_seconds=0., independent_verification_seconds=0.,
                     observation_export_seconds=0.,
                     headroom_wait_seconds=0.,
                     native_process_seconds=0., gpu_seconds=0., host_phases_microseconds={},
                     peak_process_rss_bytes=0, peak_system_wired_bytes=0)
        if 'evaluations' in cell:
            evaluations, discoveries, observations, counters = [], [], [], {}
            seen = {entry['scheme_id']}
            observed = set()
            initial = json.loads((path.parent/'panel'/entry['path']).read_text())
            queue = [(entry['scheme_id'], effective_factor_id(initial))]
            checked_steps = []
            block = reduction = 0
            for index, reference in enumerate(cell['steps']):
                step = checked_effectiveness_step(path.parent, reference, data['bindings'], expected_reducers=protocol['reducers'])
                checked_steps.append(step)
                operation = 'reduce' if queue or cell['arm']=='fixed' else 'search'
                if 'config' in step:
                    seed = effectiveness_seed(entry['id'], cell['seed'], operation, reduction if operation=='reduce' else block)
                    checked_effectiveness_config(path.parent, reference, step, entry, seed, calibration['settings'],
                        operation, Path('cells')/cell['id']/'history', protocol, resume=block > 0)
                elif step['complete'] or step.get('commands'):
                    raise ValueError('missing step configuration')
                if step.get('commands') and (step['started_seconds'] < cell['started_seconds'] or step['started_seconds'] >= cell['started_seconds']+20):
                    raise ValueError('step launched outside its arm window')
                key = 'generation_seconds' if step['operation']=='search' else 'reduction_seconds'
                costs[key] += step['workflow_seconds']
                costs['independent_verification_seconds'] += step.get('verification_seconds', 0.)
                costs['observation_export_seconds'] += step.get('observation_export_seconds', 0.)
                costs['headroom_wait_seconds'] += step.get('headroom_wait_seconds', 0.)
                costs['native_process_seconds'] += step.get('native_process_seconds', 0.)
                costs['gpu_seconds'] += step.get('gpu', {}).get('gpu_all_seconds', 0.)
                for key,value in step.get('host_phases', {}).items():
                    costs['host_phases_microseconds'][key] = costs['host_phases_microseconds'].get(key, 0)+value
                for key in ('peak_process_rss_bytes','peak_system_wired_bytes'):
                    costs[key] = max(costs[key], step.get(key, 0))
                if step['operation']=='search':
                    for key,value in step.get('counters', {}).items():
                        if key != 'discoveries_historical':
                            counters[key] = counters.get(key, 0)+value
                if not step['complete']:
                    if index != len(cell['steps'])-1:
                        raise ValueError('arm continued after an incomplete step')
                    continue
                when = step['verified_at_seconds']-cell['started_seconds']
                if step['operation']=='reduce':
                    item = dict(step['evaluation'], verified_seconds=when)
                    expected_id, expected_factors = queue.pop(0) if queue else (entry['scheme_id'],effective_factor_id(initial))
                    if item['scheme_id'] != expected_id or item['factors_id'] != expected_factors or (cell['arm']=='generate' and item['scheme_id'] in {e['scheme_id'] for e in evaluations}):
                        raise ValueError('reduction does not follow the cost-blind candidate queue')
                    evaluations.append(item)
                    reduction += 1
                else:
                    if cell['arm'] != 'generate' or queue:
                        raise ValueError('search launched before draining evaluations')
                    if block == 0 and step.get('input_factors_id') != effective_factor_id(initial):
                        raise ValueError('search did not start from the panel factors')
                    block += 1
                    for item in step['observations']:
                        key = (item['sequence'],item['worker'],item['slot'])
                        if key in observed:
                            continue
                        observed.add(key)
                        observations.append({k:v for k,v in item.items() if k!='scheme'})
                        if item['rank']==23 and item['scheme_id'] not in seen:
                            seen.add(item['scheme_id'])
                            factors = dict(n=item['scheme']['dimensions'],m=23,z2=False, **{k:item['scheme'][k] for k in 'uvw'})
                            queue.append((item['scheme_id'], effective_factor_id(factors)))
                            discoveries.append(dict(scheme_id=item['scheme_id'],factors_id=item['factors_id'],verified_seconds=when))
            if evaluations != cell['evaluations'] or discoveries != cell['discoveries'] or observations != cell['observations'] or counters != cell['counters']:
                raise ValueError('cell observations/evaluations differ from retained steps')
            if [item[0] for item in queue] != cell['pending_scheme_ids']:
                raise ValueError('pending queue differs from retained evidence')
            expected = effectiveness_endpoints(cell)
            accounting = effectiveness_accounting(observations, entry['scheme_id'], counters.get('capture_drops', 0))
            if expected != cell['endpoints'] or accounting != cell['capture_accounting']:
                raise ValueError('endpoint/capture summary disagrees with completed evidence')
            costs['complete_elapsed_seconds'] = cell['workflow_seconds']
            if cell['status']=='complete' and cell['workflow_seconds'] < protocol['arm_seconds']:
                raise ValueError('completed arm did not reach its endpoint')
            costs['coordination_seconds'] = cell['workflow_seconds']-costs['generation_seconds']-costs['reduction_seconds']
            checked_cells.append((cell, checked_steps))
        row = {k:cell[k] for k in ('id','panel','seed','arm','status','endpoints','capture_accounting','unexported_native_discoveries') if k in cell}
        row['costs'] = costs
        row['late_evaluations'] = sum(e['verified_seconds']>20 for e in cell.get('evaluations', []))
        row['improved_between_endpoints'] = bool(cell.get('endpoints',{}).get('10',{}).get('best_additions') is not None and
            cell['endpoints']['20']['best_additions'] < cell['endpoints']['10']['best_additions'])
        rows.append(row)
    roster = {(p,s,a) for p in FGM1_PANEL for s in (7,19,41) for a in ('fixed','generate')}
    if len(rows) != 30 or {(r['panel'],r['seed'],r['arm']) for r in rows} != roster:
        raise ValueError('measurement roster differs from protocol')
    expected_order = [(p, s, a) for si, s in enumerate(protocol['seeds'])
        for pi, p in enumerate(FGM1_PANEL)
        for a in (('fixed','generate') if (si+pi)%2 == 0 else ('generate','fixed'))]
    if [(r['panel'],r['seed'],r['arm']) for r in rows] != expected_order:
        raise ValueError('arm order differs from frozen protocol')
    validate_effectiveness_timeline(data, checked_cells)
    acquired = all(row['status'] == 'complete' for row in rows)
    if data['complete'] and not acquired:
        raise ValueError('measurement completion disagrees with cell roster')
    complete = data['complete'] and acquired
    aggregates = {}
    for arm in ('fixed', 'generate'):
        entries = [r for r in rows if r['arm'] == arm and r['status'] == 'complete']
        endpoints = {}
        for endpoint in ('10', '20'):
            costs = [r['endpoints'][endpoint]['best_additions'] for r in entries
                     if r['endpoints'][endpoint]['best_additions'] is not None]
            endpoints[endpoint] = dict(cells_with_circuits=len(costs), best=min(costs) if costs else None,
                                      median_best=statistics.median(costs) if costs else None,
                                      discoveries=sum(r['endpoints'][endpoint]['distinct_discoveries'] for r in entries))
        aggregates[arm] = dict(complete_cells=len(entries), endpoints=endpoints)
    return dict(schema='fgm-effectiveness-summary-v1', complete=complete, cells=rows,
                aggregates=aggregates, elapsed_seconds=data.get('elapsed_seconds'), error=data.get('error'),
                protocol_version=protocol['version'],
                reference_verification=data['reference_verification'],
                naive_additions={e['id']:verify(json.loads((path.parent/'panel'/e['path']).read_text()))['additions'] for e in panel['entries']},
                optimized_reference_unavailable=[e['id'] for e in panel['entries'] if not e['references']],
                scope='descriptive short-run quality; no general equivalence, optimizer optimality or long-run effectiveness claim')


def effectiveness_report(summary):
    lines = ['# FGM-1 effectiveness baseline', '',
             'Status: '+('complete' if summary['complete'] else 'incomplete')+'.', '',
             f'Protocol version: {summary.get("protocol_version", "unavailable")}. Elapsed: {summary.get("elapsed_seconds", "unavailable")} seconds.', '',
             'Costs below are independently verified circuits found from factors. Supplied reference circuits are verified separately.', '',
             '| Start | Seed | Arm | Status | Best at 10 s | Best at 20 s | Discoveries at 20 s | Pending at 20 s |',
             '|---|---:|---|---|---:|---:|---:|---:|']
    for row in summary['cells']:
        end = row.get('endpoints', {})
        first, last = end.get('10', {}), end.get('20', {})
        values = [row['panel'],row['seed'],row['arm'],row['status'],first.get('best_additions'),
                  last.get('best_additions'),last.get('distinct_discoveries'),last.get('unevaluated_discoveries')]
        lines.append('| '+' | '.join('unavailable' if v is None else str(v) for v in values)+' |')
    if summary.get('error'):
        lines.extend(['', 'Incomplete reason: '+summary['error']+'.'])
    lines.extend(['', 'Supplied circuit verification: '+', '.join(
        f'{r["id"]} = {r["additions"]} ({r["additions_by_stage"]["u"]}/{r["additions_by_stage"]["v"]}/{r["additions_by_stage"]["w"]} U/V/W)'
        for r in summary.get('reference_verification', []))+'.',
        'No optimized reference circuit is retained for: '+', '.join(summary.get('optimized_reference_unavailable', []))+'.',
        'Naive factor counts: '+', '.join(f'{k} = {v}' for k,v in summary.get('naive_additions', {}).items())+'.', '',
        '| Arm | Complete cells | Generation s | Reduction s | GPU s | Independent verification s | Export s | Headroom wait s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for arm in ('fixed','generate'):
        cells = [c for c in summary['cells'] if c['arm']==arm and c['status']=='complete']
        values = [arm,len(cells)]+[f'{sum(c["costs"].get(k,0) for c in cells):.3f}' for k in
            ('generation_seconds','reduction_seconds','gpu_seconds','independent_verification_seconds','observation_export_seconds','headroom_wait_seconds')]
        lines.append('| '+' | '.join(map(str,values))+' |')
    lines.extend(['', 'Generation uses cost-blind reseeded blocks and charges startup, resume, export, persistence and verification.',
                  'GPU, verification, export and headroom timings are included in generation/reduction totals; they are not additive partitions.',
                  'Canonical IDs remove term-order and sign-gauge aliases only. Capture drops count encounters, not known lost novel schemes.',
                  'Unevaluated candidates and late completions remain visible. Three seeds describe this protocol; longer runs remain unmeasured.', ''])
    return '\n'.join(lines)


def execute_effectiveness(panel_path, binary_dir, output, calibration_path=None):
    panel_path = Path(panel_path).resolve(strict=True)
    panel, protocol = load_effectiveness_panel(panel_path)
    binary_dir = Path(binary_dir).resolve(strict=True)
    bindings = effectiveness_identity(binary_dir, panel_path)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(panel_path.parent, output/'panel')
    for name in bindings['build_inventory']:
        destination = output/'binaries'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary_dir/name, destination)
    for name in bindings['source_inventory']:
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    shutil.copy2(ROOT/'makefile', output/'source/makefile')
    write_json(output/'bindings.json', bindings)
    data = dict(schema='fgm-effectiveness-measurement-v1', complete=False, bindings=bindings,
                protocol=protocol, reference_verification=[], cells=[])
    for trial_index, seed in enumerate(protocol['seeds']):
        for panel_index, entry in enumerate(panel['entries']):
            arms = ['fixed','generate'] if (trial_index+panel_index)%2 == 0 else ['generate','fixed']
            for arm in arms:
                data['cells'].append(dict(id=f'{entry["id"]}-{seed}-{arm}', panel=entry['id'], seed=seed,
                    arm=arm, status='unrun', reference_costs=[r['claimed_additions'] for r in entry['references']]))
    def save():
        write_json(output/'measurement.json', data)
    save()
    run = EffectivenessRun(output, output/'binaries')
    try:
        # Supplied-circuit verification is preparation, never rediscovery evidence.
        preparation = run.clock()
        for entry in panel['entries']:
            for reference in entry['references']:
                attempt = output/'references'/reference['id']
                attempt.mkdir(parents=True)
                record = dict(commands=[])
                destination = attempt/'verified.jsonl'
                run.command([str(run.binary_dir/'scheme_tool'), 'verify', '--input', str(output/'panel'/reference['path']),
                             '--format', 'circuit-json', '--output', str(destination)], attempt, record)
                verified = json.loads(destination.read_text())
                expected = json.loads((output/'panel'/entry['path']).read_text())
                if any(verified[k] != expected[k] for k in 'uvw') or verified['verified_circuit_additions'] != reference['claimed_additions']:
                    raise ValueError('native reference factor/count mismatch')
                if verified['verified_circuit_additions_by_stage'] != reference['claimed_additions_by_stage']:
                    raise ValueError('native reference stage count mismatch')
                data['reference_verification'].append(dict(id=reference['id'], additions=verified['verified_circuit_additions'],
                    additions_by_stage=verified['verified_circuit_additions_by_stage'], status='verified supplied circuit',
                    source_sha256=reference['source_sha256'], circuit_sha256=reference['sha256']))
                save()
        data['preparation_seconds'] = run.clock()-preparation
        run.started = run.clock()
        run.deadline = run.started+protocol['total_seconds']
        if wired_memory() > LIMIT:
            raise RuntimeError('wired memory exceeds existing guard cutoff')
        if calibration_path is not None:
            calibration_path = Path(calibration_path).resolve(strict=True)
            calibration = json.loads(calibration_path.read_text())
            if not calibration_valid(calibration, bindings, calibration_path.parent):
                raise ValueError('calibration identity or evidence mismatch; recalibrate without --calibration')
            calibration = copy.deepcopy(calibration)
            # Retain the evidence required to validate this reused calibration again.
            for row in calibration['pilots']:
                for step in row.get('steps', []):
                    source = calibration_path.parent/step['path']
                    destination = output/'calibration-evidence'/step['path']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source.parent, destination.parent, dirs_exist_ok=True)
                    config = json.loads((source.parent/'config.json').read_text())
                    if config['operation'] == 'search':
                        history = retained_effectiveness_history(calibration_path.parent, step, config)
                        shutil.copytree(history, output/'calibration-evidence'/history.relative_to(calibration_path.parent),
                                        dirs_exist_ok=True)
                    step['path'] = str(destination.relative_to(output))
            calibration['reused_from_sha256'] = digest(calibration_path)
            write_json(output/'calibration.json', calibration)
            data['calibration_reused'] = True
        else:
            calibration = calibrate_effectiveness(run, panel['entries'], output/'panel', bindings,
                                                  lambda c:write_json(output/'calibration.json', c))
            data['calibration_reused'] = False
        data['calibration_sha256'] = digest(output/'calibration.json')
        save()
        for cell in data['cells']:
            run.check_launch()
            entry = next(e for e in panel['entries'] if e['id'] == cell['panel'])
            run_effectiveness_cell(run, cell, entry, output/'panel', calibration['settings'], save)
            if cell['status'] != 'complete':
                break
        data['complete'] = all(cell['status'] == 'complete' for cell in data['cells'])
    except Exception as error:
        data['error'] = str(error)
        if not isinstance(error, EffectivenessStop):
            data['failure'] = True
    finally:
        finalization_started = run.clock()
        data['elapsed_seconds'] = run.clock()-run.started if run.started is not None else 0
        if data['elapsed_seconds'] > protocol['total_seconds']:
            data['complete'] = False
            data['error'] = 'overall envelope exceeded'
        if any(digest(output/'binaries'/name) != value for name,value in bindings['build_inventory'].items()):
            data['complete'] = False
            data['error'] = 'frozen binary inventory changed'
        save()
        summary = summarize_effectiveness(output/'measurement.json')
        write_json(output/'summary.json', summary)
        (output/'report.md').write_text(effectiveness_report(summary))
        inventory = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
        data['finalization_seconds'] = run.clock()-finalization_started
        data['elapsed_seconds'] = run.clock()-run.started if run.started is not None else 0
        if data['elapsed_seconds'] > protocol['total_seconds']:
            data['complete'] = False
            data['error'] = 'overall envelope exceeded during finalization'
        data['elapsed_scope'] = 'through independent summary and artifact hashing; excludes final closing JSON/text writes'
        save()
        summary.update(complete=data['complete'], elapsed_seconds=data['elapsed_seconds'], error=data.get('error'))
        write_json(output/'summary.json', summary)
        (output/'report.md').write_text(effectiveness_report(summary))
        for name in ('measurement.json','summary.json','report.md'):
            inventory[name] = digest(output/name)
        write_json(output/'artifacts.json', inventory)
        print('FGM-1', 'COMPLETE' if data['complete'] else 'INCOMPLETE', str(output), data.get('error',''))
    return data


FGM2_PROTOCOL_PATH = ROOT/'benchmarks/workflow/fixtures/fgm2/protocol.json'


def fgm2_schedule(protocol):
    """The protocol fixes every invocation, including its strategy order."""
    rows = []
    names = [*protocol['public_inputs'], protocol['private_input']]
    strategies = protocol['strategies']
    for panel_index, name in enumerate(names):
        for seed_index, seed in enumerate(protocol['seeds']):
            rotation = (panel_index + seed_index) % len(strategies)
            for strategy in strategies[rotation:] + strategies[:rotation]:
                rows.append(dict(id=f'{name}-{seed}-{strategy}', input=name,
                                 seed=seed, strategy=strategy))
    if len(rows) != 72 or len({row['id'] for row in rows}) != 72:
        raise ValueError('FGM-2 schedule must contain 72 unique invocations')
    return rows


def fgm2_source(path):
    """Derive the reference from retained raw factors, before a measured trial."""
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 4*1048576:
        raise ValueError('FGM-2 source exceeds 4 MiB')
    source = json.loads(path.read_text())
    if any(key in source for key in ('u_fresh','v_fresh','w_fresh')):
        raise ValueError('FGM-2 requires factor-only inputs')
    if 'n' in source:
        raw = source
    else:
        if (source.get('schema') != 'fgm-scheme-v1' or source.get('domain') != 'ZT'
                or source.get('orientation') != 'cyclic-w'):
            raise ValueError('FGM-2 private scheme metadata mismatch')
        raw = dict(n=source['dimensions'],m=source['rank'],z2=False,
                   **{key:source[key] for key in 'uvw'})
    if raw['n'] != [3,3,3] or raw['m'] != 23 or raw['z2'] is not False:
        raise ValueError('FGM-2 requires exact signed 3x3 rank-23 factors')
    normalized = normalized_input(raw)
    raw_reference = dict(dimensions=raw['n'], rank=raw['m'], domain='ZT',
                         orientation='cyclic-w', **{key:raw[key] for key in 'uvw'})
    reference = dict(dimensions=normalized['n'], rank=normalized['m'], domain='ZT',
                     orientation='cyclic-w', **{key:normalized[key] for key in 'uvw'})
    return dict(source=path, sha256=digest(path), raw_reference=raw_reference,
                raw_factors_id=host_oracle().identity(raw_reference,False),
                reference=reference, factors_id=host_oracle().identity(reference, False))


def fgm2_inputs(private_path):
    panel_path = ROOT/'benchmarks/workflow/fixtures/fgm1/panel.json'
    panel = json.loads(panel_path.read_text())
    if panel.get('schema') != 'fgm-effectiveness-panel-v1':
        raise ValueError('wrong public FGM-1 panel')
    protocol = json.loads(FGM2_PROTOCOL_PATH.read_text())
    if protocol['public_inputs'] != [entry['id'] for entry in panel['entries']]:
        raise ValueError('FGM-2 public roster differs from retained panel')
    result = {}
    for entry in panel['entries']:
        source = panel_artifact(panel_path.parent, entry['path'], entry['sha256'])
        result[entry['id']] = fgm2_source(source)
    result['private'] = fgm2_source(private_path)
    return protocol, result


def fgm2_reference_coverage(sources, private_reference=None):
    """Audit supplied cost witnesses separately from measured construction."""
    panel_path = ROOT/'benchmarks/workflow/fixtures/fgm1/panel.json'
    panel = json.loads(panel_path.read_text())
    rows = []
    for entry in panel['entries']:
        for reference in entry['references']:
            circuit_path = panel_artifact(panel_path.parent,reference['path'],reference['sha256'])
            source_path = panel_artifact(panel_path.parent,reference['source'],reference['source_sha256'])
            circuit = json.loads(circuit_path.read_text())
            if reference_circuit(source_path,reference['id']) != circuit:
                raise ValueError('FGM-2 supplied reference differs from retained source conversion')
            rows.append((entry['id'],reference['id'],circuit_path,circuit,
                         reference['claimed_additions'],reference['claimed_additions_by_stage']))
    if private_reference is not None:
        path = Path(private_reference).resolve(strict=True)
        if path.stat().st_size > 4*1048576:
            raise ValueError('FGM-2 private reference exceeds 4 MiB')
        circuit = json.loads(path.read_text())
        rows.append(('private','private-supplied',path,circuit,None,None))
    checked = []
    for name,label,path,circuit,claim,stages in rows:
        reference = sources[name]['raw_reference']
        raw = dict(n=reference['dimensions'],m=reference['rank'],z2=False,
                   **{key:reference[key] for key in 'uvw'})
        result = verify(circuit,raw)
        counts = result['additions_by_stage']
        if claim is not None and (result['additions'] != claim or counts != stages):
            raise ValueError('FGM-2 supplied reference cost claim mismatch')
        factors = dict(dimensions=circuit['n'],rank=circuit['m'],domain='ZT',orientation='cyclic-w',
                       **dict(zip('uvw',circuit_factors(circuit))))
        if host_oracle().identity(factors,False) != sources[name]['raw_factors_id']:
            raise ValueError('FGM-2 supplied reference factor identity mismatch')
        checked.append(dict(input=name,id=label,source_path=str(path),sha256=digest(path),
            factors_id=sources[name]['raw_factors_id'],
            normalized_input_factors_id=sources[name]['factors_id'],additions=result['additions'],
            additions_by_stage=counts,witness_class='exact-factor supplied circuit cost',
            construction_family_witness='unassessed'))
    return checked


def fgm2_preliminary_circuit(path, receipt, source, record_limit=1048576):
    """Bounded structural binding before admitting the independent CLI verifier."""
    artifact = receipt['circuit_artifact']
    if (Path(artifact['path']) != path or artifact['format'] != 'jsonl'
            or artifact['records'] != 1 or digest(path) != artifact['sha256']):
        raise ValueError('FGM-2 circuit artifact path, count or hash mismatch')
    if path.stat().st_size > record_limit:
        raise ValueError('FGM-2 circuit exceeds record limit')
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError('FGM-2 requires exactly one circuit')
    circuit = json.loads(lines[0])
    reference = source['reference']
    if (circuit['n'] != reference['dimensions'] or circuit['m'] != reference['rank']
            or circuit['z2'] is not False):
        raise ValueError('FGM-2 circuit dimensions, rank or domain mismatch')
    if circuit.get('factors_id',source['factors_id']) != source['factors_id']:
        raise ValueError('FGM-2 circuit declared factor identity mismatch')
    result = receipt['results']
    if len(result) != 1:
        raise ValueError('FGM-2 requires exactly one result')
    result = result[0]
    if (result['circuit_record_index'] != 0 or result['result_rank'] != 23
            or result['result_factors_id'] != source['factors_id']
            or result['effective_input_factors_id'] != source['factors_id']
            or result['submitted_factors_id'] != source['raw_factors_id']
            or result['source_sha256'] != source['sha256']
            or result['input_presentation_index'] != 0):
        raise ValueError('FGM-2 result circuit/source binding mismatch')
    binding = {key:result[key] for key in ('input_presentation_index','submitted_factors_id',
                'effective_input_factors_id','source_sha256')}
    if circuit.get('source_binding') != binding:
        raise ValueError('FGM-2 circuit source binding mismatch')
    return dict(factors_id=source['factors_id'],circuit_sha256=digest(path),
                claimed_additions=circuit['complexity']['reduced'])


def fgm2_bind_verifier(log_path, circuit_path, reference_path, preliminary, receipt, source):
    """Bind the independent Python CLI PASS to retained bytes and native counts."""
    if log_path.stat().st_size > 1048576 or digest(circuit_path) != preliminary['circuit_sha256']:
        raise ValueError('FGM-2 verifier log bound or circuit hash mismatch')
    reference = source['reference']
    expected = dict(n=reference['dimensions'],m=reference['rank'],z2=False,
                    **{key:reference[key] for key in 'uvw'})
    if json.loads(reference_path.read_text()) != expected:
        raise ValueError('FGM-2 independently derived reference changed')
    lines = log_path.read_text().splitlines()
    prefix = str(circuit_path)+' PASS '
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise ValueError('FGM-2 independent verifier lacks one PASS record')
    verified = ast.literal_eval(lines[0][len(prefix):])
    if (not isinstance(verified,dict) or verified.get('domain')!='ZT'
            or verified.get('rank') != 23
            or verified.get('equations') != 729
            or type(verified.get('additions')) is not int
            or set(verified.get('additions_by_stage',{})) != set('uvw')
            or any(type(value) is not int or value < 0
                   for value in verified['additions_by_stage'].values())
            or sum(verified['additions_by_stage'].values()) != verified['additions']):
        raise ValueError('FGM-2 independent verifier metadata mismatch')
    result = receipt['results'][0]
    if (verified['additions'] != preliminary['claimed_additions']
            or result['verified_circuit_additions'] != verified['additions']
            or result['verified_circuit_additions_by_stage'] != verified['additions_by_stage']):
        raise ValueError('FGM-2 independent verifier/receipt cost mismatch')
    return dict(factors_id=source['factors_id'],additions=verified['additions'],
                additions_by_stage=verified['additions_by_stage'],
                circuit_sha256=preliminary['circuit_sha256'])


def fgm2_summary(data):
    cells = data['trials']
    groups = defaultdict(dict)
    for row in cells:
        if row['status'] == 'verified':
            groups[(row['input'],row['seed'])][row['strategy']] = row
    comparisons = []
    for (name,seed), strategies in sorted(groups.items()):
        baseline = strategies.get('baseline')
        for strategy in ('transpose','cancellation','combined'):
            extension = strategies.get(strategy)
            if baseline and extension:
                comparisons.append(dict(input=name,seed=seed,strategy=strategy,
                    baseline_additions=baseline['evaluation']['additions'],
                    additions=extension['evaluation']['additions'],
                    cost_change=extension['evaluation']['additions']-baseline['evaluation']['additions'],
                    extension_work=extension.get('construction'),
                    phase_microseconds=extension.get('phase_microseconds')))
    references = {'sun':[56], 'cn122':[55,58], 'private':[55]}
    attainment = {name:{str(cost):[row['id'] for row in cells if row['input']==name
                    and row['status']=='verified' and row['evaluation']['additions']<=cost]
                    for cost in costs} for name,costs in references.items()}
    supplied = data.get('supplied_reference_coverage',[])
    reference_map = {name:dict(witnesses=[row['id'] for row in supplied if row['input']==name],
                     classification=('exact-factor supplied circuit cost witnesses' if any(
                         row['input']==name for row in supplied) else 'no supplied circuit witness'))
                     for name in [*data['protocol']['public_inputs'],data['protocol']['private_input']]}
    missing = [row['id'] for row in cells if row['status'] != 'verified']
    constructor_outcomes = defaultdict(lambda:dict(requested=0,target_only=0,one_auxiliary=0,
                                               exhausted=0,other=0))
    for row in cells:
        for stage, report in row.get('construction',{}).items():
            status = report.get('status')
            if status == 'not-requested':
                continue
            counts = constructor_outcomes[stage]
            counts['requested'] += 1
            field = {'target-only':'target_only','one-auxiliary':'one_auxiliary',
                     'family-exhausted':'exhausted'}.get(status,'other')
            counts[field] += 1
    return dict(schema='fgm2-construction-summary-v1',protocol_revision=data['protocol'].get('revision',1),
                complete=data['complete'],
                status=data['status'],verified=len(cells)-len(missing),
                total_trials=len(cells),comparisons=comparisons,reference_attainment=attainment,
                supplied_reference_coverage=supplied,reference_map=reference_map,
                missing_verified=missing,constructor_outcomes=dict(constructor_outcomes),
                comparison_scope='fixed reduction work plus strategy extensions, not matched time')


def fgm2_report(data, summary):
    """Compact per-invocation account; missing results remain visible."""
    lines = ['# FGM-2 construction measurement', '',
             f"Protocol revision: {summary['protocol_revision']}.",
             f"Status: {summary['status']}; verified: {summary['verified']}/{summary['total_trials']}.",
             'Fixed reduction work is followed by strategy extensions. Elapsed times are not matched.', '',
             '| Input | Seed | Strategy | Status | Baseline | Final | Change | Trial s | Extension s | GPU s | Peak wired bytes |',
             '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in data['trials']:
        evaluation = row.get('evaluation',{})
        final = evaluation.get('additions')
        baseline = row.get('baseline_additions')
        gpu = row.get('gpu',{}).get('gpu_all_seconds')
        wired = row.get('native_guard',{}).get('peak_system_wired_bytes')
        phases = row.get('phase_microseconds',{})
        extension = sum(phases.get(key,0) for key in
            ('transpose_pair','construction_u','construction_v','construction_wt','transposition','verification'))/1000000 if phases else None
        fmt = lambda value: '-' if value is None else str(value)
        lines.append('| '+ ' | '.join(map(fmt,(row['input'],row['seed'],row['strategy'],row['status'],
            baseline,final,final-baseline if final is not None and baseline is not None else None,
            round(row['elapsed_seconds'],3) if 'elapsed_seconds' in row else None,
            round(extension,3) if extension is not None else None,
            round(gpu,3) if gpu is not None else None,wired)))+' |')
    lines += ['',f"Missing verified results: {', '.join(summary['missing_verified']) or 'none'}.",
              f"Setup: {data.get('setup_seconds',0):.3f} s; finalization: {data.get('finalization_seconds',0):.3f} s.",
              data.get('timing_scope',''),
              (f"Budget check after closing writes: {data['budget_check_elapsed_seconds']:.3f} s."
               if 'budget_check_elapsed_seconds' in data else ''),
              'Per-row elapsed observations (seconds):', '',
              '| Trial | Preparation | Headroom wait | Native guard | Native receipt/binding | Verifier guard | Final binding | Intermediate persistence |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    phase_fields = ('preparation_seconds','headroom_wait_seconds','native_supervised_seconds',
                    'native_receipt_binding_seconds','verifier_supervised_seconds',
                    'final_binding_seconds','evidence_persistence_seconds')
    for row in data['trials']:
        values = [f"{row[key]:.3f}" if key in row else '-' for key in phase_fields]
        lines.append('| '+ ' | '.join([row['id'],*values])+' |')
    lines += ['', 'Intermediate persistence overlaps the active phase; phase observations are not additive. Final trial snapshot writes are included in overall elapsed time.',
              'Constructor outcomes count observed searches only; supplied reference circuits are separate cost witnesses.', '']
    return '\n'.join(lines)


def fgm2_build_files(binary_dir):
    binary = binary_dir/'additions_reducer'
    build = binary_dir/'additions_reducer.build.json'
    library = binary_dir/'shaders/signed.metallib'
    library_build = binary_dir/'shaders/signed.metallib.build.json'
    receipt = json.loads(build.read_text())
    stat = binary.stat()
    if receipt['output'] != {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}:
        raise ValueError('FGM-2 build receipt does not match executable')
    for dependency,checksum in receipt['inputs']['dependencies'].items():
        path = Path(dependency)
        if digest(path if path.is_absolute() else ROOT/path) != checksum:
            raise ValueError('FGM-2 build dependency changed: '+dependency)
    if json.loads(library_build.read_text())['sha256'] != digest(library):
        raise ValueError('FGM-2 signed library receipt mismatch')
    return (binary,build,library,library_build)


def execute_fgm2(private_path, binary_dir, output, *, private_reference=None,
                 clock=None, sleeper=None, guard=None):
    """One bounded FGM-2 campaign. A deadline miss never means no circuit exists."""
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    guard = guard or guarded_run
    started = clock()
    protocol, sources = fgm2_inputs(private_path)
    reference_coverage = fgm2_reference_coverage(sources,private_reference)
    binary_dir = Path(binary_dir).resolve(strict=True)
    bound_files = fgm2_build_files(binary_dir)
    original_hashes = {str(path.relative_to(binary_dir)):digest(path) for path in bound_files}
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output/'.gitignore').write_text('*\n')
    deadline = started + protocol['total_seconds']
    native_admission = (protocol['native_seconds']+protocol['verification_seconds']
                        +2*protocol['phase_bookkeeping_seconds']
                        +protocol['finalization_reserve_seconds'])
    verifier_admission = (protocol['verification_seconds']
                          +protocol['phase_bookkeeping_seconds']
                          +protocol['finalization_reserve_seconds'])
    schedule = fgm2_schedule(protocol)
    data = dict(schema='fgm2-construction-measurement-v1',status='incomplete',complete=False,
                protocol=protocol,started_monotonic=started,trials=[dict(row,status='unrun') for row in schedule],
                inputs={},build_inventory={},harness={},supplied_reference_coverage=reference_coverage)
    def save():
        write_json(output/'measurement.json',data)
    save()
    forced = False
    supervisor_failure = False
    correctness_failure = False
    try:
        data['hardware'] = host_machine_identity()
        data['source_inventory'] = source_identity(ROOT)
        for path in bound_files:
            relative = path.relative_to(binary_dir)
            destination = output/'binaries'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            if digest(destination) != original_hashes[str(relative)]:
                raise ValueError('FGM-2 frozen build file differs from source')
        data['build_inventory'] = original_hashes
        binary = output/'binaries/additions_reducer'
        library = output/'binaries/shaders/signed.metallib'
        harness_paths = (Path(__file__),ROOT/'benchmarks/metal/guard.py',
            ROOT/'benchmarks/metal/application.py',ROOT/'tests/metal/verify.py',
            ROOT/'tests/workflow/identity_oracle.py',FGM2_PROTOCOL_PATH)
        for path in harness_paths:
            relative = path.relative_to(ROOT)
            destination = output/'harness'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            if digest(destination) != digest(path):
                raise ValueError('FGM-2 frozen harness file differs from source')
            data['harness'][str(relative)] = digest(destination)
        public_panel = ROOT/'benchmarks/workflow/fixtures/fgm1'
        shutil.copytree(public_panel,output/'public-panel')
        panel = json.loads((output/'public-panel/panel.json').read_text())
        for name,checksum in panel['artifacts'].items():
            if digest(output/'public-panel'/name) != checksum:
                raise ValueError('FGM-2 retained public panel artifact mismatch')
        data['public_panel_sha256'] = digest(output/'public-panel/panel.json')
        for row in reference_coverage:
            if row['input'] != 'private':
                row['retained_path'] = str((output/'public-panel'/Path(row['source_path']).relative_to(public_panel)).relative_to(output))
        for name, source in sources.items():
            destination = output/'inputs'/f'{name}.json'
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source['source'],destination)
            if digest(destination) != source['sha256']:
                raise ValueError('FGM-2 input changed during retention')
            data['inputs'][name] = dict(path=str(destination.relative_to(output)),
                sha256=source['sha256'],factors_id=source['factors_id'],
                raw_factors_id=source.get('raw_factors_id'),
                dimensions=source['reference']['dimensions'],rank=source['reference']['rank'],domain='ZT')
            source['retained'] = destination
        if private_reference is not None:
            destination = output/'references/private-supplied.json'
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(private_reference).resolve(strict=True),destination)
            if digest(destination) != next(row['sha256'] for row in reference_coverage
                                           if row['id']=='private-supplied'):
                raise ValueError('FGM-2 private reference changed during retention')
            data['private_reference_retained'] = str(destination.relative_to(output))
            next(row for row in reference_coverage if row['id']=='private-supplied')['retained_path'] = data['private_reference_retained']
        save()
        data['setup_seconds'] = clock()-started
        for index, row in enumerate(data['trials']):
            if forced or deadline-clock() < native_admission:
                data['stop_reason'] = 'forced GPU termination' if forced else 'global admission cutoff'
                if not forced:
                    row['unrun_reason'] = 'insufficient global budget for native admission'
                break
            trial_start = clock()
            trial = output/'trials'/f'{index:02d}-{row["id"]}'
            trial.mkdir(parents=True)
            row.update(trial_start_seconds=trial_start-started,
                       evidence_persistence_seconds=0.)
            phase_name = 'preparation_seconds'
            phase_started = clock()
            def next_phase(name):
                nonlocal phase_name, phase_started
                now = clock()
                row[phase_name] = row.get(phase_name,0.)+now-phase_started
                phase_name, phase_started = name, now
            def persist():
                persistence_started = clock()
                save()
                row['evidence_persistence_seconds'] += clock()-persistence_started
            persist()
            source = sources[row['input']]
            try:
                if (any(digest(output/'binaries'/name) != checksum
                        for name,checksum in data['build_inventory'].items())
                        or any(digest(output/'harness'/name) != checksum
                               for name,checksum in data['harness'].items())):
                    raise ValueError('FGM-2 frozen binary or harness identity changed')
                if digest(source['retained']) != source['sha256']:
                    raise ValueError('FGM-2 retained input hash changed')
                config = dict(schema='fgm-run-v1',operation='reduce',output=str(trial/'receipt.json'),
                    input=dict(kind='files',files=[dict(path=str(source['retained']),format='json',domain='ZT')]),
                    execution=dict(workers=1,batch_steps=64,block_size=32,backend='general',
                                   memory_bytes=536870912),
                    reduction=dict(domain='ZT',seed=row['seed'],strategy=row['strategy'],
                                   **protocol['reduction']))
                config_path = trial/'config.json'
                write_json(config_path,config)
                reference = source['reference']
                reference_path = trial/'independent-reference.json'
                write_json(reference_path,dict(n=reference['dimensions'],m=reference['rank'],z2=False,
                    **{key:reference[key] for key in 'uvw'}))
                row['config_sha256'] = digest(config_path)
                row['independent_reference_sha256'] = digest(reference_path)
                row['input_sha256'] = source['sha256']
                next_phase('headroom_wait_seconds')
                while True:
                    sample_allowance = deadline-clock()-native_admission
                    if sample_allowance <= 0:
                        row['unrun_reason'] = 'insufficient global budget after preparation or headroom wait'
                        data['stop_reason'] = 'global admission cutoff'
                        break
                    try:
                        available = wired_memory(timeout=min(2,sample_allowance)) <= LIMIT-FGM1_PROTOCOL['launch_wired_reserve_bytes']
                    except subprocess.TimeoutExpired:
                        if deadline-clock() <= native_admission:
                            row['unrun_reason'] = 'global admission budget exhausted during headroom sample'
                            data['stop_reason'] = 'global admission cutoff'
                            break
                        raise
                    if deadline-clock() < native_admission:
                        row['unrun_reason'] = 'insufficient global budget after headroom sample'
                        data['stop_reason'] = 'global admission cutoff'
                        break
                    if available:
                        break
                    sleeper(min(.025,max(0,deadline-clock()-native_admission)))
                if row.get('unrun_reason'):
                    break
                next_phase('preparation_seconds')
                if deadline-clock() < native_admission:
                    row['unrun_reason'] = 'insufficient global budget before native guard'
                    data['stop_reason'] = 'global admission cutoff'
                    break
                argv = ['/usr/bin/time','-l','-p',str(binary),'--run-config',str(config_path)]
                row['native_command'] = argv
                persist()
                if deadline-clock() < native_admission:
                    row['unrun_reason'] = 'insufficient global budget after native command persistence'
                    data['stop_reason'] = 'global admission cutoff'
                    break
                native_deadline = clock()+protocol['native_seconds']
                row['native_deadline_seconds'] = native_deadline-started
                row['status'] = 'failed'
                next_phase('native_supervised_seconds')
                try:
                    guarded = guard(argv,trial/'native-guard',absolute_deadline=native_deadline)
                except Exception as error:
                    supervisor_failure = True
                    row['status'] = 'terminated'
                    row['native_supervision_error'] = str(error)
                    raise RuntimeError('native guard failed with unknown process cleanup') from error
                finally:
                    next_phase('native_receipt_binding_seconds')
                forced = bool(guarded.get('forced_termination'))
                supervisor_failure = bool(guarded.get('cleanup_failure'))
                if forced or supervisor_failure:
                    row['status'] = 'terminated'
                row['native_guard'] = dict(result_sha256=digest(trial/'native-guard/result.json'),
                    log_sha256=digest(trial/'native-guard/run.log') if (trial/'native-guard/run.log').is_file() else None,
                    complete=guarded['complete'],exit_code=guarded.get('exit_code'),
                    forced_termination=guarded.get('forced_termination',False),
                    cleanup_failure=guarded.get('cleanup_failure',False),
                    observed_process_seconds=guarded.get('observed_process_seconds'),
                    wall_seconds=guarded.get('wall_seconds'),
                    peak_system_wired_bytes=max((sample['wired_bytes'] for sample in guarded.get('memory',[])),default=None))
                if forced:
                    row['status'] = 'terminated'
                    raise RuntimeError('native GPU process required termination')
                if supervisor_failure:
                    row['status'] = 'terminated'
                    raise RuntimeError('native GPU cleanup failed')
                if not guarded['complete'] and guarded.get('exit_code') not in (None,0):
                    raise ValueError('FGM-2 native process failed: '+str(guarded.get('error')))
                if not guarded['complete']:
                    raise TimeoutError('native process did not finish before native deadline')
                receipt_path = trial/'receipt.json'
                receipt = json.loads(receipt_path.read_text())
                row['receipt_sha256'] = digest(receipt_path)
                if (receipt['status'] != 'complete' or not receipt['execution_started']
                        or receipt['configuration_sha256'] != digest(config_path)
                        or receipt['executable_sha256'] != digest(binary)
                        or receipt.get('library_mode') != 'metallib'
                        or receipt.get('library_sha256') != digest(library)):
                    raise ValueError('FGM-2 native receipt/configuration/executable mismatch')
                if (len(receipt['presentations']) != 1
                        or receipt['presentations'][0]['source_sha256'] != source['sha256']
                        or receipt['presentations'][0]['effective_factors_id'] != source['factors_id']):
                    raise ValueError('FGM-2 native source presentation mismatch')
                result = receipt['results'][0]
                if result.get('strategy') != row['strategy']:
                    raise ValueError('FGM-2 strategy receipt mismatch')
                row['baseline_additions'] = result['baseline_additions']
                row['baseline_additions_by_stage'] = result['baseline_additions_by_stage']
                row['stage_sources'] = result['stage_sources']
                row['construction'] = result['construction']
                row['phase_microseconds'] = result['phase_microseconds']
                log = (trial/'native-guard/run.log').read_text()
                row['gpu'] = native_dispatch_evidence(log,receipt)
                kernels = {item[0] for item in row['gpu'].get('dispatches',[])}
                if (row['strategy'] in ('transpose','combined') and 'transposePairKernel' not in kernels
                        or row['strategy'] in ('cancellation','combined') and 'constructorClosureKernel' not in kernels):
                    raise ValueError('FGM-2 strategy lacks required GPU constructor dispatch')
                artifact = Path(str(receipt_path)+'.circuits.jsonl')
                preliminary = fgm2_preliminary_circuit(artifact,receipt,source)
                row['artifact'] = dict(path=str(artifact.relative_to(output)),sha256=preliminary['circuit_sha256'])
                row['status'] = 'artifact-awaiting-verification'
                persist()
                if deadline-clock() < verifier_admission:
                    row['verification_admission'] = 'insufficient time'
                    continue
                argv = [sys.executable,str(output/'harness/tests/metal/verify.py'),str(artifact),
                        '--reference',str(reference_path)]
                row['verification_command'] = argv
                persist()
                if deadline-clock() < verifier_admission:
                    row['verification_admission'] = 'insufficient time after verifier command persistence'
                    continue
                verifier_deadline = clock()+protocol['verification_seconds']
                row['verifier_deadline_seconds'] = verifier_deadline-started
                row['verification_started_seconds'] = clock()-trial_start
                next_phase('verifier_supervised_seconds')
                try:
                    checked = guard(argv,trial/'verifier-guard',absolute_deadline=verifier_deadline)
                except Exception as error:
                    supervisor_failure = True
                    row['verifier_supervision_error'] = str(error)
                    raise RuntimeError('verifier guard failed with unknown process cleanup') from error
                finally:
                    next_phase('final_binding_seconds')
                supervisor_failure = bool(checked.get('cleanup_failure'))
                row['verifier_guard'] = dict(result_sha256=digest(trial/'verifier-guard/result.json'),
                    log_sha256=digest(trial/'verifier-guard/run.log') if (trial/'verifier-guard/run.log').is_file() else None,
                    complete=checked['complete'],exit_code=checked.get('exit_code'),
                    forced_termination=checked.get('forced_termination',False),
                    cleanup_failure=checked.get('cleanup_failure',False),
                    observed_process_seconds=checked.get('observed_process_seconds'),
                    wall_seconds=checked.get('wall_seconds'))
                if checked.get('cleanup_failure'):
                    supervisor_failure = True
                    raise RuntimeError('FGM-2 verifier cleanup failed')
                verifier_log = trial/'verifier-guard/run.log'
                timed_out = 'time limit' in str(checked.get('error',''))
                if checked.get('forced_termination') or (timed_out and checked.get('exit_code') != 0):
                    row['verification_admission'] = 'verifier deadline or forced termination'
                    continue
                if checked.get('exit_code') not in (None,0):
                    raise ValueError('FGM-2 independent verifier rejected circuit')
                if not checked['complete'] and not (timed_out and checked.get('exit_code')==0 and verifier_log.is_file()):
                    row['verification_admission'] = 'verifier incomplete'
                    continue
                if digest(receipt_path) != row['receipt_sha256'] or digest(reference_path) != row['independent_reference_sha256']:
                    raise ValueError('FGM-2 receipt or independent reference changed during verification')
                evaluation = fgm2_bind_verifier(verifier_log,artifact,reference_path,preliminary,receipt,source)
                row['verifier_log_sha256'] = digest(verifier_log)
                row['verification_completed_seconds'] = clock()-trial_start
                row['evaluation'] = {key:evaluation[key] for key in
                    ('factors_id','additions','additions_by_stage','circuit_sha256')}
                row['status'] = 'verified' if checked['complete'] and clock() <= deadline else 'late-verified'
            except Exception as error:
                if row['status'] == 'unrun' and not row.get('unrun_reason'):
                    row['status'] = 'failed'
                row['error'] = str(error)
                correctness_failure = isinstance(error,(ValueError,SyntaxError,KeyError,TypeError))
            finally:
                next_phase('closed')
                row['elapsed_seconds'] = clock()-trial_start
                write_json(trial/'trial.json',row)
                row['trial_sha256'] = digest(trial/'trial.json')
                save()
            if row.get('unrun_reason'):
                break
            if forced:
                data['stop_reason'] = 'forced GPU termination'
                break
            if supervisor_failure:
                data['stop_reason'] = 'unknown process cleanup after guard failure'
                break
            if correctness_failure:
                data['stop_reason'] = 'correctness or identity failure'
                break
        data['complete'] = len(data['trials']) == 72 and all(row['status']=='verified' for row in data['trials'])
        data['status'] = 'complete' if data['complete'] else 'incomplete'
    except Exception as error:
        data['error'] = str(error)
    finally:
        finalization_started = clock()
        try:
            if any(digest(output/'binaries'/name) != checksum
                   for name,checksum in data['build_inventory'].items()):
                data['complete'] = False
                data['status'] = 'incomplete'
                data['error'] = 'frozen build inventory changed during campaign'
            if any(digest(output/'harness'/name) != checksum
                   for name,checksum in data['harness'].items()):
                data['complete'] = False
                data['status'] = 'incomplete'
                data['error'] = 'frozen harness inventory changed during campaign'
        except (OSError,ValueError) as error:
            data['complete'] = False
            data['status'] = 'incomplete'
            data['error'] = 'retained inventory unavailable: '+str(error)
        data['elapsed_seconds'] = clock()-started
        data['finalization_seconds'] = clock()-finalization_started
        if data['elapsed_seconds'] > protocol['total_seconds']:
            data['complete'] = False
            data['status'] = 'budget-exceeded'
        save()
        summary = fgm2_summary(data)
        write_json(output/'summary.json',summary)
        (output/'report.md').write_text(fgm2_report(data,summary))
        timing_snapshot = clock()
        data['final_elapsed_seconds'] = timing_snapshot-started
        data['finalization_seconds'] = timing_snapshot-finalization_started
        data['timing_scope'] = ('Finalization and final elapsed time end at the snapshot before closing summary, report, '
                                'measurement, and checksum writes. The budget check includes those writes.')
        if data['final_elapsed_seconds'] > protocol['total_seconds']:
            data['complete'] = False
            data['status'] = 'budget-exceeded'
        summary = fgm2_summary(data)
        write_json(output/'summary.json',summary)
        (output/'report.md').write_text(fgm2_report(data,summary))
        save()
        (output/'measurement.sha256').write_text(digest(output/'measurement.json')+'\n')
        checked_elapsed = clock()-started
        if checked_elapsed > protocol['total_seconds']:
            data['complete'] = False
            data['status'] = 'budget-exceeded'
            data['budget_check_elapsed_seconds'] = checked_elapsed
            write_json(output/'summary.json',fgm2_summary(data))
            (output/'report.md').write_text(fgm2_report(data,fgm2_summary(data)))
            save()
            (output/'measurement.sha256').write_text(digest(output/'measurement.json')+'\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--freeze', type=Path, metavar='NEW_DIRECTORY',
                           help='Prepare pinned source only; no build or GPU execution')
    operation.add_argument('--source', type=Path, help='Frozen production source to measure')
    operation.add_argument('--native-config', type=Path, help='Measure one configured native workflow')
    operation.add_argument('--prepare-effectiveness-panel', type=Path, metavar='CORPUS', help='Freeze the FGM-1 panel from a retained corpus without GPU work')
    operation.add_argument('--effectiveness-panel', type=Path, metavar='PANEL_JSON', help='Run one FGM-1 baseline with a 15-minute measurement cap')
    operation.add_argument('--summarize-effectiveness', type=Path, metavar='MEASUREMENT_JSON', help='Independently check and summarize retained FGM-1 circuits')
    operation.add_argument('--fgm2', action='store_true', help='Run the fixed 72-invocation FGM-2 construction protocol')
    parser.add_argument('--fgm2-private-input', type=Path, help='Required external factor-only private input for FGM-2')
    parser.add_argument('--fgm2-private-reference', type=Path, help='Optional private supplied circuit for separate FGM-2 reference audit')
    parser.add_argument('--binary-dir', type=Path, help='Native binaries for FGM-1 (default build/metal)')
    parser.add_argument('--calibration', type=Path, help='Reuse matching FGM-1 calibration.json')
    parser.add_argument('--native-binary', type=Path)
    operation.add_argument('--host', type=Path, metavar='NATIVE_BINARY', help='Measure native-host workflows with independent correctness checks')
    operation.add_argument('--qualify-profile', type=Path, metavar='PROFILE_DIRECTORY')
    operation.add_argument('--summarize', type=Path, metavar='QUALIFICATION_JSON')
    parser.add_argument('--baseline', type=Path, help='Frozen baseline directory for profile qualification')
    parser.add_argument('--production-run', type=Path, help='Completed production qualification directory')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--rows', nargs='+', choices=[r[0] for r in ROWS])
    parser.add_argument('--repetitions', type=int)
    args = parser.parse_args()
    if args.fgm2:
        if (args.output is None or args.fgm2_private_input is None
                or any(v is not None for v in (args.rows,args.protocol,args.baseline,args.production_run,
                                               args.repetitions,args.native_binary,args.calibration))):
            parser.error('FGM-2 requires --output and --fgm2-private-input, with optional --binary-dir')
        result = execute_fgm2(args.fgm2_private_input,args.binary_dir or ROOT/'build/metal',args.output,
                              private_reference=args.fgm2_private_reference)
        if not result['complete']:
            raise SystemExit(1)
        return
    if args.fgm2_private_input is not None or args.fgm2_private_reference is not None:
        parser.error('--fgm2-private-input and --fgm2-private-reference require --fgm2')
    if args.prepare_effectiveness_panel or args.effectiveness_panel or args.summarize_effectiveness:
        if args.output is None or any(v is not None for v in (args.rows,args.protocol,args.baseline,args.production_run,args.repetitions,args.native_binary)):
            parser.error('FGM-1 requires --output and does not accept legacy measurement options')
        if args.prepare_effectiveness_panel:
            if args.binary_dir or args.calibration:
                parser.error('panel preparation only accepts corpus and output')
            prepare_effectiveness_panel(args.prepare_effectiveness_panel, args.output)
        elif args.summarize_effectiveness:
            if args.binary_dir or args.calibration:
                parser.error('offline summary only accepts measurement and output')
            result = summarize_effectiveness(args.summarize_effectiveness)
            with args.output.open('x') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write('\n')
        else:
            result = execute_effectiveness(args.effectiveness_panel, args.binary_dir or ROOT/'build/metal',
                                           args.output, args.calibration)
            if not result['complete']:
                raise SystemExit(1)
        return
    if args.binary_dir or args.calibration:
        parser.error('--binary-dir and --calibration require --effectiveness-panel')
    if args.native_config is not None:
        if args.native_binary is None or args.output is None:
            parser.error('--native-config requires --native-binary and --output')
        if any(value is not None for value in (args.rows,args.protocol,args.baseline,args.production_run)):
            parser.error('native workflow adapter accepts configuration, binary, output and repetitions only')
        execute_native(args.native_config,args.native_binary,args.output,
                       args.repetitions if args.repetitions is not None else 1)
        return
    if args.native_binary is not None:
        parser.error('--native-binary requires --native-config')
    if args.host is not None:
        if args.output is None:
            parser.error('--host requires --output')
        if any(value is not None for value in (args.rows,args.protocol,args.baseline,args.production_run)):
            parser.error('--host uses fixed public workloads and accepts only --output and --repetitions')
        execute_host(args.host,args.output,args.repetitions if args.repetitions is not None else 6)
        return
    if args.freeze is not None:
        if any(value is not None for value in (args.output, args.protocol, args.rows, args.repetitions, args.baseline, args.production_run)):
            parser.error('--freeze accepts only a new directory')
        freeze_baseline(args.freeze)
        return
    if args.output is None:
        parser.error('--output is required')
    if args.summarize is not None:
        if any(value is not None for value in (args.protocol, args.rows, args.repetitions, args.baseline, args.production_run)):
            parser.error('--summarize accepts only evidence and --output')
        result = summarize(args.summarize)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print('Baseline repeatability only; no practical allowance or candidate verdict established.')
        return
    if args.qualify_profile is not None:
        if args.baseline is None or args.production_run is None:
            parser.error('--qualify-profile requires --baseline and --production-run')
        if args.protocol is not None or args.rows is not None or args.repetitions is not None:
            parser.error('profile qualification uses the production protocol and all eight workloads once')
        execute((args.baseline/'source').resolve(strict=True), args.output.resolve(), None,
                [r[0] for r in ROWS], 1, profile=args.qualify_profile.resolve(strict=True),
                production_run=args.production_run.resolve(strict=True))
        return
    if args.baseline is not None or args.production_run is not None:
        parser.error('--baseline and --production-run require --qualify-profile')
    repetitions = args.repetitions if args.repetitions is not None else 1
    if not 1 <= repetitions <= 12:
        parser.error('repetitions must be 1..12')
    config = json.loads(args.protocol.read_text()) if args.protocol else protocol({r[0]:6 for r in ROWS})
    if config['baseline'] != BASELINE or config['shader_library_mode'] != 'metallib':
        parser.error('wrong baseline or library mode')
    if set(config['rounds']) != {r[0] for r in ROWS} or any(type(n) is not int or n < 2 for n in config['rounds'].values()):
        parser.error('every workload needs a positive round count >=2')
    execute(args.source.resolve(strict=True), args.output.resolve(), config,
            args.rows or [r[0] for r in ROWS], repetitions)


if __name__ == '__main__':
    main()
