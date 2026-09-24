"""Qualify pinned production programs using public fixtures and retained guards."""
import argparse
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
import sys
import time
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
sys.path.insert(0, str(ROOT / 'tests/metal'))
from application import digest, hardware_inventory, write_json, source_identity
from guard import run as guarded_run
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
                record['resource_violation']='time limit' in reason or 'wired memory exceeded' in reason
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


def execute_host(binary, output, plan_path):
    """Acquire six new trials only after explicit prospective approvals are present."""
    import performance
    binary=binary.resolve(strict=True)
    plan_bytes=plan_path.read_bytes();plan=json.loads(plan_bytes)
    if set(plan)!= {'schema','evaluations','parent'} or plan['schema']!='fgm-host-plan-v1':
        raise ValueError('invalid host plan')
    evaluations=plan['evaluations']
    if set(evaluations)!=set(HOST_WORKLOADS):raise ValueError('host plan requires all five workloads')
    candidate=host_candidate_identity(binary)
    hardware=host_machine_identity();hardware_hash=content_hash(hardware)
    active=[]
    for name,document in evaluations.items():
        verdict=performance.evaluate(document)  # Validate declared prior evidence; launch nothing.
        budget=document['budget']
        if budget['status']!='approved' or budget['workload']!=name or any(budget[key]!=value for key,value in candidate.items()):
            raise ValueError('host approval or candidate identity mismatch')
        family=document['protocol']['mandatory_endpoints']
        if (document['protocol']['alpha']!=.05 or len(family)!=5
                or {item['workload'] for item in family}!=set(HOST_WORKLOADS)
                or any(item['endpoint']!='workflow_seconds' for item in family)):
            raise ValueError('host latency family must include all five endpoints at alpha .05')
        if document['look']==1 and len(document['observations'])==6:
            if verdict['verdict']!='PASS':raise ValueError('retained first look must pass or declare a second look')
        else:
            expected_count=0 if document['look']==1 else 6
            if len(document['observations'])!=expected_count:
                raise ValueError('host acquisition requires zero or six retained observations')
            active.append(name)
    if not active:raise ValueError('host plan has no pending acquisition')
    if plan['parent'] is not None:
        parent=plan['parent']
        if set(parent)!= {'path','sha256'} or digest(Path(parent['path']))!=parent['sha256']:
            raise ValueError('host parent receipt identity mismatch')
        previous=json.loads(Path(parent['path']).read_text())
        if previous.get('complete') is not True or previous.get('error'):
            raise ValueError('host parent acquisition incomplete')
        for name,document in evaluations.items():
            if document['observations']!=previous['evaluations'][name]['observations']:
                raise ValueError('host parent observations changed')
            for index,observation in enumerate(document['observations']):
                artifact=Path(parent['path']).parent/f'{index:02d}-{name}'/'trial.json'
                if digest(artifact)!=observation['evidence_sha256']:
                    raise ValueError('host parent trial evidence changed')
    elif any(document['observations'] or document['look']!=1 for document in evaluations.values()):
        raise ValueError('retained observations require a parent receipt')
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_bytes(plan_bytes)
    build_receipt=output/'native-build.json'
    build_receipt.write_bytes(binary.with_name(binary.name+'.build.json').read_bytes())
    binary_hash=digest(binary)
    if content_hash({'binary_sha256':binary_hash,'receipt_sha256':digest(build_receipt)})!=candidate['candidate_build']:
        raise ValueError('host build changed while retaining receipt')
    work=prepare_host_inputs(ROOT,output/'inputs')
    harness={str(path.relative_to(ROOT)):digest(path) for path in
             (Path(__file__),ROOT/'benchmarks/workflow/performance.py',ROOT/'benchmarks/metal/guard.py',
              ROOT/'benchmarks/metal/application.py',ROOT/'tests/metal/verify.py',ROOT/'tests/workflow/identity_oracle.py')}
    receipt=dict(schema='fgm-host-qualification-v1',complete=False,plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                 parent=plan['parent'],candidate=candidate,hardware=hardware,harness=harness,
                 source_files=source_identity(ROOT),binary_sha256=binary_hash,
                 build_receipt='native-build.json',build_receipt_sha256=digest(build_receipt),
                 revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                 attempts=[],evaluations=copy.deepcopy(evaluations),performance_acceptance='NOT EVALUATED')
    for name in active:
        receipt['evaluations'][name].update(acquisition_complete=False,acquisition_error=None)
    def save():write_json(output/'qualification.json',receipt)
    save()
    try:
        for name,document in receipt['evaluations'].items():
            for trial in document['protocol']['prospective_trials']:
                identity=trial['identity']
                expected=dict(fixture=work[name]['fixture_sha256'],config=content_hash(HOST_LIMITS),seed='7',work=name,
                              build_settings=candidate['candidate_build'],shader_mode='not-applicable',hardware=hardware_hash)
                if identity!=expected:raise ValueError('host prospective input/configuration identity mismatch')
        for offset in range(6):
            for name in HOST_WORKLOADS:
                if name not in active:continue
                index=offset+(0 if receipt['evaluations'][name]['look']==1 else 6)
                if host_candidate_identity(binary)!=candidate:raise ValueError('host source/build changed')
                document=receipt['evaluations'][name]
                if content_hash({path.name:digest(path) for path in work[name]['input_files']})!=work[name]['fixture_sha256']:
                    raise ValueError('host fixture changed')
                record={'workload':name,'trial_id':document['protocol']['prospective_trials'][index]['trial_id'],
                        'complete':False,'error':None}
                receipt['attempts'].append(record);save()
                attempt=output/f'{index:02d}-{name}'
                try:run_host_attempt(binary,name,work[name],attempt,record,save)
                finally:
                    if record.get('peak_process_rss_bytes',0)>document['budget']['peak_process_rss_limit']:
                        record['resource_violation']=True
                    write_json(attempt/'trial.json',record)
                    document['observations'].append(dict(trial_id=record['trial_id'],complete=record['complete'],
                        value=record.get('workflow_seconds'),peak_process_rss_bytes=record.get('peak_process_rss_bytes'),
                        **candidate,identity=document['protocol']['prospective_trials'][index]['identity'],
                        evidence_sha256=digest(attempt/'trial.json'),resource_violation=record.get('resource_violation',False),
                        error=record['error']))
                    save()
                if record['resource_violation']:raise RuntimeError('host process memory budget exceeded')
                if content_hash({path.name:digest(path) for path in work[name]['input_files']})!=work[name]['fixture_sha256']:
                    raise ValueError('host fixture changed')
        if host_candidate_identity(binary)!=candidate:raise ValueError('host source/build changed')
        for name in active:receipt['evaluations'][name]['acquisition_complete']=True
        receipt['complete']=True
    except Exception as error:
        receipt['error']=str(error)
        for name in active:
            receipt['evaluations'][name].update(acquisition_complete=False,acquisition_error=str(error))
        raise
    finally:
        save();(output/'qualification.sha256').write_text(digest(output/'qualification.json')+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--freeze', type=Path, metavar='NEW_DIRECTORY',
                           help='Prepare pinned source only; no build or GPU execution')
    operation.add_argument('--source', type=Path, help='Frozen production source to measure')
    operation.add_argument('--host', type=Path, metavar='NATIVE_BINARY', help='Acquire approved native-host workflow trials')
    operation.add_argument('--qualify-profile', type=Path, metavar='PROFILE_DIRECTORY')
    operation.add_argument('--summarize', type=Path, metavar='QUALIFICATION_JSON')
    parser.add_argument('--baseline', type=Path, help='Frozen baseline directory for profile qualification')
    parser.add_argument('--production-run', type=Path, help='Completed production qualification directory')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--rows', nargs='+', choices=[r[0] for r in ROWS])
    parser.add_argument('--repetitions', type=int)
    args = parser.parse_args()
    if args.host is not None:
        if args.output is None or args.protocol is None:
            parser.error('--host requires --protocol and --output')
        if any(value is not None for value in (args.rows,args.repetitions,args.baseline,args.production_run)):
            parser.error('--host uses the frozen plan only')
        execute_host(args.host,args.output,args.protocol)
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
