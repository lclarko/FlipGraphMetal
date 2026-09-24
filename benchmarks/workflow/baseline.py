"""Qualify pinned production programs using public fixtures and retained guards."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
sys.path.insert(0, str(ROOT / 'tests/metal'))
from application import digest, hardware_inventory, write_json
from guard import run as guarded_run
from smoke import dispatch_evidence
from verify import verify, reconstruct
from profile_baseline import archive_files

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
        source.mkdir()
        for name, (payload, mode) in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
            path.chmod(mode)
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
    expected = set()
    with tarfile.open(archive_path, mode='r:') as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                raise ValueError('unsafe baseline source archive')
            if member.isdir():
                continue
            if not member.isfile() or member.name in expected:
                raise ValueError('special or duplicate baseline source member')
            expected.add(member.name)
            live = source / relative
            if live.is_symlink() or not live.is_file() or live.read_bytes() != archive.extractfile(member).read():
                raise ValueError('live source differs from baseline archive: ' + member.name)
    actual = set()
    for live in source.rglob('*'):
        relative = live.relative_to(source)
        if relative.parts[0] == 'build' or '__pycache__' in relative.parts:
            continue
        if live.is_symlink():
            raise ValueError('linked baseline source')
        if live.is_file():
            actual.add(relative.as_posix())
    if actual != expected:
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


def execute(source, output, config, names, repetitions):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', config)
    (output/'protocol.sha256').write_text(digest(output/'protocol.json')+'\n')
    freeze = json.loads((source.parent / 'freeze.json').read_text())
    if freeze['commit'] != BASELINE or digest(source.parent/'source.tar') != freeze['archive_sha256']:
        raise ValueError('wrong or changed baseline snapshot')
    assert_source_snapshot(source, freeze)
    identity = inventory(source)
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
    receipt = {'schema': 'fgm-baseline-qualification-v1', 'complete': False,
               'baseline': freeze, 'build_inventory': identity, 'hardware': hardware_inventory(),
               'os': subprocess.check_output(['sw_vers'], text=True).strip(),
               'compiler': subprocess.check_output(['xcrun', 'clang++', '--version'], text=True).strip(),
               'protocol': config, 'protocol_sha256': content_hash(config), 'harness': harness,
               'attempts': [], 'performance_acceptance': 'NOT EVALUATED'}
    write_json(output/'qualification.json', receipt)
    try:
        for repeat in range(repetitions):
            for row in ROWS:
                name, program, fixture, f2, kernel = row
                if name not in names:
                    continue
                assert_inventory(source, identity)
                fixture_path = source/'tests/metal/fixtures'/fixture
                data = read_raw(fixture_path, f2)
                kind = 'search' if program.startswith('flip_graph') else 'minimizer' if program.startswith('complexity') else 'reducer'
                attempt = output/f'{repeat:02d}-{name}'
                attempt.mkdir()
                adapted = attempt/'input.txt'
                adapted.write_bytes(adapter_bytes(data, kind))
                reference = normalized_input(data) if not f2 else data
                write_json(attempt/'effective-input.json', reference)
                record = {'workload': name, 'repeat': repeat, 'complete': False,
                          'fixture_sha256': digest(fixture_path), 'input_sha256': digest(adapted),
                          'expected_kernel': kernel, 'exports': [], 'verification_seconds': None}
                receipt['attempts'].append(record)
                write_json(output/'qualification.json', receipt)
                start = time.monotonic()
                argv = command(row, config, source/'build/metal', adapted, attempt/'exports')
                record['argv'] = argv
                write_json(output/'qualification.json', receipt)
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
                record['complete'] = True
                write_json(output/'qualification.json', receipt)
                print(name, f"process={record['process_seconds']:.2f}s GPU={record['gpu_work_seconds']:.3f}s exports={len(record['exports'])}", flush=True)
        assert_inventory(source, identity)
        assert_source_snapshot(source, freeze)
        receipt['complete'] = True
        receipt['measurement_acquisition'] = 'complete'
        receipt['qualification_gaps'] = ['host phases need a behavior-qualified diagnostic', 'Metal allocation profile not yet qualified', 'practical budgets unapproved']
        if any(row['workload'] == 'mutation-reducer' and row['applied_mutation_evidence'] == 'NOT VERIFIED' for row in receipt['attempts']):
            receipt['qualification_gaps'].append('actual reducer mutation not established for every attempted case')
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        write_json(output/'qualification.json', receipt)
        (output/'qualification.sha256').write_text(digest(output/'qualification.json')+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--freeze', type=Path, metavar='NEW_DIRECTORY',
                           help='Prepare pinned source only; no build or GPU execution')
    operation.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--rows', nargs='+', choices=[r[0] for r in ROWS], default=[r[0] for r in ROWS])
    parser.add_argument('--repetitions', type=int, default=1)
    args=parser.parse_args()
    if args.freeze is not None:
        if args.output is not None or args.protocol is not None:
            parser.error('--freeze does not accept --output or --protocol')
        freeze_baseline(args.freeze)
        return
    if args.output is None:
        parser.error('--source requires --output')
    if not 1 <= args.repetitions <= 12:
        parser.error('repetitions must be 1..12')
    config=json.loads(args.protocol.read_text()) if args.protocol else protocol({r[0]:6 for r in ROWS})
    if config['baseline'] != BASELINE or config['shader_library_mode'] != 'metallib':
        parser.error('wrong baseline or library mode')
    if set(config['rounds']) != {r[0] for r in ROWS} or any(type(n) is not int or n < 2 for n in config['rounds'].values()):
        parser.error('every workload needs a positive round count >=2')
    execute(args.source.resolve(strict=True), args.output.resolve(), config, args.rows, args.repetitions)


if __name__ == '__main__':
    main()
