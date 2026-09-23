import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import selectors
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = '- iteration time (last / min / max / mean):'
GPU = re.compile(r'Metal dispatch (?P<kernel>randomWalkKernel|randomWalkCompactKernel):.*?, (?P<ms>[\d.]+) ms GPU')
LIMIT = 3 * 1024**3
SOURCE_TIMING = 'source-elapsed-v1'


def source_timing(stdout, backend, count, rounds):
    """Read cumulative application clocks, independent of pipe delivery batching.

    The frozen CPU has no round cap. Its first requested reports define the
    measured window; further completed/partial reports remain explicit overshoot.
    The frozen Metal executable must complete exactly its requested round cap.
    """
    cpu = backend == 'cpu'
    reports, pending = [], {}
    trailing_fragment = ''
    header = False
    for raw_line in stdout.splitlines(keepends=True):
        line = raw_line.rstrip('\r\n')
        if cpu and len(reports) >= rounds and not raw_line.endswith('\n'):
            trailing_fragment = raw_line
            break
        if cpu:
            if line.startswith('| threads:'):
                match = re.fullmatch(r'\| threads:\s+8\s+flip iters:\s+1\.00K\s+iteration:\s+(\d+)\s*\|', line)
                if not match or pending:
                    raise ValueError('invalid or overlapping CPU report ordinal')
                pending = {'iteration': int(match[1])}
            elif line.startswith('| count:'):
                match = re.fullmatch(r'\| count:\s+(\d+)\s+reset iters:\s+1\.0B\s+elapsed:\s+(\d+\.\d{2})\s*\|', line)
                if not match or set(pending) != {'iteration'}:
                    raise ValueError('invalid CPU report elapsed row')
                pending.update(count=int(match[1]), elapsed_seconds=float(match[2]))
        else:
            if re.fullmatch(r'\| Schemes\s+Iteration\s+Elapsed time\s*\|', line):
                if header or pending:
                    raise ValueError('overlapping Metal report header')
                header = True
            elif header:
                match = re.fullmatch(r'\|\s+(\d+)\s+(\d+)\s+(\d+\.\d{3})\s*\|', line)
                if not match:
                    raise ValueError('invalid Metal report elapsed row')
                pending = dict(count=int(match[1]), iteration=int(match[2]), elapsed_seconds=float(match[3]))
                header = False
        if line.startswith(REPORT):
            if set(pending) != {'iteration', 'count', 'elapsed_seconds'}:
                raise ValueError('report ending without complete source timing')
            if pending['iteration'] != len(reports) + 1 or pending['count'] != count:
                raise ValueError('nonconsecutive source ordinal or wrong work count')
            elapsed = pending['elapsed_seconds']
            if not math.isfinite(elapsed) or elapsed < 0 or (reports and elapsed <= reports[-1]['elapsed_seconds']):
                raise ValueError('source elapsed times must increase')
            reports.append(pending)
            pending = {}
    if len(reports) < rounds or (not cpu and (len(reports) != rounds or pending or header)):
        raise ValueError('source report count or ending does not match requested rounds')
    # Both programs truncate the underlying cumulative clock to milliseconds.
    # CPU then rounds to two decimals; Metal prints three. These conservative
    # difference bounds cover both endpoint quantization errors.
    uncertainty = .011 if cpu else .002
    elapsed = reports[rounds - 1]['elapsed_seconds'] - reports[0]['elapsed_seconds']
    if elapsed <= uncertainty:
        raise ValueError('source timing window is too short for printed precision')
    work = (rounds - 1) * count * 1000
    return dict(timing_method=SOURCE_TIMING, source_reports=reports,
                source_partial_report=pending, source_completed_overshoot=len(reports) - rounds,
                source_trailing_fragment=trailing_fragment,
                measured_report_indices=[1, rounds], steady_seconds=elapsed,
                steady_seconds_uncertainty=uncertainty,
                steady_steps_per_second=work / elapsed,
                steady_steps_per_second_bounds=[work / (elapsed + uncertainty), work / (elapsed - uncertainty)],
                quality_scope='CPU termination is asynchronous; exports can include work beyond the measured window. No equal-budget quality claim.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(path):
    suffixes = {'.cpp', '.hpp', '.h', '.mm', '.metal', '.cu', '.cuh', '.py'}
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.suffix in suffixes and not any(
                part in {'.git', 'build', '__pycache__'} for part in p.relative_to(path).parts)}



def hardware_inventory():
    """Retain reproduction fields, never raw profiler output or identifiers."""
    argv = ['system_profiler', '-json', 'SPHardwareDataType', 'SPDisplaysDataType']
    allowed = {
        'SPHardwareDataType': ('machine_name', 'machine_model', 'chip_type', 'cpu_type',
                               'number_processors', 'physical_memory'),
        'SPDisplaysDataType': ('sppci_model', 'spdisplays_cores', 'spdisplays_metal'),
    }
    try:
        capture = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        if capture.returncode:
            return {'argv': argv, 'exit_code': capture.returncode, 'error': 'profiler failed'}
        raw = json.loads(capture.stdout)
        fields = {category: [{key: item[key] for key in keys
                             if key in item and isinstance(item[key], (str, int, float))}
                            for item in raw.get(category, []) if isinstance(item, dict)]
                  for category, keys in allowed.items()}
        return {'argv': argv, 'exit_code': 0, 'fields': fields}
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError):
        return {'argv': argv, 'error': 'profiler unavailable or malformed'}

def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def wired_memory(timeout=2):
    stat = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=timeout, check=True).stdout
    return int(re.search(r'page size of (\d+) bytes', stat)[1]) * int(
        re.search(r'Pages wired down:\s+(\d+)', stat)[1])


def terminate(process):
    deadline = time.monotonic() + 2
    process.poll()
    denied = None
    members = None

    def send(sig, phase):
        nonlocal denied, members
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            leader = process.poll()
            denied = PermissionError(f'process group {process.pid}: EPERM during {phase}; leader={leader}; members={members}')
            if leader is None:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            snapshot = subprocess.run(['/bin/ps', '-A', '-o', 'pid=,pgid='],
                                      capture_output=True, text=True, check=True, timeout=remaining)
            rows = [line.split() for line in snapshot.stdout.splitlines() if line.strip()]
            if not rows or any(len(row) != 2 or not all(field.isdecimal() for field in row)
                               for row in rows):
                raise RuntimeError(f'cannot corroborate process-group cleanup during {phase} from ps')
            members = [[int(field) for field in row] for row in rows if int(row[1]) == process.pid]
            if members:
                denied = PermissionError(f'process group {process.pid}: EPERM during {phase}; leader={leader}; members={members}')
                return None
            return False

    status = send(signal.SIGTERM, 'TERM')
    while status is not False and time.monotonic() < deadline:
        process.poll()
        status = send(0, 'TERM grace probe')
        if status is not False:
            time.sleep(min(.01, max(0, deadline - time.monotonic())))
    if status is not False:
        status = send(signal.SIGKILL, 'KILL after TERM grace')
        if status is None:
            raise denied
    process.wait(timeout=2)


def artifacts(directory):
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.relative_to(directory).as_posix() not in {'result.json', 'result.sha256'}}


def check_metal_library(binary, library, commands, *, resolver=None, recorded_binary=None):
    """Verify the packaged asset, generated header and executable binding."""
    if not isinstance(library, dict) or library.get('mode') != 'metallib':
        raise ValueError('Metal library requires explicit metallib mode')
    name, checksum = library.get('library'), library.get('sha256')
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or '..' in Path(name).parts or str(Path(name)) != name
            or not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum)):
        raise ValueError('invalid Metal library name or digest')
    binary = Path(binary)
    asset = binary.parent / name
    if any(p.is_symlink() for p in (asset, *asset.parents)) or not asset.is_file():
        raise ValueError('Metal library must be an ordinary installed file')
    if resolver:
        resolver.check_tree(asset)
    if digest(asset) != checksum:
        raise ValueError('Metal library digest mismatch')
    header_name = library.get('header')
    if not isinstance(header_name, str) or not Path(header_name).is_absolute():
        raise ValueError('Metal library header must be recorded explicitly')
    header = resolver.resolve(Path(header_name)) if resolver else Path(header_name).resolve(strict=True)
    contents = header.read_text()
    for macro, value in [('METAL_LIBRARY_NAME', name), ('METAL_LIBRARY_SHA256', checksum)]:
        if not re.search(r'^\s*#\s*define\s+' + macro + r'\s+"' + re.escape(value) + r'"\s*$', contents, re.M):
            raise ValueError('Metal library header binding mismatch')
    executable = binary.read_bytes()
    if name.encode() not in executable or checksum.encode() not in executable:
        raise ValueError('Metal library binding is not present in executable')
    bound = [command for command in commands if any(
        arg == '-include' and index + 1 < len(command) and command[index + 1] == header_name
        for index, arg in enumerate(command))]
    target = str(recorded_binary or binary)
    if not any('-o' in command and command[command.index('-o') + 1] == target for command in bound):
        raise ValueError('build argv does not bind the Metal library header to executable')
    if any('METAL_SOURCE_DIR' in arg for command in bound for arg in command):
        raise ValueError('packaged build cannot substitute runtime source')
    return {'library': name, 'sha256': checksum}


def build_identity(backend, binary, source, manifest_path, *, resolver=None):
    original_binary, original_source, original_manifest = binary, source, manifest_path
    resolve = resolver.resolve if resolver else lambda path: path.resolve(strict=True)
    binary, source, manifest_path = map(resolve, (binary, source, manifest_path))
    recorded_binary = str(original_binary) if resolver else str(binary)
    recorded_source = str(original_source) if resolver else str(source)
    recorded_manifest = str(original_manifest) if resolver else str(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    commands = manifest.get('commands')
    if (manifest.get('version') != 1 or not isinstance(commands, list) or not commands
            or any(not isinstance(command, list) or not command or
                   any(not isinstance(arg, str) for arg in command) for command in commands)
            or not isinstance(manifest.get('source_revision'), str) or not manifest['source_revision']):
        raise ValueError('build manifest requires version 1, source_revision and build argv lists')
    files = source_identity(source)
    if (manifest.get('source_root') != recorded_source or manifest.get('source_files') != files
            or not files or manifest.get('binary') != recorded_binary
            or manifest.get('binary_sha256') != digest(binary)):
        raise ValueError('build manifest does not match source and executable')
    if backend != 'cpu' and 'metal_library' in manifest:
        check_metal_library(binary, manifest['metal_library'], commands, resolver=resolver,
                            recorded_binary=recorded_binary)
    elif backend != 'cpu':
        recorded_shader = Path(manifest.get('metal_source_dir', ''))
        shader = resolve(recorded_shader)
        if (not shader.is_dir() or not shader.is_relative_to(source)
                or not recorded_shader.is_relative_to(Path(recorded_source))):
            raise ValueError('Metal shader directory must be inside the recorded source root')
        if str(recorded_shader).encode() not in binary.read_bytes():
            raise ValueError('Metal shader directory is not present in the compiled executable')
        if not any('METAL_SOURCE_DIR=' in arg and str(recorded_shader) in arg
                   for command in commands for arg in command):
            raise ValueError('build argv does not bind the Metal shader directory')
    return {'binary': recorded_binary, 'sha256': digest(binary), 'source': recorded_source, 'files': files,
            'build_manifest': recorded_manifest, 'build_manifest_sha256': digest(manifest_path),
            'build': manifest}


def custom_fixture(path):
    sys.path.insert(0, str(ROOT / 'tests/metal'))
    from verify import verify
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    tokens = text.split()
    if len(tokens) < 4 or any(re.fullmatch(r'[+-]?[0-9]+', token) is None for token in tokens):
        raise ValueError('custom fixture must contain only raw signed-scheme integers')
    values = [int(token) for token in tokens]
    n, rank = values[:3], values[3]
    widths = [n[0] * n[1], n[1] * n[2], n[2] * n[0]]
    if any(not 1 <= dim <= 16 for dim in n) or max(widths) > 64 or not 1 <= rank <= 350:
        raise ValueError('custom fixture requires dimensions 1..16, factor widths <=64 and rank 1..350')
    if len(values) != 4 + sum(widths) * rank or any(value not in (-1, 0, 1) for value in values[4:]):
        raise ValueError('custom fixture coefficient count must match dimensions and rank; values must be -1, 0 or 1')
    data = {'n': n, 'm': rank, 'z2': False}
    offset = 4
    for width, key in zip(widths, 'uvw'):
        data[key] = [values[offset + width * r:offset + width * (r + 1)] for r in range(rank)]
        offset += width * rank
    verify(data)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'rank': rank, 'n': n}, text


def command(binary, backend, count, seed, rounds, fixture, directory, *, fixture_path=None):
    if fixture not in ('naive', 'rank26', 'custom'):
        raise ValueError('unknown fixture: ' + fixture)
    if (fixture == 'custom') != (fixture_path is not None):
        raise ValueError('fixture_path is required only for the custom fixture')
    custom, custom_text = custom_fixture(fixture_path) if fixture == 'custom' else (None, None)
    dimensions = custom['n'] if custom else [3, 3, 3]
    common = ['--seed', str(seed)]
    if fixture == 'naive' or backend != 'cpu':
        for index, dimension in enumerate(dimensions, 1):
            common += [f'-n{index}', str(dimension)]
    if fixture in ('rank26', 'custom'):
        text = (custom_text if fixture == 'custom'
                else (ROOT / 'tests/metal/fixtures/strassen_3x3.txt').read_text())
        input_path = directory / 'input.txt'
        input_path.write_text(('' if backend == 'cpu' else '1\n') + text)
        common += ['--input-path', str(input_path)]
    output = directory / 'schemes'
    output.mkdir()
    if backend == 'cpu':
        options = ['--ring', 'ZT', '--count', str(count), '--threads', '8', '--flip-iterations', '1000',
                   '--min-plus-iterations', '1000000000', '--max-plus-iterations', '1000000000',
                   '--reset-iterations', '1000000000', '--reduce-probability', '0',
                   '--sandwiching-probability', '0', '--copy-best-probability', '0',
                   '--top-count', '1', '--output-path', str(output), '--format', 'json']
    else:
        options = ['--schemes', str(count), '--block-size', '32', '--max-iterations', '1000',
                   '--plus-iterations', '1000000000', '--resize-probability', '0',
                   '--expand-probability', '0', '--reduce-probability', '0', '--basis-probability', '0',
                   '--sandwiching-probability', '0', '--rounds', str(rounds), '--path', str(output)]
    return [str(binary), *common, *options]


def gpu_evidence(stdout, stderr, rounds, expected_kernel=None):
    """Validate retained device and search-dispatch evidence, without GPU execution."""
    if expected_kernel not in (None, 'randomWalkKernel', 'randomWalkCompactKernel'):
        raise ValueError('unknown expected search kernel')
    combined = stdout + '\n' + stderr
    devices = re.findall(r'^Metal device: (.+)$', combined, re.MULTILINE)
    matches = list(GPU.finditer(combined))
    seconds = [float(match['ms']) / 1000 for match in matches]
    kernels = [match['kernel'] for match in matches]
    dispatch_lines = [line for line in combined.splitlines() if line.startswith('Metal dispatch randomWalk')]
    if (len(devices) != 1 or not devices[0].startswith('Apple ')
            or len(matches) != rounds or len(dispatch_lines) != rounds
            or any(not math.isfinite(value) or value <= 0 for value in seconds)
            or (expected_kernel is not None and any(kernel != expected_kernel for kernel in kernels))):
        raise ValueError('Metal Apple device, search kernel or positive dispatch timing evidence invalid')
    return dict(gpu_device=devices[0], gpu_seconds=seconds, gpu_kernels=kernels)


def run_case(case, argv, directory, rounds, timing='arrival', *, expected_kernel=None):
    sys.path.insert(0, str(ROOT / 'tests/metal'))
    from verify import verify
    record = {**case, 'argv': argv, 'complete': False, 'reports': [], 'gpu_seconds': [], 'gpu_kernels': [], 'memory': [], 'coalesced_reports': []}
    if expected_kernel is not None:
        record['expected_kernel'] = expected_kernel
    write_json(directory / 'result.json', record)
    start = time.monotonic()
    reason = None
    intentional = False
    process = None
    sampler = None
    stop_sampling = threading.Event()
    memory_events = queue.Queue()
    try:
        initial = wired_memory()
        record['memory'].append({'seconds': 0, 'wired_bytes': initial})
        if initial > LIMIT:
            raise RuntimeError('wired memory exceeded 3 GiB before launch')
        process_start = time.monotonic()
        process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, env={**{key: os.environ[key] for key in
                                       ['PATH', 'HOME', 'TMPDIR', 'DEVELOPER_DIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME']
                                       if key in os.environ}, 'OMP_NUM_THREADS': '8'})
        selector = selectors.DefaultSelector()
        buffers = {}
        logs = {}
        for name, stream in [('stdout', process.stdout), ('stderr', process.stderr)]:
            selector.register(stream, selectors.EVENT_READ, name)
            buffers[name] = b''
            logs[name] = (directory / (name + '.log')).open('wb')
        def sample_memory():
            while not stop_sampling.wait(1):
                remaining = start + 45 - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    value = wired_memory(timeout=min(2, remaining))
                    memory_events.put({'seconds': time.monotonic() - start, 'wired_bytes': value})
                except Exception as error:
                    memory_events.put({'error': str(error)})
                    return

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        try:
            while selector.get_map():
                now = time.monotonic()
                if now - start >= 45:
                    reason = 'time limit'
                    break
                while not memory_events.empty():
                    sample = memory_events.get_nowait()
                    if 'error' in sample:
                        raise RuntimeError('memory sampling failed: ' + sample['error'])
                    record['memory'].append(sample)
                    if sample['wired_bytes'] > LIMIT:
                        reason = 'wired memory exceeded 3 GiB'
                if reason:
                    break
                for key, _ in selector.select(timeout=0.05):
                    data = os.read(key.fileobj.fileno(), 65536)
                    received = time.monotonic() - start
                    reports_in_chunk = []
                    name = key.data
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    logs[name].write(data)
                    logs[name].flush()
                    buffers[name] += data
                    while b'\n' in buffers[name]:
                        line, buffers[name] = buffers[name].split(b'\n', 1)
                        text = line.decode(errors='replace')
                        if name == 'stdout' and REPORT in text:
                            reports_in_chunk.append(len(record['reports']))
                            record['reports'].append(received)
                            if case['backend'] == 'cpu' and len(record['reports']) == rounds:
                                intentional = True
                                terminate(process)
                        match = GPU.search(text)
                        if match:
                            record['gpu_seconds'].append(float(match['ms']) / 1000)
                            record['gpu_kernels'].append(match['kernel'])
                    if len(reports_in_chunk) > 1:
                        record['coalesced_reports'].append(reports_in_chunk)
        finally:
            stop_sampling.set()
            sampler.join(timeout=2.1)
            terminate(process)
            drain_deadline = time.monotonic() + 2
            while selector.get_map() and time.monotonic() < drain_deadline:
                for key, _ in selector.select(timeout=.05):
                    remaining = os.read(key.fileobj.fileno(), 65536)
                    if remaining:
                        logs[key.data].write(remaining)
                    else:
                        selector.unregister(key.fileobj)
            if selector.get_map():
                reason = reason or 'output pipes did not close after process group termination'
            selector.close()
            for log in logs.values():
                log.close()
            for stream in (process.stdout, process.stderr):
                stream.close()
        record['exit_code'] = process.wait(timeout=2)
        record['process_seconds'] = time.monotonic() - process_start
        while not memory_events.empty():
            sample = memory_events.get_nowait()
            if 'error' in sample:
                reason = reason or ('memory sampling failed: ' + sample['error'])
            else:
                record['memory'].append(sample)
                if sample['wired_bytes'] > LIMIT:
                    reason = reason or 'wired memory exceeded 3 GiB'
        if reason:
            raise RuntimeError(reason)
        if timing == 'arrival' and record['coalesced_reports']:
            raise RuntimeError('multiple reports arrived in one read; timing is unreliable')
        if (len(record['reports']) < rounds or
                ((timing == 'arrival' or case['backend'] != 'cpu') and len(record['reports']) != rounds)):
            raise RuntimeError(f'expected {rounds} reports, got {len(record["reports"])}')
        if case['backend'] == 'cpu':
            if not intentional or record['exit_code'] not in (0, -signal.SIGTERM):
                raise RuntimeError('CPU did not stop cleanly at report limit')
        else:
            if record['exit_code']:
                raise RuntimeError('Metal exit invalid')
            record.update(gpu_evidence((directory / 'stdout.log').read_text(),
                                       (directory / 'stderr.log').read_text(), rounds, expected_kernel))
        exports = {}
        for path in sorted((directory / 'schemes').rglob('*.json')):
            data = json.loads(path.read_text())
            if data.get('n') != case.get('custom_fixture', {}).get('n', [3, 3, 3]) or data.get('z2') is not False:
                raise RuntimeError(f'export domain mismatch: {path}')
            verify(data)
            exports[str(path.relative_to(directory))] = digest(path)
        record['exports'] = exports
        record['verified_exports'] = len(exports)
        if timing == 'source-elapsed':
            record.update(source_timing((directory / 'stdout.log').read_text(), case['backend'], case['count'], rounds))
            if record['source_reports'][-1]['elapsed_seconds'] > record['process_seconds'] + record['steady_seconds_uncertainty']:
                raise RuntimeError('source clock elapsed exceeds observed process duration')
            arrival_interval = record['reports'][rounds - 1] - record['reports'][0]
            record['arrival_steps_per_second'] = ((rounds - 1) * case['count'] * 1000 / arrival_interval
                                                  if arrival_interval > 0 else None)
        elif timing == 'arrival':
            record['steady_steps_per_second'] = (rounds - 1) * case['count'] * 1000 / (
                record['reports'][-1] - record['reports'][0])
        else:
            raise ValueError('unknown timing method')
        record['complete'] = True
    except Exception as error:
        record['error'] = str(error)
    finally:
        stop_sampling.set()
        if sampler:
            sampler.join(timeout=2.1)
        if process is not None:
            try:
                terminate(process)
            except Exception as error:
                record['complete'] = False
                record['cleanup_error'] = str(error)
        record['wall_seconds'] = time.monotonic() - start
        record['intentional_termination'] = intentional
        record['artifacts'] = artifacts(directory)
        write_json(directory / 'result.json', record)
        (directory / 'result.sha256').write_text(digest(directory / 'result.json') + '\n')
    return record


def case_matrix(backends, fixtures, populations, seeds, repeats, counterbalance=False):
    if counterbalance and (set(backends) != {'cpu', 'baseline', 'candidate'} or repeats < 2 or repeats % 2):
        raise ValueError('counterbalance requires cpu, baseline, candidate and a positive even repeat count')
    cases = []
    for fixture in fixtures:
        for count in populations:
            for seed_index, seed in enumerate(seeds):
                for repeat in range(repeats):
                    panel = repeat // 2
                    orientation = 'ABBA' if (seed_index + panel) % 2 == 0 else 'BAAB'
                    order = backends
                    if counterbalance:
                        order = ['cpu', 'baseline', 'candidate'] if orientation == 'ABBA' else ['cpu', 'candidate', 'baseline']
                    if repeat % 2:
                        order = order[::-1]
                    for position, backend in enumerate(order):
                        case = dict(name=f'{fixture}-{backend}-n{count}-s{seed}-r{repeat}', fixture=fixture,
                                    backend=backend, count=count, seed=seed, repeat=repeat)
                        if counterbalance:
                            case.update(panel=panel, orientation=orientation, panel_position=3 * (repeat % 2) + position)
                        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description='Bounded production CPU/Metal comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'baseline', 'candidate'], default=['cpu', 'baseline', 'candidate'])
    for backend in ['cpu', 'baseline', 'candidate']:
        parser.add_argument('--' + backend, type=Path)
        parser.add_argument('--' + backend + '-source', type=Path)
        parser.add_argument('--' + backend + '-build-manifest', type=Path)
    parser.add_argument('--populations', nargs='+', type=int, choices=[512, 1024, 2048], default=[512, 1024, 2048])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 19])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rounds', type=int, choices=[6, 17, 33], default=6)
    parser.add_argument('--timing', choices=['arrival', 'source-elapsed'], default='arrival',
                        help='source-elapsed uses frozen report-entry clocks and retains arrival batching as a diagnostic')
    parser.add_argument('--fixtures', nargs='+', choices=['naive', 'rank26', 'custom'], default=['naive', 'rank26'])
    parser.add_argument('--fixture-path', type=Path)
    parser.add_argument('--expected-kernel', choices=['randomWalkKernel', 'randomWalkCompactKernel'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--counterbalance', action='store_true', help='CPU-bracketed ABBA/BAAB panels; requires all three backends and even repeats')
    args = parser.parse_args()
    if ('custom' in args.fixtures) != (args.fixture_path is not None):
        parser.error('--fixture-path is required if and only if custom is selected')
    custom = None
    if args.fixture_path is not None:
        try:
            custom, _ = custom_fixture(args.fixture_path)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.repeats < 1 or any(seed < 1 or seed > 2147483647 for seed in args.seeds):
        parser.error('repeats and seeds must be positive; seeds must fit signed int')
    for values in [args.backends, args.populations, args.seeds, args.fixtures]:
        if len(values) != len(set(values)):
            parser.error('duplicate matrix entries are not allowed')
    if args.counterbalance and (set(args.backends) != {'cpu', 'baseline', 'candidate'} or args.repeats % 2):
        parser.error('--counterbalance requires all three backends and even repeats')
    identities = {}
    for backend in ['cpu', 'baseline', 'candidate']:
        binary, source = getattr(args, backend), getattr(args, backend + '_source')
        manifest = getattr(args, backend + '_build_manifest')
        if backend not in args.backends:
            if binary or source or manifest:
                parser.error(f'{backend} paths provided but backend not selected')
            continue
        if not binary or not source or not binary.is_file() or not source.is_dir():
            parser.error(f'{backend} requires an executable and source directory')
        if not manifest or not manifest.is_file():
            parser.error(f'{backend} requires a build manifest')
        try:
            identities[backend] = build_identity(backend, binary, source, manifest)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    cases = case_matrix(args.backends, args.fixtures, args.populations, args.seeds, args.repeats, args.counterbalance)
    if custom:
        for case in cases:
            if case['fixture'] == 'custom':
                case['custom_fixture'] = custom
    config = {'version': 1, 'rounds': args.rounds, 'cases': cases, 'identities': identities,
              'fixture_sha256': digest(ROOT / 'tests/metal/fixtures/strassen_3x3.txt'),
              'runner_sha256': digest(Path(__file__)), 'platform': platform.platform(),
              'developer_dir': os.environ.get('DEVELOPER_DIR'), 'iterations_per_round': 1000,
              'time_limit': 45, 'wired_limit': LIMIT}
    if args.expected_kernel is not None:
        config['expected_kernel'] = args.expected_kernel
    if args.timing == 'source-elapsed':
        config['timing_method'] = SOURCE_TIMING
    if custom:
        config['custom_fixture'] = custom
    if args.counterbalance:
        config['counterbalance'] = {'enabled': True, 'seed_order': args.seeds, 'panels_per_seed': args.repeats // 2,
                                  'orientation_rule': '(seed_index + panel) % 2: 0=ABBA, 1=BAAB; CPU brackets GPU quartet'}
    output = args.output.resolve()
    if args.resume:
        if not (output / 'config.json').is_file() or json.loads((output / 'config.json').read_text()) != config:
            parser.error('resume requires identical configuration, source and executable hashes')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', config)
        machine = {'hardware': hardware_inventory()}
        for name, argv in [('compiler', ['xcrun', 'clang++', '--version']),
                           ('power', ['pmset', '-g', 'therm'])]:
            try:
                capture = subprocess.run(argv, capture_output=True, text=True, timeout=10)
                machine[name] = {'argv': argv, 'exit_code': capture.returncode, 'stdout': capture.stdout, 'stderr': capture.stderr}
            except (OSError, subprocess.TimeoutExpired) as error:
                machine[name] = {'argv': argv, 'error': str(error)}
        write_json(output / 'machine.json', machine)
    results = []
    for case in cases:
        backend = case['backend']
        identity = identities[backend]
        if build_identity(backend, Path(identity['binary']), Path(identity['source']),
                          Path(identity['build_manifest'])) != identity:
            raise RuntimeError('source, executable or build manifest changed during matrix')
        if digest(ROOT / 'tests/metal/fixtures/strassen_3x3.txt') != config['fixture_sha256']:
            raise RuntimeError('input fixture changed during matrix')
        if custom and digest(Path(custom['path'])) != custom['sha256']:
            raise RuntimeError('custom input fixture changed during matrix')
        directory = output / case['name']
        if directory.exists():
            path = directory / 'result.json'
            if not path.is_file():
                raise RuntimeError(f'incomplete case directory: {directory}; use a new output directory')
            if not (directory / 'result.sha256').is_file() or digest(path) != (directory / 'result.sha256').read_text().strip():
                raise RuntimeError(f'altered or unsealed record: {directory}')
            record = json.loads(path.read_text())
            if artifacts(directory) != record['artifacts']:
                raise RuntimeError(f'altered artifact inventory: {directory}')
            if not record.get('complete') or any(record.get(k) != v for k, v in case.items()):
                raise RuntimeError(f'incomplete or altered record: {directory}; use a new output directory')
            if args.expected_kernel is not None:
                if record.get('expected_kernel') != args.expected_kernel:
                    raise RuntimeError('retained record expected kernel differs from configured protocol')
                if backend != 'cpu':
                    evidence = gpu_evidence((directory / 'stdout.log').read_text(),
                                            (directory / 'stderr.log').read_text(), args.rounds, args.expected_kernel)
                    if any(record.get(key) != value for key, value in evidence.items()):
                        raise RuntimeError('retained GPU evidence differs from logs')
            for relative, expected in record['exports'].items():
                if digest(directory / relative) != expected:
                    raise RuntimeError(f'altered export: {directory / relative}')
        else:
            directory.mkdir()
            argv = command(identities[case['backend']]['binary'], case['backend'], case['count'], case['seed'],
                           args.rounds, case['fixture'], directory,
                           fixture_path=Path(custom['path']) if case['fixture'] == 'custom' else None)
            if case['fixture'] == 'custom':
                copied = (directory / 'input.txt').read_bytes()
                raw = copied if case['backend'] == 'cpu' else copied.removeprefix(b'1\n')
                if hashlib.sha256(raw).hexdigest() != custom['sha256']:
                    raise RuntimeError('custom fixture changed while preparing the case')
            record = run_case(case, argv, directory, args.rounds, args.timing, expected_kernel=args.expected_kernel)
        results.append(record)
        write_json(output / 'results.json', results)
        print(case['name'], 'PASS' if record['complete'] else 'INCOMPLETE', record.get('steady_steps_per_second'), flush=True)
        if not record['complete']:
            raise RuntimeError(f'{record.get("error")}; evidence: {directory}')


if __name__ == '__main__':
    main()
