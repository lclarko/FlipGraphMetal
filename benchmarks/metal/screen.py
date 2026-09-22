import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from application import artifacts, digest, source_identity, terminate, wired_memory, write_json

ROOT = Path(__file__).resolve().parents[2]


def check_log(log, expected_rounds=None):
    dispatches = [(name, float(value) / 1000) for name, value in re.findall(
        r'Metal dispatch (\w+):.*?, ([\d.]+) ms GPU', log)]
    if not dispatches or any(seconds <= 0 for _, seconds in dispatches):
        raise RuntimeError('missing or nonpositive GPU dispatch timings')
    gpu_seconds = [seconds for name, seconds in dispatches if name in ('randomWalkKernel', 'randomWalkCompactKernel')]
    rounds = re.findall(r'^(?:ROUND|REPEAT) (\d+).* MATCH$', log, re.M)
    if expected_rounds is not None:
        if rounds != [str(i) for i in range(expected_rounds)]:
            raise RuntimeError('missing, duplicated or reordered exact walk comparisons')
        if len(gpu_seconds) != expected_rounds:
            raise RuntimeError('GPU dispatch count does not match exact walk comparisons')
    return gpu_seconds, rounds


def execute(argv, directory, expected_exports, expected_rounds=None):
    directory.mkdir(parents=True, exist_ok=False)
    record = {'argv': argv, 'complete': False, 'memory': []}
    process = None
    start = time.monotonic()
    try:
        initial = wired_memory()
        record['memory'].append({'seconds': 0, 'wired_bytes': initial})
        if initial > 3 * 1024**3:
            raise RuntimeError('wired memory exceeded 3 GiB before launch')
        with (directory / 'run.log').open('w') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={key: os.environ[key] for key in
                    ['PATH', 'HOME', 'TMPDIR', 'DEVELOPER_DIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME'] if key in os.environ})
            while process.poll() is None:
                remaining = 45 - (time.monotonic() - start)
                if remaining <= 0:
                    raise RuntimeError('time limit')
                value = wired_memory(timeout=min(2, remaining))
                record['memory'].append({'seconds': time.monotonic() - start, 'wired_bytes': value})
                if value > 3 * 1024**3:
                    raise RuntimeError('wired memory exceeded 3 GiB')
                if time.monotonic() - start >= 45:
                    raise RuntimeError('time limit')
                try:
                    process.wait(timeout=min(.25, 45 - (time.monotonic() - start)))
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f'exit {process.returncode}')
        sys.path.insert(0, str(ROOT / 'tests/metal'))
        from verify import verify
        files = sorted((directory / 'schemes').glob('*.json'))
        if len(files) != expected_exports:
            raise RuntimeError(f'expected {expected_exports} exports, found {len(files)}')
        for path in files:
            verify(json.loads(path.read_text()))
        record['verified_exports'] = len(files)
        log = (directory / 'run.log').read_text()
        record['gpu_seconds'], record['matched_rounds'] = check_log(log, expected_rounds)
        record['complete'] = True
    except Exception as error:
        record['error'] = str(error)
    finally:
        if process is not None:
            try:
                terminate(process)
            except Exception as error:
                record['complete'] = False
                record['cleanup_error'] = str(error)
            record['exit_code'] = process.returncode
        record['wall_seconds'] = time.monotonic() - start
        record['artifacts'] = artifacts(directory)
        write_json(directory / 'result.json', record)
        (directory / 'result.sha256').write_text(digest(directory / 'result.json') + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description='Screen frozen CPU/Metal walks under process and memory bounds')
    parser.add_argument('--builds', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--populations', nargs='+', type=int, default=[512, 2048])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 19])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--mode', choices=['matched', 'repeat'], default='matched')
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 or n > 4096 for n in args.populations) or any(s < 1 or s > 2147483647 for s in args.seeds):
        parser.error('invalid population, seed or repetition count')
    builds = [p.resolve(strict=True) for p in args.builds]
    if len(set(p.name for p in builds)) != len(builds):
        parser.error('build directory names must be unique')
    identities = {str(p): {'binary': digest(p / 'matched'), 'manifest': digest(p / 'build.json'),
                           'source': source_identity(p)} for p in builds}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture_sha256 = digest(args.fixture) if args.fixture else None
    write_json(output / 'config.json', {'builds': identities, 'arguments': {k: str(v) for k, v in vars(args).items()},
        'runner_sha256': digest(Path(__file__)), 'fixture_sha256': fixture_sha256})
    records = []
    for count in args.populations:
        for seed in args.seeds:
            for repeat in range(args.repeats):
                for build in builds if repeat % 2 == 0 else builds[::-1]:
                    identity = identities[str(build)]
                    if identity != {'binary': digest(build / 'matched'), 'manifest': digest(build / 'build.json'), 'source': source_identity(build)}:
                        raise RuntimeError('benchmark source or executable changed')
                    if args.fixture and digest(args.fixture) != fixture_sha256:
                        raise RuntimeError('input fixture changed during screen')
                    name = f'{build.name}-n{count}-s{seed}-r{repeat}'
                    directory = output / name
                    argv = [str(build / 'matched'), str(count), '8', str(seed), str(directory / 'schemes'), args.mode]
                    if args.fixture:
                        argv += ['32', str(args.fixture.resolve())]
                    expected = 13 if args.mode == 'repeat' else 6
                    record = execute(argv, directory, count, expected_rounds=expected)
                    record.update(name=name, build=str(build), count=count, seed=seed, repeat=repeat)
                    records.append(record)
                    write_json(output / 'results.json', records)
                    print(name, 'PASS' if record['complete'] else 'INCOMPLETE', record.get('error', ''), flush=True)
                    if not record['complete']:
                        raise RuntimeError(f'incomplete screen: {directory}')


if __name__ == '__main__':
    main()
