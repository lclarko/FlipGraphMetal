"""Run one build or regression command with the Metal process and memory limits."""

import argparse
import os
from pathlib import Path
import subprocess
import time

from application import artifacts, digest, terminate, wired_memory, write_json


DEFAULT_WIRED_LIMIT_BYTES = 4294967296
LAUNCH_WIRED_RESERVE_BYTES = 2348810240
LAUNCH_WIRED_CEILING_BYTES = DEFAULT_WIRED_LIMIT_BYTES - LAUNCH_WIRED_RESERVE_BYTES


def launch_headroom_available(wired_bytes):
    """Apply the prospective prelaunch admission ceiling."""
    return wired_bytes <= LAUNCH_WIRED_CEILING_BYTES


def run(argv, output, *, absolute_deadline=None, wired_limit_bytes=DEFAULT_WIRED_LIMIT_BYTES):
    if type(wired_limit_bytes) is not int or wired_limit_bytes <= 0:
        raise ValueError('wired limit must be a positive integer byte count')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    if absolute_deadline is not None and not isinstance(absolute_deadline, (int, float)):
        raise ValueError('absolute deadline must be a monotonic clock value')
    deadline = min(start + 45, absolute_deadline) if absolute_deadline is not None else start + 45
    record = dict(argv=argv, complete=False, time_limit=45, absolute_deadline=absolute_deadline,
                  wired_limit=wired_limit_bytes, memory=[], forced_termination=False, cleanup_failure=False)
    limit_text = (f'{wired_limit_bytes // 1024**3} GiB' if wired_limit_bytes % 1024**3 == 0
                  else f'{wired_limit_bytes} bytes')
    write_json(output / 'result.json', record)
    process = None
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('time limit before memory sample')
        initial = wired_memory(timeout=min(2, remaining))
        record['memory'].append(dict(seconds=0, wired_bytes=initial))
        if initial > wired_limit_bytes:
            raise RuntimeError(f'wired memory exceeded {limit_text} before launch')
        if time.monotonic() >= deadline:
            raise RuntimeError('time limit before launch')
        with (output / 'run.log').open('x') as log:
            if time.monotonic() >= deadline:
                raise RuntimeError('time limit before launch')
            process_start = time.monotonic()
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={key: os.environ[key] for key in
                ['PATH', 'HOME', 'TMPDIR', 'DEVELOPER_DIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME']
                if key in os.environ})
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('time limit')
                value = wired_memory(timeout=min(2, remaining))
                record['memory'].append(dict(seconds=time.monotonic() - start, wired_bytes=value))
                if value > wired_limit_bytes:
                    raise RuntimeError(f'wired memory exceeded {limit_text}')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('time limit')
                try:
                    process.wait(timeout=min(.25, remaining))
                except subprocess.TimeoutExpired:
                    pass
            record['observed_process_seconds'] = time.monotonic() - process_start
            if time.monotonic() >= deadline:
                raise RuntimeError('time limit at completion')
            if process.returncode:
                raise RuntimeError(f'exit {process.returncode}')
        record['complete'] = True
    except Exception as error:
        record['error'] = str(error)
    finally:
        if process is not None:
            try:
                record['forced_termination'] = process.poll() is None
                terminate(process)
            except Exception as error:
                record['complete'] = False
                record['cleanup_failure'] = True
                record['cleanup_error'] = str(error)
            record['exit_code'] = process.returncode
        record['wall_seconds'] = time.monotonic() - start
        if time.monotonic() >= deadline:
            record['complete'] = False
            record.setdefault('error', 'time limit during cleanup')
        record['artifacts'] = artifacts(output)
        write_json(output / 'result.json', record)
        (output / 'result.sha256').write_text(digest(output / 'result.json') + '\n')
    print(output, 'PASS' if record['complete'] else 'INCOMPLETE', record.get('error', ''), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wired-limit-bytes', type=int, default=DEFAULT_WIRED_LIMIT_BYTES)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv:
        parser.error('a command is required')
    raise SystemExit(0 if run(argv, args.output, wired_limit_bytes=args.wired_limit_bytes)['complete'] else 1)


if __name__ == '__main__':
    main()
