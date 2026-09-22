"""Run one build or regression command with the Metal process and memory limits."""

import argparse
import os
from pathlib import Path
import subprocess
import time

from application import LIMIT, artifacts, digest, terminate, wired_memory, write_json


def run(argv, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    record = dict(argv=argv, complete=False, time_limit=45, wired_limit=LIMIT, memory=[])
    write_json(output / 'result.json', record)
    process = None
    try:
        initial = wired_memory()
        record['memory'].append(dict(seconds=0, wired_bytes=initial))
        if initial > LIMIT:
            raise RuntimeError('wired memory exceeded 3 GiB before launch')
        with (output / 'run.log').open('w') as log:
            process_start = time.monotonic()
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={key: os.environ[key] for key in
                ['PATH', 'HOME', 'TMPDIR', 'DEVELOPER_DIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME']
                if key in os.environ})
            while process.poll() is None:
                remaining = 45 - (time.monotonic() - start)
                if remaining <= 0:
                    raise RuntimeError('time limit')
                value = wired_memory(timeout=min(2, remaining))
                record['memory'].append(dict(seconds=time.monotonic() - start, wired_bytes=value))
                if value > LIMIT:
                    raise RuntimeError('wired memory exceeded 3 GiB')
                remaining = 45 - (time.monotonic() - start)
                if remaining <= 0:
                    raise RuntimeError('time limit')
                try:
                    process.wait(timeout=min(.25, remaining))
                except subprocess.TimeoutExpired:
                    pass
            record['observed_process_seconds'] = time.monotonic() - process_start
            if process.returncode:
                raise RuntimeError(f'exit {process.returncode}')
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
        record['artifacts'] = artifacts(output)
        write_json(output / 'result.json', record)
        (output / 'result.sha256').write_text(digest(output / 'result.json') + '\n')
    print(output, 'PASS' if record['complete'] else 'INCOMPLETE', record.get('error', ''), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv:
        parser.error('a command is required')
    raise SystemExit(0 if run(argv, args.output)['complete'] else 1)


if __name__ == '__main__':
    main()
