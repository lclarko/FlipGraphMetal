import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time


def stop(processes):
    def signal_group(process, value):
        process.poll()
        try:
            os.killpg(process.pid, value)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            if process.poll() is None:
                raise
            try:
                os.killpg(process.pid, value)
                return True
            except ProcessLookupError:
                return False

    errors = []

    def attempt(process, value):
        try:
            return signal_group(process, value)
        except OSError as error:
            errors.append(f'process group {process.pid}: {error}')
            return False

    groups = [process for process in processes if process]
    for process in groups:
        attempt(process, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(attempt(process, 0) for process in groups):
        for process in groups:
            process.poll()
        time.sleep(0.02)
    for process in groups:
        attempt(process, signal.SIGKILL)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError('; '.join(errors))


def main():
    parser = argparse.ArgumentParser(description='Capture prepared Metal walks before exporting schemes')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schemes', type=int, choices=[512, 2048], required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--kernel', choices=['randomWalkKernel', 'randomWalkCompactKernel'], default='randomWalkKernel')
    args = parser.parse_args()
    if args.seed < 1:
        parser.error('seed must be positive')
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(binary), str(args.schemes), '8', str(args.seed), str(output / 'schemes'), 'capture']
    events = queue.Queue()
    records = []
    recorder = None
    recorder_log = None
    reason = None
    ready = False
    done = False
    released = False
    environment = {key: os.environ[key] for key in ['PATH', 'HOME', 'TMPDIR', 'DEVELOPER_DIR', 'LANG', 'LC_ALL', 'USER', 'LOGNAME'] if key in os.environ}
    environment['OMP_NUM_THREADS'] = '8'
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    cleanup_error = None
    start = time.monotonic()
    last_sample = -1
    with (output / 'target.log').open('w') as log:
        target = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                                  text=True, start_new_session=True, env=environment)

        def read_output():
            for line in target.stdout:
                log.write(line)
                log.flush()
                events.put(line.rstrip('\n'))

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while target.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed - last_sample >= 1:
                    stat = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=max(0.01, min(2, 45-elapsed)), check=True).stdout
                    page = int(re.search(r'page size of (\d+) bytes', stat)[1])
                    wired = int(re.search(r'Pages wired down:\s+(\d+)', stat)[1]) * page
                    records.append(dict(seconds=elapsed, wired_bytes=wired))
                    last_sample = elapsed
                    if wired > 3 * 1024**3:
                        reason = 'wired memory exceeded 3 GiB'
                if time.monotonic() - start > 45:
                    reason = 'time limit'
                if reason:
                    break
                try:
                    line = events.get(timeout=0.05)
                except queue.Empty:
                    line = ''
                if line.startswith('CAPTURE_READY '):
                    if ready:
                        raise RuntimeError('duplicate capture readiness')
                    _, pid, notification = line.split()
                    if int(pid) != target.pid:
                        raise RuntimeError('capture process identity mismatch')
                    ready = True
                    capture_command = ['xcrun', 'xctrace', 'record', '--template', 'Metal System Trace',
                                       '--instrument', 'Metal GPU Counters', '--time-limit', '8s',
                                       '--notify-tracing-started', notification, '--attach', pid,
                                       '--output', str(output / 'capture.trace')]
                    recorder_log = (output / 'record.log').open('w')
                    recorder = subprocess.Popen(capture_command, stdout=recorder_log, stderr=subprocess.STDOUT,
                                                start_new_session=True, env=environment)
                    (output / 'capture-command.json').write_text(json.dumps(capture_command, indent=2) + '\n')
                if line == 'CAPTURE_DONE':
                    if not ready or done or recorder.poll() is not None:
                        raise RuntimeError('capture ended before target completion')
                    done = True
                    recorder.send_signal(signal.SIGINT)
                if recorder and recorder.poll() is not None and not released:
                    if recorder.returncode or not done:
                        raise RuntimeError('recording failed or ended before the prepared walks')
                    target.stdin.write('export\n')
                    target.stdin.flush()
                    target.stdin.close()
                    released = True
            if reason:
                raise RuntimeError(reason)
            target.wait()
            reader.join(timeout=2)
            if target.returncode or not released or reader.is_alive():
                raise RuntimeError('target did not complete capture and export')
        except Exception as error:
            reason = str(error)
            raise
        finally:
            try:
                stop([target, recorder])
            except Exception as error:
                cleanup_error = str(error)
                reason = reason or ('capture cleanup failed: ' + cleanup_error)
            reader.join(timeout=2)
            if recorder_log:
                recorder_log.close()
            result = dict(argv=command, kernel=args.kernel, target_pid=target.pid, verification='not run', binary_sha256=binary_sha256, cleanup_error=cleanup_error,
                          exit_code=target.returncode, recorder_exit=recorder.returncode if recorder else None,
                          reason=reason, wall_seconds=time.monotonic()-start, memory=records,
                          capture_ready=ready, capture_done=done, export_released=released)
            (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if cleanup_error:
        raise RuntimeError('capture cleanup failed: ' + cleanup_error)
    try:
        log = (output / 'target.log').read_text()
        samples = re.findall(r'^REPEAT (\d+) ' + re.escape(args.kernel) + r' METAL_WALL [\d.e+-]+ MATCH$', log, re.M)
        if samples != ['0', '1', '2']:
            raise RuntimeError('expected three CPU-matched captured walks')
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests/metal'))
        from verify import verify
        files = list((output / 'schemes').glob('*.json'))
        if len(files) != args.schemes:
            raise RuntimeError('incomplete scheme export')
        for path in files:
            verify(json.loads(path.read_text()))
        result['verified_exports'] = len(files)
        result['verification'] = 'passed'
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    except Exception as error:
        result['verification'] = 'failed'
        result['reason'] = str(error)
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        raise
    print('Capture complete:', output, 'verified exports:', len(files))


if __name__ == '__main__':
    main()
