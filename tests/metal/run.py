"""Run Metal checks serially, retaining a new evidence directory for every attempt."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from application import digest, write_json
from guard import run
from screen import check_log
from verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    binary = ROOT / 'build/metal'
    attempt = Path(tempfile.mkdtemp(prefix='check-', dir=binary))
    summary = {'complete': False, 'steps': [], 'attempt': str(attempt)}
    write_json(attempt / 'checks.json', summary)
    print('Evidence:', attempt, flush=True)

    def guarded(name, command):
        result = run(command, attempt / (name + '-guard'))
        summary['steps'].append({'name': name, 'complete': result['complete']})
        write_json(attempt / 'checks.json', summary)
        if not result['complete']:
            raise RuntimeError(name + ' incomplete; retained evidence: ' + str(attempt))
        return (attempt / (name + '-guard/run.log')).read_text()

    try:
        probe = guarded('probe', [str(binary / 'probe')])
        if not re.search(r'^Metal device: Apple .+', probe, re.M) or 'PASS: four integer arithmetic results from a completed Metal dispatch' not in probe:
            raise RuntimeError('probe did not confirm Apple GPU execution and arithmetic')
        if not args.probe:
            # Lifecycle tests supervise their own subprocesses and do not launch GPU kernels.
            command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/metal', '-p', 'test_*.py', '-v']
            summary['host_command'] = command
            write_json(attempt / 'checks.json', summary)
            with (attempt / 'host-tests.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary['steps'].append({'name': 'host-regressions', 'complete': True})
            for name, program in [('signed', 'correctness'), ('f2', 'f2_correctness')]:
                exports = attempt / (name + '-exports')
                log = guarded(name, [str(binary / program), '--output-dir', str(exports)])
                check_log(log)
                if not re.search(r'^Metal device: Apple .+', log, re.M):
                    raise RuntimeError('missing Apple GPU device evidence')
                files = sorted(exports.glob('*.json'))
                if not files:
                    raise RuntimeError('correctness test exported no tensors')
                verified = {p.name: {'sha256': digest(p), **verify(json.loads(p.read_text()))} for p in files}
                write_json(attempt / (name + '-verification.json'), verified)
            if args.smoke:
                from smoke import fixture_hashes
                receipt = attempt / 'fixture-receipt.json'
                write_json(receipt, fixture_hashes(attempt / 'signed-exports', attempt / 'f2-exports'))
                guarded('smoke', [sys.executable, str(ROOT / 'tests/metal/smoke.py'),
                                 '--binary-dir', str(binary), '--signed-fixtures', str(attempt / 'signed-exports'),
                                 '--f2-fixtures', str(attempt / 'f2-exports'), '--fixture-receipt', str(receipt),
                                 '--output', str(attempt / 'smoke-exports')])
            summary['steps'].append({'name': 'independent-verification', 'complete': True})
        summary['complete'] = True
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        write_json(attempt / 'checks.json', summary)
    print('Checks passed; evidence:', attempt)


if __name__ == '__main__':
    main()
