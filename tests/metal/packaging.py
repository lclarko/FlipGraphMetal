"""Check a relocated shader package serially under the standard GPU guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
from metal_library import package, PRODUCTION
from guard import run
from application import write_json
from smoke import check_fixtures, dispatch_evidence
from verify import verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def library_evidence(log, variant, manifest):
    name = f'shaders/{variant}.metallib'
    expected = (name, manifest['files'][name])
    found = re.findall(r'^Metal library: (\S+) SHA256 ([0-9a-f]{64})$', log, re.MULTILINE)
    require(found == [expected], f'expected exactly one matching {variant} library record')
    return {'library': name, 'sha256': expected[1]}


def search_command(binary, exports):
    return [str(binary), '-n1', '3', '-n2', '3', '-n3', '3',
            '--schemes', '4', '--max-iterations', '10', '--rounds', '1',
            '--resize-probability', '0', '--block-size', '4', '--seed', '7',
            '--path', str(exports)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'build/metal')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--signed-fixtures', type=Path, required=True)
    parser.add_argument('--f2-fixtures', type=Path, required=True)
    parser.add_argument('--fixture-receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    binary, output, signed, f2, receipt = (path.resolve() for path in
        (args.binary_dir, args.output, args.signed_fixtures, args.f2_fixtures, args.fixture_receipt))
    output.mkdir(parents=True, exist_ok=False)
    summary = {'complete': False, 'steps': [], 'fixture_receipt_sha256': None}
    write_json(output / 'summary.json', summary)
    original_cwd = Path.cwd()
    try:
        receipt_bytes = receipt.read_bytes()
        fixtures = json.loads(receipt_bytes)
        check_fixtures(signed, f2, fixtures)
        summary['fixture_receipt_sha256'] = hashlib.sha256(receipt_bytes).hexdigest()
        relocated = output / 'relocated package'
        manifest = package(binary, relocated)
        summary['package_manifest'] = manifest
        cwd = output / 'unrelated working directory'
        cwd.mkdir()
        os.chdir(cwd)
        summary['working_directory'] = str(cwd)
        write_json(output / 'summary.json', summary)

        def guarded(name, command):
            step = {'name': name, 'command': command, 'complete': False}
            summary['steps'].append(step)
            write_json(output / 'summary.json', summary)
            result = run(command, output / (name + '-guard'))
            step['guard_complete'] = result['complete']
            write_json(output / 'summary.json', summary)
            log_path = output / (name + '-guard') / 'run.log'
            log = log_path.read_text() if log_path.exists() else ''
            return step, result, log

        smoke_output = output / 'smoke-exports'
        step, result, _ = guarded('relocated-smoke', [sys.executable, str(ROOT / 'tests/metal/smoke.py'),
            '--binary-dir', str(relocated), '--signed-fixtures', str(signed),
            '--f2-fixtures', str(f2), '--fixture-receipt', str(receipt),
            '--working-dir', str(cwd), '--output', str(smoke_output)])
        require(result['complete'], 'relocated smoke incomplete; retained guard evidence')
        smoke = json.loads((smoke_output / 'results.json').read_text())
        require(smoke['complete'] and len(smoke['runs']) == 8, 'relocated smoke did not complete all cases')
        step['runs'] = []
        for record in smoke['runs']:
            require(record['complete'], 'incomplete smoke case')
            program = Path(record['command'][0]).name
            log = (smoke_output / record['log']).read_text()
            step['runs'].append({'program': program, **library_evidence(log, PRODUCTION[program], manifest),
                                 **dispatch_evidence(log)})
        step['complete'] = True
        write_json(output / 'summary.json', summary)

        # Resolve shaders relative to the actual executable, not this symlink or cwd.
        link_dir = output / 'links elsewhere'
        link_dir.mkdir()
        link = link_dir / 'signed-search'
        link.symlink_to(relocated / 'flip_graph')
        step, result, log = guarded('symlink', search_command(link, output / 'symlink-exports'))
        require(result['complete'], 'symlink search incomplete; retained guard evidence')
        step.update(library_evidence(log, 'signed', manifest), **dispatch_evidence(log))
        step['verified'] = {path.name: verify(json.loads(path.read_text()))
                            for path in sorted((output / 'symlink-exports').glob('*.json'))}
        step['verified_export_count'] = len(step['verified'])
        step['complete'] = True
        write_json(output / 'summary.json', summary)

        mismatch = ('Error: Metal: packaged shader library does not match this executable; '
                    'rebuild or reinstall the complete package')
        for name in ('missing-library', 'corrupt-library', 'wrong-domain-library'):
            case = output / name
            (case / 'shaders').mkdir(parents=True)
            shutil.copy2(relocated / 'flip_graph', case / 'flip_graph')
            shader = case / 'shaders/signed.metallib'
            if name == 'corrupt-library':
                data = bytearray((relocated / 'shaders/signed.metallib').read_bytes())
                require(bool(data), 'empty production library')
                data[len(data) // 2] ^= 1
                shader.write_bytes(data)
            elif name == 'wrong-domain-library':
                shutil.copy2(relocated / 'shaders/f2.metallib', shader)
            diagnostic = (f'Error: Metal: cannot read packaged shader library {shader}; '
                          'keep the matching shaders directory with the executable') if name == 'missing-library' else mismatch
            step, result, log = guarded(name, search_command(case / 'flip_graph', case / 'exports'))
            # Expected failures keep their original INCOMPLETE guard record. A
            # separate verdict distinguishes rejection from timeout/unavailable GPU.
            checks = {
                'expected_exit': result.get('exit_code') == 1,
                'guard_reports_exit': result.get('error') == f"exit {result.get('exit_code')}",
                'guard_incomplete': result['complete'] is False,
                'cleanup_ok': 'cleanup_error' not in result,
                'exact_diagnostic': diagnostic in log.splitlines(),
                'no_dispatch': not re.search(r'^Metal dispatch ', log, re.MULTILINE),
                'apple_gpu': bool(re.search(r'^Metal device: Apple .+$', log, re.MULTILINE)),
            }
            verdict = {'expected_rejection': True, 'complete': all(checks.values()),
                       'diagnostic': diagnostic, 'checks': checks}
            write_json(case / 'verdict.json', verdict)
            step['expected_rejection_verdict'] = str(case / 'verdict.json')
            require(verdict['complete'], f'{name}: expected explicit shader rejection was not established')
            step['complete'] = True
            write_json(output / 'summary.json', summary)
        check_fixtures(signed, f2, fixtures)
        summary['complete'] = True
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        os.chdir(original_cwd)
        write_json(output / 'summary.json', summary)
    print('Packaging checks passed; retained evidence:', output)


if __name__ == '__main__':
    main()
