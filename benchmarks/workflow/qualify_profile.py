"""Check diagnostic runtime observations against matched frozen production runs."""
import argparse
import json
import math
from pathlib import Path
import re

import baseline as b


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


def run(baseline, production_run, profile, output):
    production=baseline/'source'
    diagnostic=profile/'source'
    production_receipt = production_run / 'qualification.json'
    prior=json.loads(production_receipt.read_text())
    declaration=json.loads((profile/'profile.json').read_text())
    if (prior.get('schema') != 'fgm-baseline-qualification-v1'
            or prior.get('complete') is not True
            or prior['baseline']['commit'] != b.BASELINE
            or declaration['baseline_commit'] != b.BASELINE):
        raise ValueError('incomplete or wrong baseline/profile')
    b.assert_source_snapshot(production,prior['baseline'])
    b.assert_inventory(production,prior['build_inventory'])
    for path,sha in declaration['diagnostic_files'].items():
        if b.digest(diagnostic/path)!=sha:
            raise ValueError('diagnostic source changed')
    inventory=b.inventory(diagnostic)
    for variant in ('signed','f2'):
        key=f'build/metal/shaders/{variant}.metallib'
        if inventory[key]!=prior['build_inventory'][key]:
            raise ValueError('profiling changed compiled shader bytes')
    output.mkdir(parents=True,exist_ok=False)
    summary={'schema':'fgm-profile-qualification-v1','complete':False,
             'production_evidence_sha256':b.digest(production_receipt),
             'profile_declaration_sha256':b.digest(profile/'profile.json'),
             'harness_sha256':b.digest(Path(__file__)), 'build_inventory':inventory,
             'qualification_scope':'byte-identical shaders, matched inputs, independently verified identical exports; not full-walk equivalence',
             'performance_acceptance':'NOT EVALUATED; diagnostic observations only','rows':[]}
    b.write_json(output/'qualification.json',summary)
    try:
        for row in b.ROWS:
            name,program,_,f2,kernel=row
            original, source_attempt = production_attempt(prior, production_run, name)
            input_path=source_attempt/'input.txt'
            if b.digest(production/'tests/metal/fixtures'/row[2]) != original['fixture_sha256']:
                raise ValueError('baseline fixture changed')
            if original['expected_kernel'] != kernel:
                raise ValueError('baseline kernel does not match workload')
            if b.digest(input_path)!=original['input_sha256']:
                raise ValueError('baseline input changed')
            attempt=output/name;attempt.mkdir()
            argv=b.command(row,prior['protocol'],diagnostic/'build/metal',input_path,attempt/'exports')
            record={'workload':name,'complete':False,'argv':argv,
                    'production_repeat': original['repeat'],
                    'input_sha256': original['input_sha256'],
                    'fixture_sha256': original['fixture_sha256']}
            summary['rows'].append(record);b.write_json(output/'qualification.json',summary)
            guard=b.guarded_run(['/usr/bin/time','-l','-p',*argv],attempt/'guard')
            if not guard['complete']:
                raise RuntimeError('incomplete profile attempt')
            log=(attempt/'guard/run.log').read_text()
            metrics=b.parse_metrics(log,kernel,prior['protocol']['rounds'][name],name=='mutation-reducer')
            observations = parse_profile_observations(log, [r[0] for r in metrics['dispatches']])
            exports=[]
            reference=json.loads((source_attempt/'effective-input.json').read_text())
            for path in sorted((attempt/'exports').rglob('*.json')):
                result=b.verify(json.loads(path.read_text()),reference if name=='fixed-reducer' else None)
                exports.append({'sha256':b.digest(path),**result})
            if sorted(r['sha256'] for r in exports)!=sorted(r['sha256'] for r in original['exports']):
                raise ValueError('diagnostic export bytes differ from production')
            record.update(complete=True,metrics=metrics,profile=observations,exports=exports)
            b.write_json(output/'qualification.json',summary)
            print(name,'matched shaders and verified exports',flush=True)
        b.assert_inventory(production,prior['build_inventory']);b.assert_inventory(diagnostic,inventory)
        summary['complete']=True
    except Exception as error:
        summary['error']=str(error);raise
    finally:
        b.write_json(output/'qualification.json',summary)
        (output/'qualification.sha256').write_text(b.digest(output/'qualification.json')+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--production-run',type=Path,required=True,
                   help='Completed baseline run containing qualification.json')
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.baseline.resolve(),a.production_run.resolve(),a.profile.resolve(),a.output.resolve())
