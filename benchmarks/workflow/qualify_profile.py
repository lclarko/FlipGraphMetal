"""Check diagnostic runtime observations against matched frozen production runs."""
import argparse
import json
import math
from pathlib import Path
import re
import time

import baseline as b


def run(baseline, profile, output):
    production=baseline/'source'
    diagnostic=profile/'source'
    prior=json.loads((baseline/'pilot-01/qualification.json').read_text())
    declaration=json.loads((profile/'profile.json').read_text())
    if not prior['complete'] or prior['baseline']['commit']!=b.BASELINE or declaration['baseline_commit']!=b.BASELINE:
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
             'production_evidence_sha256':b.digest(baseline/'pilot-01/qualification.json'),
             'profile_declaration_sha256':b.digest(profile/'profile.json'),
             'harness_sha256':b.digest(Path(__file__)), 'build_inventory':inventory,
             'qualification_scope':'byte-identical shaders, matched inputs, independently verified identical exports; not full-walk equivalence',
             'performance_acceptance':'NOT EVALUATED; diagnostic observations only','rows':[]}
    b.write_json(output/'qualification.json',summary)
    try:
        for row in b.ROWS:
            name,program,_,f2,kernel=row
            original=next(r for r in prior['attempts'] if r['workload']==name)
            source_attempt=baseline/'pilot-01'/f'00-{name}'
            input_path=source_attempt/'input.txt'
            if b.digest(input_path)!=original['input_sha256']:
                raise ValueError('baseline input changed')
            attempt=output/name;attempt.mkdir()
            argv=b.command(row,prior['protocol'],diagnostic/'build/metal',input_path,attempt/'exports')
            record={'workload':name,'complete':False,'argv':argv}
            summary['rows'].append(record);b.write_json(output/'qualification.json',summary)
            guard=b.guarded_run(['/usr/bin/time','-l','-p',*argv],attempt/'guard')
            if not guard['complete']:
                raise RuntimeError('incomplete profile attempt')
            log=(attempt/'guard/run.log').read_text()
            metrics=b.parse_metrics(log,kernel,prior['protocol']['rounds'][name],name=='mutation-reducer')
            observations=[]
            for line in log.splitlines():
                if not line.startswith('FGM_PROFILE_V1 '):continue
                values=dict(part.split('=',1) for part in line.split()[1:])
                expected={'kernel','setup_seconds','commit_wait_seconds','validation_report_seconds','tracked_shared_live_bytes','tracked_shared_peak_bytes'}
                if set(values)!=expected:raise ValueError('unexpected profiling fields')
                values={key:value if key=='kernel' else int(value) if key.endswith('_bytes') else float(value) for key,value in values.items()}
                if any(not math.isfinite(v) or v<0 for k,v in values.items() if k!='kernel'):
                    raise ValueError('invalid profile value')
                if values['tracked_shared_live_bytes']>values['tracked_shared_peak_bytes']:
                    raise ValueError('invalid tracked memory')
                observations.append(values)
            if [r['kernel'] for r in observations]!=[r[0] for r in metrics['dispatches']]:
                raise ValueError('profile dispatch observations missing or reordered')
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
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.baseline.resolve(),a.profile.resolve(),a.output.resolve())
