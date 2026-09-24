"""Describe retained baseline variability without granting regression allowance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict


def summarize(path):
    path=Path(path)
    raw=path.read_bytes()
    data=json.loads(raw)
    if not data.get('complete') or data.get('error'):
        raise ValueError('incomplete qualification evidence')
    rows=defaultdict(list)
    for entry in data['attempts']:
        if not entry.get('complete') or not entry.get('guard_complete'):
            raise ValueError('incomplete retained attempt')
        rows[entry['workload']].append(entry)
    expected=set(data['protocol']['mandatory_rows'])
    if set(rows)!=expected:
        raise ValueError('missing mandatory workload')
    result={'schema':'fgm-baseline-variability-v1',
            'evidence_sha256':hashlib.sha256(raw).hexdigest(),
            'protocol_sha256':data['protocol_sha256'],
            'performance_verdict':'NOT EVALUATED',
            'practical_regression_allowance':'unapproved; never inferred from repeatability',
            'workloads':{}}
    for name,entries in sorted(rows.items()):
        entries.sort(key=lambda r:r['repeat'])
        if len(entries)<4 or len(entries)%2 or [r['repeat'] for r in entries]!=list(range(len(entries))):
            raise ValueError('need complete sequential baseline pairs')
        metrics={}
        for key in ('gpu_work_seconds','gpu_all_seconds','process_seconds','workflow_seconds',
                    'verification_seconds','peak_process_rss_bytes','peak_system_wired_bytes'):
            values=[r[key] for r in entries]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
                raise ValueError('invalid baseline metric')
            mean=statistics.mean(values)
            metrics[key]={'median':statistics.median(values),'mean':mean,
                          'min':min(values),'max':max(values),'stdev':statistics.stdev(values),
                          'coefficient_of_variation':statistics.stdev(values)/mean if mean else None}
            if min(values)>0:
                pairs=[math.log(values[i+1]/values[i]) for i in range(0,len(values),2)]
                metrics[key]['paired_log_differences']=pairs
                metrics[key]['paired_log_stdev']=statistics.stdev(pairs)
        exports=[[(e['sha256'],e['domain'],e['rank']) for e in row['exports']] for row in entries]
        result['workloads'][name]={'repeats':len(entries),'metrics':metrics,
                                  'identical_verified_exports':all(x==exports[0] for x in exports),
                                  'actual_mutation_verified':all(r['applied_mutation_evidence']=='changed verified circuit factors' for r in entries) if name=='mutation-reducer' else None}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=summarize(args.evidence)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    for name,entry in result['workloads'].items():
        timing=entry['metrics']['process_seconds']
        print(f"{name}: {timing['median']:.3f}s median; {timing['min']:.3f}..{timing['max']:.3f}s range")
    print('Baseline repeatability only; no practical allowance or candidate verdict established.')


if __name__=='__main__':main()
