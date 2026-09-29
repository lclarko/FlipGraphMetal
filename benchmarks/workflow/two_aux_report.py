"""Host-only descriptive report for the frozen M4 two-auxiliary schedule."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

MANIFEST_SHA = '8fe9aa117a4679de845a91ad4f995a7e2bec44fac3dba36f6aa30dd76d8158fa'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(root, relative, expected):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('unsafe evidence path')
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('linked evidence path')
    if not path.is_file() or not re.fullmatch('[0-9a-f]{64}', str(expected)) or sha(path) != expected:
        raise ValueError('missing or changed evidence: ' + relative)
    return path


def evidence(root, row):
    if row.get('status') != 'verified':
        return None
    doc = json.loads(checked(root, row['evidence_path'], row['verification_sha256']).read_text())
    if doc.get('schema') == 'fgm-two-aux-m4-fixed-verification-v1':
        for stem in ('receipt', 'artifact'):
            checked(root, row[stem + '_path'], row[stem + '_sha256'])
        if doc['receipt_sha256'] != row['receipt_sha256'] or doc['artifact_sha256'] != row['artifact_sha256']:
            raise ValueError('fixed verification binding mismatch')
        verified = doc['verified']
        if verified['additions'] != row['additions'] or verified['additions_by_stage'] != row['additions_by_stage']:
            raise ValueError('fixed verified cost mismatch')
        receipt = json.loads((root / row['receipt_path']).read_text())
        if receipt.get('results', [None])[0] != doc['result']:
            raise ValueError('fixed receipt result differs from verification')
        return doc
    if doc.get('schema') == 'fgm3-chunk-verification-v1':
        chunk = row['chunks'][0]
        for stem in ('receipt', 'evaluation_export', 'observation_export'):
            checked(root, chunk[stem + '_path'] if stem == 'receipt' else chunk[stem],
                    chunk[stem + '_sha256'])
            if doc[stem + '_sha256'] != chunk[stem + '_sha256']:
                raise ValueError('native verification binding mismatch')
        if doc['best_additions'] != row['best_additions']:
            raise ValueError('native verified cost mismatch')
        new = [{k: v for k, v in item.items() if k != 'verified_seconds'}
               for item in chunk['new_evaluations']]
        if new != doc['new_evaluations']:
            raise ValueError('native new evaluations differ from exact verification')
        return doc
    raise ValueError('unknown verification schema')


def details(root, row, doc):
    """Read report and guard fields only after their source rows are verified."""
    stages = []
    if doc['schema'] == 'fgm-two-aux-m4-fixed-verification-v1':
        sources = [doc['result'].get('two_auxiliary')]
        guard_dirs = [(row['id'] + '/native-guard', row.get('native_guard')),
                      (row['id'] + '/verifier-guard', row.get('verifier_guard'))]
    else:
        chunk = row['chunks'][0]
        export = checked(root, chunk['evaluation_export'], chunk['evaluation_export_sha256'])
        new_ids = {entry['scheme_id'] for entry in chunk['new_evaluations']}
        sources = [entry.get('evaluation', {}).get('two_auxiliary') for line in export.read_text().splitlines()
                   if (entry := json.loads(line)).get('schema') == 'fgm-journal-evaluation-v1'
                   and entry.get('evaluation', {}).get('scheme_id') in new_ids]
        base = str(Path(row['evidence_path']).parent)
        guard_dirs = [(base + '/' + name.replace('_', '-') + '-guard', chunk.get(name + '_guard'))
                      for name in ('native', 'evaluation_export', 'observation_export')]
        guard_dirs.append((row['id'] + '/native-verifier-guard', row.get('native-verifier_guard')))
    for source in sources:
        if isinstance(source, dict):
            stages.append(source)
    memory = []
    for directory, guard in guard_dirs:
        if not isinstance(guard, dict) or not guard.get('result_sha256'):
            continue
        sidecar = json.loads(checked(root, directory + '/result.json', guard['result_sha256']).read_text())
        log = checked(root, directory + '/run.log', guard['log_sha256']).read_text()
        rss = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', log, re.M)
        memory.append(dict(scope=directory.rsplit('/', 1)[-1],
                           process_rss_bytes=max(map(int, rss)) if rss else None,
                           system_wired_bytes=max((s['wired_bytes'] for s in sidecar.get('memory', [])), default=None),
                           wall_seconds=sidecar.get('wall_seconds')))
    return stages, memory


def receipt_metrics(root, row, doc):
    """Keep receipt accounting and clocks distinct from observed guard memory."""
    native = doc['schema'] == 'fgm3-chunk-verification-v1'
    chunk = row['chunks'][0] if native else row
    receipt = json.loads(checked(root, chunk['receipt_path'], chunk['receipt_sha256']).read_text())
    accounted = {key: receipt.get(key) for key in
                 ('planned_buffer_bytes', 'reserved_host_bytes', 'admission_content_bytes',
                  'memory_accounting')}
    result = receipt.get('best_evaluation', {}) if native else doc['result']
    gpu = chunk.get('gpu', row.get('gpu', {}))
    timing = dict(receipt_microseconds={k: v for k, v in receipt.items() if k.endswith('_microseconds')},
                  result_phase_microseconds=result.get('phase_microseconds'),
                  result_dispatch_microseconds=result.get('dispatch_microseconds'),
                  result_verification_microseconds=result.get('verification_microseconds'),
                  gpu_seconds=gpu.get('gpu_all_seconds'),
                  gpu_dispatch_count=len(gpu.get('dispatches', [])),
                  independent_verifier_wall_seconds=row.get(
                      'native-verifier_guard' if native else 'verifier_guard', {}).get('wall_seconds'),
                  elapsed_seconds=row.get('elapsed_seconds'),
                  headroom_wait_seconds=row.get('headroom_wait_seconds', 0))
    timing['result_clock_scope'] = ('selected best evaluation, possibly from an earlier invocation'
                                    if native else 'this fixed-factor evaluation')
    timing['elapsed_scope'] = 'complete native/export/independent-verification unit including headroom waits'
    if native:
        timing['native_host_phases_microseconds'] = chunk.get('host_phases_microseconds')
        timing['new_evaluation_phase_microseconds'] = [e.get('phase_microseconds')
                                                        for e in chunk['new_evaluations']]
    return accounted, timing


def units(attempt, kind):
    return attempt['cells'] if kind in ('fixed', 'A', 'B') else attempt['units'] if kind == 'native' else attempt['arms']


def plan(manifest):
    for kind, field in (('fixed', 'fixed_factor_schedule'), ('native', 'native_schedule')):
        for attempt in manifest['qualification'][field]:
            yield 'qualification', kind, attempt
    for kind in ('A', 'B', 'C-S', 'C-N'):
        for attempt in manifest['experiments'][kind]['attempts']:
            yield 'measurement', kind, attempt


def quantiles(values):
    if not values:
        return None
    s = sorted(values)
    return dict(n=len(s), median=statistics.median(s), p95=s[max(0, (95 * len(s) + 99) // 100 - 1)], max=s[-1])


def stage_outcomes(report):
    counts = {'eligible': 0, 'escalated': 0, 'improved': 0}
    for stage in report.get('stages', {}).values():
        eligible = (stage['stop_reason'] not in ('smaller-family-succeeded', 'proved-no-improvement')
                    and stage['incumbent_cost'] > stage['improvement_floor'])
        counts['eligible'] += eligible
        counts['escalated'] += stage['raw_scanned'] > 0
        counts['improved'] += stage['final_cost'] < stage['incumbent_cost']
    return counts


def native_feedback(records, factor_ids, cutoff):
    verified = [r for r in records if r.get('status') == 'verified']
    initial = next((r for r in verified if r.get('chunks', [{}])[0].get('index') == 0), None)
    initial_count = len(initial['chunks'][0]['new_evaluations']) if initial else 0
    total = sum(len(r['chunks'][0]['new_evaluations']) for r in verified)
    installs = [i for r in verified for i in r['chunks'][0].get('new_installations', [])]
    return dict(scope='all verified units including late work',
                initial_candidates=min(initial_count, len(factor_ids)),
                descendant_candidates=max(0, total - len(factor_ids)),
                late_evaluations=sum(len(r['chunks'][0]['new_evaluations']) for r in verified
                                     if r.get('durable_seconds', float('inf')) > cutoff),
                feedback_installations=sum(bool(i.get('installed')) for i in installs),
                feedback_records=len(installs))


def report(manifest_path, root):
    manifest_path, root = Path(manifest_path), Path(root)
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('M4 manifest hash differs from approved frozen manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'fgm-two-aux-experiment-manifest-v1':
        raise ValueError('M4 manifest schema mismatch')
    cohorts = {f['id']: f['cohort'] for f in manifest['panel']['factors']}
    factor_identity = {f['id']: f.get('effective_factors_id') for f in manifest['panel']['factors']}
    summary = dict(schema='fgm-two-aux-m4-summary-v1', manifest_sha256=MANIFEST_SHA,
                   result_root=str(root), attempts=[], fixed={}, search={},
                   memory={'observed': [], 'planned': manifest['resources']},
                   stage_outcome_definitions={
                       'eligible': 'verified enabled stage beyond the smaller family, with incumbent_cost above improvement_floor',
                       'escalated': 'verified enabled stage with raw_scanned above zero',
                       'improved': 'verified enabled stage with final_cost below incumbent_cost'},
                   findings=[], uncertainty='Paired uncertainty interval unavailable: this report does not compute one.',
                   performance_verdict='descriptive only; no approved numeric regression margin')
    fixed = {'A': {}, 'B': {}}
    search = {'C-S': [], 'C-N': []}
    authority = None
    for phase, kind, attempt in plan(manifest):
        directory = root / phase / kind / attempt['id']
        record = dict(phase=phase, experiment=kind, id=attempt['id'], status='missing',
                      planned=len(units(attempt, kind)), verified=0, rows=[])
        path = directory / 'result.json'
        if path.exists() or (directory / 'result.sha256').exists():
            try:
                result = json.loads(checked(directory, 'result.json',
                                            (directory / 'result.sha256').read_text().strip()).read_text())
                if (result.get('schema'), result.get('phase'), result.get('experiment'),
                    result.get('attempt_id'), result.get('manifest_sha256')) != (
                        'fgm-two-aux-m4-attempt-v1', phase, kind, attempt['id'], MANIFEST_SHA):
                    raise ValueError('attempt identity or manifest binding mismatch')
                binding = (result.get('review_sha256'), result.get('authorization_sha256'))
                if not all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in binding):
                    raise ValueError('missing review or authorization binding')
                if authority is None:
                    authority = binding
                elif binding != authority:
                    raise ValueError('mixed review or authorization bindings')
                attempt_file = json.loads(checked(directory, 'attempt-manifest.json',
                                                  result['attempt_manifest_sha256']).read_text())
                if (attempt_file.get('frozen_manifest_sha256') != MANIFEST_SHA or
                    attempt_file.get('phase') != phase or attempt_file.get('experiment') != kind or
                    attempt_file.get('attempt') != attempt):
                    raise ValueError('attempt manifest binding mismatch')
                rows = result['rows']
                if len(rows) != record['planned'] or [r['id'] for r in rows] != [u['id'] for u in units(attempt, kind)]:
                    raise ValueError('scheduled row identity mismatch')
                record.update(status=result['status'], result_sha256=sha(path),
                              headroom_wait_seconds=0)
                for unit, row in zip(units(attempt, kind), rows):
                    state = row.get('status', 'unrun')
                    item = dict(id=unit['id'], status=state)
                    source = row
                    if state == 'verified':
                        try:
                            doc = evidence(directory, row)
                            if kind in ('fixed', 'A', 'B') and factor_identity[unit['factor_id']] is not None:
                                if doc['verified'].get('factors_id') != factor_identity[unit['factor_id']]:
                                    raise ValueError('fixed factor identity differs from frozen panel')
                            stages, memory = details(directory, row, doc)
                            accounted, timing = receipt_metrics(directory, row, doc)
                        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                            item.update(status='evidence-invalid', error=str(exc))
                        else:
                            record['verified'] += 1
                            item['additions'] = row.get('additions', row.get('best_additions'))
                            item['two_auxiliary_stages'] = stages
                            item['receipt_accounted_memory'] = accounted
                            item['timing'] = timing
                            summary['memory']['observed'].extend(dict(phase=phase, experiment=kind,
                                                                 attempt=attempt['id'], row=row['id'], **m)
                                                                 for m in memory)
                            if kind in fixed:
                                key = (unit['factor_id'], unit['trial_seed'], unit['repeat'])
                                fixed[kind].setdefault(key, []).append((unit, row, directory, doc))
                    elif kind in search:
                        for child in row.get('blocks', row.get('chunks', [])):
                            if child.get('status') == 'verified':
                                try:
                                    doc = evidence(directory, child)
                                    if kind == 'C-S' and factor_identity[unit['factor_ids'][0]] is not None:
                                        if doc['verified'].get('factors_id') != factor_identity[unit['factor_ids'][0]]:
                                            raise ValueError('search factor identity differs from frozen panel')
                                    stages, memory = details(directory, child, doc)
                                    accounted, timing = receipt_metrics(directory, child, doc)
                                    summary['memory']['observed'].extend(dict(phase=phase, experiment=kind,
                                                                         attempt=attempt['id'], row=child['id'], **m)
                                                                         for m in memory)
                                    item.setdefault('verified_children', []).append(dict(
                                        id=child['id'], durable_seconds=child.get('durable_seconds'),
                                        late_for_endpoints=[p for p in unit['endpoints'] if
                                                            child.get('durable_seconds', float('inf')) > row['started_seconds'] + p],
                                        two_auxiliary_stages=stages,
                                        receipt_accounted_memory=accounted, timing=timing))
                                except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                                    item.update(status='evidence-invalid', error=str(exc))
                                    break
                        if item['status'] == 'complete':
                            search[kind].append((unit, row))
                    waits = (source.get('blocks', source.get('chunks', []))
                             if kind in ('C-S', 'C-N') else [source])
                    record['headroom_wait_seconds'] += sum(
                        child.get('headroom_wait_seconds', 0) for child in waits
                        if isinstance(child.get('headroom_wait_seconds', 0), (int, float)))
                    record['rows'].append(item)
                if result['status'] == 'complete' and any(x['status'] in ('evidence-invalid', 'failed', 'unrun') for x in record['rows']):
                    record['status'] = 'evidence-invalid'
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                record.update(status='result-invalid', error=str(exc))
        summary['attempts'].append(record)
    for kind, groups in fixed.items():
        by_cohort = {name: dict(planned=0, paired=0, improvements={}, ties={}, latency_seconds={},
                                paired_latency_delta_seconds=[], disabled_semantic_mismatches=0,
                                stage_outcomes={'eligible': 0, 'escalated': 0, 'improved': 0})
                     for name in manifest['panel']['cohorts']}
        for attempt in manifest['experiments'][kind]['attempts']:
            by_cohort[cohorts[attempt['factor_id']]]['planned'] += len(attempt['cells']) // 2
        for key, values in groups.items():
            cohort = by_cohort[cohorts[key[0]]]
            if len(values) != 2 or len({v[0]['treatment']['id'] for v in values}) != 2:
                continue
            values.sort(key=lambda value: value[0]['treatment']['id'])
            left, right = values
            if kind == 'A':
                left, right = sorted(values, key=lambda v: v[0]['treatment']['id'] != 'baseline-omitted')
            else:
                left, right = sorted(values, key=lambda v: v[0]['treatment']['id'] != 'candidate-omitted')
            a, b = left[1], right[1]
            cohort['paired'] += 1
            for stage in ('u', 'v', 'w', 'total'):
                av = a['additions'] if stage == 'total' else a['additions_by_stage'][stage]
                bv = b['additions'] if stage == 'total' else b['additions_by_stage'][stage]
                delta = av - bv
                cohort['improvements'].setdefault(stage, []).append(delta)
                cohort['ties'][stage] = cohort['ties'].get(stage, 0) + (delta == 0)
            for value in (left, right):
                cohort['latency_seconds'].setdefault(value[0]['treatment']['id'], []).append(value[1]['elapsed_seconds'])
            cohort['paired_latency_delta_seconds'].append(a['elapsed_seconds'] - b['elapsed_seconds'])
            if kind == 'A':
                circuits = [json.loads((value[2] / value[1]['artifact_path']).read_text()) for value in (left, right)]
                factors = [value[3]['verified'].get('factors_id') for value in (left, right)]
                if (a['additions'] != b['additions'] or a['additions_by_stage'] != b['additions_by_stage'] or
                    circuits[0] != circuits[1] or factors[0] != factors[1]):
                    cohort['disabled_semantic_mismatches'] += 1
            else:
                report = right[3]['result'].get('two_auxiliary', {})
                for key, count in stage_outcomes(report).items():
                    cohort['stage_outcomes'][key] += count
        for cohort in by_cohort.values():
            cohort['latency_seconds'] = {k: quantiles(v) for k, v in cohort['latency_seconds'].items()}
            cohort['paired_latency_delta_seconds'] = quantiles(cohort['paired_latency_delta_seconds'])
            cohort['improvements'] = {k: dict(samples=v, summary=quantiles(v)) for k, v in cohort['improvements'].items()}
        summary['fixed'][kind] = by_cohort
        if kind == 'A':
            for name, cohort in by_cohort.items():
                if cohort['disabled_semantic_mismatches']:
                    summary['findings'].append(f"A {name}: {cohort['disabled_semantic_mismatches']} disabled semantic mismatches")
    for kind, arms in search.items():
        endpoints = (10, 20) if kind == 'C-S' else (30, 60)
        output = []
        for unit, arm in arms:
            records = arm.get('blocks', arm.get('chunks', []))
            points = {}
            for point in endpoints:
                timely = [r for r in records if r.get('status') == 'verified' and
                          r.get('durable_seconds', float('inf')) <= arm['started_seconds'] + point]
                costs = [r.get('additions', r.get('best_additions')) for r in timely]
                ids = ({e['scheme_id'] for r in timely for chunk in r.get('chunks', [])
                        for e in chunk.get('new_evaluations', [])} if kind == 'C-N' else None)
                count = len(ids) if ids is not None else len(timely)
                points[str(point)] = dict(best_additions=min(costs) if costs else None,
                                          verified_evaluations=count,
                                          target_54_attained=bool(costs and min(costs) <= 54),
                                          throughput_per_second=count / point)
            native_detail = None
            if kind == 'C-N':
                native_detail = native_feedback(records, unit['factor_ids'],
                                                arm['started_seconds'] + unit['seconds'])
            output.append(dict(id=unit['id'], factor_ids=unit['factor_ids'],
                               treatment=unit['treatment']['id'], endpoints=points,
                               native_init_descendant_feedback=native_detail))
        summary['search'][kind] = dict(planned_arms=sum(len(a['arms']) for a in manifest['experiments'][kind]['attempts']),
                                       complete_evidence_arms=len(output), arms=output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    value = report(args.manifest, args.root)
    destination = args.output / 'summary.json'
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    (args.output / 'summary.sha256').write_text(sha(destination) + '\n')
    lines = ['# M4 two-auxiliary descriptive report', '', f'Manifest SHA-256: `{MANIFEST_SHA}`',
             f'Summary SHA-256: `{sha(destination)}`', '',
             'No numeric performance regression margin is approved. This report does not compute a paired uncertainty interval.', '',
             '| Phase | Experiment | Planned attempts | Complete | Missing | Invalid |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for phase, kind in (('qualification', 'fixed'), ('qualification', 'native'),
                        ('measurement', 'A'), ('measurement', 'B'), ('measurement', 'C-S'), ('measurement', 'C-N')):
        records = [r for r in value['attempts'] if (r['phase'], r['experiment']) == (phase, kind)]
        lines.append(f"| {phase} | {kind} | {len(records)} | {sum(r['status']=='complete' for r in records)} | "
                     f"{sum(r['status']=='missing' for r in records)} | {sum(r['status'] in ('result-invalid','evidence-invalid') for r in records)} |")
    lines += ['', '| Fixed comparison | Cohort | Planned pairs | Verified pairs | Total cost delta samples |',
             '| --- | --- | ---: | ---: | --- |']
    for kind, cohorts in value['fixed'].items():
        for name, item in cohorts.items():
            lines.append(f"| {kind} | {name} | {item['planned']} | {item['paired']} | "
                         f"{item['improvements'].get('total', {}).get('samples', [])} |")
    lines += ['', '| Search | Planned arms | Complete evidence arms |', '| --- | ---: | ---: |']
    for kind, item in value['search'].items():
        lines.append(f"| {kind} | {item['planned_arms']} | {item['complete_evidence_arms']} |")
    lines += ['', 'The JSON retains every scheduled row, per-stage pairs, endpoint counts, late results, and evidence failures.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'report.sha256').write_text(sha(args.output / 'report.md') + '\n')


if __name__ == '__main__':
    main()
