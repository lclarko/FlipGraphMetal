"""Execute one frozen M4 qualification or measurement attempt with retained evidence."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
sys.path.insert(0, str(ROOT / 'tests/metal'))
from application import digest, write_json, wired_memory
from guard import run as guarded_run
import baseline as b
import two_aux_protocol as proposal

REVIEW_SOURCES = ('benchmarks/workflow/two_aux_execute.py',
                  'benchmarks/workflow/two_aux_protocol.py',
                  'benchmarks/workflow/baseline.py', 'benchmarks/metal/guard.py',
                  'benchmarks/metal/application.py', 'tests/metal/verify.py',
                  'tests/workflow/identity_oracle.py', 'tests/metal/smoke.py')
WIRED_LIMIT = 4294967296
PRELAUNCH_CEILING = 1946157056
FINAL_RESERVE = 50
CHILD_SECONDS = 45
APPROVED_MANIFEST_SHA256 = '8fe9aa117a4679de845a91ad4f995a7e2bec44fac3dba36f6aa30dd76d8158fa'


class ArmClosed(Exception):
    pass


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def durable_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def confined(root, relative):
    proposal._path(relative, 'packet path')
    path = root
    for part in relative.split('/'):
        path = path / part
        require(not path.is_symlink(), 'linked packet path: ' + relative)
    path = path.resolve(strict=True)
    require(path.is_relative_to(root), 'packet path escapes root')
    require(path.is_file(), 'packet path is not a file')
    return path


def packet_file(root, relative, checksum):
    path = confined(root, relative)
    require(digest(path) == checksum, 'packet hash mismatch: ' + relative)
    return path


def read_inputs(packet, authorization, review):
    packet = Path(packet).resolve(strict=True)
    manifest_path = confined(packet, 'manifest.json')
    manifest_sha = digest(manifest_path)
    require(manifest_sha == APPROVED_MANIFEST_SHA256,
            'packet differs from the approved frozen M4 manifest')
    auth = json.loads(Path(authorization).read_text())
    require(auth.get('schema') == 'fgm-two-aux-m4-authorization-v1' and
            auth.get('protocol', {}).get('approved') is True and
            auth['protocol'].get('manifest_sha256') == manifest_sha,
            'authorization does not bind this manifest')
    manifest = proposal.validate_manifest(json.loads(manifest_path.read_text()))
    rev_path = Path(review).resolve(strict=True)
    rev = json.loads(rev_path.read_text())
    require(rev.get('schema') == 'fgm-two-aux-m4-executor-review-v1' and
            rev.get('status') == 'approved' and rev.get('manifest_sha256') == manifest_sha and
            rev.get('sources') == {name: digest(ROOT / name) for name in REVIEW_SOURCES},
            'independent implementation review or source hashes missing')
    for row in manifest['panel']['factors']:
        source = packet_file(packet, row['path'], row['sha256'])
        parsed = b.fgm2_source(source)
        require(b.host_oracle().identity(parsed['raw_reference']) == row['canonical_id'] and
                parsed['factors_id'] == row['effective_factors_id'],
                'factor identity mismatch: ' + row['id'])
    builds = {}
    for role, declaration in manifest['builds'].items():
        paths = {}
        for name, entry in declaration['executables'].items():
            paths[name] = packet_file(packet, entry['path'], entry['sha256'])
            packet_file(packet, entry['receipt_path'], entry['receipt_sha256'])
        paths['shaders/signed.metallib'] = packet_file(
            packet, declaration['library_path'], declaration['library_sha256'])
        packet_file(packet, declaration['library_receipt_path'],
                    declaration['library_receipt_sha256'])
        require(declaration.get('library_mode') == 'metallib', 'compiled library mode changed')
        builds[role] = paths
    return manifest, manifest_sha, digest(rev_path), builds


def build_inventory(manifest, role):
    build = manifest['builds'][role]
    return {**{name: item['sha256'] for name, item in build['executables'].items()},
            'shaders/signed.metallib': build['library_sha256']}


def check_build(paths, inventory):
    require(all(digest(paths[name]) == expected for name, expected in inventory.items()),
            'frozen executable or library changed before launch')


def planned_rows(manifest, phase, experiment, attempt_id):
    if phase == 'qualification':
        key = {'fixed': 'fixed_factor_schedule', 'native': 'native_schedule'}.get(experiment)
        require(key is not None, 'invalid qualification experiment')
        schedule = manifest['qualification'][key]
        attempt = next((item for item in schedule if item['id'] == attempt_id), None)
        require(attempt is not None, 'attempt outside frozen qualification schedule')
        units = attempt['cells'] if experiment == 'fixed' else attempt['units']
    else:
        require(experiment in manifest['experiments'], 'invalid measurement experiment')
        attempt = next((item for item in manifest['experiments'][experiment]['attempts']
                        if item['id'] == attempt_id), None)
        require(attempt is not None, 'attempt outside frozen measurement schedule')
        units = attempt['cells'] if experiment in ('A', 'B') else attempt['arms']
    return attempt, [dict(id=item['id'], status='unrun') for item in units]


def checked_guard(root, relative, recorded):
    """Require retained supervision evidence for a completed work unit."""
    sidecar = packet_file(root, relative + '/result.json', recorded['result_sha256'])
    packet_file(root, relative + '/run.log', recorded['log_sha256'])
    value = json.loads(sidecar.read_text())
    require(value.get('complete') is True and value.get('forced_termination') is False and
            value.get('cleanup_failure') is False and value.get('time_limit') == CHILD_SECONDS and
            value.get('wired_limit') == WIRED_LIMIT,
            'qualification guard did not complete under frozen limits')


def qualification_allowances(root, manifest, manifest_sha, review_sha, authorization_sha):
    """Bind all twenty passing attempts and derive frozen per-treatment allowances."""
    root = Path(root).resolve(strict=True)
    fixed = {}
    native = {}
    expected = [('fixed', a) for a in manifest['qualification']['fixed_factor_schedule']]
    expected += [('native', a) for a in manifest['qualification']['native_schedule']]
    for kind, attempt in expected:
        path = root / kind / attempt['id'] / 'result.json'
        require(path.is_file() and not path.is_symlink(), 'missing qualification: ' + attempt['id'])
        require((path.parent / 'result.sha256').read_text() == digest(path) + '\n',
                'qualification result sidecar mismatch: ' + attempt['id'])
        result = json.loads(path.read_text())
        require(result.get('schema') == 'fgm-two-aux-m4-attempt-v1' and
                result.get('phase') == 'qualification' and result.get('experiment') == kind and
                result.get('attempt_id') == attempt['id'] and
                result.get('manifest_sha256') == manifest_sha and
                result.get('review_sha256') == review_sha and
                result.get('authorization_sha256') == authorization_sha and
                result.get('status') == 'complete', 'qualification is not passing: ' + attempt['id'])
        declaration = confined(path.parent, 'attempt-manifest.json')
        require(digest(declaration) == result.get('attempt_manifest_sha256') and
                json.loads(declaration.read_text()).get('attempt') == attempt,
                'qualification attempt declaration changed')
        rows = result.get('rows', [])
        units = attempt['cells'] if kind == 'fixed' else attempt['units']
        require(len(rows) == len(units) and all(row.get('id') == unit['id'] and
                row.get('status') == 'verified' for row, unit in zip(rows, units)),
                'qualification rows incomplete: ' + attempt['id'])
        for row, unit in zip(rows, units):
            evidence = confined(path.parent, row['evidence_path'])
            require(digest(evidence) == row['verification_sha256'],
                    'qualification verification evidence changed: ' + row['id'])
            if kind == 'fixed':
                for name in ('native', 'verifier'):
                    checked_guard(path.parent, row['id'] + '/' + name + '-guard',
                                  row[name + '_guard'])
                receipt = confined(path.parent, row['receipt_path'])
                artifact = confined(path.parent, row['artifact_path'])
                require(digest(receipt) == row['receipt_sha256'] and
                        digest(artifact) == row['artifact_sha256'],
                        'qualification fixed receipt or artifact changed')
            else:
                chunk = row['chunks'][0]
                base = str(Path(row['evidence_path']).parent)
                for name in ('native', 'evaluation_export', 'observation_export'):
                    checked_guard(path.parent, base + '/' + name.replace('_', '-') + '-guard',
                                  chunk[name + '_guard'])
                checked_guard(path.parent, row['id'] + '/native-verifier-guard',
                              row['native-verifier_guard'])
                packet_file(path.parent, row['state_path'], row['state_sha256'])
                for field, key in (('receipt_path', 'receipt_sha256'),
                                   ('evaluation_export', 'evaluation_export_sha256'),
                                   ('observation_export', 'observation_export_sha256')):
                    require(digest(confined(path.parent, chunk[field])) == chunk[key],
                            'qualification native export changed')
            work = b.fgm3_qualification_work_seconds(row)
            if kind == 'fixed':
                key = (attempt['factor_id'], unit['treatment']['id'])
                fixed.setdefault(key, []).append(work)
            else:
                key = (tuple(attempt['factor_ids']), unit['treatment']['id'], unit['operation'])
                native.setdefault(key, []).append(work)
    result = {}
    for kind, groups, ceiling in [('fixed', fixed, 20), ('native', native, 60)]:
        for key, works in groups.items():
            allowance = b.fgm3_chunk_allowance(works)
            require(allowance <= ceiling, kind + ' qualification allowance exceeds frozen ceiling')
            result[(kind, key)] = allowance
    return result


class Attempt:
    def __init__(self, output, manifest, manifest_sha, review_sha, phase, experiment,
                 attempt, rows, builds, *, clock, sleep, sample, guard, started,
                 packet, authorization, review):
        self.output = Path(output).absolute()
        self.manifest = manifest
        self.phase = phase
        self.experiment = experiment
        self.attempt = attempt
        self.phase_limits = (attempt.get('phases_seconds', {}) if phase == 'qualification' else
                             manifest['experiments'][experiment]['phases_seconds_per_attempt'])
        self.builds = builds
        self.clock, self.sleep, self.sample, self.guard = clock, sleep, sample, guard
        self.started = started
        self.packet = Path(packet)
        self.authorization = Path(authorization)
        self.review = Path(review)
        self.authorization_sha256 = digest(self.authorization)
        self.launch_builds = None
        self.work_deadline = None
        self.headroom_seconds = 0.
        self.deadline = self.started + manifest['resources']['outer_seconds']
        self.data = dict(schema='fgm-two-aux-m4-attempt-v1', phase=phase,
                         experiment=experiment, attempt_id=attempt['id'],
                         manifest_sha256=manifest_sha, review_sha256=review_sha,
                         authorization_sha256=self.authorization_sha256,
                         status='running', started_monotonic=self.started,
                         deadline_monotonic=self.deadline, rows=rows, allowances={})
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()

    def save(self):
        durable_json(self.output / 'result.json', self.data)

    def check_launch(self):
        require(digest(self.packet / 'manifest.json') == self.data['manifest_sha256'] and
                digest(self.review) == self.data['review_sha256'] and
                digest(self.authorization) == self.authorization_sha256,
                'manifest, authorization, or review changed before launch')
        authority = json.loads(self.authorization.read_text())
        require(authority.get('schema') == 'fgm-two-aux-m4-authorization-v1' and
                authority.get('protocol', {}).get('approved') is True and
                authority['protocol'].get('manifest_sha256') == self.data['manifest_sha256'],
                'authorization changed before launch')
        require(json.loads(self.review.read_text())['sources'] ==
                {name: digest(ROOT / name) for name in REVIEW_SOURCES},
                'reviewed sources changed before launch')
        for entry in self.manifest['panel']['factors']:
            packet_file(self.packet, entry['path'], entry['sha256'])
        for build in self.manifest['builds'].values():
            for item in build['executables'].values():
                packet_file(self.packet, item['receipt_path'], item['receipt_sha256'])
            packet_file(self.packet, build['library_receipt_path'],
                        build['library_receipt_sha256'])

    def enough(self, seconds=CHILD_SECONDS):
        return self.clock() + seconds + FINAL_RESERVE <= self.deadline

    def child_deadline(self):
        return min(self.deadline - FINAL_RESERVE,
                   self.work_deadline if self.work_deadline is not None else math.inf)

    def headroom(self, row, allowance, arm_deadline=None, global_work_seconds=CHILD_SECONDS):
        start = self.clock()
        row.setdefault('headroom_wait_seconds', 0.)
        while True:
            require(self.enough(global_work_seconds),
                    'global finalization reserve before headroom sample')
            if arm_deadline is not None:
                if arm_deadline - self.clock() <= allowance:
                    raise ArmClosed('arm admission allowance')
            remaining = self.child_deadline() - self.clock()
            require(remaining > 0, 'work phase deadline before headroom sample')
            wait_ceiling = self.phase_limits.get('headroom_wait', math.inf)
            require(self.headroom_seconds < wait_ceiling, 'headroom phase ceiling exhausted')
            remaining = min(remaining, wait_ceiling - self.headroom_seconds)
            value = self.sample(timeout=min(2, remaining))
            elapsed = self.clock() - start
            row['headroom_wait_seconds'] += elapsed
            self.headroom_seconds += elapsed
            start = self.clock()
            require(self.headroom_seconds <= wait_ceiling, 'headroom phase ceiling exceeded')
            require(self.enough(global_work_seconds),
                    'global finalization reserve after headroom sample')
            if arm_deadline is not None:
                if arm_deadline - self.clock() <= allowance:
                    raise ArmClosed('arm admission after headroom sample')
            if value <= PRELAUNCH_CEILING:
                return
            delay = min(.025, self.deadline - FINAL_RESERVE - global_work_seconds - self.clock(),
                        self.child_deadline() - self.clock(),
                        wait_ceiling - self.headroom_seconds)
            require(delay > 0, 'headroom wait exhausted admission')
            self.sleep(delay)

    def child(self, name, argv, role, row, *, arm_deadline=None, allowance=0,
              work_admission=True, expected_files=None,
              global_work_seconds=CHILD_SECONDS, timed=True):
        require(self.enough(global_work_seconds),
                'global finalization reserve before child')
        if arm_deadline is not None and work_admission:
            if arm_deadline - self.clock() <= allowance:
                raise ArmClosed('arm admission before child')
        self.headroom(row, allowance if work_admission else 0,
                      arm_deadline if work_admission else None, global_work_seconds)
        self.check_launch()
        check_build(self.builds[role], build_inventory(self.manifest, role))
        require(self.launch_builds is not None, 'launched build inventory unavailable')
        check_build(self.launch_builds[role], build_inventory(self.manifest, role))
        for path, checksum in (expected_files or {}).items():
            require(digest(path) == checksum, 'launched input changed: ' + str(path))
        row['status'] = name + '-starting'
        self.save()
        require(self.enough(global_work_seconds),
                'global finalization reserve after durable save')
        for path, checksum in (expected_files or {}).items():
            require(digest(path) == checksum, 'launched input changed after save: ' + str(path))
        if arm_deadline is not None and work_admission:
            if arm_deadline - self.clock() <= allowance:
                raise ArmClosed('arm admission after durable save')
        deadline = self.child_deadline()
        destination = self.output / row['id'] / (name + '-guard')
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            command = (['/usr/bin/time', '-l', '-p'] if timed else []) + list(map(str, argv))
            result = self.guard(command, destination,
                                absolute_deadline=deadline, wired_limit_bytes=WIRED_LIMIT)
        except Exception as exc:
            row['status'] = 'unknown-cleanup'
            self.save()
            raise RuntimeError('guard cleanup unknown') from exc
        sidecar = destination / 'result.json'
        log = destination / 'run.log'
        row[name + '_guard'] = dict(result_sha256=digest(sidecar) if sidecar.is_file() else None,
                                   log_sha256=digest(log) if log.is_file() else None,
                                   wall_seconds=result.get('wall_seconds'),
                                   forced_termination=result.get('forced_termination'),
                                   cleanup_failure=result.get('cleanup_failure'),
                                   wired_limit_bytes=result.get('wired_limit'))
        self.save()
        require(result.get('wired_limit') == WIRED_LIMIT and result.get('complete') and
                not result.get('forced_termination') and not result.get('cleanup_failure') and
                sidecar.is_file() and log.is_file(), name + ' child failed')
        return log

    def finish(self, error=None):
        final_started = self.clock()
        if error is not None:
            self.data['status'] = 'failed'
            self.data['error'] = repr(error)
        else:
            require(all(row['status'] in ('verified', 'complete') for row in self.data['rows']),
                    'attempt has unrun or failed rows')
            self.data['status'] = 'complete'
        self.data['finished_monotonic'] = self.clock()
        if self.clock() > self.deadline:
            self.data['status'] = 'failed'
            self.data['error'] = 'outer deadline exceeded during finalization'
        self.save()
        durable_text(self.output / 'result.sha256',
                     digest(self.output / 'result.json') + '\n')
        self.data.setdefault('phase_seconds', {})['finalization'] = self.clock() - final_started
        if self.clock() > self.deadline or self.clock() - final_started > FINAL_RESERVE:
            self.data['status'] = 'failed'
            self.data['error'] = 'outer deadline or finalization ceiling exceeded after closing writes'
        self.save()
        durable_text(self.output / 'result.sha256',
                     digest(self.output / 'result.json') + '\n')
        if self.clock() > self.deadline or self.clock() - final_started > FINAL_RESERVE:
            self.data['status'] = 'failed'
            self.data['error'] = 'outer deadline or finalization ceiling exceeded after final seal'
            self.data['phase_seconds']['finalization'] = self.clock() - final_started
            self.save()
            durable_text(self.output / 'result.sha256', digest(self.output / 'result.json') + '\n')
        return self.data


def copy_builds(run, packet):
    copies = {}
    for role, paths in run.builds.items():
        inventory = build_inventory(run.manifest, role)
        target = paths['flip_graph'].parent
        require(all(path == target / name for name, path in paths.items()),
                'frozen build paths are not co-located')
        copies[role] = paths
        check_build(copies[role], inventory)
    run.launch_builds = copies
    return copies


def factor(run, packet, factor_id):
    entry = next(item for item in run.manifest['panel']['factors'] if item['id'] == factor_id)
    source = packet_file(packet, entry['path'], entry['sha256'])
    target = run.output / 'factors' / (factor_id + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    require(digest(target) == entry['sha256'], 'copied factor changed')
    parsed = b.fgm2_source(target)
    require(b.host_oracle().identity(parsed['raw_reference']) == entry['canonical_id'] and
            parsed['factors_id'] == entry['effective_factors_id'], 'copied factor identity changed')
    return target, parsed, entry


def fixed_unit(run, packet, builds, unit, row, *, arm_deadline=None, allowance=0):
    start = run.clock()
    row['started_seconds'] = start
    row['status'] = 'prepared'
    row['headroom_wait_seconds'] = 0.
    run.save()
    role = unit['treatment']['build']
    factor_id = unit['factor_id'] if 'factor_id' in unit else unit['factor_ids'][0]
    factor_path, source, entry = factor(run, packet, factor_id)
    work = run.output / row['id']
    work.mkdir(parents=True, exist_ok=True)
    reduction = dict(unit['evaluation_settings'])
    reduction['seed'] = unit.get('reduction_seed', unit.get('first_reduction_seed'))
    require(type(reduction['seed']) is int and reduction['seed'] > 0,
            'fixed reduction seed missing')
    if 'constructor' in unit['treatment']:
        reduction['constructor'] = unit['treatment']['constructor']
    config = dict(schema='fgm-run-v1', operation='reduce',
                  input=dict(kind='files', files=[dict(path=str(factor_path), format='json', domain='ZT')]),
                  execution=dict(workers=1, batch_steps=1, block_size=32, backend='general',
                                 memory_bytes=run.manifest['resources']['execution_memory_bytes']),
                  reduction=reduction, output=str(work / 'receipt.json'))
    config_path = work / 'config.json'
    write_json(config_path, config)
    row['config_sha256'] = digest(config_path)
    row['input_sha256'] = digest(factor_path)
    run.save()
    require(digest(config_path) == row['config_sha256'] and digest(factor_path) == entry['sha256'],
            'fixed configuration or factor changed')
    log = run.child('native', [builds[role]['additions_reducer'], '--run-config', config_path],
                    role, row, arm_deadline=arm_deadline, allowance=allowance,
                    global_work_seconds=2 * CHILD_SECONDS + 10,
                    expected_files={config_path: row['config_sha256'],
                                    factor_path: row['input_sha256']})
    receipt_path = work / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    producer = build_inventory(run.manifest, role)
    require(receipt['status'] == 'complete' and receipt['configuration_sha256'] == row['config_sha256']
            and receipt['executable_sha256'] == producer['additions_reducer']
            and receipt.get('library_mode') == 'metallib'
            and receipt.get('library_sha256') == producer['shaders/signed.metallib']
            and len(receipt['presentations']) == 1
            and receipt['presentations'][0]['source_sha256'] == entry['sha256']
            and receipt['presentations'][0]['effective_factors_id'] == entry['effective_factors_id'],
            'fixed receipt, factor, or producer mismatch')
    require(digest(config_path) == row['config_sha256'], 'fixed config changed during native child')
    row['receipt_sha256'] = digest(receipt_path)
    row['receipt_path'] = str(receipt_path.relative_to(run.output))
    artifact = Path(str(receipt_path) + '.circuits.jsonl')
    preliminary = b.fgm2_preliminary_circuit(artifact, receipt, source)
    row['artifact_sha256'] = digest(artifact)
    row['artifact_path'] = str(artifact.relative_to(run.output))
    row['gpu'] = b.native_dispatch_evidence(log.read_text(), receipt)
    # The omitted treatment must not enter two-auxiliary construction.
    if 'constructor' not in unit['treatment']:
        require(not receipt['results'][0].get('two_auxiliary') and
                not receipt.get('two_aux_preparation_bytes') and
                not any(item[0] == 'constructorTwoAuxClosureKernel'
                        for item in row['gpu'].get('dispatches', [])),
                'omitted treatment dispatched two-auxiliary construction')
    reference = source['reference']
    reference_path = work / 'independent-reference.json'
    write_json(reference_path, dict(n=reference['dimensions'], m=reference['rank'], z2=False,
                                   **{key: reference[key] for key in 'uvw'}))
    row['reference_sha256'] = digest(reference_path)
    run.save()
    require(digest(receipt_path) == row['receipt_sha256'] and
            digest(artifact) == row['artifact_sha256'], 'fixed artifact changed before verification')
    verifier_log = run.child('verifier', [sys.executable, ROOT / 'tests/metal/verify.py',
                                          artifact, '--reference', reference_path],
                             role, row, work_admission=False, timed=False,
                             expected_files={artifact: row['artifact_sha256'],
                                             reference_path: row['reference_sha256'],
                                             receipt_path: row['receipt_sha256']})
    require(digest(receipt_path) == row['receipt_sha256'] and
            digest(artifact) == row['artifact_sha256'] and
            digest(reference_path) == row['reference_sha256'],
            'fixed evidence changed during verification')
    verified = b.fgm2_bind_verifier(verifier_log, artifact, reference_path,
                                    preliminary, receipt, source)
    result = receipt['results'][0]
    evidence = dict(schema='fgm-two-aux-m4-fixed-verification-v1',
                    config_sha256=row['config_sha256'], receipt_sha256=row['receipt_sha256'],
                    artifact_sha256=row['artifact_sha256'],
                    verifier_log_sha256=digest(verifier_log), verified=verified,
                    result=result, treatment=unit['treatment'])
    durable_json(work / 'verification.json', evidence)
    row['verification_sha256'] = digest(work / 'verification.json')
    row['evidence_path'] = str((work / 'verification.json').relative_to(run.output))
    row['additions'] = verified['additions']
    row['additions_by_stage'] = verified['additions_by_stage']
    row['verified_seconds'] = run.clock()
    row['elapsed_seconds'] = run.clock() - start
    row['status'] = 'verified'
    run.save()
    # Endpoint credit starts after the verification file and progress record exist.
    row['durable_seconds'] = run.clock()
    run.save()
    row['elapsed_seconds'] = run.clock() - start
    run.save()
    return row


def population(run, packet, ids, label):
    root = run.output / 'populations' / label
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, factor_id in enumerate(ids):
        original, parsed, panel = factor(run, packet, factor_id)
        relative = f'{index:02d}-{factor_id}.json'
        shutil.copyfile(original, root / relative)
        require(digest(root / relative) == panel['sha256'], 'population copy hash mismatch')
        entries.append(dict(path=relative, label=factor_id, sha256=panel['sha256'],
                            source_factors_id=parsed['raw_factors_id'],
                            effective_factors_id=parsed['factors_id'],
                            canonical_id=panel['canonical_id']))
    write_json(root / 'population.json', dict(schema='fgm-two-aux-m4-population-v1',
                                              count=len(entries), entries=entries))
    return root, entries


def native_protocol(run, unit):
    frozen = unit['native_search_settings']
    pinned = json.loads(b.FGM3_PROTOCOL_PATH.read_text())
    require(digest(b.FGM3_PROTOCOL_PATH) == run.manifest['fgm3_protocol']['sha256'],
            'pinned FGM-3 protocol changed')
    protocol = copy.deepcopy(pinned)
    for key in ('execution', 'policy', 'pool', 'evaluation', 'history', 'circuit_target'):
        protocol[key] = copy.deepcopy(frozen[key])
    protocol.update(seeds=run.manifest['seeds'], selectors=['cost-diverse'],
                    wired_limit_bytes=WIRED_LIMIT,
                    launch_wired_reserve_bytes=run.manifest['resources']['launch_reserve_bytes'],
                    finalization_reserve_seconds=FINAL_RESERVE,
                    native_guard_seconds=CHILD_SECONDS,
                    native_admission_seconds=4 * CHILD_SECONDS + 10 + FINAL_RESERVE)
    if 'constructor' in unit['treatment']:
        protocol['evaluation']['constructor'] = unit['treatment']['constructor']
    # Retain the pinned FGM-3 read budget. Search, history and evaluator settings
    # come only from the frozen M4 unit.
    return protocol


def native_unit(run, packet, builds, unit, row, ids, *, allowance, arm_deadline=None):
    started = run.clock()
    row['started_seconds'] = started
    row['status'] = 'prepared'
    row['headroom_wait_seconds'] = 0.
    run.save()
    role = unit['treatment']['build']
    require(role == 'candidate', 'native qualification uses candidate build')
    history_id = unit.get('history_id', row['id'])
    context = run.output / 'native-context' / history_id
    if unit.get('operation') == 'resume':
        require(context.is_dir(), 'native resume lacks matching initialization')
    else:
        context.mkdir(parents=True, exist_ok=False)
    pop_dir = context / 'population'
    if not pop_dir.exists():
        _, entries = population(run, packet, ids, history_id)
        shutil.copytree(run.output / 'populations' / history_id, pop_dir)
    else:
        entries = json.loads((pop_dir / 'population.json').read_text())['entries']
    population_sha = digest(pop_dir / 'population.json')
    protocol = native_protocol(run, unit)
    binaries = builds[role]['flip_graph'].parent
    inventory = build_inventory(run.manifest, role)
    check_build({name: binaries / name for name in inventory}, inventory)
    state_path = context / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = dict(build_inventory=inventory, evaluations=[], installations=[],
                     observations=[], initial_ids=[entry['canonical_id'] for entry in entries])
    chunk = unit.get('chunk', 0)
    require((chunk == 0 and unit.get('operation') == 'initialize') or
            (chunk >= 1 and unit.get('operation') == 'resume'),
            'native chunk or operation mismatch')
    require(unit.get('evaluation_base_seed', unit['trial_seed']) == unit['trial_seed'] and
            unit.get('generation_seed', b.fgm3_generation_seed(unit['trial_seed'], chunk)) ==
            b.fgm3_generation_seed(unit['trial_seed'], chunk),
            'native frozen seed binding mismatch')
    local_arm = str(context.relative_to(run.output))
    # The old helper owns receipt-bound exports and exact circuit verification.
    # A wrapper adds the M4 global deadline and a fresh byte check for each child.
    effective_arm_deadline = min(arm_deadline if arm_deadline is not None else math.inf,
                                 run.child_deadline())
    def before_native(config_path, record):
        run.save()
        run.check_launch()
        check_build({name: binaries / name for name in inventory}, inventory)
        require(digest(config_path) == record['config_sha256'],
                'native configuration changed before launch')
        require(digest(pop_dir / 'population.json') == population_sha and
                all(digest(pop_dir / entry['path']) == entry['sha256'] for entry in entries),
                'native population changed before launch')
        # These checks can take time. Admission must use the clock after them.
        return b.fgm3_admission(run.clock(), effective_arm_deadline, run.deadline,
                                allowance, protocol)
    def guard(argv, destination, *, wired_limit_bytes):
        if record['status'] == 'native-started':
            # before_native just completed admission outside the supervision
            # exception handler, so a closed arm remains an unrun unit.
            return run.guard(argv, destination, absolute_deadline=run.child_deadline(),
                             wired_limit_bytes=wired_limit_bytes)
        reserve = {'evaluation-export-started': 145,
                   'observation-export-started': 100}.get(record['status'], CHILD_SECONDS)
        require(run.enough(reserve), 'native stage lacks remaining child reserves')
        run.headroom(row, 0, global_work_seconds=reserve)
        run.save()
        require(run.enough(reserve), 'native stage reserve crossed during save')
        run.check_launch()
        check_build({name: binaries / name for name in inventory}, inventory)
        require(run.enough(reserve), 'native stage reserve crossed during input checks')
        return run.guard(argv, destination, absolute_deadline=run.child_deadline(),
                         wired_limit_bytes=wired_limit_bytes)
    def sample(*, timeout):
        remaining = run.child_deadline() - run.clock()
        require(remaining > 0, 'native work phase deadline before headroom sample')
        return run.sample(timeout=min(timeout, remaining))
    record = dict(index=chunk, status='scheduled', started_seconds=started,
                  headroom_wait_seconds=0.)
    row['chunks'] = [record]
    run.save()
    def verify_callback(output,population,protocol,selector,chunk,state,receipt,
                        receipt_path,config,record,export_path,observation_path,
                        started,expected_population_count,clock):
        record['status'] = 'verification-started'
        payload = dict(schema='fgm-two-aux-m4-native-verifier-input-v1',
                       output=str(output), population=str(population), protocol=protocol,
                       selector=selector, chunk=chunk, state=state, record=record,
                       started=started, expected_population_count=expected_population_count,
                       paths={name: str(path) for name,path in (
                           ('receipt', receipt_path), ('config', Path(record['config_path'])
                            if 'config_path' in record else Path(receipt_path).parent / 'config.json'),
                           ('evaluations', export_path), ('observations', observation_path),
                           ('population', Path(population) / 'population.json'))},
                       hashes={name: digest(path) for name,path in (
                           ('receipt', receipt_path),
                           ('config', Path(receipt_path).parent / 'config.json'),
                           ('evaluations', export_path), ('observations', observation_path),
                           ('population', Path(population) / 'population.json'))})
        request = Path(receipt_path).parent / 'm4-verifier-input.json'
        response = Path(receipt_path).parent / 'm4-verifier-output.json'
        durable_json(request, payload)
        request_sha = digest(request)
        row['verifier_input_sha256'] = request_sha
        run.save()
        run.child('native-verifier', [sys.executable, Path(__file__), '--verify-native',
                                      request, response], role, row, work_admission=False,
                                      expected_files={request: request_sha})
        require(digest(request) == request_sha, 'native verifier input changed')
        checked = json.loads(response.read_text())
        verification_path = Path(receipt_path).parent / 'verification.json'
        require(checked.get('input_sha256') == request_sha and
                checked.get('record', {}).get('status') == 'verified' and
                checked.get('record', {}).get('verification_sha256') ==
                digest(verification_path),
                'native verifier result lacks bound exact verification')
        with verification_path.open('rb') as stream:
            os.fsync(stream.fileno())
        for name,path in payload['paths'].items():
            require(digest(Path(path)) == payload['hashes'][name],
                    'native verifier input changed during verification')
        record.update(checked['record'])
        state.clear()
        state.update(checked['state'])
        row['verifier_output_sha256'] = digest(response)
        return record
    result = b.fgm3_run_chunk(run.output, pop_dir, binaries, protocol, 'cost-diverse',
                              unit['trial_seed'], 1, local_arm, chunk, state,
                              run.deadline, effective_arm_deadline, allowance,
                              clock=run.clock, sleeper=run.sleep, guard=guard,
                              memory_sample=sample, record=record,
                              expected_population_count=len(ids),
                              verify_callback=verify_callback, before_native=before_native)
    row['headroom_wait_seconds'] += result['headroom_wait_seconds']
    row['elapsed_seconds'] = run.clock() - started
    if result['status'] == 'unrun':
        row.update(status='unrun', stop_reason=result.get('stop_reason', 'arm admission'))
        run.save()
        return row
    require(result['status'] == 'verified', 'native unit did not complete exact verification')
    # Persist the entire state and raw export paths before assigning M4 credit.
    durable_json(state_path, state)
    snapshot = context / f'state-{chunk:03d}.json'
    durable_json(snapshot, state)
    row['state_sha256'] = digest(snapshot)
    row['state_path'] = str(snapshot.relative_to(run.output))
    row['best_additions'] = result['best_additions']
    row['verification_sha256'] = result['verification_sha256']
    evidence = context / 'chunks' / f'{chunk:03d}' / 'verification.json'
    row['evidence_path'] = str(evidence.relative_to(run.output))
    result['receipt_path'] = str((context / 'chunks' / f'{chunk:03d}' / 'receipt.json').relative_to(run.output))
    row['status'] = 'verified'
    run.save()
    row['durable_seconds'] = run.clock()
    for evaluation in result['new_evaluations']:
        evaluation['verified_seconds'] = row['durable_seconds']
    row['two_auxiliary'] = [item['evaluation'].get('two_auxiliary') for item in
                            b.fgm3_export_rows(run.output / result['evaluation_export'])[0]]
    for evaluation in state['evaluations']:
        if evaluation['scheme_id'] in {item['scheme_id'] for item in result['new_evaluations']}:
            evaluation['verified_seconds'] = row['durable_seconds']
    durable_json(state_path, state)
    durable_json(snapshot, state)
    row['state_sha256'] = digest(snapshot)
    run.save()
    row['elapsed_seconds'] = run.clock() - started
    run.save()
    return row


def endpoint(rows, started, seconds):
    cutoff = started + seconds
    timely = [row for row in rows if row.get('status') == 'verified' and
              row.get('durable_seconds', math.inf) <= cutoff]
    native = any('chunks' in row for row in timely)
    evaluated = (len({entry['scheme_id'] for row in timely for chunk in row.get('chunks', [])
                      for entry in chunk.get('new_evaluations', [])}) if native else len(timely))
    return dict(best_additions=min((row.get('additions', row.get('best_additions'))
                                    for row in timely), default=None),
                verified_evaluations=evaluated,
                target_54_attained=any(row.get('additions', row.get('best_additions', math.inf)) <= 54
                                       for row in timely))


def measure_fixed_arm(run, packet, builds, unit, row, allowance):
    started = run.clock()
    deadline = started + unit['seconds']
    row.update(status='running', started_seconds=started, blocks=[], endpoints={})
    run.save()
    require(unit['first_reduction_seed'] == b.effectiveness_seed(
        unit['factor_ids'][0], unit['trial_seed'], 'reduce', 0),
        'fixed arm first seed mismatch')
    for block in range(1000000):
        if run.clock() >= deadline or deadline - run.clock() <= allowance or not run.enough(100):
            break
        current = dict(unit)
        current['reduction_seed'] = b.effectiveness_seed(unit['factor_ids'][0],
            unit['trial_seed'], 'reduce', block)
        work = dict(id=f"{row['id']}-block-{block:03d}", status='unrun')
        row['blocks'].append(work)
        run.save()
        try:
            fixed_unit(run, packet, builds, current, work, arm_deadline=deadline,
                       allowance=allowance)
        except ArmClosed as exc:
            work.update(status='unrun', stop_reason=str(exc))
            run.save()
            break
        if work['status'] != 'verified':
            raise ValueError('fixed timed block incomplete')
    while run.clock() < deadline and run.enough(0):
        run.sleep(min(30, deadline - run.clock()))
    row['elapsed_seconds'] = run.clock() - started
    row['endpoints'] = {str(point): endpoint(row['blocks'], started, point)
                        for point in unit['endpoints']}
    row['status'] = 'complete' if run.clock() >= deadline else 'failed'
    run.save()
    require(row['status'] == 'complete', 'timed fixed arm incomplete')


def measure_native_arm(run, packet, builds, unit, row, allowances):
    started = run.clock()
    deadline = started + unit['seconds']
    ids = unit['factor_ids']
    row.update(status='running', started_seconds=started, chunks=[], endpoints={})
    run.save()
    require(unit['first_generation_seed'] == b.fgm3_generation_seed(unit['trial_seed'], 0),
            'native arm first seed mismatch')
    for chunk in range(1000000):
        operation = 'initialize' if chunk == 0 else 'resume'
        key = ('native', (tuple(ids), unit['treatment']['id'], operation))
        allowance = allowances[key]
        if run.clock() >= deadline or deadline - run.clock() <= allowance or not run.enough(45):
            break
        work = dict(id=f"{row['id']}-chunk-{chunk:03d}", status='unrun')
        row['chunks'].append(work)
        run.save()
        frozen = dict(unit, operation=operation, chunk=chunk,
                      history_id=row['id'])
        native_unit(run, packet, builds, frozen, work, ids, allowance=allowance,
                    arm_deadline=deadline)
        if work['status'] == 'unrun':
            break
        if work['status'] != 'verified':
            raise ValueError('native timed chunk incomplete')
        if work['best_additions'] <= 54:
            row['target_observed'] = True
            break
    while run.clock() < deadline and run.enough(0):
        run.sleep(min(30, deadline - run.clock()))
    row['elapsed_seconds'] = run.clock() - started
    row['endpoints'] = {str(point): endpoint(row['chunks'], started, point)
                        for point in unit['endpoints']}
    row['status'] = 'complete' if run.clock() >= deadline else 'failed'
    run.save()
    require(row['status'] == 'complete', 'timed native arm incomplete')


def check_phase_ceilings(run):
    """Account disjoint attempt phases against the frozen reservations."""
    phases = dict(preparation=run.data['preparation_seconds'])
    rows = run.data['rows']
    if run.experiment in ('fixed', 'A', 'B'):
        for key, name in (('native', 'native_guard'), ('independent_verification', 'verifier_guard')):
            phases[key] = sum(row.get(name, {}).get('wall_seconds', 0) for row in rows)
        phases['headroom_wait'] = sum(row.get('headroom_wait_seconds', 0) for row in rows)
    elif run.experiment == 'native':
        phases['work_units'] = sum(row.get('elapsed_seconds', 0) for row in rows)
        phases['drain'] = 0.
    else:
        units = run.attempt['arms']
        phases['arms'] = sum(min(row.get('elapsed_seconds', 0), unit['seconds'])
                             for row, unit in zip(rows, units))
        phases['drain'] = sum(max(0, row.get('elapsed_seconds', 0) - unit['seconds'])
                              for row, unit in zip(rows, units))
    phases['audit'] = max(0, run.clock() - run.started - sum(phases.values()))
    run.data['phase_seconds'] = phases
    for name, seconds in phases.items():
        require(seconds <= run.phase_limits[name], name + ' phase ceiling exceeded')


def execute(packet, authorization, review, phase, experiment, attempt_id, output,
            *, qualification_root=None, clock=time.monotonic, sleep=time.sleep,
            sample=wired_memory, guard=guarded_run):
    started = clock()
    manifest, manifest_sha, review_sha, builds = read_inputs(packet, authorization, review)
    attempt, rows = planned_rows(manifest, phase, experiment, attempt_id)
    allowances = None
    if phase == 'measurement':
        require(qualification_root is not None, 'measurement needs complete qualification root')
        allowances = qualification_allowances(qualification_root, manifest, manifest_sha,
                                                review_sha, digest(Path(authorization)))
    run = Attempt(output, manifest, manifest_sha, review_sha, phase, experiment,
                  attempt, rows, builds, clock=clock, sleep=sleep, sample=sample,
                  guard=guard, started=started, packet=packet,
                  authorization=authorization, review=review)
    durable_json(run.output / 'attempt-manifest.json',
                 dict(schema='fgm-two-aux-m4-attempt-manifest-v1',
                      frozen_manifest_sha256=manifest_sha, phase=phase,
                      experiment=experiment, attempt=attempt))
    run.data['attempt_manifest_sha256'] = digest(run.output / 'attempt-manifest.json')
    run.save()
    try:
        copies = copy_builds(run, packet)
        require(clock() - started <= 75, 'frozen preparation budget exceeded')
        run.data['preparation_seconds'] = clock() - started
        units = attempt['cells'] if experiment in ('fixed', 'A', 'B') else (
            attempt['units'] if experiment == 'native' else attempt['arms'])
        for unit, row in zip(units, rows):
            if phase == 'qualification' and experiment == 'fixed' or experiment in ('A', 'B'):
                require(run.enough(100), 'fixed native and verifier admission reserve')
                fixed_unit(run, packet, copies, unit, row)
            elif phase == 'qualification' and experiment == 'native':
                remaining = attempt['phases_seconds']['work_units'] - sum(
                    previous.get('elapsed_seconds', 0) for previous in rows)
                require(remaining > 0, 'native qualification work phase ceiling exhausted')
                run.work_deadline = clock() + remaining
                require(run.enough(45), 'native qualification admission reserve')
                native_unit(run, packet, copies, unit, row, attempt['factor_ids'],
                            allowance=60)
                run.work_deadline = None
            elif experiment == 'C-S':
                factor_id = unit['factor_ids'][0]
                key = ('fixed', (factor_id, unit['treatment']['id']))
                measure_fixed_arm(run, packet, copies, unit, row, allowances[key])
            else:
                measure_native_arm(run, packet, copies, unit, row, allowances)
        if phase == 'qualification':
            if experiment == 'fixed':
                for treatment in ('candidate-omitted', 'candidate-enabled'):
                    work = [b.fgm3_qualification_work_seconds(row)
                            for row, unit in zip(rows, units)
                            if unit['treatment']['id'] == treatment]
                    allowance = b.fgm3_chunk_allowance(work)
                    require(allowance <= 20, 'fixed qualification allowance exceeds 20 seconds')
                    run.data['allowances'][treatment] = allowance
            else:
                for operation in ('initialize', 'resume'):
                    work = [b.fgm3_qualification_work_seconds(row)
                            for row, unit in zip(rows, units)
                            if unit['operation'] == operation]
                    allowance = b.fgm3_chunk_allowance(work)
                    require(allowance <= 60, 'native qualification allowance exceeds 60 seconds')
                    run.data['allowances'][operation] = allowance
        check_phase_ceilings(run)
        run.finish()
    except Exception as exc:
        current = next((row for row in rows if row['status'] not in ('verified', 'complete')),
                       None)
        if current is not None and current['status'] != 'unrun':
            current['failed_stage'] = current['status']
            current['status'] = 'failed'
            current['error'] = repr(exc)
            current['elapsed_seconds'] = clock() - current.get('started_seconds', clock())
        run.finish(exc)
    return run.data


def verify_native(request_path, response_path):
    """Run the existing exact FGM-3 binder inside one guarded host child."""
    request_path = Path(request_path).resolve(strict=True)
    response_path = Path(response_path)
    require(not response_path.exists(), 'native verifier output already exists')
    request_sha = digest(request_path)
    data = json.loads(request_path.read_text())
    require(data.get('schema') == 'fgm-two-aux-m4-native-verifier-input-v1',
            'native verifier input schema mismatch')
    for name, path in data['paths'].items():
        require(digest(Path(path)) == data['hashes'][name],
                'native verifier input hash mismatch: ' + name)
    receipt_path = Path(data['paths']['receipt'])
    record = data['record']
    state = data['state']
    b.fgm3_verify_chunk(data['output'], data['population'], data['protocol'],
                        data['selector'], data['chunk'], state,
                        json.loads(receipt_path.read_text()), receipt_path,
                        json.loads(Path(data['paths']['config']).read_text()), record,
                        Path(data['paths']['evaluations']),
                        Path(data['paths']['observations']), data['started'],
                        data['expected_population_count'], time.monotonic)
    require(digest(request_path) == request_sha, 'native verifier request changed')
    durable_json(response_path, dict(schema='fgm-two-aux-m4-native-verifier-output-v1',
                                     input_sha256=request_sha, record=record, state=state))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ['--verify-native']:
        require(len(argv) == 3, 'native verifier needs input and new output')
        verify_native(argv[1], argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--phase', choices=('qualification', 'measurement'), required=True)
    parser.add_argument('--experiment', choices=('fixed', 'native', 'A', 'B', 'C-S', 'C-N'),
                        required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification-root', type=Path)
    args = parser.parse_args(argv)
    result = execute(args.packet, args.authorization, args.review, args.phase,
                     args.experiment, args.attempt, args.output,
                     qualification_root=args.qualification_root)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
