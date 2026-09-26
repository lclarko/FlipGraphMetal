"""Check a relocated shader package serially under the standard GPU guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'benchmarks/metal'))
sys.path.insert(0, str(ROOT / 'tests/workflow'))
from metal_library import package, PRODUCTION
from guard import run
from application import write_json
from smoke import check_fixtures, dispatch_evidence
from verify import verify
from copy import deepcopy
from host_rng import HostRNG


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


def native_host_checks(binary, output, cwd):
    """Invoke the packaged native verifier directly with no PATH executables."""
    output.mkdir(parents=True, exist_ok=False)
    empty_path = output / 'empty-path'
    empty_path.mkdir()
    environment = {key: os.environ[key] for key in ('HOME', 'TMPDIR', 'LANG', 'LC_ALL')
                   if key in os.environ}
    environment['PATH'] = str(empty_path)
    records = []
    summary = {'complete': False, 'path': str(empty_path), 'runs': records}
    write_json(output / 'results.json', summary)
    try:
        # A generated scalar product is exact in both declared domains.
        for domain in ('ZT', 'F2'):
            source = output / (domain + '.json')
            source.write_text(json.dumps(dict(domain=domain, orientation='cyclic-w',
                dimensions=[1, 1, 1], rank=1, u=[[1]], v=[[1]], w=[[1]])))
            for operation in ('verify', 'analyze'):
                destination = output / (domain + '-' + operation + '.jsonl')
                command = [str(binary), operation, '--input', str(source),
                           '--format', 'json', '--output', str(destination)]
                log = output / (domain + '-' + operation + '.log')
                record = {'command': command, 'complete': False, 'log': log.name}
                records.append(record)
                write_json(output / 'results.json', summary)
                with log.open('x') as stream:
                    result = subprocess.run(command, cwd=cwd, env=environment,
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=45)
                record['exit_code'] = result.returncode
                require(result.returncode == 0, 'native host invocation failed')
                result_record = json.loads(destination.read_text())
                require(result_record['verification'] == ('exact-Z' if domain == 'ZT' else 'exact-F2'),
                        'native host verification domain mismatch')
                require(result_record['dimensions'] == [1, 1, 1] and result_record['rank'] == 1,
                        'native host verification changed dimensions or rank')
                record.update(complete=True, output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
                write_json(output / 'results.json', summary)
        summary['complete'] = True
        return summary
    finally:
        write_json(output / 'results.json', summary)


def native_workflow_checks(binaries, output, cwd):
    """Exercise production entry points directly while PATH has no Python."""
    output.mkdir(parents=True, exist_ok=False)
    empty_path = output/'empty-path'; empty_path.mkdir()
    records = []
    def execute(name, program, config):
        config_path = output/(name+'.json')
        destination = output/(name+'-record.json')
        config = deepcopy(config); config['output'] = str(destination)
        config_path.write_text(json.dumps(config, sort_keys=True)+'\n')
        argv = ['/usr/bin/env', 'PATH='+str(empty_path), str(binaries/program),
                '--run-config', str(config_path)]
        result = run(argv, output/(name+'-guard'))
        require(result['complete'], name+' native workflow failed')
        record = json.loads(destination.read_text())
        require(record['status']=='complete' and record['execution_started'], name+' incomplete native run record')
        records.append(dict(name=name, command=argv, record=record))
        write_json(output/'results.json', dict(complete=False, runs=records))
        return record
    base = dict(schema='fgm-run-v1', operation='search',
        policy=dict(schema='fgm-controlled-config-v1', policy='controlled-v1', mode='alternatives', domain='ZT',
            seed=7, dimensions=[3,3,3], collection_rank=26, excursion=1, interval_min=2, interval_max=3,
            reduction_q=0, stagnation_limit=5, flip_budget=6, control_budget=20, optional_quota=2, target_rank=None),
        input=dict(kind='files', files=[]),
        execution=dict(workers=1, batch_steps=2, block_size=32, backend='general', memory_bytes=268435456),
        pool=dict(capacity_per_rank=4, reserve_per_rank=4, memory_bytes=1048576, stage_threshold=2, selector='uniform'),
        history=dict(path='', storage_bytes=33554432, transaction_bytes=1048576, index_memory_bytes=65536))
    # Public generated scalar identities expose pool selection without a GPU
    # dispatch. Capacity two evicts the first rank-two seed; rank one is ineligible.
    scalar = dict(dimensions=[1,1,1], rank=1, domain='ZT', orientation='cyclic-w',
                  u=[[1]], v=[[1]], w=[[1]])
    inventory = []
    for factor in 'uvw':
        item = deepcopy(scalar); item['rank']=2
        for key in 'uvw': item[key]=[[1],[1]]
        item[factor][1][0]=0
        inventory.append(item)
    inventory.append(scalar)
    source = output/'initial-parents.jsonl'
    source.write_text(''.join(json.dumps(item)+'\n' for item in inventory))
    verified = output/'initial-parents-verified.jsonl'
    check = run(['/usr/bin/env','PATH='+str(empty_path),str(binaries/'scheme_tool'),'verify',
                 '--input',str(source),'--format','jsonl','--output',str(verified)], output/'initial-parents-verify-guard')
    require(check['complete'], 'generated initial parents failed verification')
    identities = [json.loads(line)['scheme_id'] for line in verified.read_text().splitlines()]
    for mode in ('alternatives','rank-reduction'):
        for selector in ('uniform','flips'):
            name='initial-'+mode+'-'+selector
            config=deepcopy(base); config['policy'].update(mode=mode,dimensions=[1,1,1],flip_budget=0,control_budget=0)
            config['policy'].pop('collection_rank')
            config['policy']['collection_rank' if mode=='alternatives' else 'stage_rank']=2
            config['input']['files']=[dict(path=str(source),format='jsonl')]
            config['execution']['workers']=4
            config['pool'].update(capacity_per_rank=2,reserve_per_rank=2,selector=selector)
            config['history']['path']=str(output/(name+'-history'))
            record=execute(name,'flip_graph',config)
            rng=HostRNG(7)
            expected=[identities[1:3][rng.bounded(2) if selector=='uniform' else rng.select([2,2])] for _ in range(4)]
            require([worker['parent_id'] for worker in record['workers']]==expected, 'initial selection bypassed bounded eligible pool or N5 draws')
            require(record['actual_backend'] is None and record['counters']['flip_attempts']==0, 'zero-budget selection dispatched GPU work')
            config['policy']['collection_rank' if mode=='alternatives' else 'stage_rank']=3
            config['history']['path']=str(output/(name+'-empty-history'))
            empty=execute(name+'-empty','flip_graph',config)
            require(empty['terminal_reason']=='no_eligible_parent' and empty['actual_backend'] is None, 'empty eligible roster did not stop before allocation')
            config['policy']['target_rank']=1
            config['history']['path']=str(output/(name+'-target-history'))
            target=execute(name+'-target','flip_graph',config)
            require(target['terminal_reason']=='existing_target_met' and 'workers' not in target, 'input target preflight did not precede selection')
    # The pinned scalar reference independently reduces these exact scalar
    # tensors from rank three to one on seed 7's first flip in both domains.
    # A stage charge uses the second credit; installation needs a third.
    for domain in ('ZT','F2'):
        source=output/(domain+'-stage-input.json')
        item=deepcopy(scalar); item.update(domain=domain,rank=3,u=[[1],[1],[1]],v=[[1],[1],[1]],
                                         w=[[1],[1],[1 if domain=='F2' else -1]])
        source.write_text(json.dumps(item))
        for credits in (2,3):
            name=domain+'-stage-'+str(credits)
            config=deepcopy(base); config['policy'].pop('collection_rank')
            config['policy'].update(mode='rank-reduction',domain=domain,dimensions=[1,1,1],stage_rank=3,
                                    flip_budget=10,control_budget=credits)
            config['execution']['batch_steps']=1
            config['pool']['stage_threshold']=1
            config['input']['files']=[dict(path=str(source),format='json')]
            config['history']['path']=str(output/(name+'-history'))
            result=execute(name,'flip_graph_f2' if domain=='F2' else 'flip_graph',config)
            worker,=result['workers']
            require(result['final_stage']==1 and worker['current_rank']==1 and worker['best_rank']==1,
                    'stage did not advance to the verified scalar result')
            require(worker['controls']==credits and worker['flips']==1, 'stage or installation credit mismatch')
            initial_id=result['presentations'][0]['scheme_id']
            require((worker['parent_id']==initial_id)==(credits==2), 'uninstalled parent replaced installed-parent provenance')
    selected_source=output/'ZT-stage-input.json'
    manifest=output/'public-selection.jsonl'
    manifest.write_text(json.dumps(dict(schema='fgm-collection-v1',namespace='public',id='scalar-three',
        path=selected_source.name,sha256=hashlib.sha256(selected_source.read_bytes()).hexdigest(),
        format='json',domain='ZT',dimensions=[1,1,1]))+'\n')
    selection=deepcopy(base); selection['policy'].update(dimensions=[1,1,1],collection_rank=3,target_rank=3)
    selection['input']=dict(kind='selection',manifest=str(manifest),count=1,seed=7)
    selection['history']['path']=str(output/'selection-history')
    selected=execute('external-selection','flip_graph',selection)
    require(selected['terminal_reason']=='existing_target_met' and
            selected['presentations'][0]['source_binding']['id']=='scalar-three', 'external selection lost its input binding')
    previous = None
    for domain, backend in [('ZT','general'),('F2','general'),('ZT','packed')]:
        name = domain+'-'+backend
        fixture = output/(domain+'-input.txt')
        if not fixture.exists():
            source = ROOT/'tests/metal/fixtures'/('strassen_3x3_f2.txt' if domain=='F2' else 'strassen_3x3.txt')
            shutil.copy2(source, fixture)
        config = deepcopy(base); config['policy']['domain']=domain
        config['input']['files']=[dict(path=str(fixture),format='cpu-text',domain=domain)]
        config['execution']['backend']=backend; config['history']['path']=str(output/(name+'-history'))
        program='flip_graph_f2' if domain=='F2' else 'flip_graph'
        result = execute(name,program,config)
        require(result['actual_backend']==backend, 'wrong controlled backend')
        require(result['counters']['flip_attempts']==6, 'controlled lifetime budget mismatch')
        if name=='ZT-general': previous=result
        if backend=='packed':
            require(previous['counters']==result['counters'], 'packaged general/packed counters differ')
            require(previous['workers']==result['workers'], 'packaged general/packed final workers differ')
        resume = deepcopy(config);resume['input']=dict(kind='resume',journal=config['history']['path'])
        resume['policy']['seed']=11;resume['policy']['flip_budget']=2
        resumed = execute(name+'-resume',program,resume)
        require(resumed['counters']['discoveries_historical']==result['counters']['discoveries_historical']+resumed['counters']['discoveries_current_run'],
                'resume discovery accounting reset or duplicated')
        resume['discovery_target']=resumed['counters']['discoveries_historical']
        stopped=execute(name+'-history-target',program,resume)
        require(stopped['terminal_reason']=='discovery_target_met' and stopped['counters']['flip_attempts']==0,
                'committed historical target did not stop before walking')
        corpus=output/(name+'-corpus.jsonl')
        exported=run(['/usr/bin/env','PATH='+str(empty_path),str(binaries/'scheme_tool'),'verify',
            '--input',config['history']['path'],'--format','journal','--output',str(corpus)],output/(name+'-corpus-guard'))
        require(exported['complete'],'native read-only journal corpus access failed')
        schemes=[json.loads(line) for line in corpus.read_text().splitlines()]
        for scheme in schemes:
            verify(dict(n=scheme['dimensions'],m=scheme['rank'],z2=scheme['domain']=='F2',**{key:scheme[key] for key in 'uvw'}))
        require(sum(s['rank']==26 for s in schemes)==stopped['counters']['discoveries_historical']+1,
                'corpus membership differs from committed discoveries and imported seed')
    rank_general=None
    for backend in ('general','packed'):
        config=deepcopy(base); config['policy'].pop('collection_rank')
        config['policy'].update(mode='rank-reduction',stage_rank=26)
        config['input']['files']=[dict(path=str(output/'ZT-input.txt'),format='cpu-text',domain='ZT')]
        config['execution']['backend']=backend
        config['history']['path']=str(output/('rank-'+backend+'-history'))
        result=execute('rank-'+backend,'flip_graph',config)
        if backend=='general': rank_general=result
        else:
            require(rank_general['workers']==result['workers'] and rank_general['counters']==result['counters'],
                    'packaged rank-reduction general/packed state differs')
    for flips in (0,3):
        config=dict(schema='fgm-run-v1',operation='reduce',
            reduction=dict(domain='ZT',seed=7,rounds=3,reducers=16,schemes=8,max_flips=flips,no_improvements=2,target_additions=0),
            input=dict(kind='files',files=[dict(path=str(output/'ZT-input.txt'),format='cpu-text',domain='ZT')]),
            execution=dict(workers=8,batch_steps=1,block_size=8,backend='general',memory_bytes=536870912))
        record=execute('reduction-'+str(flips),'additions_reducer',config)
        artifact=record['circuit_artifact'];artifact_path=Path(artifact['path'])
        require(hashlib.sha256(artifact_path.read_bytes()).hexdigest()==artifact['sha256'],'circuit artifact hash mismatch')
        circuits=[json.loads(line) for line in artifact_path.read_text().splitlines()]
        require(len(circuits)==artifact['records']==len(record['results']),'circuit artifact record mismatch')
        for circuit in circuits:
            verify(circuit)
        native_output=output/('reduction-'+str(flips)+'-native-verified.jsonl')
        result=run(['/usr/bin/env','PATH='+str(empty_path),str(binaries/'scheme_tool'),'verify','--format','jsonl',
                    '--input',str(artifact_path),'--output',str(native_output)],output/('reduction-'+str(flips)+'-verify-guard'))
        require(result['complete'],'native circuit sidecar verification failed')
        if flips:
            require(record['counters']['flips_applied']>0,'flip-enabled reducer did not apply a mutation')
        else:
            require(record['counters']['flips_applied']==0,'fixed reducer reported mutations')
    summary=dict(complete=True,python_available=False,runs=records)
    write_json(output/'results.json',summary)
    return summary


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
        host_link = cwd / 'scheme-tool-link'
        host_link.symlink_to(relocated / 'scheme_tool')
        summary['native_host_checks'] = native_host_checks(
            host_link, output / 'native-host-no-python', cwd)
        summary['native_workflow_checks'] = native_workflow_checks(
            relocated, output / 'native-workflow-no-python', cwd)
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
