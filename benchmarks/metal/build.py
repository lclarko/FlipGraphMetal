import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
from application import source_identity
parser = argparse.ArgumentParser(description="Build the signed CPU/Metal profiling executables")
parser.add_argument('--rank-capacity', type=int, choices=[32, 350], default=350)
parser.add_argument('--output', type=Path, help='directory for source snapshots and executables')
parser.add_argument('--source', type=Path, help='Metal source directory for the CPU reference and diagnostics')
parser.add_argument('--gpu-source', type=Path, help='Metal source directory for the production GPU comparison')
parser.add_argument('--gpu-kernel', choices=['randomWalkKernel', 'randomWalkCompactKernel'], default='randomWalkKernel')
args = parser.parse_args()
if args.gpu_kernel == "randomWalkCompactKernel" and args.rank_capacity != 350:
    parser.error("compact walks require rank capacity 350")

root = Path(__file__).resolve().parents[2]
output = args.output.resolve() if args.output else root / ('build/metal/profile' if args.rank_capacity == 350 else 'build/metal/profile-cap32')
production = output / 'production'
source = output / 'source'
input_source = args.source.resolve() if args.source else root / 'src/metal'
gpu_source = args.gpu_source.resolve() if args.gpu_source else input_source
if not input_source.is_dir() or not gpu_source.is_dir():
    parser.error('source directories must exist')
if input_source in (source, production) or gpu_source in (source, production):
    parser.error('input source directories must differ from output snapshots')
if args.rank_capacity != 350 and (gpu_source / 'storage.json').is_file():
    parser.error('storage diagnostics require rank capacity 350')
output.mkdir(parents=True, exist_ok=False)
production.mkdir()
source.mkdir()
(output / 'build.json').write_text(json.dumps({'complete': False, 'gpu_kernel': args.gpu_kernel, 'rank_capacity': args.rank_capacity, 'source': str(input_source), 'gpu_source': str(gpu_source)}, indent=2) + '\n')
inputs = list(input_source.glob('*'))
gpu_inputs = list(gpu_source.glob('*'))
for path in inputs:
    if path.is_file():
        shutil.copy2(path, source / path.name)
for path in gpu_inputs:
    if path.is_file():
        shutil.copy2(path, production / path.name)
for name in ['build.py', 'profile.cpp', 'kernels.metal']:
    shutil.copy2(root / 'benchmarks/metal' / name, output / name)
runtime_input = gpu_source / 'runtime.mm'
runtime = runtime_input.read_text()
# Match the loader to the GPU snapshot, including headers it compiles eagerly.
if '@"compact.h"' in runtime:
    shutil.copy2(gpu_source / 'compact.h', source / 'compact.h')
if args.gpu_kernel == 'randomWalkCompactKernel':
    benchmark = output / 'profile.cpp'
    benchmark.write_text(benchmark.read_text().replace('"randomWalkKernel"', '"randomWalkCompactKernel"'))

creation_check = 'if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);'
if runtime.count(creation_check) != 1:
    raise RuntimeError('pipeline creation check changed')
runtime = runtime.replace(creation_check, creation_check + '''
            std::cout << "PIPELINE " << name << " SIMD_WIDTH " << pipeline.threadExecutionWidth
                      << " MAX_THREADS " << pipeline.maxTotalThreadsPerThreadgroup
                      << " STATIC_THREADGROUP_BYTES " << pipeline.staticThreadgroupMemoryLength << std::endl;''')
storage = json.loads((gpu_source / 'storage.json').read_text()) if (gpu_source / 'storage.json').is_file() else None
if storage:
    padding = storage['padded_multiple']
    lanes = storage.get('threads_per_walk', 1)
    if lanes not in (1, 32):
        raise RuntimeError('unexpected lanes per walk')
    buffers = {int(index): size for index, size in storage['buffer_bytes_per_worker'].items()}
    if padding not in (1, 32) or not buffers or any(index not in (13, 14, 15) or size <= 0 for index, size in buffers.items()):
        raise RuntimeError('unexpected storage allocation contract')
    fields = ''.join(f'\n    id<MTLBuffer> storage{index};' for index in buffers)
    runtime = runtime.replace('    id<MTLLibrary> library;', '    id<MTLLibrary> library;' + fields)
    binding = '        [encoder setComputePipelineState:pipeline];'
    if runtime.count(binding) != 1:
        raise RuntimeError('pipeline binding changed')
    allocation = '\n        if (std::string(name) == "randomWalkKernel") {\n'
    allocation += f'            size_t workers = (threads + {padding-1}) / {padding} * {padding};\n'
    for index, size in buffers.items():
        allocation += f'''            {{
                size_t length = workers * {size};
                if (length > r.device.maxBufferLength) throw std::runtime_error("Metal: unsupported storage buffer size");
                if (!r.storage{index} || r.storage{index}.length < length)
                    r.storage{index} = [r.device newBufferWithLength:length options:MTLResourceStorageModeShared];
                if (!r.storage{index}) throw std::runtime_error("Metal: storage buffer allocation failed");
                [encoder setBuffer:r.storage{index} offset:0 atIndex:{index}];
            }}
'''
    runtime = runtime.replace(binding, binding + allocation + '        }')
    if lanes == 32:
        dispatch_line = '        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];'
        if runtime.count(dispatch_line) != 1:
            raise RuntimeError('dispatch invocation changed')
        runtime = runtime.replace(dispatch_line, '\n'.join([
            '        size_t physicalThreads = threads;',
            '        if (std::string(name) == "randomWalkKernel") {',
            '            if (blockSize != 32 || pipeline.threadExecutionWidth != 32)',
            '                throw std::runtime_error("SIMD diagnostic requires a 32-lane group");',
            '            physicalThreads *= 32;',
            '        }',
            dispatch_line.replace('MTLSizeMake(threads,', 'MTLSizeMake(physicalThreads,')]))

(source / 'runtime.mm').write_text(runtime)
if args.rank_capacity != 350:
    for directory in [source, production]:
        path = directory / 'core.h'
        contents = path.read_text()
        original = 'constant constexpr int MAX_RANK = 350;'
        if contents.count(original) != 1:
            raise RuntimeError('rank-capacity declaration changed')
        path.write_text(contents.replace(original, 'constant constexpr int MAX_RANK = 32;'))
kernels = (source / 'kernels.metal').read_text()
start = kernels.index('kernel void randomWalkKernel(')
end = kernels.index('kernel void resizeKernel(', start)
trace = kernels[start:end].replace('randomWalkKernel', 'profileTraceKernel', 1)
trace = trace.replace('device int *errors [[buffer(30)]]', 'device uint *trace, device int *errors [[buffer(30)]]', 1)
trace = trace.replace('    for (int iteration = 0; iteration < iterations; iteration++) {',
    '    uint totalPairs = 0, maxPairs = 0;\n    for (int iteration = 0; iteration < iterations; iteration++) {\n        uint pairs = scheme.flips[0].size + scheme.flips[1].size + scheme.flips[2].size;\n        totalPairs += pairs;\n        if (pairs > maxPairs) maxPairs = pairs;')
trace = trace.replace('    errors[idx] = !scheme.validate();', '    trace[2*idx] = totalPairs;\n    trace[2*idx+1] = maxPairs;\n    errors[idx] = !scheme.validate();')
(source / 'profile_trace.h').write_text(trace)
walk = kernels[start:end].replace('randomWalkKernel', 'profileWalkKernel', 1)
walk = walk.replace('errors[idx] = !scheme.validate();', 'errors[idx] = 0;')
scheme = (source / 'scheme_integer.h').read_text()
flip = scheme[scheme.index('bool SchemeInteger::tryFlip('):scheme.index('bool SchemeInteger::tryPlus(')]
narrow = flip.replace('SchemeInteger::tryFlip(', 'SchemeInteger::tryFlipNarrow(', 1)
if narrow.count('int indices[MAX_PAIRS * 3];') != 1:
    raise RuntimeError('permutation scratch declaration changed')
narrow = narrow.replace('int indices[MAX_PAIRS * 3];', 'uint16_t indices[MAX_PAIRS * 3];')
flip_declaration = '    bool tryFlip(LOCAL RandomState &state, bool checkReduce = true) LOCAL_METHOD;'
if scheme.count(flip_declaration) != 1:
    raise RuntimeError('tryFlip declaration changed')
scheme = scheme.replace(flip_declaration, flip_declaration + '\n' + flip_declaration.replace('tryFlip(', 'tryFlipNarrow(')) + '\n' + narrow
core = (source / 'core.h').read_text()
permutation = core[core.index('inline void randomPermutation('):core.index('inline void randomMatrixZ2(')]
if permutation.count('LOCAL int *array') != 1:
    raise RuntimeError('permutation argument changed')
(source / 'core.h').write_text(core + '\n' + permutation.replace('LOCAL int *array', 'LOCAL uint16_t *array'))
declaration = '    void flip(int i, int j, int k, int index1, int index2, bool checkReduce) LOCAL_METHOD;'
if scheme.count(declaration) != 1:
    raise RuntimeError('flip declaration changed')
(source / 'scheme_integer.h').write_text(scheme.replace(declaration, 'public:\n' + declaration + '\nprivate:'))
selection = scheme[scheme.index('bool SchemeInteger::tryFlip('):scheme.index('bool SchemeInteger::tryPlus(')]
selection = selection[selection.index('{'):]
selection = selection.replace('flips[', 'scheme.flips[').replace('uvw[', 'scheme.uvw[')
def selected(match):
    i, j, k, first, second, check = [word.strip() for word in match[1].split(',')]
    return f'return uint({i}) | (uint({j}) << 2) | (uint({k}) << 4) | (uint({first}) << 6) | (uint({second}) << 15);'
selection, changes = re.subn(r'\bflip\(([^;]+)\);\s*return true;', selected, selection)
if changes not in (1, 4):
    raise RuntimeError('flip selection implementation changed')
selection = 'uint profileSelect(thread const Scheme &scheme, thread RandomState &state) ' + selection.replace('return false;', 'return 0xffffffffu;')
decision = 'struct ProfileDecision { uint operation; uint state; };\n'
record = kernels[start:end].replace('randomWalkKernel', 'profileRecordKernel', 1)
record = record.replace('device int *errors [[buffer(30)]]', 'device ProfileDecision *decisions [[buffer(13)]], device int *errors [[buffer(30)]]', 1)
original = '        if (!scheme.tryFlip(state)) {'
if record.count(original) != 1:
    raise RuntimeError('walk selection call changed')
record = record.replace(original, '''        uint operation = profileSelect(scheme, state);
        decisions[iteration * schemesCount + idx] = {operation, state.value};
        if (operation != 0xffffffffu)
            scheme.flip(operation & 3, (operation >> 2) & 3, (operation >> 4) & 3, (operation >> 6) & 511, (operation >> 15) & 511, true);
        if (operation == 0xffffffffu) {''')
(source / 'profile_record.h').write_text(record)
selected_walk = record.replace('profileRecordKernel', 'profileSelectedKernel', 1)
selected_walk = selected_walk.replace('device ProfileDecision *decisions [[buffer(13)]], ', '', 1)
selected_walk = selected_walk.replace('        decisions[iteration * schemesCount + idx] = {operation, state.value};\n', '', 1)
replay = record.replace('profileRecordKernel', 'profileReplayKernel', 1)
replay = replay.replace('device ProfileDecision *decisions', 'device const ProfileDecision *decisions', 1)
replay = replay.replace('        uint operation = profileSelect(scheme, state);\n        decisions[iteration * schemesCount + idx] = {operation, state.value};',
    '        uint operation = decisions[iteration * schemesCount + idx].operation;\n        state.value = decisions[iteration * schemesCount + idx].state;')
narrow_walk = kernels[start:end].replace('randomWalkKernel', 'profileNarrowKernel', 1).replace('scheme.tryFlip(state)', 'scheme.tryFlipNarrow(state)')
if args.gpu_kernel == 'randomWalkCompactKernel':
    compact_kernels = (gpu_source / 'kernels.metal').read_text()
    compact_start = compact_kernels.index('kernel void randomWalkCompactKernel(')
    compact_end = compact_kernels.index('\n}\n', compact_start) + 3
    selected_compact = compact_kernels[compact_start:compact_end]
    if 'kernel void randomWalkCompactKernel(' in kernels:
        previous_start = kernels.index('kernel void randomWalkCompactKernel(')
        previous_end = kernels.index('\n}\n', previous_start) + 3
        kernels = kernels[:previous_start] + selected_compact + kernels[previous_end:]
    else:
        kernels += '\n#if defined(__METAL_VERSION__) && !defined(METAL_F2)\n' + selected_compact + '\n#endif\n'
(source / 'kernels.metal').write_text(kernels + '\n' + walk + selection + decision + selected_walk + replay + narrow_walk + (root / 'benchmarks/metal/kernels.metal').read_text())
commands = [
    ['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off', '-Wno-unknown-attributes', '-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include', '-I'+str(source), '-c', str(output / 'profile.cpp'), '-o', str(output / 'profile.o')],
    ['xcrun', 'clang++', '-std=c++17', '-O2', '-fobjc-arc', '-ffp-contract=off', '-framework', 'Foundation', '-framework', 'Metal', '-DMETAL_SOURCE_DIR="'+str(source)+'"', str(output / 'profile.o'), str(source / 'runtime.mm'), '-L/opt/homebrew/opt/libomp/lib', '-lomp', '-o', str(output / 'profile')],
]
matched = commands[-1].copy()
matched[matched.index('-DMETAL_SOURCE_DIR="'+str(source)+'"')] = '-DMETAL_SOURCE_DIR="'+str(production)+'"'
matched[-1] = str(output / 'matched')
commands.append(matched)
manifest = {'complete': False, 'gpu_kernel': args.gpu_kernel, 'runtime_input': str(runtime_input), 'rank_capacity': args.rank_capacity, 'source': str(input_source), 'gpu_source': str(gpu_source), 'commands': commands, 'inputs': {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*inputs, *gpu_inputs, *Path(__file__).parent.glob('*')] if p.is_file()}}
(output / 'build.json').write_text(json.dumps(manifest, indent=2)+'\n')

for command in commands:
    subprocess.run(command, cwd=root, check=True)
manifest['snapshot_files'] = source_identity(output)
manifest['complete'] = True
manifest['binary_sha256'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ('profile', 'matched')}
(output / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
