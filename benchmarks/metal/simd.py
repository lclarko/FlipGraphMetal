import argparse
import json
from pathlib import Path

from storage import generate as generate_storage


def candidate_function(original):
    start = original.index('bool SchemeInteger::tryFlip(')
    end = original.index('bool SchemeInteger::tryPlus(', start)
    method = original[start:end]
    start = method.index('        if (index < flips[0].size)')
    end = method.index('\n    }\n\n    if (p == size)')
    body = method[start:end]
    if body.count('if (randomWord(&state) % 2)') != 2 or body.count('break;') != 2:
        raise ValueError('candidate orientation or selection changed')
    body = body.replace('if (randomWord(&state) % 2)', 'if (orientation & 1)', 1)
    body = body.replace('if (randomWord(&state) % 2)', 'if (orientation & 2)', 1)
    packed = 'return uint(i) | (uint(j) << 2) | (uint(k) << 4) | (uint(index1) << 6) | (uint(index2) << 15);'
    body = body.replace('break;', packed)
    body = body.replace('flips[', 'scheme.storage->flips[').replace('uvw[', 'scheme.storage->uvw[')
    return '''uint simdCandidate(thread StorageScheme &scheme, int index, uint orientation) {
    int i = 0, j = 0, k = 0, index1 = 0, index2 = 0;
''' + body + '\n    return 0xffffffffu;\n}\n'


SELECTION = '''
bool simdTryFlip(thread StorageScheme &scheme, thread RandomState &state, device uint *choices, uint lane) {
    int size = scheme.storage->flips[0].size + scheme.storage->flips[1].size + scheme.storage->flips[2].size;
    if (lane == 0) randomPermutation(scheme.scratch, size, state);
    simdgroup_barrier(mem_flags::mem_device);
    for (int begin = 0; begin < size; begin += 32) {
        uint count = uint(min(32, size - begin));
        if (lane == 0) {
            RandomState preview = state;
            for (uint p = 0; p < count; p++) {
                uint orientation = randomWord(&preview) % 2;
                orientation |= (randomWord(&preview) % 2) << 1;
                choices[p] = orientation;
                choices[32 + p] = preview.value;
            }
        }
        simdgroup_barrier(mem_flags::mem_device);
        uint operation = lane < count ? simdCandidate(scheme, scheme.scratch[begin + lane], choices[lane]) : 0xffffffffu;
        uint winner = simd_min(operation == 0xffffffffu ? 32u : lane);
        uint selected = simd_shuffle(operation, min(winner, 31u));
        simdgroup_barrier(mem_flags::mem_device);
        if (lane == 0) {
            state.value = choices[32 + (winner < 32 ? winner : count - 1)];
            if (winner < 32)
                scheme.flip(selected & 3, (selected >> 2) & 3, (selected >> 4) & 3,
                            (selected >> 6) & 511, (selected >> 15) & 511, true);
        }
        simdgroup_barrier(mem_flags::mem_device);
        if (winner < 32) return true;
    }
    return false;
}
'''


def generate(source, output):
    original = (source / 'scheme_integer.h').read_text()
    selection = candidate_function(original)
    generate_storage(source, output, 'device')
    path = output / 'kernels.metal'
    kernels = path.read_text()
    start = kernels.index('kernel void randomWalkKernel(')
    end = kernels.index('kernel void resizeKernel(', start)
    walk = kernels[start:end]
    signature = 'uint idx [[thread_position_in_grid]]) {'
    if walk.count(signature) != 1:
        raise ValueError('walk invocation changed')
    walk = walk.replace(signature, '''uint threadIndex [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]], uint groupSize [[threads_per_threadgroup]]) {
    uint idx = threadIndex / 32;''')
    walk = walk.replace('device int *scratch [[buffer(13)]]', 'device int *scratch [[buffer(13)]], device uint *selectionStates [[buffer(14)]]')
    walk = walk.replace('    if (idx >= uint(schemesCount)) return;', '''    if (idx >= uint(schemesCount)) return;
    if (width != 32 || groupSize != 32) { errors[idx] = 2; return; }''')
    walk = walk.replace('{ errors[idx] = 2; return; }', '{ if (lane == 0) errors[idx] = 2; return; }')
    loop_start = walk.index('    for (int iteration = 0; iteration < iterations; iteration++) {')
    success_start = walk.index('        if (scheme.storage->m < rank)', loop_start)
    loop_end = walk.index('\n    }\n\n    flips[idx]', success_start)
    success = walk[success_start:loop_end]
    tail = walk[loop_end + len('\n    }\n'):]
    closing = tail.rfind('}')
    walk = walk[:loop_start] + '''    device uint *choices = selectionStates + idx * 64;
    for (int iteration = 0; iteration < iterations; iteration++) {
        int rank = scheme.storage->m;
        bool flipped = simdTryFlip(scheme, state, choices, lane);
        if (!flipped) {
            if (lane == 0) scheme.tryExpand(randint(1, 2, state), state);
            simdgroup_barrier(mem_flags::mem_device);
            continue;
        }
        if (lane == 0) {
''' + success + '''
        }
        simdgroup_barrier(mem_flags::mem_device);
    }
    if (lane == 0) {
''' + tail[:closing] + '\n    }\n}\n\n'
    path.write_text(kernels[:start] + selection + SELECTION + walk + kernels[end:])
    path = output / 'storage.json'
    manifest = json.loads(path.read_text())
    manifest.update(mode='simd', variant='simd', threads_per_walk=32,
                    buffer_bytes_per_worker={'13': 6000, '14': 256},
                    eligibility=manifest['eligibility'] + ', SIMD width 32 and threadgroup size 32')
    path.write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate a diagnostic SIMD-cooperative signed 3x3 Metal walk')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.source.resolve(strict=True), args.output.resolve())


if __name__ == '__main__':
    main()
