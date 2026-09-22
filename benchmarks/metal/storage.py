import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def definitions(text, cls):
    found = []
    pattern = re.compile(r'^([^\n]*\b' + cls + r'::[^\n]+) \{', re.M)
    for match in pattern.finditer(text):
        depth = 1
        end = match.end()
        while depth:
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        found.append((match[1], text[match.end():end - 1]))
    return found


def address_overloads(text, cls):
    text = text.replace('LOCAL const Addition &addition', 'Addition addition')
    additions = []
    for signature, body in definitions(text, cls):
        if 'LOCAL_METHOD' not in signature or f'{cls}::{cls}(' in signature:
            continue
        if 'operator[]' in signature:
            continue
        new = signature.replace('LOCAL_METHOD', 'device')
        if new.startswith('LOCAL Addition &'):
            new = new.replace('LOCAL Addition &', 'device Addition &', 1)
        additions.append((new, body))
        if 'LOCAL Addition &target' in signature:
            additions.append((signature.replace('LOCAL Addition &target', 'device Addition &target'), body))
            additions.append((new.replace('LOCAL Addition &target', 'device Addition &target'), body))
    declarations = '\n'.join('    ' + signature.replace(cls + '::', '') + ';' for signature, _ in additions)
    declarations = declarations.replace('limitSub(Addition addition, bool firstPositiveNonZero)', 'limitSub(Addition addition, bool firstPositiveNonZero = false)')
    end = text.index('\n};')
    text = text[:end] + '\n#ifdef __METAL_VERSION__\n' + declarations + '\n#endif' + text[end:]
    text += '\n#ifdef __METAL_VERSION__\n' + '\n'.join(signature + ' {' + body + '}\n' for signature, body in additions) + '#endif\n'
    return text


def interleave_snapshot(output):
    path = output / 'scheme_integer.h'
    source = path.read_text()
    begin = source.index('struct StorageScheme {')
    view = source[begin:]
    view = re.sub(r'storage->uvw\[([^\]]+)\]\[([^\]]+)\]', r'term(\1, \2)', view)
    view = view.replace('storage->flips', 'flips')
    view = re.sub(r'indices\[([^\]]+)\]', r'indices[(\1) * 32]', view)
    view = view.replace('randomPermutation(indices, size, state)', 'randomPermutationStrided(indices, size, state)')
    view = view.replace('    device SchemeInteger *storage;', '    device SchemeInteger *storage;\n    device Addition *terms;\n    StorageFlipSet flips[3];')
    declaration = '    device Addition &term(int p, int r) const thread { return terms[(p * MAX_RANK + r) * 32]; }\n'
    view = view.replace('struct StorageScheme {\n', 'struct StorageScheme {\n' + declaration)
    pairs = (output / 'flip_set.h').read_text()
    pair_methods = []
    for signature, body in definitions(pairs, 'FlipSet'):
        if 'LOCAL_METHOD' not in signature or 'FlipSet::FlipSet(' in signature:
            continue
        signature = signature.replace('FlipSet::', 'StorageFlipSet::')
        body = re.sub(r'pairs\[([^\]]+)\]', r'pairs[(\1) * 32]', body)
        pair_methods.append((signature, body))
    pair_view = 'static_assert(sizeof(Addition) == 32, \"interleaved Addition stride changed\");\nstruct StorageFlipSet {\n    size_t size;\n    device uint32_t *pairs;\n'
    pair_view += '\n'.join('    ' + signature.replace('StorageFlipSet::', '') + ';' for signature, _ in pair_methods) + '\n};\n'
    pair_view += '\n'.join(signature + ' {' + body + '}\n' for signature, body in pair_methods)
    path.write_text(source[:begin] + pair_view + view)
    path = output / 'core.h'
    core = path.read_text()
    permutation = core[core.index('inline void randomPermutation('):core.index('inline void randomMatrixZ2(')]
    permutation = permutation.replace('randomPermutation(', 'randomPermutationStrided(').replace('LOCAL int *array', 'device int *array')
    permutation = re.sub(r'array\[([^\]]+)\]', r'array[(\1) * 32]', permutation)
    path.write_text(core + '\n#ifdef __METAL_VERSION__\n' + permutation + '#endif\n')
    path = output / 'kernels.metal'
    kernels = path.read_text()
    start, end = kernels.index('kernel void randomWalkKernel('), kernels.index('kernel void resizeKernel(')
    walk = kernels[start:end]
    walk = walk.replace('device int *scratch [[buffer(13)]]', 'device int *scratch [[buffer(13)]], device Addition *terms [[buffer(14)]], device uint32_t *pairs [[buffer(15)]]')
    old = '    scheme.scratch = scratch + idx * (MAX_PAIRS * 3);'
    setup = """    uint tile = idx / 32, lane = idx % 32;
    scheme.scratch = scratch + tile * (MAX_PAIRS * 3 * 32) + lane;
    scheme.terms = terms + tile * (MAX_RANK * 3 * 32) + lane;
    for (int p = 0; p < 3; p++) {
        scheme.flips[p].pairs = pairs + tile * (MAX_PAIRS * 3 * 32) + p * MAX_PAIRS * 32 + lane;
        scheme.flips[p].size = schemes[idx].flips[p].size;
        for (size_t pair = 0; pair < scheme.flips[p].size; pair++)
            scheme.flips[p].pairs[pair * 32] = schemes[idx].flips[p].pairs[pair];
        for (int r = 0; r < scheme.m; r++) {
            scheme.term(p, r).n = schemes[idx].uvw[p][r].n;
            scheme.term(p, r).values = schemes[idx].uvw[p][r].values;
            scheme.term(p, r).signs = schemes[idx].uvw[p][r].signs;
            scheme.term(p, r).valid = schemes[idx].uvw[p][r].valid;
        }
    }"""
    if walk.count(old) != 1:
        raise RuntimeError('hybrid scratch initialization changed')
    walk = walk.replace(old, setup)
    copy_start = walk.index('            schemesBest[idx].m =')
    copy_end = walk.index('            savedRank =', copy_start)
    copy = walk[copy_start:copy_end]
    copy = re.sub(r'schemes\[idx\].uvw\[([^\]]+)\]\[([^\]]+)\]', r'scheme.term(\1, \2)', copy)
    copy = copy.replace('schemes[idx].n[p]', 'scheme.n[p]').replace('schemes[idx].nn[p]', 'scheme.nn[p]')
    copy = copy.replace('schemes[idx].flips[p].size', 'scheme.flips[p].size')
    copy = copy.replace('schemes[idx].flips[p].pairs[pair]', 'scheme.flips[p].pairs[pair * 32]')
    walk = walk[:copy_start] + copy + walk[copy_end:]
    final = '\n'.join(line[8:] if line.startswith('        ') else line for line in copy.replace('schemesBest[idx]', 'schemes[idx]').split('\n'))
    walk = walk.replace('    schemes[idx].m = scheme.m;', final)
    path.write_text(kernels[:start] + walk + kernels[end:])


def compact_snapshot(output):
    path = output / 'addition.h'
    addition = path.read_text()
    compact = re.sub(r'\bAddition\b', 'CompactAddition', addition)
    compact = re.sub(r'\bT\b', 'ushort', compact)
    path.write_text(addition + '\n#ifdef __METAL_VERSION__\n' + compact + '\nstatic_assert(sizeof(CompactAddition) == 12, "compact Addition stride changed");\n#endif\n')
    path = output / 'scheme_integer.h'
    scheme = path.read_text()
    begin = scheme.index('struct StorageScheme {')
    scheme = scheme[:begin] + re.sub(r'\bAddition\b', 'CompactAddition', scheme[begin:])
    path.write_text(scheme)
    path = output / 'kernels.metal'
    kernels = path.read_text()
    start, end = kernels.index('kernel void randomWalkKernel('), kernels.index('kernel void resizeKernel(')
    walk = kernels[start:end]
    walk = walk.replace('device Addition *terms [[buffer(14)]]', 'device CompactAddition *terms [[buffer(14)]]')
    guard = """    if (schemes[idx].m < 1 || schemes[idx].m > MAX_RANK) { errors[idx] = 3; return; }
    for (int p = 0; p < 3; p++) {
        if (schemes[idx].nn[p] != 9 || schemes[idx].flips[p].size > MAX_PAIRS) { errors[idx] = 3; return; }
        for (int r = 0; r < schemes[idx].m; r++) {
            device const Addition &term = schemes[idx].uvw[p][r];
            if (term.n != 9 || (term.values & ~T(511)) || (term.signs & ~T(511))) { errors[idx] = 3; return; }
        }
    }
"""
    walk = walk.replace('    StorageScheme scheme;', guard + '    StorageScheme scheme;')
    walk = walk.replace('scheme.term(p, r).values = schemes[idx].uvw[p][r].values;', 'scheme.term(p, r).values = ushort(schemes[idx].uvw[p][r].values);')
    walk = walk.replace('scheme.term(p, r).signs = schemes[idx].uvw[p][r].signs;', 'scheme.term(p, r).signs = ushort(schemes[idx].uvw[p][r].signs);')
    path.write_text(kernels[:start] + walk + kernels[end:])


def generate(source, output, mode):
    requested_mode = mode
    if mode in ('interleaved', 'interleaved16'):
        mode = 'hybrid'
    core_source = (source / 'core.h').read_text()
    for declaration in ['typedef uint64_t T;', 'constant constexpr int MAX_RANK = 350;', 'constant constexpr int MAX_PAIRS = 500;']:
        if declaration not in core_source:
            raise RuntimeError('storage diagnostics require the original widths and capacities')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, cls in [('addition.h', 'Addition'), ('flip_set.h', 'FlipSet')]:
        path = output / name
        path.write_text(address_overloads(path.read_text(), cls))
    original = (source / 'scheme_integer.h').read_text()
    names = {'validate', 'validateEquation', 'initFlips', 'removeZeroes', 'removeAt', 'addTriplet',
             'fixSigns', 'flip', 'plus', 'split', 'reduceAdd', 'reduceSub', 'tryFlip', 'tryPlus',
             'trySplit', 'trySplitExisted', 'tryExpand', 'tryReduce'}
    methods = []
    for signature, body in definitions(original, 'SchemeInteger'):
        name = re.search(r'SchemeInteger::(\w+)', signature)[1]
        if name not in names:
            continue
        signature = signature.replace('SchemeInteger::', 'StorageScheme::')
        signature = re.sub(r'LOCAL const Addition &\s*(\w+)', r'Addition \1', signature)
        for field in ['uvw', 'flips'] + (['n', 'nn', 'm'] if mode == 'device' else []):
            body = re.sub(r'\b' + field + r'\b', 'storage->' + field, body)
        if name == 'tryFlip':
            body = body.replace('int indices[MAX_PAIRS * 3];', 'device int *indices = scratch;')
        methods.append((signature, body))
    if len(methods) != len(names):
        raise RuntimeError('required scheme methods changed')
    fields = '    device SchemeInteger *storage;\n    device int *scratch;\n'
    if mode == 'hybrid':
        fields += '    int n[3], nn[3], m;\n'
    declarations = '\n'.join('    ' + signature.replace('StorageScheme::', '') + ';' for signature, _ in methods)
    declarations = declarations.replace('bool tryFlip(LOCAL RandomState &state, bool checkReduce)', 'bool tryFlip(LOCAL RandomState &state, bool checkReduce = true)')
    generated = '#ifdef __METAL_VERSION__\n#ifdef METAL_F2\n#error Storage diagnostics support signed ternary only\n#endif\nstruct StorageScheme {\n' + fields + declarations + '\n};\n'
    generated += '\n'.join(signature + ' {' + body + '}\n' for signature, body in methods) + '#endif\n'
    core = (output / 'core.h').read_text()
    permutation = core[core.index('inline void randomPermutation('):core.index('inline void randomMatrixZ2(')]
    core += '\n#ifdef __METAL_VERSION__\n' + permutation.replace('LOCAL int *array', 'device int *array') + '#endif\n'
    (output / 'core.h').write_text(core)
    scheme_path = output / 'scheme_integer.h'
    scheme_path.write_text(scheme_path.read_text() + '\n' + generated)
    kernels = (output / 'kernels.metal').read_text()
    start = kernels.index('kernel void randomWalkKernel(')
    end = kernels.index('kernel void resizeKernel(', start)
    for pattern in ['    Scheme scheme;', '    loadObject(scheme, schemes + idx);', '            storeObject(schemesBest + idx, scheme);', '    storeObject(schemes + idx, scheme);', '        if (randomUniform(&state) * maxIterations < basisProbability)\n            scheme.swapBasis(state);', '        if (flipsCount >= plusIterations && scheme.tryPlus(state))\n            flipsCount = 0;']:
        if kernels[start:end].count(pattern) != 1:
            raise RuntimeError('random walk storage transformation no longer matches: ' + pattern)
    walk = kernels[start:end].replace('device int *errors [[buffer(30)]]', 'device int *scratch [[buffer(13)]], device int *errors [[buffer(30)]]')
    guard = '''    if (maxIterations != 1000 || plusIterations != 1000000000 || randomIterations ||
        reduceProbability != 0 || expandProbability != 0 || sandwichingProbability != 0 || basisProbability != 0 ||
        schemes[idx].n[0] != 3 || schemes[idx].n[1] != 3 || schemes[idx].n[2] != 3 ||
        flips[idx] >= plusIterations - maxIterations) { errors[idx] = 2; return; }
'''
    walk = walk.replace('    Scheme scheme;', guard + '    StorageScheme scheme;\n    scheme.storage = schemes + idx;')
    setup = '    scheme.scratch = scratch + idx * (MAX_PAIRS * 3);'
    finish = ''
    rank = 'scheme.storage->m'
    if mode == 'hybrid':
        setup = '''    for (int i = 0; i < 3; i++) { scheme.n[i] = schemes[idx].n[i]; scheme.nn[i] = schemes[idx].nn[i]; }
    scheme.m = schemes[idx].m;
    scheme.scratch = scratch + idx * (MAX_PAIRS * 3);'''
        finish = '    schemes[idx].m = scheme.m;'
        rank = 'scheme.m'
    walk = walk.replace('    loadObject(scheme, schemes + idx);', setup)
    walk = walk.replace('scheme.m', rank) if mode == 'device' else walk
    copy = f"""            schemesBest[idx].m = {rank};
            for (int p = 0; p < 3; p++) {{
                schemesBest[idx].n[p] = schemes[idx].n[p];
                schemesBest[idx].nn[p] = schemes[idx].nn[p];
                for (int r = 0; r < {rank}; r++) {{
                    schemesBest[idx].uvw[p][r].n = schemes[idx].uvw[p][r].n;
                    schemesBest[idx].uvw[p][r].values = schemes[idx].uvw[p][r].values;
                    schemesBest[idx].uvw[p][r].signs = schemes[idx].uvw[p][r].signs;
                    schemesBest[idx].uvw[p][r].valid = schemes[idx].uvw[p][r].valid;
                }}
                schemesBest[idx].flips[p].size = schemes[idx].flips[p].size;
                for (size_t pair = 0; pair < schemes[idx].flips[p].size; pair++)
                    schemesBest[idx].flips[p].pairs[pair] = schemes[idx].flips[p].pairs[pair];
            }}"""
    walk = walk.replace('            storeObject(schemesBest + idx, scheme);', copy)
    walk = walk.replace('    storeObject(schemes + idx, scheme);', finish)
    walk = walk.replace('        if (randomUniform(&state) * maxIterations < basisProbability)\n            scheme.swapBasis(state);', '        randomUniform(&state);')
    (output / 'kernels.metal').write_text(kernels[:start] + walk + kernels[end:])
    if requested_mode in ('interleaved', 'interleaved16'):
        interleave_snapshot(output)
    if requested_mode == 'interleaved16':
        compact_snapshot(output)
    mode = requested_mode
    (output / 'storage.json').write_text(json.dumps({'mode': mode, 'input': str(source), 'sha256': hashes,
        'variant': mode, 'scratch_ints_per_worker': 1500,
        'padded_multiple': 32 if mode in ('interleaved', 'interleaved16') else 1,
        'buffer_bytes_per_worker': {'13': 6000, **({'14': 12600 if mode == 'interleaved16' else 33600, '15': 6000} if mode in ('interleaved', 'interleaved16') else {})},
        'conversion_in_dispatch': mode in ('interleaved', 'interleaved16'),
        'working_coefficient_bits': 16 if mode == 'interleaved16' else 64,
        'eligibility': 'signed 3x3, 1000 iterations, plus threshold 1000000000, disabled optional transformations'}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic device or hybrid Metal scheme storage')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['device', 'hybrid', 'interleaved', 'interleaved16'], required=True)
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve(), args.mode)


if __name__ == '__main__':
    main()
