"""Build Metal shader libraries and assemble portable command-line installations."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile

from build_config import configuration, update

DEFAULT_FLAGS = ('-std=metal3.2 -mmacosx-version-min=15.0 '
                 '-fmetal-math-mode=safe -fmetal-math-fp32-functions=precise')
VARIANTS = ('signed', 'f2', 'signed-testing', 'f2-testing')
PRODUCTION = {'flip_graph': 'signed', 'complexity_minimizer': 'signed',
              'additions_reducer': 'signed', 'flip_graph_f2': 'f2',
              'complexity_minimizer_f2': 'f2'}
# These native utilities require no shader library or Metal device.
HOST_PRODUCTION = ('scheme_tool',)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assemble(source_dir, variant='signed'):
    """Match the runtime source loader, including its literal pragma removal."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown Metal variant: {variant}')
    f2 = variant.startswith('f2')
    names = ['core.h', 'addition.h', 'flip_set.h', 'scheme_integer.h',
             'scheme_z2.h', 'pairs_counter.h', 'additions_reducer.h']
    if not f2:
        names.append('compact.h')
    names.extend(['controlled.h', 'controlled_capture.h', 'controlled_packed.h', 'kernels.metal', 'controlled_kernels.metal', 'reduction_kernels.metal'])
    if variant.endswith('-testing'):
        names.append('test_kernels.metal')
    source = '#define METAL_F2\n' if f2 else ''
    if variant.endswith('-testing'):
        source += '#define METAL_TESTING\n'
    for name in names:
        source += (Path(source_dir) / name).read_bytes().decode('utf-8').replace('#pragma once', '') + '\n'
    return source


def atomic_bytes(path, data):
    path = Path(path)
    if path.exists() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def relative_library_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or str(path) != name
            or not re.fullmatch(r'[A-Za-z0-9_./-]+', name)):
        raise ValueError('library name must be a safe relative installation path')
    return name


def build_library(source_dir, output, header, *, variant='signed',
                  compiler='xcrun -sdk macosx metal', flags=DEFAULT_FLAGS,
                  library_name=None):
    output, header = Path(output), Path(header)
    library_name = relative_library_name(library_name or 'shaders/' + output.name)
    source = assemble(source_dir, variant).encode('utf-8')
    assembled = output.with_suffix('.metal')
    receipt = output.with_name(output.name + '.build.json')
    command = [*shlex.split(compiler), *shlex.split(flags), str(assembled), '-o', str(output)]
    inputs = {'configuration': configuration(compiler, flags, source_dir),
              'variant': variant, 'source_sha256': digest(source),
              'helper_sha256': digest(Path(__file__).read_bytes()),
              'command': command, 'library_name': library_name}
    saved = None
    try:
        saved = json.loads(receipt.read_text())
    except (FileNotFoundError, ValueError):
        pass
    valid = (isinstance(saved, dict) and saved.get('inputs') == inputs
             and output.is_file() and saved.get('sha256') == digest(output.read_bytes()))
    commands = saved.get('commands', []) if valid else []
    if not commands:
        valid = False
    if not valid:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.metal-build-', dir=output.parent) as temporary:
            temporary = Path(temporary)
            source_path, library_path = temporary / 'library.metal', temporary / 'library.metallib'
            source_path.write_bytes(source)
            actual_command = [*shlex.split(compiler), *shlex.split(flags), str(source_path), '-o', str(library_path)]
            subprocess.run(actual_command, check=True)
            commands = [actual_command]
            if not library_path.is_file() or not library_path.stat().st_size:
                raise ValueError('Metal compiler did not produce a nonempty library')
            library_path.replace(output)
    sha256 = digest(output.read_bytes())
    atomic_bytes(assembled, source)
    atomic_bytes(header, (f'#pragma once\n#define METAL_LIBRARY_NAME "{library_name}"\n'
                          f'#define METAL_LIBRARY_SHA256 "{sha256}"\n').encode())
    update(receipt, {'inputs': inputs, 'sha256': sha256, 'commands': commands})
    return {'mode': 'metallib', 'library': library_name, 'sha256': sha256,
            'header': str(header), 'commands': commands, 'source_sha256': digest(source)}


def program_receipt(path, header=None):
    """Validate the existing build receipt without requiring original sources."""
    receipt = path.with_name(path.name + '.build.json')
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError(f'expected regular build receipt: {path.name}')
    try:
        saved = json.loads(receipt.read_text())
        stat = path.stat()
        if saved['output'] != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            raise ValueError('output metadata differs')
        inputs = saved['inputs']
        if not inputs['command'] or not isinstance(inputs['configuration'], dict):
            raise ValueError('missing build inputs')
        dependencies = inputs['dependencies']
        if not isinstance(dependencies, dict) or not dependencies:
            raise ValueError('missing dependency hashes')
        if header is not None:
            matches = [sha for name, sha in dependencies.items()
                       if Path(name).parts[-2:] == ('shaders', header.name)]
            if matches != [digest(header.read_bytes())]:
                raise ValueError('shader header dependency differs')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'program/build receipt mismatch: {path.name}: {error}') from error


def package(binary_dir, output):
    """Copy native production programs and required hash-bound shader assets."""
    binary_dir, output = Path(binary_dir), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('package output already exists')
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    contents = readme.read_bytes()
    section = re.search(rb'^## Attribution and rights\r?\n.*?(?=^## |\Z)', contents, re.MULTILINE | re.DOTALL)
    if section is None:
        raise ValueError('project README is missing its Attribution and rights section')
    attribution = section.group(0)
    files = {'ATTRIBUTION.md': digest(attribution)}
    expected = {}
    for variant in ('signed', 'f2'):
        name = f'shaders/{variant}.metallib'
        library = binary_dir / name
        header = binary_dir / 'shaders' / f'{variant}.h'
        for path in (library, header):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'expected regular package input: {path}')
        sha256 = digest(library.read_bytes())
        definitions = dict(re.findall(r'^#define (METAL_LIBRARY_(?:NAME|SHA256)) "([^"\n]+)"$',
                                      header.read_text(), re.MULTILINE))
        if definitions != {'METAL_LIBRARY_NAME': name, 'METAL_LIBRARY_SHA256': sha256}:
            raise ValueError(f'shader/header binding mismatch: {variant}')
        expected[variant] = (name, sha256)
        files[name] = sha256
    for name in (*PRODUCTION, *HOST_PRODUCTION):
        variant = PRODUCTION.get(name)
        path = binary_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'expected regular program: {name}')
        data = path.read_bytes()
        if not path.stat().st_mode & 0o111:
            raise ValueError(f'program is not executable: {name}')
        if variant is not None:
            library_name, sha256 = expected[variant]
            if library_name.encode() + b'\0' not in data or sha256.encode() + b'\0' not in data:
                raise ValueError(f'program/shader binding mismatch: {name}')
        program_receipt(path, binary_dir / 'shaders' / f'{variant}.h' if variant else None)
        files[name] = digest(data)
    # All validation precedes creation; a failed copy leaves an inspectable partial package.
    output.mkdir(parents=True, exist_ok=False)
    for name, sha256 in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == 'ATTRIBUTION.md':
            target.write_bytes(attribution)
        else:
            shutil.copy2(binary_dir / name, target)
        if digest(target.read_bytes()) != sha256:
            raise ValueError(f'package input changed while copying: {name}')
    manifest = {'format': 1, 'files': files}
    update(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    build = commands.add_parser('build')
    for name in ('source-dir', 'output', 'header'):
        build.add_argument('--' + name, required=True)
    build.add_argument('--variant', choices=VARIANTS, default='signed')
    build.add_argument('--compiler', default='xcrun -sdk macosx metal')
    build.add_argument('--flags', default=DEFAULT_FLAGS)
    build.add_argument('--library-name')
    build.add_argument('--manifest', type=Path)
    pack = commands.add_parser('package')
    pack.add_argument('--binary-dir', required=True)
    pack.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    manifest = args.pop('manifest', None)
    result = build_library(**args) if action == 'build' else package(**args)
    if manifest is not None:
        update(manifest, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
