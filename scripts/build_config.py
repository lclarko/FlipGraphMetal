"""Record build configuration and compile only when inputs or configuration change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def executable_identity(command):
    resolved = shutil.which(command)
    if resolved is None:
        raise ValueError(f"compiler executable not found: {command}")
    path = Path(resolved).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def configuration(compiler, flags, source_dir):
    words = shlex.split(compiler)
    if not words:
        raise ValueError("compiler command is empty")
    identity = executable_identity(words[0])
    result = {
        "compiler": compiler,
        "launcher": identity,
        "flags": flags,
        "source_dir": str(Path(source_dir).resolve(strict=True)),
        "environment": {key: os.environ.get(key) for key in
                        ("DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CPATH",
                         "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")},
    }
    if Path(words[0]).name == "xcrun":
        # Preserve xcrun selectors such as --sdk while resolving its compiler.
        resolved = subprocess.run([*words[:-1], "--find", words[-1]],
                                  check=True, text=True, capture_output=True).stdout.strip()
        result["compiler_executable"] = executable_identity(resolved)
    return result


def update(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = Path(path)
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def build(output, config, dependencies, command):
    output = Path(output)
    receipt = output.with_name(output.name + ".build.json")
    expected = {
        "configuration": json.loads(Path(config).read_text()),
        "command": command,
        "dependencies": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                         for name in dependencies},
    }
    if output.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        stat = output.stat()
        if (saved.get("inputs") == expected and saved.get("output") ==
                {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}):
            return False
    output.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    stat = output.stat()
    update(receipt, {"inputs": expected,
                     "output": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    stamp = commands.add_parser("stamp")
    stamp.add_argument("--output", required=True)
    stamp.add_argument("--compiler", required=True)
    stamp.add_argument("--flags", required=True)
    stamp.add_argument("--source-dir", required=True)
    compile_parser = commands.add_parser("build")
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument("--config", required=True)
    compile_parser.add_argument("--dependency", action="append", default=[])
    compile_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "stamp":
        update(args.output, configuration(args.compiler, args.flags, args.source_dir))
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("build requires a compiler command")
        build(args.output, args.config, args.dependency, command)


if __name__ == "__main__":
    main()
