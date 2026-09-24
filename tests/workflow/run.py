"""Run host-only workflow and independent-reference checks with retained receipts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from scalar_bridge import build, check_build


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory():
    paths = [ROOT / "makefile", ROOT / "scripts/build_config.py",
             ROOT / "tests/metal/verify.py"]
    for name in ("strassen_3x3.txt", "strassen_3x3_f2.txt", "strassen_4x4.txt",
                 "rank23_3x3.txt"):
        paths.append(ROOT / "tests/metal/fixtures" / name)
    for directory in ("tests/workflow", "src/workflow", "benchmarks/workflow",
                      "docs/specifications"):
        paths.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def run_suite(name, output):
    directory = ROOT / "tests/workflow"
    if name == "qualification":
        directory /= "qualification"
    suite = unittest.defaultTestLoader.discover(str(directory), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    complete = result.wasSuccessful() and not result.skipped and result.testsRun > 0
    summary = {"complete": complete, "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped)}
    with output.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--suite", choices=("routine", "qualification"), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.suite:
        if args.result is None:
            parser.error("suite execution requires a result path")
        return run_suite(args.suite, args.result)
    binary = ROOT / "build/metal/scheme_tool"
    build_receipt = binary.with_name(binary.name + ".build.json")
    if not binary.is_file() or not build_receipt.is_file():
        raise RuntimeError("make scheme-tool first")
    drivers = {
        "FGM_RUN_CONFIG_DRIVER": ROOT / "build/workflow/test_run_config",
        "FGM_HOST_RNG_DRIVER": ROOT / "build/workflow/test_host_rng",
    }
    driver_artifacts = [path for binary_path in drivers.values()
                        for path in (binary_path, binary_path.with_name(binary_path.name + ".build.json"))]
    if any(not path.is_file() for path in driver_artifacts):
        raise RuntimeError("make test-workflow or qualify-workflow to build native test drivers")
    directory = ROOT / "build/parity"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="host-check-", dir=directory))
    receipt = {
        "schema": "fgm-host-check-v1",
        "complete": False,
        "gpu_execution": "NOT RUN",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scheme_tool_sha256": digest(binary),
        "build_receipt_sha256": digest(build_receipt),
        "native_test_artifacts": {str(path.relative_to(ROOT)): digest(path) for path in driver_artifacts},
        "sources": source_inventory(),
        "specification_sha256": digest(ROOT / "docs/specifications/FGM-CONTRACT-v1.md"),
    }

    def save():
        temporary = attempt / "checks.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(attempt / "checks.json")

    save()
    try:
        environment = dict(os.environ, FGM_SCHEME_TOOL=str(binary))
        environment.update({name: str(path) for name, path in drivers.items()})
        reference = None
        if args.qualification:
            reference = build(attempt / "scalar-reference")
            receipt["scalar_receipt_sha256"] = digest(reference.parent / "receipt.json")
            environment["FGM_SCALAR_REFERENCE"] = str(reference)
        groups = ["routine"] + (["qualification"] if args.qualification else [])
        receipt["steps"] = []
        for name in groups:
            result_path = attempt / (name + "-result.json")
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--suite", name, "--result", str(result_path)]
            step = {"suite": name, "command": command, "complete": False}
            receipt["steps"].append(step)
            save()
            with (attempt / (name + ".log")).open("xb") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            step["exit_code"] = result.returncode
            if result_path.exists():
                step.update(json.loads(result_path.read_text()))
            save()
            if result.returncode or not step["complete"]:
                raise RuntimeError("host checks failed or skipped: " + name)
        if reference is not None:
            check_build(reference)
        if digest(binary) != receipt["scheme_tool_sha256"]:
            raise RuntimeError("native tool changed during tests")
        if digest(build_receipt) != receipt["build_receipt_sha256"]:
            raise RuntimeError("native build receipt changed during tests")
        if any(digest(ROOT / name) != expected for name, expected in receipt["native_test_artifacts"].items()):
            raise RuntimeError("native test driver or build receipt changed during tests")
        if source_inventory() != receipt["sources"]:
            raise RuntimeError("test inputs changed during execution")
        receipt["complete"] = True
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        receipt["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        (attempt / "checks.sha256").write_text(digest(attempt / "checks.json") + "\n")
        print("Retained host checks:", attempt, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
