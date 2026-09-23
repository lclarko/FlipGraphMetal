#!/usr/bin/env python3
"""Run bounded production checks under the caller's process-group guard."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from verify import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def converted_f2(path):
    data = json.loads(path.read_text())
    return " ".join(map(str, [*data["n"], data["m"], 1,
                               *[v for key in "uvw" for row in data[key] for v in row]])).encode()


def fixture_hashes(signed, f2):
    """Create the receipt once from baseline exports, then reuse it unchanged."""
    signed, f2 = Path(signed), Path(f2)
    hashes = {f"signed/{name}": digest(signed / name)
              for name in ("input.json", "input.txt", "minimizer.txt")}
    hashes["f2/transform-0.json"] = digest(f2 / "transform-0.json")
    hashes["converted/f2.txt"] = hashlib.sha256(converted_f2(f2 / "transform-0.json")).hexdigest()
    return hashes


def check_fixtures(signed, f2, expected, converted=None):
    actual = fixture_hashes(signed, f2)
    if actual != expected:
        raise ValueError("fixture receipt mismatch")
    if converted is not None and digest(converted) != expected["converted/f2.txt"]:
        raise ValueError("converted F2 fixture receipt mismatch")
    return actual


def save_summary(output, summary):
    # Atomic replacement retains the previous complete JSON on abrupt termination.
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "results.json")


def execute(command, root, output, summary, log_name, expected_error=None):
    """Persist intent before launch; stream directly without changing process groups."""
    record = {"command": command, "complete": False, "log": log_name}
    summary["runs"].append(record)
    save_summary(output, summary)
    start = time.monotonic()
    with (output / log_name).open("xb") as log:
        child = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=child.returncode, wall_seconds=time.monotonic() - start)
    save_summary(output, summary)
    if expected_error is not None:
        if child.returncode != 1 or expected_error not in (output / log_name).read_text():
            raise RuntimeError("expected explicit rejection: " + expected_error)
        record['expected_rejection'] = expected_error
        save_summary(output, summary)
    elif child.returncode:
        raise RuntimeError(f"child exited {child.returncode}; retained log: {output / log_name}")
    return record


def dispatch_evidence(log):
    devices = re.findall(r"^Metal device: (.+)$", log, re.MULTILINE)
    if not devices or any(not name.startswith("Apple ") for name in devices):
        raise ValueError("no Apple GPU device evidence")
    dispatches = [(name, int(threads), float(ms)) for name, threads, ms in re.findall(
        r"^Metal dispatch (\w+): (\d+) threads, (\S+) ms GPU$", log, re.MULTILINE)]
    if not dispatches or any(threads <= 0 or not math.isfinite(ms) or ms <= 0
                             for _, threads, ms in dispatches):
        raise ValueError("no positive finite Metal dispatch timing evidence")
    return {"devices": devices, "dispatches": dispatches}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--signed-fixtures", type=Path, required=True)
    parser.add_argument("--f2-fixtures", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.working_dir.resolve(strict=True)
    binary, signed, f2 = (path.resolve() for path in
                          (args.binary_dir, args.signed_fixtures, args.f2_fixtures))
    output = args.output.resolve()
    output.mkdir()  # Parent must exist; an existing attempt is never reused.
    summary = {"complete": False, "runs": [], "verified": [], "working_directory": str(root)}
    save_summary(output, summary)
    try:
        receipt_bytes = args.fixture_receipt.read_bytes()
        expected = json.loads(receipt_bytes)
        summary.update(fixture_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                       fixture_hashes=expected)
        save_summary(output, summary)
        check_fixtures(signed, f2, expected)
        reference = json.loads((signed / "input.json").read_text())
        verify(reference)
        verify(json.loads((f2 / "transform-0.json").read_text()))
        f2_input = output / "f2.txt"
        with f2_input.open("xb") as stream:
            stream.write(converted_f2(f2 / "transform-0.json"))
        runs = [
            ("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "8", "--max-iterations", "100", "--resize-probability", "0", "--path", str(output / "flip")]),
            ("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "4", "--max-iterations", "10", "--resize-probability", "1", "--path", str(output / "resize"), "--rounds", "2"]),
            ("complexity_minimizer", ["--input-path", str(signed / "minimizer.txt"), "--schemes", "8", "--max-iterations", "100", "--path", str(output / "minimize")]),
            ("additions_reducer", ["-i", str(signed / "input.txt"), "--count", "8", "-o", str(output / "reduce")]),
            ("additions_reducer", ["-i", str(signed / "input.txt"), "--count", "8", "--schemes-count", "2", "--max-flips", "10", "-o", str(output / "reduce-flips")]),
            ("flip_graph_f2", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "8", "--max-iterations", "100", "--resize-probability", "0", "--sandwiching-probability", "1", "--path", str(output / "flip-f2")]),
            ("complexity_minimizer_f2", ["--input-path", str(f2_input), "--schemes", "8", "--max-iterations", "100", "--path", str(output / "minimize-f2")]),
        ]
        runs = [(program, arguments, None) for program, arguments in runs]
        runs.append(("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "4",
            "--max-iterations", "10", "--resize-probability", "1",
            "--path", str(output / "resize-reject")], "flip candidate capacity exceeded (500 pairs per factor) in resizeKernel"))
        for index, (program, arguments, expected_error) in enumerate(runs):
            check_fixtures(signed, f2, expected, f2_input)
            command = [str(binary / program), *arguments, "--block-size", "4", "--seed", "7"]
            if "--rounds" not in arguments:
                command += ["--rounds", "3"]
            record = execute(command, root, output, summary, f"{index}-{program}.log", expected_error)
            record.update(dispatch_evidence((output / record["log"]).read_text()))
            already_verified = {item["file"] for item in summary["verified"]}
            for path in sorted(output.glob("*/*.json")):
                relative = str(path.relative_to(output))
                if relative not in already_verified:
                    result = verify(json.loads(path.read_text()), reference if path.parent.name == "reduce" else None)
                    summary["verified"].append({"file": relative, "sha256": digest(path), **result})
            record["complete"] = True
            save_summary(output, summary)
            print(program, "PASS", record["dispatches"], flush=True)
        if not any(row["file"].startswith("reduce/") for row in summary["verified"]):
            raise RuntimeError("reduction smoke test exported no circuit")
        check_fixtures(signed, f2, expected, f2_input)
        summary["complete"] = True
        save_summary(output, summary)
        print(f"PASS: {len(summary['verified'])} independently verified exports; evidence in {output}")
    except Exception as error:
        summary["error"] = str(error)
        save_summary(output, summary)
        raise


if __name__ == "__main__":
    main()
