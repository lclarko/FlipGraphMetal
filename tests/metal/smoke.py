#!/usr/bin/env python3
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from verify import verify


def main():
    root = Path(__file__).resolve().parents[2]
    binary = root / "build/metal"
    output = Path(tempfile.mkdtemp(prefix="smoke-", dir=binary))
    reference = json.loads((binary / "test-output/input.json").read_text())
    f2 = json.loads((binary / "f2-test-output/transform-0.json").read_text())
    f2_input = output / "f2.txt"
    f2_input.write_text(" ".join(map(str, [*f2["n"], f2["m"], 1, *[v for key in "uvw" for row in f2[key] for v in row]])))
    runs = [
        ("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "8", "--max-iterations", "100", "--resize-probability", "0", "--path", str(output / "flip")]),
        ("flip_graph", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "4", "--max-iterations", "10", "--resize-probability", "1", "--path", str(output / "resize")]),
        ("complexity_minimizer", ["--input-path", str(binary / "test-output/minimizer.txt"), "--schemes", "8", "--max-iterations", "100", "--path", str(output / "minimize")]),
        ("additions_reducer", ["-i", str(binary / "test-output/input.txt"), "--count", "8", "-o", str(output / "reduce")]),
        ("additions_reducer", ["-i", str(binary / "test-output/input.txt"), "--count", "8", "--schemes-count", "2", "--max-flips", "10", "-o", str(output / "reduce-flips")]),
        ("flip_graph_f2", ["-n1", "3", "-n2", "3", "-n3", "3", "--schemes", "8", "--max-iterations", "100", "--resize-probability", "0", "--sandwiching-probability", "1", "--path", str(output / "flip-f2")]),
        ("complexity_minimizer_f2", ["--input-path", str(f2_input), "--schemes", "8", "--max-iterations", "100", "--path", str(output / "minimize-f2")]),
    ]
    results = []
    for index, (program, args) in enumerate(runs):
        command = [str(binary / program), *args, "--block-size", "4", "--seed", "7", "--rounds", "3"]
        start = time.monotonic()
        run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
        log = run.stdout + run.stderr
        (output / f"{index}-{program}.log").write_text(log)
        if run.returncode:
            raise RuntimeError(f"{program} failed: {log}")
        dispatches = [(name, float(ms)) for name, ms in re.findall(r"Metal dispatch (\w+): \d+ threads, ([\d.e+-]+) ms GPU", log)]
        if not dispatches:
            raise RuntimeError("no Metal dispatch evidence")
        results.append({"command": command, "wall_seconds": time.monotonic() - start, "dispatches": dispatches})
        print(program, "PASS", dispatches, flush=True)
    verified = []
    for path in output.glob("*/*.json"):
        data = json.loads(path.read_text())
        verified.append({"file": str(path.relative_to(output)), **verify(data, reference if path.parent.name == "reduce" else None)})
    if not any(row["file"].startswith("reduce/") for row in verified):
        raise RuntimeError("reduction smoke test exported no circuit")
    summary = {"runs": results, "verified": verified}
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {len(verified)} independently verified exports; evidence in {output}")


if __name__ == "__main__":
    main()
