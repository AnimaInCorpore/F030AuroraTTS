#!/usr/bin/env python3
"""The network on the emulated Falcon's DSP against the host reference.

AURTEST.TOS runs the encoder and decoder passes on the DSP (and the length
regulator on the 68030) for one phoneme sequence; its checksums of the
encoder output, the predictors and every head word must equal the host
reference's. Also reports the 200 Hz ticks of each stage (emulated time:
a model result, not a hardware measurement)."""
import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, run_hatari  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", default="1,6,16,33,12,71,5,60,24,1", help="phoneme ids, comma separated")
    ap.add_argument("--output", type=Path, default=ROOT / "build/net-gate")
    ap.add_argument("--vbls", type=int, default=20000)
    args = ap.parse_args()
    case = args.output
    case.mkdir(parents=True, exist_ok=True)
    for stale in ("RESULT.BIN", "results.json"):
        (case / stale).unlink(missing_ok=True)
    ids = bytes(int(v) for v in args.ids.split(","))
    (case / "IDS.BIN").write_bytes(ids)
    shutil.copy(ROOT / "build/AURORA.BIN", case / "AURORA.BIN")
    shutil.copy(ROOT / "build/falcon/AURTEST.TOS", case / "AURTEST.TOS")
    host = subprocess.run([str(ROOT / "build/aurora-ref.exe"), str(ROOT / "build/AURORA.BIN"), "--ids-file",
                           str(case / "IDS.BIN"), "--checksums"], capture_output=True, text=True, check=True)
    expected = json.loads(host.stdout)
    run_hatari(case, "AURTEST.TOS", "finished", args.vbls)
    rf = case / "RESULT.BIN"
    if not rf.is_file():
        raise SystemExit(f"no result; see {case / 'debug.log'}")
    r = struct.unpack(">48I", rf.read_bytes()[:192])
    got = dict(status=r[0], tokens=r[1], frames=r[2], duration_sum=r[3], x=r[4], predict=r[5], head=r[6],
               samples=r[13], pcm=r[14])
    result = dict(expected=expected, falcon=got,
                  ticks_200hz=dict(encode=r[7], upsample=r[8], decode=r[9], synth=r[15]),
                  port_words=dict(to_dsp=r[10], from_dsp=r[11]),
                  equal={k: got[k] == expected[k] for k in expected})
    vm = subprocess.run([str(ROOT / "build/aurora-vmtest.exe"), str(ROOT / "build/AURORA.BIN"), args.ids],
                        capture_output=True, text=True, env=dict(__import__("os").environ, PASSES="1"))
    if "passes:" not in vm.stdout:
        raise SystemExit("aurora-vmtest failed: " + vm.stdout + vm.stderr)
    host_passes = [int(v) for v in vm.stdout.split("passes:")[1].splitlines()[0].split()]
    falcon_passes = list(r[16:16 + r[12]])
    first = next((i for i, (a, b) in enumerate(zip(host_passes, falcon_passes)) if a != b), None)
    result["first_differing_pass"] = None if first is None else first - 1
    (case / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if got["status"] != 0 or not all(result["equal"].values()):
        raise SystemExit("MISMATCH")


if __name__ == "__main__":
    main()
