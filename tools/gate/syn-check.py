#!/usr/bin/env python3
"""Runs the synthesizer alone for N steps on the emulated DSP and compares its
outputs and state with the reference (aurora-syncheck)."""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, run_hatari  # noqa: E402

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ids = sys.argv[2] if len(sys.argv) > 2 else "1,6,16,33,12,71,5,60,24,1"
case = ROOT / "build/syn-check"
case.mkdir(parents=True, exist_ok=True)
(case / "DUMP.BIN").unlink(missing_ok=True)
subprocess.run([str(ROOT / "build/aurora-ref.exe"), str(ROOT / "build/AURORA.BIN"), "--ids", ids, "--dump",
                str(case / "ref.bin")], check=True, capture_output=True)
b = (case / "ref.bin").read_bytes()
N, T = struct.unpack("<2i", b[:8])
head = struct.unpack("<%di" % (T * 66), b[8 + 12 * N:8 + 12 * N + 4 * T * 66])
(case / "HEAD.BIN").write_bytes(struct.pack(">%di" % len(head), *head))
(case / "STEPS.BIN").write_bytes(bytes([steps]))
shutil.copy(ROOT / "build/AURORA.BIN", case)
shutil.copy(ROOT / "build/falcon/SYNTEST.TOS", case)
run_hatari(case, "SYNTEST.TOS", "finished", 20000)
sys.exit(subprocess.run([str(ROOT / "build/aurora-syncheck.exe"), str(ROOT / "build/AURORA.BIN"),
                         str(case / "HEAD.BIN"), str(case / "DUMP.BIN")]).returncode)
