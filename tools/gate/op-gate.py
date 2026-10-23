#!/usr/bin/env python3
"""Each network-machine operation once on the emulated Falcon's DSP against the host interpreter."""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, run_hatari  # noqa: E402

case = ROOT / "build/op-gate"
case.mkdir(parents=True, exist_ok=True)
(case / "RESULT.BIN").unlink(missing_ok=True)
shutil.copy(ROOT / "build/AURORA.BIN", case / "AURORA.BIN")
shutil.copy(ROOT / "build/falcon/OPTEST.TOS", case / "OPTEST.TOS")
run_hatari(case, "OPTEST.TOS", "finished", 3000)
sys.exit(subprocess.run([str(ROOT / "build/aurora-optest.exe"), str(ROOT / "build/AURORA.BIN"),
                         str(case / "RESULT.BIN")]).returncode)
