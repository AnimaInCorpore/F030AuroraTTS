"""Where the gates find the emulator and TOS (the F030OPL2 conventions)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HATARI = Path(os.environ.get("HATARI", "/c/Arbeit/F030Arcade/third_party/hatari/build-ucrt64/src/hatari.exe"
                             if os.name != "nt" else r"C:\Arbeit\F030Arcade\third_party\hatari\build-ucrt64\src\hatari.exe"))
TOS = Path(os.environ.get("FALCON_TOS", r"C:\Arbeit\F030OPL2\third_party\f030dsp3d\tools\tos402.rom"))
UCRT = r"C:\msys64\ucrt64\bin"


def listing_symbol(listing, wanted):
    """The P address of a label in an ASM56000 listing."""
    import re
    label_re = re.compile(r"^\s*\d+\s+([A-Za-z_][A-Za-z0-9_]*):\s*(;.*)?$")
    addr_re = re.compile(r"^\s*\d+\s+P:([0-9A-F]+)\b")
    pending = []
    for line in Path(listing).read_text(errors="replace").splitlines():
        m = label_re.match(line)
        if m:
            pending.append(m.group(1))
            continue
        a = addr_re.match(line)
        if a and pending:
            if wanted in pending:
                return int(a.group(1), 16)
            pending.clear()
    raise SystemExit(f"{wanted} is missing from {listing}")


def run_hatari(case, program, stop_label, vbls, profile=False):
    """Run program (in case/) on the emulated Falcon until the DSP reaches stop_label."""
    import subprocess
    stop = listing_symbol(ROOT / "build/dsp/AURORA.LST", stop_label)
    (case / "start.ini").write_text(f"db pc = ${stop:04x} :once :trace :file {(case / 'end.ini').resolve()}\n")
    (case / "end.ini").write_text("quit 0\n")
    env = dict(os.environ)
    env["PATH"] = UCRT + os.pathsep + env.get("PATH", "")
    env.setdefault("HOME", str(Path.home()))
    with (case / "debug.log").open("w") as log:
        subprocess.run([str(HATARI), "--machine", "falcon", "--dsp", "emu", "--memsize", "14",
                        "--conout", "2", "--tos", str(TOS), "--patch-tos", "true",
                        "--fast-boot", "true", "--fast-forward", "true", "--sound", "off",
                        "--confirm-quit", "false", "--run-vbls", str(vbls),
                        "--parse", str((case / "start.ini").resolve()),
                        "--log-file", str((case / "hatari.log").resolve()), program],
                       cwd=case, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
