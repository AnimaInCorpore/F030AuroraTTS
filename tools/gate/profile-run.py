#!/usr/bin/env python3
"""Profile one AURTEST run on the emulated Falcon: DSP cycles by kernel label
and 68030 cycles by symbol. Emulated time; not a hardware measurement."""
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, run_hatari, listing_symbol  # noqa: E402

LABEL_RE = re.compile(r"^\s*\d+\s+([A-Za-z_][A-Za-z0-9_]*):?\s*(;.*)?$")
ADDRESS_RE = re.compile(r"^\s*\d+\s+P:([0-9A-F]+)\b")


def labels():
    """P addresses of code labels (data labels skipped)."""
    out, pending = [], []
    data_re = re.compile(r"^\s*\d+\s+[XYL]:[0-9A-F]+")
    label_re = re.compile(r"^\s*\d+\s+([A-Za-z_][A-Za-z0-9_]*):\s*(;.*)?$")
    for line in (ROOT / "build/dsp/AURORA.LST").read_text(errors="replace").splitlines():
        m = label_re.match(line)
        if m:
            pending.append(m.group(1))
            continue
        if data_re.match(line):
            pending.clear()
            continue
        a = ADDRESS_RE.match(line)
        if a and pending:
            for n in pending:
                out.append((int(a.group(1), 16), n))
            pending.clear()
    return sorted(out)


def main():
    ids = sys.argv[1] if len(sys.argv) > 1 else "1,6,16,33,12,71,5,60,24,1"
    player = "--player" in sys.argv
    case = ROOT / "build/profile"
    case.mkdir(parents=True, exist_ok=True)
    for f in ("RESULT.BIN", "dsp.txt", "cpu.txt"):
        (case / f).unlink(missing_ok=True)
    (case / "IDS.BIN").write_bytes(bytes(int(v) for v in ids.split(",")))
    shutil.copy(ROOT / "build/AURORA.BIN", case)
    shutil.copy(ROOT / "build/falcon/AURTEST.TOS", case)
    prog = "AURTEST.TOS"
    if player:
        syms = (["<pad>", "<sil>"] + "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
                + [v + s2 for v in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split() for s2 in "012"]
                + [" ", ",", ".", "?", "!", ";", ":", "-"])
        phon = " ".join("_" if syms[int(i)] == " " else syms[int(i)] for i in ids.split(",") if int(i) > 1)
        shutil.copy(ROOT / "build/falcon/AURORA.TTP", case / "AURORA.TOS")
        (case / "AURORA.CMD").write_text("-q -p " + phon + "\n")
        prog = "AURORA.TOS"
    stop = listing_symbol(ROOT / "build/dsp/AURORA.LST", "finished")
    (case / "start.ini").write_text(f"dp on\nprofile on\n"
                                    f"db pc = ${stop:04x} :once :trace :file {(case / 'end.ini').resolve()}\n")
    (case / "end.ini").write_text(f"dp save {(case / 'dsp.txt').resolve()}\nprofile save {(case / 'cpu.txt').resolve()}\nquit 0\n")
    import os
    env = dict(os.environ)
    from gate_env import HATARI, TOS, UCRT
    env["PATH"] = UCRT + os.pathsep + env.get("PATH", "")
    with (case / "debug.log").open("w") as log:
        subprocess.run([str(HATARI), "--machine", "falcon", "--dsp", "emu", "--memsize", "14", "--conout", "2",
                        "--tos", str(TOS), "--patch-tos", "true", "--fast-boot", "true", "--fast-forward", "true",
                        "--sound", "off", "--confirm-quit", "false", "--run-vbls", "200000",
                        "--parse", str((case / "start.ini").resolve()), prog],
                       cwd=case, stdout=log, stderr=subprocess.STDOUT, env=env)
    lab = labels()
    prof = re.compile(r"^p:([0-9a-f]+).*?\((\d+), (\d+), (\d+)\)")
    by = defaultdict(int)
    total = 0
    for line in (case / "dsp.txt").read_text(errors="replace").splitlines():
        m = prof.match(line.strip())
        if not m:
            continue
        addr, cycles = int(m.group(1), 16), int(m.group(3))
        if addr < 0x80:
            continue   # the boot code's idle loop before the program starts
        name = "?"
        for a, n in lab:
            if a <= addr:
                name = n
            else:
                break
        by[name] += cycles
        total += cycles
    print(f"DSP total {total} cycles ({total / 32e6:.2f} s at 32 MHz)")
    for n, c in sorted(by.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {n:24s} {c:12d} {100 * c / total:5.1f}%")


if __name__ == "__main__":
    main()
