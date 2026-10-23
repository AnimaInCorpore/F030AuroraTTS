#!/usr/bin/env python3
"""Profile AURORA.TTP on the emulated Falcon in two windows: the network
(program start to the synthesizer's first reset) and the synthesizer (the
rest), so each internal-P overlay's addresses get its own labels.
  profile-split.py ids [--top N]"""
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, HATARI, TOS, UCRT, listing_symbol  # noqa: E402

SYMS = (["<pad>", "<sil>"] + "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
        + [v + s for v in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split() for s in "012"]
        + [" ", ",", ".", "?", "!", ";", ":", "-"])
LST = ROOT / "build/dsp/AURORA.LST"


def labels():
    """(run address, name, overlay) for code labels; overlay 'vm', 'syn' or None."""
    out, pending = [], []
    label_re = re.compile(r"^\s*\d+\s+([A-Za-z_][A-Za-z0-9_]*):\s*(;.*)?$")
    data_re = re.compile(r"^\s*\d+\s+[XYL]:[0-9A-F]+\b")
    addr_re = re.compile(r"^\s*\d+\s+P:([0-9A-F]+)\b")
    ovl = None
    for line in LST.read_text(errors="replace").splitlines():
        m = label_re.match(line)
        if m:
            name = m.group(1)
            if name == "ovl_vm_start":
                ovl = "vm"
            elif name == "ovl_syn_start":
                ovl = "syn"
            pending.append(name)
            continue
        if data_re.match(line):
            pending.clear()
            continue
        a = addr_re.match(line)
        if a and pending:
            addr = int(a.group(1), 16)
            for n in pending:
                out.append((addr, n, ovl if addr < 0x200 else None))
            if pending[-1] in ("ovl_vm_end", "ovl_syn_end"):
                ovl = None
            pending.clear()
    return sorted(out)


def read_profile(path):
    prof = re.compile(r"^p:([0-9a-f]+).*\((\d+), (\d+), (\d+)\)\s*$")
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = prof.match(line.strip())
        if m:
            out[int(m.group(1), 16)] = int(m.group(3))
    return out


def attribute(cycles, lab, ovl):
    by = defaultdict(int)
    for addr, c in cycles.items():
        if addr < 0x80 or c <= 0:
            continue
        name = "?"
        for a, n, o in lab:
            if a > addr:
                break
            if o is None or o == ovl:
                name = n
        by[name] += c
    return by


def main():
    ids = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 22
    case = ROOT / "build/profile-split"
    case.mkdir(parents=True, exist_ok=True)
    for f in ("nn.txt", "all.txt"):
        (case / f).unlink(missing_ok=True)
    phon = " ".join("_" if SYMS[int(i)] == " " else SYMS[int(i)] for i in ids.split(",") if int(i) > 1)
    shutil.copy(ROOT / "build/AURORA.BIN", case)
    shutil.copy(ROOT / "build/falcon/AURORA.TTP", case / "AURORA.TOS")
    (case / "AURORA.CMD").write_text("-q -p " + phon + "\n")
    stop = listing_symbol(LST, "finished")
    syn = listing_symbol(LST, "synth_reset")
    (case / "start.ini").write_text(
        "dp on\n"
        f"db pc = ${syn:04x} :once :trace :file {(case / 'mid.ini').resolve()}\n"
        f"db pc = ${stop:04x} :once :trace :file {(case / 'end.ini').resolve()}\n")
    (case / "mid.ini").write_text(f"dp save {(case / 'nn.txt').resolve()}\n")
    (case / "end.ini").write_text(f"dp save {(case / 'all.txt').resolve()}\nquit 0\n")
    env = dict(os.environ)
    env["PATH"] = UCRT + os.pathsep + env.get("PATH", "")
    with (case / "debug.log").open("w") as log:
        subprocess.run([str(HATARI), "--machine", "falcon", "--dsp", "emu", "--memsize", "14", "--conout", "2",
                        "--tos", str(TOS), "--patch-tos", "true", "--fast-boot", "true", "--fast-forward", "true",
                        "--sound", "off", "--confirm-quit", "false", "--run-vbls", "200000",
                        "--parse", str((case / "start.ini").resolve()), "AURORA.TOS"],
                       cwd=case, stdout=log, stderr=subprocess.STDOUT, env=env)
    nn, allp = read_profile(case / "nn.txt"), read_profile(case / "all.txt")
    synp = {a: allp.get(a, 0) - nn.get(a, 0) for a in allp}
    lab = labels()
    for title, cyc, ovl in (("network (to the first synth reset)", nn, "vm"), ("synthesizer", synp, "syn")):
        by = attribute(cyc, lab, ovl)
        tot = sum(by.values())
        print(f"{title}: {tot / 32084988:.2f} s DSP")
        for n, c in sorted(by.items(), key=lambda kv: -kv[1])[:top]:
            print(f"  {n:22s} {c / 32084988:6.2f} s {100 * c / tot:5.1f}%")


if __name__ == "__main__":
    main()
