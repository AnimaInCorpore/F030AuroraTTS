#!/usr/bin/env python3
"""AURORA.TTP under FreeMiNT on the emulated Falcon.

A 1.44 MB boot floppy carries the FreeMiNT 1.19 snapshot's loader and 68030
kernel (memory protection off), a mint.cnf naming MINTTEST.TOS as INIT, and
the player. MINTTEST.TOS measures a busy loop alone, starts the player beside
it, and logs the loop's speed each second (AURORA/MINT.LOG). The player's
OUT.WAV must equal the host reference. Hatari's GEMDOS folder drive cannot be
used: MiNT replaces the GEMDOS trap Hatari hooks.

  mint-gate.py [--snapshot DIR] [--phonemes "HH AH0 L OW1 ."]
The snapshot is the unpacked freemint-1-19-648983e1-tt_falcon_clones.zip."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, HATARI, TOS, UCRT, listing_symbol  # noqa: E402
import fatimage  # noqa: E402

SYMS = (["<pad>", "<sil>"] + "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
        + [v + s for v in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split() for s in "012"]
        + [" ", ",", ".", "?", "!", ";", ":", "-"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=ROOT / "build/mint/drive")
    ap.add_argument("--phonemes", default="HH AH0 L OW1 _ W ER1 L D , _ HH AW1 _ AA1 R _ Y UW1 .")
    ap.add_argument("--output", type=Path, default=ROOT / "build/mint-gate")
    ap.add_argument("--alone", action="store_true", help="no busy neighbour: the player's run time alone")
    ap.add_argument("--player", type=Path, default=ROOT / "build/falcon/AURORA.TTP")
    args = ap.parse_args()
    case = args.output
    stage = case / "floppy"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "AUTO").mkdir(parents=True)
    kdir = stage / "MINT" / "1-19-648"
    kdir.mkdir(parents=True)
    snap = args.snapshot
    shutil.copy(snap / "auto/mint-648.prg", stage / "AUTO/MINT-648.PRG")
    shutil.copy(snap / "mint/1-19-648/mint030.prg", kdir / "MINT030.PRG")
    (kdir / "MINT.CNF").write_bytes(b"INIT=u:/a/aurora/minttest.tos\r\n")
    (kdir / "MINT.INI").write_bytes(b"MEM_PROT=NO\r\n")
    app = stage / "AURORA"
    app.mkdir()
    shutil.copy(ROOT / "build/falcon/MINTTEST.TOS", app / "MINTTEST.TOS")
    shutil.copy(args.player, app / "AURORA.TOS")
    if args.alone:
        (app / "ALONE.FLG").write_bytes(b"1")
    shutil.copy(ROOT / "build/AURORA.BIN", app / "AURORA.BIN")
    (app / "AURORA.CMD").write_bytes(("-q -w OUT.WAV -p " + args.phonemes + "\r\n").encode())
    image = case / "MINT.ST"
    image.write_bytes(fatimage.Image().build(stage))

    ids = [1] + [SYMS.index(" " if t == "_" else t) for t in args.phonemes.split()] + [1]
    subprocess.run([str(ROOT / "build/aurora-ref.exe"), str(ROOT / "build/AURORA.BIN"), "--ids",
                    ",".join(map(str, ids)), "--wav", str(case / "REF.WAV")], check=True, capture_output=True)

    stop = listing_symbol(ROOT / "build/dsp/AURORA.LST", "finished")
    (case / "start.ini").write_text(f"db pc = ${stop:04x} :once :trace :file {(case / 'end.ini').resolve()}\n")
    (case / "end.ini").write_text("quit 0\n")
    env = dict(os.environ)
    env["PATH"] = UCRT + os.pathsep + env.get("PATH", "")
    with (case / "debug.log").open("w") as log:
        subprocess.run([str(HATARI), "--machine", "falcon", "--dsp", "emu", "--memsize", "14", "--fpu", "68882",
                        "--conout", "2", "--tos", str(TOS), "--patch-tos", "true", "--fast-boot", "true",
                        "--fast-forward", "true", "--sound", "off", "--confirm-quit", "false",
                        "--protect-floppy", "off", "--disk-a", str(image.resolve()), "--run-vbls", "40000",
                        "--parse", str((case / "start.ini").resolve())],
                       cwd=case, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1800)
    img = image.read_bytes()
    log = fatimage.get_file(img, "AURORA/MINT.LOG")
    print(log.decode(errors="replace") if log else "no MINT.LOG (see debug.log)")
    for line in (case / "debug.log").read_text(errors="replace").splitlines():
        if "s of speech" in line:
            print(line.strip())
    wav = fatimage.get_file(img, "AURORA/OUT.WAV")
    if wav is None:
        raise SystemExit("no OUT.WAV")
    ref = (case / "REF.WAV").read_bytes()
    equal = wav[44:] == ref[44:]
    print(f"OUT.WAV {len(wav[44:]) // 2} samples, reference {len(ref[44:]) // 2}: {'EQUAL' if equal else 'DIFFERENT'}")
    if not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
