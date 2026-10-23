#!/usr/bin/env python3
"""AURORA.TTP on the emulated Falcon against the host reference: the player
renders phonemes (or text, with DICT.BIN) to OUT.WAV without playing; every
sample must equal aurora-ref's. Reports the player's own timing line.
Emulated time is a model result, not a hardware measurement."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import ROOT, run_hatari  # noqa: E402

SYMBOLS = (["<pad>", "<sil>"] + "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
           + [v + s for v in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split() for s in "012"]
           + [" ", ",", ".", "?", "!", ";", ":", "-"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", default="1,6,16,33,12,71,5,60,24,71,22,54,11,53,14,71,19,69,71,9,63,13,71,53,13,71,53,17,"
                    "19,30,2,11,53,18,53,14,71,5,54,17,71,8,60,15,6,47,4,71,15,54,16,56,32,4,71,33,21,71,15,57,17,71,72,1")
    ap.add_argument("--text", help="one sentence of text (the Falcon frontend reads it; the host frontend gives the ids)")
    ap.add_argument("--output", type=Path, default=ROOT / "build/tts-gate")
    ap.add_argument("--vbls", type=int, default=200000)
    args = ap.parse_args()
    if args.text:
        out = subprocess.run([str(ROOT / "build/frontend/g2p-test.exe"), str(ROOT / "build/frontend/DICT.BIN"),
                              args.text], capture_output=True, text=True, check=True).stdout.split()
        args.ids = ",".join(out)
    ids = [int(v) for v in args.ids.split(",")]
    phon = " ".join("_" if SYMBOLS[i] == " " else SYMBOLS[i] for i in ids if i > 1)
    case = args.output
    case.mkdir(parents=True, exist_ok=True)
    (case / "OUT.WAV").unlink(missing_ok=True)
    shutil.copy(ROOT / "build/AURORA.BIN", case)
    shutil.copy(ROOT / "build/falcon/AURORA.TTP", case / "AURORA.TOS")
    if args.text:
        shutil.copy(ROOT / "build/frontend/DICT.BIN", case)
        (case / "AURORA.CMD").write_text("-q -w OUT.WAV " + args.text + "\n")
    else:
        (case / "AURORA.CMD").write_text("-q -w OUT.WAV -p " + phon + "\n")
    subprocess.run([str(ROOT / "build/aurora-ref.exe"), str(ROOT / "build/AURORA.BIN"), "--ids", args.ids, "--wav",
                    str(case / "REF.WAV")], check=True, capture_output=True)
    run_hatari(case, "AURORA.TOS", "finished", args.vbls)
    log = (case / "debug.log").read_text(errors="replace")
    for line in log.splitlines():
        if "s of speech" in line or "symbols" in line or "AuroraTTS" in line:
            print(line.strip())
    out, ref = (case / "OUT.WAV"), (case / "REF.WAV")
    if not out.is_file():
        raise SystemExit("no OUT.WAV")
    a, b = out.read_bytes()[44:], ref.read_bytes()[44:]
    print(f"samples: falcon {len(a) // 2}, reference {len(b) // 2}; {'EQUAL' if a == b else 'DIFFERENT'}")
    if a != b:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
