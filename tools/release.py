#!/usr/bin/env python3
"""Packages a binary release from the current build.

  release.py VERSION [--gates build/gates.log]

Writes build/release/:
  F030AuroraTTS-VERSION.zip   folder F030AuroraTTS-VERSION/ with AURORA.TTP,
                              AURORA.BIN, DICT.BIN, README.TXT (usage and
                              SHA-256 sums), LICENSE.TXT, NOTICE.TXT
  SHA256SUMS.txt              the archive's sum
  validation-VERSION.json     source commit, sums, sizes, gate results

Run tools/build-all.sh and the gates on a clean checkout of the release
commit first; the gate log's "==" headers and result lines are copied."""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ("AURORA.TTP", ROOT / "build/falcon/AURORA.TTP", "the program, with the DSP kernel embedded"),
    ("AURORA.BIN", ROOT / "build/AURORA.BIN", "the quantized model"),
    ("DICT.BIN", ROOT / "build/frontend/DICT.BIN", "the pronunciation dictionary and letter-to-sound rules"),
]

README = """F030AuroraTTS {version}
{rule}

NOTE: this project was developed with AI assistance (Claude). The code,
DSP kernel and documentation were written largely by an AI model under
human direction and checked by the emulator gates.

English text-to-speech for the Atari Falcon030: a port of AuroraTTS-100K,
a 95,638-parameter neural voice, to the Falcon's DSP56001. The network and
its synthesizer run on the DSP; speech plays by DMA at 24,585 Hz.

Source, documentation and issues:
  https://github.com/AnimaInCorpore/F030AuroraTTS

STATUS
  Tested only in the DSP-calibrated Hatari emulator, where every gate is
  bit-exact. It has NOT yet run on a physical Falcon. It is not real time:
  a sentence renders in about 3.3 times its spoken length, and each
  sentence plays while the next one renders.

REQUIREMENTS
  Falcon030 with its DSP56001, TOS 4.0x, about 2 MB of free RAM (some of it
  ST-RAM for playback). Keep the three files in one folder.

FILES
{files}

USAGE
  Start AURORA.TTP from the desktop and enter the text in the parameter
  box, or from a shell:

  AURORA.TTP Hello there. This is the Falcon speaking.
  AURORA.TTP -f STORY.TXT                   read the text from a file
  AURORA.TTP -p HH AH0 L OW1 _ W ER1 L D .  ARPAbet phonemes, "_" between words
  AURORA.TTP -w OUT.WAV Some text.          also write a WAV (24,585 Hz, mono)
  AURORA.TTP -q -w OUT.WAV Some text.       write the WAV without playing

  With no text it reads lines from the keyboard until an empty line. A key
  press stops after the current sentence. When started without arguments
  and a file AURORA.CMD sits beside it, its contents are the command line.
  Input is 8-bit ASCII or Latin-1.

FREEMINT
  AURORA.TTP runs under FreeMiNT and lets other programs run while it
  renders (supervisor mode only per DSP transfer, Syield while the DSP
  computes). Memory protection must be off (MEM_PROT=NO). Ctrl-C stops it
  after the current sentence. Tested in Hatari with FreeMiNT 1.19 only.

SHA-256
{sums}

LICENCE
  GNU General Public License, version 3 or later (LICENSE.TXT). The model
  is AuroraTTS-100K (Apache-2.0); the frontend uses data from g2p_en
  (Apache-2.0), inflect (MIT), CMUdict (BSD-style) and NLTK's averaged
  perceptron tagger (MIT). See NOTICE.TXT.
"""


def sha(path_or_bytes):
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return hashlib.sha256(data).hexdigest()


def crlf(text):
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("latin-1", errors="replace")


def gate_results(log):
    """[{gate, lines}] from the '== name' sections of a gate log."""
    out = []
    for block in re.split(r"^== ", log.read_text(errors="replace"), flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        out.append({"gate": name.strip(), "result": [l.strip() for l in body.splitlines() if l.strip()]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version")
    ap.add_argument("--gates", type=Path, default=ROOT / "build/gates.log")
    args = ap.parse_args()
    v = args.version
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("the working tree has uncommitted changes; release from a clean commit")

    files = [(n, p.read_bytes(), what) for n, p, what in FILES]
    readme = README.format(
        version=v, rule="=" * (len("F030AuroraTTS ") + len(v)),
        files="\n".join(f"  {n:<11}{len(b) // 1024:>5} KB  {what}" for n, b, what in files),
        sums="\n".join(f"  {sha(b)}  {n}" for n, b, _ in files))
    extra = [("README.TXT", crlf(readme)),
             ("LICENSE.TXT", crlf((ROOT / "LICENSE").read_text(encoding="utf-8"))),
             ("NOTICE.TXT", crlf((ROOT / "NOTICE.md").read_text(encoding="utf-8")))]

    out = ROOT / "build/release"
    out.mkdir(parents=True, exist_ok=True)
    folder = f"F030AuroraTTS-{v}"
    archive = out / f"{folder}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b, _ in files:
            z.writestr(f"{folder}/{n}", b)
        for n, b in extra:
            z.writestr(f"{folder}/{n}", b)
    digest = sha(archive)
    (out / "SHA256SUMS.txt").write_text(f"{digest}  {archive.name}\n", newline="\n")
    validation = {
        "version": v,
        "source_commit": commit,
        "archive": archive.name,
        "archive_sha256": digest,
        "files": [{"name": n, "bytes": len(b), "sha256": sha(b)} for n, b, _ in files],
        "emulator": "Hatari, Falcon with DSP emulation, TOS 4.02, 14 MB; FreeMiNT 1.19 snapshot 648983e1",
        "gates": gate_results(args.gates) if args.gates.exists() else [],
        "not_verified": ["a physical Falcon030", "real DSP external-memory timing",
                         "audio quality by listening tests"],
    }
    (out / f"validation-{v}.json").write_text(json.dumps(validation, indent=2) + "\n", newline="\n")
    print(f"{archive}\n{digest}")


if __name__ == "__main__":
    main()
