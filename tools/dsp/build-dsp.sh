#!/bin/sh
# Assemble dsp/aurora.asm and the two-stage loader with Motorola's ASM56000
# under DOSBox, then emit build/dsp/aurora-image.h for the 68030 program.
#   ASM56K  directory with ASM56000.EXE, CLDLOD.EXE, DOS4GW.EXE, ioequ.inc
#           (default: the F030OPL2 checkout's pinned f030dsp3d submodule)
#   DOSBOX  DOSBox Staging executable
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
asm56k=${ASM56K:-/c/Arbeit/F030OPL2/third_party/f030dsp3d/tools/asm56k}
dosbox=${DOSBOX:-/c/Arbeit/F030Comanche/tools/toolchain/dosbox/dosbox.exe}
python=${PYTHON:-python3}
out="$root/build/dsp"
mkdir -p "$out"
for f in ASM56000.EXE CLDLOD.EXE DOS4GW.EXE ioequ.inc; do
    [ -f "$asm56k/$f" ] || { echo "error: $asm56k/$f is missing (set ASM56K)" >&2; exit 1; }
    cp "$asm56k/$f" "$out/"
done
cp "$root/dsp/aurora.asm" "$out/AURORA.ASM"
cp "$root/dsp/synth.asm" "$out/synth.asm"
cp "$root/dsp/synmap.asm" "$out/synmap.asm"
cp "$root/dsp/synhot.asm" "$out/synhot.asm"
cp "$root/dsp/stage2_loader.asm" "$out/AURBOOT.ASM"
cat > "$out/BUILD.BAT" <<'BATCH'
@ECHO OFF
ASM56000.EXE -q -a -bAURORA.CLD -z -lAURORA.LST AURORA.ASM
IF ERRORLEVEL 1 EXIT 1
CLDLOD.EXE AURORA.CLD > AURORA.LOD
IF ERRORLEVEL 1 EXIT 1
ASM56000.EXE -q -a -bAURBOOT.CLD -z -lAURBOOT.LST AURBOOT.ASM
IF ERRORLEVEL 1 EXIT 1
CLDLOD.EXE AURBOOT.CLD > AURBOOT.LOD
EXIT
BATCH
cd "$out"
rm -f AURORA.CLD AURORA.LOD AURORA.LST AURBOOT.CLD AURBOOT.LOD AURBOOT.LST
"$dosbox" --noprimaryconf --set output=texture "$out/BUILD.BAT" >/dev/null 2>&1 || true
for listing in AURORA AURBOOT; do
    [ -f "$listing.LST" ] || { echo "error: $listing.ASM did not assemble; see build/dsp/" >&2; exit 1; }
    grep -qE '^0 +Errors' "$listing.LST" || { grep -B3 -iE '^\*\*\*\*' "$listing.LST" | head -60 >&2; echo "error: $listing.ASM failed; see build/dsp/$listing.LST" >&2; exit 1; }
    grep -qE '^0 +Warnings' "$listing.LST" || { grep -B3 -iE '^\*\*\*\*' "$listing.LST" | head -60 >&2; echo "error: $listing.ASM warned; see build/dsp/$listing.LST" >&2; exit 1; }
done
"$python" "$root/tools/dsp/image-header.py" --bootstrap AURBOOT.LOD --program AURORA.LOD --output aurora-image.h
