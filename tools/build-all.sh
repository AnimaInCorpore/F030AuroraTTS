#!/bin/sh
# Rebuilds everything that depends on the model image or the DSP kernel:
# the image, the DSP kernel, the host tools and the Falcon programs.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
export PATH=/c/msys64/ucrt64/bin:$PATH
PY=${PYTHON_VENV:-$root/.venv/Scripts/python}
"$PY" tools/export-model.py
sh tools/dsp/build-dsp.sh
for t in aurora-ref aurora-vmtest aurora-optest aurora-syncheck; do
    g++ -O2 -std=c++11 -Wall -Wextra -I src/aurora -I tools/host tools/host/$t.cpp -o build/$t.exe
done
for p in "aurtest AURTEST.TOS" "optest OPTEST.TOS" "syntest SYNTEST.TOS" "lockstep LOCKSTEP.TOS" "aurora-tts AURORA.TTP" "minttest MINTTEST.TOS"; do
    set -- $p
    sh tools/falcon-build.sh "$1" "$2"
done
[ -f build/frontend/DICT.BIN ] || "$PY" tools/frontend/build-dict.py
echo "built"
