#!/bin/sh
# Build a 68030 program from src/falcon: tools/falcon-build.sh aurtest AURTEST.TOS
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cxx=${M68K_CXX:-/c/msys64/mingw64/bin/m68k-atari-mintelf-g++}
mkdir -p "$root/build/falcon"
"$cxx" -m68030 -msoft-float -O2 -std=gnu++17 -fno-exceptions -fno-rtti -Wall -Wextra \
    -I "$root/src/aurora" -I "$root/src/falcon" -I "$root/src/frontend" -I "$root/build/dsp" \
    -c "$root/src/falcon/$1.cpp" -o "$root/build/falcon/$1.o"
"$cxx" -m68000 "$root/build/falcon/$1.o" -o "$root/build/falcon/$2"
