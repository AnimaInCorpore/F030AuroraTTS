# F030AuroraTTS development

The DSP kernel must match the C++ reference word for word. A change to
`src/aurora/fixed.h`, `model.h`, `synth.h`, `vm.h` or the kernel is gated by
`sh tools/build-all.sh`, then `tools/gate/op-gate.py`, `syn-check.py 70`,
`net-gate.py` (both the default ids and a long sentence) and `tts-gate.py`.
Every Falcon program embeds the DSP image: rebuild them all after a kernel
change (build-all does). A change to the arithmetic is also checked against
PyTorch with `tools/ref/compare-net.py` and `compare-synth.py`.

Toolchain: MSYS2 ucrt64 on PATH (DOSBox and Hatari need its DLLs), ASM56000
under DOSBox, `m68k-atari-mintelf-g++`, the DSP-calibrated Hatari. The model
files under `third_party/auroratts` come from the Hugging Face model page.

DSP memory: external X aliases P $4000-$7fff, Y P $0000-$3fff. Code lives in
P $0080-$01ff (inner loops) and P $7000-$7fff (= X $3000-$3fff); X/Y
$0100-$01ff are avoided. Internal X $0080-$00ff is the matrix input stage and
the synthesizer's state; internal Y $0080-$00a7 holds the synthesizer's
48-bit state words. Emulator timing is a model result; state what was not run
on hardware.
