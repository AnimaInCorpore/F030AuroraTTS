# AuroraTTS-100K on the Falcon DSP: design and fixed point

This port runs [AuroraTTS-100K](https://huggingface.co/AuroraAI-Research/AuroraTTS-100K)
(95,638 parameters, Apache-2.0) on the Atari Falcon030: the network and the
harmonic + noise synthesizer on the DSP56001, the text frontend and the length
regulator on the 68030.

## Division of work

| Stage | Where | Source |
| --- | --- | --- |
| text to 79 phoneme symbols | 68030 | `src/frontend/g2p.h` |
| encoder (4 loops), BiGRU, predictors, pitch embedding | DSP, streamed per token | `src/aurora/driver.h` |
| durations, Gaussian upsampling | 68030 | `src/aurora/model.h` |
| decoder (9 layers), head | DSP, one pass per layer | `src/aurora/driver.h` |
| harmonic + noise synthesis, FIR, resampling | DSP, one step per frame | `dsp/synth.asm`, `src/aurora/synth.h` |
| playback | Falcon DMA at 24,585 Hz | `src/falcon/aurora-tts.cpp` |

The DSP holds 32K words; the weights do not fit beside each other, so the
68030 uploads one layer's weights and streams the sequence through it. Each
pass is a small program of vector operations (`src/aurora/vm.h`) that the DSP
kernel interprets once per token or frame: matrix rows, layer norm, table
activations, element-wise arithmetic, and rings for the depthwise
convolutions. Weight blocks already in place are not uploaded again.

## Bit-exact reference

`src/aurora/fixed.h` defines every operation's arithmetic as the DSP performs
it: integer sums are exact in the 56-bit accumulator, so only each operation's
final rounding, shift and limiting matter. Three implementations must agree
word for word, and the gates check that they do:

- `model.h` + `synth.h`: the network and synthesizer computed directly (the
  reference);
- `vm.h`'s host interpreter running the pass programs on simulated X/Y memory
  (`aurora-vmtest`);
- the DSP kernel on the emulated Falcon (`tools/gate/*.py`).

## Number formats

| Quantity | Format |
| --- | --- |
| activations | Q7.16 (range +-128; the largest measured is 35) |
| weights | per matrix: word / 2^(23-s), s = 0..3; rows are [bias, weights] |
| tables | value array then slope array, linear interpolation |
| synthesis signal | Q3.20 |
| phase | cycles * 2^48 (48 bits), increment cycles * 2^28 |
| output | 16-bit at 24,585 Hz |

Layer norm: `1/sqrt(var + eps)` is `sqrt(n) / sqrt(S)` with `S = sum d^2 + n eps`,
so there is no division by n (sqrt(n) is folded into gamma by the exporter).
S is normalized by double shifts into [2^21, 2^23), the DSP accumulator's
natural [1/4, 1), and `rsqrt` comes from a 384-interval table over [1/2, 2).

## Accuracy against PyTorch

Measured with `tools/ref/compare-net.py` and `compare-synth.py` on the 24
evaluation sentences:

- durations: identical in every sentence (so the timing is the original's);
- normalized log-F0: at most 0.0023 off (about 1 cent);
- harmonic and noise log bands: at most 0.02 off (0.15 dB);
- synthesizer, same parameters, noise silenced: 20-30 dB waveform SNR over a
  whole sentence (sub-cent phase drift accumulates; 30-40 dB over the first
  half second), plus the resampler's 10.8 kHz cutoff;
- full output: log-mel distance 0.46 against PyTorch, where two PyTorch runs
  with different noise differ by 0.44.

## Deliberate differences from upstream

- **Noise.** Upstream takes the STFT of one white noise sequence, scales each
  bin, and inverts it. The port draws each frame's spectrum directly as white
  noise (uniform components), so it needs one inverse FFT per two frames
  instead of a forward and an inverse FFT per frame. The level and spectral
  shape are the same; frames are independent, which adds about 0.4 dB where
  the noise envelope changes quickly.
- **Window normalization** is a constant 2/3 (upstream divides the first and
  last 512 samples by a smaller envelope).
- **Sine table**: 1,024 entries without interpolation; harmonic phase offsets
  rounded to the table's steps.
- **Output rate** 24,585 Hz (the Falcon codec's 25.175 MHz / 1024), from the
  model's 22,050 Hz through a 32-tap, 256-phase resampler (10.8 kHz cutoff).
- **Length regulator** weights are normalized once per token and frame
  (`floor(E 2^23 / W)`), then summed.

## Speed (emulated Falcon, calibrated Hatari)

Sentence 1 of the evaluation set, 64 symbols, 4.24 s of speech, rendered by
`AURORA.TTP` (DSP seconds from `tools/gate/profile-split.py`):

| Stage | Before | Now |
| --- | --- | --- |
| encoder, BiGRU, predictors | 1.15 | 1.04 |
| length regulator (68030) | 0.32 | hidden under decoder layer 0 |
| decoder, 9 layers | 6.27 | 5.59 |
| synthesizer | 8.41 | 7.26 |
| total | 16.1 (3.8x) | **13.9 (3.3x)** |

Where the network's time goes: matrix products 3.4 s (one multiply per
instruction cycle with one external operand, the DSP's floor for this
network), activations 0.6 s, layer norm 0.45 s, host transfers and weight
uploads 1.1 s. The synthesizer: harmonic oscillators 2.5 s (54 harmonics a
sample, 5 instructions each plus 2 multiply-adds), inverse FFT 1.1 s, resampler
0.76 s, FIR 0.43 s, noise magnitudes and spectrum combining about 0.9 s.

What made the difference, in the order it was done:

- the 68030's length regulator: only the tokens with a nonzero weight, one
  64/32 divide per weight (`divu.l`) instead of 64-bit arithmetic per channel;
- host transfers in assembly; then, as in F030MXDRV, a ready token before
  every block the DSP receives, after which the 68030 writes without polling,
  and reads that poll only the first word;
- matrix inputs staged in internal X, so every multiply has one external
  operand (Hatari charges two extra cycles for an instruction that reaches
  two external spaces, an instruction fetch from external P included);
- two internal-P overlays, the network's and the synthesizer's inner loops,
  each copied into P $0080-$01ff when a command needs it;
- the noise drawn as a random spectrum (one FFT per frame pair, not two);
- the harmonic oscillator: phase offsets in whole table steps and a
  two-period sine table, then two harmonics per iteration, software-pipelined;
- the FFT butterfly with paired twiddle loads and `subl` (b' = 2a - a'), 12
  instructions instead of 20;
- the resampler's clamp folded into the accumulator limiter; the FIR in
  256-sample blocks; the noise magnitudes in runs of bins sharing a band pair;
- the decoder's first layer fed by the length regulator frame by frame, so the
  68030 upsamples while the DSP computes.

Not pursued: feeding the DSP through DMA and the SSI. F030SID found that
Hatari loses words on a free-running SSI stream without flagging an overrun,
so the route cannot be gated bit-exactly in the emulator, and it pays off
only for many small writes; here the host port's cost is a fraction of the
DSP's compute.

These are emulator results; nothing has run on a physical Falcon. Hatari
models external memory as zero wait states and ignores the bus control
register (the kernel clears it at start), and the Falcon's external memory
aliasing (external X at P $4000-$7fff, Y at P $0000-$3fff) should be probed on
hardware before trusting the overlays and table placement.
