# F030AuroraTTS

English text-to-speech for the Atari Falcon030. It is a port of
[AuroraTTS-100K](https://huggingface.co/AuroraAI-Research/AuroraTTS-100K), a
95,638-parameter neural voice trained on LJSpeech, to the Falcon's DSP56001:

- the neural network and its harmonic + noise synthesizer run on the DSP in
  24-bit fixed point;
- the text frontend and the length regulator run on the 68030;
- speech plays by DMA at 24,585 Hz.

```
AURORA.TTP Hello there. This is the Falcon speaking.
```

The DSP's output is bit-identical to a C++ reference on the PC. That reference
is checked against the original PyTorch model: phoneme durations are identical
and pitch is within about a cent.

## Status

- Runs end to end in the DSP-calibrated Hatari, from text to audible DMA
  playback. **It has not yet run on a physical Falcon.**
- It is not real time. A sentence renders in about 3.3 times its spoken
  length: 4.24 s of speech takes 13.9 s on the emulated Falcon. Each sentence
  plays while the next one renders.
- Every emulator gate passes bit-exactly: each DSP operation, the synthesizer
  step by step, five whole sentences, and the player from plain text to WAV.
  None of this establishes hardware behaviour or audio quality.

## Running it on a Falcon

Requirements:

- a Falcon030 with a working DSP56001 (TOS 4.02 is the tested emulator
  configuration);
- about 2 MB of free RAM, some of it ST-RAM for playback.

Copy these three files into one folder:

| File | Size | Contents |
| --- | ---: | --- |
| `AURORA.TTP` | 300 KB | the program, with the DSP kernel embedded |
| `AURORA.BIN` | 476 KB | the quantized model |
| `DICT.BIN` | 983 KB | the pronunciation dictionary and letter-to-sound rules |

Start `AURORA.TTP` from the desktop, entering the text in the parameter box, or
from a shell:

```
AURORA.TTP Hello there. This is the Falcon speaking.
AURORA.TTP -f STORY.TXT                      read the text from a file
AURORA.TTP -p HH AH0 L OW1 _ W ER1 L D .     ARPAbet phonemes, "_" between words
AURORA.TTP -w OUT.WAV Some text.             also write a WAV (24,585 Hz, mono)
AURORA.TTP -q -w OUT.WAV Some text.          write the WAV without playing
```

How it behaves:

- **No text:** it reads lines from the keyboard until an empty line.
- **Splitting and stopping:** text is split into sentences (abbreviations such
  as "Dr." and initials do not end one), and a key press stops after the
  current sentence.
- **Progress:** each sentence prints its symbol count and render time.
- **`AURORA.CMD`:** when the program is started with no arguments and this file
  sits beside it, its contents are used as the command line. The emulator
  gates use this.

Input is 8-bit ASCII or Latin-1. Numbers, years, ordinals and currency are
spoken as the model's training frontend (g2p_en) would say them.

## How it works

| Stage | Where | Cost for 4.24 s of speech |
| --- | --- | --- |
| text to 79 phoneme symbols (`src/frontend`) | 68030 | about 0.1 s |
| encoder, BiGRU context, duration and pitch predictors | DSP | 1.0 s |
| length regulator (Gaussian upsampling) | 68030, while the DSP computes | hidden |
| decoder, 9 dilated-convolution layers, output head | DSP | 5.6 s |
| harmonic + noise synthesis, 65-tap FIR, resampling to 24,585 Hz | DSP | 7.3 s |
| playback | DMA | while the next sentence renders |

The DSP has 32K words of memory. The weights do not all fit, so the 68030
uploads one layer at a time and streams the sentence through it, one token or
frame at a time. The DSP runs each pass as a small program of vector
operations: matrix rows, layer norm, table activations, and rings for the
convolutions. The synthesizer is a dedicated DSP routine, one step per 11.6 ms
frame:

- up to 80 harmonic oscillators, with pitch and amplitude interpolated per
  sample;
- a noise spectrum shaped by 24 bands, then one inverse FFT per two frames and
  overlap-add;
- the model's learned output filter;
- a 256-phase resampler to the codec's rate.

See [docs/fixed-point.md](docs/fixed-point.md) for:

- the number formats and the fixed-point layer norm;
- the accuracy measurements against PyTorch;
- the deliberate differences from the upstream synthesizer;
- the speed breakdown and how it was reached.

The frontend is described in [src/frontend/README.md](src/frontend/README.md).

## Quality

- **Network:** durations are identical on all 24 evaluation sentences. The
  normalized pitch is within 0.0023 (about one cent). The spectral bands are
  within 0.02 in log amplitude (0.15 dB).
- **Synthesizer:** the output's spectrum is as close to PyTorch's as two
  PyTorch runs are to each other (the noise is random in both).
- **Upstream vs port:** the noise frames are drawn directly as random spectra.
  This halves the FFT work and adds about 0.4 dB where the noise envelope
  changes quickly. Output is resampled to 24,585 Hz with a 10.8 kHz cutoff.
- **Frontend:** it matches g2p_en on 98% of words. Every dictionary word,
  number and punctuation mark matches; rare unknown words and some homographs
  can differ.

## Building

The build runs on Windows in an MSYS2 shell. Requirements:

| Tool | Used for |
| --- | --- |
| MSYS2 ucrt64 (`g++`, Python), on `PATH` | host tools; DOSBox and Hatari need its DLLs |
| `m68k-atari-mintelf-g++` (set `M68K_CXX`) | the 68030 programs |
| DOSBox Staging (set `DOSBOX`) | running Motorola's DOS assembler |
| Motorola ASM56000, CLDLOD, DOS4GW and `ioequ.inc` (set `ASM56K`) | the DSP kernel, from F030OPL2's `third_party/f030dsp3d/tools/asm56k` |
| the DSP-calibrated Hatari (set `HATARI`) | the emulator gates |
| a Python venv with torch, safetensors, numpy, scipy, soundfile, nltk and g2p_en | the exporter, the reference comparisons and the dictionary builder |

Fetch the model files from the
[model page](https://huggingface.co/AuroraAI-Research/AuroraTTS-100K) into
`third_party/auroratts/`: `config.json`, `model.safetensors`,
`modeling_auroratts.py`, `configuration_auroratts.py` and
`eval_sentences.json`. Then:

```sh
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install g2p_en nltk safetensors soundfile scipy
sh tools/build-all.sh
```

`build-all.sh` builds:

- `build/AURORA.BIN` (the quantized model) and `src/aurora/aurora-layout.h`;
- the DSP kernel (`build/dsp/aurora-image.h`);
- the host tools in `build/`;
- the Falcon programs in `build/falcon/`;
- `build/frontend/DICT.BIN`, if it is missing (the dictionary takes about
  2.5 minutes).

Every Falcon program embeds the DSP kernel, so after any change to `dsp/` they
must all be rebuilt; `build-all.sh` does that.

## Checks

| Command | What it establishes |
| --- | --- |
| `.venv/Scripts/python tools/ref/compare-net.py` | the fixed-point network against PyTorch, 24 sentences |
| `.venv/Scripts/python tools/ref/compare-synth.py` | the fixed-point synthesizer against PyTorch's |
| `.venv/Scripts/python tools/frontend/compare.py` | the frontend against g2p_en |
| `build/aurora-vmtest.exe build/AURORA.BIN 1,6,16,...` | the DSP pass programs, run by the host interpreter, equal the reference |
| `python3 tools/gate/op-gate.py` | every DSP operation on the emulated Falcon equals the interpreter |
| `python3 tools/gate/syn-check.py 70` | the DSP synthesizer's output and state, step by step |
| `python3 tools/gate/net-gate.py --ids ...` | phonemes to PCM on the emulated Falcon, every checksum equal |
| `python3 tools/gate/tts-gate.py [--text "..."]` | `AURORA.TTP` writes a WAV equal to the reference, sample for sample |
| `python3 tools/gate/profile-split.py ids` | DSP time by kernel label, network and synthesizer separately |

Emulator timing is a model result: Hatari does not model the DSP's external
memory wait states. The memory layout relies on the Falcon's external DSP
memory mapping, which should be probed on real hardware first.

## Source map

| Path | Contents |
| --- | --- |
| `tools/export-model.py` | quantizes the checkpoint into `AURORA.BIN` and `aurora-layout.h` |
| `src/aurora/fixed.h` | the DSP's arithmetic, operation by operation |
| `src/aurora/model.h`, `synth.h` | the reference network and synthesizer |
| `src/aurora/vm.h` | the pass programs' operations and the host interpreter |
| `src/aurora/driver.h`, `synth-driver.h` | the passes the 68030 runs, shared with the PC gates |
| `src/aurora/symbols.h` | the 79 symbols and ARPAbet input |
| `dsp/aurora.asm` | the DSP kernel: commands, network machine, overlays |
| `dsp/synth.asm`, `synhot.asm`, `synmap.asm` | the DSP synthesizer and its memory map |
| `src/falcon/aurora-tts.cpp` | the player, `AURORA.TTP` |
| `src/falcon/falcon-dsp.h` | DSP boot and host-port transfers |
| `src/falcon/aurtest.cpp`, `optest.cpp`, `syntest.cpp`, `lockstep.cpp` | gate and debugging programs |
| `src/frontend/g2p.h` | text to phonemes |
| `tools/frontend/` | dictionary builder, letter-to-sound training, comparisons |
| `tools/ref/` | PyTorch runners and comparisons |
| `tools/gate/` | emulator gates and profilers |
| `tools/dsp/` | DSP build: assembly under DOSBox, the boot image header |
| `docs/fixed-point.md` | design, accuracy, deviations, speed |

## Credits and licences

F030AuroraTTS is licensed under the GNU General Public License, version 3 or
(at your option) any later version ([LICENSE](LICENSE)). The GPL is required by
the DSP boot loader taken from F030OPL2, and every other upstream licence is
compatible with it. [NOTICE.md](NOTICE.md) carries the upstream notices:

- **AuroraTTS-100K** (model, architecture, synthesizer design) by
  AuroraAI-Research: Apache-2.0. The weights are not in the repository; fetch
  `model.safetensors` from the model page.
- **g2p_en** (the frontend's behaviour, homograph table, number words):
  Apache-2.0. **inflect** (number words): MIT.
- **The CMU Pronouncing Dictionary** (in `DICT.BIN`): BSD-style, notice required.
- **NLTK's averaged perceptron tagger** (pruned, in `DICT.BIN`): MIT.
- **F030OPL2** (DSP boot loader, build and gate conventions): GPL-3.0.
  F030MXDRV and F030SID informed the host-port protocol and DSP placement rules.

A binary release of `AURORA.TTP`, `AURORA.BIN` and `DICT.BIN` must ship with
`LICENSE`, `NOTICE.md` and the corresponding source.
