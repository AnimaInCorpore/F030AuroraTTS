# Notices

F030AuroraTTS is licensed under the GNU General Public License, version 3 or
(at your option) any later version; see `LICENSE`. It contains or is derived
from the following works, whose terms are compatible with the GPL-3.0 and are
reproduced or referenced here.

## AuroraTTS-100K (Apache-2.0)

The network architecture, the synthesizer design and the model weights are
AuroraTTS-100K by AuroraAI-Research,
<https://huggingface.co/AuroraAI-Research/AuroraTTS-100K>, licensed under the
Apache License, Version 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>).
`third_party/auroratts/` holds unmodified upstream files (`config.json`,
`configuration_auroratts.py`, `modeling_auroratts.py`, `eval_sentences.json`,
`eval_results.json`). `AURORA.BIN`, built by `tools/export-model.py`, is a
modified (fixed-point, re-laid-out) form of the upstream weights, and the C++
and DSP code reimplement upstream's network and synthesizer.

## g2p_en (Apache-2.0)

The text frontend (`src/frontend/`, `DICT.BIN`) reproduces the behaviour of
g2p_en by Kyubyong Park (<https://github.com/Kyubyong/g2p>), Apache License 2.0:
its text normalization, its homograph table (`homographs.en`) and, for number
words and the letter-to-sound training data, the outputs of its neural model.

## inflect (MIT)

The number-to-words logic follows inflect (<https://github.com/jaraco/inflect>),
MIT License.

## The CMU Pronouncing Dictionary

`DICT.BIN` contains pronunciations from the CMU Pronouncing Dictionary
(cmudict 0.7a as distributed with NLTK data):

Copyright (C) 1993-2008 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Averaged perceptron tagger (MIT)

`DICT.BIN` contains a pruned form of NLTK's `averaged_perceptron_tagger_eng`
model, from textblob-aptagger:

Copyright 2013 Matthew Honnibal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## F030OPL2 (GPL-3.0)

The DSP two-stage boot loader (`dsp/stage2_loader.asm`,
`tools/dsp/generate_dsp_stage2.py`) and the build and gate conventions come
from F030OPL2 (<https://github.com/AnimaInCorpore/F030OPL2>), GPL-3.0.

## Not included

Motorola's ASM56000 tools, DOSBox, Hatari, TOS images and the MiNT toolchain
are used to build and test but are not part of this repository.
