# English text frontend (g2p) for the Falcon port

`g2p.h` turns English text into the AuroraTTS token ids, reproducing
`text_to_ids()` in `third_party/auroratts/modeling_auroratts.py`, which uses
g2p_en. It is header-only C++ (C++11 and up) with no exceptions, RTTI, STL,
iostreams, floating point or heap allocation. All data comes from
`DICT.BIN` (983 KB), which the caller loads into RAM.

```cpp
#include "g2p.h"
static AuroraFront::G2p g2p;            // ~23 KB of work buffers: not on the TOS stack
g2p.load(dictBytes, dictSize);          // pointers into the buffer, no copy
int n = g2p.textToIds(text, ids, max);  // SIL ... SIL; -1 max too small, -2 text too long
```

Input is 8-bit ASCII or Latin-1 text. For text in the Atari ST character
set, call `G2p::atariToLatin1(buf)` first. The expanded text must fit in 8 KB
(`WORK`) and contain at most 1536 tokens.

## Pipeline: what is exact and what is approximated

| g2p_en step | here | status |
|---|---|---|
| `normalize_numbers` (commas, £, $, decimals, ordinals, years 1001–2999, inflect `number_to_words`) | 6 passes plus an inflect re-implementation | **exact**: 4,868 generated number/currency/ordinal cases give the same ids. The only differences are inside hyphenated tokens such as `1-2`, which g2p_en sends to its OOV network |
| NFD accent stripping, `lower()`, `[^ a-z'.,?!-]` filter, `i.e.`/`e.g.` | Latin-1 table | **exact** |
| TweetTokenizer (naked-domain URLs, `'`/`-` words, ellipsis, `HANG_RE`) | hand-written matcher | **exact**: 0 tokenization differences on 9,000 fuzz lines |
| CMUdict lookup (nltk cmudict, first pronunciation) | 122,608 entries, Huffman-coded | **exact**: 100% of dictionary words |
| Space and punctuation tokens, SIL handling | — | **exact** |
| Homographs (`homographs.en`, chosen by the nltk perceptron POS tag) | homograph table plus a **pruned port of the nltk averaged perceptron** (tag dictionary, suffix/prefix/previous-tag features for every word, word/context features at the homograph) | approximated: same choice as nltk for 97.7% of homographs in technical text and 94.8% in the homograph-heavy test sentences |
| OOV words (g2p_en GRU seq2seq) | see below | approximated |

g2p_en does **not** expand abbreviations. `Mr.` becomes the tokens `mr` and
`.`, and `mr` is in CMUdict. The frontend does the same.

### Out-of-vocabulary words

g2p_en runs a GRU encoder/decoder with hidden size 256 on every word that is
not in CMUdict. A fixed-point port would cost about 200k MACs per letter, or
roughly 1 s per letter on the 68030, so it is replaced by the following steps:

1. **Cache.** g2p_en's own network output is stored in the dictionary for
   every number word inflect can produce that CMUdict lacks: `twenty-one` …
   `ninety-ninth`, `quadrillion`, `twentieth`, and so on. These outputs are
   often odd (for example `ninety-two` → `N IY2 IH0 N T UW1 T IY0 OW0`), but
   they are what the model was trained with, so numbers are exact.
2. **Morphology.** `-'s -s -es -ies -ed -ied -ing -er -ier -ly -ily -ically
   -able -ably -ness -less -ment -ful -ism -ist -ish` on a dictionary stem
   (also stem+`e` and undoubled consonant).
3. **Compounds.** Two dictionary words, each with at least 4 letters (or 3+5),
   where the second may be a plural. The second part's primary stress becomes
   secondary, as in CMUdict (`windowsill`, `timestamp`, `filename`).
4. **Letter-to-sound trees.** There is one CART tree per letter (26,582
   nodes, 81 KB). Each tree looks at ±4 letters, their vowel/consonant class,
   and the vowel groups before and after. The leaf is the phone string for
   that letter, with stress. After that, exactly one primary stress is
   enforced and a vowel is inserted into vowel-less output (`hmm`). The trees
   are trained on the network's own predictions for the CMUdict vocabulary,
   which agrees with the network on unseen words better than training on
   CMUdict itself (36% vs 28% on the test words).

Hyphenated and dotted tokens (`well-known`, `www.example.com`) are split and
handled part by part. g2p_en's network mangles these (`well-known` →
`W EH1 L K AO1 N ER0`), so the frontend deliberately says them properly and
does not match the reference there.

## Measured match (tools/frontend/compare.py)

| corpus | sentences | exact id sequence | word-level |
|---|---|---|---|
| `third_party/auroratts/eval_sentences.json` | 24 | 20/24 = 83.3% | 352/356 = 98.9% |
| `tools/frontend/corpus.txt`: prose, numbers, money, ordinals, abbreviations, punctuation, names and rare words, homographs | 572 | 506/572 = 88.5% | 4198/4281 = 98.1% |
| total | 596 | 88.3% | 98.1% |

By category in `corpus.txt`:

- dictionary words: 3992/3992
- punctuation: 701/701
- homographs: 127/134
- OOV words: 79/155

In `eval_sentences.json` the OOV words are 6/10. Every differing sentence
differs only in OOV words or homographs.

The 76 OOV differences in `corpus.txt` break down as:

- 54 plain rare words, names and acronyms (`bouillabaisse`, `fbi`, `hmm`)
- 17 hyphenated compounds
- 3 joined URL-like tokens
- 2 with apostrophes

On a further 6,000 sentences of technical prose (GNU info manuals, not
included in the repo), dictionary words still match at 100%. Homographs match
97.7% and sentences 57.7%, which reflects how much identifier-like jargon that
text contains.

## DICT.BIN format (big-endian)

Header (64 bytes): `"AGP1"`, u32 file size, u32 word count, u32 block count,
u32 block size (32), u32 phone count (69), then u32 offsets of nine sections:

| # | section | bytes | content |
|---|---|---|---|
| 0 | block index | 30,656 | per block: u32 bit offset into the stream, u32 offset of its first word |
| 1 | first words | 32,364 | NUL-terminated first word of each block (binary search) |
| 2 | dictionary stream | 679,732 | per block entry: front-coding prefix length, suffix letters, END, phones, END. Entry 0 has no word (it is in the index) |
| 3 | prefix Huffman table | 40 | |
| 4 | letter tables | 1,168 | 29 contexts (previous letter), u16 offsets, then tables |
| 5 | phone tables | 3,972 | 70 contexts (previous phone) |
| 6 | homographs | 10,444 | u16 n, u16 offsets, entries `word\0 pos n1 phones n2 phones` |
| 7 | LTS | 80,828 | u16 nlabels, u16 nfeat, u32 tree offset per letter, labels (`n phones`), trees |
| 8 | POS tagger | 144,000 | class count and POS mask, tag dictionary (1,173 words), 7,853 features with 27,142 weights packed as u16 (6-bit class, 10-bit weight in 1/16 units) |

The Huffman tables are canonical: u8 maxlen, u8 count[maxlen], then the
symbols. maxlen 0 means a single symbol. Phone ids are positions in
`PHONES = _CON + vowels×stress`, so the model id is the phone id + 2. Words
use the alphabet `' - a-z`.

LTS tree nodes are stored in preorder. An internal node is
`0x80|feature (bit 6: long jump), value, jump` (u16, or u24 when long). If
`feature == value` the walk jumps, otherwise it falls through to the next
node. A leaf is a u16 label.

Tagger features are the nltk feature strings with the name replaced by a
type byte and the tags by `class+1` bytes.

## Cost on the 68030

There are no floats, multiplies or divides in hot paths; the 68030 build has
no libgcc calls. Per eval sentence the host counts about 15 dictionary
lookups, about 10k decoded Huffman bits, a few tagger features and a few
hundred tree nodes. At roughly 50–80 cycles per decoded bit, that is an
estimated 50–150 ms per sentence at 16 MHz. A long OOV word adds up to about
30 lookups for morphology and compound checks. This is an estimate and has
not been measured: run `build/frontend/G2PTIME.TOS` (with `DICT.BIN` beside
it) on the Falcon or in Hatari to get real figures.

Worst-case stack is about 1.5 KB. Code is about 23 KB.

## Rebuilding

```sh
# data (about 2.5 min: cmudict, g2p_en network predictions, LTS training)
.venv/Scripts/python.exe tools/frontend/build-dict.py        # -> build/frontend/DICT.BIN
# host tool
export PATH=/c/msys64/ucrt64/bin:$PATH
g++ -O2 -std=c++11 -Wall -Wextra -o build/frontend/g2p-test.exe tools/frontend/g2p-test.cpp
build/frontend/g2p-test.exe build/frontend/DICT.BIN "Hello, world."     # ids
build/frontend/g2p-test.exe build/frontend/DICT.BIN -f lines.txt -stats # batch
build/frontend/g2p-test.exe build/frontend/DICT.BIN -oov words.txt      # word -> phones
build/frontend/g2p-test.exe build/frontend/DICT.BIN -tags "I read it."  # POS tags
# comparison against the real g2p_en
.venv/Scripts/python.exe tools/frontend/compare.py [--show N] [corpus files]
# 68030 compile check and timing program
M=/c/msys64/mingw64/bin/m68k-atari-mintelf-g++
F="-m68030 -msoft-float -O2 -std=gnu++17 -fno-exceptions -fno-rtti -Wall -Wextra"
$M $F -c tools/frontend/g2p-m68k-check.cpp -o build/frontend/g2p-m68k-check.o
$M $F -c tools/frontend/g2ptime.cpp -o build/frontend/g2ptime.o
$M -m68030 build/frontend/g2ptime.o -o build/frontend/G2PTIME.TOS
```

`build-dict.py` is deterministic: two runs produce byte-identical files.
Building with `-DAURORA_G2P_STATS` adds work counters, which `g2p-test -time`
prints.
