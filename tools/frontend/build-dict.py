#!/usr/bin/env python3
"""Build build/frontend/DICT.BIN for the Falcon English text frontend.

DICT.BIN (all integers big-endian) holds
  * the CMUdict entries g2p_en can reach (nltk cmudict, first pronunciation),
    plus g2p_en's own neural predictions for the hyphenated number words that
    inflect produces ("twenty-one", "thirty-first", ...), which g2p_en sends
    to its OOV network;
  * the g2p_en homograph table;
  * letter-to-sound decision trees for out-of-vocabulary words.

Dictionary coding: sorted words in blocks of BLOCK entries.  The first word of
each block is stored plainly in an index (binary search).  Inside a block,
words are front-coded (shared-prefix length + suffix letters) and the
phonemes follow each word.  Prefix lengths, letters (context = previous
letter) and phonemes (context = previous phoneme) are canonical-Huffman coded.

Usage:  python tools/frontend/build-dict.py [out=build/frontend/DICT.BIN]
"""
import collections
import heapq
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lts_train  # noqa: E402

_VOW = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
_CON = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
PHONES = _CON + [v + s for v in _VOW for s in "012"]
PH2ID = {p: i for i, p in enumerate(PHONES)}
NPH = len(PHONES)  # 69; symbol NPH = end / start context
CHARS = "'-abcdefghijklmnopqrstuvwxyz"  # code = index + 1, 0 = end
CH2C = {c: i + 1 for i, c in enumerate(CHARS)}
BLOCK = 32
POSCODES = {"V": 0, "N": 1, "ADJ": 2, "VBD": 3, "VBN": 4}
WORD_RE = re.compile(r"^[a-z](?:[a-z'\-]*[a-z])?$")
LTS_MIN_LEAF, LTS_MIN_GAIN = 5, 8.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- Huffman
def huff_lengths(freq, maxlen=15):
    """Canonical Huffman code lengths for {sym: count}; deterministic."""
    syms = sorted(s for s, c in freq.items() if c > 0)
    if len(syms) == 1:
        return {syms[0]: 0}
    f = {s: freq[s] for s in syms}
    while True:
        heap = [(f[s], i, (s,)) for i, s in enumerate(syms)]
        heapq.heapify(heap)
        L = {s: 0 for s in syms}
        k = len(syms)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            for s in a[2] + b[2]:
                L[s] += 1
            heapq.heappush(heap, (a[0] + b[0], k, a[2] + b[2]))
            k += 1
        if max(L.values()) <= maxlen:
            return L
        f = {s: (c + 1) // 2 for s, c in f.items()}


class Huff:
    def __init__(self, freq):
        self.len = huff_lengths(freq)
        self.code = {}
        order = sorted(self.len, key=lambda s: (self.len[s], s))
        code = 0
        prev = order and self.len[order[0]]
        for s in order:
            code <<= self.len[s] - prev
            prev = self.len[s]
            self.code[s] = code
            code += 1
        self.order = order

    def table(self):
        if len(self.order) == 1:
            return bytes([0, self.order[0]])
        maxlen = max(self.len.values())
        cnt = [0] * (maxlen + 1)
        for s in self.order:
            cnt[self.len[s]] += 1
        assert all(c < 256 for c in cnt)
        return bytes([maxlen] + cnt[1:] + self.order)


class Bits:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0
        self.total = 0

    def put(self, code, n):
        for i in range(n - 1, -1, -1):
            self.acc = (self.acc << 1) | ((code >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                self.acc = 0
                self.n = 0
        self.total += n

    def bytes(self):
        b = bytearray(self.out)
        if self.n:
            b.append(self.acc << (8 - self.n))
        return bytes(b)


# ---------------------------------------------------------------- data
def load_entries():
    import nltk
    for p in ("cmudict", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find("corpora/" + p if p == "cmudict" else "taggers/" + p)
        except LookupError:
            nltk.download(p, quiet=True)
    from nltk.corpus import cmudict
    d = cmudict.dict()
    ent = {}
    for w in sorted(d):
        if WORD_RE.match(w):
            ent[w] = [PH2ID[p] for p in d[w][0]]
    log("cmudict: %d entries, %d reachable" % (len(d), len(ent)))
    return d, ent


def number_words():
    unit = "one two three four five six seven eight nine".split()
    ordu = "first second third fourth fifth sixth seventh eighth ninth".split()
    tens = "twenty thirty forty fifty sixty seventy eighty ninety".split()
    teen = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    mill = ("thousand million billion trillion quadrillion quintillion sextillion septillion "
            "octillion nonillion decillion hundred zero oh").split()
    out = []
    for t in tens:
        for u in unit:
            out.append(t + "-" + u)
        for u in ordu:
            out.append(t + "-" + u)
        out.append(t[:-1] + "ieth")
    for w in unit + teen + tens + mill:
        out.append(w)
        out.append(w + "th")
    out += ordu
    return sorted(set(out))


def neural_cache(cmu, ent):
    """g2p_en's own OOV-network output for hyphenated number words."""
    from g2p_en import G2p
    g = G2p()
    n = 0
    for w in number_words():
        if w in cmu:
            continue
        ph = [PH2ID[p] for p in g.predict(w) if p in PH2ID]
        ent[w] = ph
        n += 1
    log("neural cache: %d number words" % n)


def homographs():
    import g2p_en
    path = os.path.join(os.path.dirname(g2p_en.__file__), "homographs.en")
    out = {}
    for line in open(path, encoding="utf8").read().splitlines():
        if line.startswith("#"):
            continue
        hw, p1, p2, pos = line.strip().split("|")
        out[hw.lower()] = ([PH2ID[p] for p in p1.split()], [PH2ID[p] for p in p2.split()], POSCODES[pos])
    return out


# ---------------------------------------------------------------- sections
def build_dict_section(ent):
    words = sorted(ent)
    # statistics for Huffman tables
    pre_f = collections.Counter()
    let_f = collections.defaultdict(collections.Counter)
    ph_f = collections.defaultdict(collections.Counter)

    def walk(emit_pre, emit_let, emit_ph):
        for bi in range(0, len(words), BLOCK):
            blk = words[bi:bi + BLOCK]
            prev = ""
            for k, w in enumerate(blk):
                if k:
                    p = 0
                    while p < len(w) and p < len(prev) and w[p] == prev[p]:
                        p += 1
                    p = min(p, 31)
                    emit_pre(bi, p)
                    ctx = CH2C[w[p - 1]] if p else 0
                    for c in w[p:]:
                        emit_let(bi, ctx, CH2C[c])
                        ctx = CH2C[c]
                    emit_let(bi, ctx, 0)
                ctx = NPH
                for ph in ent[w]:
                    emit_ph(bi, ctx, ph)
                    ctx = ph
                emit_ph(bi, ctx, NPH)
                prev = w

    walk(lambda b, p: pre_f.update([p]),
         lambda b, c, s: let_f[c].update([s]),
         lambda b, c, s: ph_f[c].update([s]))
    hpre = Huff(pre_f)
    hlet = {c: Huff(let_f[c]) for c in range(len(CHARS) + 1) if let_f[c]}
    hph = {c: Huff(ph_f[c]) for c in range(NPH + 1) if ph_f[c]}

    bits = Bits()
    blockoff = {}

    def bstart(b):
        if b not in blockoff:
            blockoff[b] = bits.total

    def e_pre(b, p):
        bstart(b)
        bits.put(hpre.code[p], hpre.len[p])

    def e_let(b, c, s):
        bits.put(hlet[c].code[s], hlet[c].len[s])

    def e_ph(b, c, s):
        bstart(b)
        bits.put(hph[c].code[s], hph[c].len[s])

    walk(e_pre, e_let, e_ph)
    stream = bits.bytes()

    nblocks = (len(words) + BLOCK - 1) // BLOCK
    first = bytearray()
    index = bytearray()
    for i in range(nblocks):
        index += struct.pack(">II", blockoff[i * BLOCK], len(first))
        first += words[i * BLOCK].encode("ascii") + b"\0"

    def tabset(tabs, n):
        offs = bytearray()
        body = bytearray()
        base = 2 * n
        for c in range(n):
            if c in tabs:
                offs += struct.pack(">H", base + len(body))
                body += tabs[c].table()
            else:
                offs += struct.pack(">H", 0)
        return bytes(offs + body)

    hp = hpre.table()
    hl = tabset(hlet, len(CHARS) + 1)
    hh = tabset(hph, NPH + 1)
    log("dict: %d words, %d blocks; index %d, first words %d, bits %d, tables %d bytes"
        % (len(words), nblocks, len(index), len(first), len(stream), len(hp) + len(hl) + len(hh)))
    return len(words), nblocks, bytes(index), bytes(first), stream, hp, hl, hh


def build_homo_section(h):
    words = sorted(h)
    offs = bytearray()
    body = bytearray()
    base = 2 + 2 * len(words)
    for w in words:
        p1, p2, pos = h[w]
        offs += struct.pack(">H", base + len(body))
        body += w.encode("ascii") + b"\0" + bytes([pos, len(p1)] + p1 + [len(p2)] + p2)
    return struct.pack(">H", len(words)) + bytes(offs) + bytes(body)


def build_lts_section(model):
    # labels
    lab = bytearray()
    for l in model.labels:
        lab += bytes([len(l)] + [PH2ID[p] for p in l])
    trees = bytearray()
    toffs = []
    for letter in lts_train.LETTERS:
        nodes = model.trees[letter]
        # byte positions: compute sizes with long/short offsets iteratively
        sizes = [2 if n[0] == "l" else 4 for n in nodes]
        while True:
            pos = [0] * (len(nodes) + 1)
            for i, s in enumerate(sizes):
                pos[i + 1] = pos[i] + s
            changed = False
            for i, n in enumerate(nodes):
                if n[0] == "q":
                    rel = pos[n[3]] - pos[i]
                    need = 4 if rel < 0x10000 else 5
                    if need != sizes[i]:
                        sizes[i] = need
                        changed = True
            if not changed:
                break
        toffs.append(len(trees))
        for i, n in enumerate(nodes):
            if n[0] == "l":
                assert n[1] < 0x8000
                trees += struct.pack(">H", n[1])
            else:
                rel = pos[n[3]] - pos[i]
                if sizes[i] == 4:
                    trees += bytes([0x80 | n[1], n[2]]) + struct.pack(">H", rel)
                else:
                    trees += bytes([0xC0 | n[1], n[2], rel >> 16, (rel >> 8) & 255, rel & 255])
    hdr = struct.pack(">HH", len(model.labels), lts_train.NFEAT)
    lab_off = 4 + 4 * 26
    tree_off = lab_off + len(lab)
    tbl = b"".join(struct.pack(">I", tree_off + o) for o in toffs)
    sec = hdr + tbl + bytes(lab) + bytes(trees)
    log("lts: %d labels, %d nodes, %d bytes" % (len(model.labels), sum(len(t) for t in model.trees.values()), len(sec)))
    return sec


# ---------------------------------------------------------------- POS tagger
# g2p_en picks homograph pronunciations from nltk's averaged-perceptron tag.
# We keep a pruned copy of that model: the tag dictionary, the context-free
# features (suffix, first letter, previous tags) for all words, and for the
# homograph position additionally the word, previous-tag+word and the
# neighbouring-word features (only for frequent context words).
FEAT_TYPES = ["bias", "i suffix", "i pref1", "i-1 tag", "i-2 tag", "i tag+i-2 tag", "i word",
              "i-1 tag+i word", "i-1 word", "i-1 suffix", "i-2 word", "i+1 word", "i+1 suffix",
              "i+2 word"]
R_TYPES = {"bias", "i suffix", "i pref1", "i-1 tag", "i-2 tag", "i tag+i-2 tag"}
TAG_TH_R, TAG_TH_CTXWORD, TAG_TH_CTXSUF = 1.0, 1.5, 1.5
WSCALE = 16  # weight units of 1/16, 10-bit signed


def build_tagger_section(homo):
    from nltk.tag.perceptron import PerceptronTagger
    t = PerceptronTagger()
    classes = sorted(t.classes)
    assert len(classes) < 64
    tagbyte = {c: i + 1 for i, c in enumerate(classes)}
    tagbyte["-START-"] = len(classes) + 1
    tagbyte["-START2-"] = len(classes) + 2
    voc = set(k for k in t.tagdict if k == k.lower()) | set(homo) | {
        "-START-", "-START2-", "-END-", "-END2-", "!HYPHEN"}

    def parse(f):
        if f == "bias":
            return "bias", []
        for name in sorted(FEAT_TYPES, key=len, reverse=True):
            if f.startswith(name + " "):
                return name, f[len(name) + 1:].split(" ")
        raise ValueError(f)

    feats = {}
    for f in sorted(t.model.weights):
        d = t.model.weights[f]
        name, args = parse(f)
        if name in R_TYPES:
            th = TAG_TH_R
        elif name in ("i word", "i-1 tag+i word"):
            if args[-1] not in homo:
                continue
            th = 0.0
        elif name in ("i-1 word", "i-2 word", "i+1 word", "i+2 word"):
            if args[0] not in voc:
                continue
            th = TAG_TH_CTXWORD
        else:
            th = TAG_TH_CTXSUF
        key = bytes([FEAT_TYPES.index(name) + 1])
        ok = True
        for k, a in enumerate(args):
            if name in ("i-1 tag", "i-2 tag", "i tag+i-2 tag") or (name == "i-1 tag+i word" and k == 0):
                if a not in tagbyte:
                    ok = False
                    break
                key += bytes([tagbyte[a]])
            else:
                try:
                    key += a.encode("ascii")
                except UnicodeEncodeError:
                    ok = False
                    break
        if not ok:
            continue
        pairs = []
        for c in sorted(d):
            w = d[c]
            if abs(w) < th or c not in tagbyte:
                continue
            q = max(-511, min(511, int(round(w * WSCALE))))
            if q:
                pairs.append((tagbyte[c] - 1, q))
        if pairs:
            feats[key] = pairs
    # serialise
    posmask = []
    for c in classes:
        m = 0
        for p, code in POSCODES.items():
            if c.startswith(p):
                m |= 1 << code
        posmask.append(m)
    td = sorted((k, v) for k, v in t.tagdict.items() if k == k.lower() and k.isascii())
    tdoff = bytearray()
    tdbody = bytearray()
    for k, v in td:
        tdoff += struct.pack(">H", len(tdbody))
        tdbody += k.encode("ascii") + b"\0" + bytes([tagbyte[v] - 1])
    assert len(tdbody) < 0x10000
    foff = bytearray()
    fbody = bytearray()
    for key in sorted(feats):
        assert len(fbody) % 2 == 0 and len(fbody) < 0x20000
        foff += struct.pack(">H", len(fbody) // 2)
        pairs = feats[key]
        ent = bytes([len(key)]) + key + bytes([len(pairs)])
        if len(ent) % 2:
            ent += b"\0"
        for c, q in pairs:
            ent += struct.pack(">H", (c << 10) | (q & 0x3FF))
        fbody += ent
    head = bytes([len(classes), 0]) + bytes(posmask)
    if len(head) % 2:
        head += b"\0"
    # layout: head | u16 ntd | u16 nfeat | u32 off_tdoff | u32 off_tdbody | u32 off_foff | u32 off_fbody
    fixed = len(head) + 4 + 16
    o_tdoff = fixed
    o_tdbody = o_tdoff + len(tdoff)
    o_foff = o_tdbody + len(tdbody)
    o_foff += o_foff & 1
    o_fbody = o_foff + len(foff)
    sec = head + struct.pack(">HHIIII", len(td), len(feats), o_tdoff, o_tdbody, o_foff, o_fbody)
    sec += tdoff + tdbody
    if len(sec) % 2:
        sec += b"\0"
    assert len(sec) == o_foff
    sec += foff + fbody
    log("tagger: %d classes, %d tagdict words, %d features, %d weights, %d bytes"
        % (len(classes), len(td), len(feats), sum(len(p) for p in feats.values()), len(sec)))
    return bytes(sec)


def train_lts(cmu):
    """Trees imitate g2p_en's OOV network: the training targets are the
    network's own predictions for the (a-z only) CMUdict words.  This matches
    the reference on unseen words noticeably better than training on the
    CMUdict pronunciations themselves (36% vs 28% exact on the test corpus)."""
    from g2p_en import G2p
    g = G2p()
    words, prons = [], []
    for w in sorted(w for w in cmu if re.fullmatch("[a-z]+", w)):
        p = [x for x in g.predict(w) if x in PH2ID]
        if p:
            words.append(w)
            prons.append(p)
    log("lts: %d training words (targets: g2p_en network predictions)" % len(words))
    return lts_train.train(words, prons, LTS_MIN_LEAF, LTS_MIN_GAIN, log=log)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "frontend", "DICT.BIN")
    cmu, ent = load_entries()
    neural_cache(cmu, ent)
    nw, nb, index, first, stream, hp, hl, hh = build_dict_section(ent)
    hg = homographs()
    homo = build_homo_section(hg)
    tagger = build_tagger_section(hg)
    lts = build_lts_section(train_lts(cmu))

    HDR = 64
    secs = [index, first, stream, hp, hl, hh, homo, lts, tagger]
    offs = []
    pos = HDR
    for s in secs:
        offs.append(pos)
        pos += len(s)
        pos += (-pos) & 3
    hdr = b"AGP1" + struct.pack(">IIIII", pos, nw, nb, BLOCK, NPH) + b"".join(struct.pack(">I", o) for o in offs)
    hdr += b"\0" * (HDR - len(hdr))
    blob = bytearray(hdr)
    for s, o in zip(secs, offs):
        blob += b"\0" * (o - len(blob))
        blob += s
    blob += b"\0" * (pos - len(blob))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    log("wrote %s: %d bytes" % (out, len(blob)))


if __name__ == "__main__":
    main()
