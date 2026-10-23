#!/usr/bin/env python3
"""Compare the Falcon frontend (g2p.h via g2p-test) with the reference
text_to_ids of the upstream model (g2p_en).

Usage:
  python tools/frontend/compare.py [--exe build/frontend/g2p-test.exe]
         [--dict build/frontend/DICT.BIN] [--show N] [corpus files...]

Default corpora: third_party/auroratts/eval_sentences.json and
tools/frontend/corpus.txt.  Reports exact-sequence and word-level match
rates and the categories of word differences.
"""
import argparse
import collections
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "ref"))


def load_reference():
    import aurora_torch
    m = aurora_torch._load_module()
    return m


def trace(m, text):
    """Mirror of g2p_en.G2p.__call__ recording the source of every token.
    Returns (ids, [(word, tag, source, pron)])."""
    from g2p_en.expand import normalize_numbers
    from g2p_en import g2p as g2pmod
    g = m.get_g2p()
    t = normalize_numbers(text)
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = t.lower()
    t = re.sub("[^ a-z'.,?!\\-]", "", t)
    t = t.replace("i.e.", "that is").replace("e.g.", "for example")
    words = g2pmod.word_tokenize(t)
    tokens = g2pmod.pos_tag(words)
    out = []
    for word, pos in tokens:
        if re.search("[a-z]", word) is None:
            out.append((word, pos, "punct", [word]))
        elif word in g.homograph2features:
            p1, p2, pos1 = g.homograph2features[word]
            out.append((word, pos, "homograph", p1 if pos.startswith(pos1) else p2))
        elif word in g.cmu:
            out.append((word, pos, "dict", g.cmu[word][0]))
        else:
            out.append((word, pos, "oov", g.predict(word)))
    return out


def segments(ids, space):
    segs, cur = [], []
    for i in ids[1:-1]:
        if i == space:
            segs.append(tuple(cur))
            cur = []
        else:
            cur.append(i)
    segs.append(tuple(cur))
    return [s for s in segs if s]


def run_frontend(exe, dictp, texts):
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".txt") as f:
        for t in texts:
            f.write(t.encode("latin-1") + b"\n")
        path = f.name
    try:
        out = subprocess.run([exe, dictp, "-f", path, "-stats"], capture_output=True, check=True).stdout.decode()
    finally:
        os.unlink(path)
    res, stats = [], []
    for line in out.splitlines():
        if line.startswith("#stats"):
            stats.append(list(map(int, line.split()[1:])))
        elif line.startswith("ERR"):
            res.append(None)
        else:
            res.append(list(map(int, line.split())))
    return res, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "frontend", "g2p-test.exe"))
    ap.add_argument("--dict", default=os.path.join(ROOT, "build", "frontend", "DICT.BIN"))
    ap.add_argument("--show", type=int, default=12, help="examples per category")
    ap.add_argument("corpora", nargs="*")
    a = ap.parse_args()
    corpora = a.corpora or [os.path.join(ROOT, "third_party", "auroratts", "eval_sentences.json"),
                            os.path.join(HERE, "corpus.txt")]
    m = load_reference()
    sym = m.SYMBOLS
    space = m.SPACE_ID

    def name(seg):
        return " ".join(sym[i] for i in seg)

    grand = collections.Counter()
    for cp in corpora:
        if cp.endswith(".json"):
            texts = json.load(open(cp, encoding="utf8"))
        else:
            texts = [l.rstrip("\n") for l in open(cp, encoding="utf8") if l.strip() and not l.startswith("#")]
        got, _ = run_frontend(a.exe, a.dict, texts)
        c = collections.Counter()
        cat_examples = collections.defaultdict(list)
        for text, mine in zip(texts, got):
            try:
                ref = m.text_to_ids(text)
            except Exception:  # g2p_en itself fails (e.g. int(",5") in $ amounts)
                c["ref_error"] += 1
                continue
            tr = trace(m, text)
            c["sent"] += 1
            if mine == ref:
                c["sent_ok"] += 1
            # reference segments with categories (word tokens only)
            refseg, refcat = [], []
            for word, pos, src, pron in tr:
                ids = [m.SYM2ID[p] for p in pron if p in m.SYM2ID and p not in ("<pad>", "<sil>")]
                if not ids:
                    continue
                refseg.append(tuple(ids))
                refcat.append((src, word, pos))
            assert [s for s in segments(ref, space)] == refseg, (text, segments(ref, space), refseg)
            mseg = segments(mine or [m.SIL_ID, m.SIL_ID], space)
            sm = difflib.SequenceMatcher(None, refseg, mseg, autojunk=False)
            matched = set()
            for blk in sm.get_matching_blocks():
                for k in range(blk.size):
                    matched.add(blk.a + k)
            # pair up replaced segments for the report
            pairs = {}
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "replace" and i2 - i1 == j2 - j1:
                    for k in range(i2 - i1):
                        pairs[i1 + k] = mseg[j1 + k]
            for i, (seg, (src, word, pos)) in enumerate(zip(refseg, refcat)):
                if src == "punct":
                    c["punct"] += 1
                    c["punct_ok"] += i in matched
                    if i not in matched:
                        cat_examples["punct"].append((text[:60], word, name(seg), name(pairs.get(i, ()))))
                    continue
                c["words"] += 1
                c["w_" + src] += 1
                if i in matched:
                    c["words_ok"] += 1
                    c["ok_" + src] += 1
                else:
                    key = src
                    if i not in pairs:
                        key = src + "/alignment"
                    cat_examples[key].append((word, pos, name(seg), name(pairs.get(i, ()))))
        grand.update(c)
        print("=" * 78)
        print("%s: %d sentences" % (os.path.relpath(cp, ROOT), c["sent"]))
        if c["ref_error"]:
            print("  (%d sentences skipped: reference g2p_en raised an exception)" % c["ref_error"])
        print("  exact sequence match: %d/%d = %.1f%%" % (c["sent_ok"], c["sent"], 100.0 * c["sent_ok"] / c["sent"]))
        print("  word-level match:     %d/%d = %.2f%%" % (c["words_ok"], c["words"], 100.0 * c["words_ok"] / max(1, c["words"])))
        for src in ("dict", "homograph", "oov"):
            if c["w_" + src]:
                print("    %-10s %5d/%5d = %.1f%%" % (src, c["ok_" + src], c["w_" + src], 100.0 * c["ok_" + src] / c["w_" + src]))
        if c["punct"]:
            print("    %-10s %5d/%5d" % ("punct", c["punct_ok"], c["punct"]))
        for key in sorted(cat_examples):
            ex = cat_examples[key]
            print("  -- %s: %d differences" % (key, len(ex)))
            for e in ex[:a.show]:
                print("     %-18s %-5s ref: %-32s got: %s" % e)
    if len(corpora) > 1:
        c = grand
        print("=" * 78)
        print("TOTAL: exact %d/%d = %.1f%%, words %d/%d = %.2f%%" % (
            c["sent_ok"], c["sent"], 100.0 * c["sent_ok"] / c["sent"], c["words_ok"], c["words"],
            100.0 * c["words_ok"] / max(1, c["words"])))


if __name__ == "__main__":
    main()
