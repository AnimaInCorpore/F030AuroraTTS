"""Letter-to-sound training for the Falcon text frontend.

Learns one small decision tree per letter from CMUdict (the same data the
g2p_en neural OOV model was trained on):

1. Viterbi-EM alignment of letters to phones (each letter emits 0, 1 or 2
   phones, stress ignored while aligning).
2. Per letter a CART tree over a +-4 letter window, vowel/consonant classes
   of that window and the number of vowel groups before/after the letter.
   The leaf label is the phone string emitted for the letter, with stress.

Everything is deterministic (no randomness, stable tie-breaking).
"""
import collections
import math

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"
VOWELS = set("aeiouy")
WIN = (-4, -3, -2, -1, 1, 2, 3, 4)
NFEAT = 8 + 8 + 2  # letters, classes, vowel groups before/after
FEAT_NAMES = ["L%+d" % o for o in WIN] + ["C%+d" % o for o in WIN] + ["VB", "VA"]


def strip_stress(p):
    return p[:-1] if p[-1].isdigit() else p


def align(words, prons, iters=4, log=print):
    """Return per-word lists of per-letter phone tuples (with stress)."""
    base = [[strip_stress(p) for p in pr] for pr in prons]
    co = collections.defaultdict(collections.Counter)
    for w, b in zip(words, base):
        ps = set(b)
        for l in set(w):
            for p in ps:
                co[l][p] += 1
    c1 = {}
    for l in LETTERS:
        tot = sum(co[l].values()) or 1
        c1[l] = {p: -math.log(c / tot) for p, c in co[l].items()}
    ceps = {l: -math.log(0.25) for l in LETTERS}
    c2 = {l: {} for l in LETTERS}
    big1, big2 = 25.0, 30.0
    # initial pair costs: possible but expensive
    pair_init = True
    result = None
    for it in range(iters):
        out = []
        cnt = {l: collections.Counter() for l in LETTERS}
        total = 0.0
        for w, b in zip(words, base):
            n, m = len(w), len(b)
            if m > 2 * n:
                out.append(None)
                continue
            INF = 1e30
            prev = [INF] * (m + 1)
            prev[0] = 0.0
            backs = []
            for i in range(n):
                l = w[i]
                e = ceps[l]
                d1 = c1[l]
                d2 = c2[l]
                cur = [INF] * (m + 1)
                bk = [0] * (m + 1)
                lo = max(0, m - 2 * (n - i - 1) - 2)
                for j in range(lo, min(m, 2 * (i + 1)) + 1):
                    best = prev[j] + e
                    k = 0
                    if j >= 1:
                        v = prev[j - 1] + d1.get(b[j - 1], big1)
                        if v < best:
                            best, k = v, 1
                        if j >= 2:
                            if pair_init:
                                pc = 9.0 + d1.get(b[j - 2], big1) + d1.get(b[j - 1], big1)
                            else:
                                pc = d2.get((b[j - 2], b[j - 1]), big2)
                            v = prev[j - 2] + pc
                            if v < best:
                                best, k = v, 2
                    cur[j] = best
                    bk[j] = k
                backs.append(bk)
                prev = cur
            if prev[m] >= INF:
                out.append(None)
                continue
            total += prev[m]
            segs = []
            j = m
            for i in range(n - 1, -1, -1):
                k = backs[i][j]
                segs.append((j - k, j))
                j -= k
            segs.reverse()
            out.append(segs)
            for i, (a, z) in enumerate(segs):
                cnt[w[i]][tuple(b[a:z])] += 1
        # re-estimate
        for l in LETTERS:
            tot = sum(cnt[l].values()) + 1.0
            c1[l] = {}
            c2[l] = {}
            ceps[l] = -math.log((cnt[l].get((), 0) + 0.5) / tot)
            for o, c in cnt[l].items():
                if len(o) == 1:
                    c1[l][o[0]] = -math.log(c / tot)
                elif len(o) == 2:
                    if c >= 3:
                        c2[l][o] = -math.log(c / tot)
        pair_init = False
        log("  align iter %d: cost %.0f, unaligned %d" % (it, total, sum(1 for o in out if o is None)))
        result = out
    labels = []
    for pr, segs in zip(prons, result):
        if segs is None:
            labels.append(None)
        else:
            labels.append([tuple(pr[a:z]) for a, z in segs])
    return labels


def features(word):
    """Feature rows (len(word) x NFEAT) for a lowercase a-z word."""
    n = len(word)
    codes = [LETTERS.index(c) + 1 for c in word]
    isv = [c in VOWELS for c in word]
    # vowel group index: groups start where a vowel follows a non-vowel
    gstart = [isv[i] and (i == 0 or not isv[i - 1]) for i in range(n)]
    ngroups = sum(gstart)
    rows = []
    before = 0
    for i in range(n):
        # groups strictly before i (a group containing i does not count)
        row = []
        for o in WIN:
            k = i + o
            row.append(codes[k] if 0 <= k < n else 0)
        for o in WIN:
            k = i + o
            row.append(0 if not (0 <= k < n) else (1 if isv[k] else 2))
        # count groups ending before i
        b = 0
        a = 0
        cur_group = isv[i]
        # groups before: number of group starts at index < i, minus 1 if i is in a group that started before i
        starts_before = sum(1 for k in range(i) if gstart[k])
        if cur_group and not gstart[i]:
            starts_before -= 1
        b = starts_before
        starts_after = sum(1 for k in range(i + 1, n) if gstart[k])
        a = starts_after
        row.append(min(b, 3))
        row.append(min(a, 3))
        rows.append(row)
    return rows


class Tree:
    """Flat preorder tree: node = (feat, val, yes_index) or leaf label."""

    def __init__(self):
        self.nodes = []  # ('q', f, v, yes_idx) or ('l', label)


def _nlogn(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, x * np.log2(np.maximum(x, 1)), 0.0)


def build_tree(F, Y, nlab, min_leaf=2, min_gain=1.0):
    """F: (N, NFEAT) uint8, Y: (N,) int. Returns list of nodes (preorder)."""
    V = 32
    nodes = []

    def rec(idx):
        y = Y[idx]
        cnt = np.bincount(y, minlength=nlab)
        n = len(idx)
        maj = int(np.argmax(cnt))
        if cnt[maj] == n or n < 2 * min_leaf:
            nodes.append(("l", maj))
            return
        parent = n * math.log2(n) - float(_nlogn(cnt).sum())
        best = None
        Fi = F[idx]
        for f in range(NFEAT):
            joint = np.bincount(Fi[:, f].astype(np.int64) * nlab + y, minlength=V * nlab).reshape(V, nlab)
            yes_n = joint.sum(1)
            no = cnt[None, :] - joint
            no_n = n - yes_n
            yes_h = _nlogn(yes_n) - _nlogn(joint).sum(1)
            no_h = _nlogn(no_n) - _nlogn(no).sum(1)
            gain = parent - yes_h - no_h
            ok = (yes_n >= min_leaf) & (no_n >= min_leaf)
            gain = np.where(ok, gain, -1.0)
            v = int(np.argmax(gain))
            g = float(gain[v])
            if best is None or g > best[0] + 1e-9:
                best = (g, f, v)
        if best is None or best[0] < min_gain:
            nodes.append(("l", maj))
            return
        g, f, v = best
        me = len(nodes)
        nodes.append(None)
        mask = Fi[:, f] == v
        # "no" branch first (falls through), then "yes"
        rec(idx[~mask])
        yes_at = len(nodes)
        rec(idx[mask])
        nodes[me] = ("q", f, v, yes_at)

    rec(np.arange(len(Y)))
    return nodes


def tree_predict(nodes, row):
    i = 0
    while True:
        nd = nodes[i]
        if nd[0] == "l":
            return nd[1]
        _, f, v, yes = nd
        i = yes if row[f] == v else i + 1


def fix_stress(phones):
    """Exactly one primary stress per word (CMUdict convention)."""
    vow = [i for i, p in enumerate(phones) if p[-1].isdigit()]
    if not vow:
        return phones
    prim = [i for i in vow if phones[i][-1] == "1"]
    out = list(phones)
    if not prim:
        sec = [i for i in vow if phones[i][-1] == "2"]
        k = sec[0] if sec else vow[0]
        out[k] = out[k][:-1] + "1"
    elif len(prim) > 1:
        for i in prim[1:]:
            out[i] = out[i][:-1] + "2"
    return out


class LTS:
    def __init__(self, trees, labels):
        self.trees = trees  # letter -> nodes
        self.labels = labels  # label id -> tuple of phones

    def predict(self, word, fix=True):
        out = []
        for i, row in enumerate(features(word)):
            lab = tree_predict(self.trees[word[i]], row)
            out.extend(self.labels[lab])
        return fix_stress(out) if fix else out


def train(words, prons, min_leaf=2, min_gain=1.0, log=print):
    """words: a-z only. Returns LTS."""
    labs = align(words, prons, log=log)
    label_ids = {(): 0}
    per_letter = {l: ([], []) for l in LETTERS}
    for w, ls in zip(words, labs):
        if ls is None:
            continue
        for row, lab, c in zip(features(w), ls, w):
            if lab not in label_ids:
                label_ids[lab] = len(label_ids)
            per_letter[c][0].append(row)
            per_letter[c][1].append(label_ids[lab])
    labels = [None] * len(label_ids)
    for k, v in label_ids.items():
        labels[v] = k
    trees = {}
    for l in LETTERS:
        F = np.array(per_letter[l][0], dtype=np.uint8).reshape(-1, NFEAT)
        Y = np.array(per_letter[l][1], dtype=np.int64)
        if len(Y) == 0:
            trees[l] = [("l", 0)]
            continue
        trees[l] = build_tree(F, Y, len(labels), min_leaf, min_gain)
    log("  LTS: %d labels, %d tree nodes" % (len(labels), sum(len(t) for t in trees.values())))
    return LTS(trees, labels)
