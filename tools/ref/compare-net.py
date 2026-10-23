#!/usr/bin/env python3
"""Compare the fixed-point network (build/aurora-ref) with the PyTorch model.

For each sentence: durations equal or not, and the head outputs' error when the
float decoder is fed the fixed-point durations (so frame counts agree)."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import aurora_torch as A  # noqa: E402

EXE = ROOT / "build" / "aurora-ref.exe"


def run_fixed(ids):
    out = ROOT / "build" / "cmp.bin"
    subprocess.run([str(EXE), str(ROOT / "build/AURORA.BIN"), "--ids", ",".join(map(str, ids)), "--dump", str(out)],
                   check=True, capture_output=True)
    b = out.read_bytes()
    N, T = struct.unpack("<2i", b[:8])
    a = np.frombuffer(b[8:], dtype="<i4")
    dur, logd, pit = a[:N], a[N:2 * N] / 65536, a[2 * N:3 * N] / 65536
    head = a[3 * N:].reshape(T, 66) / 65536
    return dur, logd, pit, head


@torch.no_grad()
def float_head(net, ids, dur):
    ids_t = torch.tensor([ids])
    x, mask, logd, pit = net.encode(ids_t, torch.tensor([len(ids)]))
    x = x + net.pit_emb(pit.unsqueeze(1)) * mask
    d = torch.tensor(dur, dtype=torch.float32).view(1, -1)
    T = int(d.sum())
    h = net.decode(x, d, mask, T, torch.ones(1, 1, T))
    o = net.head(h)[0].T.numpy()
    fdur = ((logd.exp() - 1).round().clamp(min=1))[0].numpy()
    return o, logd[0].numpy(), pit[0].numpy(), fdur


def main():
    m, net = A.load_model()
    sents = json.loads((ROOT / "third_party/auroratts/eval_sentences.json").read_text())
    worst = 0
    for i, s in enumerate(sents):
        ids = m.text_to_ids(s)
        dur, logd, pit, head = run_fixed(ids)
        o, flogd, fpit, fdur = float_head(net, ids, dur)
        dd = int((fdur != dur).sum())
        e = np.abs(head - o)
        # f0n column 0, vuv 1, harmonic bands 2..41, noise 42..65 (clamped for synthesis)
        lo = np.clip(o[:, 2:], -14, 2)
        lf = np.clip(head[:, 2:], -14, 2)
        print(f"{i:2d} N={len(ids):3d} T={len(head):4d} durdiff={dd:2d} logd {np.abs(logd - flogd).max():.5f} "
              f"pit {np.abs(pit - fpit).max():.5f} f0n {e[:, 0].max():.5f} vuv {e[:, 1].max():.5f} "
              f"bands {np.abs(lf - lo).max():.5f}")
        worst = max(worst, e[:, :2].max())
    print("worst f0n/vuv error", worst)


if __name__ == "__main__":
    main()
