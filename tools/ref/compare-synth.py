#!/usr/bin/env python3
"""Compare the fixed-point synthesizer with upstream's torch HNSynth on the
same head outputs. Harmonic-only (noise silenced on both sides) gives a
sample-by-sample SNR; the full output is compared by log-mel spectra since
the noise is random on both sides."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import aurora_torch as A  # noqa: E402

EXE = ROOT / "build" / "aurora-ref.exe"
RATE = 25175000 / 1024


def fixed(ids, noise):
    dump, wav = ROOT / "build/cs.bin", ROOT / "build/cs.wav"
    cmd = [str(EXE), str(ROOT / "build/AURORA.BIN"), "--ids", ",".join(map(str, ids)), "--dump", str(dump),
           "--wav", str(wav)] + ([] if noise else ["--no-noise"])
    subprocess.run(cmd, check=True, capture_output=True)
    b = dump.read_bytes()
    N, T = struct.unpack("<2i", b[:8])
    head = np.frombuffer(b[8:], dtype="<i4")[3 * N:].reshape(T, 66) / 65536
    y, _ = sf.read(str(wav))
    return head, y


@torch.no_grad()
def upstream(net, head, noise):
    c = net.cfg
    h = torch.tensor(head, dtype=torch.float32).T[None]
    f0 = torch.exp(h[:, 0] * c.f0_std + c.f0_mean).clamp(60, 600)
    voiced = (h[:, 1] > 0).float()
    logAh = h[:, 2:42].transpose(1, 2).clamp(-14, 2)
    logAn = h[:, 42:].transpose(1, 2).clamp(-14, 2)
    L = (h.shape[2] - 1) * c.hop
    nz = None if noise else torch.zeros(1, L)
    return net.dsp(f0, voiced, logAh, logAn, noise=nz)[0].clamp(-1, 1).numpy().astype(np.float64)


def logmel(y, sr):
    import torchaudio
    m = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, hop_length=256, n_mels=80, f_max=10000)
    return torch.log(m(torch.tensor(y, dtype=torch.float32)) + 1e-5).numpy()


def main():
    torch.manual_seed(0)
    m, net = A.load_model()
    sents = json.loads((ROOT / "third_party/auroratts/eval_sentences.json").read_text())
    for i, s in enumerate(sents[:int(sys.argv[1]) if len(sys.argv) > 1 else 8]):
        ids = m.text_to_ids(s)
        head, yh = fixed(ids, False)
        ref = resample(upstream(net, head, False), len(yh))
        err = yh - ref
        snr = 10 * np.log10((ref ** 2).sum() / (err ** 2).sum())
        head, yf = fixed(ids, True)
        rf = resample(upstream(net, head, True), len(yf))
        a, b = logmel(yf, RATE), logmel(rf, RATE)
        n = min(a.shape[1], b.shape[1])
        lsd = np.abs(a[:, :n] - b[:, :n]).mean()
        base = np.abs(logmel(resample(upstream(net, head, True), len(yf)), RATE)[:, :n] - b[:, :n]).mean()
        print(f"{i:2d} harmonic SNR {snr:6.2f} dB   log-mel L1 fixed-vs-torch {lsd:.3f} (torch-vs-torch noise {base:.3f})"
              f"  peak {np.abs(yf).max():.3f}")
        if i == 0:
            sf.write(str(ROOT / "build/s00_fixed.wav"), yf, 24585)
            sf.write(str(ROOT / "build/s00_torch.wav"), rf, 24585)


if __name__ == "__main__":
    main()
