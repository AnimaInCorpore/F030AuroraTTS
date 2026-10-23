import math
import os
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_auroratts import AuroraTTSConfig

@dataclass
class Config:
    sr: int = 22050
    hop: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    vocab: int = 79
    d: int = 56               # hidden width
    enc_k: int = 5
    enc_loops: int = 4        # one shared block applied 4x (separate norms per loop)
    gru_h: int = 28           # BiGRU hidden per direction (2*gru_h == d)
    pred_ch: int = 32
    dec_layers: int = 9
    dec_k: int = 5
    dec_dil: tuple = (1, 2, 4)
    dec_ffn: int = 3          # distinct FFNs shared across dec_layers
    kh: int = 40              # harmonic envelope bands (mel-spaced)
    kn: int = 24              # noise envelope bands (mel-spaced)
    n_harm: int = 80
    fir_taps: int = 65
    f0_mean: float = 5.3      # mean log-Hz, overwritten from data stats
    f0_std: float = 0.25

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        if "dec_dil" in d:
            d["dec_dil"] = tuple(d["dec_dil"])
        return cls(**d)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def hz2mel(f):
    return 2595.0 * torch.log10(1.0 + f / 700.0)


class CLN(nn.Module):
    """LayerNorm over the channel dim of a (B, C, T) tensor."""
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        return F.layer_norm(x.transpose(1, 2), (x.shape[1],), self.w, self.b).transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.a = nn.Conv1d(d, mult * d, 1)
        self.b = nn.Conv1d(mult * d, d, 1)

    def forward(self, x):
        return self.b(F.gelu(self.a(x)))


class LoopedEncoder(nn.Module):
    """One conv-GLU + FFN block applied `loops` times (weights shared, norms per loop)."""
    def __init__(self, d, k, loops):
        super().__init__()
        self.loops = loops
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.glu = nn.Conv1d(d, 2 * d, 1)
        self.ffn = FFN(d)
        self.n1 = nn.ModuleList([CLN(d) for _ in range(loops)])
        self.n2 = nn.ModuleList([CLN(d) for _ in range(loops)])

    def forward(self, x, mask):
        for i in range(self.loops):
            y = self.n1[i](x) * mask
            x = x + F.glu(self.glu(self.dw(y)), dim=1)
            x = x + self.ffn(self.n2[i](x) * mask)
            x = x * mask
        return x


class ContextGRU(nn.Module):
    """Phoneme-rate BiGRU for phrase-level context (packed, so padding never leaks)."""
    def __init__(self, d, h):
        super().__init__()
        assert 2 * h == d
        self.norm = CLN(d)
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)

    def forward(self, x, lens_cpu):
        y = self.norm(x).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(y, lens_cpu, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[2])
        return x + out.transpose(1, 2).to(x.dtype)


class VarPred(nn.Module):
    def __init__(self, d, c, k=5):
        super().__init__()
        self.n = CLN(d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.p1 = nn.Conv1d(d, c, 1)
        self.p2 = nn.Conv1d(c, c, 1)
        self.o = nn.Conv1d(c, 1, 1)

    def forward(self, x, mask):
        y = self.dw(self.n(x) * mask)
        y = F.relu(self.p1(y))
        y = F.relu(self.p2(y))
        return self.o(y).squeeze(1)


def gaussian_upsample(h, dur, sigma, T, tok_mask):
    """Soft (Gaussian) length regulator. h (B,d,N), dur (B,N), tok_mask (B,N) bool.
    Returns upsampled features (B,d,T) and two relative-position features (B,2,T)."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        h = h.float()
        dur = dur.float()
        centers = dur.cumsum(1) - dur / 2
        t = torch.arange(T, device=h.device).float() + 0.5
        diff = t[None, :, None] - centers[:, None, :]                     # (B,T,N)
        logits = -(diff ** 2) / (2 * sigma ** 2)
        logits = logits.masked_fill(~tok_mask[:, None, :], -1e9)
        w = logits.softmax(-1)
        up = torch.bmm(h, w.transpose(1, 2))                              # (B,d,T)
        dsafe = dur.clamp(min=1.0)
        pos = (w * (diff / dsafe[:, None, :])).sum(-1)
        ld = (w * torch.log(dsafe)[:, None, :]).sum(-1)
        return up, torch.stack([pos, ld], 1)


class Decoder(nn.Module):
    """Frame-rate dilated depthwise-conv stack; FFN weights are shared across layers."""
    def __init__(self, c):
        super().__init__()
        d, k = c.d, c.dec_k
        self.L, self.G = c.dec_layers, c.dec_ffn
        self.inp = nn.Conv1d(d + 2, d, 1)
        dils = [c.dec_dil[i % len(c.dec_dil)] for i in range(self.L)]
        self.dw = nn.ModuleList([nn.Conv1d(d, d, k, padding=dl * (k - 1) // 2, dilation=dl, groups=d) for dl in dils])
        self.ln = nn.ModuleList([CLN(d) for _ in range(self.L)])
        self.ffn = nn.ModuleList([FFN(d) for _ in range(self.G)])
        self.out_ln = CLN(d)

    def forward(self, x, fmask):
        x = self.inp(x) * fmask
        for i in range(self.L):
            y = self.dw[i](x)
            y = self.ffn[i % self.G](self.ln[i](y))
            x = (x + y) * fmask
        return self.out_ln(x)


class HNSynth(nn.Module):
    """Differentiable harmonic + filtered-noise synthesizer. Learned: per-harmonic phase
    offsets `theta` and a static output FIR (145 params total)."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        H = c.n_harm
        self.register_buffer("harm", torch.arange(1, H + 1).float(), persistent=False)
        self.register_buffer("win", torch.hann_window(c.n_fft), persistent=False)
        nyq = c.sr / 2
        self.nyq_mel = float(2595 * math.log10(1 + nyq / 700))
        bins = c.n_fft // 2 + 1
        freqs = torch.linspace(0, nyq, bins)
        pos = (hz2mel(freqs) / self.nyq_mel).clamp(0, 1) * (c.kn - 1)
        i0 = pos.floor().clamp(max=c.kn - 2).long()
        fr = pos - i0
        M = torch.zeros(bins, c.kn)
        ar = torch.arange(bins)
        M[ar, i0] = 1 - fr
        M[ar, i0 + 1] = fr
        self.register_buffer("interp", M.t().contiguous(), persistent=False)   # (kn, bins)
        self.theta = nn.Parameter(torch.zeros(H))
        fir = torch.zeros(c.fir_taps)
        fir[c.fir_taps // 2] = 1.0
        self.fir = nn.Parameter(fir)

    def forward(self, f0, voiced, logAh, logAn, noise=None):
        """f0 (B,n+1) Hz, voiced (B,n+1), logAh (B,n+1,kh), logAn (B,n+1,kn)
        -> waveform (B, n*hop). Frame j is centred on sample j*hop."""
        c = self.c
        f0, voiced, logAh, logAn = f0.float(), voiced.float(), logAh.float(), logAn.float()
        B, Tp = f0.shape
        L = (Tp - 1) * c.hop
        # harmonic amplitudes: sample the log-envelope at h*f0 (mel-spaced linear interp)
        freqs = f0.unsqueeze(-1) * self.harm                                   # (B,Tp,H)
        alive = (freqs < 0.95 * c.sr / 2).float()
        pos = (hz2mel(freqs) / self.nyq_mel).clamp(0, 1) * (c.kh - 1)
        i0 = pos.floor().clamp(max=c.kh - 2)
        fr = pos - i0
        i0 = i0.long()
        e = logAh.gather(-1, i0) * (1 - fr) + logAh.gather(-1, i0 + 1) * fr
        amp = e.exp() * alive * voiced.unsqueeze(-1)
        amp = F.interpolate(amp.transpose(1, 2), size=L + 1, mode="linear", align_corners=True)[..., :L]
        f0s = F.interpolate(f0.unsqueeze(1), size=L + 1, mode="linear", align_corners=True)[:, 0, :L]
        # phase: fp32 cumsum is accurate enough for short training crops and much
        # faster than fp64 on commodity GPUs; long utterances (inference) keep fp64.
        pd = torch.float64 if L > 32768 else torch.float32
        phase = torch.cumsum(f0s.to(pd) * (2 * math.pi / c.sr), dim=1).remainder(2 * math.pi).float()
        harm = (amp * torch.sin(phase.unsqueeze(1) * self.harm.view(1, -1, 1) + self.theta.view(1, -1, 1))).sum(1)
        # noise: white noise shaped by a time-varying magnitude filter in the STFT domain
        if noise is None:
            noise = torch.randn(B, L, device=f0.device)
        S = torch.stft(noise, c.n_fft, c.hop, window=self.win, center=True, return_complex=True)
        mag = (logAn @ self.interp).exp().transpose(1, 2)
        nz = torch.istft(S * mag, c.n_fft, c.hop, window=self.win, center=True, length=L)
        y = harm + nz
        return F.conv1d(y.unsqueeze(1), self.fir.view(1, 1, -1), padding=c.fir_taps // 2).squeeze(1)


class TinyTTS(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.cfg = c
        d = c.d
        self.emb = nn.Embedding(c.vocab, d, padding_idx=0)
        self.enc = LoopedEncoder(d, c.enc_k, c.enc_loops)
        self.ctx = ContextGRU(d, c.gru_h)
        self.dur_pred = VarPred(d, c.pred_ch)
        self.pit_pred = VarPred(d, c.pred_ch)
        self.pit_emb = nn.Conv1d(1, d, 1)
        self.sigma_raw = nn.Parameter(torch.tensor(1.8546))                    # softplus -> 2.0 frames
        self.dec = Decoder(c)
        self.head = nn.Conv1d(d, 2 + c.kh + c.kn, 1)
        self.dsp = HNSynth(c)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()
            self.head.bias[2:2 + c.kh] = torch.linspace(-3.5, -7.0, c.kh)
            self.head.bias[2 + c.kh:] = torch.linspace(-4.0, -6.0, c.kn)

    def encode(self, ids, lens_cpu):
        mask = (ids != 0).unsqueeze(1).to(self.emb.weight.dtype)
        x = self.emb(ids).transpose(1, 2) * mask
        x = self.enc(x, mask)
        x = self.ctx(x, lens_cpu) * mask
        return x, mask, self.dur_pred(x, mask), self.pit_pred(x, mask)

    def decode(self, x, dur, mask, T, fmask):
        up, ex = gaussian_upsample(x, dur, F.softplus(self.sigma_raw), T, mask.squeeze(1) > 0)
        return self.dec(torch.cat([up, ex], 1) * fmask, fmask)

    def heads(self, h):
        c = self.cfg
        o = self.head(h).float()
        logAh = o[:, 2:2 + c.kh].transpose(1, 2).clamp(-14, 2)
        logAn = o[:, 2 + c.kh:].transpose(1, 2).clamp(-14, 2)
        return o[:, 0], o[:, 1], logAh, logAn

    def forward(self, ids, lens_cpu, dur, pitch_tok, T, fmask):
        """Training pass (teacher-forced durations and token-level pitch)."""
        x, mask, logd, pit = self.encode(ids, lens_cpu)
        x = x + self.pit_emb(pitch_tok.unsqueeze(1).to(x.dtype)) * mask
        h = self.decode(x, dur, mask, T, fmask)
        f0n, vuv, logAh, logAn = self.heads(h)
        return dict(logd=logd.float(), pit=pit.float(), f0n=f0n, vuv=vuv, logAh=logAh, logAn=logAn)

    @torch.no_grad()
    def infer(self, ids, speed=1.0, pitch_shift=0.0):
        """ids (1,N) long -> waveform (samples,) float in [-1,1]."""
        c = self.cfg
        lens = torch.tensor([ids.shape[1]])
        x, mask, logd, pit = self.encode(ids, lens)
        dur = ((logd.float().exp() - 1) / speed).round().clamp(min=1)
        x = x + self.pit_emb(pit.unsqueeze(1).to(x.dtype)) * mask
        T = int(dur.sum().item())
        fmask = torch.ones(1, 1, T, device=ids.device, dtype=x.dtype)
        h = self.decode(x, dur, mask, T, fmask)
        f0n, vuv, logAh, logAn = self.heads(h)
        f0 = torch.exp(f0n * c.f0_std + c.f0_mean + pitch_shift).clamp(60, 600)
        voiced = (torch.sigmoid(vuv) > 0.5).float()
        return self.dsp(f0, voiced, logAh, logAn)[0].clamp(-1, 1)




_VOW = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
_CON = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
PHONES = _CON + [v + s for v in _VOW for s in "012"]
PUNCT = [" ", ",", ".", "?", "!", ";", ":", "-"]
SYMBOLS = ["<pad>", "<sil>"] + PHONES + PUNCT            # 79 symbols
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
VOCAB, SIL_ID, SPACE_ID = len(SYMBOLS), SYM2ID["<sil>"], SYM2ID[" "]
_PUNCT_MAP = {"(": ",", ")": ",", "—": "-", "–": "-", "--": "-", "…": "."}
_g2p = None


def get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        for p in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict", "punkt", "punkt_tab"):
            try:
                nltk.download(p, quiet=True)
            except Exception:
                pass
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


def text_to_ids(text):
    toks = [SIL_ID]
    for t in get_g2p()(text):
        t = _PUNCT_MAP.get(t, t)
        if t in SYM2ID and t not in ("<pad>", "<sil>"):
            if t == " " and toks[-1] in (SIL_ID, SPACE_ID):
                continue
            toks.append(SYM2ID[t])
    while toks[-1] == SPACE_ID:
        toks.pop()
    toks.append(SIL_ID)
    return toks



@dataclass
class AuroraTTSOutput(ModelOutput):
    waveform: torch.Tensor = None
    sampling_rate: int = None


class AuroraTTSModel(PreTrainedModel):
    config_class = AuroraTTSConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.net = TinyTTS(Config.from_dict(config.to_dict()))
        self._buffers_ok = False
        self.post_init()

    def _init_weights(self, module):
        pass  # keep the custom initialisation done in TinyTTS.__init__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, variant=None, **kwargs):
        # variant=None -> model.safetensors (fp32); "fp16" -> model.fp16.safetensors (stock loader);
        # "int8" -> model.int8.safetensors (int8 weights + fp16 scales, dequantised to fp32 at load)
        if variant == "int8":
            return cls._from_int8(pretrained_model_name_or_path, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, variant=variant, **kwargs)

    @classmethod
    def _from_int8(cls, path, **kwargs):
        from safetensors.torch import load_file
        hub = {k: kwargs[k] for k in ("token", "revision", "cache_dir", "local_files_only", "force_download") if k in kwargs}
        config = AuroraTTSConfig.from_pretrained(path, **hub)
        if os.path.isdir(path):
            fn = os.path.join(path, "model.int8.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            fn = hf_hub_download(repo_id=path, filename="model.int8.safetensors", **hub)
        raw = load_file(fn)
        sd = {}
        for k, v in raw.items():
            if k.endswith(".scale"):
                continue
            if v.dtype == torch.int8:
                s = raw[k + ".scale"].float().reshape(-1, *([1] * (v.ndim - 1)))
                sd[k] = v.float() * s
            else:
                sd[k] = v.float()
        model = cls(config)
        model.load_state_dict(sd)
        return model.eval()

    def _ensure_buffers(self):
        # non-persistent DSP buffers are not in the checkpoint; rebuild them if they were created on "meta"
        dsp = self.net.dsp
        if self._buffers_ok and not dsp.win.is_meta:
            return
        fresh = HNSynth(self.net.cfg)
        dev = next(self.net.parameters()).device
        for n in ("harm", "win", "interp"):
            setattr(dsp, n, getattr(fresh, n).to(dev))
        self._buffers_ok = True

    @torch.no_grad()
    def forward(self, input_ids, speed=1.0, pitch_shift=0.0):
        """input_ids: LongTensor (1, N) from `tokenize`. Returns AuroraTTSOutput(waveform (samples,), sampling_rate)."""
        self._ensure_buffers()
        ids = input_ids.to(next(self.net.parameters()).device)
        wav = self.net.infer(ids, speed=speed, pitch_shift=pitch_shift)
        return AuroraTTSOutput(waveform=wav, sampling_rate=self.config.sr)

    def tokenize(self, text):
        return torch.tensor([text_to_ids(text)], dtype=torch.long)

    def synthesize(self, text, speed=1.0, pitch_shift=0.0):
        out = self(self.tokenize(text), speed=speed, pitch_shift=pitch_shift)
        return out.waveform.float().cpu().numpy(), out.sampling_rate
