"""The network's forward pass written out op by op, with every intermediate
recorded, so fixed-point formats can be chosen from measured ranges.
Checked against the upstream infer() by check_equal()."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def ln(x, n):
    return F.layer_norm(x.transpose(1, 2), (x.shape[1],), n.w, n.b).transpose(1, 2)


@torch.no_grad()
def forward_taps(net, ids):
    taps = {}

    def tap(name, t):
        a = t.detach().abs().max().item() if t.numel() else 0.0
        taps[name] = max(taps.get(name, 0.0), a)
        return t

    c = net.cfg
    ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
    x = tap("emb", net.emb(ids).transpose(1, 2))
    e = net.enc
    for i in range(e.loops):
        y = tap("enc.n1", ln(x, e.n1[i]))
        y = tap("enc.dw", e.dw(y))
        g = tap("enc.glu_pre", e.glu(y))
        x = tap("enc.x1", x + F.glu(g, dim=1))
        y = tap("enc.n2", ln(x, e.n2[i]))
        hdn = tap("enc.ffn_a", e.ffn.a(y))
        hdn = tap("enc.ffn_gelu", F.gelu(hdn))
        x = tap("enc.x2", x + tap("enc.ffn_b", e.ffn.b(hdn)))
    y = tap("ctx.norm", ln(x, net.ctx.norm)).transpose(1, 2)
    g = net.ctx.gru
    N = y.shape[1]
    outs = []
    for d, sfx in ((0, ""), (1, "_reverse")):
        Wi, Wh = getattr(g, "weight_ih_l0" + sfx), getattr(g, "weight_hh_l0" + sfx)
        bi, bh = getattr(g, "bias_ih_l0" + sfx), getattr(g, "bias_hh_l0" + sfx)
        h = torch.zeros(28)
        o = [None] * N
        order = range(N) if d == 0 else range(N - 1, -1, -1)
        for t in order:
            gi = tap("gru.gi", Wi @ y[0, t] + bi)
            gh = tap("gru.gh", Wh @ h + bh)
            r = torch.sigmoid(gi[:28] + gh[:28])
            z = torch.sigmoid(gi[28:56] + gh[28:56])
            tap("gru.rz_pre", gi[:56] + gh[:56])
            npre = tap("gru.n_pre", gi[56:] + r * gh[56:])
            n = torch.tanh(npre)
            h = (1 - z) * n + z * h
            o[t] = h
        outs.append(torch.stack(o, 1))
    x = tap("ctx.out", x + torch.cat(outs, 0).unsqueeze(0))
    res = {}
    for name, vp in (("dur", net.dur_pred), ("pit", net.pit_pred)):
        y = tap(name + ".n", ln(x, vp.n))
        y = tap(name + ".dw", vp.dw(y))
        y = tap(name + ".p1", F.relu(vp.p1(y)))
        y = tap(name + ".p2", F.relu(vp.p2(y)))
        res[name] = tap(name + ".o", vp.o(y).squeeze(1))
    logd, pit = res["dur"], res["pit"]
    dur = (logd.exp() - 1).round().clamp(min=1)
    tap("dur", dur)
    x = tap("enc.final", x + net.pit_emb(pit.unsqueeze(1)))
    T = int(dur.sum().item())
    sigma = F.softplus(net.sigma_raw)
    from auroratts_upstream import gaussian_upsample
    up, ex = gaussian_upsample(x, dur, sigma, T, torch.ones(1, x.shape[2], dtype=torch.bool))
    tap("up", up); tap("ex.pos", ex[:, 0]); tap("ex.ld", ex[:, 1])
    dec = net.dec
    x = tap("dec.inp", dec.inp(torch.cat([up, ex], 1)))
    for i in range(dec.L):
        y = tap(f"dec.dw", dec.dw[i](x))
        y = tap(f"dec.ln", ln(y, dec.ln[i]))
        f = dec.ffn[i % dec.G]
        hdn = tap("dec.ffn_a", f.a(y))
        hdn = tap("dec.ffn_gelu", F.gelu(hdn))
        y = tap("dec.ffn_b", f.b(hdn))
        x = tap(f"dec.x{i}", x + y)
    h = tap("dec.out_ln", ln(x, dec.out_ln))
    o = tap("head", net.head(h))
    return taps, h, dur


def check_equal(net, ids):
    import aurora_torch as A
    ref = A.infer_parts(net, ids)
    _, h, dur = forward_taps(net, ids)
    assert np.array_equal(dur[0].numpy(), ref["dur"])
    return float(np.abs(h[0].numpy() - ref["dec"]).max())
