"""Run the original AuroraTTS-100K PyTorch code without the transformers package.

The upstream modeling file only needs transformers for its HF wrapper class;
TinyTTS, Config and the text frontend are plain PyTorch. A stub module stands
in for transformers so the file can be imported unchanged.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "third_party" / "auroratts"


def _load_module():
    if "auroratts_upstream" in sys.modules:
        return sys.modules["auroratts_upstream"]
    stub = types.ModuleType("transformers")
    stub.PreTrainedModel = torch.nn.Module
    stub.PretrainedConfig = object
    utils = types.ModuleType("transformers.utils")
    utils.ModelOutput = dict
    stub.utils = utils
    sys.modules.setdefault("transformers", stub)
    sys.modules.setdefault("transformers.utils", utils)
    src = (MODEL_DIR / "modeling_auroratts.py").read_text()
    src = src.replace("from .configuration_auroratts import AuroraTTSConfig", "AuroraTTSConfig = object")
    mod = types.ModuleType("auroratts_upstream")
    exec(compile(src, str(MODEL_DIR / "modeling_auroratts.py"), "exec"), mod.__dict__)
    sys.modules["auroratts_upstream"] = mod
    return mod


def load_model():
    import json
    from safetensors.torch import load_file
    m = _load_module()
    cfg = m.Config.from_dict(json.loads((MODEL_DIR / "config.json").read_text()))
    net = m.TinyTTS(cfg)
    sd = {k[len("net."):]: v for k, v in load_file(str(MODEL_DIR / "model.safetensors")).items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"dsp.harm", "dsp.win", "dsp.interp"}, missing
    return m, net.eval()


@torch.no_grad()
def infer_parts(net, ids, noise=None, speed=1.0):
    """The upstream infer() split open: returns every intermediate the ports compare."""
    c = net.cfg
    ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
    lens = torch.tensor([ids.shape[1]])
    x, mask, logd, pit = net.encode(ids, lens)
    dur = ((logd.float().exp() - 1) / speed).round().clamp(min=1)
    xp = x + net.pit_emb(pit.unsqueeze(1).to(x.dtype)) * mask
    T = int(dur.sum().item())
    fmask = torch.ones(1, 1, T)
    h = net.decode(xp, dur, mask, T, fmask)
    f0n, vuv, logAh, logAn = net.heads(h)
    f0 = torch.exp(f0n * c.f0_std + c.f0_mean).clamp(60, 600)
    voiced = (torch.sigmoid(vuv) > 0.5).float()
    if noise is not None:
        noise = torch.as_tensor(noise, dtype=torch.float32).view(1, -1)
    wav = net.dsp(f0, voiced, logAh, logAn, noise=noise)[0].clamp(-1, 1)
    out = dict(enc=x[0], logd=logd[0], pit=pit[0], dur=dur[0], dec=h[0], f0=f0[0], voiced=voiced[0],
               logAh=logAh[0], logAn=logAn[0], wav=wav)
    return {k: v.numpy().astype(np.float64) for k, v in out.items()}
