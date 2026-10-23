from transformers import PretrainedConfig


class AuroraTTSConfig(PretrainedConfig):
    model_type = "auroratts"

    def __init__(self, sr=22050, hop=256, n_fft=1024, n_mels=80, vocab=79, d=56, enc_k=5, enc_loops=4,
                 gru_h=28, pred_ch=32, dec_layers=9, dec_k=5, dec_dil=(1, 2, 4), dec_ffn=3, kh=40, kn=24,
                 n_harm=80, fir_taps=65, f0_mean=5.3, f0_std=0.25, **kwargs):
        self.sr = sr
        self.hop = hop
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.vocab = vocab
        self.d = d
        self.enc_k = enc_k
        self.enc_loops = enc_loops
        self.gru_h = gru_h
        self.pred_ch = pred_ch
        self.dec_layers = dec_layers
        self.dec_k = dec_k
        self.dec_dil = list(dec_dil)
        self.dec_ffn = dec_ffn
        self.kh = kh
        self.kn = kn
        self.n_harm = n_harm
        self.fir_taps = fir_taps
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        super().__init__(**kwargs)
