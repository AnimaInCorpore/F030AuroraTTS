// AuroraTTS-100K network in the DSP's fixed point: phoneme ids to the head's
// per-frame synthesis parameters.
//
// This is the reference the DSP passes must reproduce. It computes each layer
// over the whole sequence; the DSP streams a layer frame by frame through a
// ring, which gives the same words because every operation's integer sum is
// exact and the sequence edges are zero-padded the same way.
//
// The length regulator (durations and the Gaussian upsampling) runs on the
// 68030 in this same code.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "aurora-layout.h"
#include "fixed.h"

namespace Aurora {

const int kD = 56;          // channels
const int kHead = 66;       // f0n, vuv, 40 harmonic bands, 24 noise bands
const int kMaxDur = 255;

#define AU_BLOCK(name) (img + AU_##name##_OFF)

// 32 x 32 -> 64 multiply and 64 / 32 divide: single instructions on the
// 68030 (muls.l, divu.l), plain C elsewhere; the results are the same.
inline int64_t mul32x32(int32_t a, int32_t b) {
	return (int64_t)a * b;
}

inline uint32_t udiv64by32(uint64_t n, uint32_t d) {
#if defined(__mc68020__) || defined(__mc68030__) || defined(__mc68040__)
	uint32_t hi = (uint32_t)(n >> 32), lo = (uint32_t)n;
	__asm__("divu.l %2,%0:%1" : "+d"(hi), "+d"(lo) : "d"(d));
	return lo;
#else
	return (uint32_t)(n / d);
#endif
}

struct Model {
	const int32_t *img;     // AURORA.BIN as sign-extended words

	// ---------------------------------------------------------------- helpers

	// Depthwise convolution, kernel 5, zero-padded: x is (T, kD) row-major.
	void dwconv(const int32_t *x, int T, int dil, const int32_t *rows, int shift, int32_t *y) const {
		for (int t = 0; t < T; ++t)
			for (int c = 0; c < kD; ++c) {
				int32_t taps[5];
				for (int k = 0; k < 5; ++k) {
					const int u = t + (k - 2) * dil;
					taps[k] = u < 0 || u >= T ? 0 : x[u * kD + c];
				}
				y[t * kD + c] = matrow(taps, 1, 5, rows + c * 6, shift);
			}
	}

	void norm(const int32_t *x, int T, const int32_t *gb, int shift, int32_t *y) const {
		for (int t = 0; t < T; ++t)
			layernorm(x + t * kD, kD, gb, shift, AU_BLOCK(TAB_RSQRT), y + t * kD);
	}

	// ---------------------------------------------------------------- encoder

	// x: (N, kD) out
	void encode(const uint8_t *ids, int N, int32_t *x, int32_t *work) const {
		int32_t *y = work, *z = work + N * kD;
		int32_t g[2 * kD], h[2 * kD];
		for (int n = 0; n < N; ++n)
			memcpy(x + n * kD, AU_BLOCK(EMB) + ids[n] * kD, kD * sizeof(int32_t));
		static const int n1off[4] = {AU_ENC_N1_0_OFF, AU_ENC_N1_1_OFF, AU_ENC_N1_2_OFF, AU_ENC_N1_3_OFF};
		static const int n1sh[4] = {AU_ENC_N1_0_SHIFT, AU_ENC_N1_1_SHIFT, AU_ENC_N1_2_SHIFT, AU_ENC_N1_3_SHIFT};
		static const int n2off[4] = {AU_ENC_N2_0_OFF, AU_ENC_N2_1_OFF, AU_ENC_N2_2_OFF, AU_ENC_N2_3_OFF};
		static const int n2sh[4] = {AU_ENC_N2_0_SHIFT, AU_ENC_N2_1_SHIFT, AU_ENC_N2_2_SHIFT, AU_ENC_N2_3_SHIFT};
		for (int loop = 0; loop < 4; ++loop) {
			norm(x, N, img + n1off[loop], n1sh[loop], y);
			dwconv(y, N, 1, AU_BLOCK(ENC_DW), AU_ENC_DW_SHIFT, z);
			for (int n = 0; n < N; ++n) {
				int32_t *xn = x + n * kD;
				matvec(z + n * kD, kD, AU_BLOCK(ENC_GLU), 2 * kD, AU_ENC_GLU_SHIFT, g);
				for (int c = 0; c < kD; ++c)
					xn[c] = add(xn[c], mul(g[c], activate(ACT_SIGMOID, AU_BLOCK(TAB_SIGMOID), g[kD + c])));
				int32_t ln[kD], f[kD];
				layernorm(xn, kD, img + n2off[loop], n2sh[loop], AU_BLOCK(TAB_RSQRT), ln);
				matvec(ln, kD, AU_BLOCK(ENC_FFA), 2 * kD, AU_ENC_FFA_SHIFT, h);
				for (int c = 0; c < 2 * kD; ++c)
					h[c] = activate(ACT_GELU, AU_BLOCK(TAB_GELU), h[c]);
				matvec(h, 2 * kD, AU_BLOCK(ENC_FFB), kD, AU_ENC_FFB_SHIFT, f);
				for (int c = 0; c < kD; ++c)
					xn[c] = add(xn[c], f[c]);
			}
		}
	}

	// One GRU step: h (28) from input y (56).
	void gruStep(const int32_t *y, int32_t *h, const int32_t *ih, int ihs, const int32_t *hh, int hhs) const {
		const int H = kD / 2;
		int32_t gi[3 * H], gh[3 * H];
		matvec(y, kD, ih, 3 * H, ihs, gi);
		matvec(h, H, hh, 3 * H, hhs, gh);
		for (int j = 0; j < H; ++j) {
			const int32_t r = activate(ACT_SIGMOID, AU_BLOCK(TAB_SIGMOID), add(gi[j], gh[j]));
			const int32_t z = activate(ACT_SIGMOID, AU_BLOCK(TAB_SIGMOID), add(gi[H + j], gh[H + j]));
			const int32_t n = activate(ACT_TANH, AU_BLOCK(TAB_TANH), add(gi[2 * H + j], mul(r, gh[2 * H + j])));
			h[j] = add(n, mul(z, sub(h[j], n)));
		}
	}

	// x += [forward GRU, backward GRU] over layer-normed x
	void context(int32_t *x, int N, int32_t *work) const {
		const int H = kD / 2;
		int32_t *y = work, *o = work + N * kD;
		norm(x, N, AU_BLOCK(CTX_NORM), AU_CTX_NORM_SHIFT, y);
		int32_t h[kD / 2];
		memset(h, 0, sizeof h);
		for (int n = 0; n < N; ++n) {
			gruStep(y + n * kD, h, AU_BLOCK(GRU_F_IH), AU_GRU_F_IH_SHIFT, AU_BLOCK(GRU_F_HH), AU_GRU_F_HH_SHIFT);
			memcpy(o + n * kD, h, sizeof h);
		}
		memset(h, 0, sizeof h);
		for (int n = N - 1; n >= 0; --n) {
			gruStep(y + n * kD, h, AU_BLOCK(GRU_B_IH), AU_GRU_B_IH_SHIFT, AU_BLOCK(GRU_B_HH), AU_GRU_B_HH_SHIFT);
			memcpy(o + n * kD + H, h, sizeof h);
		}
		for (int i = 0; i < N * kD; ++i)
			x[i] = add(x[i], o[i]);
	}

	// A variance predictor: one output per token.
	void predict(const int32_t *x, int N, int which, int32_t *out, int32_t *work) const {
		int32_t *y = work, *z = work + N * kD;
		const int32_t *n_ = which ? AU_BLOCK(PIT_N) : AU_BLOCK(DUR_N);
		norm(x, N, n_, which ? AU_PIT_N_SHIFT : AU_DUR_N_SHIFT, y);
		dwconv(y, N, 1, which ? AU_BLOCK(PIT_DW) : AU_BLOCK(DUR_DW), which ? AU_PIT_DW_SHIFT : AU_DUR_DW_SHIFT, z);
		for (int n = 0; n < N; ++n) {
			int32_t a[32], b[32];
			matvec(z + n * kD, kD, which ? AU_BLOCK(PIT_P1) : AU_BLOCK(DUR_P1), 32,
			       which ? AU_PIT_P1_SHIFT : AU_DUR_P1_SHIFT, a);
			for (int j = 0; j < 32; ++j)
				a[j] = relu(a[j]);
			matvec(a, 32, which ? AU_BLOCK(PIT_P2) : AU_BLOCK(DUR_P2), 32, which ? AU_PIT_P2_SHIFT : AU_DUR_P2_SHIFT, b);
			for (int j = 0; j < 32; ++j)
				b[j] = relu(b[j]);
			matvec(b, 32, which ? AU_BLOCK(PIT_O) : AU_BLOCK(DUR_O), 1, which ? AU_PIT_O_SHIFT : AU_DUR_O_SHIFT, out + n);
		}
	}

	// The whole phoneme-rate side: x (N, kD) ready for upsampling, durations.
	// work holds 2 N kD words.
	void encodeAll(const uint8_t *ids, int N, int32_t *x, int32_t *logd, int32_t *pit, uint8_t *dur,
	               int32_t *work) const {
		encode(ids, N, x, work);
		context(x, N, work);
		predict(x, N, 0, logd, work);
		predict(x, N, 1, pit, work);
		for (int n = 0; n < N; ++n) {
			int32_t e[kD];
			const int32_t in[2] = {pit[n], 0};
			matvec(in, 2, AU_BLOCK(PIT_EMB), kD, AU_PIT_EMB_SHIFT, e);
			for (int c = 0; c < kD; ++c)
				x[n * kD + c] = add(x[n * kD + c], e[c]);
			dur[n] = (uint8_t)durationOf(logd[n]);
		}
	}

	// round(exp(logd) - 1), at least 1, at most 255
	int durationOf(int32_t logd) const {
		const int32_t *th = AU_BLOCK(TAB_DURTH);
		int d = 1;
		while (d < kMaxDur && logd >= th[d + 1])
			++d;
		return d;
	}

	// ---------------------------------------------------------------- length regulator

	// Frame t's decoder input: the softmax-weighted token vectors, then the
	// relative position and log duration features (kD + 2 words).
	//   d_n = 2 (t + 1/2) - 2 centre_n, q_n = d_n^2 - min d^2,
	//   E_n = upw[q_n] (zero past the table), W = sum E,
	//   w_n = floor(E_n 2^23 / W): the softmax weight, Q0.23,
	//   out = round(sum w_n v_n / 2^23) for v = x (per channel), the position
	//   floor(d_n 2^15 / dur_n) and log(dur_n).
	// Only tokens with E_n > 0 take part; the centres increase, so they are a
	// run around the nearest one, which *near tracks from frame to frame.
	void upsampleFrame(const int32_t *x, const uint8_t *dur, const int32_t *center2, int N, int t, int32_t *out,
	                   int *near) const {
		const int32_t *w = AU_BLOCK(TAB_UPW);
		const int32_t *ld = AU_BLOCK(TAB_LOGDUR);
		const int32_t tt2 = 2 * t + 1;
		int n0 = *near;
		if (n0 < 0 || n0 >= N)
			n0 = 0;
		while (n0 + 1 < N && iabs(tt2 - center2[n0 + 1]) <= iabs(tt2 - center2[n0]))
			++n0;
		*near = n0;
		const int32_t d0 = tt2 - center2[n0];
		const int32_t dmin = d0 * d0;
		int lo = n0, hi = n0;
		while (lo > 0 && sq(tt2 - center2[lo - 1]) - dmin < AU_TAB_UPW_LEN)
			--lo;
		while (hi + 1 < N && sq(tt2 - center2[hi + 1]) - dmin < AU_TAB_UPW_LEN)
			++hi;
		uint32_t W = 0;
		for (int n = lo; n <= hi; ++n)
			W += (uint32_t)w[sq(tt2 - center2[n]) - dmin];
		int64_t acc[kD + 2];
		for (int c = 0; c < kD + 2; ++c)
			acc[c] = (int64_t)1 << 22;
		for (int n = lo; n <= hi; ++n) {
			const int32_t d = tt2 - center2[n];
			const int32_t e = w[sq(d) - dmin];
			const int32_t wn = (int32_t)udiv64by32((uint64_t)(uint32_t)e << 23, W);
			const int32_t *xn = x + n * kD;
			for (int c = 0; c < kD; ++c)
				acc[c] += mul32x32(wn, xn[c]);
			acc[kD] += mul32x32(wn, sdiv32(d * 32768, dur[n]));
			acc[kD + 1] += mul32x32(wn, ld[dur[n]]);
		}
		for (int c = 0; c < kD + 2; ++c)
			out[c] = sat24(acc[c] >> 23);
	}

	static int32_t iabs(int32_t v) { return v < 0 ? -v : v; }
	static int32_t sq(int32_t v) { return v * v; }
	// floor(a / b) for b > 0
	static int32_t sdiv32(int32_t a, int32_t b) {
		int32_t q = a / b;
		if (a % b != 0 && a < 0)
			--q;
		return q;
	}

	// ---------------------------------------------------------------- decoder

	// in: (T, kD + 2) upsampled frames; out: (T, kHead). work holds 2 T kD words.
	void decode(const int32_t *in, int T, int32_t *out, int32_t *work) const {
		int32_t *x = work, *y = work + T * kD;
		for (int t = 0; t < T; ++t)
			matvec(in + t * (kD + 2), kD + 2, AU_BLOCK(DEC_INP), kD, AU_DEC_INP_SHIFT, x + t * kD);
		static const int dwoff[9] = {AU_DEC_DW_0_OFF, AU_DEC_DW_1_OFF, AU_DEC_DW_2_OFF, AU_DEC_DW_3_OFF, AU_DEC_DW_4_OFF,
		                             AU_DEC_DW_5_OFF, AU_DEC_DW_6_OFF, AU_DEC_DW_7_OFF, AU_DEC_DW_8_OFF};
		static const int dwsh[9] = {AU_DEC_DW_0_SHIFT, AU_DEC_DW_1_SHIFT, AU_DEC_DW_2_SHIFT, AU_DEC_DW_3_SHIFT,
		                            AU_DEC_DW_4_SHIFT, AU_DEC_DW_5_SHIFT, AU_DEC_DW_6_SHIFT, AU_DEC_DW_7_SHIFT,
		                            AU_DEC_DW_8_SHIFT};
		static const int lnoff[9] = {AU_DEC_LN_0_OFF, AU_DEC_LN_1_OFF, AU_DEC_LN_2_OFF, AU_DEC_LN_3_OFF, AU_DEC_LN_4_OFF,
		                             AU_DEC_LN_5_OFF, AU_DEC_LN_6_OFF, AU_DEC_LN_7_OFF, AU_DEC_LN_8_OFF};
		static const int lnsh[9] = {AU_DEC_LN_0_SHIFT, AU_DEC_LN_1_SHIFT, AU_DEC_LN_2_SHIFT, AU_DEC_LN_3_SHIFT,
		                            AU_DEC_LN_4_SHIFT, AU_DEC_LN_5_SHIFT, AU_DEC_LN_6_SHIFT, AU_DEC_LN_7_SHIFT,
		                            AU_DEC_LN_8_SHIFT};
		static const int faoff[3] = {AU_DEC_FFA_0_OFF, AU_DEC_FFA_1_OFF, AU_DEC_FFA_2_OFF};
		static const int fash[3] = {AU_DEC_FFA_0_SHIFT, AU_DEC_FFA_1_SHIFT, AU_DEC_FFA_2_SHIFT};
		static const int fboff[3] = {AU_DEC_FFB_0_OFF, AU_DEC_FFB_1_OFF, AU_DEC_FFB_2_OFF};
		static const int fbsh[3] = {AU_DEC_FFB_0_SHIFT, AU_DEC_FFB_1_SHIFT, AU_DEC_FFB_2_SHIFT};
		static const int dil[3] = {1, 2, 4};
		for (int i = 0; i < 9; ++i) {
			dwconv(x, T, dil[i % 3], img + dwoff[i], dwsh[i], y);
			for (int t = 0; t < T; ++t) {
				int32_t ln[kD], h[2 * kD], f[kD];
				layernorm(y + t * kD, kD, img + lnoff[i], lnsh[i], AU_BLOCK(TAB_RSQRT), ln);
				matvec(ln, kD, img + faoff[i % 3], 2 * kD, fash[i % 3], h);
				for (int c = 0; c < 2 * kD; ++c)
					h[c] = activate(ACT_GELU, AU_BLOCK(TAB_GELU), h[c]);
				matvec(h, 2 * kD, img + fboff[i % 3], kD, fbsh[i % 3], f);
				for (int c = 0; c < kD; ++c)
					x[t * kD + c] = add(x[t * kD + c], f[c]);
			}
		}
		for (int t = 0; t < T; ++t) {
			int32_t ln[kD];
			layernorm(x + t * kD, kD, AU_BLOCK(DEC_OUT_LN), AU_DEC_OUT_LN_SHIFT, AU_BLOCK(TAB_RSQRT), ln);
			matvec(ln, kD, AU_BLOCK(HEAD), kHead, AU_HEAD_SHIFT, out + t * kHead);
		}
	}
};

// Twice each token's centre: 2 cumsum(dur) - dur.
inline void centres(const uint8_t *dur, int N, int32_t *center2, int *T) {
	int32_t cum = 0;
	for (int n = 0; n < N; ++n) {
		cum += dur[n];
		center2[n] = 2 * cum - dur[n];
	}
	*T = cum;
}

}  // namespace Aurora
