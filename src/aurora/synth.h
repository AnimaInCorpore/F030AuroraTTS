// AuroraTTS harmonic + noise synthesizer in the DSP's fixed point.
//
// Streaming form, one call per network frame, as the DSP runs it. Step m takes
// frame m's head output (66 Q7.16 words: f0n, vuv, 40 harmonic log bands, 24
// noise log bands) and
//   - renders the harmonic segment between frames m-1 and m (256 samples),
//   - at odd m, shapes noise frames m-1 and m: one random complex spectrum,
//     split into the two frames' magnitudes, one inverse FFT,
//   - reads out segment m-3 of the overlap-add buffer, filters it with the
//     model's 65-tap FIR and resamples it to the Falcon's 24,585 Hz.
// Output samples are 16-bit. A pad step (past the last frame) adds nothing.
//
// Deliberate differences from the upstream torch synthesizer (see
// docs/fixed-point.md): the noise frames are drawn as independent random
// spectra (upstream takes the STFT of one white noise sequence: the same
// level and spectral shape, without the frames' mutual correlation, and one
// FFT per frame pair instead of two), with uniform rather than Gaussian
// components; a constant 2/3 window normalization (upstream divides the first
// and last 512 samples by a smaller envelope); and a 1,024-entry sine table
// without interpolation.
#pragma once
#include <stdint.h>
#include <string.h>

#include "aurora-layout.h"
#include "fixed.h"

namespace Aurora {

const int kHop = 256;
const int kFft = 1024;
const int kHarm = 80;
const int kBins = kFft / 2 + 1;
const int kResTaps = 32;
const int kResPhases = 256;

// Q3.20 signal path
inline int32_t mul20(int32_t a, int32_t b) {
	return sat24(asr64((int64_t)a * b, 20));
}
inline int32_t mul23(int32_t a, int32_t b) {
	return sat24(asr64((int64_t)a * b, 23));
}

struct Synth {
	const int32_t *img;
	// per-frame parameters, the previous frame's kept
	int32_t inc28[2];               // phase increment, cycles * 2^28 per sample
	int32_t amp[2][kHarm];          // harmonic amplitudes, Q3.20
	int32_t mag[2][kBins];          // noise bin magnitudes, Q3.20 (frames m-1, m)
	bool pad[2];
	int64_t phase48;                // cycles * 2^48, modulo 2^48
	uint32_t lcg;                   // 24-bit noise state
	int32_t ola[2048];              // overlap-add by position modulo 2048 (Q3.20)
	int32_t fre[kFft], fim[kFft];   // FFT work
	int32_t yhist[96];              // FIR input history
	int32_t xring[64];              // resampler input, by index modulo 64
	int64_t xcount;                 // FIR outputs produced
	int64_t rpos;                   // resampler time, input samples * 2^24
	int step;
	bool noNoise;                   // debug: silence the noise path (the generator still runs)

	void reset(const int32_t *image, bool silentNoise = false) {
		memset(this, 0, sizeof *this);
		img = image;
		noNoise = silentNoise;
		lcg = (uint32_t)img[AU_SYN_CONST_OFF + 4] & 0xffffff;
		// the first FIR output (position 0) looks 32 samples back: zeros
		xcount = -32;
	}

	int32_t nextNoise() {
		const uint32_t a = (uint32_t)img[AU_SYN_CONST_OFF + 5] & 0xffffff;
		const uint32_t c = (uint32_t)img[AU_SYN_CONST_OFF + 6] & 0xffffff;
		lcg = (lcg * a + c) & 0xffffff;
		const int32_t s = (int32_t)(lcg << 8) >> 8;   // signed 24-bit
		if (noNoise)
			return 0;
		return sat24(asr64((int64_t)s * img[AU_SYN_CONST_OFF + 7], 23));   // uniform, Q3.20
	}

	// ---------------------------------------------------------------- frame parameters

	int32_t expQ20(int32_t e) const {   // e Q7.16 in [-16, 4)
		return interp(img + AU_TAB_EXP_OFF, AU_TAB_EXP_LEN / 2, (uint32_t)(e + 16 * kOne), 11);
	}

	static int32_t clampBand(int32_t v) {
		return v < -14 * kOne ? -14 * kOne : v > 2 * kOne ? 2 * kOne : v;
	}

	void frameParams(const int32_t *head, int slot) {
		const int32_t *k = img + AU_SYN_CONST_OFF;
		// x = f0n std + mean - ln(sr) in Q3.20, clamped to f0 in [60, 600]:
		// the matrix row's sum kept to 20 fraction bits
		const int32_t *row = img + AU_SYN_F0ROW_OFF;
		const int64_t sum = ((int64_t)1 << (18 - AU_SYN_F0ROW_SHIFT)) + (int64_t)row[0] * kOne + (int64_t)row[1] * head[0];
		int32_t x = sat24(asr64(sum, 19 - AU_SYN_F0ROW_SHIFT));
		if (x < k[0])
			x = k[0];
		if (x > k[1])
			x = k[1];
		inc28[slot] = interp(img + AU_TAB_INC_OFF, AU_TAB_INC_LEN / 2, (uint32_t)(x + (6 << 20)), 12);
		const bool voiced = head[1] > 0;
		int32_t bands[40];
		for (int b = 0; b < 40; ++b)
			bands[b] = clampBand(head[2 + b]);
		for (int h = 1; h <= kHarm; ++h) {
			int32_t a = 0;
			const int64_t nu28 = (int64_t)h * inc28[slot];
			if (voiced && (nu28 >> 4) < k[2]) {
				int32_t pos = interp(img + AU_TAB_MELH_OFF, AU_TAB_MELH_LEN / 2, (uint32_t)(nu28 >> 4), 14);
				if (pos > 39 * kOne)
					pos = 39 * kOne;
				int i0 = pos >> 16;
				if (i0 > 38)
					i0 = 38;
				const int32_t fr = pos - (i0 << 16);
				const int32_t e = add(bands[i0], mul(sub(bands[i0 + 1], bands[i0]), fr));
				a = expQ20(e);
			}
			amp[slot][h - 1] = a;
		}
		int32_t nb[24];
		for (int b = 0; b < 24; ++b)
			nb[b] = clampBand(head[42 + b]);
		const int32_t *map = img + AU_TAB_NMAP_OFF;
		for (int i = 0; i < kBins; ++i) {
			const int i0 = map[2 * i];
			const int32_t e = add(nb[i0], mul(sub(nb[i0 + 1], nb[i0]), map[2 * i + 1]));
			mag[slot][i] = expQ20(e);
		}
		pad[slot] = false;
	}

	void padParams(int slot) {
		inc28[slot] = inc28[slot ^ 1];
		memset(amp[slot], 0, sizeof amp[slot]);
		memset(mag[slot], 0, sizeof mag[slot]);
		pad[slot] = true;
	}

	// ---------------------------------------------------------------- harmonics

	// Segment q = m-1: positions (m-1)*256 .. m*256-1, frames in slots prev, cur.
	void harmonics(int prev, int cur, int64_t base) {
		const int32_t *sine = img + AU_TAB_SIN_OFF;
		const int32_t *theta = img + AU_SYN_THETA_OFF;   // sine-table steps, 0..1023
		int32_t da[kHarm];
		for (int h = 0; h < kHarm; ++h)
			da[h] = sub(amp[cur][h], amp[prev][h]);
		int64_t inc48 = (int64_t)inc28[prev] << 20;
		const int64_t dinc48 = ((int64_t)inc28[cur] - inc28[prev]) << 12;
		const int64_t mask48 = ((int64_t)1 << 48) - 1;
		for (int k = 0; k < kHop; ++k) {
			phase48 = (phase48 + inc48) & mask48;
			inc48 += dinc48;
			const int32_t ph = (int32_t)((uint32_t)(phase48 >> 24) << 8) >> 8;
			// sine index: floor of the harmonic's phase h ph (mod 2^24, as a
			// signed fraction of a cycle) over 2^14, plus its offset, mod 1024
			int64_t acc0 = 0, acc1 = 0;
			uint32_t phh = 0;
			for (int h = 0; h < kHarm; ++h) {
				phh += (uint32_t)ph;
				const int32_t i = (((int32_t)(phh << 8) >> 8) >> 14) + theta[h];
				const int32_t sv = sine[i & 1023];
				acc0 += 2 * (int64_t)amp[prev][h] * sv;
				acc1 += 2 * (int64_t)da[h] * sv;
			}
			const int32_t s0 = sat24(acc0 >> 24), s1 = sat24(acc1 >> 24);
			const int32_t v = sat24((int64_t)s0 + asr64((int64_t)s1 * (k << 15), 23));
			int32_t &o = ola[(base + k) & 2047];
			o = add(o, v);
		}
	}

	// ---------------------------------------------------------------- noise

	void fft(bool inverse) {
		const int32_t *sine = img + AU_TAB_SIN_OFF;
		for (int i = 1, j = 0; i < kFft; ++i) {   // bit reversal
			int bit = kFft >> 1;
			for (; j & bit; bit >>= 1)
				j ^= bit;
			j |= bit;
			if (i < j) {
				int32_t t = fre[i]; fre[i] = fre[j]; fre[j] = t;
				t = fim[i]; fim[i] = fim[j]; fim[j] = t;
			}
		}
		for (int size = 2; size <= kFft; size <<= 1) {
			const int half = size >> 1, stride = kFft / size;
			for (int start = 0; start < kFft; start += size)
				for (int k = 0; k < half; ++k) {
					const int32_t wr = sine[(k * stride + 256) & 1023];
					const int32_t ws = sine[k * stride];
					const int32_t wi = inverse ? ws : -ws;
					int32_t &ar = fre[start + k], &ai = fim[start + k];
					int32_t &br = fre[start + k + half], &bi = fim[start + k + half];
					const int64_t tr = (int64_t)wr * br - (int64_t)wi * bi;
					const int64_t ti = (int64_t)wr * bi + (int64_t)wi * br;
					const int64_t a2r = (int64_t)ar << 23, a2i = (int64_t)ai << 23;
					if (inverse) {
						br = sat24(asr64(a2r - tr, 23));
						bi = sat24(asr64(a2i - ti, 23));
						ar = sat24(asr64(a2r + tr, 23));
						ai = sat24(asr64(a2i + ti, 23));
					} else {
						br = sat24(asr64(a2r - tr, 24));
						bi = sat24(asr64(a2i - ti, 24));
						ar = sat24(asr64(a2r + tr, 24));
						ai = sat24(asr64(a2i + ti, 24));
					}
				}
		}
	}

	// Noise frames f-1 (slot prev) and f (slot cur), centred at (f-1)*256, f*256.
	// Z is white: Re and Im of every bin uniform, so that an unshaped frame
	// leaves the overlap-add with unit variance, as upstream's noise does.
	// Z' = p Z(k) + q conj Z(N-k) shapes the real part's spectrum by the
	// earlier frame's magnitudes and the imaginary part's by the later's.
	void noisePair(int prev, int cur, int64_t f) {
		const int64_t a0 = (f - 1) * kHop - kFft / 2, b0 = f * kHop - kFft / 2;
		for (int n = 0; n < kFft; ++n) {
			fre[n] = nextNoise();
			fim[n] = nextNoise();
		}
		for (int k = 0; k <= kFft / 2; ++k) {
			const int j = (kFft - k) & (kFft - 1);
			const int32_t p = (mag[prev][k] + mag[cur][k]) >> 1;
			const int32_t q = (mag[prev][k] - mag[cur][k]) >> 1;
			const int32_t zr = fre[k], zi = fim[k], yr = fre[j], yi = fim[j];
			fre[k] = sat24(asr64((int64_t)p * zr + (int64_t)q * yr, 20));
			fim[k] = sat24(asr64((int64_t)p * zi - (int64_t)q * yi, 20));
			if (j != k) {
				fre[j] = sat24(asr64((int64_t)p * yr + (int64_t)q * zr, 20));
				fim[j] = sat24(asr64((int64_t)p * yi - (int64_t)q * zi, 20));
			}
		}
		fft(true);
		const int32_t *wola = img + AU_TAB_WOLA_OFF;
		for (int n = 0; n < kFft; ++n) {
			int32_t &oa = ola[(a0 + n) & 2047];
			oa = add(oa, mul23(fre[n], wola[n]));
			int32_t &ob = ola[(b0 + n) & 2047];
			ob = add(ob, mul23(fim[n], wola[n]));
		}
	}

	// ---------------------------------------------------------------- output

	// One FIR input sample; returns the resampled 16-bit outputs it releases.
	int pushY(int32_t y, int16_t *out) {
		memmove(yhist, yhist + 1, 64 * sizeof(int32_t));
		yhist[64] = y;
		// FIR output for the sample 32 positions back
		const int32_t *fir = img + AU_SYN_FIR_OFF;
		int64_t s = (int64_t)1 << 22;
		for (int k = 0; k < 65; ++k)
			s += (int64_t)fir[k] * yhist[k];
		const int32_t x = sat24(asr64(s, 23));
		const int64_t xi = xcount++;
		if (xi < 0)
			return 0;   // positions before 0 hold nothing (the history is zero)
		xring[xi & 63] = x;
		int count = 0;
		const int32_t *res = img + AU_TAB_RES_OFF;
		const int64_t step = ((int64_t)1 << 24) + img[AU_SYN_CONST_OFF + 3];
		for (;;) {
			const int64_t n = rpos >> 24;
			if (n + kResTaps / 2 > xi)
				break;
			const int p = (int)((rpos >> 16) & (kResPhases - 1));
			int64_t a = (int64_t)1 << 22;
			for (int i = 0; i < kResTaps; ++i) {
				const int64_t at = n - (kResTaps / 2 - 1) + i;
				a += (int64_t)res[p * kResTaps + i] * (at < 0 ? 0 : xring[at & 63]);
			}
			int32_t v = sat24(asr64(a, 23)) >> 5;
			out[count++] = (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
			rpos += step;
		}
		return count;
	}

	// One step. head is frame `step`'s 66 words, or null past the last frame.
	// Returns the number of 16-bit samples written to out (at most 300).
	// The utterance has (T - 1) * 256 samples at 22,050 Hz; outputCount() of
	// them at the Falcon rate.
	int run(const int32_t *head, int16_t *out) {
		const int m = step++;
		const int cur = m & 1, prev = cur ^ 1;
		if (head)
			frameParams(head, cur);
		else
			padParams(cur);
		if (m >= 1 && !pad[cur])
			harmonics(prev, cur, (int64_t)(m - 1) * kHop);
		if (m & 1)
			noisePair(prev, cur, m);
		// segment m-3 is complete; segments before position 0 are dropped
		int count = 0;
		const int64_t q0 = (int64_t)(m - 3) * kHop;
		for (int k = 0; k < kHop; ++k) {
			int32_t &o = ola[(q0 + k) & 2047];
			if (m >= 3)
				count += pushY(o, out + count);
			o = 0;
		}
		return count;
	}
	// outputs j with j * step < (T - 1) * 256 * 2^24
	int64_t outputCount(int T) const {
		const int64_t step = ((int64_t)1 << 24) + img[AU_SYN_CONST_OFF + 3];
		const int64_t len = ((int64_t)(T - 1) * kHop) << 24;
		return (len + step - 1) / step;
	}
};

}  // namespace Aurora
