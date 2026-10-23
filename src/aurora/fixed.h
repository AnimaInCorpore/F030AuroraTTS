// AuroraTTS fixed-point arithmetic: the DSP56001's, word for word.
//
// Every function here defines what one DSP operation produces. The host
// reference and the 68030 program both use these definitions; the DSP kernel
// must match them exactly. Integer accumulation is exact (the DSP's 56-bit
// accumulator never overflows in these sums), so only the final rounding,
// shift and limiting of each operation matter, not the order of its terms.
//
// Formats
//   Q7.16   activations: word / 2^16, range [-128, 128)
//   matrix  per block shift s: word / 2^(23-s); rows are [bias, w0 .. w(n-1)]
//           and the bias multiplies 1.0
//   table   (value, slope) pairs, linear interpolation
#pragma once
#include <stdint.h>

namespace Aurora {

const int32_t kOne = 1 << 16;          // 1.0 in Q7.16
const int32_t kMax24 = (1 << 23) - 1;
const int32_t kMin24 = -(1 << 23);

inline int32_t sat24(int64_t v) {
	return v > kMax24 ? kMax24 : v < kMin24 ? kMin24 : (int32_t)v;
}

// floor(v / 2^n) for n >= 0, as the DSP's arithmetic shifts give it
inline int64_t asr64(int64_t v, int n) {
	return v >> n;   // arithmetic on every compiler this project uses
}

// y[o] = sat(round((bias + sum w x) / 2^(23-s))), the DSP's matrix row: the
// accumulator starts at the rounding constant, the bias row element multiplies
// 1.0, every weight its input, then s left shifts and a limited move.
inline void matvec(const int32_t *x, int n, const int32_t *rows, int out, int shift, int32_t *y) {
	const int64_t round = (int64_t)1 << (22 - shift);
	for (int o = 0; o < out; ++o) {
		const int32_t *r = rows + o * (n + 1);
		int64_t s = round + (int64_t)r[0] * kOne;
		for (int i = 0; i < n; ++i)
			s += (int64_t)r[i + 1] * x[i];
		y[o] = sat24(asr64(s, 23 - shift));
	}
}

// One row against a strided input: the depthwise convolution's tap sum.
inline int32_t matrow(const int32_t *x, int stride, int n, const int32_t *row, int shift) {
	int64_t s = ((int64_t)1 << (22 - shift)) + (int64_t)row[0] * kOne;
	for (int i = 0; i < n; ++i)
		s += (int64_t)row[i + 1] * x[i * stride];
	return sat24(asr64(s, 23 - shift));
}

inline int32_t add(int32_t a, int32_t b) {
	return sat24((int64_t)a + b);
}

inline int32_t sub(int32_t a, int32_t b) {
	return sat24((int64_t)a - b);
}

// Q7.16 product, truncated: the DSP's mpy then seven left shifts.
inline int32_t mul(int32_t a, int32_t b) {
	return sat24(asr64((int64_t)a * b, 16));
}

inline int32_t relu(int32_t a) {
	return a < 0 ? 0 : a;
}

// Linear interpolation in a table of n intervals, stored as the n start
// values then the n slopes: u is the offset from the table's start, `bits` the
// fraction bits of one interval. The DSP takes the interval and the fraction
// from one multiply (A1 and A0), then value + slope * fraction.
inline int32_t interp(const int32_t *table, int n, uint32_t u, int bits) {
	const uint32_t i = u >> bits;
	const int32_t frac = (int32_t)(u & ((1u << bits) - 1)) << (23 - bits);
	return sat24(table[i] + asr64((int64_t)table[n + i] * frac, 23));
}

enum Activation { ACT_GELU, ACT_SIGMOID, ACT_TANH };

// Tables of 1024 intervals: GELU and tanh over [-8, 8), sigmoid over [-16, 16).
inline int32_t activate(Activation kind, const int32_t *table, int32_t x) {
	switch (kind) {
	case ACT_GELU:
		if (x < -8 * kOne)
			return 0;
		if (x >= 8 * kOne)
			return x;
		return interp(table, 1024, (uint32_t)(x + 8 * kOne), 10);
	case ACT_SIGMOID:
		if (x < -16 * kOne)
			return 0;
		if (x >= 16 * kOne)
			return kOne;
		return interp(table, 1024, (uint32_t)(x + 16 * kOne), 11);
	case ACT_TANH:
	default:
		if (x < -8 * kOne)
			return -kOne;
		if (x >= 8 * kOne)
			return kOne;
		return interp(table, 1024, (uint32_t)(x + 8 * kOne), 10);
	}
}

// Layer norm over n channels (biased variance, eps 1e-5), then the per-channel
// [beta, gamma * sqrt(n)] rows. 1/sqrt(var + eps) = sqrt(n) / sqrt(S) with
// S = sum d^2 + n eps, so no division by n is needed; sqrt(n) is folded into
// gamma by the exporter. See docs/fixed-point.md.
//   mean  round(sum x * C / 2^23), C = round(2^23 / n)
//   S     sum d^2 + n * EPS, units of 2^-32 (d is Q7.16)
//   m     S / 2^e in [2^21, 2^23) for an even e (the DSP's accumulator
//         normalized to a fraction in [1/4, 1) by double shifts)
//   R     rsqrt(m / 2^22) / 2 from the table (Q0.23), m / 2^22 in [1/2, 2)
//   z     d / sqrt(S) as Q0.23: floor(d R / 2^(10 + e/2)), |z| <= 1
//   out   round((beta 2^23 + G z) / 2^(30 - s))
const int32_t kLnEps = 42950;          // round(1e-5 * 2^32)

inline void layernorm(const int32_t *x, int n, const int32_t *gb, int shift, const int32_t *rsqrt,
                      int32_t *out) {
	const int64_t c = (((int64_t)1 << 23) + n / 2) / n;
	int64_t sum = 0;
	for (int i = 0; i < n; ++i)
		sum += x[i];
	const int32_t mean = sat24(asr64(sum * c + ((int64_t)1 << 22), 23));
	int32_t d[128];
	int64_t s2 = (int64_t)n * kLnEps;
	for (int i = 0; i < n; ++i) {
		d[i] = sub(x[i], mean);
		s2 += (int64_t)d[i] * d[i];
	}
	int e = 0;
	int64_t m = s2;
	while (m >= ((int64_t)1 << 23)) {
		m >>= 2;
		e += 2;
	}
	while (m < ((int64_t)1 << 21)) {
		m <<= 2;
		e -= 2;
	}
	const int32_t r = interp(rsqrt, 384, (uint32_t)(m - ((int64_t)1 << 21)), 14);
	const int sh = 10 + e / 2;
	for (int i = 0; i < n; ++i) {
		const int32_t z = sat24(asr64((int64_t)d[i] * r, sh));
		const int32_t *row = gb + 2 * i;
		int64_t s = ((int64_t)1 << (29 - shift)) + ((int64_t)row[0] << 23) + (int64_t)row[1] * z;
		out[i] = sat24(asr64(s, 30 - shift));
	}
}

}  // namespace Aurora
