// The DSP kernel's network machine: a list of vector operations in DSP X
// memory, run once per frame (or token) as the 68030 streams frames through
// the host port.
//
// This file defines the operations, the DSP memory regions, a bump allocator
// for a pass's weights and vectors, and a host interpreter that executes a
// program on simulated X/Y memory with exactly the DSP's arithmetic (fixed.h)
// and exactly its addressing. The DSP kernel (dsp/aurora.asm) implements the
// same operations.
//
// Program words: an opcode, then its operands (addresses are DSP addresses).
//   END
//   IN     dst n                read n host words into X:dst
//   OUT    src n                send X:src .. +n-1 to the host
//   MVY    src n w rows dst sh  X:dst[r] = row r of Y:w times X:src (n inputs)
//   MVX    src n w rows dst sh  the same with the rows in X, the input in Y
//   COPYXY src dst n            X to Y
//   LN     src n gb sh dst      layer norm, [beta, gamma sqrt(n)] rows in Y:gb
//   ACT    desc src dst n       table activation; desc in X (see Activation)
//   ADD    a b dst n            saturating
//   SUB    a b dst n
//   MUL    a b dst n            Q7.16 product
//   RELU   src dst n
//   CLEAR  dst n                zero X words
//   PUSH   ring src             append frame X:src to a ring (n from the ring)
//   TAP    ring delay dst       copy the frame `delay` frames before the newest
//   DW     ring dil w sh dst    depthwise conv, kernel 5, rows [bias, 5 taps] in
//                               Y:w, the newest frame being t + 2 dil
// A ring is a descriptor in X: base, size (frames * n, the modulo), n, write
// offset. Its base is aligned to the power of two at or above its size so
// the DSP can use modulo addressing.
#pragma once
#include <stdint.h>
#include <string.h>

#include "fixed.h"

namespace Aurora {

enum Op {
	OP_END, OP_IN, OP_OUT, OP_MVY, OP_MVX, OP_COPYXY, OP_LN, OP_ACT, OP_ADD, OP_SUB, OP_MUL, OP_RELU,
	OP_CLEAR, OP_PUSH, OP_TAP, OP_DW, OP_COUNT
};

// ---------------------------------------------------------------- DSP memory

const int kMemWords = 0x4000;
// X: $0000-$003f kernel scalars, $0040-$00bf layer norm deviations,
// $00c0-$00ff spare; $0100-$01ff avoided (the data ROMs' window);
// $0200-$2fff the pass's; $3000-$3fff code (P $7000-$7fff).
const int kXScalars = 0x0000;
const int kXLnTemp = 0x0040;
const int kXFreeLo = 0x0200, kXFreeHi = 0x3000;
// Y: $0000-$00ff and $0200-$3fff the pass's.
const int kYFreeLo = 0x0200, kYFreeHi = 0x4000;
const int kYIntLo = 0x0000, kYIntHi = 0x0100;

// Kernel scalars the host writes (X).
const int kScRsqrt = 0x0010;            // layer norm's rsqrt table: values, slopes at +384

// An activation descriptor in X (8 words).
enum {
	ACTD_TABLE,     // values; slopes follow at +n
	ACTD_SLOPES,
	ACTD_LO,        // below: LOVAL
	ACTD_HI,        // at or above: HIVAL, or the input when IDENT is set
	ACTD_OFFSET,    // added before the lookup: -LO
	ACTD_MULT,      // 2^(23 - bits): the multiplier that splits interval and fraction
	ACTD_LOVAL,
	ACTD_HIVAL,
	ACTD_IDENT,
	ACTD_WORDS
};

// ---------------------------------------------------------------- allocation

struct Region {
	int lo, hi, at;
	void reset() { at = lo; }
	int take(int n, int align = 1) {
		int a = (at + align - 1) & ~(align - 1);
		if (a + n > hi)
			return -1;
		at = a + n;
		return a;
	}
	int room() const { return hi - at; }
};

struct Layout {
	Region x, y, yint;
	void reset() {
		x = Region{kXFreeLo, kXFreeHi, kXFreeLo};
		y = Region{kYFreeLo, kYFreeHi, kYFreeLo};
		yint = Region{kYIntLo, kYIntHi, kYIntLo};
	}
	int takeX(int n, int align = 1) { return x.take(n, align); }
	int takeY(int n) {
		int a = yint.take(n);
		return a >= 0 ? a : y.take(n);
	}
};

// ---------------------------------------------------------------- programs

struct Program {
	int32_t w[512];
	int n;
	void clear() { n = 0; }
	void put(int32_t v) { w[n++] = v; }
	void op(int code, int a = -1, int b = -1, int c = -1, int d = -1, int e = -1, int f = -1) {
		put(code);
		const int args[6] = {a, b, c, d, e, f};
		for (int i = 0; i < 6 && args[i] >= 0; ++i)
			put(args[i]);
	}
};

// ---------------------------------------------------------------- host interpreter

struct HostVm {
	int32_t X[kMemWords], Y[kMemWords];
	// the host port as the program sees it
	const int32_t *in;
	int inCount, inRead;
	int32_t *out;
	int outCount, outCap;
	bool fault;

	void reset() {
		memset(X, 0, sizeof X);
		memset(Y, 0, sizeof Y);
		fault = false;
	}

	int32_t &x(int a) {
		if (a < 0 || a >= kMemWords)
			fault = true;
		return X[a & (kMemWords - 1)];
	}
	int32_t &y(int a) {
		if (a < 0 || a >= kMemWords)
			fault = true;
		return Y[a & (kMemWords - 1)];
	}

	void matrix(const int32_t *src, int n, const int32_t *w, int rows, int dstAddr, int sh) {
		int32_t v[256];
		matvec(src, n, w, rows, sh, v);
		for (int r = 0; r < rows; ++r)
			x(dstAddr + r) = v[r];
	}

	int ringSlot(int ring, int framesBack, int c) {
		const int base = x(ring), size = x(ring + 1), n = x(ring + 2), wr = x(ring + 3);
		int off = (wr - (1 + framesBack) * n + c) % size;
		if (off < 0)
			off += size;
		return base + off;
	}

	// One run of the program at X:pc.
	void run(int pc, const int32_t *input, int nin, int32_t *output, int cap) {
		in = input;
		inCount = nin;
		inRead = 0;
		out = output;
		outCount = 0;
		outCap = cap;
		for (;;) {
			const int op = x(pc++);
			switch (op) {
			case OP_END:
				return;
			case OP_IN: {
				const int dst = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i)
					x(dst + i) = inRead < inCount ? sat24(in[inRead++]) : (fault = true, 0);
				break;
			}
			case OP_OUT: {
				const int src = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i)
					if (outCount < outCap)
						out[outCount++] = x(src + i);
					else
						fault = true;
				break;
			}
			case OP_MVY:
			case OP_MVX: {
				const int src = x(pc++), n = x(pc++), w = x(pc++), rows = x(pc++), dst = x(pc++), sh = x(pc++);
				int32_t v[256];
				for (int i = 0; i < n; ++i)
					v[i] = op == OP_MVY ? x(src + i) : y(src + i);
				if (w < 0 || w + rows * (n + 1) > kMemWords)
					fault = true;
				matrix(v, n, (op == OP_MVY ? Y : X) + (w & (kMemWords - 1)), rows, dst, sh);
				break;
			}
			case OP_COPYXY: {
				const int src = x(pc++), dst = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i)
					y(dst + i) = x(src + i);
				break;
			}
			case OP_LN: {
				const int src = x(pc++), n = x(pc++), gb = x(pc++), sh = x(pc++), dst = x(pc++);
				int32_t v[128], o[128];
				for (int i = 0; i < n; ++i)
					v[i] = x(src + i);
				int32_t rs[768];
				for (int i = 0; i < 768; ++i)
					rs[i] = x(x(kScRsqrt) + i);
				layernorm(v, n, Y + gb, sh, rs, o);
				for (int i = 0; i < n; ++i)
					x(dst + i) = o[i];
				break;
			}
			case OP_ACT: {
				const int d = x(pc++), src = x(pc++), dst = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i)
					x(dst + i) = activateDesc(d, x(src + i));
				break;
			}
			case OP_ADD:
			case OP_SUB:
			case OP_MUL: {
				const int a = x(pc++), b = x(pc++), dst = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i) {
					const int32_t va = x(a + i), vb = x(b + i);
					x(dst + i) = op == OP_ADD ? add(va, vb) : op == OP_SUB ? sub(va, vb) : mul(va, vb);
				}
				break;
			}
			case OP_RELU: {
				const int src = x(pc++), dst = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i)
					x(dst + i) = relu(x(src + i));
				break;
			}
			case OP_CLEAR: {
				const int dst = x(pc++), n = x(pc++);
				for (int i = 0; i < n; ++i)
					x(dst + i) = 0;
				break;
			}
			case OP_PUSH: {
				const int ring = x(pc++), src = x(pc++);
				const int n = x(ring + 2);
				for (int c = 0; c < n; ++c)
					x(ringSlot(ring, -1, c)) = x(src + c);
				x(ring + 3) = (x(ring + 3) + n) % x(ring + 1);
				break;
			}
			case OP_TAP: {
				const int ring = x(pc++), delay = x(pc++), dst = x(pc++);
				const int n = x(ring + 2);
				for (int c = 0; c < n; ++c)
					x(dst + c) = x(ringSlot(ring, delay, c));
				break;
			}
			case OP_DW: {
				const int ring = x(pc++), dil = x(pc++), w = x(pc++), sh = x(pc++), dst = x(pc++);
				const int n = x(ring + 2);
				for (int c = 0; c < n; ++c) {
					int32_t taps[5];
					for (int k = 0; k < 5; ++k)
						taps[k] = x(ringSlot(ring, (4 - k) * dil, c));
					x(dst + c) = matrow(taps, 1, 5, Y + w + c * 6, sh);
				}
				break;
			}
			default:
				fault = true;
				return;
			}
			if (fault)
				return;
		}
	}

	int32_t activateDesc(int d, int32_t v) {
		if (v < x(d + ACTD_LO))
			return x(d + ACTD_LOVAL);
		if (v >= x(d + ACTD_HI))
			return x(d + ACTD_IDENT) ? v : x(d + ACTD_HIVAL);
		// u = v - lo; interval and fraction from u * MULT
		const int64_t p = (int64_t)(v + x(d + ACTD_OFFSET)) * x(d + ACTD_MULT) * 2;   // the DSP's A
		const int i = (int)(p >> 24);
		const int32_t f = (int32_t)((p & 0xffffff) >> 1);
		return sat24(x(x(d + ACTD_TABLE) + i) + asr64((int64_t)x(x(d + ACTD_SLOPES) + i) * f, 23));
	}
};

}  // namespace Aurora
