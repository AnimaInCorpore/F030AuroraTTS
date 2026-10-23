// Single-operation tests shared by the Falcon program (OPTEST.TOS) and the
// host checker: each case fills memory with pseudo-random words, runs one
// operation and sends its output words back. Both sides build the same cases
// from the same seed, so the host interpreter's answer is the expected one.
#pragma once
#include <stdint.h>

#include "aurora-layout.h"
#include "vm.h"

namespace Aurora {

struct OpRng {
	uint32_t s;
	uint32_t next() {
		s = s * 1664525u + 1013904223u;
		return s;
	}
	// a 24-bit word with |v| < 2^bits
	int32_t word(int bits) {
		const int32_t v = (int32_t)(next() >> 8) >> (24 - bits);
		return v;
	}
};

const int kOpCases = 16;

// Build case k into dev (writeX/writeY) and return the program address,
// the output count in *nout.
template <class Dev>
int buildOpCase(Dev *dev, const int32_t *img, int k, int *nout) {
	OpRng r;
	r.s = 12345u + 977u * (uint32_t)k;
	static int32_t buf[4096];
	const int vin = 0x0200, vin2 = 0x0300, vout = 0x0400, w = 0x0400, prog = 0x2f00;
	Program p;
	p.clear();
	const int n = 56;
	for (int i = 0; i < 256; ++i)
		buf[i] = r.word(21);
	dev->writeX(vin, buf, 256);
	for (int i = 0; i < 256; ++i)
		buf[i] = r.word(21);
	dev->writeX(vin2, buf, 256);
	*nout = n;
	switch (k) {
	case 0:   // MVY, shift 0..3 by rows
	case 1:
	case 2:
	case 3: {
		const int rows = 40, in = 56;
		for (int i = 0; i < rows * (in + 1); ++i)
			buf[i] = r.word(22);
		dev->writeY(w, buf, rows * (in + 1));
		p.op(OP_MVY, vin, in, w, rows, vout, k);
		*nout = rows;
		break;
	}
	case 4: {   // MVX: rows in X, input copied to Y
		const int rows = 30, in = 2;
		for (int i = 0; i < rows * (in + 1); ++i)
			buf[i] = r.word(22);
		dev->writeX(0x1000, buf, rows * (in + 1));
		p.op(OP_COPYXY, vin, 0x0010, in);
		p.op(OP_MVX, 0x0010, in, 0x1000, rows, vout, 2);
		*nout = rows;
		break;
	}
	case 5: {   // LN
		for (int i = 0; i < 2 * n; ++i)
			buf[i] = r.word(21);
		dev->writeY(w, buf, 2 * n);
		const int32_t t = 0x0800;
		dev->writeX(kScRsqrt, &t, 1);
		dev->writeX(0x0800, img + AU_TAB_RSQRT_OFF, AU_TAB_RSQRT_LEN);
		p.op(OP_LN, vin, n, w, 3, vout);
		break;
	}
	case 6:
	case 7: {   // ACT: GELU, sigmoid (inputs beyond the table's range too)
		const int t = 0x0800, d = 0x1800;
		const bool sig = k == 7;
		dev->writeX(t, img + (sig ? AU_TAB_SIGMOID_OFF : AU_TAB_GELU_OFF), 2048);
		const int32_t desc[ACTD_WORDS] = {t, t + 1024, sig ? -16 * kOne : -8 * kOne, sig ? 16 * kOne : 8 * kOne,
		                                  sig ? 16 * kOne : 8 * kOne, sig ? 1 << 12 : 1 << 13, 0, sig ? kOne : 0,
		                                  sig ? 0 : 1};
		dev->writeX(d, desc, ACTD_WORDS);
		p.op(OP_ACT, d, vin, vout, n);
		break;
	}
	case 8:   // ADD, SUB, MUL, RELU into consecutive outputs
		p.op(OP_ADD, vin, vin2, vout, 14);
		p.op(OP_SUB, vin, vin2, vout + 14, 14);
		p.op(OP_MUL, vin, vin2, vout + 28, 14);
		p.op(OP_RELU, vin, vout + 42, 14);
		break;
	case 9:
	case 10:
	case 11: {   // rings: PUSH three frames, TAP, DW with dil 1 / 2 and shift 0 / 1
		const int ring = 0x0040, base = 0x1000, frames = 9, rn = 8;
		const int32_t desc[4] = {base, frames * rn, rn, 0};
		dev->writeX(ring, desc, 4);
		for (int i = 0; i < frames * rn; ++i)
			buf[i] = r.word(21);
		dev->writeX(base, buf, frames * rn);
		for (int i = 0; i < rn * 6; ++i)
			buf[i] = r.word(22);
		dev->writeY(w, buf, rn * 6);
		p.op(OP_PUSH, ring, vin);
		p.op(OP_PUSH, ring, vin + rn);
		p.op(OP_PUSH, ring, vin + 2 * rn);
		p.op(OP_TAP, ring, k - 9, vout);
		p.op(OP_DW, ring, k == 11 ? 2 : 1, w, k == 10 ? 1 : 0, vout + rn);
		p.op(OP_PUSH, ring, vin + 3 * rn);
		p.op(OP_DW, ring, 2, w, 0, vout + 2 * rn);
		p.op(OP_TAP, ring, 8, vout + 3 * rn);
		*nout = 4 * rn;
		break;
	}
	case 12:   // LN on inputs near 1.0 (left shifts) and near the limit (right shifts)
	case 13: {
		for (int i = 0; i < 256; ++i)
			buf[i] = r.word(k == 12 ? 17 : 23);
		dev->writeX(vin, buf, 256);
		for (int i = 0; i < 2 * n; ++i)
			buf[i] = r.word(21);
		dev->writeY(0x0080, buf, 2 * n);
		const int32_t t = 0x0800;
		dev->writeX(kScRsqrt, &t, 1);
		dev->writeX(0x0800, img + AU_TAB_RSQRT_OFF, AU_TAB_RSQRT_LEN);
		p.op(OP_LN, vin, n, 0x0080, k == 12 ? 4 : 5, vout);
		break;
	}
	case 14: {   // MVX with 112 inputs, shift 0
		const int rows = 20, in = 112;
		for (int i = 0; i < rows * (in + 1); ++i)
			buf[i] = r.word(22);
		dev->writeX(0x1000, buf, rows * (in + 1));
		p.op(OP_COPYXY, vin, 0x0000, in);
		p.op(OP_MVX, 0x0000, in, 0x1000, rows, vout, 0);
		*nout = rows;
		break;
	}
	case 15: {   // MVY with 112 inputs from internal Y rows
		const int rows = 2, in = 112;
		for (int i = 0; i < rows * (in + 1); ++i)
			buf[i] = r.word(22);
		dev->writeY(0x0010, buf, rows * (in + 1));
		p.op(OP_MVY, vin, in, 0x0010, rows, vout, 1);
		*nout = rows;
		break;
	}
	}
	p.op(OP_OUT, vout, *nout);
	p.put(OP_END);
	dev->writeX(prog, p.w, p.n);
	return prog;
}

}  // namespace Aurora
