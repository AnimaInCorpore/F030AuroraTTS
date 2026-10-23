// AuroraTTS on the DSP: the passes the 68030 runs, one layer (or layer
// group) at a time, streaming tokens or frames through the network machine.
//
// The same code drives the host interpreter (HostVm, for the PC gates) and
// the Falcon's DSP (through the host port). Dev provides:
//   void writeX(int addr, const int32_t *w, int n);
//   void writeY(int addr, const int32_t *w, int n);
//   void stepSend(int pc, const int32_t *in, int nin);   run a program, its input
//   void stepReceive(int32_t *out, int nout);            then its output: the
//   68030 works on the next item between the two while the DSP computes.
#pragma once
#include <stdint.h>
#include <string.h>

#include "aurora-layout.h"
#include "model.h"
#include "vm.h"

namespace Aurora {

struct Chunk {
	bool inX;          // rows in X (input from Y) or in Y (input from X)
	int addr, rows, first;
};

struct Matrix {
	Chunk c[4];
	int count, rows, n, shift;
};

template <class Dev>
struct Driver {
	Dev *dev;
	const int32_t *img;
	Layout L;
	Program prog, pad;
	int progAddr, padAddr;
	int zeroAddr;          // a zero vector in X (128 words)
	int ytemp;             // a Y copy of an input vector (128 words)
	int ytempOf;           // the X vector it currently copies, in prog/pad
	int32_t zeros[128];
	uint32_t outSum;
	bool passSums;         // keep passSum (the gates); off in the player

	// what the DSP memory holds: skip uploads that are already in place
	struct Held {
		bool inX;
		int addr, off, len;
	};
	Held held[64];
	int heldCount;

	void init(Dev *d, const int32_t *image) {
		dev = d;
		img = image;
		heldCount = 0;
		passCount = 0;
		outSum = 0;
		passSums = true;
		memset(zeros, 0, sizeof zeros);
	}

	void upload(bool inX, int addr, int off, int len) {
		for (int i = 0; i < heldCount; ++i)
			if (held[i].inX == inX && held[i].addr == addr && held[i].off == off && held[i].len == len)
				return;
		// forget what this overwrites
		int k = 0;
		for (int i = 0; i < heldCount; ++i) {
			const Held &h = held[i];
			if (h.inX == inX && h.addr < addr + len && addr < h.addr + h.len)
				continue;
			held[k++] = h;
		}
		heldCount = k;
		if (inX)
			dev->writeX(addr, img + off, len);
		else
			dev->writeY(addr, img + off, len);
		if (heldCount < 64)
			held[heldCount++] = Held{inX, addr, off, len};
	}

	// Data the pass writes itself (rings, vectors) is no longer what was held.
	void forget(bool inX, int addr, int len) {
		int k = 0;
		for (int i = 0; i < heldCount; ++i) {
			const Held &h = held[i];
			if (h.inX == inX && h.addr < addr + len && addr < h.addr + h.len)
				continue;
			held[k++] = h;
		}
		heldCount = k;
	}

	void writeXwords(int addr, const int32_t *w, int n) {
		forget(true, addr, n);
		dev->writeX(addr, w, n);
	}

	// ------------------------------------------------------------ layout

	// per pass: a checksum of every word the passes returned so far (the
	// gates compare these to find the first pass that differs)
	uint32_t passSum[32];
	int passCount;

	void begin() {
		if (passCount < 32)
			passSum[passCount++] = outSum;
		L.reset();
		prog.clear();
		pad.clear();
		zeroAddr = L.takeX(128);
		ytemp = L.yint.take(128);
		ytempOf = -1;
		writeXwords(zeroAddr, zeros, 128);
	}

	int vecX(int n) {
		const int a = L.takeX(n);
		return a;
	}

	int table(int off, int len) {
		const int a = L.takeX(len);
		upload(true, a, off, len);
		return a;
	}

	int activation(Activation kind) {
		int off, lo, hi, offset, mult, loval, hival, ident;
		switch (kind) {
		case ACT_GELU:
			off = AU_TAB_GELU_OFF; lo = -8 * kOne; hi = 8 * kOne; mult = 1 << 13; loval = 0; hival = 0; ident = 1;
			break;
		case ACT_SIGMOID:
			off = AU_TAB_SIGMOID_OFF; lo = -16 * kOne; hi = 16 * kOne; mult = 1 << 12; loval = 0; hival = kOne;
			ident = 0;
			break;
		case ACT_TANH:
		default:
			off = AU_TAB_TANH_OFF; lo = -8 * kOne; hi = 8 * kOne; mult = 1 << 13; loval = -kOne; hival = kOne;
			ident = 0;
			break;
		}
		offset = -lo;
		const int t = table(off, 2048);
		const int d = L.takeX(ACTD_WORDS);
		const int32_t desc[ACTD_WORDS] = {t, t + 1024, lo, hi, offset, mult, loval, hival, ident};
		writeXwords(d, desc, ACTD_WORDS);
		return d;
	}

	void rsqrt() {
		const int t = table(AU_TAB_RSQRT_OFF, AU_TAB_RSQRT_LEN);
		const int32_t a = t;
		writeXwords(kScRsqrt, &a, 1);
	}

	// small Y blocks: layer norm rows, depthwise rows
	int blockY(int off, int len) {
		const int a = L.takeY(len);
		upload(false, a, off, len);
		return a;
	}

	// A matrix of `rows` rows of n inputs: as many rows in Y as fit, the rest in X.
	Matrix matrix(int off, int rows, int n, int shift) {
		Matrix m;
		m.count = 0;
		m.rows = rows;
		m.n = n;
		m.shift = shift;
		const int rw = n + 1;
		int first = 0;
		Region *spaces[3] = {&L.yint, &L.y, &L.x};
		for (int s = 0; s < 3 && first < rows; ++s) {
			const int fit = spaces[s]->room() / rw;
			const int take = fit < rows - first ? fit : rows - first;
			if (take <= 0)
				continue;
			const int a = spaces[s]->take(take * rw);
			const bool inX = s == 2;
			upload(inX, a, off + first * rw, take * rw);
			m.c[m.count++] = Chunk{inX, a, take, first};
			first += take;
		}
		return m;
	}

	int ring(int frames, int n) {
		const int size = frames * n;
		int align = 1;
		while (align < size)
			align <<= 1;
		const int base = L.takeX(size, align);
		const int d = L.takeX(4);
		const int32_t desc[4] = {base, size, n, 0};
		writeXwords(d, desc, 4);
		for (int i = 0; i < size; i += 128)
			writeXwords(base + i, zeros, size - i < 128 ? size - i : 128);
		return d;
	}

	bool fits() const { return L.x.at >= 0 && L.y.at >= 0; }

	// ------------------------------------------------------------ op emitters

	void mv(Program &p, int src, const Matrix &m, int dst) {
		for (int i = 0; i < m.count; ++i) {
			const Chunk &c = m.c[i];
			if (c.inX) {
				p.op(OP_COPYXY, src, ytemp, m.n);
				p.op(OP_MVX, ytemp, m.n, c.addr, c.rows, dst + c.first, m.shift);
			} else {
				p.op(OP_MVY, src, m.n, c.addr, c.rows, dst + c.first, m.shift);
			}
		}
	}

	void finish() {
		prog.put(OP_END);
		pad.put(OP_END);
		progAddr = L.takeX(prog.n);
		padAddr = L.takeX(pad.n);
		writeXwords(progAddr, prog.w, prog.n);
		writeXwords(padAddr, pad.w, pad.n);
#ifdef AURORA_TRACE
		AURORA_TRACE(passCount - 1, progAddr, prog.w, prog.n, L.x.at, L.y.at, L.yint.at);
#endif
	}

	// Stream `count` items of nin words through prog, then `lag` pad steps;
	// output item i (nout words) is written once step i + lag has run.
	// Outputs land directly in place: output o is received after input o + lag
	// was sent, so in and out may share a buffer when nout <= nin. With a
	// producer, input item s comes from produce(ctx, s, buffer), called for
	// s + 1 while the DSP computes item s.
	typedef const int32_t *(*Producer)(void *ctx, int item, int32_t *buffer);

	void stream(const int32_t *in, int count, int nin, int32_t *out, int nout, int lag, bool reverse = false,
	            Producer produce = 0, void *ctx = 0, bool raw = false) {
		int32_t spare[128], made[2][128];
		const int32_t *next = 0;
		if (produce && count > 0)
			next = produce(ctx, 0, made[0]);
		for (int s = 0; s < count + lag; ++s) {
			const int item = reverse ? count - 1 - s : s;
			if (s < count)
				dev->stepSend(progAddr, produce ? next : in + (size_t)item * nin, nin);
			else
				dev->stepSend(padAddr, 0, 0);
			if (produce && s + 1 < count)
				next = produce(ctx, s + 1, made[(s + 1) & 1]);
			const int o = s - lag;
			int32_t *dst = spare;
			if (o >= 0)
				dst = out + (size_t)(reverse ? count - 1 - o : o) * nout;
			dev->stepReceive(dst, nout, raw);
			if (passSums)
				for (int i = 0; i < nout; ++i)
					outSum = outSum * 31u + (uint32_t)(dst[i] & 0xffffff);
		}
	}

	// ------------------------------------------------------------ passes

	// Encoder loop `loop` over x (N, kD), in place.
	void encoderLoop(int32_t *x, int N, int loop) {
		static const int n1off[4] = {AU_ENC_N1_0_OFF, AU_ENC_N1_1_OFF, AU_ENC_N1_2_OFF, AU_ENC_N1_3_OFF};
		static const int n1sh[4] = {AU_ENC_N1_0_SHIFT, AU_ENC_N1_1_SHIFT, AU_ENC_N1_2_SHIFT, AU_ENC_N1_3_SHIFT};
		static const int n2off[4] = {AU_ENC_N2_0_OFF, AU_ENC_N2_1_OFF, AU_ENC_N2_2_OFF, AU_ENC_N2_3_OFF};
		static const int n2sh[4] = {AU_ENC_N2_0_SHIFT, AU_ENC_N2_1_SHIFT, AU_ENC_N2_2_SHIFT, AU_ENC_N2_3_SHIFT};
		begin();
		const int sig = activation(ACT_SIGMOID), gelu = activation(ACT_GELU);
		rsqrt();
		const int n1 = blockY(n1off[loop], 2 * kD), n2 = blockY(n2off[loop], 2 * kD);
		const int dw = blockY(AU_ENC_DW_OFF, AU_ENC_DW_LEN);
		const Matrix glu = matrix(AU_ENC_GLU_OFF, 2 * kD, kD, AU_ENC_GLU_SHIFT);
		const Matrix ffa = matrix(AU_ENC_FFA_OFF, 2 * kD, kD, AU_ENC_FFA_SHIFT);
		const Matrix ffb = matrix(AU_ENC_FFB_OFF, kD, 2 * kD, AU_ENC_FFB_SHIFT);
		const int rx = ring(3, kD), ry = ring(5, kD);
		const int vx = vecX(kD), vy = vecX(kD), vz = vecX(kD), vg = vecX(2 * kD), vs = vecX(kD), vd = vecX(kD),
		          vh = vecX(2 * kD);
		prog.op(OP_IN, vx, kD);
		prog.op(OP_PUSH, rx, vx);
		prog.op(OP_LN, vx, kD, n1, n1sh[loop], vy);
		prog.op(OP_PUSH, ry, vy);
		pad.op(OP_PUSH, rx, zeroAddr);
		pad.op(OP_PUSH, ry, zeroAddr);
		Program *both[2] = {&prog, &pad};
		for (int i = 0; i < 2; ++i) {
			Program &p = *both[i];
			p.op(OP_DW, ry, 1, dw, AU_ENC_DW_SHIFT, vz);
			mv(p, vz, glu, vg);
			p.op(OP_ACT, sig, vg + kD, vs, kD);
			p.op(OP_MUL, vg, vs, vs, kD);
			p.op(OP_TAP, rx, 2, vd);
			p.op(OP_ADD, vd, vs, vd, kD);          // x1
			p.op(OP_LN, vd, kD, n2, n2sh[loop], vy);
			mv(p, vy, ffa, vh);
			p.op(OP_ACT, gelu, vh, vh, 2 * kD);
			mv(p, vh, ffb, vz);
			p.op(OP_ADD, vd, vz, vd, kD);
			p.op(OP_OUT, vd, kD);
		}
		finish();
		stream(x, N, kD, x, kD, 2);
	}

	// One GRU direction over the context norm of x: h (N, 28) out.
	void gruPass(const int32_t *x, int N, bool backward, int32_t *h) {
		const int H = kD / 2;
		begin();
		const int sig = activation(ACT_SIGMOID), th = activation(ACT_TANH);
		rsqrt();
		const int nrm = blockY(AU_CTX_NORM_OFF, 2 * kD);
		const Matrix ih = backward ? matrix(AU_GRU_B_IH_OFF, 3 * H, kD, AU_GRU_B_IH_SHIFT)
		                           : matrix(AU_GRU_F_IH_OFF, 3 * H, kD, AU_GRU_F_IH_SHIFT);
		const Matrix hh = backward ? matrix(AU_GRU_B_HH_OFF, 3 * H, H, AU_GRU_B_HH_SHIFT)
		                           : matrix(AU_GRU_F_HH_OFF, 3 * H, H, AU_GRU_F_HH_SHIFT);
		const int vx = vecX(kD), vy = vecX(kD), gi = vecX(3 * H), gh = vecX(3 * H), rz = vecX(2 * H), t = vecX(H),
		          nn = vecX(H), vh = vecX(H);
		writeXwords(vh, zeros, H);
		prog.op(OP_IN, vx, kD);
		prog.op(OP_LN, vx, kD, nrm, AU_CTX_NORM_SHIFT, vy);
		mv(prog, vy, ih, gi);
		mv(prog, vh, hh, gh);
		prog.op(OP_ADD, gi, gh, rz, 2 * H);
		prog.op(OP_ACT, sig, rz, rz, 2 * H);
		prog.op(OP_MUL, rz, gh + 2 * H, t, H);           // r * gh_n
		prog.op(OP_ADD, gi + 2 * H, t, t, H);
		prog.op(OP_ACT, th, t, nn, H);                    // n
		prog.op(OP_SUB, vh, nn, t, H);
		prog.op(OP_MUL, rz + H, t, t, H);                 // z (h - n)
		prog.op(OP_ADD, nn, t, vh, H);
		prog.op(OP_OUT, vh, H);
		finish();
		stream(x, N, kD, h, H, 0, backward);
	}

	// Duration and pitch predictors and the pitch embedding: x (N, kD) in
	// place (x + pit_emb(pit)), logd and pit out.
	void predictPass(int32_t *x, int N, int32_t *logd, int32_t *pit) {
		begin();
		rsqrt();
		const int dn = blockY(AU_DUR_N_OFF, 2 * kD), pn = blockY(AU_PIT_N_OFF, 2 * kD);
		const int ddw = blockY(AU_DUR_DW_OFF, AU_DUR_DW_LEN), pdw = blockY(AU_PIT_DW_OFF, AU_PIT_DW_LEN);
		const Matrix d1 = matrix(AU_DUR_P1_OFF, 32, kD, AU_DUR_P1_SHIFT), d2 = matrix(AU_DUR_P2_OFF, 32, 32, AU_DUR_P2_SHIFT),
		             d3 = matrix(AU_DUR_O_OFF, 1, 32, AU_DUR_O_SHIFT);
		const Matrix p1 = matrix(AU_PIT_P1_OFF, 32, kD, AU_PIT_P1_SHIFT), p2 = matrix(AU_PIT_P2_OFF, 32, 32, AU_PIT_P2_SHIFT),
		             p3 = matrix(AU_PIT_O_OFF, 1, 32, AU_PIT_O_SHIFT);
		const Matrix pe = matrix(AU_PIT_EMB_OFF, kD, 2, AU_PIT_EMB_SHIFT);
		const int rx = ring(3, kD), rd = ring(5, kD), rp = ring(5, kD);
		const int vx = vecX(kD), vy = vecX(kD), va = vecX(32), vb = vecX(32), vo = vecX(kD + 2), vp2 = vecX(2);
		writeXwords(vp2, zeros, 2);
		prog.op(OP_IN, vx, kD);
		prog.op(OP_PUSH, rx, vx);
		prog.op(OP_LN, vx, kD, dn, AU_DUR_N_SHIFT, vy);
		prog.op(OP_PUSH, rd, vy);
		prog.op(OP_LN, vx, kD, pn, AU_PIT_N_SHIFT, vy);
		prog.op(OP_PUSH, rp, vy);
		pad.op(OP_PUSH, rx, zeroAddr);
		pad.op(OP_PUSH, rd, zeroAddr);
		pad.op(OP_PUSH, rp, zeroAddr);
		Program *both[2] = {&prog, &pad};
		for (int i = 0; i < 2; ++i) {
			Program &p = *both[i];
			// vo = [x + pit_emb(pit) (kD), logd, pit]
			p.op(OP_DW, rd, 1, ddw, AU_DUR_DW_SHIFT, vy);
			mv(p, vy, d1, va);
			p.op(OP_RELU, va, va, 32);
			mv(p, va, d2, vb);
			p.op(OP_RELU, vb, vb, 32);
			mv(p, vb, d3, vo + kD);
			p.op(OP_DW, rp, 1, pdw, AU_PIT_DW_SHIFT, vy);
			mv(p, vy, p1, va);
			p.op(OP_RELU, va, va, 32);
			mv(p, va, p2, vb);
			p.op(OP_RELU, vb, vb, 32);
			mv(p, vb, p3, vp2);
			p.op(OP_TAP, rx, 2, vo);
			mv(p, vp2, pe, vy);
			p.op(OP_ADD, vo, vy, vo, kD);
			p.op(OP_ADD, vp2, zeroAddr, vo + kD + 1, 1);
			p.op(OP_OUT, vo, kD + 2);
		}
		finish();
		int32_t *res = new int32_t[(size_t)N * (kD + 2)];
		stream(x, N, kD, res, kD + 2, 2);
		for (int n = 0; n < N; ++n) {
			memcpy(x + (size_t)n * kD, res + (size_t)n * (kD + 2), kD * sizeof(int32_t));
			logd[n] = res[(size_t)n * (kD + 2) + kD];
			pit[n] = res[(size_t)n * (kD + 2) + kD + 1];
		}
		delete[] res;
	}

	// Decoder layer i: layer 0 reads (T, kD + 2) upsampled frames, the others
	// (T, kD); layer 8 writes (T, kHead), the others (T, kD). in and out may be
	// the same buffer when an output frame is no larger than an input frame:
	// frame t's output is written after frame t + 2 dil was read.
	void decoderLayer(const int32_t *in, int32_t *out, int T, int i, Producer produce = 0, void *ctx = 0) {
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
		const int dil = 1 << (i % 3), g = i % 3;
		const bool first = i == 0, last = i == 8;
		begin();
		const int gelu = activation(ACT_GELU);
		rsqrt();
		const int ln = blockY(lnoff[i], 2 * kD);
		const int dw = blockY(dwoff[i], AU_DEC_DW_0_LEN);
		const int oln = last ? blockY(AU_DEC_OUT_LN_OFF, 2 * kD) : -1;
		const Matrix ffa = matrix(faoff[g], 2 * kD, kD, fash[g]);
		const Matrix ffb = matrix(fboff[g], kD, 2 * kD, fbsh[g]);
		Matrix inp, head;
		if (first)
			inp = matrix(AU_DEC_INP_OFF, kD, kD + 2, AU_DEC_INP_SHIFT);
		if (last)
			head = matrix(AU_HEAD_OFF, kHead, kD, AU_HEAD_SHIFT);
		const int rx = ring(4 * dil + 1, kD);
		const int vin = vecX(kD + 2), vx = vecX(kD), vy = vecX(kD), vh = vecX(2 * kD), vo = vecX(kHead);
		const int nin = first ? kD + 2 : kD;
		prog.op(OP_IN, vin, nin);
		if (first)
			mv(prog, vin, inp, vx);
		prog.op(OP_PUSH, rx, first ? vx : vin);
		pad.op(OP_PUSH, rx, zeroAddr);
		Program *both[2] = {&prog, &pad};
		for (int k = 0; k < 2; ++k) {
			Program &p = *both[k];
			p.op(OP_DW, rx, dil, dw, dwsh[i], vy);
			p.op(OP_LN, vy, kD, ln, lnsh[i], vx);
			mv(p, vx, ffa, vh);
			p.op(OP_ACT, gelu, vh, vh, 2 * kD);
			mv(p, vh, ffb, vy);
			p.op(OP_TAP, rx, 2 * dil, vx);
			p.op(OP_ADD, vx, vy, vx, kD);
			if (last) {
				p.op(OP_LN, vx, kD, oln, AU_DEC_OUT_LN_SHIFT, vy);
				mv(p, vy, head, vo);
				p.op(OP_OUT, vo, kHead);
			} else {
				p.op(OP_OUT, vx, kD);
			}
		}
		finish();
		// the decoder's outputs only go back to the DSP: raw words (high byte unspecified)
		stream(in, T, nin, out, last ? kHead : kD, 2 * dil, false, produce, ctx, true);
	}

	// ------------------------------------------------------------ the network

	// Phoneme ids to x (N, kD) ready for upsampling, durations, and the raw
	// predictor outputs. work holds N kD words.
	void encode(const uint8_t *ids, int N, int32_t *x, int32_t *work, int32_t *logd, int32_t *pit, uint8_t *dur) {
		for (int n = 0; n < N; ++n)
			memcpy(x + (size_t)n * kD, img + AU_EMB_OFF + ids[n] * kD, kD * sizeof(int32_t));
		for (int loop = 0; loop < 4; ++loop)
			encoderLoop(x, N, loop);
		// both directions read the context norm of the same x
		const int H = kD / 2;
		int32_t *hf = work, *hb = work + (size_t)N * H;
		gruPass(x, N, false, hf);
		gruPass(x, N, true, hb);
		for (int n = 0; n < N; ++n)
			for (int j = 0; j < H; ++j) {
				x[n * kD + j] = add(x[n * kD + j], hf[n * H + j]);
				x[n * kD + H + j] = add(x[n * kD + H + j], hb[n * H + j]);
			}
		predictPass(x, N, logd, pit);
		Model m;
		m.img = img;
		for (int n = 0; n < N; ++n)
			dur[n] = (uint8_t)m.durationOf(logd[n]);
	}

	// up (T, kD + 2) upsampled frames to head (T, kHead); work holds T kD words.
	void decode(const int32_t *up, int T, int32_t *head, int32_t *work) {
		decoderLayer(up, work, T, 0);
		for (int i = 1; i < 8; ++i)
			decoderLayer(work, work, T, i);
		decoderLayer(work, head, T, 8);
	}

	// The same from the encoder's output: the 68030 computes each upsampled
	// frame (the length regulator) while the DSP runs the previous one
	// through decoder layer 0.
	struct Upsampler {
		const int32_t *img, *x, *center2;
		const uint8_t *dur;
		int N, near;
	};

	static const int32_t *upsampleItem(void *ctx, int t, int32_t *buffer) {
		Upsampler *u = (Upsampler *)ctx;
		Model m;
		m.img = u->img;
		m.upsampleFrame(u->x, u->dur, u->center2, u->N, t, buffer, &u->near);
		return buffer;
	}

	void decodeTokens(const int32_t *x, const uint8_t *dur, const int32_t *center2, int N, int T, int32_t *head,
	                  int32_t *work) {
		Upsampler u = {img, x, center2, dur, N, 0};
		decoderLayer(0, work, T, 0, upsampleItem, &u);
		for (int i = 1; i < 8; ++i)
			decoderLayer(work, work, T, i);
		decoderLayer(work, head, T, 8);
	}
};

}  // namespace Aurora
