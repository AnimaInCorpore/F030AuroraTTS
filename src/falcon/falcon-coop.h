// The DSP for programs that must share the machine (FreeMiNT): every
// host-port transaction runs in one short Supexec, and while the DSP
// computes the program waits in short supervisor polls (6.5 ms at most) and
// yields (Syield) between them, so MiNT can run other processes.
// Under plain TOS, Syield does not exist and returns at once; a Supexec
// costs microseconds against the milliseconds of a DSP step.
//
// The same driver interface as FalconDsp (writeX, writeY, stepSend,
// stepReceive, synthReset, synthStep).
#pragma once
#include <mint/mintbind.h>
#include <mint/osbind.h>
#include <stdint.h>

#include "falcon-dsp.h"

namespace Aurora {

struct CoopDsp {
	FalconDsp dsp;
	long yields;   // Syield calls while waiting (a measure of the time given away)

	// ---- the transaction in flight (Supexec takes no arguments)
	static CoopDsp *s_self;
	static int s_addr, s_n, s_pc;
	static const int32_t *s_in;
	static int32_t *s_out;
	static int16_t *s_pcm;
	static long s_result;
	static bool s_raw, s_pending;
	static int s_sendN;

	static long superWriteX() {
		s_self->dsp.writeX(s_addr, s_in, s_n);
		return 0;
	}
	static long superWriteY() {
		s_self->dsp.writeY(s_addr, s_in, s_n);
		return 0;
	}
	static long superStepSend() {
		s_self->dsp.stepSend(s_pc, s_in, s_n);
		return 0;
	}
	// Polls the port for up to about 6.5 ms (8,000 tests): a DSP step usually
	// ends within that, so one Supexec both waits and receives. A longer
	// wait returns to user mode, where MiNT can run other processes.
	static bool superWaitBounded() {
		volatile uint8_t *isr = (volatile uint8_t *)0xffffa202;
		for (int i = 0; i < 8000; ++i)
			if (*isr & 1)
				return true;
		return false;
	}
	static long superStepFull0() {
		if (s_pending) {
			s_self->dsp.stepSend(s_pc, s_in, s_sendN);
			s_pending = false;
		}
		return 0;
	}
	static long superStepFull() {
		if (s_pending) {
			s_self->dsp.stepSend(s_pc, s_in, s_sendN);
			s_pending = false;
		}
		return superStepReceive();
	}
	static long superStepReceive() {
		if (!superWaitBounded())
			return 0;
		s_self->dsp.stepReceive(s_out, s_n, s_raw);
		return 1;
	}
	static long superSynthReset() {
		s_self->dsp.synthReset();
		return 0;
	}
	static long superSynthSend() {
		FalconDsp &d = s_self->dsp;
		d.send(DSP_CMD_SYNTH_STEP | (s_in ? 0 : 1));
		if (s_in) {
			d.ready();
			FalconDsp::sendBlock(s_in, 66);
			d.wordsIn += 66;
		}
		return 0;
	}
	static long superSynthReceive() {
		if (!superWaitBounded())
			return 0;
		FalconDsp &d = s_self->dsp;
		const int n = d.receive();
		FalconDsp::receiveBlock16(s_pcm, n < 400 ? n : 400);
		d.wordsOut += n;
		return n + 1;
	}
	static long superPing() {
		return s_self->dsp.ping();
	}
	static long superDone() {
		s_self->dsp.done();
		return 0;
	}
	static long superTicks() {
		return *(volatile long *)0x4ba;
	}

	// Runs a bounded receive until it succeeds, yielding between tries.
	long receiveLoop(long (*fn)()) {
		long r;
		while ((r = Supexec(fn)) == 0) {
			Syield();
			++yields;
		}
		return r;
	}

	// ---- the device interface

	bool boot() {
		s_self = this;
		yields = 0;
		return dsp.boot();
	}
	void release() { dsp.release(); }
	bool failed() const { return dsp.failed; }

	bool ping() {
		s_self = this;
		return Supexec(superPing) != 0;
	}

	void done() {
		s_self = this;
		Supexec(superDone);
	}

	static long ticks() { return Supexec(superTicks); }

	// Uploads in pieces of 1,024 words, each its own command.
	void writeX(int addr, const int32_t *w, int n) { write(superWriteX, addr, w, n); }
	void writeY(int addr, const int32_t *w, int n) { write(superWriteY, addr, w, n); }

	void write(long (*fn)(), int addr, const int32_t *w, int n) {
		s_self = this;
		for (int at = 0; at < n; at += 1024) {
			s_addr = addr + at;
			s_in = w + at;
			s_n = n - at < 1024 ? n - at : 1024;
			Supexec(fn);
		}
		s_in = 0;
	}

	void stepSend(int pc, const int32_t *in, int nin) {
		s_self = this;
		s_pc = pc;
		s_in = in;
		s_sendN = nin;
		s_pending = true;
		Supexec(superStepFull0);
	}

	void stepReceive(int32_t *out, int nout, bool raw = false) {
		s_self = this;
		s_out = out;
		s_n = nout;
		s_raw = raw;
		receiveLoop(superStepFull);
	}

	void synthReset() {
		s_self = this;
		Supexec(superSynthReset);
	}

	int synthStep(const int32_t *head, int16_t *out) {
		s_self = this;
		s_in = head;
		Supexec(superSynthSend);
		s_pcm = out;
		return (int)receiveLoop(superSynthReceive) - 1;
	}
};

CoopDsp *CoopDsp::s_self;
int CoopDsp::s_addr, CoopDsp::s_n, CoopDsp::s_pc;
const int32_t *CoopDsp::s_in;
int32_t *CoopDsp::s_out;
int16_t *CoopDsp::s_pcm;
long CoopDsp::s_result;
bool CoopDsp::s_raw, CoopDsp::s_pending;
int CoopDsp::s_sendN;

}  // namespace Aurora
