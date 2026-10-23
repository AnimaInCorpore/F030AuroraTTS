// The Falcon's DSP as the network driver's device: boot through the two-stage
// loader, then commands and words over the host port. Every port access runs
// in supervisor mode; the program enters it once (Super) for the whole run.
#pragma once
#include <mint/falcon.h>
#include <mint/osbind.h>
#include <stdint.h>

#include "aurora-image.h"

namespace Aurora {

enum {
	DSP_CMD_PING = 0x010000, DSP_CMD_WRITE_X = 0x020000, DSP_CMD_WRITE_Y = 0x030000, DSP_CMD_RUN = 0x040000,
	DSP_CMD_READ_X = 0x050000, DSP_CMD_DONE = 0x060000, DSP_CMD_READ_Y = 0x070000,
	DSP_CMD_SYNTH_RESET = 0x100000, DSP_CMD_SYNTH_STEP = 0x110000,
	DSP_REPLY_PING = 0x415552, DSP_REPLY_READY = 0x524459
};

struct FalconDsp {
	volatile uint8_t *isr() { return (volatile uint8_t *)0xffffa202; }
	volatile uint32_t *data() { return (volatile uint32_t *)0xffffa204; }
	long spinLimit;
	bool failed;
	long wordsIn, wordsOut;

	void send(uint32_t w) {
		for (long i = 0; !(*isr() & 2); ++i)
			if (i > spinLimit) {
				failed = true;
				return;
			}
		*data() = w;
		++wordsIn;
	}

	int32_t receive() {
		for (long i = 0; !(*isr() & 1); ++i)
			if (i > spinLimit) {
				failed = true;
				return 0;
			}
		volatile uint8_t *b = (volatile uint8_t *)data();
		uint32_t w = (uint32_t)b[1] << 16;
		w |= (uint32_t)b[2] << 8;
		w |= b[3];
		++wordsOut;
		return (int32_t)(w << 8) >> 8;
	}

	// Boot from user mode (XBIOS); the rest needs supervisor mode.
	bool boot() {
		spinLimit = 40000000;
		failed = false;
		wordsIn = wordsOut = 0;
		if (Dsp_Lock() != 0)
			return false;
		Dsp_Reserve(16, 16);
		Dsp_ExecBoot((char *)kAuroraDspBoot, AURORA_DSP_BOOT_WORDS, 3);
		static long reply;
		reply = 0;
		Dsp_BlkUnpacked((long *)kAuroraDspStream, AURORA_DSP_STREAM_WORDS, &reply, 1);
		return reply == AURORA_DSP_STREAM_REPLY_OK;
	}

	void release() { Dsp_Unlock(); }

	bool ping() {
		send(DSP_CMD_PING);
		return receive() == DSP_REPLY_PING;
	}

	// Block transfers. A block the DSP receives follows its ready token: the
	// DSP is then in a receive loop that takes a word in about 0.5 us, so
	// the 68030 (a word every 1.5 us or more) writes without polling, as
	// F030MXDRV does. A block the DSP sends is polled for its first word
	// only: the DSP refills the port within a microsecond of each read.
	static void sendBlock(const int32_t *w, long n) {
		if (n <= 0)
			return;
		const int32_t *p = w;
		long c = n - 1;
		__asm__ volatile(
		    "1:	move.l	(%0)+,0xffffa204.w\n"
		    "	dbra	%1,1b\n"
		    : "+a"(p), "+d"(c)
		    :
		    : "cc", "memory");
	}

	static void receiveBlock(int32_t *w, long n) {
		if (n <= 0)
			return;
		int32_t *p = w;
		long c = n - 1;
		long t;
		__asm__ volatile(
		    "0:	btst	#0,0xffffa202.w\n"
		    "	beq.s	0b\n"
		    "1:	move.l	0xffffa204.w,%2\n"
		    "	lsl.l	#8,%2\n"
		    "	asr.l	#8,%2\n"
		    "	move.l	%2,(%0)+\n"
		    "	dbra	%1,1b\n"
		    : "+a"(p), "+d"(c), "=&d"(t)
		    :
		    : "cc", "memory");
	}

	static void receiveBlockRaw(int32_t *w, long n) {
		if (n <= 0)
			return;
		int32_t *p = w;
		long c = n - 1;
		__asm__ volatile(
		    "0:	btst	#0,0xffffa202.w\n"
		    "	beq.s	0b\n"
		    "1:	move.l	0xffffa204.w,(%0)+\n"
		    "	dbra	%1,1b\n"
		    : "+a"(p), "+d"(c)
		    :
		    : "cc", "memory");
	}

	void ready() {
		if (receive() != DSP_REPLY_READY)
			failed = true;
	}

	void writeX(int addr, const int32_t *w, int n) {
		send(DSP_CMD_WRITE_X);
		send(addr);
		send(n);
		if (n <= 0)
			return;
		ready();
		sendBlock(w, n);
		wordsIn += n;
	}

	void writeY(int addr, const int32_t *w, int n) {
		send(DSP_CMD_WRITE_Y);
		send(addr);
		send(n);
		if (n <= 0)
			return;
		ready();
		sendBlock(w, n);
		wordsIn += n;
	}

	void step(int pc, const int32_t *in, int nin, int32_t *out, int nout) {
		stepSend(pc, in, nin);
		stepReceive(out, nout);
	}

	void stepSend(int pc, const int32_t *in, int nin) {
		send(DSP_CMD_RUN | pc);
		if (nin > 0) {   // a program with input starts with IN, which sends the token
			ready();
			sendBlock(in, nin);
		}
		wordsIn += nin;
	}

	// raw: the 24-bit words without sign extension, for outputs that only go
	// back to the DSP (the decoder's)
	void stepReceive(int32_t *out, int nout, bool raw = false) {
		if (raw)
			receiveBlockRaw(out, nout);
		else
			receiveBlock(out, nout);
		wordsOut += nout;
	}

	// Samples straight into 16-bit storage.
	static void receiveBlock16(int16_t *w, long n) {
		if (n <= 0)
			return;
		int16_t *p = w;
		long c = n - 1;
		long t;
		__asm__ volatile(
		    "0:	btst	#0,0xffffa202.w\n"
		    "	beq.s	0b\n"
		    "1:	move.l	0xffffa204.w,%2\n"
		    "	move.w	%2,(%0)+\n"
		    "	dbra	%1,1b\n"
		    : "+a"(p), "+d"(c), "=&d"(t)
		    :
		    : "cc", "memory");
	}

	int32_t readX(int a) {
		send(DSP_CMD_READ_X | a);
		return receive();
	}
	int32_t readY(int a) {
		send(DSP_CMD_READ_Y | a);
		return receive();
	}

	void synthReset() { send(DSP_CMD_SYNTH_RESET); }

	int synthStep(const int32_t *head, int16_t *out) {
		send(DSP_CMD_SYNTH_STEP | (head ? 0 : 1));
		if (head) {
			ready();
			sendBlock(head, 66);
			wordsIn += 66;
		}
		const int n = receive();
		receiveBlock16(out, n < 400 ? n : 400);
		wordsOut += n;
		return n;
	}

	void done() {
		send(DSP_CMD_DONE);
		receive();
	}
};

}  // namespace Aurora
