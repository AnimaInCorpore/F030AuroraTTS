// LOCKSTEP.TOS: runs the encoder on the DSP and on the host interpreter at
// once, compares every step, and at the first difference writes DUMP.BIN:
// big-endian words: pass, step, pc, output count, the DSP's outputs (128),
// the interpreter's (128), DSP X (16384), interpreter X, DSP Y, interpreter Y.
// Reads AURORA.BIN and IDS.BIN beside the program.
#include <mint/osbind.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver.h"
#include "falcon-dsp.h"

using namespace Aurora;

static int32_t g_dx[16384], g_dy[16384], g_vx[16384], g_vy[16384];
static int32_t g_out[128], g_ref[128];

struct Lockstep {
	FalconDsp dsp;
	HostVm vm;
	bool stopped;
	long steps;
	int pc, nout;
	void writeX(int a, const int32_t *w, int n) {
		dsp.writeX(a, w, n);
		for (int i = 0; i < n; ++i)
			vm.x(a + i) = w[i];
	}
	void writeY(int a, const int32_t *w, int n) {
		dsp.writeY(a, w, n);
		for (int i = 0; i < n; ++i)
			vm.y(a + i) = w[i];
	}
	int sendPc, sendN;
	const int32_t *sendIn;
	void stepSend(int at, const int32_t *in, int nin) {
		sendPc = at;
		sendIn = in;
		sendN = nin;
	}
	void stepReceive(int32_t *out, int n, bool = false) {
		step(sendPc, sendIn, sendN, out, n);
	}
	void step(int at, const int32_t *in, int nin, int32_t *out, int n) {
		if (stopped) {
			memset(out, 0, n * 4);
			return;
		}
		dsp.step(at, in, nin, out, n);
		vm.run(at, in, nin, g_ref, n);
		++steps;
		if (memcmp(out, g_ref, n * 4) != 0) {
			stopped = true;
			pc = at;
			nout = n;
			memcpy(g_out, out, n * 4);
			memcpy(g_vx, vm.X, sizeof g_vx);
			memcpy(g_vy, vm.Y, sizeof g_vy);
			for (int a = 0; a < 16384; ++a) {
				g_dx[a] = (a >= 0x100 && a < 0x200) || a >= 0x3000 ? 0 : dsp.readX(a);
				g_dy[a] = a >= 0x100 && a < 0x200 ? 0 : dsp.readY(a);
			}
		}
	}
};

static Lockstep g_dev;
static Driver<Lockstep> g_drv;
static int32_t g_img[AURORA_IMAGE_WORDS];

static void put(FILE *f, int32_t v) {
	const unsigned char b[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16), (unsigned char)(v >> 8),
	                            (unsigned char)v};
	fwrite(b, 1, 4, f);
}

int main() {
	FILE *f = fopen("AURORA.BIN", "rb");
	if (!f || fread(g_img, 4, AURORA_IMAGE_WORDS, f) != AURORA_IMAGE_WORDS)
		return 1;
	fclose(f);
	for (long i = 0; i < AURORA_IMAGE_WORDS; ++i)
		g_img[i] = (g_img[i] << 8) >> 8;
	static uint8_t ids[256];
	f = fopen("IDS.BIN", "rb");
	const int N = f ? (int)fread(ids, 1, 256, f) : 0;
	if (f)
		fclose(f);
	if (!g_dev.dsp.boot())
		return 2;
	g_dev.vm.reset();
	void *ssp = (void *)Super(0L);
	g_dev.dsp.ping();
	g_drv.init(&g_dev, g_img);
	static int32_t x[256 * 56], work[256 * 56], logd[256], pit[256];
	static uint8_t dur[256];
	g_drv.encode(ids, N, x, work, logd, pit, dur);
	SuperToUser(ssp);
	printf("lockstep: %ld steps, %s\n", g_dev.steps, g_dev.stopped ? "DIFFERENT" : "equal");
	f = fopen("DUMP.BIN", "wb");
	put(f, g_drv.passCount - 1);
	put(f, g_dev.steps);
	put(f, g_dev.stopped ? g_dev.pc : -1);
	put(f, g_dev.nout);
	for (int i = 0; i < 128; ++i)
		put(f, g_out[i]);
	for (int i = 0; i < 128; ++i)
		put(f, g_ref[i]);
	for (int a = 0; a < 16384; ++a)
		put(f, g_dx[a]);
	for (int a = 0; a < 16384; ++a)
		put(f, g_vx[a]);
	for (int a = 0; a < 16384; ++a)
		put(f, g_dy[a]);
	for (int a = 0; a < 16384; ++a)
		put(f, g_vy[a]);
	fclose(f);
	ssp = (void *)Super(0L);
	g_dev.dsp.done();
	SuperToUser(ssp);
	g_dev.dsp.release();
	return 0;
}
