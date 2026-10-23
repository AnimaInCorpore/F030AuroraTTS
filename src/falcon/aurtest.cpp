// AURTEST.TOS: the network gate's Falcon side. Reads AURORA.BIN and IDS.BIN
// (phoneme ids, one byte each) beside the program, runs the encoder and
// decoder passes on the DSP and the length regulator on the 68030, and writes
// RESULT.BIN: big-endian words
//   status (0 ok, 1 no image, 2 boot, 3 ping, 4 port), tokens, frames,
//   duration sum, checksum of the encoder output x, of logd and pit, of the
//   head output, then 200 Hz ticks spent encoding, upsampling, decoding,
//   and host port words in and out.
#include <mint/osbind.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver.h"
#include "synth-driver.h"
#include "falcon-dsp.h"

using namespace Aurora;

static FalconDsp g_dsp;

static uint8_t *loadFile(const char *path, long *size) {
	FILE *f = fopen(path, "rb");
	if (!f)
		return 0;
	fseek(f, 0, SEEK_END);
	*size = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *p = (uint8_t *)malloc(*size + 4);
	if (p && fread(p, 1, *size, f) != (size_t)*size) {
		free(p);
		p = 0;
	}
	fclose(f);
	return p;
}

static uint32_t sum(const int32_t *v, long n) {
	uint32_t s = 0;
	for (long i = 0; i < n; ++i)
		s = s * 31u + (uint32_t)(v[i] & 0xffffff);
	return s;
}

static uint32_t ticks() {
	return *(volatile uint32_t *)0x4ba;
}

static void writeResult(const uint32_t *w, int n) {
	FILE *f = fopen("RESULT.BIN", "wb");
	if (!f)
		return;
	for (int i = 0; i < n; ++i) {
		const unsigned char b[4] = {(unsigned char)(w[i] >> 24), (unsigned char)(w[i] >> 16), (unsigned char)(w[i] >> 8),
		                            (unsigned char)w[i]};
		fwrite(b, 1, 4, f);
	}
	fclose(f);
}

static Driver<FalconDsp> g_drv;
static int16_t *g_pcm;
static long g_pcmCount;

int main() {
	uint32_t r[48];
	memset(r, 0, sizeof r);
	long isize = 0, nsize = 0;
	uint8_t *raw = loadFile("AURORA.BIN", &isize);
	uint8_t *ids = loadFile("IDS.BIN", &nsize);
	if (!raw || !ids || isize != AURORA_IMAGE_WORDS * 4) {
		r[0] = 1;
		writeResult(r, 48);
		return 1;
	}
	// big-endian words already: sign-extend the 24-bit values in place
	int32_t *img = (int32_t *)raw;
	for (long i = 0; i < AURORA_IMAGE_WORDS; ++i)
		img[i] = (img[i] << 8) >> 8;
	printf("AuroraTTS network gate: %ld tokens\n", nsize);
	if (!g_dsp.boot()) {
		r[0] = 2;
		writeResult(r, 48);
		return 1;
	}
	void *ssp = (void *)Super(0L);
	const int N = (int)nsize;
	r[1] = N;
	if (!g_dsp.ping()) {
		r[0] = 3;
	} else {
		int32_t *x = (int32_t *)malloc(N * kD * 4), *work = (int32_t *)malloc(N * kD * 4);
		int32_t *logd = (int32_t *)malloc(N * 4), *pit = (int32_t *)malloc(N * 4), *c2 = (int32_t *)malloc(N * 4);
		uint8_t *dur = (uint8_t *)malloc(N);
		g_drv.init(&g_dsp, img);
		const uint32_t t0 = ticks();
		g_drv.encode(ids, N, x, work, logd, pit, dur);
		const uint32_t t1 = ticks();
		int T;
		centres(dur, N, c2, &T);
		int32_t *head = (int32_t *)malloc((long)T * kHead * 4);
		int32_t *dw = (int32_t *)malloc((long)T * kD * 4);
		const uint32_t t2 = ticks();   // the length regulator runs inside the decoder's first pass
		g_drv.decodeTokens(x, dur, c2, N, T, head, dw);
		const uint32_t t3 = ticks();
		const long want = synthOutputCount(img, T);
		int16_t *pcm = (int16_t *)malloc(want * 2 + 1024);
		const long got = synthesize(&g_dsp, img, head, T, pcm);
		g_pcm = pcm;
		g_pcmCount = got;
		const uint32_t t4 = ticks();
		uint32_t ps = 0;
		for (long i = 0; i < got; ++i)
			ps = ps * 31u + (uint32_t)(pcm[i] & 0xffffff);
		r[13] = got;
		r[14] = ps;
		r[15] = t4 - t3;
		uint32_t ds = 0;
		for (int n = 0; n < N; ++n)
			ds += dur[n];
		r[2] = T;
		r[3] = ds;
		r[4] = sum(x, (long)N * kD);
		r[5] = sum(logd, N) ^ (sum(pit, N) * 7u);
		r[6] = sum(head, (long)T * kHead);
		r[7] = t1 - t0;
		r[8] = t2 - t1;
		r[9] = t3 - t2;
		r[10] = g_dsp.wordsIn;
		r[11] = g_dsp.wordsOut;
		r[0] = g_dsp.failed ? 4 : 0;
		g_drv.begin();   // closes the last pass's checksum
		r[12] = g_drv.passCount;
		for (int i = 0; i < g_drv.passCount && i < 32; ++i)
			r[16 + i] = g_drv.passSum[i];
	}
	SuperToUser(ssp);
	if (g_pcm && g_pcmCount > 0) {
		FILE *pf = fopen("PCM.BIN", "wb");
		fwrite(g_pcm, 2, g_pcmCount, pf);
		fclose(pf);
	}
	// the result first: the gate stops the emulator when the DSP sees DONE
	writeResult(r, 48);
	ssp = (void *)Super(0L);
	g_dsp.done();
	SuperToUser(ssp);
	g_dsp.release();
	printf("status %lu, %lu frames, ticks %lu %lu %lu\n", (unsigned long)r[0], (unsigned long)r[2],
	       (unsigned long)r[7], (unsigned long)r[8], (unsigned long)r[9]);
	return 0;
}
