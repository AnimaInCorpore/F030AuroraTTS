// SYNTEST.TOS: the synthesizer alone. Reads AURORA.BIN, HEAD.BIN (frames of
// 66 big-endian words) and STEPS.BIN (one byte: steps to run), runs that many
// steps on the DSP and writes DUMP.BIN: the step count, the output samples of
// every step (count, then the samples, 320 words per step), then DSP X
// $0000-$2fff and Y $0000-$3fff.
#include <mint/osbind.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "falcon-dsp.h"
#include "synth-driver.h"

using namespace Aurora;

static FalconDsp g_dsp;
static int32_t g_img[AURORA_IMAGE_WORDS];
static int32_t g_head[400 * 66];
static int32_t g_out[96 * 321];
static int32_t g_x[0x3000], g_y[0x4000];

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
	f = fopen("HEAD.BIN", "rb");
	const int T = (int)(fread(g_head, 4, 400 * 66, f) / 66);
	fclose(f);
	f = fopen("STEPS.BIN", "rb");
	const int steps = fgetc(f);
	fclose(f);
	if (!g_dsp.boot())
		return 2;
	void *ssp = (void *)Super(0L);
	g_dsp.ping();
	synthUpload(&g_dsp, g_img);
	g_dsp.synthReset();
	for (int m = 0; m < steps && m < 96; ++m) {
		int16_t buf[400];
		const int n = g_dsp.synthStep(m < T ? g_head + m * 66 : 0, buf);
		g_out[m * 321] = n;
		for (int i = 0; i < n && i < 320; ++i)
			g_out[m * 321 + 1 + i] = buf[i];
	}
	for (int a = 0; a < 0x3000; ++a)
		g_x[a] = (a >= 0x100 && a < 0x200) ? 0 : g_dsp.readX(a);
	for (int a = 0; a < 0x4000; ++a)
		g_y[a] = (a >= 0x100 && a < 0x200) ? 0 : g_dsp.readY(a);
	SuperToUser(ssp);
	f = fopen("DUMP.BIN", "wb");
	put(f, steps);
	for (int i = 0; i < steps * 321; ++i)
		put(f, g_out[i]);
	for (int a = 0; a < 0x3000; ++a)
		put(f, g_x[a]);
	for (int a = 0; a < 0x4000; ++a)
		put(f, g_y[a]);
	fclose(f);
	ssp = (void *)Super(0L);
	g_dsp.done();
	SuperToUser(ssp);
	g_dsp.release();
	return 0;
}
