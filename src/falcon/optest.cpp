// OPTEST.TOS: runs each single-operation case (optest.h) on the DSP and
// writes every output word to RESULT.BIN (big-endian, 64 per case).
#include <mint/osbind.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "falcon-dsp.h"
#include "optest.h"

using namespace Aurora;

static FalconDsp g_dsp;
static int32_t g_img[AURORA_IMAGE_WORDS];
static uint32_t g_out[kOpCases * 64 + 4];

int main() {
	FILE *f = fopen("AURORA.BIN", "rb");
	if (!f || fread(g_img, 4, AURORA_IMAGE_WORDS, f) != AURORA_IMAGE_WORDS)
		return 1;
	fclose(f);
	for (long i = 0; i < AURORA_IMAGE_WORDS; ++i)
		g_img[i] = (g_img[i] << 8) >> 8;
	if (!g_dsp.boot())
		return 2;
	void *ssp = (void *)Super(0L);
	g_out[0] = g_dsp.ping() ? 1 : 0;
	for (int k = 0; k < kOpCases && !g_dsp.failed; ++k) {
		int nout;
		const int pc = buildOpCase(&g_dsp, g_img, k, &nout);
		int32_t o[64];
		g_dsp.step(pc, 0, 0, o, nout);
		for (int i = 0; i < nout; ++i)
			g_out[1 + k * 64 + i] = (uint32_t)o[i];
	}
	g_out[1 + kOpCases * 64] = g_dsp.failed;
	SuperToUser(ssp);
	f = fopen("RESULT.BIN", "wb");
	fwrite(g_out, 4, kOpCases * 64 + 2, f);
	fclose(f);
	ssp = (void *)Super(0L);
	g_dsp.done();
	SuperToUser(ssp);
	g_dsp.release();
	return 0;
}
