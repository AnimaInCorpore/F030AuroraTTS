// The synthesizer pass: the head's frames to 24,585 Hz 16-bit samples.
//
// Dev provides writeX/writeY (as for the network passes) and
//   void synthReset();
//   int synthStep(const int32_t *head66 /* null: a pad step */, int16_t *out);
// The DSP keeps the synthesizer's tables at fixed addresses (dsp/synth.asm);
// the host device runs synth.h's reference instead.
#pragma once
#include <stdint.h>

#include "aurora-layout.h"
#include "synth.h"

namespace Aurora {

// dsp/synth.asm's memory map: the tables the host uploads
const int kSynXThq = 0x00a8, kSynXTw = 0x1c00, kSynXSin2 = 0x2000, kSynXWola = 0x0a00;
const int kSynYExp = 0x0200, kSynYMelh = 0x0700, kSynYInc = 0x0b00, kSynYNmap = 0x1000, kSynYFir = 0x1500,
          kSynYConst = 0x1580, kSynYRes = 0x2000;

template <class Dev>
void synthUpload(Dev *dev, const int32_t *img) {
	// each harmonic's offset as the address of its table entry for index 0
	int32_t thq[81];
	for (int h = 0; h < 80; ++h)
		thq[h] = kSynXSin2 + 512 + img[AU_SYN_THETA_OFF + h];
	thq[80] = kSynXSin2 + 512;   // the spare harmonic of an odd count (dsp/synhot.asm)
	dev->writeX(kSynXThq, thq, 81);
	// the sine table rotated by half a cycle: entry j holds sin(2 pi (j + 512) / 1024)
	// FFT twiddles as L words: X cos(2 pi k / 1024) = sin(.. k + 256 ..), Y sin
	dev->writeX(kSynXTw, img + AU_TAB_SIN_OFF + 256, 512);
	dev->writeY(kSynXTw, img + AU_TAB_SIN_OFF, 512);
	// the harmonics' table: two periods, entry j for index j - 512
	dev->writeX(kSynXSin2, img + AU_TAB_SIN_OFF + 512, 512);
	dev->writeX(kSynXSin2 + 512, img + AU_TAB_SIN_OFF, 1024);
	dev->writeX(kSynXSin2 + 1536, img + AU_TAB_SIN_OFF, 512);
	dev->writeX(kSynXWola, img + AU_TAB_WOLA_OFF, AU_TAB_WOLA_LEN);
	dev->writeY(kSynYExp, img + AU_TAB_EXP_OFF, AU_TAB_EXP_LEN);
	dev->writeY(kSynYMelh, img + AU_TAB_MELH_OFF, AU_TAB_MELH_LEN);
	dev->writeY(kSynYInc, img + AU_TAB_INC_OFF, AU_TAB_INC_LEN);
	// the noise bin map as runs of bins sharing a band pair: [runs], then per
	// run [band, count, the count fractions] (dsp/synhot.asm's noise_mags)
	{
		int32_t runs[AU_TAB_NMAP_LEN / 2 + 64];
		int n = 1, count = 0, at = -1;
		const int32_t *map = img + AU_TAB_NMAP_OFF;
		for (int k = 0; k < AU_TAB_NMAP_LEN / 2; ++k) {
			if (at < 0 || map[2 * k] != runs[at]) {
				at = n;
				runs[n++] = map[2 * k];
				runs[n++] = 0;
				++count;
			}
			++runs[at + 1];
			runs[n++] = map[2 * k + 1];
		}
		runs[0] = count;
		dev->writeY(kSynYNmap, runs, n);
	}
	dev->writeY(kSynYFir, img + AU_SYN_FIR_OFF, AU_SYN_FIR_LEN);
	dev->writeY(kSynYConst, img + AU_SYN_CONST_OFF, AU_SYN_CONST_LEN);
	dev->writeY(kSynYConst + 8, img + AU_SYN_F0ROW_OFF, AU_SYN_F0ROW_LEN);
	dev->writeY(kSynYRes, img + AU_TAB_RES_OFF, AU_TAB_RES_LEN);
}

// The number of output samples for T frames: (T - 1) 256 at 22,050 Hz.
inline long synthOutputCount(const int32_t *img, int T) {
	const int64_t step = ((int64_t)1 << 24) + img[AU_SYN_CONST_OFF + 3];
	const int64_t len = ((int64_t)(T - 1) * kHop) << 24;
	return (long)((len + step - 1) / step);
}

// Renders head (T, kHead) into pcm (synthOutputCount samples, and room for
// 400 more: each step's output lands in place). Calls progress(done, total)
// after every step when given.
template <class Dev>
long synthesize(Dev *dev, const int32_t *img, const int32_t *head, int T, int16_t *pcm,
                void (*progress)(long, long) = 0) {
	synthUpload(dev, img);
	dev->synthReset();
	const long want = synthOutputCount(img, T);
	long have = 0;
	for (int m = 0; have < want; ++m) {
		const int n = dev->synthStep(m < T ? head + (long)m * 66 : 0, pcm + have);
		have += n;
		if (have > want)
			have = want;
		if (progress)
			progress(have, want);
		if (m > T + 8)
			break;   // never: a guard against a device that stops answering
	}
	return have;
}

}  // namespace Aurora
