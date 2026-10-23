// Compares SYNTEST.TOS's DUMP.BIN with the reference synthesizer run for
// the same steps: outputs per step, then each piece of state.
//   aurora-syncheck IMAGE HEAD.BIN DUMP.BIN
#include <stdio.h>
#include <string.h>
#include <vector>

#include "image.h"
#include "synth.h"

using namespace Aurora;

static std::vector<int32_t> readBe(const char *path) {
	FILE *f = fopen(path, "rb");
	std::vector<int32_t> v;
	unsigned char b[4];
	while (f && fread(b, 1, 4, f) == 4)
		v.push_back((int32_t)((uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3]));
	if (f)
		fclose(f);
	return v;
}

static int s24(int32_t v) {
	return (int32_t)((uint32_t)v << 8) >> 8;
}

static void compare(const char *name, const int32_t *dsp, const int32_t *ref, int n, int base = 0) {
	int bad = 0, first = -1;
	for (int i = 0; i < n; ++i)
		if (s24(dsp[i]) != s24(ref[i])) {
			if (first < 0)
				first = i;
			++bad;
		}
	if (!bad) {
		printf("  %-10s equal (%d)\n", name, n);
		return;
	}
	printf("  %-10s %d of %d differ; first [%d] at %04x: dsp %06x ref %06x", name, bad, n, first, base + first,
	       dsp[first] & 0xffffff, ref[first] & 0xffffff);
	for (int i = first + 1; i < n && i < first + 4; ++i)
		printf(" | %06x %06x", dsp[i] & 0xffffff, ref[i] & 0xffffff);
	printf("\n");
}

static Synth syn;

int main(int, char **argv) {
	std::vector<int32_t> img = loadImage(argv[1]);
	std::vector<int32_t> head = readBe(argv[2]);
	std::vector<int32_t> d = readBe(argv[3]);
	const int T = (int)head.size() / 66;
	const int steps = d[0];
	const int32_t *outs = &d[1];
	const int32_t *X = &d[1 + steps * 321];
	const int32_t *Y = X + 0x3000;
	syn.reset(img.data());
	for (int m = 0; m < steps; ++m) {
		int16_t buf[400];
		const int n = syn.run(m < T ? &head[m * 66] : 0, buf);
		const int32_t *o = outs + m * 321;
		int bad = o[0] != n, first = -1;
		for (int i = 0; i < n && i < o[0]; ++i)
			if (s24(o[1 + i]) != buf[i]) {
				if (first < 0)
					first = i;
				bad = 1;
			}
		printf("step %d: %d samples (dsp %d) %s", m, n, o[0], bad ? "DIFFER" : "equal");
		if (first >= 0)
			printf(" from [%d]: dsp %d %d %d ref %d %d %d", first, s24(o[1 + first]), s24(o[2 + first]),
			       s24(o[3 + first]), buf[first], buf[first + 1], buf[first + 2]);
		printf("\n");
	}
	const int cur = (steps - 1) & 1;
	printf("state after %d steps:\n", steps);
	std::vector<int32_t> v(2048);
	for (int s = 0; s < 2; ++s) {
		char nm[32];
		sprintf(nm, "amp[%d]", s);
		compare(nm, X + 0x0200 + 80 * s, syn.amp[s], 80, 0x0200 + 80 * s);
		sprintf(nm, "mag[%d]", s);
		compare(nm, X + 0x2800 + 513 * s, syn.mag[s], 513, 0x2800 + 513 * s);
	}
	const int32_t inc[2] = {X[0x95], X[0x96]};
	compare("inc28", inc, syn.inc28, 2, 0x95);
	const int32_t ph[2] = {X[0x80], Y[0x80]};
	const int32_t phr[2] = {(int32_t)(syn.phase48 >> 24), (int32_t)(syn.phase48 & 0xffffff)};
	compare("phase", ph, phr, 2, 0x80);
	const int32_t lcg[1] = {X[0x91]};
	const int32_t lcgr[1] = {(int32_t)syn.lcg};
	compare("lcg", lcg, lcgr, 1, 0x91);
	compare("ola", X + 0x1000, syn.ola, 2048, 0x1000);
	compare("fre", X + 0x1800, syn.fre, 1024, 0x1800);
	compare("fim", Y + 0x1800, syn.fim, 1024, 0x1800);
	const int32_t xc[1] = {X[0x92]};
	const int32_t xcr[1] = {(int32_t)syn.xcount};
	compare("xcount", xc, xcr, 1, 0x92);
	(void)cur;
	return 0;
}
