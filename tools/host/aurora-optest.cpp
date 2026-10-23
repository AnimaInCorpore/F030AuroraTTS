// Checks OPTEST.TOS's RESULT.BIN against the host interpreter, op by op.
#include <stdio.h>
#include <vector>

#include "image.h"
#include "optest.h"

using namespace Aurora;

struct VmDev {
	HostVm vm;
	void writeX(int a, const int32_t *w, int n) {
		for (int i = 0; i < n; ++i)
			vm.x(a + i) = w[i];
	}
	void writeY(int a, const int32_t *w, int n) {
		for (int i = 0; i < n; ++i)
			vm.y(a + i) = w[i];
	}
};

static VmDev dev;

int main(int, char **argv) {
	std::vector<int32_t> img = loadImage(argv[1]);
	FILE *f = fopen(argv[2], "rb");
	std::vector<uint32_t> got(kOpCases * 64 + 2);
	if (!f || fread(got.data(), 4, got.size(), f) != got.size()) {
		fprintf(stderr, "no result\n");
		return 2;
	}
	fclose(f);
	for (auto &w : got)
		w = (w >> 24) | ((w >> 8) & 0xff00) | ((w << 8) & 0xff0000) | (w << 24);
	printf("ping %s, port %s\n", got[0] ? "ok" : "FAILED", got[1 + kOpCases * 64] ? "FAILED" : "ok");
	static const char *names[kOpCases] = {"MVY s0", "MVY s1", "MVY s2", "MVY s3", "MVX", "LN", "ACT gelu",
	                                      "ACT sigmoid", "ADD/SUB/MUL/RELU", "ring dil1", "ring sh1", "ring dil2",
	                                      "LN small", "LN large", "MVX 112", "MVY 112 int"};
	int bad = 0;
	for (int k = 0; k < kOpCases; ++k) {
		dev.vm.reset();
		int nout;
		const int pc = buildOpCase(&dev, img.data(), k, &nout);
		int32_t o[64];
		dev.vm.run(pc, 0, 0, o, nout);
		int first = -1, count = 0;
		for (int i = 0; i < nout; ++i) {
			const int32_t g = (int32_t)(got[1 + k * 64 + i] << 8) >> 8;
			if (g != o[i]) {
				if (first < 0)
					first = i;
				++count;
			}
		}
		printf("%-18s %s", names[k], count ? "MISMATCH" : "equal");
		if (count) {
			const int32_t g = (int32_t)(got[1 + k * 64 + first] << 8) >> 8;
			printf(" (%d of %d words, first [%d] dsp %06x host %06x)", count, nout, first, g & 0xffffff, o[first] & 0xffffff);
			++bad;
		}
		printf("\n");
	}
	return bad ? 1 : 0;
}
