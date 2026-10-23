// Runs the network as DSP programs on the host interpreter and checks every
// output word against the direct fixed-point reference (model.h).
//   aurora-vmtest IMAGE a,b,c [a,b,c ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "driver.h"
#include "image.h"

using namespace Aurora;


struct HostDevice {
	HostVm vm;
	long wordsIn, wordsOut, steps;
	int maxX, maxY;
	void writeX(int addr, const int32_t *w, int n) {
		for (int i = 0; i < n; ++i)
			vm.x(addr + i) = w[i];
		wordsIn += n + 3;
	}
	void writeY(int addr, const int32_t *w, int n) {
		for (int i = 0; i < n; ++i)
			vm.y(addr + i) = w[i];
		wordsIn += n + 3;
	}
	int sendPc, sendN;
	const int32_t *sendIn;
	void stepSend(int pc, const int32_t *in, int nin) {
		sendPc = pc;
		sendIn = in;
		sendN = nin;
	}
	void stepReceive(int32_t *out, int nout, bool = false) {
		step(sendPc, sendIn, sendN, out, nout);
	}
	void step(int pc, const int32_t *in, int nin, int32_t *out, int nout) {
		vm.run(pc, in, nin, out, nout);
		if (vm.fault || vm.outCount != nout || vm.inRead != nin) {
			fprintf(stderr, "program fault at %04x (in %d/%d, out %d/%d)\n", pc, vm.inRead, nin, vm.outCount, nout);
			exit(1);
		}
		wordsIn += nin + 1;
		wordsOut += nout;
		++steps;
	}
};

static HostDevice dev;

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: aurora-vmtest IMAGE ids [ids ...]\n");
		return 2;
	}
	std::vector<int32_t> img = loadImage(argv[1]);
	Model m;
	m.img = img.data();
	int failures = 0;
	for (int a = 2; a < argc; ++a) {
		std::vector<uint8_t> ids;
		for (char *p = argv[a]; *p;) {
			ids.push_back((uint8_t)strtol(p, &p, 10));
			if (*p == ',')
				++p;
		}
		const int N = (int)ids.size();
		// reference
		std::vector<int32_t> x(N * kD), work(2 * N * kD + 64), logd(N), pit(N), c2(N);
		std::vector<uint8_t> dur(N);
		m.encodeAll(ids.data(), N, x.data(), logd.data(), pit.data(), dur.data(), work.data());
		int T;
		centres(dur.data(), N, c2.data(), &T);
		std::vector<int32_t> up((size_t)T * (kD + 2)), head((size_t)T * kHead), dwork((size_t)2 * T * kD);
		int near = 0;
		for (int t = 0; t < T; ++t)
			m.upsampleFrame(x.data(), dur.data(), c2.data(), N, t, &up[(size_t)t * (kD + 2)], &near);
		m.decode(up.data(), T, head.data(), dwork.data());
		// programs
		dev.vm.reset();
		dev.wordsIn = dev.wordsOut = dev.steps = 0;
		static Driver<HostDevice> drv;
		drv.init(&dev, img.data());
		std::vector<int32_t> x2(N * kD), w2(N * kD), logd2(N), pit2(N), c22(N);
		std::vector<uint8_t> dur2(N);
		drv.encode(ids.data(), N, x2.data(), w2.data(), logd2.data(), pit2.data(), dur2.data());
		const long encIn = dev.wordsIn, encOut = dev.wordsOut;
		int bad = 0;
		for (int n = 0; n < N; ++n)
			bad += dur[n] != dur2[n] || logd[n] != logd2[n] || pit[n] != pit2[n];
		for (int i = 0; i < N * kD; ++i)
			bad += x[i] != x2[i];
		std::vector<int32_t> head2((size_t)T * kHead), dw2((size_t)T * kD);
		if (!bad) {
			drv.decode(up.data(), T, head2.data(), dw2.data());
			for (size_t i = 0; i < head.size(); ++i)
				bad += head[i] != head2[i];
			std::vector<int32_t> head3((size_t)T * kHead);
			drv.decodeTokens(x2.data(), dur2.data(), c2.data(), N, T, head3.data(), dw2.data());
			for (size_t i = 0; i < head.size(); ++i)
				bad += head[i] != head3[i];
		}
		if (getenv("PASSES")) {
			drv.begin();
			printf("passes:");
			for (int i = 0; i < drv.passCount; ++i)
				printf(" %u", drv.passSum[i]);
			printf("\n");
		}
		printf("%3d tokens %4d frames: %s; host port: encoder %ld in %ld out, decoder %ld in %ld out, %ld steps\n",
		       N, T, bad ? "MISMATCH" : "equal", encIn, encOut, dev.wordsIn - encIn, dev.wordsOut - encOut, dev.steps);
		failures += bad != 0;
	}
	return failures ? 1 : 0;
}
