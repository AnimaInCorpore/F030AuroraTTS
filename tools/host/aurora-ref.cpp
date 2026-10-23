// Host reference: phoneme ids -> fixed-point network outputs (and later audio).
//   aurora-ref IMAGE --ids 1,6,16,... --dump out.bin
// The dump is little-endian int32: N, T, then dur[N], logd[N], pit[N],
// head[T * 66].
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "model.h"
#include "synth.h"
#include "image.h"

using namespace Aurora;


static void put32(FILE *f, uint32_t v) {
	unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
	fwrite(b, 1, 4, f);
}

static void writeWav(const char *path, const std::vector<int16_t> &pcm, uint32_t rate) {
	FILE *f = fopen(path, "wb");
	const uint32_t bytes = (uint32_t)pcm.size() * 2;
	fwrite("RIFF", 1, 4, f);
	put32(f, 36 + bytes);
	fwrite("WAVEfmt ", 1, 8, f);
	put32(f, 16);
	put32(f, 1 | (1 << 16));
	put32(f, rate);
	put32(f, rate * 2);
	put32(f, 2 | (16 << 16));
	fwrite("data", 1, 4, f);
	put32(f, bytes);
	for (size_t i = 0; i < pcm.size(); ++i) {
		const unsigned char b[2] = {(unsigned char)pcm[i], (unsigned char)(pcm[i] >> 8)};
		fwrite(b, 1, 2, f);
	}
	fclose(f);
}

static uint32_t checksum(const int32_t *v, long n) {
	uint32_t s = 0;
	for (long i = 0; i < n; ++i)
		s = s * 31u + (uint32_t)(v[i] & 0xffffff);
	return s;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: aurora-ref IMAGE --ids a,b,c [--dump FILE]\n");
		return 2;
	}
	std::vector<int32_t> img = loadImage(argv[1]);
	std::vector<uint8_t> ids;
	const char *dump = 0, *wav = 0;
	bool silentNoise = false, checksums = false;
	for (int i = 2; i < argc; ++i) {
		if (!strcmp(argv[i], "--ids") && i + 1 < argc) {
			for (char *p = argv[++i]; *p;) {
				ids.push_back((uint8_t)strtol(p, &p, 10));
				if (*p == ',')
					++p;
			}
		} else if (!strcmp(argv[i], "--wav") && i + 1 < argc) {
			wav = argv[++i];
		} else if (!strcmp(argv[i], "--checksums")) {
			checksums = true;
		} else if (!strcmp(argv[i], "--ids-file") && i + 1 < argc) {
			FILE *f = fopen(argv[++i], "rb");
			int c;
			while (f && (c = fgetc(f)) != EOF)
				ids.push_back((uint8_t)c);
			if (f)
				fclose(f);
		} else if (!strcmp(argv[i], "--no-noise")) {
			silentNoise = true;
		} else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
			dump = argv[++i];
		}
	}
	const int N = (int)ids.size();
	Model m;
	m.img = img.data();
	std::vector<int32_t> x(N * kD), work(2 * N * kD + 64), logd(N), pit(N), center2(N);
	std::vector<uint8_t> dur(N);
	m.encodeAll(ids.data(), N, x.data(), logd.data(), pit.data(), dur.data(), work.data());
	int T;
	centres(dur.data(), N, center2.data(), &T);
	std::vector<int32_t> up((size_t)T * (kD + 2)), head((size_t)T * kHead), dwork((size_t)2 * T * kD);
	int near = 0;
	for (int t = 0; t < T; ++t)
		m.upsampleFrame(x.data(), dur.data(), center2.data(), N, t, &up[(size_t)t * (kD + 2)], &near);
	m.decode(up.data(), T, head.data(), dwork.data());
	if (checksums) {
		// the fields AURTEST.TOS writes
		uint32_t ds = 0;
		for (int n = 0; n < N; ++n)
			ds += dur[n];
		static Synth syn;
		syn.reset(img.data());
		const int64_t want = syn.outputCount(T);
		std::vector<int32_t> pcm;
		int16_t buf[400];
		for (int m = 0; (int64_t)pcm.size() < want; ++m) {
			const int n = syn.run(m < T ? &head[(size_t)m * kHead] : 0, buf);
			pcm.insert(pcm.end(), buf, buf + n);
		}
		pcm.resize(want);
		printf("{\"tokens\": %d, \"frames\": %d, \"duration_sum\": %u, \"x\": %u, \"predict\": %u, \"head\": %u, "
		       "\"samples\": %lld, \"pcm\": %u}\n",
		       N, T, ds, checksum(x.data(), (long)N * kD), checksum(logd.data(), N) ^ (checksum(pit.data(), N) * 7u),
		       checksum(head.data(), (long)T * kHead), (long long)want, checksum(pcm.data(), (long)want));
		return 0;
	}
	printf("%d tokens, %d frames\n", N, T);
	if (wav) {
		static Synth syn;
		syn.reset(img.data(), silentNoise);
		const int64_t want = syn.outputCount(T);
		std::vector<int16_t> pcm;
		int16_t buf[400];
		for (int m = 0; (int64_t)pcm.size() < want; ++m) {
			const int n = syn.run(m < T ? &head[(size_t)m * kHead] : 0, buf);
			pcm.insert(pcm.end(), buf, buf + n);
		}
		pcm.resize(want);
		writeWav(wav, pcm, 24585);
		printf("%lld samples at 24585 Hz\n", (long long)want);
	}
	if (dump) {
		FILE *f = fopen(dump, "wb");
		int32_t hdr[2] = {N, T};
		fwrite(hdr, 4, 2, f);
		for (int n = 0; n < N; ++n) {
			int32_t d = dur[n];
			fwrite(&d, 4, 1, f);
		}
		fwrite(logd.data(), 4, N, f);
		fwrite(pit.data(), 4, N, f);
		fwrite(head.data(), 4, head.size(), f);
		fclose(f);
	}
	return 0;
}
