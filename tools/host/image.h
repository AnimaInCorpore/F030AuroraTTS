// Loads AURORA.BIN: big-endian words, each a sign-extended 24-bit DSP word.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "aurora-layout.h"

inline std::vector<int32_t> loadImage(const char *path) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		perror(path);
		exit(1);
	}
	std::vector<int32_t> img;
	unsigned char b[4];
	while (fread(b, 1, 4, f) == 4) {
		uint32_t w = (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
		img.push_back((int32_t)w);
	}
	fclose(f);
	if (img.size() != AURORA_IMAGE_WORDS) {
		fprintf(stderr, "%s: %zu words, the layout expects %d\n", path, img.size(), AURORA_IMAGE_WORDS);
		exit(1);
	}
	return img;
}
