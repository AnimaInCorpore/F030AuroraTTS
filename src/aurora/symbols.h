// AuroraTTS-100K's 79 input symbols (upstream SYMBOLS), and ARPAbet phoneme
// input: whitespace-separated symbols, "_" for the space between words,
// punctuation , . ? ! ; : - as themselves. Like upstream text_to_ids, the
// result starts and ends with <sil>, never repeats a space, and has no space
// before the final <sil>.
#pragma once
#include <stdint.h>
#include <string.h>

namespace Aurora {

const int kSymPad = 0, kSymSil = 1, kSymSpace = 71;

inline const char *symbolName(int id) {
	static const char *const kNames[79] = {
	    "<pad>", "<sil>", "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
	    "T", "TH", "V", "W", "Y", "Z", "ZH", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
	    "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0",
	    "EY1", "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "UH0",
	    "UH1", "UH2", "UW0", "UW1", "UW2", " ", ",", ".", "?", "!", ";", ":", "-"};
	return id >= 0 && id < 79 ? kNames[id] : "?";
}

inline int symbolId(const char *s, int len) {
	if (len == 1 && s[0] == '_')
		return kSymSpace;
	for (int id = 2; id < 79; ++id) {
		const char *n = symbolName(id);
		if ((int)strlen(n) == len && !memcmp(n, s, len))
			return id;
	}
	// an unstressed vowel written without its digit: stress 0
	for (int id = 2; id < 79; ++id) {
		const char *n = symbolName(id);
		if ((int)strlen(n) == len + 1 && !memcmp(n, s, len) && n[len] == '0')
			return id;
	}
	return -1;
}

// Phoneme text to ids; returns the count, or -(position + 1) of a symbol it
// does not know, or 0 when max is too small.
inline int phonemesToIds(const char *text, uint8_t *ids, int max) {
	int n = 0;
	if (max < 2)
		return 0;
	ids[n++] = kSymSil;
	const char *p = text;
	while (*p) {
		while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
			++p;
		if (!*p)
			break;
		const char *s = p;
		if (strchr(",.?!;:-", *p))
			++p;
		else
			while (*p && !strchr(" \t\n\r,.?!;:-", *p))
				++p;
		char tok[8];
		int len = (int)(p - s);
		if (len > 7)
			return -(int)(s - text) - 1;
		for (int i = 0; i < len; ++i)
			tok[i] = (s[i] >= 'a' && s[i] <= 'z') ? (char)(s[i] - 32) : s[i];
		const int id = symbolId(tok, len);
		if (id < 0)
			return -(int)(s - text) - 1;
		if (id == kSymSpace && (ids[n - 1] == kSymSil || ids[n - 1] == kSymSpace))
			continue;
		if (n + 2 > max)
			return 0;
		ids[n++] = (uint8_t)id;
	}
	while (n > 1 && ids[n - 1] == kSymSpace)
		--n;
	ids[n++] = kSymSil;
	return n;
}

}  // namespace Aurora
