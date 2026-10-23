// Falcon timing/self-test for the text frontend (TOS program).
//
//   G2PTIME.TOS [DICT.BIN [lines.txt [repeats]]]
//
// Loads DICT.BIN, converts each line (default: built-in sentences) and
// prints the ids of the first sentence, the per-sentence time and totals.
// Build: see src/frontend/README.md.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../src/frontend/g2p.h"

static AuroraFront::G2p g2p;
static uint8_t ids[4096];

static const char *kBuiltin[] = {
    "The first physician to see the President at Parkland Hospital was Dr. Charles J. Carrico, a resident in general surgery.",
    "Mr. Wakefield winds up his graphic but somewhat sensational account by describing another religious service, which may appropriately be inserted here.",
    "On the seventh July, eighteen thirty-seven,",
    "He paid $3.50 for 1,234 apples on the 21st of May 1984.",
    "This is the approximate time he entered the roominghouse, according to Earlene Roberts, the housekeeper there.",
    "I will read the book tomorrow, and the record will show that I used it.",
    0};

int main(int argc, char **argv) {
    const char *dictPath = argc > 1 ? argv[1] : "DICT.BIN";
    FILE *f = fopen(dictPath, "rb");
    if (!f) { printf("cannot open %s\n", dictPath); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *dict = (uint8_t *)malloc(n);
    if (!dict || fread(dict, 1, n, f) != (size_t)n) { printf("read error\n"); return 1; }
    fclose(f);
    if (!g2p.load(dict, n)) { printf("bad DICT.BIN\n"); return 1; }
    printf("DICT.BIN: %ld bytes\n", n);

    static char text[64 * 1024];
    const char *lines[512];
    int nl = 0;
    if (argc > 2) {
        FILE *t = fopen(argv[2], "rb");
        if (!t) { printf("cannot open %s\n", argv[2]); return 1; }
        size_t m = fread(text, 1, sizeof(text) - 1, t);
        fclose(t);
        text[m] = 0;
        char *p = text;
        while (*p && nl < 512) {
            char *e = p;
            while (*e && *e != '\n' && *e != '\r') e++;
            char c = *e;
            *e = 0;
            if (e > p) lines[nl++] = p;
            if (!c) break;
            p = e + 1;
        }
    } else {
        for (; kBuiltin[nl]; nl++) lines[nl] = kBuiltin[nl];
    }
    int reps = argc > 3 ? atoi(argv[3]) : 1;
    if (reps < 1) reps = 1;

    long totalIds = 0;
    clock_t t0 = clock();
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < nl; i++) {
            clock_t s = clock();
            int k = g2p.textToIds(lines[i], ids, sizeof(ids));
            clock_t e = clock();
            if (r == 0) {
                printf("%3d: %4d ids %5ld ms  (%d words, %d oov)\n", i, k,
                       (long)(e - s) * 1000L / CLOCKS_PER_SEC, g2p.statWords, g2p.statOov);
                if (i == 0) {
                    for (int j = 0; j < k; j++) printf("%d ", ids[j]);
                    printf("\n");
                }
            }
            totalIds += k;
        }
    }
    clock_t t1 = clock();
    long ms = (long)(t1 - t0) * 1000L / CLOCKS_PER_SEC;
    printf("total: %d sentences x %d, %ld ids, %ld ms, %ld ms/sentence\n", nl, reps, totalIds, ms,
           ms / ((long)nl * reps));
    return 0;
}
