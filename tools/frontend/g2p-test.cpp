// Host test CLI for src/frontend/g2p.h
//
//   g2p-test DICT.BIN "text"          print token ids of one text
//   g2p-test DICT.BIN -f lines.txt    one line of ids per input line
//   g2p-test DICT.BIN -oov words.txt  OOV path (letter-to-sound) phones per word
//   g2p-test DICT.BIN -time lines.txt N   convert the file N times, print timing
//
// Input files are 8-bit (Latin-1).  Output: ids separated by spaces; for
// -f also "#stats words dict homograph oov" lines when -stats is given.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../src/frontend/g2p.h"

static const char *kPhones[69] = {
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH",
    "V", "W", "Y", "Z", "ZH", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1",
    "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1",
    "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "UH0", "UH1",
    "UH2", "UW0", "UW1", "UW2"};

static AuroraFront::G2p g2p;  // large: keep off the stack
static uint8_t ids[16384];

static uint8_t *readFile(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = (uint8_t *)malloc(n + 1);
    if (fread(p, 1, n, f) != (size_t)n) { fclose(f); free(p); return 0; }
    p[n] = 0;
    fclose(f);
    *size = n;
    return p;
}

static void printIds(const char *text) {
    int n = g2p.textToIds(text, ids, (int)sizeof(ids));
    if (n < 0) { printf("ERR %d\n", n); return; }
    for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
    printf("\n");
}

// iterate lines of a buffer (handles \r\n)
template <class F> static void forLines(char *buf, F fn) {
    char *p = buf;
    while (*p) {
        char *e = p;
        while (*e && *e != '\n') e++;
        char save = *e;
        char *t = e;
        if (t > p && t[-1] == '\r') t--;
        char s2 = *t;
        *t = 0;
        fn(p);
        *t = s2;
        if (!save) break;
        p = e + 1;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: g2p-test DICT.BIN \"text\" | -f file [-stats] | -oov file | -time file N\n");
        return 2;
    }
    long dsize;
    uint8_t *dict = readFile(argv[1], &dsize);
    if (!dict || !g2p.load(dict, dsize)) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (!strcmp(argv[2], "-f") && argc >= 4) {
        long n;
        char *buf = (char *)readFile(argv[3], &n);
        if (!buf) return 1;
        bool stats = argc >= 5 && !strcmp(argv[4], "-stats");
        forLines(buf, [&](char *line) {
            printIds(line);
            if (stats) printf("#stats %d %d %d %d\n", g2p.statWords, g2p.statDict, g2p.statHomograph, g2p.statOov);
        });
        return 0;
    }
    if (!strcmp(argv[2], "-oov") && argc >= 4) {
        long n;
        char *buf = (char *)readFile(argv[3], &n);
        if (!buf) return 1;
        forLines(buf, [&](char *line) {
            uint8_t ph[AuroraFront::G2p::MAXPH];
            int np = g2p.wordPhones(line, (int)strlen(line), ph);
            printf("%s", line);
            for (int i = 0; i < np; i++) printf(" %s", kPhones[ph[i]]);
            printf("\n");
        });
        return 0;
    }
    if (!strcmp(argv[2], "-time") && argc >= 5) {
        long n;
        char *buf = (char *)readFile(argv[3], &n);
        if (!buf) return 1;
        int reps = atoi(argv[4]);
        long lines = 0, total = 0;
        clock_t t0 = clock();
        for (int r = 0; r < reps; r++)
            forLines(buf, [&](char *line) { total += g2p.textToIds(line, ids, (int)sizeof(ids)); lines++; });
        double s = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("%ld lines, %ld ids, %.3f s, %.1f us/line\n", lines, total, s, 1e6 * s / lines);
#ifdef AURORA_G2P_STATS
        AuroraFront::G2pCounters &c = AuroraFront::g2pCounters();
        printf("per line: %.0f bits, %.0f symbols, %.1f dict lookups, %.1f tagger features, %.1f LTS nodes\n",
               (double)c.bits / lines, (double)c.syms / lines, (double)c.lookups / lines,
               (double)c.feats / lines, (double)c.ltsNodes / lines);
#endif
        return 0;
    }
    if (!strcmp(argv[2], "-tags") && argc >= 4) {
        printIds(argv[3]);
        char tok[256];
        for (int t = 0;; t++) {
            int tag = g2p.debugToken(t, tok, sizeof(tok));
            if (tag == -2) break;
            printf("%s/%d ", tok, tag);
        }
        printf("\n");
        return 0;
    }
    printIds(argv[2]);
    return 0;
}
