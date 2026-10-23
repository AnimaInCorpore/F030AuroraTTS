// AuroraTTS English text frontend for the Atari Falcon030.
//
// Reproduces third_party/auroratts text_to_ids(), i.e. g2p_en (number
// expansion via inflect, accent stripping, TweetTokenizer, homographs,
// CMUdict) followed by the symbol mapping of the model.  Data comes from
// DICT.BIN (tools/frontend/build-dict.py).  See src/frontend/README.md.
//
// Header-only, C++11, no exceptions/RTTI/STL/iostreams/floating point, no
// dynamic allocation.  The G2p object holds ~20 KB of work buffers: allocate
// it statically or on the heap, not on a small TOS stack.
#ifndef AURORA_FRONTEND_G2P_H
#define AURORA_FRONTEND_G2P_H

#include <stdint.h>
#include <string.h>

namespace AuroraFront {

#ifdef AURORA_G2P_STATS  // work counters for the host cost estimate
struct G2pCounters { long bits, syms, lookups, feats, ltsNodes; };
inline G2pCounters &g2pCounters() { static G2pCounters c; return c; }
#define G2P_COUNT(f, n) (AuroraFront::g2pCounters().f += (n))
#else
#define G2P_COUNT(f, n) ((void)0)
#endif

// Model symbol ids (SYMBOLS in modeling_auroratts.py).
enum {
    ID_PAD = 0, ID_SIL = 1, ID_PHONE0 = 2, NUM_PHONES = 69,
    ID_SPACE = 71, ID_COMMA = 72, ID_PERIOD = 73, ID_QUESTION = 74,
    ID_EXCLAIM = 75, ID_SEMICOLON = 76, ID_COLON = 77, ID_DASH = 78
};

// Phone ids inside DICT.BIN (= model id - 2): 24 consonants, then the 15
// vowels x stress 0/1/2.
enum { PH_FIRST_VOWEL = 24 };
enum { PH_B = 0, PH_CH, PH_D, PH_DH, PH_F, PH_G, PH_HH, PH_JH, PH_K, PH_L, PH_M, PH_N, PH_NG,
       PH_P, PH_R, PH_S, PH_SH, PH_T, PH_TH, PH_V, PH_W, PH_Y, PH_Z, PH_ZH };
enum { PH_AA = 24, PH_AE = 27, PH_AH = 30, PH_AO = 33, PH_AW = 36, PH_AY = 39, PH_EH = 42,
       PH_ER = 45, PH_EY = 48, PH_IH = 51, PH_IY = 54, PH_OW = 57, PH_OY = 60, PH_UH = 63,
       PH_UW = 66 };

struct G2p {
    enum { WORK = 8192, MAXTOK = 1536, MAXWORD = 64, MAXPH = 96 };

    // Statistics of the last textToIds() call (for testing).
    int statWords, statDict, statHomograph, statOov;

    G2p() : d(0), dsize(0) {}

    bool load(const uint8_t *dict, long size) {
        d = 0;
        if (!dict || size < 64 || memcmp(dict, "AGP1", 4) != 0) return false;
        if ((long)be32(dict + 4) > size) return false;
        d = dict;
        dsize = size;
        nwords = be32(d + 8);
        nblocks = be32(d + 12);
        blockSize = be32(d + 16);
        index = d + be32(d + 24);
        firstWords = d + be32(d + 28);
        bits = d + be32(d + 32);
        hPre = d + be32(d + 36);
        hLet = d + be32(d + 40);
        hPh = d + be32(d + 44);
        homo = d + be32(d + 48);
        lts = d + be32(d + 52);
        nhomo = be16(homo);
        const uint8_t *tg = d + be32(d + 56);
        ncls = tg[0];
        posMask = tg + 2;
        int h = 2 + ncls + (ncls & 1);
        ntd = be16(tg + h);
        nfeat = be16(tg + h + 2);
        tdOff = tg + be32(tg + h + 4);
        tdBody = tg + be32(tg + h + 8);
        fOff = tg + be32(tg + h + 12);
        fBody = tg + be32(tg + h + 16);
        return true;
    }

    // Converts text (8-bit ASCII/Latin-1, NUL-terminated) to model token ids.
    // Returns the number of ids, -1 if `max` is too small, -2 if the
    // (expanded) text does not fit the work buffers, -3 if not loaded.
    int textToIds(const char *text, uint8_t *ids, int max) {
        if (!d) return -3;
        statWords = statDict = statHomograph = statOov = 0;
        tagged = false;
        int len = normalize(text);
        if (len < 0) return -2;
        int nt = tokenize(bufA, len);
        if (nt < 0) return -2;
        int n = 0;
        if (max < 2) return -1;
        ids[n++] = ID_SIL;
        uint8_t ph[MAXPH];
        for (int t = 0; t < nt; t++) {
            const char *w = bufA + tokStart[t];
            int wl = tokLen[t];
            if (!tokAlpha[t]) {
                int id = -1;
                if (wl == 1) {
                    switch (w[0]) {
                    case ',': id = ID_COMMA; break;
                    case '.': id = ID_PERIOD; break;
                    case '?': id = ID_QUESTION; break;
                    case '!': id = ID_EXCLAIM; break;
                    case '-': id = ID_DASH; break;
                    case ';': id = ID_SEMICOLON; break;
                    case ':': id = ID_COLON; break;
                    }
                }
                if (id >= 0) {
                    if (n >= max) return -1;
                    ids[n++] = (uint8_t)id;
                }
            } else {
                int np = wordToPhones(t, ph);
                for (int i = 0; i < np; i++) {
                    if (n >= max) return -1;
                    ids[n++] = (uint8_t)(ph[i] + ID_PHONE0);
                }
            }
            if (ids[n - 1] != ID_SIL && ids[n - 1] != ID_SPACE) {
                if (n >= max) return -1;
                ids[n++] = ID_SPACE;
            }
        }
        while (ids[n - 1] == ID_SPACE) n--;
        if (n >= max) return -1;
        ids[n++] = ID_SIL;
        return n;
    }

    // Converts Atari ST character set text (TOS) to Latin-1 in place, so
    // accented letters reach textToIds() the way g2p_en would see them.
    // Characters without a Latin-1 equivalent become DEL (dropped later).
    static void atariToLatin1(char *s) {
        static const uint8_t map[0x39] = {
            0xC7, 0xFC, 0xE9, 0xE2, 0xE4, 0xE0, 0xE5, 0xE7, 0xEA, 0xEB, 0xE8, 0xEF, 0xEE, 0xEC, 0xC4, 0xC5,
            0xC9, 0xE6, 0xC6, 0xF4, 0xF6, 0xF2, 0xFB, 0xF9, 0xFF, 0xD6, 0xDC, 0xA2, 0xA3, 0xA5, 0xDF, 0x7F,
            0xE1, 0xED, 0xF3, 0xFA, 0xF1, 0xD1, 0xAA, 0xBA, 0xBF, 0x7F, 0xAC, 0xBD, 0xBC, 0xA1, 0xAB, 0xBB,
            0xE3, 0xF5, 0xD8, 0xF8, 0x7F, 0x7F, 0xC0, 0xC3, 0xD5 };
        for (; *s; s++) {
            uint8_t c = (uint8_t)*s;
            if (c >= 0x80) *s = (char)(c < 0x80 + 0x39 ? map[c - 0x80] : 0x7F);
        }
    }

    // --- lower-level entry points (also used by the test tool) ---------

    // CMUdict (+cache) lookup of a lowercase word. Returns #phones or -1.
    int lookup(const char *w, int wl, uint8_t *out) const {
        if (wl <= 0 || wl >= MAXWORD) return -1;
        G2P_COUNT(lookups, 1);
        // binary search: last block whose first word <= w
        long lo = 0, hi = (long)nblocks - 1, blk = -1;
        while (lo <= hi) {
            long mid = (lo + hi) >> 1;
            const char *fw = (const char *)firstWords + be32(index + 8 * mid + 4);
            if (cmpWord(fw, w, wl) <= 0) { blk = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (blk < 0) return -1;
        BitReader br;
        br.p = bits;
        br.pos = be32(index + 8 * blk);
        char cur[MAXWORD];
        const char *fw = (const char *)firstWords + be32(index + 8 * blk + 4);
        int cl = (int)strlen(fw);
        memcpy(cur, fw, cl);
        long count = (long)nwords - blk * (long)blockSize;
        if (count > (long)blockSize) count = blockSize;
        for (long k = 0; k < count; k++) {
            if (k) {
                int p = decode(br, hPre);
                int ctx = p ? charCode(cur[p - 1]) : 0;
                for (;;) {
                    int s = decode(br, table(hLet, ctx));
                    if (!s) break;
                    if (p < MAXWORD - 1) cur[p++] = kChar(s - 1);
                    ctx = s;
                }
                cl = p;
            }
            int c = cmpN(cur, cl, w, wl);
            if (c > 0) return -1;
            int ctx = NUM_PHONES, np = 0;
            for (;;) {
                int s = decode(br, table(hPh, ctx));
                if (s == NUM_PHONES) break;
                if (c == 0 && np < MAXPH) out[np++] = (uint8_t)s;
                ctx = s;
            }
            if (c == 0) return np;
        }
        return -1;
    }

    // Letter-to-sound rules for a lowercase a-z word.  Returns #phones.
    int letterToSound(const char *w, int wl, uint8_t *out, int maxOut) const {
        int np = 0;
        if (wl > MAXWORD) wl = MAXWORD;
        uint8_t code[MAXWORD], vow[MAXWORD], gstart[MAXWORD];
        for (int i = 0; i < wl; i++) {
            char c = w[i];
            code[i] = (uint8_t)(c - 'a' + 1);
            vow[i] = (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');
            gstart[i] = vow[i] && (i == 0 || !vow[i - 1]);
        }
        int totalStarts = 0;
        for (int i = 0; i < wl; i++) totalStarts += gstart[i];
        int startsBefore = 0;  // group starts at index < i
        const uint8_t *labTab = lts + 4 + 4 * 26;
        for (int i = 0; i < wl; i++) {
            uint8_t f[18];
            static const int win[8] = { -4, -3, -2, -1, 1, 2, 3, 4 };
            for (int k = 0; k < 8; k++) {
                int j = i + win[k];
                bool in = j >= 0 && j < wl;
                f[k] = in ? code[j] : 0;
                f[8 + k] = in ? (vow[j] ? 1 : 2) : 0;
            }
            int vb = startsBefore - ((vow[i] && !gstart[i]) ? 1 : 0);
            int va = totalStarts - startsBefore - gstart[i];
            f[16] = (uint8_t)(vb > 3 ? 3 : vb);
            f[17] = (uint8_t)(va > 3 ? 3 : va);
            startsBefore += gstart[i];
            const uint8_t *nd = lts + be32(lts + 4 + 4 * (code[i] - 1));
            for (;;) {
                uint8_t b0 = nd[0];
                G2P_COUNT(ltsNodes, 1);
                if (!(b0 & 0x80)) break;
                if (f[b0 & 0x1F] == nd[1]) {
                    uint32_t rel = (b0 & 0x40) ? ((uint32_t)nd[2] << 16 | (uint32_t)nd[3] << 8 | nd[4]) : be16(nd + 2);
                    nd += rel;
                } else {
                    nd += (b0 & 0x40) ? 5 : 4;
                }
            }
            int lab = be16(nd);
            // walk the label table (labels are short; table is small)
            const uint8_t *lp = labTab;
            while (lab--) lp += 1 + lp[0];
            for (int k = 0; k < lp[0]; k++)
                if (np < maxOut) out[np++] = lp[1 + k];
        }
        bool anyV = false;
        for (int i = 0; i < np; i++) anyV |= isVowelPh(out[i]);
        if (!anyV && np < maxOut) {  // e.g. "hmm": insert a vowel
            int at = np > 1 ? 1 : np;
            for (int i = np; i > at; i--) out[i] = out[i - 1];
            out[at] = PH_AH + 1;
            np++;
        }
        fixStress(out, np);
        return np;
    }

    // Pronunciation of one lowercase token without homograph handling:
    // dictionary, else the OOV path (morphology, compounds, letter-to-sound).
    int wordPhones(const char *w, int wl, uint8_t *out) const {
        int np = lookup(w, wl, out);
        return np >= 0 ? np : oov(w, wl, out);
    }

    // Debug: POS tag (class index, -1 if not tagged) of token t of the last
    // textToIds() call; copies the token text to buf.
    int debugToken(int t, char *buf, int bufLen) {
        if (t < 0 || t >= ntok) return -2;
        int l = tokLen[t] < bufLen - 1 ? tokLen[t] : bufLen - 1;
        memcpy(buf, bufA + tokStart[t], l);
        buf[l] = 0;
        if (!tagged) { tagAll(); tagged = true; }
        return tokTag[t] - 1;
    }

private:
    const uint8_t *d;
    long dsize;
    uint32_t nwords, nblocks, blockSize, nhomo;
    const uint8_t *index, *firstWords, *bits, *hPre, *hLet, *hPh, *homo, *lts;

    char bufA[WORK], bufB[WORK];
    uint16_t tokStart[MAXTOK];
    uint8_t tokLen[MAXTOK];
    uint8_t tokAlpha[MAXTOK];
    uint8_t tokHomo[MAXTOK];
    uint8_t tokTag[MAXTOK];
    bool tagged;
    int ncls;
    uint32_t ntd, nfeat;
    const uint8_t *posMask, *tdOff, *tdBody, *fOff, *fBody;
    int ntok;

    static char kChar(int s) { return "'-abcdefghijklmnopqrstuvwxyz"[s]; }

    // ------------------------------------------------------------ helpers
    static uint32_t be32(const uint8_t *p) {
        return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    }
    static uint32_t be16(const uint8_t *p) { return (uint32_t)p[0] << 8 | p[1]; }
    static bool isDigit(int c) { return c >= '0' && c <= '9'; }
    static bool isLower(int c) { return c >= 'a' && c <= 'z'; }
    static int charCode(char c) {
        if (c == '\'') return 1;
        if (c == '-') return 2;
        return c - 'a' + 3;
    }
    // compare NUL-terminated a with w[0..wl)
    static int cmpWord(const char *a, const char *w, int wl) {
        for (int i = 0; i < wl; i++) {
            uint8_t x = (uint8_t)a[i], y = (uint8_t)w[i];
            if (x != y) return x < y ? -1 : 1;  // also covers a ending (x == 0)
        }
        return a[wl] ? 1 : 0;
    }
    static int cmpN(const char *a, int al, const char *b, int bl) {
        int n = al < bl ? al : bl;
        for (int i = 0; i < n; i++)
            if (a[i] != b[i]) return (uint8_t)a[i] < (uint8_t)b[i] ? -1 : 1;
        return al < bl ? -1 : (al > bl ? 1 : 0);
    }

    struct BitReader {
        const uint8_t *p;
        uint32_t pos;
        int bit() {
            G2P_COUNT(bits, 1);
            int b = (p[pos >> 3] >> (7 - (pos & 7))) & 1;
            pos++;
            return b;
        }
    };
    const uint8_t *table(const uint8_t *set, int ctx) const { return set + be16(set + 2 * ctx); }
    static int decode(BitReader &br, const uint8_t *t) {
        int maxlen = t[0];
        G2P_COUNT(syms, 1);
        if (!maxlen) return t[1];
        const uint8_t *cnt = t + 1, *sym = t + 1 + maxlen;
        int code = 0, first = 0, idx = 0;
        for (int len = 1; len <= maxlen; len++) {
            code |= br.bit();
            int c = cnt[len - 1];
            if (code - c < first) return sym[idx + (code - first)];
            idx += c;
            first = (first + c) << 1;
            code <<= 1;
        }
        return 0;  // corrupt data
    }

    static bool isVowelPh(int p) { return p >= PH_FIRST_VOWEL; }
    static int stressOf(int p) { return (p - PH_FIRST_VOWEL) % 3; }
    static int withStress(int p, int s) { return p - stressOf(p) + s; }

    // exactly one primary stress (CMUdict convention)
    static void fixStress(uint8_t *ph, int n) {
        int firstV = -1, firstSec = -1, nprim = 0;
        for (int i = 0; i < n; i++) {
            if (!isVowelPh(ph[i])) continue;
            if (firstV < 0) firstV = i;
            int s = stressOf(ph[i]);
            if (s == 1) {
                if (nprim++) ph[i] = (uint8_t)withStress(ph[i], 2);
            } else if (s == 2 && firstSec < 0) firstSec = i;
        }
        if (!nprim && firstV >= 0) {
            int k = firstSec >= 0 ? firstSec : firstV;
            ph[k] = (uint8_t)withStress(ph[k], 1);
        }
    }

    // ------------------------------------------------- string building
    struct Out {
        char *p;
        int n, cap;
        bool ovf;
        void put(char c) {
            if (n < cap - 1) p[n++] = c;
            else ovf = true;
        }
        void puts(const char *s) { while (*s) put(*s++); }
        void put(const char *s, int l) { for (int i = 0; i < l; i++) put(s[i]); }
    };

    // --------------------------------------------- inflect number words
    static const char *unitW(int u) {
        static const char *t[10] = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
        return t[u];
    }
    static const char *teenW(int u) {
        static const char *t[10] = { "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                                     "sixteen", "seventeen", "eighteen", "nineteen" };
        return t[u];
    }
    static const char *tenW(int t) {
        static const char *x[10] = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                     "eighty", "ninety" };
        return x[t];
    }
    static const char *millW(int k) {
        static const char *m[12] = { "", " thousand", " million", " billion", " trillion", " quadrillion",
                                     " quintillion", " sextillion", " septillion", " octillion",
                                     " nonillion", " decillion" };
        return k < 12 ? m[k] : m[11];
    }
    static void tenfn(Out &o, int t, int u) {
        if (t == 1) { o.puts(teenW(u)); return; }
        o.puts(tenW(t));
        if (t && u) o.put('-');
        o.puts(unitW(u));
    }

    // inflect number_to_words(digits, andword=and_/"" , group=0); ordinal
    // when `ord`.  Appends to o.
    static void numberToWords(Out &o, const char *dg, int n, bool useAnd, bool ord) {
        while (n > 0 && *dg == '0') { dg++; n--; }
        int start = o.n;
        if (n == 0) o.puts("zero");
        else if (n == 1 && dg[0] == '1') o.puts("one");
        else {
            int ngroups = (n + 2) / 3;
            int lastSep = -1;      // position of last ", " separator
            bool lowNonZero = false, lowSingle = false;
            bool any = false;
            for (int g = ngroups - 1; g >= 0; g--) {
                // digits of group g (from the right)
                int end = n - 3 * g, beg = end - 3;
                int h = 0, t = 0, u = 0;
                if (beg >= 0) h = dg[beg] - '0';
                if (end - 2 >= 0) t = dg[end - 2] - '0';
                u = dg[end - 1] - '0';
                if (!h && !t && !u) continue;
                if (any) { lastSep = o.n; o.puts(", "); }
                any = true;
                if (h) {
                    o.puts(unitW(h));
                    o.puts(" hundred");
                    if (t || u) {
                        if (useAnd) o.puts(" and");
                        o.put(' ');
                        tenfn(o, t, u);
                    }
                } else {
                    tenfn(o, t, u);
                }
                o.puts(millW(g));
                if (g == 0) {
                    lowNonZero = true;
                    lowSingle = !h;
                }
            }
            // COMMA_WORD: ", word" at the end -> " and word"
            if (lastSep >= 0 && lowNonZero && lowSingle && !o.ovf) {
                char tail[32];
                int tl = o.n - (lastSep + 2);
                if (tl > 0 && tl < 32) {
                    memcpy(tail, o.p + lastSep + 2, tl);
                    o.n = lastSep;
                    o.puts(useAnd ? " and " : " ");
                    o.put(tail, tl);
                }
            }
        }
        if (ord && !o.ovf) {
            // inflect _sub_ord on the final word
            static const char *suf[8] = { "ty", "one", "two", "three", "five", "eight", "nine", "twelve" };
            static const char *rep[8] = { "tieth", "first", "second", "third", "fifth", "eighth", "ninth", "twelfth" };
            int len = o.n - start;
            for (int k = 0; k < 8; k++) {
                int sl = (int)strlen(suf[k]);
                if (len >= sl && memcmp(o.p + o.n - sl, suf[k], sl) == 0) {
                    o.n -= sl;
                    o.puts(rep[k]);
                    return;
                }
            }
            o.puts("th");
        }
    }

    // g2p_en expand._expand_number
    static void expandNumber(Out &o, const char *dg, int n) {
        while (n > 1 && *dg == '0') { dg++; n--; }
        int v = -1;
        if (n <= 4) {
            v = 0;
            for (int i = 0; i < n; i++) v = v * 10 + dg[i] - '0';
        }
        if (v > 1000 && v < 3000) {
            if (v == 2000) { o.puts("two thousand"); return; }
            if (v > 2000 && v < 2010) {
                o.puts("two thousand ");
                o.puts(v % 10 ? unitW(v % 10) : "zero");
                return;
            }
            int hi = v / 100, lo = v % 100;
            tenfn(o, hi / 10, hi % 10);
            if (lo == 0) { o.puts(" hundred"); return; }
            o.put(' ');
            if (lo < 10) { o.puts("oh "); o.puts(unitW(lo)); }
            else tenfn(o, lo / 10, lo % 10);
            return;
        }
        numberToWords(o, dg, n, false, false);
    }

    static void putInt(Out &o, const char *dg, int n) {  // str(int(dg))
        while (n > 1 && *dg == '0') { dg++; n--; }
        o.put(dg, n);
    }
    static bool isOne(const char *dg, int n) {
        while (n > 1 && *dg == '0') { dg++; n--; }
        return n == 1 && dg[0] == '1';
    }
    static bool isZero(const char *dg, int n) {
        for (int i = 0; i < n; i++) if (dg[i] != '0') return false;
        return true;
    }

    // g2p_en normalize_numbers: six regex passes.  in -> out
    static int passCommas(const char *s, int n, Out &o) {
        int i = 0;
        while (i < n) {
            if (isDigit(s[i])) {
                int j = i + 1;
                while (j < n && (isDigit(s[j]) || s[j] == ',')) j++;
                int end = j;
                while (end > i && !isDigit(s[end - 1])) end--;
                if (end - 1 >= i + 2) {
                    for (int k = i; k < end; k++) if (s[k] != ',') o.put(s[k]);
                    i = end;
                    continue;
                }
            }
            o.put(s[i++]);
        }
        return o.n;
    }
    // £/$ amount: [0-9,]*[0-9]+ (dollars also '.'); returns end or -1
    static int amountEnd(const char *s, int n, int i, bool dot) {
        int j = i;
        while (j < n && (isDigit(s[j]) || s[j] == ',' || (dot && s[j] == '.'))) j++;
        while (j > i && !isDigit(s[j - 1])) j--;
        return j > i ? j : -1;
    }
    static void passPounds(const char *s, int n, Out &o) {
        int i = 0;
        while (i < n) {
            if ((uint8_t)s[i] == 0xA3) {
                int e = amountEnd(s, n, i + 1, false);
                if (e > 0) {
                    o.put(s + i + 1, e - i - 1);
                    o.puts(" pounds");
                    i = e;
                    continue;
                }
            }
            o.put(s[i++]);
        }
    }
    static void passDollars(const char *s, int n, Out &o) {
        int i = 0;
        while (i < n) {
            if (s[i] == '$') {
                int e = amountEnd(s, n, i + 1, true);
                if (e > 0) {
                    const char *g = s + i + 1;
                    int gl = e - i - 1;
                    int dots = 0, d1 = -1;
                    for (int k = 0; k < gl; k++) if (g[k] == '.') { if (!dots) d1 = k; dots++; }
                    if (dots > 1) {
                        o.put(g, gl);
                        o.puts(" dollars");
                    } else {
                        // strip commas (Python would raise on int("1,0"))
                        char a[64], b[64];
                        int al = 0, bl = 0;
                        int dl = dots ? d1 : gl;
                        for (int k = 0; k < dl && al < 63; k++) if (g[k] != ',') a[al++] = g[k];
                        if (dots)
                            for (int k = d1 + 1; k < gl && bl < 63; k++) if (g[k] != ',') b[bl++] = g[k];
                        bool hasD = al && !isZero(a, al), hasC = bl && !isZero(b, bl);
                        if (hasD) {
                            putInt(o, a, al);
                            o.puts(isOne(a, al) ? " dollar" : " dollars");
                            if (hasC) o.puts(", ");
                        }
                        if (hasC) {
                            putInt(o, b, bl);
                            o.puts(isOne(b, bl) ? " cent" : " cents");
                        }
                        if (!hasD && !hasC) o.puts("zero dollars");
                    }
                    i = e;
                    continue;
                }
            }
            o.put(s[i++]);
        }
    }
    static void passDecimal(const char *s, int n, Out &o) {
        int i = 0;
        while (i < n) {
            if (isDigit(s[i])) {
                int j = i;
                while (j < n && isDigit(s[j])) j++;
                if (j + 1 < n && s[j] == '.' && isDigit(s[j + 1])) {
                    int k = j + 1;
                    while (k < n && isDigit(s[k])) k++;
                    o.put(s + i, j - i);
                    o.puts(" point ");
                    o.put(s + j + 1, k - j - 1);
                    i = k;
                } else {
                    o.put(s + i, j - i);
                    i = j;
                }
                continue;
            }
            o.put(s[i++]);
        }
    }
    static void passOrdinal(const char *s, int n, Out &o) {
        int i = 0;
        while (i < n) {
            if (isDigit(s[i])) {
                int j = i;
                while (j < n && isDigit(s[j])) j++;
                if (j + 1 < n && ((s[j] == 's' && s[j + 1] == 't') || (s[j] == 'n' && s[j + 1] == 'd') ||
                                  (s[j] == 'r' && s[j + 1] == 'd') || (s[j] == 't' && s[j + 1] == 'h'))) {
                    numberToWords(o, s + i, j - i, true, true);
                    i = j + 2;
                } else {
                    o.put(s + i, j - i);
                    i = j;
                }
                continue;
            }
            o.put(s[i++]);
        }
    }
    static void passNumber(const char *s, int n, Out &o) {
        int i = 0;
        while (i < n) {
            if (isDigit(s[i])) {
                int j = i;
                while (j < n && isDigit(s[j])) j++;
                expandNumber(o, s + i, j - i);
                i = j;
                continue;
            }
            o.put(s[i++]);
        }
    }

    // Latin-1 -> what NFD + strip Mn + lower() + [^ a-z'.,?!-] removal leaves
    static char foldChar(uint8_t c) {
        if (c >= 'A' && c <= 'Z') return (char)(c + 32);
        if ((c >= 'a' && c <= 'z') || c == ' ' || c == '\'' || c == '.' || c == ',' || c == '?' ||
            c == '!' || c == '-')
            return (char)c;
        if (c < 0xC0) return 0;
        static const char hi[64 + 1] =
            "aaaaaa\0ceeeeiiii\0nooooo\0\0uuuuy\0\0"   // C0-DF
            "aaaaaa\0ceeeeiiii\0nooooo\0\0uuuuy\0y";   // E0-FF
        return hi[c - 0xC0];
    }

    // Full g2p_en preprocessing; result in bufA, returns length or -1.
    int normalize(const char *text) {
        Out o;
        int n = (int)strlen(text);
        if (n >= WORK) return -1;
        // pass 1: commas (text -> B)
        o.p = bufB; o.n = 0; o.cap = WORK; o.ovf = false;
        passCommas(text, n, o);
        if (o.ovf) return -1;
        n = o.n;
        o.p = bufA; o.n = 0; o.ovf = false;
        passPounds(bufB, n, o);
        if (o.ovf) return -1;
        n = o.n;
        o.p = bufB; o.n = 0;
        passDollars(bufA, n, o);
        if (o.ovf) return -1;
        n = o.n;
        o.p = bufA; o.n = 0;
        passDecimal(bufB, n, o);
        if (o.ovf) return -1;
        n = o.n;
        o.p = bufB; o.n = 0;
        passOrdinal(bufA, n, o);
        if (o.ovf) return -1;
        n = o.n;
        o.p = bufA; o.n = 0;
        passNumber(bufB, n, o);
        if (o.ovf) return -1;
        n = o.n;
        // accents, lowercase, character filter (A -> B)
        int m = 0;
        for (int i = 0; i < n; i++) {
            char c = foldChar((uint8_t)bufA[i]);
            if (c) bufB[m++] = c;
        }
        // "i.e." -> "that is", "e.g." -> "for example" (B -> A)
        o.p = bufA; o.n = 0;
        for (int i = 0; i < m;) {
            if (i + 4 <= m && !memcmp(bufB + i, "i.e.", 4)) { o.puts("that is"); i += 4; }
            else o.put(bufB[i++]);
        }
        if (o.ovf) return -1;
        m = o.n;
        o.p = bufB; o.n = 0;
        for (int i = 0; i < m;) {
            if (i + 4 <= m && !memcmp(bufA + i, "e.g.", 4)) { o.puts("for example"); i += 4; }
            else o.put(bufA[i++]);
        }
        if (o.ovf) return -1;
        m = o.n;
        // HANG_RE: runs of >= 4 equal non-alphanumerics -> 3 (B -> A)
        int k = 0;
        for (int i = 0; i < m;) {
            char c = bufB[i];
            int j = i + 1;
            while (j < m && bufB[j] == c) j++;
            int run = j - i;
            if (!isLower(c) && run > 3) run = 3;
            for (int r = 0; r < run; r++) bufA[k++] = c;
            i = j;
        }
        bufA[k] = 0;
        return k;
    }

    // ------------------------------------------------- TweetTokenizer
    static bool isWordCh(char c) { return isLower(c); }  // \w after filtering

    // naked-domain URL alternative; returns end or -1
    static int urlEnd(const char *s, int n, int p) {
        int i = p;
        while (i < n && isWordCh(s[i])) i++;
        int best = -1;
        // segments separated by single '.' or '-'
        while (i + 1 < n && (s[i] == '.' || s[i] == '-') && isWordCh(s[i + 1])) {
            char sep = s[i];
            int b = i + 1, e = b;
            while (e < n && isWordCh(s[e])) e++;
            if (sep == '.' && e - b >= 2 && e - b <= 13) best = e;
            i = e;
        }
        return best;
    }

    int tokenize(const char *s, int n) {
        ntok = 0;
        int p = 0;
        while (p < n) {
            char c = s[p];
            if (c == ' ') { p++; continue; }
            int e = -1;
            bool alpha = false;
            if (isLower(c)) {
                alpha = true;
                e = urlEnd(s, n, p);
                if (e < 0) {
                    // word with apostrophes or dashes
                    int r = p + 1;
                    while (r < n && (isLower(s[r]) || s[r] == '\'' || s[r] == '-')) r++;
                    int q = r - 1;
                    while (q >= p + 2 && !isLower(s[q])) q--;
                    if (q >= p + 2) e = q + 1;
                }
                if (e < 0) {
                    e = p + 1;
                    while (e < n && isLower(s[e])) e++;
                }
            } else if (c == '.') {
                int j = p + 1, last = -1;
                for (;;) {
                    int k = j;
                    while (k < n && s[k] == ' ') k++;
                    if (k < n && s[k] == '.') { j = k + 1; last = j; }
                    else break;
                }
                e = last > 0 ? last : p + 1;
            } else {
                e = p + 1;
            }
            if (ntok >= MAXTOK || e - p > 255) return -1;
            tokStart[ntok] = (uint16_t)p;
            tokLen[ntok] = (uint8_t)(e - p);
            tokAlpha[ntok] = alpha;
            tokHomo[ntok] = alpha && e - p < MAXWORD && findHomograph(s + p, e - p) != 0;
            ntok++;
            p = e;
        }
        return ntok;
    }

    // ----------------------------------------------------- homographs
    const uint8_t *findHomograph(const char *w, int wl) const {
        long lo = 0, hi = (long)nhomo - 1;
        while (lo <= hi) {
            long mid = (lo + hi) >> 1;
            const uint8_t *e = homo + be16(homo + 2 + 2 * mid);
            int c = cmpWord((const char *)e, w, wl);
            if (c == 0) return e + wl + 1;
            if (c < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return 0;
    }

    // ------------------------------------------- pruned nltk POS tagger
    // Tags are kept as "tag bytes": class index + 1; ncls+1 = -START-,
    // ncls+2 = -START2-.
    enum { F_BIAS = 1, F_SUF, F_PREF1, F_TAG1, F_TAG2, F_TAG12, F_WORD, F_TAG1WORD, F_WM1, F_SUFM1,
           F_WM2, F_WP1, F_SUFP1, F_WP2 };

    int tagdictLookup(const char *w, int wl) const {
        long lo = 0, hi = (long)ntd - 1;
        while (lo <= hi) {
            long mid = (lo + hi) >> 1;
            const uint8_t *e = tdBody + be16(tdOff + 2 * mid);
            int c = cmpWord((const char *)e, w, wl);
            if (c == 0) return e[wl + 1];
            if (c < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    // adds the weights of feature `key` to scores
    void addFeature(const uint8_t *key, int kl, int32_t *scores) const {
        G2P_COUNT(feats, 1);
        long lo = 0, hi = (long)nfeat - 1;
        while (lo <= hi) {
            long mid = (lo + hi) >> 1;
            const uint8_t *e = fBody + 2 * (uint32_t)be16(fOff + 2 * mid);
            int el = e[0];
            int c = cmpN((const char *)e + 1, el, (const char *)key, kl);
            if (c == 0) {
                int np = e[1 + el];
                const uint8_t *p = e + 2 + el;
                if ((2 + el) & 1) p++;
                for (int i = 0; i < np; i++, p += 2) {
                    int v = (int)be16(p);
                    int w = v & 0x3FF;
                    if (w & 0x200) w -= 0x400;
                    scores[v >> 10] += w;
                }
                return;
            }
            if (c < 0) lo = mid + 1;
            else hi = mid - 1;
        }
    }

    // normalised context word of token j (nltk PerceptronTagger.normalize)
    int ctxWord(int j, const char *&s) const {
        if (j == -2) { s = "-START-"; return 7; }
        if (j == -1) { s = "-START2-"; return 8; }
        if (j == ntok) { s = "-END-"; return 5; }
        if (j == ntok + 1) { s = "-END2-"; return 6; }
        s = bufA + tokStart[j];
        int l = tokLen[j];
        if (s[0] != '-' && memchr(s, '-', l)) { s = "!HYPHEN"; return 7; }
        return l;
    }

    int keyWord(uint8_t *k, int type, int j) const {
        const char *s;
        int l = ctxWord(j, s);
        k[0] = (uint8_t)type;
        memcpy(k + 1, s, l);
        return 1 + l;
    }
    int keySuffix(uint8_t *k, int type, int j) const {
        const char *s;
        int l = ctxWord(j, s);
        int sl = l < 3 ? l : 3;
        k[0] = (uint8_t)type;
        memcpy(k + 1, s + l - sl, sl);
        return 1 + sl;
    }

    void tagAll() {
        int32_t sc[64];
        uint8_t key[MAXWORD + 8];
        int prev = ncls + 1, prev2 = ncls + 2;
        for (int t = 0; t < ntok; t++) {
            const char *w = bufA + tokStart[t];
            int wl = tokLen[t];
            int tag = tagdictLookup(w, wl);
            if (tag >= 0) {
                tag += 1;
            } else {
                for (int c = 0; c < ncls; c++) sc[c] = 0;
                key[0] = F_BIAS;
                addFeature(key, 1, sc);
                int sl = wl < 3 ? wl : 3;
                key[0] = F_SUF;
                memcpy(key + 1, w + wl - sl, sl);
                addFeature(key, 1 + sl, sc);
                key[0] = F_PREF1;
                key[1] = (uint8_t)w[0];
                addFeature(key, 2, sc);
                key[0] = F_TAG1; key[1] = (uint8_t)prev;
                addFeature(key, 2, sc);
                key[0] = F_TAG2; key[1] = (uint8_t)prev2;
                addFeature(key, 2, sc);
                key[0] = F_TAG12; key[1] = (uint8_t)prev; key[2] = (uint8_t)prev2;
                addFeature(key, 3, sc);
                if (tokHomo[t] && wl < MAXWORD) {
                    int kl = keyWord(key, F_WORD, t);
                    addFeature(key, kl, sc);
                    key[0] = F_TAG1WORD; key[1] = (uint8_t)prev;
                    memcpy(key + 2, w, wl);
                    addFeature(key, 2 + wl, sc);
                    if (tokLen[t > 0 ? t - 1 : 0] < MAXWORD && (t + 1 >= ntok || tokLen[t + 1] < MAXWORD) &&
                        (t < 2 || tokLen[t - 2] < MAXWORD) && (t + 2 >= ntok || tokLen[t + 2] < MAXWORD)) {
                        kl = keyWord(key, F_WM1, t - 1);  addFeature(key, kl, sc);
                        kl = keySuffix(key, F_SUFM1, t - 1); addFeature(key, kl, sc);
                        kl = keyWord(key, F_WM2, t - 2);  addFeature(key, kl, sc);
                        kl = keyWord(key, F_WP1, t + 1);  addFeature(key, kl, sc);
                        kl = keySuffix(key, F_SUFP1, t + 1); addFeature(key, kl, sc);
                        kl = keyWord(key, F_WP2, t + 2);  addFeature(key, kl, sc);
                    }
                }
                // argmax; ties go to the alphabetically larger class
                int best = 0;
                for (int c = 1; c < ncls; c++)
                    if (sc[c] >= sc[best]) best = c;
                tag = best + 1;
            }
            tokTag[t] = (uint8_t)tag;
            prev2 = prev;
            prev = tag;
        }
    }

    // nltk tag of homograph t startswith(pos1)?  pos: 0 V,1 N,2 ADJ,3 VBD,4 VBN
    bool posMatches(int t, int pos) {
        if (!tagged) { tagAll(); tagged = true; }
        int tag = tokTag[t];
        return tag >= 1 && tag <= ncls && (posMask[tag - 1] >> pos & 1);
    }

    // ------------------------------------------------------ word -> phones
    static int addSuffixS(uint8_t *ph, int np, int maxOut) {
        // plural/possessive -s after phones
        int last = np ? ph[np - 1] : -1;
        if (np + 2 > maxOut) return np;
        if (last == PH_S || last == PH_Z || last == PH_SH || last == PH_ZH || last == PH_CH || last == PH_JH) {
            ph[np++] = PH_IH;  // IH0
            ph[np++] = PH_Z;
        } else if (last == PH_P || last == PH_T || last == PH_K || last == PH_F || last == PH_TH) {
            ph[np++] = PH_S;
        } else {
            ph[np++] = PH_Z;
        }
        return np;
    }

    // dictionary lookup of w[0..bl) + optional extra letter (for "e"/"y")
    int lookupBase(const char *w, int bl, char extra, uint8_t *out) const {
        if (bl < 2 || bl + 1 >= MAXWORD) return -1;
        if (!extra) return lookup(w, bl, out);
        char tmp[MAXWORD];
        memcpy(tmp, w, bl);
        tmp[bl] = extra;
        return lookup(tmp, bl + 1, out);
    }
    static bool endsWith(const char *w, int wl, const char *suf) {
        int sl = (int)strlen(suf);
        return wl > sl && !memcmp(w + wl - sl, suf, sl);
    }
    static bool isVowelCh(char c) { return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'; }
    // base of an inflected form: stem, stem+"e", or undoubled stem
    int stemLookup(const char *w, int bl, uint8_t *out) const {
        if (bl < 3) return -1;
        int np = lookupBase(w, bl, 0, out);
        if (np >= 0) return np;
        if (!isVowelCh(w[bl - 1])) {
            np = lookupBase(w, bl, 'e', out);
            if (np >= 0) return np;
        }
        if (bl >= 4 && w[bl - 1] == w[bl - 2] && !isVowelCh(w[bl - 1]))
            return lookupBase(w, bl - 1, 0, out);
        return -1;
    }
    static int appendPh(uint8_t *out, int np, int maxOut, const uint8_t *ph, int n) {
        for (int i = 0; i < n && np < maxOut; i++) out[np++] = ph[i];
        return np;
    }
    static int addSuffixEd(uint8_t *ph, int np, int maxOut) {
        int last = np ? ph[np - 1] : -1;
        if (np + 2 > maxOut) return np;
        if (last == PH_T || last == PH_D) { ph[np++] = PH_IH; ph[np++] = PH_D; }
        else if (last == PH_P || last == PH_K || last == PH_F || last == PH_S || last == PH_SH ||
                 last == PH_CH || last == PH_TH) ph[np++] = PH_T;
        else ph[np++] = PH_D;
        return np;
    }

    // Inflected/derived forms of dictionary words. Returns #phones or -1.
    int morph(const char *w, int wl, uint8_t *out, int maxOut) const {
        int np;
        if (wl < 5) return -1;
        if (endsWith(w, wl, "ies")) {
            np = lookupBase(w, wl - 3, 'y', out);
            if (np >= 0) { if (np < maxOut) out[np++] = PH_Z; return np; }
        }
        if (endsWith(w, wl, "es") && (w[wl - 3] == 's' || w[wl - 3] == 'x' || w[wl - 3] == 'z' || w[wl - 3] == 'h')) {
            np = lookupBase(w, wl - 2, 0, out);
            if (np >= 0) return addSuffixS(out, np, maxOut);
        }
        if (w[wl - 1] == 's' && w[wl - 2] != 's') {
            np = lookupBase(w, wl - 1, 0, out);
            if (np < 0) np = morph(w, wl - 1, out, maxOut);  // e.g. "-ers", "-ings"
            if (np >= 0) return addSuffixS(out, np, maxOut);
        }
        if (endsWith(w, wl, "ied")) {
            np = lookupBase(w, wl - 3, 'y', out);
            if (np >= 0) { if (np < maxOut) out[np++] = PH_D; return np; }
        }
        if (endsWith(w, wl, "ed")) {
            np = stemLookup(w, wl - 2, out);
            if (np >= 0) return addSuffixEd(out, np, maxOut);
        }
        if (endsWith(w, wl, "ing")) {
            np = stemLookup(w, wl - 3, out);
            if (np >= 0 && np + 2 <= maxOut) { out[np++] = PH_IH; out[np++] = PH_NG; return np; }
        }
        if (endsWith(w, wl, "er")) {
            np = stemLookup(w, wl - 2, out);
            if (np >= 0 && np < maxOut) { out[np++] = PH_ER; return np; }
        }
        if (endsWith(w, wl, "ily")) {
            np = lookupBase(w, wl - 3, 'y', out);
            if (np > 0 && out[np - 1] == PH_IY && np + 3 <= maxOut) {
                np--;
                static const uint8_t s[3] = { PH_AH, PH_L, PH_IY };
                return appendPh(out, np, maxOut, s, 3);
            }
        }
        if (endsWith(w, wl, "ably") || endsWith(w, wl, "able")) {
            np = stemLookup(w, wl - 4, out);
            if (np >= 0 && np + 4 <= maxOut) {
                out[np++] = PH_AH; out[np++] = PH_B;
                if (w[wl - 1] == 'y') { out[np++] = PH_L; out[np++] = PH_IY; }
                else { out[np++] = PH_AH; out[np++] = PH_L; }
                return np;
            }
        }
        if (endsWith(w, wl, "ically")) {
            np = lookupBase(w, wl - 4, 0, out);  // "dynamic" + "ally"
            if (np >= 0 && np + 2 <= maxOut) { out[np++] = PH_L; out[np++] = PH_IY; return np; }
        }
        if (endsWith(w, wl, "ier")) {
            np = lookupBase(w, wl - 3, 'y', out);
            if (np >= 0 && np < maxOut) { out[np++] = PH_ER; return np; }
        }
        static const char *const suf[] = { "ly", "ness", "less", "ment", "ful", "ism", "ist", "ish", 0 };
        static const uint8_t sufPh[][4] = { { 2, PH_L, PH_IY }, { 3, PH_N, PH_AH, PH_S }, { 3, PH_L, PH_AH, PH_S },
                                            { 3, PH_M, PH_AH, PH_N }, { 3, PH_F, PH_AH, PH_L },
                                            { 3, PH_IH + 2, PH_Z, PH_AH }, { 3, PH_IH, PH_S, PH_T },
                                            { 2, PH_IH, PH_SH } };
        for (int k = 0; suf[k]; k++) {
            if (!endsWith(w, wl, suf[k])) continue;
            int sl = (int)strlen(suf[k]);
            np = lookupBase(w, wl - sl, 0, out);
            if (np < 0 && k >= 5) np = stemLookup(w, wl - sl, out);
            if (np >= 0) {
                np = appendPh(out, np, maxOut, sufPh[k] + 1, sufPh[k][0]);
                if (k == 3 && np < maxOut) out[np++] = PH_T;       // ment
                if (k == 5 && np < maxOut) out[np++] = PH_M;       // ism
                return np;
            }
        }
        return -1;
    }

    // plural of a dictionary word
    int pluralLookup(const char *w, int wl, uint8_t *out, int maxOut) const {
        if (wl < 4 || w[wl - 1] != 's' || w[wl - 2] == 's') return -1;
        int np = lookup(w, wl - 1, out);
        if (np < 0 && w[wl - 2] == 'e') {
            np = lookup(w, wl - 2, out);
            if (np >= 0 && np + 2 <= maxOut) { out[np++] = PH_IH; out[np++] = PH_Z; return np; }
            return -1;
        }
        return np >= 0 ? addSuffixS(out, np, maxOut) : -1;
    }

    // Compound of two dictionary words (the second may be a plural).  Parts
    // have >= 4 letters (a 3-letter first part needs a >= 5-letter second).
    // The second part's primary stress becomes secondary.
    int compound(const char *w, int wl, uint8_t *out, int maxOut) const {
        int bestK = -1, bestScore = -100;
        uint8_t tmp[MAXPH];
        for (int k = 3; k <= wl - 4; k++) {
            int l2 = wl - k;
            if (k == 3 && l2 < 5) continue;
            int score = 2 * (k < l2 ? k : l2) + (l2 >= k ? 1 : 0) - (w[k - 1] == 's' ? 4 : 0);
            if (score <= bestScore) continue;
            if (lookup(w, k, tmp) < 0) continue;
            if (lookup(w + k, l2, tmp) < 0 && pluralLookup(w + k, l2, tmp, MAXPH) < 0) continue;
            bestK = k;
            bestScore = score;
        }
        if (bestK < 0) return -1;
        int n1 = lookup(w, bestK, out);
        int n2 = lookup(w + bestK, wl - bestK, out + n1);
        if (n2 < 0) n2 = pluralLookup(w + bestK, wl - bestK, out + n1, maxOut - n1);
        for (int q = n1; q < n1 + n2; q++)
            if (isVowelPh(out[q]) && stressOf(out[q]) == 1) out[q] = (uint8_t)withStress(out[q], 2);
        return n1 + n2;
    }

    // one hyphen/dot-free part of an OOV token
    int oovPart(const char *w, int wl, uint8_t *out, int maxOut) const {
        int np = lookup(w, wl, out);
        if (np >= 0) return np;
        // possessive
        if (wl > 2 && w[wl - 2] == '\'' && w[wl - 1] == 's') {
            np = lookup(w, wl - 2, out);
            if (np < 0) np = morph(w, wl - 2, out, maxOut);
            if (np >= 0) return addSuffixS(out, np, maxOut);
        }
        char tmp[MAXWORD];
        int tl = 0;
        for (int i = 0; i < wl && tl < MAXWORD; i++)
            if (isLower(w[i])) tmp[tl++] = w[i];
        if (!tl) return 0;
        if (tl == wl || (wl > 2 && w[wl - 2] == '\'')) {
            np = morph(tmp, tl, out, maxOut);
            if (np >= 0) return np;
            np = compound(tmp, tl, out, maxOut);
            if (np >= 0) return np;
        }
        return letterToSound(tmp, tl, out, maxOut);
    }

    int oov(const char *w, int wl, uint8_t *out) const {
        int np = 0, i = 0;
        bool first = true;
        while (i < wl) {
            int j = i;
            while (j < wl && w[j] != '-' && w[j] != '.') j++;
            if (j > i) {
                int k = oovPart(w + i, j - i, out + np, MAXPH - np);
                if (!first) {  // compound: later parts lose primary stress
                    for (int q = np; q < np + k; q++)
                        if (isVowelPh(out[q]) && stressOf(out[q]) == 1) out[q] = (uint8_t)withStress(out[q], 2);
                }
                if (k > 0) first = false;
                np += k;
            }
            i = j + 1;
        }
        return np;
    }

    int wordToPhones(int t, uint8_t *out) {
        const char *w = bufA + tokStart[t];
        int wl = tokLen[t];
        statWords++;
        const uint8_t *h = findHomograph(w, wl);
        if (h) {
            statHomograph++;
            int pos = h[0];
            const uint8_t *p1 = h + 1, *p2 = p1 + 1 + p1[0];
            const uint8_t *p = posMatches(t, pos) ? p1 : p2;
            memcpy(out, p + 1, p[0]);
            return p[0];
        }
        int np = lookup(w, wl, out);
        if (np >= 0) { statDict++; return np; }
        statOov++;
        return oov(w, wl, out);
    }
};

}  // namespace AuroraFront

#endif
