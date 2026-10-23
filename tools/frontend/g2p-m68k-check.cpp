#include "../../src/frontend/g2p.h"
static AuroraFront::G2p g;
int conv(const uint8_t *d, long n, const char *t, uint8_t *ids, int m) { if (!g.load(d, n)) return -9; return g.textToIds(t, ids, m); }
