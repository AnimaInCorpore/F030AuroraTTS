// AURORA.TTP: AuroraTTS-100K text-to-speech on the Atari Falcon030.
//
//   AURORA.TTP [options] text ...
//     -p        the text is ARPAbet phonemes ("HH AH0 L OW1 _ W ER1 L D .")
//     -w FILE   also write the speech to a WAV file (24,585 Hz, mono)
//     -q        do not play (with -w: render only)
//     -f FILE   read the text from a file
// Without text it reads lines from the keyboard until an empty line.
//
// The network and the synthesizer run on the DSP (dsp/aurora.asm), the
// length regulator and the text frontend on the 68030. Each sentence is
// rendered completely, then played by DMA at 24,585 Hz while the next one
// renders. AURORA.BIN (the model) and DICT.BIN (the pronunciation
// dictionary) are read from the program's directory.
#include <mint/falcon.h>
#include <mint/osbind.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver.h"
#include "falcon-coop.h"
#include <signal.h>
#include "symbols.h"
#include "synth-driver.h"

#if __has_include("g2p.h")
#include "g2p.h"
#define AURORA_HAVE_FRONTEND 1
#endif

#ifndef AURORA_VERSION
#define AURORA_VERSION "dev"
#endif

using namespace Aurora;

namespace {

const long kRate = 24585;
const int kMaxIds = 400;

// Every host-port transaction is a short Supexec and the waits for the DSP
// happen in user mode (falcon-coop.h): FreeMiNT keeps running other
// processes while a sentence renders.
CoopDsp g_dsp;
Driver<CoopDsp> g_drv;
volatile sig_atomic_t g_stop;

void onSignal(int) {
	g_stop = 1;
}
int32_t *g_img;
bool g_dspHeld, g_soundHeld;

uint32_t ticks() {
	return (uint32_t)CoopDsp::ticks();
}

uint8_t *loadFile(const char *path, long *size) {
	FILE *f = fopen(path, "rb");
	if (!f)
		return 0;
	fseek(f, 0, SEEK_END);
	*size = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *p = (uint8_t *)malloc(*size + 1);
	if (p && fread(p, 1, *size, f) != (size_t)*size) {
		free(p);
		p = 0;
	}
	if (p)
		p[*size] = 0;
	fclose(f);
	return p;
}

// ---------------------------------------------------------------- audio

// A playback buffer in ST-RAM (DMA reads only ST-RAM): stereo 16-bit frames.
struct PlayBuffer {
	int16_t *frames;
	long capacity, count;
};

PlayBuffer g_play[2];
int g_playing = -1;

bool allocPlay(PlayBuffer &b, long frames) {
	if (b.capacity >= frames)
		return true;
	if (b.frames)
		Mfree(b.frames);
	b.frames = (int16_t *)Mxalloc(frames * 4 + 16, 0);
	b.capacity = b.frames ? frames : 0;
	return b.frames != 0;
}

bool soundOpen() {
	if (Locksnd() != 1)
		return false;
	g_soundHeld = true;
	Sndstatus(1);
	Setmode(1);                 // 16-bit stereo
	Settracks(0, 0);
	Setmontracks(0);
	Soundcmd(4, 2);             // ADDERIN: the DAC hears the matrix
	Devconnect(0, 8, 0, 3, 1);  // DMA playback to the DAC, 25.175 MHz / 1024: 24,585 Hz, no handshake
	return true;
}

bool playing() {
	return (Buffoper(-1) & 1) != 0;
}

void waitPlayback() {
	while (g_playing >= 0 && playing())
		Syield();
	g_playing = -1;
}

void play(int which) {
	PlayBuffer &b = g_play[which];
	Buffoper(0);
	Setbuffer(0, b.frames, b.frames + 2 * b.count);
	Buffoper(1);
	g_playing = which;
}

void soundClose() {
	if (g_soundHeld) {
		Buffoper(0);
		Unlocksnd();
	}
	g_soundHeld = false;
}

// ---------------------------------------------------------------- WAV output

FILE *g_wav;
long g_wavSamples;

void put16(FILE *f, int v) {
	fputc(v & 0xff, f);
	fputc((v >> 8) & 0xff, f);
}

void put32(FILE *f, long v) {
	put16(f, (int)(v & 0xffff));
	put16(f, (int)((v >> 16) & 0xffff));
}

void wavHeader(FILE *f, long samples) {
	fwrite("RIFF", 1, 4, f);
	put32(f, 36 + samples * 2);
	fwrite("WAVEfmt ", 1, 8, f);
	put32(f, 16);
	put16(f, 1);
	put16(f, 1);
	put32(f, kRate);
	put32(f, kRate * 2);
	put16(f, 2);
	put16(f, 16);
	fwrite("data", 1, 4, f);
	put32(f, samples * 2);
}

// ---------------------------------------------------------------- rendering

struct Timing {
	uint32_t encode, upsample, decode, synth;
};

int16_t *g_pcm;
long g_pcmCapacity;

// ids -> g_pcm; returns the sample count (0 on failure). User mode.
long render(const uint8_t *ids, int N, Timing *t) {
	static int32_t x[kMaxIds * kD], work[kMaxIds * kD], logd[kMaxIds], pit[kMaxIds], c2[kMaxIds];
	static uint8_t dur[kMaxIds];
	uint32_t t0 = ticks();
	g_drv.encode(ids, N, x, work, logd, pit, dur);
	uint32_t t1 = ticks();
	int T;
	centres(dur, N, c2, &T);
	if (T < 2)
		return 0;
	int32_t *head = (int32_t *)malloc((long)T * kHead * 4);
	int32_t *dw = (int32_t *)malloc((long)T * kD * 4);
	if (!head || !dw) {
		free(head);
		free(dw);
		return 0;
	}
	uint32_t t2 = ticks();   // the length regulator runs inside the decoder's first pass
	g_drv.decodeTokens(x, dur, c2, N, T, head, dw);
	free(dw);
	uint32_t t3 = ticks();
	const long want = synthOutputCount(g_img, T);
	if (want > g_pcmCapacity) {
		free(g_pcm);
		g_pcm = (int16_t *)malloc(want * 2 + 1024);
		g_pcmCapacity = g_pcm ? want : 0;
	}
	long got = g_pcm ? synthesize(&g_dsp, g_img, head, T, g_pcm) : 0;
	free(head);
	uint32_t t4 = ticks();
	t->encode = t1 - t0;
	t->upsample = t2 - t1;
	t->decode = t3 - t2;
	t->synth = t4 - t3;
	return got;
}

// ---------------------------------------------------------------- sentences

// Whether the word ending at out[end] (a '.') is an abbreviation or an
// initial, after which a sentence does not end.
bool abbreviation(const char *out, int end) {
	int s = end;
	while (s > 0 && out[s - 1] != ' ' && out[s - 1] != '(' && out[s - 1] != '"')
		--s;
	const int len = end - s;
	if (len == 1 && out[s] >= 'A' && out[s] <= 'Z')
		return true;   // an initial: "J. Smith"
	static const char *const kAbbr[] = {"Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "Prof", "Gen", "Col", "Capt", "Lt",
	                                    "Sgt", "Rev", "Hon", "Mt", "vs", "etc", "No", "Co", "Inc", "Ltd", "Jan", "Feb",
	                                    "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "e.g", "i.e"};
	for (unsigned i = 0; i < sizeof kAbbr / sizeof kAbbr[0]; ++i)
		if ((int)strlen(kAbbr[i]) == len && !strncmp(kAbbr[i], out + s, len))
			return true;
	return false;
}

// The next sentence of text: up to and including . ? ! or a newline, at most
// max - 1 characters (cut at a space when longer).
const char *nextSentence(const char *p, char *out, int max) {
	while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
		++p;
	int n = 0, lastSpace = -1;
	while (*p && n < max - 1) {
		const char c = *p++;
		if (c == '\n' || c == '\r') {
			if (n > 0)
				break;
			continue;
		}
		if (c == ' ')
			lastSpace = n;
		out[n++] = c;
		if ((c == '.' || c == '?' || c == '!') && (*p == ' ' || *p == '\n' || *p == '\r' || !*p)) {
			if (c == '.' && abbreviation(out, n - 1))
				continue;
			break;
		}
	}
	if (*p && n == max - 1 && lastSpace > 0) {
		p -= n - lastSpace;
		n = lastSpace;
	}
	out[n] = 0;
	return p;
}

#ifdef AURORA_HAVE_FRONTEND
AuroraFront::G2p g_g2p;
bool g_haveDict;
#endif

int toIds(const char *sentence, bool phonemes, uint8_t *ids) {
	if (phonemes) {
		const int n = phonemesToIds(sentence, ids, kMaxIds);
		if (n < 0)
			printf("unknown phoneme at \"%.10s\"\r\n", sentence - n - 1);
		return n;
	}
#ifdef AURORA_HAVE_FRONTEND
	if (g_haveDict)
		return g_g2p.textToIds(sentence, ids, kMaxIds);
#endif
	printf("no DICT.BIN: use -p and ARPAbet phonemes\r\n");
	return -1;
}

bool g_quiet, g_phonemes;
long g_totalSamples;
uint32_t g_totalTicks;

// Renders and plays (or writes) one sentence; false to stop.
bool speak(const char *sentence) {
	static uint8_t ids[kMaxIds];
	const int N = toIds(sentence, g_phonemes, ids);
	if (N <= 2)
		return N >= 0;
	printf("%s\r\n", sentence);
	Timing t;
	const long n = render(ids, N, &t);
	if (n <= 0 || g_dsp.failed()) {
		printf("rendering failed\r\n");
		return false;
	}
	const uint32_t total = t.encode + t.upsample + t.decode + t.synth;
	printf("  %d symbols, %ld.%02ld s of speech in %ld.%02ld s (network %lu, length %lu, decoder %lu, synth %lu ticks)\r\n",
	       N, n / kRate, (n % kRate) * 100 / kRate, (long)(total / 200), (long)(total % 200) / 2, (unsigned long)t.encode,
	       (unsigned long)t.upsample, (unsigned long)t.decode, (unsigned long)t.synth);
	g_totalSamples += n;
	g_totalTicks += total;
	if (g_wav) {
		for (long i = 0; i < n; ++i)
			put16(g_wav, g_pcm[i]);
		g_wavSamples += n;
	}
	if (!g_quiet && g_soundHeld) {
		// the other buffer: the one playing keeps playing until it ends
		const int which = g_playing == 0 ? 1 : 0;
		if (!allocPlay(g_play[which], n)) {
			printf("not enough ST-RAM to play\r\n");
			return false;
		}
		int16_t *f = g_play[which].frames;
		for (long i = 0; i < n; ++i) {
			f[2 * i] = g_pcm[i];
			f[2 * i + 1] = g_pcm[i];
		}
		g_play[which].count = n;
		waitPlayback();
		play(which);
	}
	if (g_stop)
		return false;
	if (Bconstat(2)) {
		Bconin(2);
		return false;
	}
	return true;
}

void shutdown() {
	waitPlayback();
	soundClose();
	if (g_dspHeld)
		g_dsp.release();
	g_dspHeld = false;
}

}  // namespace

int main(int argc, char **argv) {
	printf("AuroraTTS-100K for the Falcon030 DSP, %s\r\n", AURORA_VERSION);
	Pdomain(1);   // a MiNT program (TOS ignores it)
	signal(SIGINT, onSignal);   // stop after the current sentence, as a key does
	signal(SIGTERM, onSignal);
	signal(SIGQUIT, onSignal);
	signal(SIGHUP, onSignal);
	const char *wavPath = 0, *textFile = 0;
	static char text[4096];
	text[0] = 0;
	// no arguments: AURORA.CMD, when present, holds the command line
	static char cmdline[1024];
	static char *cargv[64];
	long csize = 0;
	char *cmd = argc <= 1 ? (char *)loadFile("AURORA.CMD", &csize) : 0;
	if (cmd) {
		strncpy(cmdline, cmd, sizeof cmdline - 1);
		free(cmd);
		int n = 0;
		cargv[n++] = argv[0];
		for (char *q = cmdline; *q && n < 63;) {
			while (*q == ' ' || *q == '\n' || *q == '\r')
				*q++ = 0;
			if (!*q)
				break;
			cargv[n++] = q;
			while (*q && *q != ' ' && *q != '\n' && *q != '\r')
				++q;
		}
		argc = n;
		argv = cargv;
	}
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-p"))
			g_phonemes = true;
		else if (!strcmp(argv[i], "-q"))
			g_quiet = true;
		else if (!strcmp(argv[i], "-w") && i + 1 < argc)
			wavPath = argv[++i];
		else if (!strcmp(argv[i], "-f") && i + 1 < argc)
			textFile = argv[++i];
		else {
			if (text[0])
				strncat(text, " ", sizeof text - strlen(text) - 1);
			strncat(text, argv[i], sizeof text - strlen(text) - 1);
		}
	}
	long size = 0;
	g_img = (int32_t *)loadFile("AURORA.BIN", &size);
	if (!g_img || size != (long)AURORA_IMAGE_WORDS * 4) {
		printf("AURORA.BIN is missing or does not match this program\r\n");
		return 1;
	}
	for (long i = 0; i < AURORA_IMAGE_WORDS; ++i)
		g_img[i] = (g_img[i] << 8) >> 8;
#ifdef AURORA_HAVE_FRONTEND
	long dsize = 0;
	uint8_t *dict = loadFile("DICT.BIN", &dsize);
	g_haveDict = dict && g_g2p.load(dict, dsize);
#endif
	if (textFile) {
		long tsize = 0;
		char *t = (char *)loadFile(textFile, &tsize);
		if (!t) {
			printf("cannot read %s\r\n", textFile);
			return 1;
		}
		strncpy(text, t, sizeof text - 1);
		free(t);
	}
	if (!g_dsp.boot()) {
		printf("the DSP is in use, or did not start\r\n");
		return 1;
	}
	g_dspHeld = true;
	const bool alive = g_dsp.ping();
	if (!alive) {
		printf("the DSP kernel did not answer\r\n");
		shutdown();
		return 1;
	}
	g_drv.init(&g_dsp, g_img);
	g_drv.passSums = false;
	if (!g_quiet && !soundOpen())
		printf("the sound system is locked: not playing\r\n");
	if (wavPath) {
		g_wav = fopen(wavPath, "wb");
		if (g_wav)
			wavHeader(g_wav, 0);
	}
	if (text[0] && g_phonemes) {
		speak(text);   // phonemes: one utterance
	} else if (text[0]) {
		static char sentence[512];
		for (const char *p = text; *p;) {
			p = nextSentence(p, sentence, sizeof sentence);
			if (sentence[0] && !speak(sentence))
				break;
			if (g_stop)
				break;
		}
	} else {
		printf("Type text, an empty line ends.\r\n");
		static char line[512];
		for (;;) {
			line[0] = (char)(sizeof line - 3);
			Cconrs(line);
			printf("\r\n");
			const int len = (unsigned char)line[1];
			if (len == 0)
				break;
			line[2 + len] = 0;
			if (!speak(line + 2))
				break;
		}
	}
	if (g_totalSamples > 0)
		printf("total: %ld.%02ld s of speech in %ld.%02ld s (%ld yields to other programs)\r\n", g_totalSamples / kRate,
		       (g_totalSamples % kRate) * 100 / kRate, (long)(g_totalTicks / 200), (long)(g_totalTicks % 200) / 2,
		       g_dsp.yields);
	if (g_wav) {
		fseek(g_wav, 0, SEEK_SET);
		wavHeader(g_wav, g_wavSamples);
		fclose(g_wav);
	}
	waitPlayback();
	// the gates stop the emulator when the DSP sees DONE
	g_dsp.done();
	shutdown();
	return 0;
}
