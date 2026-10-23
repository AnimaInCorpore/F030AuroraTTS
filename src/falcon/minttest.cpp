// MINTTEST.TOS: the FreeMiNT cooperation test, run as MiNT's INIT program.
// It measures a busy loop alone for two seconds, then starts AURORA.TOS
// beside it (Pexec mode 100, no shell) and keeps counting, one line a second
// in A:\AURORA\MINT.LOG: how much CPU a neighbour keeps while a sentence
// renders. The player reads its command line from AURORA.CMD.
#include <mint/mintbind.h>
#include <mint/osbind.h>
#include <stdio.h>
#include <time.h>

static volatile unsigned long g_sink;

// Busy work until the next whole second of clock(); returns the iterations.
static unsigned long countSecond() {
	const clock_t start = clock();
	unsigned long n = 0;
	while ((unsigned long)(clock() - start) < (unsigned long)CLOCKS_PER_SEC) {
		for (int i = 0; i < 200; ++i)
			g_sink += i;
		++n;
	}
	return n;
}

int main() {
	Dsetdrv(0);   // the boot floppy: MiNT cannot use Hatari's GEMDOS folder drive
	Dsetpath("\\AURORA");
	FILE *log = fopen("MINT.LOG", "w");
	if (!log)
		return 1;
	fprintf(log, "baseline\n");
	unsigned long base = 0;
	for (int s = 0; s < 2; ++s) {
		const unsigned long n = countSecond();
		base += n;
		fprintf(log, "  %lu\n", n);
		fflush(log);
	}
	base /= 2;
	// ALONE.FLG: the player alone; only its run time is logged
	FILE *flag = fopen("ALONE.FLG", "r");
	const clock_t t0 = clock();
	const long pid = Pexec(100, (char *)"AURORA.TOS", (char *)"\0", 0);
	fprintf(log, "player pid %ld%s\n", pid, flag ? ", alone" : "");
	fflush(log);
	if (flag) {
		long status = 0;
		Pwaitpid(pid, 0, &status);
		const long t = (long)(clock() - t0);
		fprintf(log, "player ended after %ld.%02ld s\n", t / CLOCKS_PER_SEC, (t % CLOCKS_PER_SEC) * 100 / CLOCKS_PER_SEC);
		fclose(log);
		for (;;)
			Syield();
	}
	for (int s = 0; s < 600; ++s) {
		const unsigned long n = countSecond();
		long status = 0;
		const long done = Pwaitpid(pid, 1 /* WNOHANG */, &status);
		fprintf(log, "%3d s: %lu (%lu%% of alone)%s\n", s + 1, n, base ? n * 100 / base : 0,
		        done == pid ? ", player ended" : "");
		fflush(log);
		if (done == pid)
			break;
	}
	fclose(log);
	for (;;)
		Syield();
	return 0;
}
