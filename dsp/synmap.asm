; The synthesizer's memory map and state (see synth.asm and src/aurora/synth-driver.h).

; ---------------------------------------------------------------- memory map
; X
SYN_AMP         equ     $0200           ; amplitudes, Q3.20: slot s at +80 s
SYN_DA          equ     $02a0           ; [amplitude, change] per harmonic for one segment
SYN_YRING       equ     $0380           ; FIR input history, 65 words (modulo 65)
SYN_XRING       equ     $0800           ; resampler input, 512 words (modulo 512)
SYN_HEAD        equ     $0440           ; the frame's 66 words
SYN_BANDS       equ     $0490           ; its 64 bands, clamped
SYN_OUT         equ     $04e0           ; one step's output samples (320 at most)
SYN_OLA         equ     $1000           ; overlap-add by position modulo 2048
SYN_FRE         equ     $1800           ; FFT real (imaginary: the same address in Y)
SYN_TW          equ     $1c00           ; FFT twiddles, L words k = 0..511: X cos, Y sin (2 pi k / 1024)
SYN_SIN2        equ     $2000           ; sine over two periods: SYN_SIN2 + j holds sin(2 pi (j - 512) / 1024)
SYN_WOLA        equ     $0a00           ; Hann / 1.5
SYN_THQ         equ     $00a8           ; per harmonic: SYN_SIN2 + 512 + its offset (internal, 81)
SYN_MAG         equ     $2800           ; noise magnitudes, Q3.20: slot s at +513 s
; Y
SYN_SINEBUF     equ     $0000           ; this sample's harmonic sines from $0001 (internal)
SYN_FIRI        equ     $00b0           ; the FIR's taps (internal copy; Y $80-$a7 hold state L words)
SYN_EXP         equ     $0200           ; exp: 577 values, 577 slopes
SYN_MELH        equ     $0700           ; harmonic band position: 512, 512
SYN_INC         equ     $0b00           ; phase increment: 640, 640
SYN_NMAP        equ     $1000           ; noise bin: [band, fraction] * 513
SYN_FIR         equ     $1500           ; 65 taps
SYN_CONST       equ     $1580           ; 8 constants, then the f0 row (2)
SYN_FIM         equ     $1800
SYN_RES         equ     $2000           ; resampler: 256 phases of 32 taps

EXP_N           equ     577
MELH_N          equ     512
INC_N           equ     640
F0ROW_SHIFT     equ     3

; state: L words at X/Y $80.., scalars in X
SY_PHASE        equ     $0080           ; L: cycles * 2^48
SY_INC          equ     $0081           ; L: this segment's increment
SY_DINC         equ     $0082           ; L: its change per sample
SY_RPOS         equ     $0083           ; L: resampler time: input sample (hi), fraction (lo)
SY_RSTEP        equ     $0084           ; L: its step
SY_STEP         equ     $0090           ; m
SY_LCG          equ     $0091
SY_XCOUNT       equ     $0092           ; FIR outputs produced, from -32
SY_HC           equ     $0093           ; harmonics alive, by slot (2)
SY_INC28        equ     $0095           ; increment cycles * 2^28, by slot (2)
SY_PAD          equ     $0097           ; pad step, by slot (2)
SY_YRP          equ     $009a           ; FIR history: the next write (the oldest)
SY_OUTN         equ     $009b           ; samples in SYN_OUT
SY_KK           equ     $009c           ; k << 15 within a segment
SY_CUR          equ     $009d           ; this step's slot
SY_PREV         equ     $009e
SY_H            equ     $009f           ; harmonics in this segment
SY_TMP          equ     $00a0           ; 8 words

