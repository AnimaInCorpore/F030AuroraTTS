; The synthesizer's overlay: its inner loops, run from internal P
; (aurora.asm loads it). Called from synth.asm.

        org     p:$0080,p:OVL_SYN_LOAD
ovl_syn_start:

; Harmonics: 256 samples into the overlap-add buffer at r0 (modulo m0),
;
hm_run:
        do      #256,hr_samples
        ; phase += inc; inc += dinc
        move    l:SY_PHASE,a
        move    l:SY_INC,x
        add     x,a
        move    a10,l:SY_PHASE
        move    a1,x:SY_TMP             ; ph
        move    l:SY_INC,a
        move    l:SY_DINC,y
        add     y,a
        move    a10,l:SY_INC
        move    x:SY_H,b
        tst     b
        jeq     hr_silent
        ; the sines: index floor(h ph / 2^14) (from the multiply) + the offset,
        ; whose table address n6 holds; each store lands one harmonic later
        ; Two harmonics an iteration, software-pipelined (5 instructions each):
        ; a sine is fetched one step after its address is formed and stored
        ; one step later still. An odd count computes one spare harmonic,
        ; whose offset (the table's 81st entry) is harmless.
        move    x:SY_TMP,x0
        move    #>$000200,y0
        move    #SYN_THQ,r1
        move    #SYN_SINEBUF,r5
        move    #SYN_SIN2,r2
        move    #0,n2
        move    x:SY_H,a
        move    #>1,x1
        add     x1,a
        asr     a
        move    a1,n4                   ; pairs: (H + 1) / 2
        move    x:SY_H,a
        move    a1,n3                   ; H, for the sums
        clr     b
        do      n4,hr_sines
        add     x0,b    x:(r1)+,n6
        move    x:(r2+n2),y1
        move    b1,x1
        mpy     x1,y0,a y1,y:(r5)+
        move    a1,r6
        add     x0,b    x:(r1)+,n2
        move    x:(r6+n6),y1
        move    b1,x1
        mpy     x1,y0,a y1,y:(r5)+
        move    a1,r2
hr_sines:
        nop
        move    x:(r2+n2),y1
        move    y1,y:(r5)+
        ; the two sums
        move    #SYN_DA,r1
        move    #SYN_SINEBUF+1,r5
        clr     a
        clr     b
        move    x:(r1)+,x0      y:(r5)+,y0
        do      n3,hr_sums
        mac     x0,y0,a x:(r1)+,x1
        mac     x1,y0,b x:(r1)+,x0      y:(r5)+,y0
hr_sums:
        move    a,x0                    ; s0
        move    b,x1                    ; s1
        move    x:SY_KK,y1
        mpy     x1,y1,b
        add     x0,b
        move    b,x1                    ; v
        move    x:(r0),a
        add     x1,a
        move    a,x:(r0)
hr_silent:
        move    (r0)+
        move    x:SY_KK,a
        move    #>$008000,x0
        add     x0,a
        move    a1,x:SY_KK
hr_samples:
        rts

; 256 samples from the overlap-add buffer (r0, modulo m0; each slot cleared)
; through the 65-tap FIR into the resampler ring (r1, modulo m1).
fir_block:
        move    x:SY_YRP,r3
        move    #64,m3
        move    #>$800000,y1
        do      #256,fb_done
        move    x:(r0),a
        clr     b
        move    b,x:(r0)+
        move    a,x:(r3)+               ; the newest; r3 now the oldest
        move    #SYN_FIRI,r4
        clr     a
        move    y1,a0
        move    x:(r3)+,x0      y:(r4)+,y0
        rep     #64
        mac     x0,y0,a x:(r3)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        move    a,x:(r1)+
fb_done:
        move    r3,x:SY_YRP
        move    #-1,m3
        rts

; Every resampler output with n + 16 <= xcount - 1 into SYN_OUT. The taps'
; ring pointer r1 (modulo 512) follows n; the sample is
; floor(sat24(floor(S / 2^20)) / 256), which equals the reference's
; clamp16(sat24(floor(S / 2^23)) >> 5): the accumulator's limiter clamps.
resample_run:
        move    x:SY_OUTN,a
        move    #>SYN_OUT,x0
        add     x0,a
        move    a1,r2
        move    x:SY_XCOUNT,a
        move    #>17,x0
        sub     x0,a
        move    a1,x:SY_TMP+7           ; the last n with its taps present
        move    #511,m1
        move    #511,m3
        move    x:SY_RPOS,a
        move    #>15,x0
        sub     x0,a
        move    #>511,x0
        and     x0,a
        move    #>SYN_XRING,x0
        add     x0,a
        move    a1,r1
rr_loop:
        move    x:SY_RPOS,a
        move    x:SY_TMP+7,x0
        cmp     x0,a
        jgt     rr_done
        move    r1,r3
        ; the phase's row: (lo >> 16) 32
        move    y:SY_RPOS,x0
        move    #>$001000,y0
        mpy     x0,y0,b                 ; floor(lo / 2^11), lo taken as signed
        move    #>$001fe0,x0
        and     x0,b
        move    #>SYN_RES,x0
        add     x0,b
        move    b1,r6
        clr     a
        move    #>$800000,x0
        move    x0,a0
        move    x:(r3)+,x0      y:(r6)+,y0
        rep     #31
        mac     x0,y0,a x:(r3)+,x0      y:(r6)+,y0
        mac     x0,y0,a
        rep     #3
        asl     a
        move    a,x0
        move    #>$008000,y0
        mpy     x0,y0,a
        move    a1,x:(r2)+
        ; rpos += step; r1 moves by the carry into n
        move    l:SY_RPOS,a
        move    l:SY_RSTEP,x
        add     x,a
        move    x:SY_RPOS,x0
        move    a10,l:SY_RPOS
        move    a1,b
        sub     x0,b
        move    b1,n1
        nop
        move    (r1)+n1
        jmp     rr_loop
rr_done:
        move    #-1,m1
        move    #-1,m3
        move    r2,a
        move    #>SYN_OUT,x0
        sub     x0,a
        move    a1,x:SY_OUTN
        rts

; Inverse radix-2 FFT of SYN_FRE (bit-reversed input; X real, Y imaginary),
; unscaled and limited: t = (wr br - ws bi, wr bi + ws br), a' = a + t and
; b' = 2a - a' (subl: the same 56-bit value as a - t). Twiddles are L words
; (X wr, Y ws). An element is reached through r0/r4 (a) and r1/r5 (b), one
; register per memory bank, so stores and loads pair in XY moves; each
; butterfly loads the next a's real part.
fft_inverse:
        move    #1,n1                   ; half
        move    #512,n6                 ; twiddle stride, and the group count
        do      #10,ffti_stage
        move    #SYN_FRE,r2
        move    n1,n2
        move    n1,n5
        do      n6,ffti_group
        move    r2,r0
        move    r2,r1
        move    r2,r4
        move    r2,r5
        move    #SYN_TW,r6
        move    (r1)+n1
        move    (r5)+n5
        move    x:(r0),a
        do      n2,ffti_fly
        move    l:(r6)+n6,y
        move    l:(r1),x
        tfr     a,b
        mac     x1,y1,a
        mac     -x0,y0,a
        subl    a,b     a,x:(r0)+
        move    b,x:(r1)+       y:(r4),a
        tfr     a,b
        mac     x0,y1,a
        mac     x1,y0,a
        subl    a,b     a,y:(r4)+
        move    x:(r0),a        b,y:(r5)+
ffti_fly:
        move    (r2)+n2
        move    (r2)+n2
        nop
ffti_group:
        move    n1,a
        asl     a
        move    a1,n1
        move    n6,a
        asr     a
        move    a1,n6
        nop
ffti_stage:
        rts


; The 513 noise magnitudes of a frame: r4 the bin map, r5/r6 exp's values
; and slopes, r3 the magnitudes.
noise_mags:
        ; runs of bins sharing a band pair: [runs], then per run [band, count,
        ; count fractions] (the host builds it from the bin map)
        move    y:(r4)+,a
        do      a1,nm_runs
        move    y:(r4)+,n0              ; band
        move    #SYN_BANDS+40,r0
        nop
        move    x:(r0+n0),a
        move    a,y1                    ; b[i0]
        move    (r0)+
        nop
        move    x:(r0+n0),a
        sub     y1,a
        move    a,x1                    ; b[i0+1] - b[i0], limited
        move    y:(r4)+,a               ; bins in the run
        do      a1,nm_bins
        move    y:(r4)+,y0              ; fraction
        mpy     x1,y0,a
        rep     #7
        asl     a
        move    a,x0
        tfr     y1,a
        add     x0,a
        move    a,x0                    ; e, limited
        ; mag = exp(e): interval (e + 16) >> 11, fraction ((e + 16) & $7ff) << 12
        move    #>16*ONE,a
        add     x0,a
        move    a,x0
        move    #>$001000,y0
        mpy     x0,y0,a
        move    a1,n5
        move    a1,n6
        move    a0,b
        lsr     b
        move    b1,x0
        move    y:(r5+n5),a             ; r5 = SYN_EXP
        move    y:(r6+n6),y0            ; r6 = its slopes
        mac     x0,y0,a
        move    a,x:(r3)+
nm_bins:
        nop
nm_runs:
        rts
; Z'k = p Zk + q conj(Z(N-k)), p = (mA + mB) >> 1, q = (mA - mB) >> 1, for
; k and N - k together (A = prev's magnitudes, B = cur's)
noise_combine:
        move    x:SY_PREV,b
        tst     b
        jne     nc_p1
        move    #SYN_MAG,r2
        move    #SYN_MAG+513,r3
        jmp     nc_go
nc_p1:
        move    #SYN_MAG+513,r2
        move    #SYN_MAG,r3
nc_go:
        move    #SYN_FRE,r0             ; k
        move    #SYN_FRE+1024,r1        ; N - k, but bin 0 pairs with itself
        do      #513,nc_done
        move    r1,r6
        move    r0,b
        move    #>SYN_FRE,x0
        cmp     x0,b
        jne     nc_pq
        move    #SYN_FRE,r6
nc_pq:
        ; p = (mA + mB) >> 1, q = (mA - mB) >> 1
        move    x:(r2)+,x1
        move    x:(r3)+,x0
        tfr     x1,a
        add     x0,a
        asr     a
        move    a,y1                    ; p
        tfr     x1,a
        sub     x0,a
        asr     a
        move    a,y0                    ; q
        move    x:(r6),a
        move    a,x:SY_TMP+3            ; yr
        move    y:(r6),a
        move    a,x:SY_TMP+4            ; yi
        move    x:(r0),x0               ; zr
        ; re k = (p zr + q yr) >> 20
        move    x:SY_TMP+3,x1
        mpy     x0,y1,a
        mac     x1,y0,a
        rep     #3
        asl     a
        move    a,x:SY_TMP+5
        ; im k = (p zi - q yi) >> 20
        move    y:(r0),x1
        mpy     x1,y1,a
        move    x:SY_TMP+4,x1
        mac     -x1,y0,a
        rep     #3
        asl     a
        move    a,x:SY_TMP+6
        ; j == k (bins 0 and N/2): k's values only
        move    r6,a
        move    r0,x1
        cmp     x1,a
        jeq     nc_store_k
        ; re j = (p yr + q zr) >> 20
        move    x:SY_TMP+3,x1
        mpy     x1,y1,a
        mac     x0,y0,a
        rep     #3
        asl     a
        move    a,x:(r6)
        ; im j = (p yi - q zi) >> 20
        move    x:SY_TMP+4,x1
        mpy     x1,y1,a
        move    y:(r0),x1
        mac     -x1,y0,a
        rep     #3
        asl     a
        move    a,y:(r6)
nc_store_k:
        move    x:SY_TMP+5,a
        move    a,x:(r0)
        move    x:SY_TMP+6,a
        move    a,y:(r0)+
        move    (r1)-
nc_done:
        rts

ovl_syn_end:
        nop
        IF      (ovl_syn_end-ovl_syn_start)>$180
        FAIL    'the synthesizer overlay exceeds internal P'
        ENDIF

        org     p:$7000
