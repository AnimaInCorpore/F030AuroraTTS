; AuroraTTS harmonic + noise synthesizer: src/aurora/synth.h on the DSP,
; word for word. Included by aurora.asm.
;
; One step per network frame (CMD_SYNTH_STEP): the frame's parameters, 256
; samples of harmonics, at odd steps two noise frames through one complex FFT
; pair, then the overlap-add segment three frames back through the output FIR
; and the 22,050 -> 24,585 Hz resampler. The host uploads the tables to the
; fixed addresses below (src/aurora/synth-map.h holds the same map).

; ---------------------------------------------------------------- reset

synth_reset:
        clr     a
        move    #SYN_AMP,r0
        rep     #160
        move    a,x:(r0)+
        move    #SYN_MAG,r0
        do      #1026,sr_mag
        move    a,x:(r0)+
sr_mag:
        move    #SYN_OLA,r0
        do      #2048,sr_ola
        move    a,x:(r0)+
sr_ola:
        move    #SYN_YRING,r0
        rep     #65
        move    a,x:(r0)+
        move    #SYN_XRING,r0
        do      #512,sr_xring
        move    a,x:(r0)+
sr_xring:
        move    #SYN_FIR,r4
        move    #SYN_FIRI,r5
        do      #65,sr_fir
        move    y:(r4)+,x0
        move    x0,y:(r5)+
sr_fir:
        move    a,x:SY_PHASE
        move    a,y:SY_PHASE
        move    a,x:SY_INC
        move    a,y:SY_INC
        move    a,x:SY_RPOS
        move    a,y:SY_RPOS
        move    a,x:SY_STEP
        move    a,x:SY_HC
        move    a,x:SY_HC+1
        move    a,x:SY_INC28
        move    a,x:SY_INC28+1
        move    a,x:SY_PAD
        move    a,x:SY_PAD+1
        ; the resampler step: 2^24 + const 3, below 1.0: hi 0, lo the fraction
        move    #SYN_CONST,r0
        move    #>3,n0
        nop
        move    y:(r0+n0),a
        move    a1,y:SY_RSTEP
        clr     a
        move    a,x:SY_RSTEP
        move    #>-32,a
        move    a1,x:SY_XCOUNT
        move    #>SYN_YRING,a
        move    a1,x:SY_YRP
        move    #>4,n0
        nop
        move    y:(r0+n0),a
        move    a1,x:SY_LCG
        rts

; ---------------------------------------------------------------- one step

synth_step:
        move    a1,x:SY_TMP+7           ; the pad flag
        ; slots: cur = m & 1
        move    x:SY_STEP,a
        move    #>1,x0
        and     x0,a
        move    a1,x:SY_CUR
        eor     x0,a
        move    a1,x:SY_PREV
        move    x:SY_TMP+7,b
        tst     b
        jne     ss_pad
        move    #SYN_HEAD,r0
        jclr    #1,x:m_hsr,*            ; ready: the host may now write the block unpaced
        movep   #>REPLY_READY,x:m_htx
        do      #66,ss_in
        jclr    #0,x:m_hsr,*
        movep   x:m_hrx,x:(r0)+
ss_in:
        jsr     frame_params
        jmp     ss_noise
ss_pad:
        jsr     pad_params
ss_noise:
        clr     a
        move    a,x:SY_OUTN
        ; harmonics between frames m-1 and m, unless m is 0 or this is a pad
        move    x:SY_STEP,a
        tst     a
        jeq     ss_pair
        move    #SY_PAD,r0
        move    x:SY_CUR,n0
        nop
        move    x:(r0+n0),b
        tst     b
        jne     ss_pair
        jsr     harmonics
ss_pair:
        move    x:SY_STEP,a
        move    #>1,x0
        and     x0,a
        jeq     ss_out
        jsr     noise_pair
ss_out:
        jsr     readout
        ; reply: count, then the samples
        move    x:SY_OUTN,a
        jsr     host_send
        move    x:SY_OUTN,b
        tst     b
        jeq     ss_sent
        move    #SYN_OUT,r0
        do      b1,ss_sent
        jclr    #1,x:m_hsr,*
        movep   x:(r0)+,x:m_htx
ss_sent:
        move    x:SY_STEP,a
        move    #>1,x0
        add     x0,a
        move    a1,x:SY_STEP
        rts

; ---------------------------------------------------------------- frame parameters

pad_params:
        move    #SY_INC28,r0
        move    x:SY_PREV,n0
        nop
        move    x:(r0+n0),a
        move    x:SY_CUR,n0
        nop
        move    a1,x:(r0+n0)
        clr     a
        move    #SY_HC,r0
        nop
        move    a1,x:(r0+n0)
        move    #>1,x0
        move    #SY_PAD,r0
        nop
        move    x0,x:(r0+n0)
        jsr     slot_pointers           ; r2 amplitudes, r3 magnitudes
        rep     #80
        move    a,x:(r2)+
        do      #513,pp_mag
        move    a,x:(r3)+
pp_mag:
        rts

; r2 = SYN_AMP + 80 cur, r3 = SYN_MAG + 513 cur
slot_pointers:
        move    x:SY_CUR,b
        tst     b
        jne     sp_one
        move    #SYN_AMP,r2
        move    #SYN_MAG,r3
        rts
sp_one:
        move    #SYN_AMP+80,r2
        move    #SYN_MAG+513,r3
        rts

; Linear interpolation: a1 = u (>= 0), y0 = 2^(23 - bits), r6 values,
; n6 = the table's interval count. Returns the value in a (limited by the caller).
; Uses b, x0, y0, r5, n5.
interp_y:
        move    a1,x0
        mpy     x0,y0,a
        move    a1,n5
        move    r6,r5
        move    a0,b
        lsr     b
        move    b1,x0                   ; the fraction, Q0.23
        move    y:(r5+n5),a             ; value
        move    r6,b
        move    n6,y0
        add     y0,b
        move    b1,r5
        nop
        move    y:(r5+n5),y0            ; slope
        mac     x0,y0,a
        rts

frame_params:
        move    x:SY_CUR,n0
        clr     a
        move    #SY_PAD,r0
        nop
        move    a1,x:(r0+n0)
        ; x = f0n std + mean - ln(sr), Q3.20: the row's sum to 20 fraction bits
        move    #SYN_CONST+8,r4
        move    #>ONE,x0
        move    x:SYN_HEAD,x1
        clr     a       y:(r4)+,y0
        move    #>$010000,y1            ; 2^(19 - shift) in A0 for shift 3
        move    y1,a0
        mac     x0,y0,a y:(r4)+,y0
        mac     x1,y0,a
        rep     #4+F0ROW_SHIFT
        asl     a
        move    a,x0
        move    #SYN_CONST,r4
        tfr     x0,a
        move    y:(r4)+,x1              ; lowest
        cmp     x1,a
        tlt     x1,a
        move    y:(r4)+,x1              ; highest
        cmp     x1,a
        tgt     x1,a
        ; inc28 = interp(inc, x + 6 2^20, 12 bits)
        move    #>$600000,x0
        add     x0,a
        move    #>$000800,y0
        move    #SYN_INC,r6
        move    #INC_N,n6
        jsr     interp_y
        move    #SY_INC28,r0
        move    x:SY_CUR,n0
        nop
        move    a,x:(r0+n0)
        move    a,x:SY_TMP              ; inc28
        ; bands, clamped to [-14, 2]
        move    #SYN_HEAD+2,r0
        move    #SYN_BANDS,r1
        move    #>-14*ONE,x0
        move    #>2*ONE,x1
        do      #64,fp_bands
        move    x:(r0)+,a
        cmp     x0,a
        tlt     x0,a
        cmp     x1,a
        tgt     x1,a
        move    a,x:(r1)+
fp_bands:
        jsr     slot_pointers
        clr     a
        move    r2,r1
        rep     #80
        move    a,x:(r1)+
        move    #SY_HC,r0
        move    x:SY_CUR,n0
        nop
        move    a,x:(r0+n0)
        ; voiced: vuv > 0
        move    x:SYN_HEAD+1,b
        tst     b
        jle     fp_noise
        ; nu: inc28 2^20 as 48 bits, summed per harmonic in B
        move    x:SY_TMP,x0
        move    #>$080000,y0
        mpy     x0,y0,a                 ; inc28 2^20
        move    a1,x:SY_TMP+1
        move    a0,y:SY_TMP+1
        move    #SYN_CONST+2,r4
        nop
        move    y:(r4),x0
        move    x0,x:SY_TMP+2           ; alive limit
        clr     b
        move    b,x:SY_TMP+3            ; harmonics alive
        move    b,y:SY_TMP+3
        do      #80,fp_harm
        move    l:SY_TMP+1,x
        move    l:SY_TMP+3,b
        add     x,b
        move    b10,l:SY_TMP+3          ; h inc28 2^20
        move    x:SY_TMP+2,x0
        cmp     x0,b
        jge     fp_dead
        ; pos = interp(melh, v, 14 bits), at most 39.0
        move    b1,a
        move    #>$000200,y0
        move    #SYN_MELH,r6
        move    #MELH_N,n6
        jsr     interp_y
        move    #>39*ONE,x0
        cmp     x0,a
        tgt     x0,a
        ; i0 = min(pos >> 16, 38), fr = pos - i0 2^16
        move    a,x:SY_TMP+4            ; pos
        move    a,x0
        move    #>$000080,y0
        mpy     x0,y0,b                 ; pos >> 16 in B1, the fraction below
        move    #>39,x0
        cmp     x0,b
        jlt     fp_i0
        move    #>38,b                  ; pos is 39.0: i0 = 38, fr = 1.0
        move    #>ONE,x0
        move    x0,x:SY_TMP+5
        jmp     fp_e
fp_i0:
        move    x:SY_TMP+4,a
        move    #>$00ffff,x0
        and     x0,a
        move    a1,x:SY_TMP+5
fp_e:
        ; e = b[i0] + mul(b[i0+1] - b[i0], fr)
        move    b1,n0
        move    #SYN_BANDS,r0
        nop
        move    x:(r0+n0),a
        move    a,y1                    ; b[i0]
        move    (r0)+
        nop
        move    x:(r0+n0),a
        sub     y1,a
        move    a,x0
        move    x:SY_TMP+5,y0
        mpy     x0,y0,a
        rep     #7
        asl     a
        move    a,x0
        tfr     y1,a
        add     x0,a                    ; e, limited below
        move    a,x0
        tfr     x0,a
        ; amp = interp(exp, e + 16, 11 bits)
        move    #>16*ONE,x0
        add     x0,a
        move    #>$001000,y0
        move    #SYN_EXP,r6
        move    #EXP_N,n6
        jsr     interp_y
        move    a,x:(r2)+
        move    #SY_HC,r0
        move    x:SY_CUR,n0
        nop
        move    x:(r0+n0),a
        move    #>1,x0
        add     x0,a
        move    a1,x:(r0+n0)
        nop
        jmp     fp_next
fp_dead:
        ; every later harmonic is above the limit too (and its sum would
        ; read back negative): leave the loop
        enddo
        jmp     fp_noise
fp_next:
        nop
fp_harm:
fp_noise:
        ; noise magnitudes: 513 bins
        jsr     slot_pointers
        move    #SYN_NMAP,r4
        move    #SYN_EXP,r5
        move    #SYN_EXP+EXP_N,r6
        jsr     noise_mags
        rts


; ---------------------------------------------------------------- harmonics

; The segment between frames m-1 (prev) and m (cur): positions (m-1) 256 ..
harmonics:
        ; H = max(hc[prev], hc[cur]); the [amplitude, change] pairs
        move    #SY_HC,r0
        move    x:SY_PREV,n0
        nop
        move    x:(r0+n0),a
        move    x:SY_CUR,n0
        nop
        move    x:(r0+n0),x0
        cmp     x0,a
        tlt     x0,a
        move    a1,x:SY_H
        move    x:SY_PREV,b
        tst     b
        jne     hm_p1
        move    #SYN_AMP,r0
        move    #SYN_AMP+80,r1
        jmp     hm_pairs
hm_p1:
        move    #SYN_AMP+80,r0
        move    #SYN_AMP,r1
hm_pairs:
        move    #SYN_DA,r2
        do      #80,hm_pairs_done
        move    x:(r0)+,x0
        move    x:(r1)+,a
        sub     x0,a    x0,x:(r2)+
        move    a,x:(r2)+
hm_pairs_done:
        ; the increment: inc28[prev] 2^20, its change (inc28[cur] - inc28[prev]) 2^12
        move    #SY_INC28,r0
        move    x:SY_PREV,n0
        nop
        move    x:(r0+n0),x0
        move    #>$080000,y0
        mpy     x0,y0,a
        move    a10,l:SY_INC
        move    x:SY_CUR,n0
        nop
        move    x:(r0+n0),a
        sub     x0,a
        move    a,x0
        move    #>$000800,y0
        mpy     x0,y0,a
        move    a10,l:SY_DINC
        ; the OLA position (m-1) 256, modulo 2048
        move    x:SY_STEP,a
        move    #>1,x0
        sub     x0,a
        move    a1,x0
        move    #>$000080,y0            ; * 256: 2 x y = x 2^8 in A0
        mpy     x0,y0,a
        move    a0,a
        move    #>2047,x0
        and     x0,a
        move    #>SYN_OLA,x0
        add     x0,a
        move    a1,r0
        move    #2047,m0
        clr     a
        move    a,x:SY_KK
        jsr     hm_run
        move    #-1,m0
        rts
hm_samples:
        move    #-1,m0
        rts

; ---------------------------------------------------------------- noise

; Noise frames m-1 (prev) and m (cur): a random spectrum shaped by both
; frames' magnitudes, one inverse FFT, overlap-added at a0 and b0.
noise_pair:
        ; a0 = (m-1) 256 - 512, b0 = m 256 - 512
        move    x:SY_STEP,x0
        move    #>$000080,y0
        mpy     x0,y0,a
        move    a0,a                    ; m 256
        move    #>512,x0
        sub     x0,a
        move    a1,x:SY_TMP+1           ; b0
        move    #>256,x0
        sub     x0,a
        move    a1,x:SY_TMP             ; a0
        jsr     spectrum_fill
        jsr     noise_combine
        jsr     bit_reverse
        jsr     fft_inverse
        ; overlap-add: a at a0, b at b0
        move    x:SY_TMP,a
        move    #>2047,x1
        and     x1,a
        move    #>SYN_OLA,x0
        add     x0,a
        move    a1,r0
        move    x:SY_TMP+1,a
        and     x1,a
        add     x0,a
        move    a1,r1
        move    #2047,m0
        move    #2047,m1
        move    #SYN_WOLA,r2
        move    #SYN_FRE,r4
        jsr     ola_add
        move    #-1,m0
        move    #-1,m1
        rts




; In-place bit reversal of SYN_FRE (X and Y).
bit_reverse:
        move    #SYN_FRE,r0
        move    #SYN_FRE,r4
        move    #0,m4
        move    #512,n4
        do      #1024,br_done
        move    r0,a
        move    r4,x0
        cmp     x0,a
        jge     br_next
        move    x:(r0),x0
        move    x:(r4),x1
        move    x1,x:(r0)
        move    x0,x:(r4)
        move    y:(r0),x0
        move    y:(r4),x1
        move    x1,y:(r0)
        move    x0,y:(r4)
br_next:
        move    (r0)+
        move    (r4)+n4
br_done:
        move    #-1,m4
        rts

; A random spectrum in SYN_FRE (X real, Y imaginary), natural order: for each
; bin the LCG's next two words, real first, scaled by constant 7.
spectrum_fill:
        move    #SYN_FRE,r1
        move    #SYN_CONST+5,r4
        nop
        move    y:(r4)+,y0              ; multiplier
        move    y:(r4)+,y1              ; increment
        move    y:(r4)+,x1              ; scale
        move    x:SY_LCG,x0
        do      #1024,sf_done
        mpy     x0,y0,a
        asr     a
        move    a0,b
        add     y1,b
        move    b1,x0
        mpy     x0,x1,a
        move    a,x:(r1)
        mpy     x0,y0,a
        asr     a
        move    a0,b
        add     y1,b
        move    b1,x0
        mpy     x0,x1,a
        move    a,y:(r1)+
sf_done:
        move    x0,x:SY_LCG
        rts

; Overlap-add of the inverse FFT: real part at r0, imaginary at r1 (modulo
; m0/m1), r2 the Hann / 1.5 window, r4 the FFT buffer.
ola_add:
        do      #1024,np_ola
        move    x:(r2)+,y0
        move    x:(r4),x0
        mpy     x0,y0,a y:(r4)+,x1
        mpy     x1,y0,b
        move    a,x1
        move    x:(r0),a
        add     x1,a
        move    a,x:(r0)+
        move    b,x1
        move    x:(r1),b
        add     x1,b
        move    b,x:(r1)+
np_ola:
        rts

; ---------------------------------------------------------------- output

; Segment m-3 of the overlap-add buffer through the FIR and the resampler;
; every slot read is cleared. Before step 3 the segment is only cleared.
readout:
        move    x:SY_STEP,a
        move    #>3,x0
        sub     x0,a
        move    a1,x0
        move    #>$000080,y0
        mpy     x0,y0,a
        move    a0,a
        move    #>2047,x0
        and     x0,a
        move    #>SYN_OLA,x0
        add     x0,a
        move    a1,r0
        move    #2047,m0
        move    x:SY_STEP,a
        move    #>3,x0
        cmp     x0,a
        jge     ro_run
        clr     a
        do      #256,ro_clear
        move    a,x:(r0)+
ro_clear:
        move    #-1,m0
        rts
ro_run:
        ; the FIR outputs go to the ring from slot xcount & 511
        move    x:SY_XCOUNT,a
        move    a1,x:SY_TMP+6
        move    #>511,x0
        and     x0,a
        move    #>SYN_XRING,x0
        add     x0,a
        move    a1,r1
        move    #511,m1
        jsr     fir_block
        move    #-1,m0
        ; outputs before position 0 hold nothing (the first block only)
        move    x:SY_TMP+6,b
        tst     b
        jge     ro_count
        move    b1,a
        move    #>511,x0
        and     x0,a
        move    #>SYN_XRING,x0
        add     x0,a
        move    a1,r1
        neg     b
        clr     a
        do      b1,ro_count
        move    a,x:(r1)+
ro_count:
        move    #-1,m1
        move    x:SY_TMP+6,a
        move    #>256,x0
        add     x0,a
        move    a1,x:SY_XCOUNT
        jsr     resample_run
        rts
