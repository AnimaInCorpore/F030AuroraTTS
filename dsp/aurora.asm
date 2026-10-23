; AuroraTTS-100K on the Falcon's DSP56001: the network machine.
;
; The 68030 streams tokens or frames through one layer at a time: it uploads
; a pass's weights, tables and two small programs (a step and a pad step),
; then runs the step program once per item. The programs are lists of vector
; operations; src/aurora/vm.h defines them and holds the host interpreter
; this kernel must match word for word (src/aurora/fixed.h has the
; arithmetic).
;
; Memory map (words). External X aliases P $4000-$7fff, external Y P $0000-$3fff.
;   P internal $0000-$003f  vectors; $0040-$007f the boot loader (transient)
;   P internal $0080-$01ff  inner loops
;   P external $7000-$7fff  everything else (= X $3000-$3fff)
;   X internal $0000-$003f  scalars ($0010 written by the host)
;   X internal $0040-$007f  layer norm deviations
;   X internal $0080-$00ff  a matrix's input (the synthesizer's state when it runs)
;   X $0200-$2fff, Y $0000-$00ff and $0200-$3fff: the pass's (host allocated)
;   X and Y $0100-$01ff     not used (the data ROMs' window)

        include 'ioequ.inc'
        include 'synmap.asm'

; ---------------------------------------------------------------- commands

CMD_PING        equ     $01
CMD_WRITE_X     equ     $02             ; address, count, words
CMD_WRITE_Y     equ     $03
CMD_RUN         equ     $04             ; arg: program address in X
CMD_READ_X      equ     $05             ; arg: address; reply the word
CMD_DONE        equ     $06             ; reply 0 and pass the gates' breakpoint
CMD_READ_Y      equ     $07             ; arg: address; reply the word
CMD_SYNTH_RESET equ     $10             ; synthesizer state to its start
CMD_SYNTH_STEP  equ     $11             ; arg 1: a pad step, else 66 head words follow;
                                        ; reply: a count, then that many 16-bit samples
REPLY_PING      equ     $415552         ; "AUR"
REPLY_ERROR     equ     $ffffff
REPLY_READY     equ     $524459         ; "RDY": sent from just before a block receive loop

; operations, in vm.h's order
OP_END          equ     0
OP_COUNT        equ     16

ONE             equ     $010000         ; 1.0 in Q7.16
LN_C56          equ     149797          ; round(2^23 / 56)
LN_EPS2N        equ     4810400         ; 2 * 56 * round(1e-5 * 2^32): 2 S's preload

; ---------------------------------------------------------------- X scalars

        org     x:$0000
last_command:   ds      1
overlay:        ds      1               ; the overlay in internal P
rc_table:       ds      4               ; matrix rounding by shift: 2^(23 - s) as A0
ln_rc_table:    ds      8               ; layer norm rounding by shift: 2^(6 - s) as A1
        org     x:$0010
rsqrt_table:    ds      1               ; host: layer norm's table (values, slopes at +384)
ln_shift:       ds      1
op_n:           ds      1
op_rows:        ds      1
op_src:         ds      1
op_dst:         ds      1
op_w:           ds      1
ring_base:      ds      1
ring_size:      ds      1
ring_n:         ds      1
ring_desc:      ds      1
scratch:        ds      4
        org     x:$0020
act_offset:     ds      1               ; the current activation descriptor
act_mult:       ds      1
act_loval:      ds      1
act_hival:      ds      1
act_ident:      ds      1

LN_TEMP         equ     $0040           ; 64 words
MV_STAGE        equ     $0080           ; a matrix's input, 128 words

; ---------------------------------------------------------------- vectors

        org     p:$0000
        jmp     start

; ================================================================ overlays
; Internal P ($0080-$01ff) holds one of two overlays: the network machine's
; inner loops or the synthesizer's. Each is assembled to run at $0080 and
; stored in external P, and load_overlay copies the one a command needs.
OVL_VM_LOAD     equ     $7800
OVL_SYN_LOAD    equ     $7a00
OVL_NONE        equ     0
OVL_VM          equ     1
OVL_SYN         equ     2

        org     p:$0080,p:OVL_VM_LOAD
ovl_vm_start:

; Matrix rows, weights in Y, input in X: r2 input, r4 rows, r1 output,
; n0 = n - 1, x1 = 1.0, y1 = rounding, b = row count. One variant per shift.
mvy_s0:
        do      b,mvy_s0_end
        move    r2,r0
        clr     a       y:(r4)+,y0
        move    y1,a0
        mac     x1,y0,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        move    a,x:(r1)+
mvy_s0_end:
        rts
mvy_s1:
        do      b,mvy_s1_end
        move    r2,r0
        clr     a       y:(r4)+,y0
        move    y1,a0
        mac     x1,y0,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        asl     a
        move    a,x:(r1)+
mvy_s1_end:
        rts
mvy_s2:
        do      b,mvy_s2_end
        move    r2,r0
        clr     a       y:(r4)+,y0
        move    y1,a0
        mac     x1,y0,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        asl     a
        asl     a
        move    a,x:(r1)+
mvy_s2_end:
        rts
mvy_s3:
        do      b,mvy_s3_end
        move    r2,r0
        clr     a       y:(r4)+,y0
        move    y1,a0
        mac     x1,y0,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        asl     a
        asl     a
        asl     a
        move    a,x:(r1)+
mvy_s3_end:
        rts


; Table activation over n0 words: r0 source, r1 destination, r3 values,
; r6 slopes, x1 = lo, y1 = hi (the descriptor's other words in act_*).
act_loop:
        do      n0,act_loop_end
        move    x:(r0)+,a
        cmp     x1,a
        jlt     act_below
        cmp     y1,a
        jge     act_above
        sub     x1,a    x:act_mult,y0   ; u = x - lo
        move    a,x0
        mpy     x0,y0,a
        move    a1,n3
        move    a1,n6
        move    a0,b
        lsr     b
        move    b1,x0
        move    x:(r3+n3),a
        move    x:(r6+n6),y0
        mac     x0,y0,a
act_store:
        move    a,x:(r1)+
        nop
act_loop_end:
        rts
act_below:
        move    x:act_loval,a
        jmp     act_store
act_above:
        move    x:act_ident,b
        tst     b
        jne     act_store
        move    x:act_hival,a
        jmp     act_store

; Depthwise taps: n0 channels, r5 the first channel's oldest tap (modulo
; m5), n1 = dil n (modulo m1 = m5), r4 rows [bias, 5 taps], r6 output,
; x1 = 1.0, y1 = rounding. b = shift (0 or 1).
dw_loop0:
        do      n0,dw_loop0_end
        move    r5,r1
        clr     a       y:(r4)+,y0
        move    y1,a0
        mac     x1,y0,a x:(r1)+n1,x0    y:(r4)+,y0
        rep     #4
        mac     x0,y0,a x:(r1)+n1,x0    y:(r4)+,y0
        mac     x0,y0,a (r5)+
        move    a,x:(r6)+
dw_loop0_end:
        rts
dw_loop1:
        do      n0,dw_loop1_end
        move    r5,r1
        clr     a       y:(r4)+,y0
        move    y1,a0
        mac     x1,y0,a x:(r1)+n1,x0    y:(r4)+,y0
        rep     #4
        mac     x0,y0,a x:(r1)+n1,x0    y:(r4)+,y0
        mac     x0,y0,a (r5)+
        asl     a
        move    a,x:(r6)+
dw_loop1_end:
        rts

; LN src n gb shift dst (n is 56)
op_ln:
        move    x:(r7)+,r0
        move    x:(r7)+,n0
        move    x:(r7)+,r4
        move    x:(r7)+,a
        move    a1,x:ln_shift
        move    x:(r7)+,r1
        ; mean = round(sum x * C / 2^23)
        move    r0,r2
        clr     a
        move    #>LN_C56,y1
        move    #>$800000,x0
        move    x0,a0
        move    x:(r0)+,x0
        rep     n0
        mac     x0,y1,a x:(r0)+,x0
        move    a,y1                    ; the mean, limited
        ; d = x - mean to LN_TEMP; B = 2 S = 2 sum d^2 + 2 n EPS
        move    r2,r0
        move    #LN_TEMP,r3
        clr     b
        move    #>LN_EPS2N,x0
        move    x0,b0
        do      n0,ln_dev_done
        move    x:(r0)+,a
        sub     y1,a
        move    a,x0
        mac     x0,x0,b x0,x:(r3)+
ln_dev_done:
        ; B = S, then normalized to [1/4, 1) by even shifts; A counts them (k)
        asr     b
        clr     a
        move    #>2,x1
        move    #>$200000,x0
ln_right:
        tst     b
        jec     ln_left
        asr     b
        asr     b
        sub     x1,a
        jmp     ln_right
ln_left:
        cmp     x0,b
        jge     ln_normal
        asl     b
        asl     b
        add     x1,a
        jmp     ln_left
ln_normal:
        move    a1,x:scratch+1          ; k
        ; m = B1 in [2^21, 2^23); u = m - 2^21; interval u >> 14, fraction (u & $3fff) << 9
        move    #>$200000,x0
        sub     x0,b
        move    b1,x0
        move    #>$000200,y0
        mpy     x0,y0,a
        move    a1,n2
        move    a1,n6
        move    a0,b
        lsr     b       x:rsqrt_table,r2
        move    b1,x0
        move    r2,b
        move    #>384,y0
        add     y0,b
        move    b1,r6
        move    x:(r2+n2),a
        move    x:(r6+n6),y0
        mac     x0,y0,a
        move    a,y1                    ; R
        ; j = k / 2 + 1: shift left j (or right -j) after d R
        move    x:scratch+1,a
        asr     a       #LN_TEMP,r3
        move    #>1,x0
        add     x0,a
        move    a1,n3                   ; j (16 bits, signed)
        move    x:ln_shift,n2
        move    #ln_rc_table,r2
        move    #>7,b
        move    n2,x0
        sub     x0,b    x:(r2+n2),x1    ; x1 = rounding, b = 7 - shift
        move    b1,n5                   ; right shifts of the output
        tst     a
        jlt     ln_neg
        jeq     ln_zero
        ; j > 0
        do      n0,ln_out_pos
        move    x:(r3)+,x0
        mpy     x0,y1,a
        rep     n3
        asl     a
        move    a,x0                    ; z, limited
        move    y:(r4)+,a               ; beta
        add     x1,a    y:(r4)+,y0      ; + rounding; gamma sqrt(n)
        mac     x0,y0,a
        rep     n5
        asr     a
        move    a,x:(r1)+
ln_out_pos:
        jmp     vm_run
ln_zero:
        do      n0,ln_out_zero
        move    x:(r3)+,x0
        mpy     x0,y1,a
        move    a,x0
        move    y:(r4)+,a
        add     x1,a    y:(r4)+,y0
        mac     x0,y0,a
        rep     n5
        asr     a
        move    a,x:(r1)+
ln_out_zero:
        jmp     vm_run
ln_neg:
        neg     a
        move    a1,n3                   ; -j
        do      n0,ln_out_neg
        move    x:(r3)+,x0
        mpy     x0,y1,a
        rep     n3
        asr     a
        move    a,x0
        move    y:(r4)+,a
        add     x1,a    y:(r4)+,y0
        mac     x0,y0,a
        rep     n5
        asr     a
        move    a,x:(r1)+
ln_out_neg:
        jmp     vm_run
ovl_vm_end:
        nop
        IF      (ovl_vm_end-ovl_vm_start)>$180
        FAIL    'the network overlay exceeds internal P'
        ENDIF

        include 'synhot.asm'

; ================================================================ external code
        org     p:$7000

start:
        movep   #1,x:m_pbc              ; Falcon host port
        movep   #0,x:m_bcr              ; no wait states
        move    #>-1,m0
        move    #>-1,m1
        move    #>-1,m2
        move    #>-1,m3
        move    #>-1,m4
        move    #>-1,m5
        move    #>-1,m6
        move    #>-1,m7
        ; internal RAM is undefined after boot
        move    #0,r0
        move    #0,r4
        clr     a
        do      #256,start_cleared
        move    a1,x:(r0)+
        move    a1,y:(r4)+
start_cleared:
        ; matrix rounding 2^(23 - s) in A0, s = 0..3
        move    #rc_table,r0
        move    #>$800000,a
        move    a1,x:(r0)+
        move    #>$400000,a
        move    a1,x:(r0)+
        move    #>$200000,a
        move    a1,x:(r0)+
        move    #>$100000,a
        move    a1,x:(r0)+
        ; layer norm rounding 2^(6 - s) in A1, s = 0..6 (7 unused)
        move    #ln_rc_table,r0
        move    #>64,a
        do      #7,ln_rc_done
        move    a1,x:(r0)+
        asr     a
        nop
ln_rc_done:

command_loop:
        jclr    #0,x:m_hsr,*
        movep   x:m_hrx,a
        move    a1,x:last_command
        move    a1,x1
        move    #>$000080,y0
        mpy     x1,y0,a                 ; opcode = command >> 16
        move    a1,b

        move    #>CMD_RUN,x0
        cmp     x0,b
        jne     try_write_x
        jsr     ensure_vm
        jsr     command_argument
        move    a1,r7
        nop
        jsr     vm_run
        jmp     command_loop

try_write_x:
        move    #>CMD_WRITE_X,x0
        cmp     x0,b
        jne     try_write_y
        jsr     host_receive
        move    a1,r0
        jsr     host_receive
        tst     a
        jeq     command_loop
        jclr    #1,x:m_hsr,*            ; ready: the host may now write the block unpaced
        movep   #>REPLY_READY,x:m_htx
        do      a1,write_x_done
        jclr    #0,x:m_hsr,*
        movep   x:m_hrx,x:(r0)+
write_x_done:
        jmp     command_loop

try_write_y:
        move    #>CMD_WRITE_Y,x0
        cmp     x0,b
        jne     try_ping
        jsr     host_receive
        move    a1,r4
        jsr     host_receive
        tst     a
        jeq     command_loop
        jclr    #1,x:m_hsr,*            ; ready: the host may now write the block unpaced
        movep   #>REPLY_READY,x:m_htx
        do      a1,write_y_done
        jclr    #0,x:m_hsr,*
        movep   x:m_hrx,y:(r4)+
write_y_done:
        jmp     command_loop

try_ping:
        move    #>CMD_PING,x0
        cmp     x0,b
        jne     try_read_x
        move    #>REPLY_PING,a
        jsr     host_send
        jmp     command_loop

try_read_x:
        move    #>CMD_READ_X,x0
        cmp     x0,b
        jne     try_done
        jsr     command_argument
        move    a1,r0
        nop
        move    x:(r0),a
        jsr     host_send
        jmp     command_loop

try_done:
        move    #>CMD_READ_Y,x0
        cmp     x0,b
        jne     try_done2
        jsr     command_argument
        move    a1,r4
        nop
        move    y:(r4),a
        jsr     host_send
        jmp     command_loop

try_done2:
        move    #>CMD_SYNTH_STEP,x0
        cmp     x0,b
        jne     try_synth_reset
        jsr     ensure_syn
        jsr     command_argument
        jsr     synth_step
        jmp     command_loop

try_synth_reset:
        move    #>CMD_SYNTH_RESET,x0
        cmp     x0,b
        jne     try_done3
        jsr     ensure_syn
        jsr     synth_reset
        jmp     command_loop

try_done3:
        move    #>CMD_DONE,x0
        cmp     x0,b
        jne     command_unknown
        clr     a
        jsr     host_send
finished:
        jmp     command_loop

command_unknown:
        move    #>REPLY_ERROR,a
        jsr     host_send
        jmp     command_loop

; The overlays: copy one into internal P unless it is there already.
ensure_vm:
        move    x:overlay,a
        move    #>OVL_VM,x0
        cmp     x0,a
        jeq     ensure_done
        move    x0,x:overlay
        move    #OVL_VM_LOAD,r0
        move    #>ovl_vm_end-ovl_vm_start,x0
        jmp     load_overlay
ensure_syn:
        move    x:overlay,a
        move    #>OVL_SYN,x0
        cmp     x0,a
        jeq     ensure_done
        move    x0,x:overlay
        move    #OVL_SYN_LOAD,r0
        move    #>ovl_syn_end-ovl_syn_start,x0
load_overlay:
        move    #$0080,r1
        do      x0,lo_copied
        movem   p:(r0)+,a
        movem   a,p:(r1)+
lo_copied:
ensure_done:
        rts

command_argument:
        move    x:last_command,a
        move    #>$00ffff,x0
        and     x0,a
        move    a1,a
        rts

host_send:
        jclr    #1,x:m_hsr,*
        movep   a1,x:m_htx
        rts

host_receive:
        jclr    #0,x:m_hsr,*
        movep   x:m_hrx,a
        rts

; Matrix rows, weights in X, input in Y: r0 rows, r2 input, r1 output,
; n0 = n - 1, y1 = 1.0, x1 = rounding, b = row count.
mvx_s0:
        do      b,mvx_s0_end
        move    r2,r4
        clr     a       x:(r0)+,x0
        move    x1,a0
        mac     x0,y1,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        move    a,x:(r1)+
mvx_s0_end:
        rts
mvx_s1:
        do      b,mvx_s1_end
        move    r2,r4
        clr     a       x:(r0)+,x0
        move    x1,a0
        mac     x0,y1,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        asl     a
        move    a,x:(r1)+
mvx_s1_end:
        rts
mvx_s2:
        do      b,mvx_s2_end
        move    r2,r4
        clr     a       x:(r0)+,x0
        move    x1,a0
        mac     x0,y1,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        asl     a
        asl     a
        move    a,x:(r1)+
mvx_s2_end:
        rts
mvx_s3:
        do      b,mvx_s3_end
        move    r2,r4
        clr     a       x:(r0)+,x0
        move    x1,a0
        mac     x0,y1,a x:(r0)+,x0      y:(r4)+,y0
        rep     n0
        mac     x0,y0,a x:(r0)+,x0      y:(r4)+,y0
        mac     x0,y0,a
        asl     a
        asl     a
        asl     a
        move    a,x:(r1)+
mvx_s3_end:
        rts

; ---------------------------------------------------------------- the machine

; r7 walks the program in X.
vm_run:
        move    x:(r7)+,n6
        move    #op_table,r6
        move    #>OP_COUNT,x0
        move    n6,a
        cmp     x0,a
        jge     vm_fault
        movem   p:(r6+n6),r5
        nop
        jmp     (r5)
vm_fault:
        rts

op_table:
        dc      op_end,op_in,op_out,op_mvy,op_mvx,op_copyxy,op_ln,op_act
        dc      op_add,op_sub,op_mul,op_relu,op_clear,op_push,op_tap,op_dw

op_end:
        rts

; IN dst n
op_in:
        move    x:(r7)+,r0
        move    x:(r7)+,n0
        jclr    #1,x:m_hsr,*            ; ready: the host may now write the block unpaced
        movep   #>REPLY_READY,x:m_htx
        do      n0,op_in_done
        jclr    #0,x:m_hsr,*
        movep   x:m_hrx,x:(r0)+
op_in_done:
        jmp     vm_run

; OUT src n
op_out:
        move    x:(r7)+,r0
        move    x:(r7)+,n0
        do      n0,op_out_done
        jclr    #1,x:m_hsr,*
        movep   x:(r0)+,x:m_htx
op_out_done:
        jmp     vm_run

; MVY src n w rows dst shift
op_mvy:
        move    x:(r7)+,r2
        move    x:(r7)+,a
        ; the input into internal X: one external operand per multiply
        move    #MV_STAGE,r5
        do      a1,op_mvy_staged
        move    x:(r2)+,x0
        move    x0,x:(r5)+
op_mvy_staged:
        move    #MV_STAGE,r2
        move    #>1,x0
        sub     x0,a    x:(r7)+,r4
        move    a1,n0
        move    x:(r7)+,b
        move    x:(r7)+,r1
        move    x:(r7)+,n3
        move    #rc_table,r3
        move    #>ONE,x1
        move    n3,a
        move    x:(r3+n3),y1
        tst     a
        jne     op_mvy_1
        jsr     mvy_s0
        jmp     vm_run
op_mvy_1:
        move    #>1,x0
        cmp     x0,a
        jne     op_mvy_2
        jsr     mvy_s1
        jmp     vm_run
op_mvy_2:
        move    #>2,x0
        cmp     x0,a
        jne     op_mvy_3
        jsr     mvy_s2
        jmp     vm_run
op_mvy_3:
        jsr     mvy_s3
        jmp     vm_run

; MVX src n w rows dst shift (src in Y, rows in X)
op_mvx:
        move    x:(r7)+,r2
        move    x:(r7)+,a
        move    #>1,x0
        sub     x0,a    x:(r7)+,r0
        move    a1,n0
        move    x:(r7)+,b
        move    x:(r7)+,r1
        move    x:(r7)+,n3
        move    #rc_table,r3
        move    #>ONE,y1
        move    n3,a
        move    x:(r3+n3),x1
        tst     a
        jne     op_mvx_1
        jsr     mvx_s0
        jmp     vm_run
op_mvx_1:
        move    #>1,x0
        cmp     x0,a
        jne     op_mvx_2
        jsr     mvx_s1
        jmp     vm_run
op_mvx_2:
        move    #>2,x0
        cmp     x0,a
        jne     op_mvx_3
        jsr     mvx_s2
        jmp     vm_run
op_mvx_3:
        jsr     mvx_s3
        jmp     vm_run

; COPYXY src dst n
op_copyxy:
        move    x:(r7)+,r0
        move    x:(r7)+,r4
        move    x:(r7)+,n0
        do      n0,op_copyxy_done
        move    x:(r0)+,x0
        move    x0,y:(r4)+
op_copyxy_done:
        jmp     vm_run


; ACT desc src dst n
op_act:
        move    x:(r7)+,r2
        move    x:(r7)+,r0
        move    x:(r7)+,r1
        move    x:(r7)+,n0
        move    x:(r2)+,r3              ; values
        move    x:(r2)+,r6              ; slopes
        move    x:(r2)+,x1              ; lo
        move    x:(r2)+,y1              ; hi
        move    x:(r2)+,a
        move    a1,x:act_offset
        move    x:(r2)+,a
        move    a1,x:act_mult
        move    x:(r2)+,a
        move    a1,x:act_loval
        move    x:(r2)+,a
        move    a1,x:act_hival
        move    x:(r2)+,a
        move    a1,x:act_ident
        jsr     act_loop
        jmp     vm_run

; ADD a b dst n, SUB, MUL
op_add:
        jsr     op3_args
        do      n0,op_add_done
        move    x:(r0)+,a
        move    x:(r2)+,x0
        add     x0,a
        move    a,x:(r1)+
op_add_done:
        jmp     vm_run

op_sub:
        jsr     op3_args
        do      n0,op_sub_done
        move    x:(r0)+,a
        move    x:(r2)+,x0
        sub     x0,a
        move    a,x:(r1)+
op_sub_done:
        jmp     vm_run

op_mul:
        jsr     op3_args
        do      n0,op_mul_done
        move    x:(r0)+,x0
        move    x:(r2)+,y0
        mpy     x0,y0,a
        rep     #7
        asl     a
        move    a,x:(r1)+
op_mul_done:
        jmp     vm_run

op3_args:
        move    x:(r7)+,r0
        move    x:(r7)+,r2
        move    x:(r7)+,r1
        move    x:(r7)+,n0
        rts

; RELU src dst n
op_relu:
        move    x:(r7)+,r0
        move    x:(r7)+,r1
        move    x:(r7)+,n0
        clr     b
        do      n0,op_relu_done
        move    x:(r0)+,a
        tst     a
        tlt     b,a
        move    a,x:(r1)+
op_relu_done:
        jmp     vm_run

; CLEAR dst n
op_clear:
        move    x:(r7)+,r1
        move    x:(r7)+,n0
        clr     a
        rep     n0
        move    a,x:(r1)+
        jmp     vm_run

; A ring descriptor (r2): base, size, n, write. Leaves ring_base, ring_size,
; ring_n, and r1 = base + write with m1 = size - 1.
ring_load:
        move    r2,x:ring_desc
        move    x:(r2)+,a
        move    a1,x:ring_base
        move    x:(r2)+,b
        move    b1,x:ring_size
        move    #>1,x0
        sub     x0,b    x:(r2)+,x0
        move    b1,m1
        move    x0,x:ring_n
        move    x:(r2)+,x0
        add     x0,a
        move    a1,r1
        rts

; PUSH ring src
op_push:
        move    x:(r7)+,r2
        move    x:(r7)+,r0
        jsr     ring_load
        move    x:ring_n,n0
        do      n0,op_push_copied
        move    x:(r0)+,x0
        move    x0,x:(r1)+
op_push_copied:
        ; write = r1 - base (r1 wrapped)
        move    r1,a
        move    x:ring_base,x0
        sub     x0,a    x:ring_desc,r2
        move    #>-1,m1
        move    (r2)+
        move    (r2)+
        move    (r2)+
        move    a1,x:(r2)
        jmp     vm_run

; TAP ring delay dst: the frame (1 + delay) n words before the write offset
op_tap:
        move    x:(r7)+,r2
        jsr     ring_load
        move    x:(r7)+,a
        move    x:(r7)+,r0
        move    #>1,x0
        add     x0,a    x:ring_n,y0
        move    a1,x0
        mpy     x0,y0,a
        asr     a
        move    a0,n1
        move    x:ring_n,n0
        move    (r1)-n1
        do      n0,op_tap_done
        move    x:(r1)+,x0
        move    x0,x:(r0)+
op_tap_done:
        move    #>-1,m1
        jmp     vm_run

; DW ring dil w shift dst
op_dw:
        move    x:(r7)+,r2
        jsr     ring_load
        move    x:(r7)+,a               ; dil
        move    x:(r7)+,r4
        move    x:(r7)+,b
        move    b1,x:scratch            ; shift
        move    x:(r7)+,r6
        ; n1 = dil n; the oldest tap of channel 0: write - (1 + 4 dil) n
        move    a1,x0
        move    x:ring_n,y0
        mpy     x0,y0,a
        asr     a
        move    a0,n1                   ; dil n
        move    a0,x0
        move    a0,b
        asl     b
        asl     b                       ; 4 dil n
        move    x:ring_n,x0
        add     x0,b
        move    b1,n5
        move    r1,r5
        move    m1,m5
        nop
        move    (r5)-n5
        move    x:ring_n,n0
        move    #>ONE,x1
        move    x:rc_table,y1           ; shifts 0 and 1 use the same loops' constants
        move    x:scratch,a
        tst     a
        jne     op_dw_1
        jsr     dw_loop0
        jmp     op_dw_done
op_dw_1:
        move    x:rc_table+1,y1
        jsr     dw_loop1
op_dw_done:
        move    #>-1,m1
        move    #>-1,m5
        jmp     vm_run

        include 'synth.asm'

main_code_end:
        nop                             ; tools/dsp/image-header.py checks it stays below the overlays

        end
