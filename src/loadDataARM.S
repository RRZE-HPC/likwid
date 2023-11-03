.text
.syntax unified
.cpu    cortex-a15
.fpu    neon-vfpv4
.globl  loadDataARM
.align  2
.section    .note.GNU-stack,"",@progbits
.type   loadDataARM,%function
loadDataARM:

push     {r4-r7, lr}
add      r7, sp, #12
push     {r8, r10, r11}

mov     lr, r0              @ copy cache size: r0 -> lr
mov     r2, r1              @ copy ptr: r1 -> r2

.loop:
ldr     r3, [r2]!
subs    lr, lr, 1
bne .loop

pop      {r8, r10, r11}
pop      {r4-r7, pc}
