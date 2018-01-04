#!/usr/bin/perl 

package isaarmv7;

$WORDLENGTH_ARMv7 = 4;
$STACKPTR_ARMv7 = 'rsp';
$BASEPTR_ARMv7  = 'rbp';

$REG_ARMv7 = { GPR1 => 'rax',
  GPR2 => 'rbx',
  GPR3 => 'rcx',
  GPR4 => 'rdx',
  GPR5 => 'rsi',
  GPR6 => 'rdi',
  GPR7 => 'r8',
  GPR8 => 'r9',
  GPR9 => 'r10',
  GPR10 => 'r11',
  GPR11 => 'r12',
  GPR12 => 'r13',
  GPR13 => 'r14',
  GPR14 => 'r15',
  FPR1 => 'xmm0',
  FPR2 => 'xmm1',
  FPR3 => 'xmm2',
  FPR4 => 'xmm3',
  FPR5 => 'xmm4',
  FPR6 => 'xmm5',
  FPR7 => 'xmm6',
  FPR8 => 'xmm7',
  FPR9 => 'xmm8',
  FPR10 => 'xmm9',
  FPR11 => 'xmm10',
  FPR12 => 'xmm11',
  FPR13 => 'xmm12',
  FPR14 => 'xmm13',
  FPR15 => 'xmm14',
  FPR16 => 'xmm15'};

$ARG_ARMv7 = {
  ARG1 => 'rdi',
  ARG2 => 'rsi',
  ARG3 => 'rdx',
  ARG4 => 'rcx',
  ARG5 => 'r8',
  ARG6 => 'r9',
  ARG7 =>  '[rbp+16]',
  ARG8 =>  '[rbp+24]',
  ARG9 =>  '[rbp+32]',
  ARG10 => '[rbp+40]',
  ARG11 => '[rbp+48]',
  ARG12 => '[rbp+56]',
  ARG13 => '[rbp+64]',
  ARG14 => '[rbp+72]',
  ARG15 => '[rbp+80]',
  ARG16 => '[rbp+88]',
  ARG17 => '[rbp+96]',
  ARG18 => '[rbp+104]',
  ARG19 => '[rbp+112]',
  ARG20 => '[rbp+120]',
  ARG21 => '[rbp+128]',
  ARG22 => '[rbp+136]',
  ARG23 => '[rbp+144]',
  ARG24 => '[rbp+152]'};

1;
