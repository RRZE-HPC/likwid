#!/usr/bin/perl

package isax86;

$WORDLENGTH_X86 = 4;
$STACKPTR_X86 = 'esp';
$BASEPTR_X86  = 'ebp';

$REG_X86 = { GPR1 => 'eax',
  GPR2 => 'ebx',
  GPR3 => 'ecx',
  GPR4 => 'edx',
  GPR5 => 'esi',
  GPR6 => 'edi',
  FPR1 => 'xmm0',
  FPR2 => 'xmm1',
  FPR3 => 'xmm2',
  FPR4 => 'xmm3',
  FPR5 => 'xmm4',
  FPR6 => 'xmm5',
  FPR7 => 'xmm6',
  FPR8 => 'xmm7'};

$ARG_X86 = {
    ARG1 =>  '[ebp+8]',
    ARG2 =>  '[ebp+12]',
    ARG3 =>  '[ebp+16]',
    ARG4 => '[ebp+20]',
    ARG5 => '[ebp+24]',
    ARG6 => '[ebp+28]',
    ARG7 => '[ebp+32]',
    ARG8 => '[ebp+36]',
    ARG9 => '[ebp+40]',
    ARG10 => '[ebp+44]',
    ARG11 => '[ebp+48]',
    ARG12 => '[ebp+52]',
    ARG13 => '[ebp+56]',
    ARG14 => '[ebp+60]',
    ARG15 => '[ebp+64]',
    ARG16 => '[ebp+68]',
    ARG17 => '[ebp+72]',
    ARG18 => '[ebp+76]'};


1;
