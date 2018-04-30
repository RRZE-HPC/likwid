#!/usr/bin/env perl 

package isaarmv8;

$WORDLENGTH_ARMv8 = 4;
$STACKPTR_ARMv8 = 'sp';
$BASEPTR_ARMv8  = 'rbp';

$REG_ARMv8 = { GPR1 => 'x1',
  GPR2 => 'x2',
  GPR3 => 'x3',
  GPR4 => 'x4',
  GPR5 => 'x5',
  GPR6 => 'x6',
  GPR7 => 'x7',
  GPR8 => 'x8',
  GPR9 => 'x9',
  GPR10 => 'x10',
  GPR11 => 'x11',
  GPR12 => 'x12',
  GPR13 => 'x13',
  GPR14 => 'x14',
  GPR15 => 'x15',
  GPR16 => 'x16',
  GPR17 => 'x17',
  GPR18 => 'x18',
  GPR19 => 'x19',
  GPR20 => 'x20',
  GPR21 => 'x21',
  GPR22 => 'x22',
  FPR1 => 'd0',
  FPR2 => 'd1',
  FPR3 => 'd2',
  FPR4 => 'd3',
  FPR5 => 'd4',
  FPR6 => 'd5',
  FPR7 => 'd6',
  FPR8 => 'd7',
  FPR9 => 'd8',
  FPR10 => 'd9',
  FPR11 => 'd10',
  FPR12 => 'd11',
  FPR13 => 'd12',
  FPR14 => 'd13',
  FPR15 => 'd14',
  FPR16 => 'd15'};

$ARG_ARMv8 = {
  ARG1 => 'x0',
  ARG2 => 'x1',
  ARG3 => 'x2',
  ARG4 => 'x3',
  ARG5 => 'x4',
  ARG6 => 'x5',
  ARG7 =>  'x6',
  ARG8 =>  'x7',
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
