#!/usr/bin/env perl
# =======================================================================================
#
#      Filename:  isax86_64.pm
#
#      Description:  Configuration for x86_64 ISA for ptt to pas converter.
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
#
#      This program is free software: you can redistribute it and/or modify it under
#      the terms of the GNU General Public License as published by the Free Software
#      Foundation, either version 3 of the License, or (at your option) any later
#      version.
#
#      This program is distributed in the hope that it will be useful, but WITHOUT ANY
#      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License along with
#      this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =======================================================================================

package isax86_64;

$WORDLENGTH_X86_64 = 8;
$STACKPTR_X86_64 = 'rsp';
$BASEPTR_X86_64  = 'rbp';

$REG_X86_64 = { GPR1 => 'rax',
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

$ARG_X86_64 = {
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
