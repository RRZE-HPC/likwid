#!/usr/bin/perl
# =======================================================================================
#
#      Filename:  gas.pm
#
#      Description:  Implements gas callbacks for likwid asm parser.
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

package as;
use Data::Dumper;
use isax86;
use isax86_64;
use isaarmv8;

$AS = { HEADER     => '.intel_syntax noprefix',
        FOOTER     => ''};

$LOCAL = {};
$MODE = 'GLOBAL';

my $CURRENT_SECTION='NONE';
my $WORDLENGTH;
my $STACKPTR;
my $BASEPTR;
my $REG;
my $ARG;

sub emit_code
{
    my $code = shift;
    $code =~ s/([GF]PR[0-9]+)/$REG->{$1}/g;
    $code =~ s/(ARG[0-9]+)/$ARG->{$1}/g;
    $code =~ s/(LOCAL[0-9]+)/$LOCAL->{$1}/g;
    print "$code\n";
}

sub align
{
    my $number = shift;
    print ".align $number\n";

}

sub mode
{
    $cmd = shift;

    if ($cmd eq 'START') {
        $MODE = 'LOCAL';
    } elsif ($cmd eq 'STOP') {
        $MODE = 'GLOBAL';
    }
}

sub function_entry
{
    my $symbolname = shift;
    my $allocate = shift;
    my $distance;

    foreach ( (0 .. $allocate) ) {
        $distance =  $_ * $WORDLENGTH;
        $LOCAL->{"LOCAL$_"} = "[$BASEPTR-$distance]";
    }

    if($CURRENT_SECTION ne 'text') {
        $CURRENT_SECTION = 'text';
        print ".text\n";
    }

    print ".globl $symbolname\n";
    if ($main::ISA eq 'ARMv7' or $main::ISA eq 'ARMv8') {
        print ".type $symbolname, %function\n";
    } else {
        print ".type $symbolname, \@function\n";
    }
    print "$symbolname :\n";

    if ($main::ISA eq 'x86') {
        print "push ebp\n";
        print "mov ebp, esp\n";
        $distance = $allocate * $WORDLENGTH;
        print "sub  esp, $distance\n" if ($allocate);
        print "push ebx\n";
        print "push esi\n";
        print "push edi\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "push rbp\n";
        print "mov rbp, rsp\n";
        $distance = $allocate * $WORDLENGTH;
        print "sub  rsp, $distance\n" if ($allocate);
        print "push rbx\n";
        print "push r12\n";
        print "push r13\n";
        print "push r14\n";
        print "push r15\n";
    } elsif ($main::ISA eq 'ARMv7') {
        print "push     {r4-r7, lr}\n";
        print "add      r7, sp, #12\n";
        print "push     {r8, r10, r11}\n";
        print "vstmdb   sp!, {d8-d15}\n";
    }
}

sub function_exit
{
    my $symbolname = shift;

    $LOCAL = {};

    if ($main::ISA eq 'x86') {
        print "pop edi\n";
        print "pop esi\n";
        print "pop ebx\n";
        print "mov  esp, ebp\n";
        print "pop ebp\n";
        print "ret\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "pop r15\n";
        print "pop r14\n";
        print "pop r13\n";
        print "pop r12\n";
        print "pop rbx\n";
        print "mov  rsp, rbp\n";
        print "pop rbp\n";
        print "ret\n";
    } elsif ($main::ISA eq 'ARMv7') {
        print "vldmia   sp!, {d8-d15}\n";
        print "pop      {r8, r10, r11}\n";
        print "pop      {r4-r7, pc}\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print ".exit:\n";
        print "\tret\n";
    }
    print ".size $symbolname, .-$symbolname\n";
    print "\n";
}

sub define_data
{
    my $symbolname = shift;
    my $type = shift;
    my $value = shift;

    if($CURRENT_SECTION ne 'data') {
        $CURRENT_SECTION = 'data';
        print ".data\n";
    }
    if  ($main::ISA ne 'ARMv7' and $main::ISA ne 'ARMv8')
    {
        print ".align 64\n";
        print "$symbolname:\n";
        if ($type eq 'DOUBLE') {
            print ".double $value, $value, $value, $value, $value, $value, $value, $value\n"
        } elsif ($type eq 'SINGLE') {
            print ".single $value, $value, $value, $value, $value, $value, $value, $value\n"
        } elsif ($type eq 'INT') {
            print ".int $value, $value\n"
        }
    }
}

sub define_offset
{
    my $symbolname = shift;
    my $type = shift;
    my $value = shift;

    if($CURRENT_SECTION ne 'data') {
        $CURRENT_SECTION = 'data';
        print ".data\n";
    }
    if ($main::ISA eq 'ARMv7' or $main::ISA eq 'ARMv8') {
        print ".align 2\n";
    } else {
        print ".align 16\n";
    }
    print "$symbolname:\n";
    print ".int $value\n";
}


sub loop_entry
{
    my $symbolname = shift;
    my $stopping_criterion = shift;
    $stopping_criterion = $REG->{$stopping_criterion} if( exists $REG->{$stopping_criterion});

    if ($main::ISA eq 'x86') {
        print "xor   eax, eax\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "xor   rax, rax\n";
    } elsif ($main::ISA eq 'ARMv7') {
        print "mov   r4, #0\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print "\tmov   x6, 0\n";
        print ".loop:\n";
        print "\tcmp w0, w6\n";
        print "\tblt .exit\n";
    }
    if ($main::ISA eq 'ARMv7') {
        print ".align 2\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print "\n";
    } else {
        print ".align 16\n";
    }
    if ($MODE eq 'GLOBAL') {
        print "$symbolname :\n";
    } elsif ($main::ISA ne 'ARMv8') {
        print "1:\n";
    }

}


sub loop_exit
{
    my $symbolname = shift;
    my $step = shift;

    if ($main::ISA eq 'x86') {
        print "add eax, $step\n";
        print "cmp eax, edi\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "addq rax, $step\n";
        print "cmpq rax, rdi\n";
    } elsif ($main::ISA eq 'ARMv7') {
        print "add r4, #$step\n";
        print "cmp r4, r0\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print "\tadd x6, x6, $step\n";
        print "\tb .loop\n";
    }
    if ($MODE eq 'GLOBAL') {
        print "jl $symbolname\n";
    }else {
        if ($main::ISA eq 'ARMv7') {
            print "blt 1b\n";
        } elsif ($main::ISA eq 'ARMv8') {
            print "\n";
        } else {
            print "jl 1b\n";
        }
    }
    print "\n";
}

sub isa_init
{
    if ($main::ISA eq 'x86') {
        $WORDLENGTH = $isax86::WORDLENGTH_X86 ;
        $STACKPTR = $isax86::STACKPTR_X86 ;
        $BASEPTR = $isax86::BASEPTR_X86 ;
        $REG = $isax86::REG_X86;
        $ARG = $isax86::ARG_X86 ;
        $AS = { HEADER     => '.intel_syntax noprefix',
                FOOTER     => '' };
    } elsif ($main::ISA eq 'x86-64') {
        $WORDLENGTH = $isax86_64::WORDLENGTH_X86_64;
        $STACKPTR = $isax86_64::STACKPTR_X86_64 ;
        $BASEPTR = $isax86_64::BASEPTR_X86_64 ;
        $REG = $isax86_64::REG_X86_64;
        $ARG = $isax86_64::ARG_X86_64 ;
        $AS = { HEADER     => '.intel_syntax noprefix',
                FOOTER     => '' };
    } elsif ($main::ISA eq 'ARMv7') {
        $WORDLENGTH = $isaarmv7::WORDLENGTH_ARMv7;
        $STACKPTR = $isaarmv7::STACKPTR_ARMv7 ;
        $BASEPTR = $isaarmv7::BASEPTR_ARMv7 ;
        $REG = $isaarmv7::REG_ARMv7;
        $ARG = $isaarmv7::ARG_ARMv7 ;
        $AS = { HEADER     => '.cpu    cortex-a15\n.fpu    neon-v',
                FOOTER     => '' };
    } elsif ($main::ISA eq 'ARMv8') {
        $WORDLENGTH = $isaarmv8::WORDLENGTH_ARMv8;
        $STACKPTR = $isaarmv8::STACKPTR_ARMv8 ;
        $BASEPTR = $isaarmv8::BASEPTR_ARMv8 ;
        $REG = $isaarmv8::REG_ARMv8;
        $ARG = $isaarmv8::ARG_ARMv8 ;
        $AS = { HEADER     => '.cpu    generic+fp+simd',
                FOOTER     => '' };
    }
}


1;
