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
use isappc64;

sub init
{
    if ($main::ISA eq 'x86') {
        $AS = { HEADER     => '.intel_syntax noprefix',
                FOOTER     => ''};
    } elsif ($main::ISA eq 'x86_64') {
        $AS = { HEADER     => '.intel_syntax noprefix',
                FOOTER     => ''};
    } elsif ($main::ISA eq 'ppc64') {
        $AS = { HEADER     => '',
                FOOTER     => ''};
    }
}

$LOCAL = {};
$MODE = 'GLOBAL';

my $CURRENT_SECTION='NONE';
my $WORDLENGTH;
my $STACKPTR;
my $BASEPTR;
my $REG;
my $ARG;
my $ALIGN='64';

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

    if ($main::ISA eq 'x86') {
        print ".globl $symbolname\n";
        print ".type $symbolname, \@function\n";
        print "$symbolname :\n";
        print "push ebp\n";
        print "mov ebp, esp\n";
        $distance = $allocate * $WORDLENGTH;
        print "sub  esp, $distance\n" if ($allocate);
        print "push ebx\n";
        print "push esi\n";
        print "push edi\n";
    } elsif ($main::ISA eq 'x86-64') {
        print ".globl $symbolname\n";
        print ".type $symbolname, \@function\n";
        print "$symbolname :\n";
        print "push rbp\n";
        print "mov rbp, rsp\n";
        $distance = $allocate * $WORDLENGTH;
        print "sub  rsp, $distance\n" if ($allocate);
        print "push rbx\n";
        print "push r12\n";
        print "push r13\n";
        print "push r14\n";
        print "push r15\n";
    } elsif ($main::ISA eq 'ppc64') {
        #if ($main::ISA eq 'ppc64') {
            print ".set r0,0; .set SP,1; .set RTOC,2; .set r3,3; .set r4,4;\n";
            print ".set r5,5; .set r6,6; .set r7,7; .set r8,8; .set r9,9; .set r10,10\n";
            print ".set x0,0; .set x1,1; .set x2,2; .set x3,3; .set x4,4\n";
            print ".set x5,5; .set x6,6; .set x7,7; .set x8,8; .set x9,9;\n";
	    print ".set vec0,0; .set vec1,1; .set vec2,2; .set vec3,3;\n";
	    print ".set vec4,4; .set vec5,5; .set vec6,6; .set vec7,7;\n";
	    print ".set vec8,8; .set vec9,9; .set vec10,10; .set vec11,11;\n";
	    print ".set vec12,12;\n";
            #}
        print ".abiversion 2\n";
        print ".section    \".toc\",\"aw\"\n";
        print ".section    \".text\"\n";
        print ".align 2\n";
        print ".globl $symbolname\n";
        print ".type $symbolname, \@function\n";
        print "$symbolname :\n";
        print ".L.$symbolname:\n";
        print ".localentry $symbolname, .-$symbolname\n";

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
        print ".size $symbolname, .-$symbolname\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "pop r15\n";
        print "pop r14\n";
        print "pop r13\n";
        print "pop r12\n";
        print "pop rbx\n";
        print "mov  rsp, rbp\n";
        print "pop rbp\n";
        print "ret\n";
        print ".size $symbolname, .-$symbolname\n";
    } elsif ($main::ISA eq 'ppc64') {
        print "blr\n";
        print ".size $symbolname, .-$symbolname\n";
    }
    #print ".size $symbolname, .-$symbolname\n";
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
    print ".align $ALIGN\n";
    print "$symbolname:\n";
    if ($type eq 'DOUBLE') {
        print ".double $value, $value, $value, $value, $value, $value, $value, $value\n"
    } elsif ($type eq 'SINGLE') {
        print ".single $value, $value, $value, $value, $value, $value, $value, $value\n"
    } elsif ($type eq 'INT') {
        print ".int $value, $value\n"
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
    print ".align $ALIGN\n";
    print "$symbolname:\n";
    print ".int $value\n";
}


sub loop_entry
{
    my $symbolname = shift;
    #my $stopping_criterion = shift;
    my $step = shift;

    if ($main::ISA eq 'x86') {
        print "xor   eax, eax\n";
        print ".align $ALIGN\n";
        if ($MODE eq 'GLOBAL') {
            print "$symbolname :\n";
        } else {
            print "1:\n";
        }
    } elsif ($main::ISA eq 'x86-64') {
        print "xor   rax, rax\n";
        print ".align $ALIGN\n";
        if ($MODE eq 'GLOBAL') {
            print "$symbolname :\n";
        } else {
            print "1:\n";
        }
    } elsif ($main::ISA eq 'ppc64') {
	print "li r0, r10\n";
        print "li r10, $step\n";
	print "divd r10, r3, r10\n";
	print "mtctr r10\n";
	print "li r10, r0\n";
        print "$symbolname:\n";
    }

}


sub loop_exit
{
    my $symbolname = shift;
    my $step = shift;

    if ($main::ISA eq 'x86') {
        print "add eax, $step\n";
        print "cmp eax, edi\n";
        if ($MODE eq 'GLOBAL') {
            print "jl $symbolname\n";
        } else {
            print "jl 1b\n";
        }
        print "\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "add rax, $step\n";
        print "cmp rax, rdi\n";
        if ($MODE eq 'GLOBAL') {
            print "jl $symbolname\n";
        } else {
            print "jl 1b\n";
        }
        print "\n";
    } elsif ($main::ISA eq 'ppc64') {
        print "bdnz $symbolname\n";
    }
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
                FOOTER     => ''};
        $ALIGN = '64';
    } elsif ($main::ISA eq 'x86-64') {
        $WORDLENGTH = $isax86_64::WORDLENGTH_X86_64;
        $STACKPTR = $isax86_64::STACKPTR_X86_64 ;
        $BASEPTR = $isax86_64::BASEPTR_X86_64 ;
        $REG = $isax86_64::REG_X86_64;
        $ARG = $isax86_64::ARG_X86_64 ;
        $AS = { HEADER     => '.intel_syntax noprefix',
                FOOTER     => ''};
        $ALIGN = '64';
    } elsif ($main::ISA eq 'ppc64') {
        $WORDLENGTH = $isappc64::WORDLENGTH_PPC64;
        $STACKPTR = $isappc64::STACKPTR_PPC64 ;
        $BASEPTR = $isappc64::BASEPTR_PPC64 ;
        $REG = $isappc64::REG_PPC64;
        $ARG = $isappc64::ARG_PPC64 ;
        $AS = { HEADER     => '',
                FOOTER     => ''};
        $ALIGN = '16';
    }
}


1;
