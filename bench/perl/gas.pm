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
use isaarmv7;
use isaarmv8;

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
    if ($main::ISA eq 'ARMv7' or $main::ISA eq 'ARMv8') {
        print ".globl $symbolname\n";
        print ".type $symbolname, %function\n";
        print "$symbolname :\n";
    } elsif ($main::ISA eq 'x86' or $main::ISA eq 'x86-64') {
        print ".globl $symbolname\n";
        print ".type $symbolname, \@function\n";
    }

    if ($main::ISA eq 'x86') {
        print "$symbolname :\n";
        print "push ebp\n";
        print "mov ebp, esp\n";
        $distance = $allocate * $WORDLENGTH;
        print "sub  esp, $distance\n" if ($allocate);
        print "push ebx\n";
        print "push esi\n";
        print "push edi\n";
    } elsif ($main::ISA eq 'x86-64') {
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
    } elsif ($main::ISA eq 'ARMv7') {
		print "push     {r4-r7, lr}\n";
		print "add      r7, sp, #12\n";
		print "push     {r8, r10, r11}\n";
		print "vstmdb   sp!, {d8-d15}\n";
	} elsif ($main::ISA eq 'ARMv8') {
	    print "stp x29, x30, [sp, -144]!\n";
	    print "mov x29, sp\n";
	    print "stp x19, x20, [sp, 16]\n";
	    print "stp x21, x22, [sp, 32]\n";
	    print "stp x24, x25, [sp, 48]\n";
	    print "stp x26, x27, [sp, 64]\n";
	    print "str x28, [sp, 80]\n";
	    print "str d15, [sp, 88]\n";
	    print "stp d8, d9, [sp, 96]\n";
	    print "stp d10, d11, [sp, 112]\n";
	    print "stp d12, d14, [sp, 128]\n";
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
    } elsif ($main::ISA eq 'ARMv7') {
        print "vldmia   sp!, {d8-d15}\n";
        print "pop      {r8, r10, r11}\n";
        print "pop      {r4-r7, pc}\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print ".exit:\n";
        print "ldp	x19, x20, [sp, 16]\n";
        print "ldp	x21, x22, [sp, 32]\n";
        print "ldp	x24, x25, [sp, 48]\n";
        print "ldp	x26, x27, [sp, 64]\n";
        print "ldr	x28, [sp, 80]\n";
        print "ldr	d15, [sp, 88]\n";
        print "ldp	d8, d9, [sp, 96]\n";
        print "ldp	d10, d11, [sp, 112]\n";
        print "ldp	d12, d14, [sp, 128]\n";
        print "ldp	x29, x30, [sp], 144\n";
        print "ret\n";
        print ".size $symbolname, .-$symbolname\n\n";
    } elsif ($main::ISA eq 'ppc64') {
        print "blr\n";
        print ".size $symbolname, .-$symbolname\n\n";
    }
    print "#if defined(__linux__) && defined(__ELF__)\n";
    print '.section .note.GNU-stack,"",%progbits';
    print "\n";
    print "#endif\n";
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
    if  ($main::ISA ne 'ARMv7' and $main::ISA ne 'ARMv8') {
        print ".align $ALIGN\n";
        print "$symbolname:\n";
        if ($type eq 'DOUBLE') {
            print ".double $value, $value, $value, $value, $value, $value, $value, $value\n"
        } elsif ($type eq 'SINGLE') {
            print ".single $value, $value, $value, $value, $value, $value, $value, $value\n"
        } elsif ($type eq 'HALF') {
            print "._Float16 $value, $value, $value, $value, $value, $value, $value, $value\n"
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
		print ".align $ALIGN\n";
	}
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
    } elsif ($main::ISA eq 'ARMv7') {
        print "mov   r4, #0\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print "mov   x6, 0\n";
        if ($symbolname =~ /sve/)
        {
            print "ptrue p1.d, vl$step\n";
            print "whilelo  p0.d, x6, x0\n";
            print "mov     p2.b, p1.b\n";
            print "mov     p3.b, p1.b\n";
            print "mov     p4.b, p1.b\n";
            print "mov     p5.b, p1.b\n";
            print "mov     p6.b, p1.b\n";
            print "mov     p7.b, p1.b\n";
        }
        print "$symbolname:\n";
    }
    if ($main::ISA ne 'ppc64') {
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
}


sub loop_exit
{
    my $symbolname = shift;
    my $step = shift;

    if ($main::ISA eq 'x86') {
        print "add eax, $step\n";
        print "cmp eax, edi\n";
    } elsif ($main::ISA eq 'x86-64') {
        print "add rax, $step\n";
        print "cmp rax, rdi\n";
    } elsif ($main::ISA eq 'ppc64') {
        print "bdnz $symbolname\n";
    } elsif ($main::ISA eq 'ARMv7') {
        print "add r4, #$step\n";
        print "cmp r4, r0\n";
    } elsif ($main::ISA eq 'ARMv8') {
        print "add x6, x6, #$step\n";
        if ($symbolname =~ /sve/)
        {
            print "whilelo  p0.d, x6, x0\n";
            print "bne $symbolname\n";

        }
        else
        {
            print "cmp x6, x0\n";
            print "blt $symbolname\n";
        }
    }
    if ($MODE eq 'GLOBAL') {
        print "jl $symbolname\n";
    }else {
        if ($main::ISA eq 'ARMv7') {
            print "blt 1b\n";
        } elsif ($main::ISA eq 'ARMv8') {
            #print "bgt 1b\n";
            print "\n";
        } elsif ($main::ISA ne 'ppc64') {
            print "jl 1b\n";
        }
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
    } elsif ($main::ISA eq 'ARMv7') {
        $BASEPTR = $isaarmv7::BASEPTR_ARMv7;
        $WORDLENGTH = $isaarmv7::WORDLENGTH_ARMv7;
        $STACKPTR = $isaarmv7::STACKPTR_ARMv7 ;
        $REG = $isaarmv7::REG_ARMv7;
        $ARG = $isaarmv7::ARG_ARMv7 ;
        $AS = { HEADER     => ".cpu    cortex-a15\n.fpu    neon-vfpv4",
                FOOTER     => '' };
    } elsif ($main::ISA eq 'ARMv8') {
        $BASEPTR = $isaarmv8::BASEPTR_ARMv8;
        $WORDLENGTH = $isaarmv8::WORDLENGTH_ARMv8;
        $STACKPTR = $isaarmv8::STACKPTR_ARMv8 ;
        $REG = $isaarmv8::REG_ARMv8;
        $ARG = $isaarmv8::ARG_ARMv8 ;
        $AS = { HEADER     => ".cpu    generic+fp+simd",
                FOOTER     => '',
                SVE_HEADER => ".arch    armv8.2-a+crc+sve"};
    }
}


1;
