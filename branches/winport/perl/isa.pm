#!/usr/bin/perl 

package isa;

$WORDLENGTH = 8;
$STACKPTR = 'rsp';
$BASEPTR  = 'rbp';

$REG = { GPR1 => 'rax',
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

$ARG = { ARG1 => 'rdi',
	     ARG2 => 'rsi',
	     ARG3 => 'rdx',
	     ARG4 => 'rcx',
	     ARG5 => 'r8',
	     ARG6 => 'r9',
	     ARG7 => '[rbp+16]',
	     ARG8 => '[rbp+24]',
	     ARG9 => '[rbp+32]',
	     ARG10 => '[rbp+40]',
	     ARG11 => '[rbp+48]',
	     ARG12 => '[rbp+56]'};

sub emit_code
{
	my $code = shift;
	$code =~ s/([GF]PR[0-9]+)/$isa::REG->{$1}/g;
	print "$code\n";
}

sub start_timer
{
	print <<END;
push rdi
push rsi
push rbx
push rcx
push rdx
push rdx
xor rax, rax
cpuid
rdtsc
xor rax, rax
cpuid
rdtsc
xor rax, rax
cpuid
rdtsc
pop rdx
xor rax, rax
cpuid
rdtsc
pop rdx
pop rcx
pop rbx
pop rsi
pop rdi
push rax
push rdx
END
}

sub stop_timer
{
	print <<END;
xor rax, rax
cpuid
rdtsc
pop rdi
pop rsi
sub rax, rsi
sbb rdx, rdi
END
}


1;
