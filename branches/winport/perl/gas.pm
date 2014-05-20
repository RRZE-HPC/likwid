#!/usr/bin/perl 

package as;
use Data::Dumper;
use isa;

$AS = { HEADER     => '.intel_syntax noprefix',
	    FOOTER     => ''};

$LOCAL = {};
$MODE = 'GLOBAL';

my $CURRENT_SECTION='NONE';

sub emit_code
{
	my $code = shift;
	$code =~ s/([GF]PR[0-9]+)/$isa::REG->{$1}/g;
	$code =~ s/(ARG[0-9]+)/$isa::ARG->{$1}/g;
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
		$distance =  $_ * $isa::WORDLENGTH;
		$LOCAL->{"LOCAL$_"} = "[$isa::BASEPTR-$distance]";
	}

	if($CURRENT_SECTION ne 'text') {
		$CURRENT_SECTION = 'text';
		print ".text\n";
	}

	print ".globl $symbolname\n";
	print ".type $symbolname, \@function\n";
	print "$symbolname :\n";

	if ($main::ISA eq 'x86') {
		print "push ebp\n";
		print "mov ebp, esp\n";
		$distance = $allocate * $isa::WORDLENGTH;
		print "sub  esp, $distance\n" if ($allocate);
		print "push ebx\n";
		print "push esi\n";
		print "push edi\n";
	} elsif ($main::ISA eq 'x86-64') {
		print "push rbp\n";
		print "mov rbp, rsp\n";
		$distance = $allocate * $isa::WORDLENGTH;
		print "sub  rsp, $distance\n" if ($allocate);
		print "push rbx\n";
		print "push r12\n";
		print "push r13\n";
		print "push r14\n";
		print "push r15\n";
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
	} elsif ($main::ISA eq 'x86-64') {
		print "pop r15\n";
		print "pop r14\n";
		print "pop r13\n";
		print "pop r12\n";
		print "pop rbx\n";
		print "mov  rsp, rbp\n";
		print "pop rbp\n";
	}
	print "ret\n";
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
	print ".align 16\n";
	print "$symbolname:\n";
	if ($type eq 'DOUBLE') {
		print ".double $value, $value\n"
	} elsif ($type eq 'SINGLE') {
		print ".single $value, $value, $value, $value\n"
	} elsif ($type eq 'INT') {
		print ".int $value, $value\n"
	}
}

sub loop_entry 
{
	my $symbolname = shift;
	my $stopping_criterion = shift;
	$stopping_criterion = $isa::REG->{$stopping_criterion} if( exists $isa::REG->{$stopping_criterion});

	print "mov   rax, $stopping_criterion\n";
	print "neg   rax\n";
	print ".align 32\n";
	if ($MODE eq 'GLOBAL') {
		print "$symbolname :\n";
	}else {
		print "1:\n";
	}
}


sub loop_exit 
{
	my $symbolname = shift;
	my $step = shift;

	print "add rax, $step\n";
	if ($MODE eq 'GLOBAL') {
		print "js $symbolname\n";
	}else {
		print "js 1b\n";
	}
	print "\n";
}


1;
