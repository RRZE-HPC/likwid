#!/usr/bin/perl 

package isappc64;

$WORDLENGTH_PPC64 = 8;
$STACKPTR_PPC64 = '1';
$BASEPTR_PPC64  = '2';

$REG_PPC64 = { GPR1 => '3',
	     GPR2 => '4',
		 GPR3 => '5',
		 GPR4 => '6',
		 GPR5 => '7',
		 GPR6 => '8',
		 GPR7 => '9',
		 GPR8 => '10',
		 GPR9 => '11',
		 GPR10 => '12',
		 GPR11 => '13',
		 GPR12 => '14',
		 GPR13 => '15',
		 GPR14 => '16',
		 GPR15 => '17',
		 GPR16 => '18',
		 GPR17 => '19',
		 GPR18 => '20',
		 GPR19 => '21',
		 GPR20 => '22',
		 GPR21 => '23',
		 GPR22 => '24',
		 GPR23 => '25',
		 GPR24 => '26',
		 GPR25 => '27',
		 GPR26 => '28',
		 GPR27 => '29',
		 GPR28 => '30',
		 GPR29 => '31',
         FPR1 => '0',
	     FPR2 => '1',
		 FPR3 => '2',
		 FPR4 => '3',
		 FPR5 => '4',
		 FPR6 => '5',
		 FPR7 => '6',
		 FPR8 => '7',
		 FPR9 => '8',
		 FPR10 => '9',
		 FPR11 => '10',
		 FPR12 => '11',
		 FPR13 => '12',
		 FPR14 => '13',
		 FPR15 => '14',
		 FPR16 => '15',
		 FPR17 => '16',
		 FPR18 => '17',
		 FPR19 => '18',
		 FPR20 => '19',
		 FPR21 => '20',
		 FPR22 => '21',
		 FPR23 => '22',
		 FPR24 => '23',
		 FPR25 => '24',
		 FPR26 => '25',
		 FPR27 => '26',
		 FPR28 => '27',
		 FPR29 => '28',
		 FPR30 => '29',
		 FPR31 => '30',
		 FPR32 => '31'};

$ARG_PPC64 = { ARG1 => '3',
	     ARG2 => '4',
	     ARG3 => '5',
	     ARG4 => '6',
	     ARG5 => '7',
	     ARG6 => '8',
	     ARG7 => '9',
	     ARG8 => '10',
	     ARG9 => '[rbp+56]'};

sub emit_code
{
	my $code = shift;
	$code =~ s/([GF]PR[0-9]+)/$isa::REG->{$1}/g;
	print "$code\n";
}


1;
