#!/usr/bin/perl -w
use strict;
no strict "refs";
use warnings;
use lib './perl';
use Parse::RecDescent;
use Data::Dumper;
use Getopt::Std;
use Cwd 'abs_path';

use isa;
use gas;


my $ROOT = abs_path('./');
my $DEBUG=0;
my $VERBOSE=0;
our $ISA = 'x86';
our $AS  = 'gas';
my $OPT_STRING = 'hpvda:i:o:';
my %OPT;
my $INPUTFILE;
my $OUTPUTFILE;
my $CPP_ARGS='';

# Enable warnings within the Parse::RecDescent module.
$::RD_ERRORS = 1; # Make sure the parser dies when it encounters an error
#$::RD_WARN   = 1; # Enable warnings. This will warn on unused rules &c.
#$::RD_HINT   = 1; # Give out hints to help fix problems.
#$::RD_TRACE  = 1;     # if defined, also trace parsers' behaviour
$::RD_AUTOACTION = q { [@item[0..$#item]] };

sub init
{
	getopts( "$OPT_STRING", \%OPT ) or usage();
	if ($OPT{h}) { usage(); };
	if ($OPT{v}) { $VERBOSE = 1;}
	if ($OPT{d}) { $DEBUG = 1;}

	if (! $ARGV[0]) {
		die "ERROR: Please specify a input file!\n\nCall script with argument -h for help.\n";
	}

	$INPUTFILE = $ARGV[0];
	$CPP_ARGS = $ARGV[1] if ($ARGV[1]);

	if ($INPUTFILE =~ /.pas$/) {
		$INPUTFILE =~ s/\.pas//; 
	} else {
		die "ERROR: Input file must have pas ending!\n";
	}
	if ($OPT{o}) { 
		$OUTPUTFILE = $OPT{o};
	}else {
		$OUTPUTFILE = "$INPUTFILE.s";
	}
	if ($OPT{i}) { 
		$ISA = $OPT{i};
		print "INFO: Using isa $ISA.\n\n" if ($VERBOSE);
	} else {
		print "INFO: No isa specified.\n Using default $ISA.\n\n" if ($VERBOSE);
	}
	if ($OPT{a}) { 
		$AS = $OPT{a};
		print "INFO: Using as $AS.\n\n" if ($VERBOSE);
	} else {
		print "INFO: No as specified.\n Using default $AS.\n\n" if ($VERBOSE);
	}

#	if (-e "$ISA.pl") { 
#		require "$ISA.pl";
#	} else {
#		die "ERROR: Required file $ISA.pl missing!\n";
#	}
#	if (-e "$AS.pl") { 
#		require "$AS.pl";
#	} else {
#		die "ERROR: Required file $AS.pl missing!\n";
#	}
}

sub usage
{
    print <<END;
usage: $0 [-$OPT_STRING]  <INFILE>

Required:
<INFILE>  : Input pas file

Optional:
-h        : this (help) message
-v        : verbose output
-d        : debug mode: prints out the parse tree
-p        : Print out intermediate preprocessed output
-o <FILE> : Output file
-a <ASM>  : Specify different assembler (Default: gas)
-i <ISA>  : Specify different isa (Default: x86)

Example: 
$0 -i x86-64  -a masm -o out.s  myfile.pas

END

exit(0);
}

#=======================================
# GRAMMAR
#=======================================
$main::grammar = <<'_EOGRAMMAR_';
# Terminals
FUNC        : /func/i
LOOP        : /loop/i
ALLOCATE    : /allocate/i
FACTOR      : /factor/i
DEFINE      : /define/i
USE         : /use/i
STOP        : /stop/i
START       : /start/i
LOCAL       : /local/i
TIMER       : /timer/i
INCREMENT   : /increment/i
ALIGN       : /align/i
INT         : /int/i
SINGLE      : /single/i
DOUBLE      : /double/i
INUMBER     : NUMBER
UNUMBER     : NUMBER
SNUMBER     : NUMBER
FNUMBER     : NUMBER
NUMBER      : /[-+]?[0-9]*\.?[0-9]+/
SYMBOL      : /[.A-Z-a-z_][A-Za-z0-9_]*/
REG         : /GPR[0-9]+/i
SREG         : /GPR[0-9]+/i
COMMENT     : /#.*/
{'skip'}

type: SINGLE 
     |DOUBLE
	 |INT

align: ALIGN <commit> NUMBER
{
{FUNC => 'as::align',
 ARGS => ["$item{NUMBER}[1]"]}
}

ASMCODE     : /[A-Za-z1-9.:]+.*/
{
{FUNC => 'as::emit_code',
 ARGS => [$item[1]]}
}

function:  FUNC SYMBOL block
{[
 {FUNC => 'as::function_entry',
  ARGS => [$item{SYMBOL}[1],0]},
 $item{block},
 {FUNC => 'as::function_exit',
  ARGS => [$item{SYMBOL}[1]]}
]}

function_allocate:  FUNC SYMBOL ALLOCATE NUMBER block
{[
 {FUNC => 'as::function_entry',
  ARGS => [$item{SYMBOL}[1],$item{NUMBER}[1]]},
 $item{block},
 {FUNC => 'as::function_exit',
  ARGS => [$item{SYMBOL}[1]]}
]}

loop:  LOOP SYMBOL INUMBER SNUMBER block
{[
{FUNC => 'as::loop_entry',
 ARGS => [$item{SYMBOL}[1],$item{SNUMBER}[1][1]]},
 $item{block},
{FUNC => 'as::loop_exit',
 ARGS => [$item{SYMBOL}[1],$item{INUMBER}[1][1]]}
]}
| LOOP SYMBOL INUMBER SREG block
{[
{FUNC => 'as::loop_entry',
 ARGS => [$item{SYMBOL}[1],$item{SREG}[1]]},
 $item{block},
{FUNC => 'as::loop_exit',
 ARGS => [$item{SYMBOL}[1],$item{INUMBER}[1][1]]}
]}

timer: START TIMER
{
{FUNC => 'isa::start_timer',
 ARGS => []}
}
| STOP TIMER
{
{FUNC => 'isa::stop_timer',
 ARGS => []}
}

mode:  START LOCAL
{
{FUNC => 'as::mode',
 ARGS => [$item[1][1]]}
}
| STOP LOCAL
{
{FUNC => 'as::mode',
 ARGS => [$item[1][1]]}
}

block: '{' expression(s) '}'
{ $item[2] }

define_data: DEFINE type  SYMBOL  NUMBER
{
{FUNC => 'as::define_data',
 ARGS => [$item{SYMBOL}[1], $item{type}[1][1],"$item{NUMBER}[1]"]}
}

expression:  align
            |COMMENT
            |loop
            |timer
            |mode
			|ASMCODE
{ $item[1] }

instruction : define_data
            | align
            | COMMENT
            | mode
            | function
            | function_allocate
{ $item[1] }

startrule: instruction(s)
{ $item[1] }

_EOGRAMMAR_


#=======================================
# MAIN
#=======================================
init();
print "INFO: Calling cpp with arguments $CPP_ARGS.\n" if ($VERBOSE);
my $text = `cpp -x assembler-with-cpp $CPP_ARGS $INPUTFILE.pas`;

if ($OPT{p}) {
	open FILE,">$INPUTFILE.Pas";
	print FILE $text;
	close FILE;
}

open STDOUT,">$OUTPUTFILE";
print "$as::AS->{HEADER}\n";

my $parser = new Parse::RecDescent ($main::grammar)  or die "ERROR: Bad grammar!\n";
my $parse_tree = $parser->startrule($text) or print STDERR "ERROR: Syntax Error\n";
tree_exec($parse_tree);

if ($DEBUG) {
	open FILE,'>parse_tree.txt';
	print FILE Dumper $parse_tree,"\n";
	close FILE;
}

print "$as::AS->{FOOTER}\n";

sub tree_exec 
{
	my $tree = shift;

	foreach my $node (@$tree) {
		if ($node !~ /^skip|^instruction|^expression|^loop/) {
			if (ref($node) eq 'ARRAY')  {
				tree_exec($node);
			}else {
				if (ref($node) eq 'HASH') {
					&{$node->{FUNC}}(@{$node->{ARGS}});
				}
			}
		}
	}
}


