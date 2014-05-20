#!/usr/bin/perl

use lib 'util';
use strict;
use warnings;
use lib './perl';
use File::Copy;
use Cwd 'abs_path';
use Data::Dumper;
use Template;

my @Testcases;
my $name;
my $streams;
my $type;
my $flops;
my $bytes;
my $prolog='';
my $loop='';
my $increment;
my $isLoop=0;
my $skip=0;
my $multi=0;

my $BenchRoot = $ARGV[0];
my $OutputDirectory = $ARGV[1];
my $TemplateRoot = $ARGV[2];
my $DEBUG = 0;

my $stream_lookup = {
    STR0 => 'ARG2',
    STR1 => 'ARG3',
    STR2 => 'ARG4',
    STR3 => 'ARG5',
    STR4 => 'ARG6',
    STR5 =>  '[rbp+16]',
    STR6 =>  '[rbp+24]',
    STR7 =>  '[rbp+32]',
    STR8 =>  '[rbp+40]',
    STR9 => '[rbp+48]',
    STR10 => '[rbp+56]',
    STR11 => '[rbp+64]',
    STR12 => '[rbp+72]',
    STR13 => '[rbp+80]',
    STR14 => '[rbp+88]',
    STR15 => '[rbp+96]',
    STR16 => '[rbp+104]',
    STR17 => '[rbp+112]',
    STR18 => '[rbp+120]',
    STR19 => '[rbp+128]',
    STR20 => '[rbp+136]',
    STR21 => '[rbp+144]',
    STR22 => '[rbp+152]',
    STR23 => '[rbp+160]',
    STR24 => '[rbp+168]',
    STR25 => '[rbp+176]',
    STR26 => '[rbp+184]',
    STR27 => '[rbp+192]',
    STR28 => '[rbp+200]',
    STR29 => '[rbp+208]',
    STR30 => '[rbp+216]',
    STR31 => '[rbp+224]',
    STR32 => '[rbp+232]',
    STR33 => '[rbp+240]',
    STR34 => '[rbp+248]',
    STR35 => '[rbp+256]',
    STR36 => '[rbp+264]',
    STR37 => '[rbp+272]',
    STR38 => '[rbp+280]',
    STR39 => '[rbp+288]',
    STR40 => '[rbp+296]'};

opendir (DIR, "./$BenchRoot") or die "Cannot open bench directory: $!\n";
my $tpl = Template->new({
        INCLUDE_PATH => ["$TemplateRoot"]
        });

while (defined(my $file = readdir(DIR))) {
    if ($file !~ /^\./) {
        print "SCANNING $file\n" if ($DEBUG);

        $file =~ /([A-Za-z_0-9]+)\.ptt/;
        $name = $1;

        $isLoop = 0;
        $skip=0;
        $multi=0;
        $prolog='';
        $loop='';
        open FILE, "<$BenchRoot/$file";
        while (<FILE>) {
            my $line = $_;

            if($line =~ /STREAMS[ ]+([0-9]+)/) {
                $streams = $1;
                if ($streams > 10) {
                    $multi = 1;
                }
            } elsif ($line =~ /TYPE[ ]+(SINGLE|DOUBLE)/) {
                $type = $1;
            } elsif ($line =~ /FLOPS[ ]+([0-9]+)/) {
                $flops = $1;
            } elsif ($line =~ /BYTES[ ]+([0-9]+)/) {
                $bytes = $1;
            } elsif ($line =~ /INC[ ]+([0-9]+)/) {
                $increment = $1;
                $skip = 1;
            } elsif ($line =~ /LOOP[ ]+([0-9]+)/) {
                $increment = $1;
                $isLoop = 1;
            } else {
                if ($isLoop) {
                    if($line =~ /SET[ ]+(STR[0-9]+)[ ]+(GPR[0-9]+)/) {
                        $loop .= "#define $1  $2\n";
                        $loop .= "mov $2, $stream_lookup->{$1}\n";
                    } else {
                        $loop .= $line;
                    }
                } else {
                    $prolog .= $line;
                }
            }
        }
        close FILE;

        if (($streams > 5) &&  ($streams < 10)) {
            my $arg = 7;
            foreach my $stream ( 5 .. $streams ) {
                $prolog .= "mov STR$stream, ARG$arg\n";
                $arg++;
            }
        }

        $streams = 'STREAM_'.$streams;
        my $Vars;
        $Vars->{name} = $name;
        $Vars->{prolog} = $prolog;
        $Vars->{increment} = $increment;
        $Vars->{loop} = $loop;
        $Vars->{skip} = $skip;
        $Vars->{multi} = $multi;

#print Dumper($Vars);

        $tpl->process('bench.tt', $Vars, "$OutputDirectory/$name.pas");
        push(@Testcases,{name    => $name,
                         streams => $streams,
                         type    => $type,
                         stride  => $increment,
                         flops   => $flops, 
                         bytes   => $bytes});
    }
}
#print Dumper(@Testcases);
my @TestcasesSorted = sort {$a->{name} cmp $b->{name}} @Testcases;

my $Vars;
$Vars->{Testcases} = \@TestcasesSorted;
$Vars->{numKernels} = $#TestcasesSorted+1;
$Vars->{allTests} = join('\n',map {$_->{name}} @TestcasesSorted);
$tpl->process('testcases.tt', $Vars, "$OutputDirectory/testcases.h");


