package Ptt;

use strict;
use warnings;
use Exporter 'import';

our @EXPORT = qw(ReadPtt);

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

sub ReadPtt {
    my ($InputPttPath, $name) = @_;

    my $type;
    my $flops;
    my $bytes;
    my $increment;
    my $isLoop = 0;
    my $skip=0;
    my $multi=0;
    my $prolog='';
    my $loop='';
    my $desc='';
    my $streams=1;
    my $loads=-1;
    my $stores=-1;
    my $branches=-1;
    my $instr=-1;
    my $loop_instr=-1;
    my $uops = -1;

    open FILE, "<$InputPttPath";
    while (<FILE>) {
        my $line = $_;

        if($line =~ /STREAMS[ ]+([0-9]+)/) {
            $streams = $1;
            if ($streams > 10) {
                $multi = 1;
            }
        } elsif ($line =~ /TYPE[ ]+(SINGLE|DOUBLE|INT)/) {
            $type = $1;
        } elsif ($line =~ /FLOPS[ ]+([0-9]+)/) {
            $flops = $1;
        } elsif ($line =~ /BYTES[ ]+([0-9]+)/) {
            $bytes = $1;
        } elsif ($line =~ /LOADS[ ]+([0-9]+)/) {
            $loads = $1;
        } elsif ($line =~ /STORES[ ]+([0-9]+)/) {
            $stores = $1;
        } elsif ($line =~ /BRANCHES[ ]+([0-9]+)/) {
            $branches = $1;
        } elsif ($line =~ /INSTR_CONST[ ]+([0-9]+)/) {
            $instr = $1;
        } elsif ($line =~ /INSTR_LOOP[ ]+([0-9]+)/) {
            $loop_instr = $1;
        } elsif ($line =~ /UOPS[ ]+([0-9]+)/) {
            $uops = $1;
        } elsif ($line =~ /DESC[ ]+([0-9a-zA-z ,.\-_\(\)\+\*\/=]+)/) {
            $desc = $1;
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

    if (($streams > 5) && ($streams < 10)) {
        my $arg = 7;
        foreach my $stream ( 5 .. $streams ) {
            $prolog .= "mov STR$stream, ARG$arg\n";
            $arg++;
        }
    }

    $streams = 'STREAM_'.$streams;

    my $TestcaseVars;
    $TestcaseVars->{name} = $name;
    $TestcaseVars->{streams} = $streams;
    $TestcaseVars->{type} = $type;
    $TestcaseVars->{stride} = $increment;
    $TestcaseVars->{flops} = $flops;
    $TestcaseVars->{bytes} = $bytes;
    $TestcaseVars->{desc} = $desc;
    $TestcaseVars->{loads} = $loads;
    $TestcaseVars->{stores} = $stores;
    $TestcaseVars->{branches} = $branches;
    $TestcaseVars->{instr_const} = $instr;
    $TestcaseVars->{instr_loop} = $loop_instr;
    $TestcaseVars->{uops} = $uops;

    my $PttVars;
    $PttVars->{name} = $name;
    $PttVars->{prolog} = $prolog;
    $PttVars->{increment} = $increment;
    $PttVars->{loop} = $loop;
    $PttVars->{skip} = $skip;
    $PttVars->{multi} = $multi;
    $PttVars->{desc} = $desc;

    return ($PttVars, $TestcaseVars);
}

1;
