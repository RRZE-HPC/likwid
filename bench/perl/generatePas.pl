#!/usr/bin/env perl
# =======================================================================================
#
#      Filename:  generatePas.pl
#
#      Description:  Converter from ptt to pas file format.
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
my $desc;
my $prolog='';
my $loop='';
my $increment;
my $isLoop=0;
my $skip=0;
my $multi=0;

my $BenchRoot = $ARGV[0];
my $OutputDirectory = $ARGV[1];
my $TemplateRoot = $ARGV[2];
my $InputFile = "";
if (@ARGV == 4)
{
    $InputFile = $ARGV[3];
}
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
        if ($name =~ /^$/) { continue; }

        $isLoop = 0;
        $skip=0;
        $multi=0;
        $prolog='';
        $loop='';
        $desc='';
        $streams=1;
        my $loads=-1;
        my $stores=-1;
        my $branches=-1;
        my $instr=-1;
        my $loop_instr=-1;
        my $uops = -1;
        open FILE, "<$BenchRoot/$file";
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
        $Vars->{desc} = $desc;

#print Dumper($Vars);

        $tpl->process('bench.tt', $Vars, "$OutputDirectory/$name.pas");
        push(@Testcases,{name    => $name,
                streams => $streams,
                type    => $type,
                stride  => $increment,
                flops   => $flops,
                bytes   => $bytes,
                desc    => $desc,
                loads    => $loads,
                stores    => $stores,
                branches    => $branches,
                instr_const    => $instr,
                instr_loop    => $loop_instr,
                uops    => $uops});
    }
}
#print Dumper(@Testcases);
my @TestcasesSorted = sort {$a->{name} cmp $b->{name}} @Testcases;

my $Vars;
$Vars->{Testcases} = \@TestcasesSorted;
$Vars->{numKernels} = $#TestcasesSorted+1;
$Vars->{allTests} = join('\n',map {$_->{name}." - ".$_->{desc}} @TestcasesSorted);
$tpl->process('testcases.tt', $Vars, "$OutputDirectory/testcases.h");


