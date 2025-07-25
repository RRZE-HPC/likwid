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
use Ptt;

my $BenchRoot = $ARGV[0];
my $OutputDirectory = $ARGV[1];
my $TemplateRoot = $ARGV[2];


opendir (DIR, "./$BenchRoot") or die "Cannot open bench directory: $!\n";
my $tpl = Template->new({
        INCLUDE_PATH => ["$TemplateRoot"]
    });

my @Testcases;
while (defined(my $file = readdir(DIR))) {
    if ($file !~ /^\./) {
        #print "SCANNING $file\n" if ($DEBUG);

        $file =~ /([A-Za-z_0-9]+)\.ptt/;
        my $name = $1;
        if ($name =~ /^$/) { continue; }

        my ($PttVars, $TestcaseVars) = Ptt::ReadPtt("$BenchRoot/$file", $name);

#print Dumper($Vars);

        $tpl->process('bench.tt', $PttVars, "$OutputDirectory/$name.pas");
        push(@Testcases, $TestcaseVars);
    }
}
#print Dumper(@Testcases);
my @TestcasesSorted = sort {$a->{name} cmp $b->{name}} @Testcases;

my $Vars;
$Vars->{Testcases} = \@TestcasesSorted;
$Vars->{numKernels} = $#TestcasesSorted+1;
$Vars->{allTests} = join('\n',map {$_->{name}." - ".$_->{desc}} @TestcasesSorted);
$tpl->process('testcases.tt', $Vars, "$OutputDirectory/testcases.h");
