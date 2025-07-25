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
use Template;
use Ptt;

my $TestcaseOutputPath = shift @ARGV;
my $TemplateRoot = shift @ARGV;
my @PttInputPaths = @ARGV;

my $tpl = Template->new({
        INCLUDE_PATH => ["$TemplateRoot"]
    });

my @Testcases;
foreach my $file (@PttInputPaths) {
    $file =~ /([A-Za-z_0-9]+)\.ptt/;
    my $name = $1;
    my ($PttVars, $TestcaseVars) = Ptt::ReadPtt($file, $name);
    push(@Testcases, $TestcaseVars);
}

my @TestcasesSorted = sort {$a->{name} cmp $b->{name}} @Testcases;

my $Vars;
$Vars->{Testcases} = \@TestcasesSorted;
$Vars->{numKernels} = $#TestcasesSorted+1;
$Vars->{allTests} = join('\n',map {$_->{name}." - ".$_->{desc}} @TestcasesSorted);
$tpl->process('testcases.tt', $Vars, $TestcaseOutputPath);
