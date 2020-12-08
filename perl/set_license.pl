#!/usr/bin/env perl

use strict;
use warnings;
use File::Find;
use File::Copy;

my $mc = '#';
my $cc = ' *';
my $fc = '!';
my $lc = ' *';

my $VERSION   = '<VERSION>';
my $DATE   = '<DATE>';
#my $VERSION   = '4.0';
#my $DATE   = '16.6.2015';
my $YEAR  = '2015';
my $AUTHOR = 'RRZE, University Erlangen-Nuremberg';
my $LICENSE = 'gpl';

my @SKIPLIST = ('ghash.c','ghash.h','loadData.S','loadDataARM.S','bstrlib.c','bstrlib.h', 'calculator_stack.h', 'calculator_stack.c');

sub print_copyright
{
    my $fh = shift;
    my $cm = shift;

    if ($LICENSE eq 'gpl') {
        print $fh <<END;
$cm      Copyright (C) $YEAR $AUTHOR
$cm
$cm      This program is free software: you can redistribute it and/or modify it under
$cm      the terms of the GNU General Public License as published by the Free Software
$cm      Foundation, either version 3 of the License, or (at your option) any later
$cm      version.
$cm
$cm      This program is distributed in the hope that it will be useful, but WITHOUT ANY
$cm      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
$cm      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
$cm
$cm      You should have received a copy of the GNU General Public License along with
$cm      this program.  If not, see <http://www.gnu.org/licenses/>.
$cm
$cm =======================================================================================
END
    }
    elsif ($LICENSE eq 'bsd') {
        print $fh <<END
$cm      Copyright (c) $YEAR, $AUTHOR
$cm      All rights reserved.
$cm
$cm      Redistribution and use in source and binary forms, with or without
$cm      modification, are permitted provided that the following conditions are met:
$cm
$cm      * Redistributions of source code must retain the above copyright notice, this
$cm        list of conditions and the following disclaimer.
$cm
$cm      * Redistributions in binary form must reproduce the above copyright notice,
$cm        this list of conditions and the following disclaimer in the documentation
$cm        and/or other materials provided with the distribution.
$cm
$cm      THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
$cm      ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
$cm      WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
$cm      DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
$cm      FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
$cm      DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
$cm      SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
$cm      CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
$cm      OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
$cm      OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
$cm
$cm =======================================================================================
END
    }
}

sub wanted
{
    my $filename;

    if (scalar(@_)) {
        $filename = shift;
    } else {
        $filename = $_;
    }

    if (($filename =~ /^\./) or (-d $filename) or (-l $filename)) {
        return;
    }

    foreach my $filter ( @SKIPLIST ) {
        if ( $filename eq $filter ) {
            print "SKIP $filename\n";
            return;
        }
    }

    my $in_copyright = 0;
    my $in_header = 0;
    my $style = $cc;
    my $enter = 0;
    open INFILE, "< $filename";
    open OUTFILE, "> $filename.tmp";
    print "Process $filename\n";

    while( <INFILE> ) {
        # Ensure UNIX line ending
        $_ =~ s/\cM\cJ|\cM|\cJ/\n/g;

        if (/\/\*/ and !$enter) {
            $style = $cc;
            $enter = 1;
            $in_header = 1;
            print  OUTFILE "/*\n";
            print  OUTFILE "$style =======================================================================================\n";
            next;
        } elsif (/# =/ and !$enter) {
            $style = $mc;
            $enter = 1;
            $in_header = 1;
            print  OUTFILE "$style =======================================================================================\n";
            next;
        } elsif (/! =/ and !$enter) {
            $style = $fc;
            $enter = 1;
            $in_header = 1;
            print  OUTFILE "$style =======================================================================================\n";
            next;
        } elsif (/#!/ and !$enter) {
            $style = $lc;
            $enter = 1;
            $in_header = 1;
            print  OUTFILE "$_";
            print  OUTFILE "--[[\n";
            print  OUTFILE "$style =======================================================================================\n";
            next;
        } elsif (/\-\-\[\[/ and !$enter) {
            $style = $lc;
            $enter = 1;
            $in_header = 1;
            print  OUTFILE "--[[\n";
            print  OUTFILE "$style =======================================================================================\n";
            next;
        } elsif (!$enter) {
            print "Skip $filename: No header found!\n";
            unlink "$filename.tmp" or die  "Failed to delete file $filename\n";
            return;
        }

        if ($in_header) {
            if(/Filename:[ ]+([A-za-z0-9._\-]+)/) {
                if ($1 ne $filename) {
                    print "File name mismatch: $filename header says $1\n";
                }
                print  OUTFILE "$_";
            } elsif(/Version:/) {
                print OUTFILE  "$style      Version:   $VERSION\n";
            } elsif(/Released:/) {
                print  OUTFILE "$style      Released:  $DATE\n";
            } elsif(/Copyright/) {
                $in_copyright = 1;
                print_copyright(\*OUTFILE, $style);
            } elsif(/# =/ or /! =/) {
                $in_copyright = 0;
                $in_header = 0;
            } elsif (/\*\//) {
                $in_copyright = 0;
                $in_header = 0;
                print  OUTFILE " */\n";
            } elsif (/\]\]$/) {
                $in_copyright = 0;
                $in_header = 0;
                print  OUTFILE "]]\n";
            } elsif (/\* =/ or /\-\-\[\[/) {
                # Skip initial hline
            } else {
                if($in_copyright eq 0) {
                    print  OUTFILE "$_";
                }
            }
        } else {
            print  OUTFILE "$_";
        }
    }

    close INFILE;
    close OUTFILE;

    unlink $filename or die  "Failed to delete file $filename\n";
    copy ("$filename.tmp", $filename) or die "Copy failed\n";
    unlink "$filename.tmp" or die  "Failed to delete file $filename\n";
}


if (defined $ARGV[0]) {
    my $filename = $ARGV[0];
    wanted($filename);
    exit (0);
}

my @directories;
push @directories, 'src';
push @directories, 'bench/src';
push @directories, 'bench/includes';
push @directories, 'examples';

find(\&wanted,  @directories);

# single files
wanted('Makefile');
chdir 'bench';
wanted('Makefile');
wanted('likwid-bench.c');



