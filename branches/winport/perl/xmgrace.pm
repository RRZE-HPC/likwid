#!/usr/bin/perl -w

package xmgrace;
use Exporter;
use Data::Dumper;

@ISA = ('Exporter');
@EXPORT = qw(&xmgrace);

my $X_MAX=0;
my $Y_MAX=0;

sub xmgrace
{
	my ($global_opts, $datasets, $outputs);
	my $global_options = shift;
	my $data_sets = shift;

	$outputs = "-hdevice $global_options->{'device'} -printfile $global_options->{'output file'} ";
	$outputs .= "-saveall $global_options->{'grace output file'}";
	$global_opts = "-autoscale none -world $global_options->{world}";
	open FILE,'>tmp.bat';
	print FILE "title \"$global_options->{'title'}\"\n";
	print FILE "subtitle \"$global_options->{'subtitle'}\"\n";
    print FILE "xaxes scale Logarithmic\n";
	print FILE "xaxis label  \"$global_options->{'xaxis label'}\"\n";
	print FILE "xaxis tick major  2\n";
	print FILE "xaxis tick minor ticks 9\n";
	print FILE "yaxis label  \"$global_options->{'yaxis label'}\"\n";
	print FILE "yaxis tick major $global_options->{'yaxis ticks'}\n";
	print FILE "yaxis tick minor ticks 1\n";
	print FILE "legend  $global_options->{'legend'}\n";
	$datasets = ' ';

	my $num_graphs=0;
	foreach my $dataset (@{$data_sets}) {
		my $tag = "s$num_graphs";
		$datasets .= "-nxy $dataset->{'data file'} ";
		print FILE "$tag legend \"$dataset->{'title'}\"\n";
		print FILE "$tag symbol $dataset->{'symbol'}->{'type'}\n";
		print FILE "$tag symbol size $dataset->{'symbol'}->{'size'}\n";
		print FILE "$tag symbol color $dataset->{'symbol'}->{'color'}\n";
		print FILE "$tag symbol pattern $dataset->{'symbol'}->{'pattern'}\n";
		print FILE "$tag symbol fill color $dataset->{'symbol'}->{'fill color'}\n";
		print FILE "$tag symbol fill pattern $dataset->{'symbol'}->{'fill pattern'}\n";
		print FILE "$tag symbol linewidth $dataset->{'symbol'}->{'linewidth'}\n";
		print FILE "$tag symbol linestyle $dataset->{'symbol'}->{'linestyle'}\n";
		print FILE "$tag line type $dataset->{'line'}->{'type'}\n";
		print FILE "$tag line color $dataset->{'line'}->{'color'}\n";
		print FILE "$tag line linestyle $dataset->{'line'}->{'linestyle'}\n";
		print FILE "$tag line linewidth $dataset->{'line'}->{'linewidth'}\n";
		print FILE "$tag line pattern $dataset->{'line'}->{'pattern'}\n";
		$num_graphs++;
	}

#     print "EXE LINE: gracebat $global_opts $datasets -param tmp.bat $outputs\n"; 
	close FILE;
	system ("gracebat $global_opts $datasets -param tmp.bat $outputs");
	unlink 'tmp.bat';
}

1;

=head1 NAME

xmgrace

=head1 SYNOPSIS

 use xmgrace;
 xmgrace(\%global_options, [\%data_set_options_01, \%data_set_options_02, ...];

=head1 DESCRIPTION

The function xmgrace() is part of the module xmgrace that lets
you generate graphs on the fly in perl. It was written as a front-end
application to Xmgrace for hassle-free generation of graphs. xmgrace()
can be supplied with many of the same options and arguments that can
be given to Xmgrace (the UNIX program that evolved from xmgr). For
more information on Xmgrace see the end of this documentation.
This module is roughly based on the Chart::Graph::Xmgrace module.


=head1 GENERAL EXAMPLE

    use xmgrace;

    xmgrace ({"title"         => "$plot->{title}",
	    "subtitle"        => "$plot->{subtitle}",
	    "legend"          => "0.7,0.25",
	    "output file"     => "$RESULT_TARGET/plot/eps/$plot->{title}.eps",
	    "grace output file" => "$RESULT_TARGET/plot/agr/$plot->{title}.agr",
	    "xaxis label"     => "number of processors",
	    "yaxis label"     => "$PLOT_CONFIG->{$plot->{title}}->{YAXIS}"
	},
	[ { "title"     =>  "$SYSTEM",
	    "data file" =>  "$RESULT_TARGET/plot/data/$plot->{title}.dat",
	    "line" => {
		"type"      => "1",
		"color"     => "1",
		"linewidth" => "2",
		"linestyle" => "1",
		"pattern"   => "1",
	    },
	    "symbol" => {
		"type"      => "2",
		"color"     => "1",
		"pattern"   => "1",
		"linewidth" => "2",
		"linestyle" => "1",
		"size"      => "1",
		"fill pattern" => "1",
		"fill color"=> "1",
	    }
	}]);

=head1 OPTION TABLES

 +----------------------------------------------------------------------------+
 |                              SYMBOL TYPE:                                  |
 +--------+-------+--------+------+-------+--------+--------------------------+
 | SYMBOL | VALUE | SYMBOL | TYPE | VALUE | SYMBOL | VALUE                    |
 +--------+-------+--------+------+-------+--------+--------------------------+
 |  none  |  "0"  |triangle|  up  |  "4"  |  plus  |  "8"                     |
 | circle |  "1"  |triangle| left |  "5"  |   x    |  "9"                     |
 | square |  "2"  |triangle| down |  "6"  |  star  |  "10"                    |
 | diamond|  "3"  |triangle| right|  "7"  |  char  |  "11"                    |
 +--------+-------+--------+------+-------+--------+--------------------------+

 +-----------------------------------------------------------------+
 |                             LINE TYPE                           |
 +------------------------+-------+------------------------+-------+
 |  LINE TYPE             | VALUE |          LINE TYPE     | VALUE |
 +------------------------+-------+------------------------+-------+
 |     none               |  "0"  |          right stairs  |  "3"  |
 |   straight             |  "1"  |            segments    |  "4"  |
 | left stairs            |  "2"  |           3-segments   |  "5"  |
 +------------------------+-------+------------------------+-------+

 +-----------------------------------------------------------------+
 |                             LINE STYLE                          |
 +------------------------+-------+------------------------+-------+
 |  LINE STYLE            | VALUE |         LINE STYLE     | VALUE |
 +------------------------+-------+------------------------+-------+
 |       none             |  "0"  |     solid              |  "1"  |
 |       dotted           |  "2"  |     en-dash            |  "3"  |
 |       em-dash          |  "4"  |     dot-en dash        |  "5"  |
 |       dot-em dash      |  "6"  |     dot-en-dot dash    |  "7"  |
 |       en-dot-en dash   |  "8"  |                        |       |
 +------------------------+-------+------------------------+-------+

 +-----------------------------------------------------------------+
 |                             COLORS                              |
 +-------+-----+-------+-----+--------+-----+-----------+----------+
 | COLOR |VALUE| COLOR |VALUE| COLOR  |VALUE| COLOR     |  VALUE   |
 | white | "0" | blue  | "4" | violet | "8" | indigo    |   "12"   |
 | black | "1" | yellow| "5" | cyan   | "9" | maroon    |   "13"   |
 | red   | "2" | brown | "6" | magenta| "10"| turquoise |   "14"   |
 | green | "3" | grey  | "7" | orange | "11"| dark green|   "15"   |
 +-------+-----+-------+-----+--------+-----+-----------+----------+

=head1 AUTHOR

 Jan Treibig (jan.treibig@rrze.uni-erlangen.de)

=head1 SEE ALSO

 xmgrace(1).


=cut


