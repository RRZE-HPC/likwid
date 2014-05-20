#!/usr/bin/perl 

use strict;
use warnings;

my $arch;
my $key;
my $eventId;
my $limit;
my $umask;
my $num_events=0;
my @events = ();

if (! $ARGV[0] ){
    die "ERROR: Usage: perl ./$0 perfmon_<ARCH>_events.txt\n\n";
}

my $filename = $ARGV[0];

if ($filename =~ /perfmon_([A-Za-z0-9]+)_events\.txt/){
    $arch = $1;
} else {
    die "The input filename must follow the scheme perfmon_<ARCH>_events.txt\n";
}

open INFILE,"<$filename";

while (<INFILE>) {

    if (/(EVENT_[A-Z0-9_]*)[ ]+(0x[A-F0-9]+)[ ]+([A-Z0-9|]+)/) {
        $eventId = $2;
        $limit = $3;
    } elsif (/UMASK_([A-Z0-9_]*)[ ]*(0x[A-F0-9]+)/) {
        $key = $1;
        $umask = $2;
        push(@events,{name=>$key, limit=>$limit, eventId=>$eventId, mask=>$umask});
        $num_events++;
    }
}
close INFILE;

my $ucArch = uc($arch);
my $delim;
$delim = "";


$filename =~ s/\.txt/.h/;
open OUTFILE,">$filename";
print OUTFILE "/* DONT TOUCH: GENERATED FILE! */\n\n";
print OUTFILE "#define NUM_ARCH_EVENTS_$ucArch $num_events\n\n";
print OUTFILE "static PerfmonEvent  ".$arch."_arch_events[NUM_ARCH_EVENTS_$ucArch] = {\n";

foreach my $event (@events) {

    print OUTFILE <<END;
$delim {\"$event->{name}\",
  \"$event->{limit}\", 
   $event->{eventId},$event->{mask}}
END
    $delim = ',';

}

print  OUTFILE "};\n";
close OUTFILE;

