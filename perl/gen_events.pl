#!/usr/bin/perl 

use strict;
use warnings;

my $arch;
my $key;
my $eventId;
my $limit;
my $umask;
my $cmask;
my $cfg;
my $num_events=0;
my @events = ();

if (! $ARGV[0] ){
    die "ERROR: Usage: perl ./$0 perfmon_<ARCH>_events.txt perfmon_<ARCH>_events.h\n\n";
}

my $IN_filename  = $ARGV[0];
my $OUT_filename = $ARGV[1];

if ($IN_filename =~ /perfmon_([A-Za-z0-9]+)_events\.txt/){
    $arch = $1;
} else {
    die "The input filename must follow the scheme perfmon_<ARCH>_events.txt\n";
}

open INFILE,"<$IN_filename";

while (<INFILE>) {

    if (/^#/) {
        # Skip comment
    }elsif (/(EVENT_[A-Z0-9_]*)[ ]+(0x[A-F0-9]+)[ ]+([A-Z0-9|]+)/) {
        $eventId = $2;
        $limit = $3;
    } elsif (/UMASK_([A-Z0-9_]*)[ ]*(0x[A-F0-9]+)[ ]*(0x[A-F0-9]+)[ ]*(0x[A-F0-9]+)/) {
        $key   = $1;
        $umask = $2;
        $cfg   = $3;
        $cmask = $4;
        push(@events,{name=>$key,
                limit=>$limit,
                eventId=>$eventId,
                cfg=>$cfg,
                cmask=>$cmask,
                mask=>$umask});
        $num_events++;
    } elsif (/UMASK_([A-Z0-9_]*)[ ]*(0x[A-F0-9]+)/) {
        $key = $1;
        $umask = $2;
        push(@events,{name=>$key,
                limit=>$limit,
                eventId=>$eventId,
                cfg=>0x00,
                cmask=>0x00,
                mask=>$umask});
        $num_events++;
    }
}
close INFILE;

my $ucArch = uc($arch);
my $delim;
$delim = "";


open OUTFILE,">$OUT_filename";
print OUTFILE "/* DONT TOUCH: GENERATED FILE! */\n\n";
print OUTFILE "#define NUM_ARCH_EVENTS_$ucArch $num_events\n\n";
print OUTFILE "static PerfmonEvent  ".$arch."_arch_events[NUM_ARCH_EVENTS_$ucArch] = {\n";

foreach my $event (@events) {

    print OUTFILE <<END;
$delim {\"$event->{name}\",
   \"$event->{limit}\", 
   $event->{eventId},$event->{mask},$event->{cfg},$event->{cmask}}
END
    $delim = ',';
}

print  OUTFILE "};\n";
close OUTFILE;

