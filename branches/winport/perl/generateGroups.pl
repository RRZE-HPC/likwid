#!/usr/bin/perl

use strict;
use warnings;
use lib './perl';
use File::Copy;
use Cwd 'abs_path';
use Data::Dumper;
use Template;

my $name;
my $shortHelp;

my %groupEnum;
my $GroupRoot = $ARGV[0];
my $OutputDirectory = $ARGV[1];
my $TemplateRoot = $ARGV[2];
my $DEBUG = 0;

my $tpl = Template->new({
        INCLUDE_PATH => ["$TemplateRoot"]
        })|| die Template->error(), "\n";

# First open the architecture directories
opendir (DIR, "./$GroupRoot") or die "Cannot open groups directory: $!\n";
my $rule;
my $metric;

while (defined(my $arch = readdir(DIR))) {
    if ($arch !~ /^\./) {
        print "SCANNING $arch\n" if ($DEBUG);
        if (-d "$GroupRoot/$arch") {

            my $Vars;
            my @groups;
            opendir (ARCHDIR, "$GroupRoot/$arch") or die "Cannot open current directory: $!\n";

            while (defined(my $group = readdir(ARCHDIR))) {

                next unless ($group !~ /^\./);
                print "SCANNING GROUP $group\n" if ($DEBUG);
                my $eventSet;
                my @metrics;
                $Vars->{groups} = [];

                $group =~ /([A-Za-z_0-9]+)\.txt/;
                $name = $1;

                open FILE, "<$GroupRoot/$arch/$group";

                my $isInSet = 0;
                my $isInMetrics = 0;
                my $isInLong = 0;
                my $msg = '';

                while (<FILE>) {
                    my $line = $_;

                    if($line =~ /SHORT[ ]+(.+)/) {
                        $shortHelp = $1;
                    } elsif ($line =~ /EVENTSET/) {
                        $isInSet = 1;
                    } elsif ($line =~ /METRICS/) {
                        $isInSet = 0;
                        $isInMetrics = 1;
                        $eventSet =~ s/,$//;
                    } elsif ($line =~ /LONG/) {
                        $isInSet = 0;
                        $isInMetrics = 0;
                        $isInLong = 1;
                    } else {
                        if ($isInSet) {
                            if ($line =~ /([A-Z0-9]+)[ ]+([A-Z_0-9]+)/) {
                                $eventSet .= "$2:$1,";
                            }
                        } elsif ($isInMetrics)  {
                            if ($line =~ /(.+)[ ]+(.+)/) {
                                $metric = $1;
                                $rule = $2;
                                $rule =~ s/(UPMC[0-9]+)/perfmon_getResult(threadId,"$1")/g;
                                if ($rule !~ /"UPMC/){
                                    $rule =~ s/(PMC[0-9]+)/perfmon_getResult(threadId,"$1")/g;
                                }
                                $rule =~ s/(FIXC[0-9]+)/perfmon_getResult(threadId,"$1")/g;
                                push (@metrics, {label => $metric,
                                        rule  => $rule});
                            }
                        } elsif ($isInLong) {
                            $msg .= $line;
                        }
                    }
                }
                close FILE;
                $msg =~ s/\n/\\n\\\n/g;
                push (@groups, {name => $name,
                        shortHelp => $shortHelp,
                        longHelp  => $msg,
                        eventSet  => $eventSet,
                        numRows   => $#metrics+1,
                        metrics   => \@metrics});

                if (not exists($groupEnum{$name})) {
                    $groupEnum{$name} = 1;
                }

            }

            $Vars->{arch} = $arch;
            my @groupsSorted = sort {$a->{name} cmp $b->{name}} @groups;
            $Vars->{groups} = \@groupsSorted;
            $Vars->{numGroups} = $#groupsSorted+1;


            $tpl->process('group.tt', $Vars, "$OutputDirectory/perfmon_$arch"."_groups.h")|| die $tpl->error(), "\n";
#            print Dumper($Vars);
            closedir ARCHDIR;
        }
    }
}
closedir DIR;

my $Vars;
$Vars->{groups} = \%groupEnum;
$tpl->process('group_types.tt', $Vars, "$OutputDirectory/perfmon_group_types.h")|| die $tpl->error(), "\n";



