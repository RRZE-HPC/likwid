#============================================================= -*-Perl-*-
#
# Template::Test
#
# DESCRIPTION
#   Module defining a test harness which processes template input and
#   then compares the output against pre-define expected output.
#   Generates test output compatible with Test::Harness.  This was 
#   originally the t/texpect.pl script.
#
# AUTHOR
#   Andy Wardley   <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Test;

use strict;
use warnings;
use Template qw( :template );
use Exporter;

our $VERSION = 2.75;
our $DEBUG   = 0;
our @ISA     = qw( Exporter );
our @EXPORT  = qw( ntests ok is match flush skip_all test_expect callsign banner );
our @EXPORT_OK = ( 'assert' );
our %EXPORT_TAGS = ( all => [ @EXPORT_OK, @EXPORT ] );
$| = 1;

our $REASON   = 'not applicable on this platform';
our $NO_FLUSH = 0;
our $EXTRA    = 0;   # any extra tests to come after test_expect()
our $PRESERVE = 0    # don't mangle newlines in output/expect
    unless defined $PRESERVE;

our ($loaded, %callsign);

# always set binmode on Win32 machines so that any output generated
# is true to what we expect 
$Template::BINMODE = ($^O eq 'MSWin32') ? 1 : 0;

my @results = ();
my ($ntests, $ok_count);
*is = \&match;

END {
    # ensure flush() is called to print any cached results 
    flush();
}


#------------------------------------------------------------------------
# ntests($n)
#
# Declare how many (more) tests are expected to come.  If ok() is called 
# before ntests() then the results are cached instead of being printed
# to STDOUT.  When ntests() is called, the total number of tests 
# (including any cached) is known and the "1..$ntests" line can be
# printed along with the cached results.  After that, calls to ok() 
# generated printed output immediately.
#------------------------------------------------------------------------

sub ntests {
    $ntests = shift;
    # add any pre-declared extra tests, or pre-stored test @results, to 
    # the grand total of tests
    $ntests += $EXTRA + scalar @results;     
    $ok_count = 1;
    print $ntests ? "1..$ntests\n" : "1..$ntests # skip $REASON\n";
    # flush cached results
    foreach my $pre_test (@results) {
        ok(@$pre_test);
    }
}


#------------------------------------------------------------------------
# ok($truth, $msg)
#
# Tests the value passed for truth and generates an "ok $n" or "not ok $n"
# line accordingly.  If ntests() hasn't been called then we cached 
# results for later, instead.
#------------------------------------------------------------------------

sub ok {
    my ($ok, $msg) = @_;

    # cache results if ntests() not yet called
    unless ($ok_count) {
        push(@results, [ $ok, $msg ]);
        return $ok;
    }

    $msg = defined $msg ? " - $msg" : '';
    if ($ok) {
        print "ok ", $ok_count++, "$msg\n";
    }
    else {
        print STDERR "FAILED $ok_count: $msg\n" if defined $msg;
        print "not ok ", $ok_count++, "$msg\n";
    }
}



#------------------------------------------------------------------------
# assert($truth, $error)
#
# Test value for truth, die if false.
#------------------------------------------------------------------------

sub assert {
    my ($ok, $err) = @_;
    return ok(1) if $ok;

    # failed
    my ($pkg, $file, $line) = caller();
    $err ||= "assert failed";
    $err .= " at $file line $line\n";
    ok(0);
    die $err;
}

#------------------------------------------------------------------------
# match( $result, $expect )
#------------------------------------------------------------------------

sub match {
    my ($result, $expect, $msg) = @_;
    my $count = $ok_count ? $ok_count : scalar @results + 1;

    # force stringification of $result to avoid 'no eq method' overload errors
    $result = "$result" if ref $result;    

    if ($result eq $expect) {
        return ok(1, $msg);
    }
    else {
        print STDERR "FAILED $count:\n  expect: [$expect]\n  result: [$result]\n";
        return ok(0, $msg);
    }
}


#------------------------------------------------------------------------
# flush()
#
# Flush any tests results.
#------------------------------------------------------------------------

sub flush {
    ntests(0)
    unless $ok_count || $NO_FLUSH;
}


#------------------------------------------------------------------------
# skip_all($reason)
#
# Skip all tests, setting $REASON to contain any message passed.  Calls
# exit(0) which triggers flush() which generates a "1..0 # $REASON"
# string to keep to test harness happy.
#------------------------------------------------------------------------

sub skip_all {
    $REASON = join('', @_);
    exit(0);
}


#------------------------------------------------------------------------
# test_expect($input, $template, \%replace)
#
# This is the main testing sub-routine.  The $input parameter should be a 
# text string or a filehandle reference (e.g. GLOB or IO::Handle) from
# which the input text can be read.  The input should contain a number 
# of tests which are split up and processed individually, comparing the 
# generated output against the expected output.  Tests should be defined
# as follows:
#
#   -- test --
#   test input
#   -- expect --
#   expected output
# 
#   -- test --
#    etc...
#
# The number of tests is determined and ntests() is called to generate 
# the "0..$n" line compatible with Test::Harness.  Each test input is
# then processed by the Template object passed as the second parameter,
# $template.  This may also be a hash reference containing configuration
# which are used to instantiate a Template object, or may be left 
# undefined in which case a default Template object will be instantiated.
# The third parameter, also optional, may be a reference to a hash array
# defining template variables.  This is passed to the template process()
# method.
#------------------------------------------------------------------------

sub test_expect {
    my ($src, $tproc, $params) = @_;
    my ($input, @tests);
    my ($output, $expect, $match);
    my $count = 0;
    my $ttprocs;

    # read input text
    eval {
        local $/ = undef;
        $input = ref $src ? <$src> : $src;
    };
    if ($@) {
        ntests(1); ok(0);
        warn "Cannot read input text from $src\n";
        return undef;
    }

    # remove any comment lines
    $input =~ s/^#.*?\n//gm;

    # remove anything before '-- start --' and/or after '-- stop --'
    $input = $' if $input =~ /\s*--\s*start\s*--\s*/;
    $input = $` if $input =~ /\s*--\s*stop\s*--\s*/;

    @tests = split(/^\s*--\s*test\s*--\s*\n/im, $input);

    # if the first line of the file was '--test--' (optional) then the 
    # first test will be empty and can be discarded
    shift(@tests) if $tests[0] =~ /^\s*$/;

    ntests(3 + scalar(@tests) * 2);

    # first test is that Template loaded OK, which it did
    ok(1, 'running test_expect()');

    # optional second param may contain a Template reference or a HASH ref
    # of constructor options, or may be undefined
    if (ref($tproc) eq 'HASH') {
        # create Template object using hash of config items
        $tproc = Template->new($tproc)
            || die Template->error(), "\n";
    }
    elsif (ref($tproc) eq 'ARRAY') {
        # list of [ name => $tproc, name => $tproc ], use first $tproc
        $ttprocs = { @$tproc };
        $tproc   = $tproc->[1];
    }
    elsif (! ref $tproc) {
        $tproc = Template->new()
            || die Template->error(), "\n";
    }
    # otherwise, we assume it's a Template reference

    # test: template processor created OK
    ok($tproc, 'template processor is engaged');

    # third test is that the input read ok, which it did
    ok(1, 'input read and split into ' . scalar @tests . ' tests');

    # the remaining tests are defined in @tests...
    foreach $input (@tests) {
        $count++;
        my $name = '';
        
        if ($input =~ s/^\s*-- name:? (.*?) --\s*\n//im) {
            $name = $1; 
        }
        else {
            $name = "template text $count";
        }

        # split input by a line like "-- expect --"
        ($input, $expect) = 
            split(/^\s*--\s*expect\s*--\s*\n/im, $input);
        $expect = '' 
            unless defined $expect;

        $output = '';

        # input text may be prefixed with "-- use name --" to indicate a
        # Template object in the $ttproc hash which we should use
        if ($input =~ s/^\s*--\s*use\s+(\S+)\s*--\s*\n//im) {
            my $ttname = $1;
            my $ttlookup;
            if ($ttlookup = $ttprocs->{ $ttname }) {
                $tproc = $ttlookup;
            }
            else {
                warn "no such template object to use: $ttname\n";
            }
        }

        # process input text
        $tproc->process(\$input, $params, \$output) || do {
            warn "Template process failed: ", $tproc->error(), "\n";
            # report failure and automatically fail the expect match
            ok(0, "$name process FAILED: " . subtext($input));
            ok(0, '(obviously did not match expected)');
            next;
        };

        # processed OK
        ok(1, "$name processed OK: " . subtext($input));

        # another hack: if the '-- expect --' section starts with 
        # '-- process --' then we process the expected output 
        # before comparing it with the generated output.  This is
        # slightly twisted but it makes it possible to run tests 
        # where the expected output isn't static.  See t/date.t for
        # an example.

        if ($expect =~ s/^\s*--+\s*process\s*--+\s*\n//im) {
            my $out;
            $tproc->process(\$expect, $params, \$out) || do {
                warn("Template process failed (expect): ", 
                     $tproc->error(), "\n");
                # report failure and automatically fail the expect match
                ok(0, "failed to process expected output ["
                   . subtext($expect) . ']');
                next;
            };
            $expect = $out;
        };      
        
        # strip any trailing blank lines from expected and real output
        foreach ($expect, $output) {
            s/[\n\r]*\Z//mg;
        }
        
        $match = ($expect eq $output) ? 1 : 0;
        if (! $match || $DEBUG) {
            print "MATCH FAILED\n"
                unless $match;
            
            my ($copyi, $copye, $copyo) = ($input, $expect, $output);
            unless ($PRESERVE) {
                foreach ($copyi, $copye, $copyo) {
                    s/\n/\\n/g;
                }
            }
            printf(" input: [%s]\nexpect: [%s]\noutput: [%s]\n", 
                   $copyi, $copye, $copyo);
        }
        
        ok($match, $match ? "$name matched expected" : "$name did not match expected");
    };
}

#------------------------------------------------------------------------
# callsign()
#
# Returns a hash array mapping lower a..z to their phonetic alphabet 
# equivalent.
#------------------------------------------------------------------------

sub callsign {
    my %callsign;
    @callsign{ 'a'..'z' } = qw( 
        alpha bravo charlie delta echo foxtrot golf hotel india 
        juliet kilo lima mike november oscar papa quebec romeo 
        sierra tango umbrella victor whisky x-ray yankee zulu );
    return \%callsign;
}


#------------------------------------------------------------------------
# banner($text)
# 
# Prints a banner with the specified text if $DEBUG is set.
#------------------------------------------------------------------------

sub banner {
    return unless $DEBUG;
    my $text = join('', @_);
    my $count = $ok_count ? $ok_count - 1 : scalar @results;
    print "-" x 72, "\n$text ($count tests completed)\n", "-" x 72, "\n";
}


sub subtext {
    my $text = shift;
    $text =~ s/\s*$//sg;
    $text = substr($text, 0, 32) . '...' if length $text > 32;
    $text =~ s/\n/\\n/g;
    return $text;
}


1;

__END__

=head1 NAME

Template::Test - Module for automating TT2 test scripts

=head1 SYNOPSIS

    use Template::Test;
   
    $Template::Test::DEBUG = 0;   # set this true to see each test running
    $Template::Test::EXTRA = 2;   # 2 extra tests follow test_expect()...
    
    # ok() can be called any number of times before test_expect
    ok( $true_or_false )
    
    # test_expect() splits $input into individual tests, processes each 
    # and compares generated output against expected output
    test_expect($input, $template, \%replace );
    
    # $input is text or filehandle (e.g. DATA section after __END__)
    test_expect( $text );
    test_expect( \*DATA );
    
    # $template is a Template object or configuration hash
    my $template_cfg = { ... };
    test_expect( $input, $template_cfg );
    my $template_obj = Template->new($template_cfg);
    test_expect( $input, $template_obj );
    
    # $replace is a hash reference of template variables
    my $replace = {
        a => 'alpha',
        b => 'bravo'
    };
    test_expect( $input, $template, $replace );
    
    # ok() called after test_expect should be declared in $EXTRA (2)
    ok( $true_or_false )   
    ok( $true_or_false )   

=head1 DESCRIPTION

The C<Template::Test> module defines the L<test_expect()> and other related
subroutines which can be used to automate test scripts for the
Template Toolkit.  See the numerous tests in the F<t> sub-directory of
the distribution for examples of use.

=head1 PACKAGE SUBROUTINES

=head2 text_expect()

The C<test_expect()> subroutine splits an input document into a number
of separate tests, processes each one using the Template Toolkit and
then compares the generated output against an expected output, also
specified in the input document.  It generates the familiar 
C<ok>/C<not ok> output compatible with C<Test::Harness>.

The test input should be specified as a text string or a reference to
a filehandle (e.g. C<GLOB> or C<IO::Handle>) from which it can be read.  In 
particular, this allows the test input to be placed after the C<__END__>
marker and read via the C<DATA> filehandle.

    use Template::Test;
    
    test_expect(\*DATA);
    
    __END__
    # this is the first test (this is a comment)
    -- test --
    blah blah blah [% foo %]
    -- expect --
    blah blah blah value_of_foo
    
    # here's the second test (no surprise, so is this)
    -- test --
    more blah blah [% bar %]
    -- expect --
    more blah blah value_of_bar

Blank lines between test sections are generally ignored.  Any line starting
with C<#> is treated as a comment and is ignored.

The second and third parameters to C<test_expect()> are optional.  The second
may be either a reference to a Template object which should be used to 
process the template fragments, or a reference to a hash array containing
configuration values which should be used to instantiate a new Template
object.

    # pass reference to config hash
    my $config = {
        INCLUDE_PATH => '/here/there:/every/where',
        POST_CHOMP   => 1,
    };
    test_expect(\*DATA, $config);
    
    # or create Template object explicitly
    my $template = Template->new($config);
    test_expect(\*DATA, $template);

The third parameter may be used to reference a hash array of template
variable which should be defined when processing the tests.  This is
passed to the L<Template> L<process()|Template#process()> method.

    my $replace = {
        a => 'alpha',
        b => 'bravo',
    };
    
    test_expect(\*DATA, $config, $replace);

The second parameter may be left undefined to specify a default L<Template>
configuration.

    test_expect(\*DATA, undef, $replace);

For testing the output of different L<Template> configurations, a
reference to a list of named L<Template> objects also may be passed as
the second parameter.

    my $tt1 = Template->new({ ... });
    my $tt2 = Template->new({ ... });
    my @tts = [ one => $tt1, two => $tt1 ];

The first object in the list is used by default.  Other objects may be 
switched in with a 'C<-- use $name -->' marker.  This should immediately 
follow a 'C<-- test -->' line.  That object will then be used for the rest 
of the test, or until a different object is selected.

    -- test --
    -- use one --
    [% blah %]
    -- expect --
    blah, blah
    
    -- test --
    still using one...
    -- expect --
    ...
    
    -- test --
    -- use two --
    [% blah %]
    -- expect --
    blah, blah, more blah

The C<test_expect()> sub counts the number of tests, and then calls L<ntests()> 
to generate the familiar "C<1..$ntests\n>" test harness line.  Each 
test defined generates two test numbers.  The first indicates 
that the input was processed without error, and the second that the 
output matches that expected. 

Additional test may be run before C<test_expect()> by calling L<ok()>. These
test results are cached until L<ntests()> is called and the final number of
tests can be calculated. Then, the "C<1..$ntests>" line is output, along with
"C<ok $n>" / "C<not ok $n>" lines for each of the cached test result.
Subsequent calls to L<ok()> then generate an output line immediately.

    my $something = SomeObject->new();
    ok( $something );
    
    my $other = AnotherThing->new();
    ok( $other );
    
    test_expect(\*DATA);

If any tests are to follow after C<test_expect()> is called then these 
should be pre-declared by setting the C<$EXTRA> package variable.  This
value (default: C<0>) is added to the grand total calculated by L<ntests()>.
The results of the additional tests are also registered by calling L<ok()>.

    $Template::Test::EXTRA = 2;
    
    # can call ok() any number of times before test_expect()
    ok( $did_that_work );             
    ok( $make_sure );
    ok( $dead_certain ); 
    
    # <some> number of tests...
    test_expect(\*DATA, $config, $replace);
    
    # here's those $EXTRA tests
    ok( defined $some_result && ref $some_result eq 'ARRAY' );
    ok( $some_result->[0] eq 'some expected value' );

If you don't want to call C<test_expect()> at all then you can call
C<ntests($n)> to declare the number of tests and generate the test 
header line.  After that, simply call L<ok()> for each test passing 
a true or false values to indicate that the test passed or failed.

    ntests(2);
    ok(1);
    ok(0);

If you're really lazy, you can just call L<ok()> and not bother declaring
the number of tests at all.  All tests results will be cached until the
end of the script and then printed in one go before the program exits.

    ok( $x );
    ok( $y );

You can identify only a specific part of the input file for testing
using the 'C<-- start -->' and 'C<-- stop -->' markers.  Anything before the 
first 'C<-- start -->' is ignored, along with anything after the next 
'C<-- stop -->' marker.

    -- test --
    this is test 1 (not performed)
    -- expect --
    this is test 1 (not performed)
    
    -- start --
    
    -- test --
    this is test 2
    -- expect --
    this is test 2
        
    -- stop --
    
    ...

=head2 ntests()

Subroutine used to specify how many tests you're expecting to run.

=head2 ok($test)

Generates an "C<ok $n>" or "C<not ok $n>" message if C<$test> is true or false.

=head2 not_ok($test)

The logical inverse of L<ok()>. Prints an "C<ok $n>" message is C<$test> is
I<false> and vice-versa.

=head2 callsign()

For historical reasons and general utility, the module also defines a
C<callsign()> subroutine which returns a hash mapping the letters C<a>
to C<z> to their phonetic alphabet equivalent (e.g. radio callsigns). 
This is used by many of the test scripts as a known source of variable values.

    test_expect(\*DATA, $config, callsign());

=head2 banner()

This subroutine prints a simple banner including any text passed as parameters.
The C<$DEBUG> variable must be set for it to generate any output.

    banner('Testing something-or-other');

example output:

    #------------------------------------------------------------
    # Testing something-or-other (27 tests completed)
    #------------------------------------------------------------

=head1 PACKAGE VARIABLES

=head2 $DEBUG

The $DEBUG package variable can be set to enable debugging mode.

=head2 $PRESERVE

The $PRESERVE package variable can be set to stop the test_expect()
from converting newlines in the output and expected output into
the literal strings '\n'. 

=head1 HISTORY

This module started its butt-ugly life as the C<t/texpect.pl> script.  It
was cleaned up to became the C<Template::Test> module some time around
version 0.29.  It underwent further cosmetic surgery for version 2.00
but still retains some remarkable rear-end resemblances.

Since then the C<Test::More> and related modules have appeared on CPAN
making this module mostly, but not entirely, redundant.

=head1 BUGS / KNOWN "FEATURES"

Imports all methods by default.  This is generally a Bad Thing, but
this module is only used in test scripts (i.e. at build time) so a) we
don't really care and b) it saves typing.

The line splitter may be a bit dumb, especially if it sees lines like
C<-- this --> that aren't supposed to be special markers.  So don't do that.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
