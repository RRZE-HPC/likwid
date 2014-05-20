#============================================================= -*-Perl-*-
#
# Template::Plugin::Date
#
# DESCRIPTION
#
#   Plugin to generate formatted date strings.
#
# AUTHORS
#   Thierry-Michel Barral  <kktos@electron-libre.com>
#   Andy Wardley           <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 2000-2007 Thierry-Michel Barral, Andy Wardley.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin::Date;

use strict;
use warnings;
use base 'Template::Plugin';

use POSIX ();

our $VERSION = 2.78;
our $FORMAT  = '%H:%M:%S %d-%b-%Y';    # default strftime() format
our @LOCALE_SUFFIX = qw( .ISO8859-1 .ISO_8859-15 .US-ASCII .UTF-8 );


#------------------------------------------------------------------------
# new(\%options)
#------------------------------------------------------------------------

sub new {
    my ($class, $context, $params) = @_;
    bless {
        $params ? %$params : ()
    }, $class;
}


#------------------------------------------------------------------------
# now()
# 
# Call time() to return the current system time in seconds since the epoch.
#------------------------------------------------------------------------

sub now {
    return time();
}


#------------------------------------------------------------------------
# format()                           
# format($time)
# format($time, $format)
# format($time, $format, $locale)
# format($time, $format, $locale, $gmt_flag)
# format(\%named_params);
# 
# Returns a formatted time/date string for the specified time, $time, 
# (or the current system time if unspecified) using the $format, $locale,
# and $gmt values specified as arguments or internal values set defined
# at construction time).  Specifying a Perl-true value for $gmt will
# override the local time zone and force the output to be for GMT.
# Any or all of the arguments may be specified as named parameters which 
# get passed as a hash array reference as the final argument.
# ------------------------------------------------------------------------

sub format {
    my $self   = shift;
    my $params = ref($_[$#_]) eq 'HASH' ? pop(@_) : { };
    my $time   = shift(@_) || $params->{ time } || $self->{ time } 
                           || $self->now();
    my $format = @_ ? shift(@_) 
                    : ($params->{ format } || $self->{ format } || $FORMAT);
    my $locale = @_ ? shift(@_)
                    : ($params->{ locale } || $self->{ locale });
    my $gmt = @_ ? shift(@_)
            : ($params->{ gmt } || $self->{ gmt });
    my (@date, $datestr);

    if ($time =~ /^\d+$/) {
        # $time is now in seconds since epoch
        if ($gmt) {
            @date = (gmtime($time))[0..6];
        }
        else {
            @date = (localtime($time))[0..6];
        }
    }
    else {
        # if $time is numeric, then we assume it's seconds since the epoch
        # otherwise, we try to parse it as either a 'Y:M:D H:M:S' or a
        # 'H:M:S D:M:Y' string

        my @parts = (split(/(?:\/| |:|-)/, $time));

        if (@parts >= 6) {
            if (length($parts[0]) == 4) {
                # year is first; assume 'Y:M:D H:M:S'
                @date = @parts[reverse 0..5];
            }
            else {
                # year is last; assume 'H:M:S D:M:Y'
                @date = @parts[2,1,0,3..5];
            }
        }

        if (!@date) {
            return (undef, Template::Exception->new('date',
                   "bad time/date string:  " .
                   "expects 'h:m:s d:m:y'  got: '$time'"));
        }
        $date[4] -= 1;     # correct month number 1-12 to range 0-11
        $date[5] -= 1900;  # convert absolute year to years since 1900
        $time = &POSIX::mktime(@date);
    }
    
    if ($locale) {
        # format the date in a specific locale, saving and subsequently
        # restoring the current locale.
        my $old_locale = &POSIX::setlocale(&POSIX::LC_ALL);

        # some systems expect locales to have a particular suffix
        for my $suffix ('', @LOCALE_SUFFIX) {
            my $try_locale = $locale.$suffix;
            my $setlocale = &POSIX::setlocale(&POSIX::LC_ALL, $try_locale);
            if (defined $setlocale && $try_locale eq $setlocale) {
                $locale = $try_locale;
                last;
            }
        }
        $datestr = &POSIX::strftime($format, @date);
        &POSIX::setlocale(&POSIX::LC_ALL, $old_locale);
    }
    else {
        $datestr = &POSIX::strftime($format, @date);
    }

    return $datestr;
}

sub calc {
    my $self = shift;
    eval { require "Date/Calc.pm" };
    $self->throw("failed to load Date::Calc: $@") if $@;
    return Template::Plugin::Date::Calc->new('no context');
}

sub manip {
    my $self = shift;
    eval { require "Date/Manip.pm" };
    $self->throw("failed to load Date::Manip: $@") if $@;
    return Template::Plugin::Date::Manip->new('no context');
}


sub throw {
    my $self = shift;
    die (Template::Exception->new('date', join(', ', @_)));
}


package Template::Plugin::Date::Calc;
use base qw( Template::Plugin );
use vars qw( $AUTOLOAD );
*throw = \&Template::Plugin::Date::throw;

sub AUTOLOAD {
    my $self = shift;
    my $method = $AUTOLOAD;

    $method =~ s/.*:://;
    return if $method eq 'DESTROY';

    my $sub = \&{"Date::Calc::$method"};
    $self->throw("no such Date::Calc method: $method")
        unless $sub;

    &$sub(@_);
}

package Template::Plugin::Date::Manip;
use base qw( Template::Plugin );
use vars qw( $AUTOLOAD );
*throw = \&Template::Plugin::Date::throw;

sub AUTOLOAD {
    my $self = shift;
    my $method = $AUTOLOAD;
    
    $method =~ s/.*:://;
    return if $method eq 'DESTROY';
    
    my $sub = \&{"Date::Manip::$method"};
    $self->throw("no such Date::Manip method: $method")
        unless $sub;
    
    &$sub(@_);
}
    
    
1;

__END__

=head1 NAME

Template::Plugin::Date - Plugin to generate formatted date strings

=head1 SYNOPSIS

    [% USE date %]
    
    # use current time and default format
    [% date.format %]
    
    # specify time as seconds since epoch
    # or as a 'h:m:s d-m-y' or 'y-m-d h:m:s' string
    [% date.format(960973980) %]
    [% date.format('4:20:36 21/12/2000') %]
    [% date.format('2000/12/21 4:20:36') %]
    
    # specify format
    [% date.format(mytime, '%H:%M:%S') %]
    
    # specify locale
    [% date.format(date.now, '%a %d %b %y', 'en_GB') %]
    
    # named parameters 
    [% date.format(mytime, format = '%H:%M:%S') %]
    [% date.format(locale = 'en_GB') %]
    [% date.format(time   = date.now, 
                   format = '%H:%M:%S', 
                   locale = 'en_GB) %]
    
    # specify default format to plugin
    [% USE date(format = '%H:%M:%S', locale = 'de_DE') %]
    
    [% date.format %]
    ...

=head1 DESCRIPTION

The C<Date> plugin provides an easy way to generate formatted time and date
strings by delegating to the C<POSIX> C<strftime()> routine.

The plugin can be loaded via the familiar USE directive.

    [% USE date %]

This creates a plugin object with the default name of 'C<date>'.  An alternate
name can be specified as such:

    [% USE myname = date %]

The plugin provides the C<format()> method which accepts a time value, a
format string and a locale name.  All of these parameters are optional
with the current system time, default format ('C<%H:%M:%S %d-%b-%Y>') and
current locale being used respectively, if undefined.  Default values
for the time, format and/or locale may be specified as named parameters 
in the C<USE> directive.

    [% USE date(format = '%a %d-%b-%Y', locale = 'fr_FR') %]

When called without any parameters, the C<format()> method returns a string
representing the current system time, formatted by C<strftime()> according 
to the default format and for the default locale (which may not be the
current one, if locale is set in the C<USE> directive).

    [% date.format %]

The plugin allows a time/date to be specified as seconds since the epoch,
as is returned by C<time()>.

    File last modified: [% date.format(filemod_time) %]

The time/date can also be specified as a string of the form C<h:m:s d/m/y>
or C<y/m/d h:m:s>.  Any of the characters : / - or space may be used to
delimit fields.

    [% USE day = date(format => '%A', locale => 'en_GB') %]
    [% day.format('4:20:00 9-13-2000') %]  

Output:

    Tuesday

A format string can also be passed to the C<format()> method, and a locale
specification may follow that.

    [% date.format(filemod, '%d-%b-%Y') %]
    [% date.format(filemod, '%d-%b-%Y', 'en_GB') %]

A fourth parameter allows you to force output in GMT, in the case of 
seconds-since-the-epoch input:

    [% date.format(filemod, '%d-%b-%Y', 'en_GB', 1) %]

Note that in this case, if the local time is not GMT, then also specifying
'C<%Z>' (time zone) in the format parameter will lead to an extremely 
misleading result.

Any or all of these parameters may be named.  Positional parameters
should always be in the order C<($time, $format, $locale)>.

    [% date.format(format => '%H:%M:%S') %]
    [% date.format(time => filemod, format => '%H:%M:%S') %]
    [% date.format(mytime, format => '%H:%M:%S') %]
    [% date.format(mytime, format => '%H:%M:%S', locale => 'fr_FR') %]
    [% date.format(mytime, format => '%H:%M:%S', gmt => 1) %]
    ...etc...

The C<now()> method returns the current system time in seconds since the 
epoch.  

    [% date.format(date.now, '%A') %]

The C<calc()> method can be used to create an interface to the C<Date::Calc>
module (if installed on your system).

    [% calc = date.calc %]
    [% calc.Monday_of_Week(22, 2001).join('/') %]

The C<manip()> method can be used to create an interface to the C<Date::Manip>
module (if installed on your system).

    [% manip = date.manip %]
    [% manip.UnixDate("Noon Yesterday","%Y %b %d %H:%M") %]

=head1 AUTHORS

Thierry-Michel Barral wrote the original plugin.

Andy Wardley provided some minor
fixups/enhancements, a test script and documentation.

Mark D. Mills cloned C<Date::Manip> from the C<Date::Calc> sub-plugin.

=head1 COPYRIGHT

Copyright (C) 2000-2007 Thierry-Michel Barral, Andy Wardley.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<POSIX>

