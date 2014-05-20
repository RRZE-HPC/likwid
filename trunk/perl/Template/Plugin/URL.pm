#============================================================= -*-Perl-*-
#
# Template::Plugin::URL
#
# DESCRIPTION
#   Template Toolkit Plugin for constructing URL's from a base stem 
#   and adaptable parameters.
#
# AUTHOR
#   Andy Wardley   <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 2000-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin::URL;

use strict;
use warnings;
use base 'Template::Plugin';

our $VERSION = 2.74;
our $JOINT   = '&amp;';


#------------------------------------------------------------------------
# new($context, $baseurl, \%url_params)
#
# Constructor method which returns a sub-routine closure for constructing
# complex URL's from a base part and hash of additional parameters.
#------------------------------------------------------------------------

sub new {
    my ($class, $context, $base, $args) = @_;
    $args ||= { };

    return sub {
        my $newbase = shift unless ref $_[0] eq 'HASH';
        my $newargs = shift || { };
        my $combo   = { %$args, %$newargs };
        my $urlargs = join($JOINT,
                           map  { args($_, $combo->{ $_ }) }
                           grep { defined $combo->{ $_ } && length $combo->{ $_ } }
                           sort keys %$combo);

        my $query = $newbase || $base || '';
        $query .= '?' if length $query && length $urlargs;
        $query .= $urlargs if length $urlargs;

        return $query
    }
}


sub args {
    my ($key, $val) = @_;
    $key = escape($key);
    
    return map {
        "$key=" . escape($_);
    } ref $val eq 'ARRAY' ? @$val : $val;
    
}

#------------------------------------------------------------------------
# escape($url)
# 
# URL-encode data.  Borrowed with minor modifications from CGI.pm.  
# Kudos to Lincold Stein.
#------------------------------------------------------------------------

sub escape {
    my $toencode = shift;
    return undef unless defined($toencode);
    $toencode=~s/([^a-zA-Z0-9_.-])/uc sprintf("%%%02x",ord($1))/eg;
    return $toencode;
}

1;

__END__

=head1 NAME

Template::Plugin::URL - Plugin to construct complex URLs

=head1 SYNOPSIS

    [% USE url('/cgi-bin/foo.pl') %]
    
    [% url(debug = 1, id = 123) %]
       # ==> /cgi/bin/foo.pl?debug=1&amp;id=123

    [% USE mycgi = url('/cgi-bin/bar.pl', mode='browse', debug=1) %]
    
    [% mycgi %]
       # ==> /cgi/bin/bar.pl?mode=browse&amp;debug=1
    
    [% mycgi(mode='submit') %]
       # ==> /cgi/bin/bar.pl?mode=submit&amp;debug=1
    
    [% mycgi(debug='d2 p0', id='D4-2k[4]') %]
       # ==> /cgi-bin/bar.pl?mode=browse&amp;debug=d2%20p0&amp;id=D4-2k%5B4%5D

=head1 DESCRIPTION

The C<URL> plugin can be used to construct complex URLs from a base stem 
and a hash array of additional query parameters.

The constructor should be passed a base URL and optionally, a hash array
reference of default parameters and values.  Used from with a template,
it would look something like the following:

    [% USE url('http://www.somewhere.com/cgi-bin/foo.pl') %]
    [% USE url('/cgi-bin/bar.pl', mode='browse') %]
    [% USE url('/cgi-bin/baz.pl', mode='browse', debug=1) %]

When the plugin is then called without any arguments, the default base
and parameters are returned as a formatted query string.  

    [% url %]

For the above three examples, these will produce the following outputs:

    http://www.somewhere.com/cgi-bin/foo.pl
    /cgi-bin/bar.pl?mode=browse
    /cgi-bin/baz.pl?mode=browse&amp;debug=1

Note that additional parameters are separated by 'C<&amp;>' rather than
simply 'C<&>'.  This is the correct behaviour for HTML pages but is,
unfortunately, incorrect when creating URLs that do not need to be
encoded safely for HTML.  This is likely to be corrected in a future
version of the plugin (most probably with TT3).  In the mean time, you
can set C<$Template::Plugin::URL::JOINT> to C<&> to get the correct
behaviour.

Additional parameters may be also be specified to the URL:

    [% url(mode='submit', id='wiz') %]

Which, for the same three examples, produces:

    http://www.somewhere.com/cgi-bin/foo.pl?mode=submit&amp;id=wiz
    /cgi-bin/bar.pl?mode=browse&amp;id=wiz
    /cgi-bin/baz.pl?mode=browse&amp;debug=1&amp;id=wiz

A new base URL may also be specified as the first option:

    [% url('/cgi-bin/waz.pl', test=1) %]

producing

    /cgi-bin/waz.pl?test=1
    /cgi-bin/waz.pl?mode=browse&amp;test=1
    /cgi-bin/waz.pl?mode=browse&amp;debug=1&amp;test=1

The ordering of the parameters is non-deterministic due to fact that 
Perl's hashes themselves are unordered.  This isn't a problem as the 
ordering of CGI parameters is insignificant (to the best of my knowledge).
All values will be properly escaped thanks to some code borrowed from
Lincoln Stein's C<CGI> module.  e.g.

    [% USE url('/cgi-bin/woz.pl') %]
    [% url(name="Elrich von Benjy d'Weiro") %]

Here the spaces and "C<'>" character are escaped in the output:

    /cgi-bin/woz.pl?name=Elrich%20von%20Benjy%20d%27Weiro

An alternate name may be provided for the plugin at construction time
as per regular Template Toolkit syntax.

    [% USE mycgi = url('cgi-bin/min.pl') %]
    [% mycgi(debug=1) %]

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
