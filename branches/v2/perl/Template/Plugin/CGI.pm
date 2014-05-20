#============================================================= -*-Perl-*-
#
# Template::Plugin::CGI
#
# DESCRIPTION
#   Simple Template Toolkit plugin interfacing to the CGI.pm module.
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

package Template::Plugin::CGI;

use strict;
use warnings;
use base 'Template::Plugin';
use CGI;

our $VERSION = 2.70;

sub new {
    my $class   = shift;
    my $context = shift;
    CGI->new(@_);
}

# monkeypatch CGI::params() method to Do The Right Thing in TT land

sub CGI::params {
    my $self = shift;
    local $" = ', ';

    return $self->{ _TT_PARAMS } ||= do {
        # must call Vars() in a list context to receive
        # plain list of key/vals rather than a tied hash
        my $params = { $self->Vars() };

        # convert any null separated values into lists
        @$params{ keys %$params } = map { 
            /\0/ ? [ split /\0/ ] : $_ 
        } values %$params;

        $params;
    };
}

1;

__END__

=head1 NAME

Template::Plugin::CGI - Interface to the CGI module

=head1 SYNOPSIS

    [% USE CGI %]
    [% CGI.param('parameter') %]
    
    [% USE things = CGI %]
    [% things.param('name') %]
    
    # see CGI docs for other methods provided by the CGI object

=head1 DESCRIPTION

This is a very simple Template Toolkit Plugin interface to the C<CGI> module.
A C<CGI> object will be instantiated via the following directive:

    [% USE CGI %]

C<CGI> methods may then be called as follows:

    [% CGI.header %]
    [% CGI.param('parameter') %]

An alias can be used to provide an alternate name by which the object should
be identified.

    [% USE mycgi = CGI %]
    [% mycgi.start_form %]
    [% mycgi.popup_menu({ Name   => 'Color'
                          Values => [ 'Green' 'Black' 'Brown' ] }) %]

Parenthesised parameters to the C<USE> directive will be passed to the plugin 
constructor:

    [% USE cgiprm = CGI('uid=abw&name=Andy+Wardley') %]
    [% cgiprm.param('uid') %]

=head1 METHODS

In addition to all the methods supported by the C<CGI> module, this
plugin defines the following.

=head2 params()

This method returns a reference to a hash of all the C<CGI> parameters.
Any parameters that have multiple values will be returned as lists.

    [% USE CGI('user=abw&item=foo&item=bar') %]
    [% CGI.params.user %]            # abw
    [% CGI.params.item.join(', ') %] # foo, bar

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<CGI>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
