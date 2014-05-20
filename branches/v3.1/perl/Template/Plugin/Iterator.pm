#============================================================= -*-Perl-*-
#
# Template::Plugin::Iterator
#
# DESCRIPTION
#
#   Plugin to create a Template::Iterator from a list of items and optional
#   configuration parameters.
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

package Template::Plugin::Iterator;

use strict;
use warnings;
use base 'Template::Plugin';
use Template::Iterator;

our $VERSION = 2.68;

#------------------------------------------------------------------------
# new($context, \@data, \%args)
#------------------------------------------------------------------------

sub new {
    my $class   = shift;
    my $context = shift;
    Template::Iterator->new(@_);
}

1;

__END__

=head1 NAME

Template::Plugin::Iterator - Plugin to create iterators (Template::Iterator)

=head1 SYNOPSIS

    [% USE iterator(list, args) %]
    
    [% FOREACH item = iterator %]
       [% '<ul>' IF iterator.first %]
       <li>[% item %]
       [% '</ul>' IF iterator.last %]
    [% END %]

=head1 DESCRIPTION

The iterator plugin provides a way to create a L<Template::Iterator> object 
to iterate over a data set.  An iterator is implicitly automatically by the
L<FOREACH> directive.  This plugin allows the iterator to be explicitly created
with a given name.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<Template::Iterator>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
