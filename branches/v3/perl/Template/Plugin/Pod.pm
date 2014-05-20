#==============================================================================
# 
# Template::Plugin::Pod
#
# DESCRIPTION
#  Pod parser and object model.
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

package Template::Plugin::Pod;

use strict;
use warnings;
use base 'Template::Plugin';
use Pod::POM;


our $VERSION = 2.69;

#------------------------------------------------------------------------
# new($context, \%config)
#------------------------------------------------------------------------

sub new {
    my $class = shift;
    my $context = shift;

    Pod::POM->new(@_);
}


1;

__END__

=head1 NAME

Template::Plugin::Pod - Plugin interface to Pod::POM (Pod Object Model)

=head1 SYNOPSIS

    [% USE Pod(podfile) %]
    
    [% FOREACH head1 = Pod.head1;
         FOREACH head2 = head1/head2;
           ...
         END;
       END
    %]

=head1 DESCRIPTION

This plugin is an interface to the L<Pod::POM> module.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<Pod::POM>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
