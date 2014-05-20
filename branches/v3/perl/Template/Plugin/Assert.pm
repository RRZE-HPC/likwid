#============================================================= -*-Perl-*-
#
# Template::Plugin::Assert
#
# DESCRIPTION
#   Template Toolkit plugin module which allows you to assert that
#   items fetchs from the stash are defined.
#
# AUTHOR
#   Andy Wardley   <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 2008 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin::Assert;
use base 'Template::Plugin';
use strict;
use warnings;
use Template::Exception;

our $VERSION   = 1.00;
our $MONAD     = 'Template::Monad::Assert';
our $EXCEPTION = 'Template::Exception';
our $AUTOLOAD;

sub load {
    my $class   = shift;
    my $context = shift;
    my $stash   = $context->stash;
    my $vmethod = sub {
        $MONAD->new($stash, shift);
    };

    # define .assert vmethods for hash and list objects
    $context->define_vmethod( hash => assert => $vmethod );
    $context->define_vmethod( list => assert => $vmethod );

    return $class;
}

sub new {
    my ($class, $context, @args) = @_;
    # create an assert plugin object which will handle simple variable
    # lookups.
    return bless { _CONTEXT => $context }, $class;
}

sub AUTOLOAD {
    my ($self, @args) = @_;
    my $item = $AUTOLOAD;
    $item =~ s/.*:://;
    return if $item eq 'DESTROY';
    
    # lookup the named values
    my $stash = $self->{ _CONTEXT }->stash;
    my $value = $stash->dotop($stash, $item, \@args);

    if (! defined $value) {
        die $EXCEPTION->new( assert => "undefined value for $item" );
    }
    return $value;
}


package Template::Monad::Assert;

our $EXCEPTION = 'Template::Exception';
our $AUTOLOAD;

sub new {
    my ($class, $stash, $this) = @_;
    bless [$stash, $this], $class;
}

sub AUTOLOAD {
    my ($self, @args) = @_;
    my ($stash, $this) = @$self;
    my $item = $AUTOLOAD;
    $item =~ s/.*:://;
    return if $item eq 'DESTROY';

    my $value = $stash->dotop($stash, $item, \@args);

    if (! defined $value) {
        die $EXCEPTION->new( assert => "undefined value for $item" );
    }
    return $value;
}

1;

__END__

=head1 NAME

Template::Plugin::Assert - trap undefined values

=head1 SYNOPSIS

    [% USE assert %]
    
    # throws error if any undefined values are returned
    [% object.assert.method %]
    [% hash.assert.key %]
    [% list.assert.item %]

=head1 DESCRIPTION

This plugin defines the C<assert> virtual method that can be used
to automatically throw errors when undefined values are used.

For example, consider this dotop:

    [% user.name %]

If C<user.name> is an undefined value then TT will silently ignore the 
fact and print nothing.  If you C<USE> the C<assert> plugin then you
can add the C<assert> vmethod between the C<user> and C<name> elements,
like so:

    [% user.assert.name %]

Now, if C<user.name> is an undefined value, an exception will be thrown:

    assert error - undefined value for name

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 2008 Andy Wardley.  All Rights Reserved.

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
