#============================================================= -*-Perl-*-
#
# Template::Exception
#
# DESCRIPTION
#   Module implementing a generic exception class used for error handling
#   in the Template Toolkit.
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
#========================================================================

package Template::Exception;

use strict;
use warnings;
use constant TYPE  => 0;
use constant INFO  => 1;
use constant TEXT  => 2;
use overload q|""| => "as_string", fallback => 1;

our $VERSION = 2.70;


#------------------------------------------------------------------------
# new($type, $info, \$text)
#
# Constructor method used to instantiate a new Template::Exception
# object.  The first parameter should contain the exception type.  This
# can be any arbitrary string of the caller's choice to represent a 
# specific exception.  The second parameter should contain any 
# information (i.e. error message or data reference) relevant to the 
# specific exception event.  The third optional parameter may be a 
# reference to a scalar containing output text from the template 
# block up to the point where the exception was thrown.
#------------------------------------------------------------------------

sub new {
    my ($class, $type, $info, $textref) = @_;
    bless [ $type, $info, $textref ], $class;
}


#------------------------------------------------------------------------
# type()
# info()
# type_info()
#
# Accessor methods to return the internal TYPE and INFO fields.
#------------------------------------------------------------------------

sub type {
    $_[0]->[ TYPE ];
}

sub info {
    $_[0]->[ INFO ];
}

sub type_info {
    my $self = shift;
    @$self[ TYPE, INFO ];
}

#------------------------------------------------------------------------
# text()
# text(\$pretext)
#
# Method to return the text referenced by the TEXT member.  A text 
# reference may be passed as a parameter to supercede the existing 
# member.  The existing text is added to the *end* of the new text
# before being stored.  This facility is provided for template blocks
# to gracefully de-nest when an exception occurs and allows them to 
# reconstruct their output in the correct order. 
#------------------------------------------------------------------------

sub text {
    my ($self, $newtextref) = @_;
    my $textref = $self->[ TEXT ];
    
    if ($newtextref) {
        $$newtextref .= $$textref if $textref && $textref ne $newtextref;
        $self->[ TEXT ] = $newtextref;
        return '';
    }
    elsif ($textref) {
        return $$textref;
    }
    else {
        return '';
    }
}


#------------------------------------------------------------------------
# as_string()
#
# Accessor method to return a string indicating the exception type and
# information.
#------------------------------------------------------------------------

sub as_string {
    my $self = shift;
    return $self->[ TYPE ] . ' error - ' . $self->[ INFO ];
}


#------------------------------------------------------------------------
# select_handler(@types)
# 
# Selects the most appropriate handler for the exception TYPE, from 
# the list of types passed in as parameters.  The method returns the
# item which is an exact match for TYPE or the closest, more 
# generic handler (e.g. foo being more generic than foo.bar, etc.)
#------------------------------------------------------------------------

sub select_handler {
    my ($self, @options) = @_;
    my $type = $self->[ TYPE ];
    my %hlut;
    @hlut{ @options } = (1) x @options;

    while ($type) {
        return $type if $hlut{ $type };

        # strip .element from the end of the exception type to find a 
        # more generic handler
        $type =~ s/\.?[^\.]*$//;
    }
    return undef;
}
    
1;

__END__

=head1 NAME

Template::Exception - Exception handling class module

=head1 SYNOPSIS

    use Template::Exception;
    
    my $exception = Template::Exception->new($type, $info);
    $type = $exception->type;
    $info = $exception->info;
    ($type, $info) = $exception->type_info;
    
    print $exception->as_string();
    
    $handler = $exception->select_handler(\@candidates);

=head1 DESCRIPTION

The C<Template::Exception> module defines an object class for
representing exceptions within the template processing life cycle.
Exceptions can be raised by modules within the Template Toolkit, or
can be generated and returned by user code bound to template
variables.

Exceptions can be raised in a template using the C<THROW> directive,

    [% THROW user.login 'no user id: please login' %]

or by calling the L<throw()|Template::Context#throw()> method on the current
L<Template::Context> object,

    $context->throw('user.passwd', 'Incorrect Password');
    $context->throw('Incorrect Password');    # type 'undef'

or from Perl code by calling C<die()> with a C<Template::Exception> object,

    die (Template::Exception->new('user.denied', 'Invalid User ID'));

or by simply calling C<die()> with an error string.  This is
automagically caught and converted to an  exception of 'C<undef>'
type (that's the literal string 'C<undef>' rather than Perl's 
undefined value) which can then be handled in the usual way.

    die "I'm sorry Dave, I can't do that";

Each exception is defined by its type and a information component
(e.g. error message).  The type can be any identifying string and may
contain dotted components (e.g. 'C<foo>', 'C<foo.bar>', 'C<foo.bar.baz>').
Exception types are considered to be hierarchical such that 'C<foo.bar>'
would be a specific type of the more general 'C<foo>' type.

=head1 METHODS

=head2 type()

Returns the exception type.

=head2 info()

Returns the exception information.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template>, L<Template::Context>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
