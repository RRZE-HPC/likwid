#============================================================= -*-perl-*-
#
# Template::Base
#
# DESCRIPTION
#   Base class module implementing common functionality for various other
#   Template Toolkit modules.
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
 
package Template::Base;

use strict;
use warnings;
use Template::Constants;

our $VERSION = 2.78;


#------------------------------------------------------------------------
# new(\%params)
#
# General purpose constructor method which expects a hash reference of 
# configuration parameters, or a list of name => value pairs which are 
# folded into a hash.  Blesses a hash into an object and calls its 
# _init() method, passing the parameter hash reference.  Returns a new
# object derived from Template::Base, or undef on error.
#------------------------------------------------------------------------

sub new {
    my $class = shift;
    my ($argnames, @args, $arg, $cfg);
#    $class->error('');         # always clear package $ERROR var?

    {   no strict 'refs';
        no warnings 'once';
        $argnames = \@{"$class\::BASEARGS"} || [ ];
    }

    # shift off all mandatory args, returning error if undefined or null
    foreach $arg (@$argnames) {
        return $class->error("no $arg specified")
            unless ($cfg = shift);
        push(@args, $cfg);
    }

    # fold all remaining args into a hash, or use provided hash ref
    $cfg  = defined $_[0] && ref($_[0]) eq 'HASH' ? shift : { @_ };

    my $self = bless {
        (map { ($_ => shift @args) } @$argnames),
        _ERROR  => '',
        DEBUG   => 0,
    }, $class;
    
    return $self->_init($cfg) ? $self : $class->error($self->error);
}


#------------------------------------------------------------------------
# error()
# error($msg, ...)
# 
# May be called as a class or object method to set or retrieve the 
# package variable $ERROR (class method) or internal member 
# $self->{ _ERROR } (object method).  The presence of parameters indicates
# that the error value should be set.  Undef is then returned.  In the
# abscence of parameters, the current error value is returned.
#------------------------------------------------------------------------

sub error {
    my $self = shift;
    my $errvar;

    { 
        no strict qw( refs );
        $errvar = ref $self ? \$self->{ _ERROR } : \${"$self\::ERROR"};
    }
    if (@_) {
        $$errvar = ref($_[0]) ? shift : join('', @_);
        return undef;
    }
    else {
        return $$errvar;
    }
}


#------------------------------------------------------------------------
# _init()
#
# Initialisation method called by the new() constructor and passing a 
# reference to a hash array containing any configuration items specified
# as constructor arguments.  Should return $self on success or undef on 
# error, via a call to the error() method to set the error message.
#------------------------------------------------------------------------

sub _init {
    my ($self, $config) = @_;
    return $self;
}


sub debug {
    my $self = shift;
    my $msg  = join('', @_);
    my ($pkg, $file, $line) = caller();

    unless ($msg =~ /\n$/) {
        $msg .= ($self->{ DEBUG } & Template::Constants::DEBUG_CALLER)
            ? " at $file line $line\n"
            : "\n";
    }

    print STDERR "[$pkg] $msg";
}


#------------------------------------------------------------------------
# module_version()
#
# Returns the current version number.
#------------------------------------------------------------------------

sub module_version {
    my $self = shift;
    my $class = ref $self || $self;
    no strict 'refs';
    return ${"${class}::VERSION"};
}


1;

__END__

=head1 NAME

Template::Base - Base class module implementing common functionality

=head1 SYNOPSIS

    package My::Module;
    use base qw( Template::Base );
    
    sub _init {
        my ($self, $config) = @_;
        $self->{ doodah } = $config->{ doodah }
            || return $self->error("No 'doodah' specified");
        return $self;
    }
    
    package main;
    
    my $object = My::Module->new({ doodah => 'foobar' })
        || die My::Module->error();

=head1 DESCRIPTION

Base class module which implements a constructor and error reporting 
functionality for various Template Toolkit modules.

=head1 PUBLIC METHODS

=head2 new(\%config)

Constructor method which accepts a reference to a hash array or a list 
of C<name =E<gt> value> parameters which are folded into a hash.  The 
C<_init()> method is then called, passing the configuration hash and should
return true/false to indicate success or failure.  A new object reference
is returned, or undef on error.  Any error message raised can be examined
via the L<error()> class method or directly via the C<$ERROR> package variable 
in the derived class.

    my $module = My::Module->new({ ... })
        || die My::Module->error(), "\n";

    my $module = My::Module->new({ ... })
        || die "constructor error: $My::Module::ERROR\n";

=head2 error($msg, ...)

May be called as an object method to get/set the internal C<_ERROR> member
or as a class method to get/set the C<$ERROR> variable in the derived class's
package.

    my $module = My::Module->new({ ... })
        || die My::Module->error(), "\n";

    $module->do_something() 
        || die $module->error(), "\n";

When called with parameters (multiple params are concatenated), this
method will set the relevant variable and return undef.  This is most
often used within object methods to report errors to the caller.

    package My::Module;
    
    sub foobar {
        my $self = shift;
        
        # some other code...
        
        return $self->error('some kind of error...')
            if $some_condition;
    }

=head2 debug($msg, ...)

Generates a debugging message by concatenating all arguments
passed into a string and printing it to C<STDERR>.  A prefix is
added to indicate the module of the caller.

    package My::Module;
    
    sub foobar {
        my $self = shift;
        
        $self->debug('called foobar()');
        
        # some other code...
    }

When the C<foobar()> method is called, the following message
is sent to C<STDERR>:

    [My::Module] called foobar()

Objects can set an internal C<DEBUG> value which the C<debug()>
method will examine.  If this value sets the relevant bits
to indicate C<DEBUG_CALLER> then the file and line number of
the caller will be appened to the message.

    use Template::Constants qw( :debug );
    
    my $module = My::Module->new({
        DEBUG => DEBUG_SERVICE | DEBUG_CONTEXT | DEBUG_CALLER,
    });
    
    $module->foobar();

This generates an error message such as:

    [My::Module] called foobar() at My/Module.pm line 6

=head2 module_version()

Returns the version number for a module, as defined by the C<$VERSION>
package variable.

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
