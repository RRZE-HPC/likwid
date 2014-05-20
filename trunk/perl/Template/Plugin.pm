#============================================================= -*-Perl-*-
#
# Template::Plugin
#
# DESCRIPTION
#
#   Module defining a base class for a plugin object which can be loaded
#   and instantiated via the USE directive.
#
# AUTHOR
#   Andy Wardley   <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it an/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin;

use strict;
use warnings;
use base 'Template::Base';

our $VERSION = 2.70;
our $DEBUG   = 0 unless defined $DEBUG;
our $ERROR   = '';
our $AUTOLOAD;


#========================================================================
#                      -----  CLASS METHODS -----
#========================================================================

#------------------------------------------------------------------------
# load()
#
# Class method called when the plugin module is first loaded.  It 
# returns the name of a class (by default, its own class) or a prototype
# object which will be used to instantiate new objects.  The new() 
# method is then called against the class name (class method) or 
# prototype object (object method) to create a new instances of the 
# object.
#------------------------------------------------------------------------

sub load {
    return $_[0];
}


#------------------------------------------------------------------------
# new($context, $delegate, @params)
#
# Object constructor which is called by the Template::Context to 
# instantiate a new Plugin object.  This base class constructor is 
# used as a general mechanism to load and delegate to other Perl 
# modules.  The context is passed as the first parameter, followed by
# a reference to a delegate object or the name of the module which 
# should be loaded and instantiated.  Any additional parameters passed 
# to the USE directive are forwarded to the new() constructor.
# 
# A plugin object is returned which has an AUTOLOAD method to delegate 
# requests to the underlying object.
#------------------------------------------------------------------------

sub new {
    my $class = shift;
    bless {
    }, $class;
}

sub old_new {
    my ($class, $context, $delclass, @params) = @_;
    my ($delegate, $delmod);

    return $class->error("no context passed to $class constructor\n")
        unless defined $context;

    if (ref $delclass) {
        # $delclass contains a reference to a delegate object
        $delegate = $delclass;
    }
    else {
        # delclass is the name of a module to load and instantiate
        ($delmod = $delclass) =~ s|::|/|g;

        eval {
            require "$delmod.pm";
            $delegate = $delclass->new(@params)
                || die "failed to instantiate $delclass object\n";
        };
        return $class->error($@) if $@;
    }

    bless {
        _CONTEXT  => $context, 
        _DELEGATE => $delegate,
        _PARAMS   => \@params,
    }, $class;
}


#------------------------------------------------------------------------
# fail($error)
# 
# Version 1 error reporting function, now replaced by error() inherited
# from Template::Base.  Raises a "deprecated function" warning and then
# calls error().
#------------------------------------------------------------------------

sub fail {
    my $class = shift;
    my ($pkg, $file, $line) = caller();
    warn "Template::Plugin::fail() is deprecated at $file line $line.  Please use error()\n";
    $class->error(@_);
}


#========================================================================
#                      -----  OBJECT METHODS -----
#========================================================================

#------------------------------------------------------------------------
# AUTOLOAD
#
# General catch-all method which delegates all calls to the _DELEGATE 
# object.  
#------------------------------------------------------------------------

sub OLD_AUTOLOAD {
    my $self     = shift;
    my $method   = $AUTOLOAD;

    $method =~ s/.*:://;
    return if $method eq 'DESTROY';

    if (ref $self eq 'HASH') {
        my $delegate = $self->{ _DELEGATE } || return;
        return $delegate->$method(@_);
    }
    my ($pkg, $file, $line) = caller();
#    warn "no such '$method' method called on $self at $file line $line\n";
    return undef;
}


1;

__END__

=head1 NAME

Template::Plugin - Base class for Template Toolkit plugins

=head1 SYNOPSIS

    package MyOrg::Template::Plugin::MyPlugin;
    use base qw( Template::Plugin );
    use Template::Plugin;
    use MyModule;
    
    sub new {
        my $class   = shift;
        my $context = shift;
        bless {
            ...
        }, $class;
    }

=head1 DESCRIPTION

A "plugin" for the Template Toolkit is simply a Perl module which 
exists in a known package location (e.g. C<Template::Plugin::*>) and 
conforms to a regular standard, allowing it to be loaded and used 
automatically.

The C<Template::Plugin> module defines a base class from which other 
plugin modules can be derived.  A plugin does not have to be derived
from Template::Plugin but should at least conform to its object-oriented
interface.

It is recommended that you create plugins in your own package namespace
to avoid conflict with toolkit plugins.  e.g. 

    package MyOrg::Template::Plugin::FooBar;

Use the L<PLUGIN_BASE|Template::Manual::Config#PLUGIN_BASE> option to specify
the namespace that you use. e.g.

    use Template;
    my $template = Template->new({ 
        PLUGIN_BASE => 'MyOrg::Template::Plugin',
    });

=head1 METHODS

The following methods form the basic interface between the Template
Toolkit and plugin modules.

=head2 load($context)

This method is called by the Template Toolkit when the plugin module
is first loaded.  It is called as a package method and thus implicitly
receives the package name as the first parameter.  A reference to the
L<Template::Context> object loading the plugin is also passed.  The
default behaviour for the C<load()> method is to simply return the class
name.  The calling context then uses this class name to call the C<new()>
package method.

    package MyPlugin;
    
    sub load {               # called as MyPlugin->load($context)
        my ($class, $context) = @_;
        return $class;       # returns 'MyPlugin'
    }

=head2 new($context, @params)

This method is called to instantiate a new plugin object for the C<USE>
directive. It is called as a package method against the class name returned by
L<load()>. A reference to the L<Template::Context> object creating the plugin
is passed, along with any additional parameters specified in the C<USE>
directive.

    sub new {                # called as MyPlugin->new($context)
        my ($class, $context, @params) = @_;
        bless {
            _CONTEXT => $context,
        }, $class;           # returns blessed MyPlugin object
    }

=head2 error($error)

This method, inherited from the L<Template::Base> module, is used for 
reporting and returning errors.   It can be called as a package method
to set/return the C<$ERROR> package variable, or as an object method to 
set/return the object C<_ERROR> member.  When called with an argument, it
sets the relevant variable and returns C<undef.>  When called without an
argument, it returns the value of the variable.

    package MyPlugin;
    use base 'Template::Plugin';
    
    sub new {
        my ($class, $context, $dsn) = @_;
        
        return $class->error('No data source specified')
            unless $dsn;
        
        bless {
            _DSN => $dsn,
        }, $class;
    }

    package main;
    
    my $something = MyPlugin->new()
        || die MyPlugin->error(), "\n";
        
    $something->do_something()
        || die $something->error(), "\n";

=head1 DEEPER MAGIC

The L<Template::Context> object that handles the loading and use of plugins
calls the L<new()> and L<error()> methods against the package name returned by
the L<load()> method. In pseudo-code terms looks something like this:

    $class  = MyPlugin->load($context);       # returns 'MyPlugin'
    
    $object = $class->new($context, @params)  # MyPlugin->new(...)
        || die $class->error();               # MyPlugin->error()

The L<load()> method may alterately return a blessed reference to an
object instance.  In this case, L<new()> and L<error()> are then called as
I<object> methods against that prototype instance.

    package YourPlugin;
    
    sub load {
        my ($class, $context) = @_;
        bless {
            _CONTEXT => $context,
        }, $class;
    }
    
    sub new {
        my ($self, $context, @params) = @_;
        return $self;
    }

In this example, we have implemented a 'Singleton' plugin.  One object 
gets created when L<load()> is called and this simply returns itself for
each call to L<new().>   

Another implementation might require individual objects to be created
for every call to L<new(),> but with each object sharing a reference to
some other object to maintain cached data, database handles, etc.
This pseudo-code example demonstrates the principle.

    package MyServer;
    
    sub load {
        my ($class, $context) = @_;
        bless {
            _CONTEXT => $context,
            _CACHE   => { },
        }, $class;
    }
    
    sub new {
        my ($self, $context, @params) = @_;
        MyClient->new($self, @params);
    }
    
    sub add_to_cache   { ... }
    
    sub get_from_cache { ... }

    package MyClient;
    
    sub new {
        my ($class, $server, $blah) = @_;
        bless {
            _SERVER => $server,
            _BLAH   => $blah,
        }, $class;
    }
    
    sub get {
        my $self = shift;
        $self->{ _SERVER }->get_from_cache(@_);
    }
    
    sub put {
        my $self = shift;
        $self->{ _SERVER }->add_to_cache(@_);
    }

When the plugin is loaded, a C<MyServer> instance is created. The L<new()>
method is called against this object which instantiates and returns a C<MyClient>
object, primed to communicate with the creating C<MyServer>.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template>, L<Template::Plugins>, L<Template::Context>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
