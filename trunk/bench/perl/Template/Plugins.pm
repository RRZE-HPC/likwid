#============================================================= -*-Perl-*-
#
# Template::Plugins
#
# DESCRIPTION
#   Plugin provider which handles the loading of plugin modules and 
#   instantiation of plugin objects.
#
# AUTHORS
#   Andy Wardley <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 1996-2006 Andy Wardley.  All Rights Reserved.
#   Copyright (C) 1998-2000 Canon Research Centre Europe Ltd.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
# REVISION
#   $Id: Plugins.pm 1179 2008-12-09 19:29:21Z abw $
#
#============================================================================

package Template::Plugins;

use strict;
use warnings;
use base 'Template::Base';
use Template::Constants;

our $VERSION = 2.77;
our $DEBUG   = 0 unless defined $DEBUG;
our $PLUGIN_BASE = 'Template::Plugin';
our $STD_PLUGINS = {
    'assert'     => 'Template::Plugin::Assert',
    'autoformat' => 'Template::Plugin::Autoformat',
    'cgi'        => 'Template::Plugin::CGI',
    'datafile'   => 'Template::Plugin::Datafile',
    'date'       => 'Template::Plugin::Date',
    'debug'      => 'Template::Plugin::Debug',
    'directory'  => 'Template::Plugin::Directory',
    'dbi'        => 'Template::Plugin::DBI',
    'dumper'     => 'Template::Plugin::Dumper',
    'file'       => 'Template::Plugin::File',
    'format'     => 'Template::Plugin::Format',
    'html'       => 'Template::Plugin::HTML',
    'image'      => 'Template::Plugin::Image',
    'iterator'   => 'Template::Plugin::Iterator',
    'latex'      => 'Template::Plugin::Latex',
    'pod'        => 'Template::Plugin::Pod',
    'scalar'     => 'Template::Plugin::Scalar',
    'table'      => 'Template::Plugin::Table',
    'url'        => 'Template::Plugin::URL',
    'view'       => 'Template::Plugin::View',
    'wrap'       => 'Template::Plugin::Wrap',
    'xml'        => 'Template::Plugin::XML',
    'xmlstyle'   => 'Template::Plugin::XML::Style',
};


#========================================================================
#                         -- PUBLIC METHODS --
#========================================================================

#------------------------------------------------------------------------
# fetch($name, \@args, $context)
#
# General purpose method for requesting instantiation of a plugin
# object.  The name of the plugin is passed as the first parameter.
# The internal FACTORY lookup table is consulted to retrieve the
# appropriate factory object or class name.  If undefined, the _load()
# method is called to attempt to load the module and return a factory
# class/object which is then cached for subsequent use.  A reference
# to the calling context should be passed as the third parameter.
# This is passed to the _load() class method.  The new() method is
# then called against the factory class name or prototype object to
# instantiate a new plugin object, passing any arguments specified by
# list reference as the second parameter.  e.g. where $factory is the
# class name 'MyClass', the new() method is called as a class method,
# $factory->new(...), equivalent to MyClass->new(...) .  Where
# $factory is a prototype object, the new() method is called as an
# object method, $myobject->new(...).  This latter approach allows
# plugins to act as Singletons, cache shared data, etc.  
#
# Returns a reference to a plugin, (undef, STATUS_DECLINE) to decline
# the request or ($error, STATUS_ERROR) on error.
#------------------------------------------------------------------------

sub fetch {
    my ($self, $name, $args, $context) = @_;
    my ($factory, $plugin, $error);

    $self->debug("fetch($name, ", 
                 defined $args ? ('[ ', join(', ', @$args), ' ]') : '<no args>', ', ',
                 defined $context ? $context : '<no context>', 
                 ')') if $self->{ DEBUG };

    # NOTE:
    # the $context ref gets passed as the first parameter to all regular
    # plugins, but not to those loaded via LOAD_PERL;  to hack around
    # this until we have a better implementation, we pass the $args
    # reference to _load() and let it unshift the first args in the 
    # LOAD_PERL case

    $args ||= [ ];
    unshift @$args, $context;

    $factory = $self->{ FACTORY }->{ $name } ||= do {
        ($factory, $error) = $self->_load($name, $context);
        return ($factory, $error) if $error;			## RETURN
        $factory;
    };

    # call the new() method on the factory object or class name
    eval {
        if (ref $factory eq 'CODE') {
            defined( $plugin = &$factory(@$args) )
                || die "$name plugin failed\n";
        }
        else {
            defined( $plugin = $factory->new(@$args) )
                || die "$name plugin failed: ", $factory->error(), "\n";
        }
    };
    if ($error = $@) {
#	chomp $error;
        return $self->{ TOLERANT } 
	       ? (undef,  Template::Constants::STATUS_DECLINED)
	       : ($error, Template::Constants::STATUS_ERROR);
    }

    return $plugin;
}



#========================================================================
#                        -- PRIVATE METHODS --
#========================================================================

#------------------------------------------------------------------------
# _init(\%config)
#
# Private initialisation method.
#------------------------------------------------------------------------

sub _init {
    my ($self, $params) = @_;
    my ($pbase, $plugins, $factory) = 
        @$params{ qw( PLUGIN_BASE PLUGINS PLUGIN_FACTORY ) };

    $plugins ||= { };

    # update PLUGIN_BASE to an array ref if necessary
    $pbase = [ ] unless defined $pbase;
    $pbase = [ $pbase ] unless ref($pbase) eq 'ARRAY';
    
    # add default plugin base (Template::Plugin) if set
    push(@$pbase, $PLUGIN_BASE) if $PLUGIN_BASE;

    $self->{ PLUGIN_BASE } = $pbase;
    $self->{ PLUGINS     } = { %$STD_PLUGINS, %$plugins };
    $self->{ TOLERANT    } = $params->{ TOLERANT }  || 0;
    $self->{ LOAD_PERL   } = $params->{ LOAD_PERL } || 0;
    $self->{ FACTORY     } = $factory || { };
    $self->{ DEBUG       } = ( $params->{ DEBUG } || 0 )
                             & Template::Constants::DEBUG_PLUGINS;

    return $self;
}



#------------------------------------------------------------------------
# _load($name, $context)
#
# Private method which attempts to load a plugin module and determine the 
# correct factory name or object by calling the load() class method in
# the loaded module.
#------------------------------------------------------------------------

sub _load {
    my ($self, $name, $context) = @_;
    my ($factory, $module, $base, $pkg, $file, $ok, $error);

    if ($module = $self->{ PLUGINS }->{ $name } || $self->{ PLUGINS }->{ lc $name }) {
        # plugin module name is explicitly stated in PLUGIN_NAME
        $pkg = $module;
        ($file = $module) =~ s|::|/|g;
        $file =~ s|::|/|g;
        $self->debug("loading $module.pm (PLUGIN_NAME)")
            if $self->{ DEBUG };
        $ok = eval { require "$file.pm" };
        $error = $@;
    }
    else {
        # try each of the PLUGIN_BASE values to build module name
        ($module = $name) =~ s/\./::/g;
        
        foreach $base (@{ $self->{ PLUGIN_BASE } }) {
            $pkg = $base . '::' . $module;
            ($file = $pkg) =~ s|::|/|g;
            
            $self->debug("loading $file.pm (PLUGIN_BASE)")
                if $self->{ DEBUG };
            
            $ok = eval { require "$file.pm" };
            last unless $@;
            
            $error .= "$@\n" 
                unless ($@ =~ /^Can\'t locate $file\.pm/);
        }
    }
    
    if ($ok) {
        $self->debug("calling $pkg->load()") if $self->{ DEBUG };

	$factory = eval { $pkg->load($context) };
        $error   = '';
        if ($@ || ! $factory) {
            $error = $@ || 'load() returned a false value';
        }
    }
    elsif ($self->{ LOAD_PERL }) {
        # fallback - is it a regular Perl module?
        ($file = $module) =~ s|::|/|g;
        eval { require "$file.pm" };
        if ($@) {
            $error = $@;
        }
        else {
            # this is a regular Perl module so the new() constructor
            # isn't expecting a $context reference as the first argument;
            # so we construct a closure which removes it before calling
            # $module->new(@_);
            $factory = sub {
                shift;
                $module->new(@_);
            };
            $error   = '';
        }
    }
    
    if ($factory) {
        $self->debug("$name => $factory") if $self->{ DEBUG };
        return $factory;
    }
    elsif ($error) {
        return $self->{ TOLERANT } 
	    ? (undef,  Template::Constants::STATUS_DECLINED) 
            : ($error, Template::Constants::STATUS_ERROR);
    }
    else {
        return (undef, Template::Constants::STATUS_DECLINED);
    }
}


#------------------------------------------------------------------------
# _dump()
# 
# Debug method which constructs and returns text representing the current
# state of the object.
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    my $output = "[Template::Plugins] {\n";
    my $format = "    %-16s => %s\n";
    my $key;

    foreach $key (qw( TOLERANT LOAD_PERL )) {
        $output .= sprintf($format, $key, $self->{ $key });
    }

    local $" = ', ';
    my $fkeys = join(", ", keys %{$self->{ FACTORY }});
    my $plugins = $self->{ PLUGINS };
    $plugins = join('', map { 
        sprintf("    $format", $_, $plugins->{ $_ });
    } keys %$plugins);
    $plugins = "{\n$plugins    }";
    
    $output .= sprintf($format, 'PLUGIN_BASE', "[ @{ $self->{ PLUGIN_BASE } } ]");
    $output .= sprintf($format, 'PLUGINS', $plugins);
    $output .= sprintf($format, 'FACTORY', $fkeys);
    $output .= '}';
    return $output;
}


1;

__END__

=head1 NAME

Template::Plugins - Plugin provider module

=head1 SYNOPSIS

    use Template::Plugins;
    
    $plugin_provider = Template::Plugins->new(\%options);
    
    ($plugin, $error) = $plugin_provider->fetch($name, @args);

=head1 DESCRIPTION

The C<Template::Plugins> module defines a provider class which can be used
to load and instantiate Template Toolkit plugin modules.

=head1 METHODS

=head2 new(\%params) 

Constructor method which instantiates and returns a reference to a
C<Template::Plugins> object.  A reference to a hash array of configuration
items may be passed as a parameter.  These are described below.  

Note that the L<Template> front-end module creates a C<Template::Plugins>
provider, passing all configuration items.  Thus, the examples shown
below in the form:

    $plugprov = Template::Plugins->new({
        PLUGIN_BASE => 'MyTemplate::Plugin',
        LOAD_PERL   => 1,
        ...
    });

can also be used via the L<Template> module as:

    $ttengine = Template->new({
        PLUGIN_BASE => 'MyTemplate::Plugin',
        LOAD_PERL   => 1,
        ...
    });

as well as the more explicit form of:

    $plugprov = Template::Plugins->new({
        PLUGIN_BASE => 'MyTemplate::Plugin',
        LOAD_PERL   => 1,
        ...
    });
    
    $ttengine = Template->new({
        LOAD_PLUGINS => [ $plugprov ],
    });

=head2 fetch($name, @args)

Called to request that a plugin of a given name be provided. The relevant
module is first loaded (if necessary) and the
L<load()|Template::Plugin#load()> class method called to return the factory
class name (usually the same package name) or a factory object (a prototype).
The L<new()|Template::Plugin#new()> method is then called as a class or object
method against the factory, passing all remaining parameters.

Returns a reference to a new plugin object or C<($error, STATUS_ERROR)>
on error.  May also return C<(undef, STATUS_DECLINED)> to decline to
serve the request.  If C<TOLERANT> is set then all errors will be
returned as declines.

=head1 CONFIGURATION OPTIONS

The following list summarises the configuration options that can be provided
to the C<Template::Plugins> L<new()> constructor.  Please consult 
L<Template::Manual::Config> for further details and examples of each 
configuration option in use.

=head2 PLUGINS

The L<PLUGINS|Template::Manual::Config#PLUGINS> option can be used to provide
a reference to a hash array that maps plugin names to Perl module names.

    my $plugins = Template::Plugins->new({
        PLUGINS => {
            cgi => 'MyOrg::Template::Plugin::CGI',
            foo => 'MyOrg::Template::Plugin::Foo',
            bar => 'MyOrg::Template::Plugin::Bar',
        },  
    }); 

=head2 PLUGIN_BASE

If a plugin is not defined in the L<PLUGINS|Template::Manual::Config#PLUGINS>
hash then the L<PLUGIN_BASE|Template::Manual::Config#PLUGIN_BASE> is used to
attempt to construct a correct Perl module name which can be successfully
loaded.

    # single value PLUGIN_BASE
    my $plugins = Template::Plugins->new({
        PLUGIN_BASE => 'MyOrg::Template::Plugin',
    });

    # multiple value PLUGIN_BASE
    my $plugins = Template::Plugins->new({
        PLUGIN_BASE => [   'MyOrg::Template::Plugin',
                           'YourOrg::Template::Plugin'  ],
    });

=head2 LOAD_PERL

The L<LOAD_PERL|Template::Manual::Config#LOAD_PERL> option can be set to allow
you to load regular Perl modules (i.e. those that don't reside in the
C<Template::Plugin> or another user-defined namespace) as plugins.

If a plugin cannot be loaded using the
L<PLUGINS|Template::Manual::Config#PLUGINS> or
L<PLUGIN_BASE|Template::Manual::Config#PLUGIN_BASE> approaches then,
if the L<LOAD_PERL|Template::Manual::Config#LOAD_PERL> is set, the
provider will make a final attempt to load the module without prepending any
prefix to the module path. 

Unlike regular plugins, modules loaded using L<LOAD_PERL|Template::Manual::Config#LOAD_PERL>
do not receive a L<Template::Context> reference as the first argument to the 
C<new()> constructor method.

=head2 TOLERANT

The L<TOLERANT|Template::Manual::Config#TOLERANT> flag can be set to indicate
that the C<Template::Plugins> module should ignore any errors encountered while
loading a plugin and instead return C<STATUS_DECLINED>.

=head2 DEBUG

The L<DEBUG|Template::Manual::Config#DEBUG> option can be used to enable
debugging messages for the C<Template::Plugins> module by setting it to
include the C<DEBUG_PLUGINS> value.

    use Template::Constants qw( :debug );
    
    my $template = Template->new({
        DEBUG => DEBUG_FILTERS | DEBUG_PLUGINS,
    });

=head1 TEMPLATE TOOLKIT PLUGINS

Please see L<Template::Manual::Plugins> For a complete list of all the plugin 
modules distributed with the Template Toolkit.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Manual::Plugins>, L<Template::Plugin>, L<Template::Context>, L<Template>.

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
