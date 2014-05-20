#============================================================= -*-Perl-*-
#
# Template::Context
#
# DESCRIPTION
#   Module defining a context in which a template document is processed.
#   This is the runtime processing interface through which templates 
#   can access the functionality of the Template Toolkit.
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

package Template::Context;

use strict;
use warnings;
use base 'Template::Base';

use Template::Base;
use Template::Config;
use Template::Constants;
use Template::Exception;
use Scalar::Util 'blessed';

use constant DOCUMENT         => 'Template::Document';
use constant EXCEPTION        => 'Template::Exception';
use constant BADGER_EXCEPTION => 'Badger::Exception';

our $VERSION = 2.98;
our $DEBUG   = 0 unless defined $DEBUG;
our $DEBUG_FORMAT = "\n## \$file line \$line : [% \$text %] ##\n";
our $VIEW_CLASS   = 'Template::View';
our $AUTOLOAD;

#========================================================================
#                     -----  PUBLIC METHODS -----
#========================================================================

#------------------------------------------------------------------------
# template($name) 
#
# General purpose method to fetch a template and return it in compiled 
# form.  In the usual case, the $name parameter will be a simple string
# containing the name of a template (e.g. 'header').  It may also be 
# a reference to Template::Document object (or sub-class) or a Perl 
# sub-routine.  These are considered to be compiled templates and are
# returned intact.  Finally, it may be a reference to any other kind 
# of valid input source accepted by Template::Provider (e.g. scalar
# ref, glob, IO handle, etc).
#
# Templates may be cached at one of 3 different levels.  The internal
# BLOCKS member is a local cache which holds references to all
# template blocks used or imported via PROCESS since the context's
# reset() method was last called.  This is checked first and if the
# template is not found, the method then walks down the BLOCKSTACK
# list.  This contains references to the block definition tables in
# any enclosing Template::Documents that we're visiting (e.g. we've
# been called via an INCLUDE and we want to access a BLOCK defined in
# the template that INCLUDE'd us).  If nothing is defined, then we
# iterate through the LOAD_TEMPLATES providers list as a 'chain of 
# responsibility' (see Design Patterns) asking each object to fetch() 
# the template if it can.
#
# Returns the compiled template.  On error, undef is returned and 
# the internal ERROR value (read via error()) is set to contain an
# error message of the form "$name: $error".
#------------------------------------------------------------------------

sub template {
    my ($self, $name) = @_;
    my ($prefix, $blocks, $defblocks, $provider, $template, $error);
    my ($shortname, $blockname, $providers);

    $self->debug("template($name)") if $self->{ DEBUG };

    # references to Template::Document (or sub-class) objects objects, or
    # CODE references are assumed to be pre-compiled templates and are
    # returned intact
    return $name
        if (blessed($name) && $name->isa(DOCUMENT))
        || ref($name) eq 'CODE';

    $shortname = $name;

    unless (ref $name) {
        
        $self->debug("looking for block [$name]") if $self->{ DEBUG };

        # we first look in the BLOCKS hash for a BLOCK that may have 
        # been imported from a template (via PROCESS)
        return $template
            if ($template = $self->{ BLOCKS }->{ $name });
        
        # then we iterate through the BLKSTACK list to see if any of the
        # Template::Documents we're visiting define this BLOCK
        foreach $blocks (@{ $self->{ BLKSTACK } }) {
            return $template
                if $blocks && ($template = $blocks->{ $name });
        }
        
        # now it's time to ask the providers, so we look to see if any 
        # prefix is specified to indicate the desired provider set.
        if ($^O eq 'MSWin32') {
            # let C:/foo through
            $prefix = $1 if $shortname =~ s/^(\w{2,})://o;
        }
        else {
            $prefix = $1 if $shortname =~ s/^(\w+)://;
        }
        
        if (defined $prefix) {
            $providers = $self->{ PREFIX_MAP }->{ $prefix } 
            || return $self->throw( Template::Constants::ERROR_FILE,
                                    "no providers for template prefix '$prefix'");
        }
    }
    $providers = $self->{ PREFIX_MAP }->{ default }
        || $self->{ LOAD_TEMPLATES }
            unless $providers;


    # Finally we try the regular template providers which will 
    # handle references to files, text, etc., as well as templates
    # reference by name.  If

    $blockname = '';
    while ($shortname) {
        $self->debug("asking providers for [$shortname] [$blockname]") 
            if $self->{ DEBUG };

        foreach my $provider (@$providers) {
            ($template, $error) = $provider->fetch($shortname, $prefix);
            if ($error) {
                if ($error == Template::Constants::STATUS_ERROR) {
                    # $template contains exception object
                    if (blessed($template) && $template->isa(EXCEPTION)
                        && $template->type eq Template::Constants::ERROR_FILE) {
                        $self->throw($template);
                    }
                    else {
                        $self->throw( Template::Constants::ERROR_FILE, $template );
                    }
                }
                # DECLINE is ok, carry on
            }
            elsif (length $blockname) {
                return $template 
                    if $template = $template->blocks->{ $blockname };
            }
            else {
                return $template;
            }
        }
        
        last if ref $shortname || ! $self->{ EXPOSE_BLOCKS };
        $shortname =~ s{/([^/]+)$}{} || last;
        $blockname = length $blockname ? "$1/$blockname" : $1;
    }
        
    $self->throw(Template::Constants::ERROR_FILE, "$name: not found");
}


#------------------------------------------------------------------------
# plugin($name, \@args)
#
# Calls on each of the LOAD_PLUGINS providers in turn to fetch() (i.e. load
# and instantiate) a plugin of the specified name.  Additional parameters 
# passed are propagated to the new() constructor for the plugin.  
# Returns a reference to a new plugin object or other reference.  On 
# error, undef is returned and the appropriate error message is set for
# subsequent retrieval via error().
#------------------------------------------------------------------------

sub plugin {
    my ($self, $name, $args) = @_;
    my ($provider, $plugin, $error);
    
    $self->debug("plugin($name, ", defined $args ? @$args : '[ ]', ')')
        if $self->{ DEBUG };
    
    # request the named plugin from each of the LOAD_PLUGINS providers in turn
    foreach my $provider (@{ $self->{ LOAD_PLUGINS } }) {
        ($plugin, $error) = $provider->fetch($name, $args, $self);
        return $plugin unless $error;
        if ($error == Template::Constants::STATUS_ERROR) {
            $self->throw($plugin) if ref $plugin;
            $self->throw(Template::Constants::ERROR_PLUGIN, $plugin);
        }
    }
    
    $self->throw(Template::Constants::ERROR_PLUGIN, "$name: plugin not found");
}


#------------------------------------------------------------------------
# filter($name, \@args, $alias)
#
# Similar to plugin() above, but querying the LOAD_FILTERS providers to 
# return filter instances.  An alias may be provided which is used to
# save the returned filter in a local cache.
#------------------------------------------------------------------------

sub filter {
    my ($self, $name, $args, $alias) = @_;
    my ($provider, $filter, $error);
    
    $self->debug("filter($name, ", 
                 defined $args  ? @$args : '[ ]', 
                 defined $alias ? $alias : '<no alias>', ')')
        if $self->{ DEBUG };
    
    # use any cached version of the filter if no params provided
    return $filter 
        if ! $args && ! ref $name
            && ($filter = $self->{ FILTER_CACHE }->{ $name });
    
    # request the named filter from each of the FILTERS providers in turn
    foreach my $provider (@{ $self->{ LOAD_FILTERS } }) {
        ($filter, $error) = $provider->fetch($name, $args, $self);
        last unless $error;
        if ($error == Template::Constants::STATUS_ERROR) {
            $self->throw($filter) if ref $filter;
            $self->throw(Template::Constants::ERROR_FILTER, $filter);
        }
        # return $self->error($filter)
        #    if $error == &Template::Constants::STATUS_ERROR;
    }
    
    return $self->error("$name: filter not found")
        unless $filter;
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # commented out by abw on 19 Nov 2001 to fix problem with xmlstyle
    # plugin which may re-define a filter by calling define_filter()
    # multiple times.  With the automatic aliasing/caching below, any
    # new filter definition isn't seen.  Don't think this will cause
    # any problems as filters explicitly supplied with aliases will
    # still work as expected.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # alias defaults to name if undefined
    # $alias = $name
    #     unless defined($alias) or ref($name) or $args;

    # cache FILTER if alias is valid
    $self->{ FILTER_CACHE }->{ $alias } = $filter
        if $alias;

    return $filter;
}


#------------------------------------------------------------------------
# view(\%config)
# 
# Create a new Template::View bound to this context.
#------------------------------------------------------------------------

sub view {
    my $self = shift;
    require Template::View;
    return $VIEW_CLASS->new($self, @_)
        || $self->throw(&Template::Constants::ERROR_VIEW, 
                        $VIEW_CLASS->error);
}


#------------------------------------------------------------------------
# process($template, \%params)         [% PROCESS template var=val ... %]
# process($template, \%params, $local) [% INCLUDE template var=val ... %]
#
# Processes the template named or referenced by the first parameter.
# The optional second parameter may reference a hash array of variable
# definitions.  These are set before the template is processed by
# calling update() on the stash.  Note that, unless the third parameter
# is true, the context is not localised and these, and any other
# variables set in the template will retain their new values after this
# method returns.  The third parameter is in place so that this method
# can handle INCLUDE calls: the stash will be localized.
#
# Returns the output of processing the template.  Errors are thrown
# as Template::Exception objects via die().  
#------------------------------------------------------------------------

sub process {
    my ($self, $template, $params, $localize) = @_;
    my ($trim, $blocks) = @$self{ qw( TRIM BLOCKS ) };
    my (@compiled, $name, $compiled);
    my ($stash, $component, $tblocks, $error, $tmpout);
    my $output = '';
    
    $template = [ $template ] unless ref $template eq 'ARRAY';
    
    $self->debug("process([ ", join(', '), @$template, ' ], ', 
                 defined $params ? $params : '<no params>', ', ', 
                 $localize ? '<localized>' : '<unlocalized>', ')')
        if $self->{ DEBUG };
    
    # fetch compiled template for each name specified
    foreach $name (@$template) {
        push(@compiled, $self->template($name));
    }

    if ($localize) {
        # localise the variable stash with any parameters passed
        $stash = $self->{ STASH } = $self->{ STASH }->clone($params);
    } else {
        # update stash with any new parameters passed
        $self->{ STASH }->update($params);
        $stash = $self->{ STASH };
    }

    eval {
        # save current component
        eval { $component = $stash->get('component') };

        foreach $name (@$template) {
            $compiled = shift @compiled;
            my $element = ref $compiled eq 'CODE' 
                ? { (name => (ref $name ? '' : $name), modtime => time()) }
                : $compiled;

            if (blessed($component) && $component->isa(DOCUMENT)) {
                $element->{ caller } = $component->{ name };
                $element->{ callers } = $component->{ callers } || [];
                push(@{$element->{ callers }}, $element->{ caller });
            }

            $stash->set('component', $element);
            
            unless ($localize) {
                # merge any local blocks defined in the Template::Document
                # into our local BLOCKS cache
                @$blocks{ keys %$tblocks } = values %$tblocks
                    if (blessed($compiled) && $compiled->isa(DOCUMENT))
                    && ($tblocks = $compiled->blocks);
            }
            
            if (ref $compiled eq 'CODE') {
                $tmpout = &$compiled($self);
            }
            elsif (ref $compiled) {
                $tmpout = $compiled->process($self);
            }
            else {
                $self->throw('file', 
                             "invalid template reference: $compiled");
            }
            
            if ($trim) {
                for ($tmpout) {
                    s/^\s+//;
                    s/\s+$//;
                }
            }
            $output .= $tmpout;

            # pop last item from callers.  
            # NOTE - this will not be called if template throws an 
            # error.  The whole issue of caller and callers should be 
            # revisited to try and avoid putting this info directly into
            # the component data structure.  Perhaps use a local element
            # instead?

            pop(@{$element->{ callers }})
                if (blessed($component) && $component->isa(DOCUMENT));
        }
        $stash->set('component', $component);
    };
    $error = $@;
    
    if ($localize) {
        # ensure stash is delocalised before dying
        $self->{ STASH } = $self->{ STASH }->declone();
    }
    
    $self->throw(ref $error 
                 ? $error : (Template::Constants::ERROR_FILE, $error))
        if $error;
    
    return $output;
}


#------------------------------------------------------------------------
# include($template, \%params)    [% INCLUDE template   var = val, ... %]
#
# Similar to process() above but processing the template in a local 
# context.  Any variables passed by reference to a hash as the second
# parameter will be set before the template is processed and then 
# revert to their original values before the method returns.  Similarly,
# any changes made to non-global variables within the template will 
# persist only until the template is processed.
#
# Returns the output of processing the template.  Errors are thrown
# as Template::Exception objects via die().  
#------------------------------------------------------------------------

sub include {
    my ($self, $template, $params) = @_;
    return $self->process($template, $params, 'localize me!');
}

#------------------------------------------------------------------------
# insert($file)
#
# Insert the contents of a file without parsing.
#------------------------------------------------------------------------

sub insert {
    my ($self, $file) = @_;
    my ($prefix, $providers, $text, $error);
    my $output = '';

    my $files = ref $file eq 'ARRAY' ? $file : [ $file ];

    $self->debug("insert([ ", join(', '), @$files, " ])") 
        if $self->{ DEBUG };


    FILE: foreach $file (@$files) {
        my $name = $file;

        if ($^O eq 'MSWin32') {
            # let C:/foo through
            $prefix = $1 if $name =~ s/^(\w{2,})://o;
        }
        else {
            $prefix = $1 if $name =~ s/^(\w+)://;
        }

        if (defined $prefix) {
            $providers = $self->{ PREFIX_MAP }->{ $prefix } 
                || return $self->throw(Template::Constants::ERROR_FILE,
                    "no providers for file prefix '$prefix'");
        }
        else {
            $providers = $self->{ PREFIX_MAP }->{ default }
                || $self->{ LOAD_TEMPLATES };
        }

        foreach my $provider (@$providers) {
            ($text, $error) = $provider->load($name, $prefix);
            next FILE unless $error;
            if ($error == Template::Constants::STATUS_ERROR) {
                $self->throw($text) if ref $text;
                $self->throw(Template::Constants::ERROR_FILE, $text);
            }
        }
        $self->throw(Template::Constants::ERROR_FILE, "$file: not found");
    }
    continue {
        $output .= $text;
    }
    return $output;
}


#------------------------------------------------------------------------
# throw($type, $info, \$output)          [% THROW errtype "Error info" %]
#
# Throws a Template::Exception object by calling die().  This method
# may be passed a reference to an existing Template::Exception object;
# a single value containing an error message which is used to
# instantiate a Template::Exception of type 'undef'; or a pair of
# values representing the exception type and info from which a
# Template::Exception object is instantiated.  e.g.
#
#   $context->throw($exception);
#   $context->throw("I'm sorry Dave, I can't do that");
#   $context->throw('denied', "I'm sorry Dave, I can't do that");
#
# An optional third parameter can be supplied in the last case which 
# is a reference to the current output buffer containing the results
# of processing the template up to the point at which the exception 
# was thrown.  The RETURN and STOP directives, for example, use this 
# to propagate output back to the user, but it can safely be ignored
# in most cases.
# 
# This method rides on a one-way ticket to die() oblivion.  It does not 
# return in any real sense of the word, but should get caught by a 
# surrounding eval { } block (e.g. a BLOCK or TRY) and handled 
# accordingly, or returned to the caller as an uncaught exception.
#------------------------------------------------------------------------

sub throw {
    my ($self, $error, $info, $output) = @_;
    local $" = ', ';

    # die! die! die!
    if (blessed($error) && $error->isa(EXCEPTION)) {
        die $error;
    }
    elsif (blessed($error) && $error->isa(BADGER_EXCEPTION)) {
        # convert a Badger::Exception to a Template::Exception so that
        # things continue to work during the transition to Badger
        die EXCEPTION->new($error->type, $error->info);
    }
    elsif (defined $info) {
        die (EXCEPTION->new($error, $info, $output));
    }
    else {
        $error ||= '';
        die (EXCEPTION->new('undef', $error, $output));
    }

    # not reached
}


#------------------------------------------------------------------------
# catch($error, \$output)
#
# Called by various directives after catching an error thrown via die()
# from within an eval { } block.  The first parameter contains the errror
# which may be a sanitized reference to a Template::Exception object
# (such as that raised by the throw() method above, a plugin object, 
# and so on) or an error message thrown via die from somewhere in user
# code.  The latter are coerced into 'undef' Template::Exception objects.
# Like throw() above, a reference to a scalar may be passed as an
# additional parameter to represent the current output buffer
# localised within the eval block.  As exceptions are thrown upwards
# and outwards from nested blocks, the catch() method reconstructs the
# correct output buffer from these fragments, storing it in the
# exception object for passing further onwards and upwards.
#
# Returns a reference to a Template::Exception object..
#------------------------------------------------------------------------

sub catch {
    my ($self, $error, $output) = @_;

    if ( blessed($error) 
      && ( $error->isa(EXCEPTION) || $error->isa(BADGER_EXCEPTION) ) ) {
        $error->text($output) if $output;
        return $error;
    }
    else {
        return EXCEPTION->new('undef', $error, $output);
    }
}


#------------------------------------------------------------------------
# localise(\%params)
# delocalise()
#
# The localise() method creates a local copy of the current stash,
# allowing the existing state of variables to be saved and later 
# restored via delocalise().
#
# A reference to a hash array may be passed containing local variable 
# definitions which should be added to the cloned namespace.  These 
# values persist until delocalisation.
#------------------------------------------------------------------------

sub localise {
    my $self = shift;
    $self->{ STASH } = $self->{ STASH }->clone(@_);
}

sub delocalise {
    my $self = shift;
    $self->{ STASH } = $self->{ STASH }->declone();
}


#------------------------------------------------------------------------
# visit($document, $blocks)
#
# Each Template::Document calls the visit() method on the context
# before processing itself.  It passes a reference to the hash array
# of named BLOCKs defined within the document, allowing them to be 
# added to the internal BLKSTACK list which is subsequently used by
# template() to resolve templates.
# from a provider.
#------------------------------------------------------------------------

sub visit {
    my ($self, $document, $blocks) = @_;
    unshift(@{ $self->{ BLKSTACK } }, $blocks)
}


#------------------------------------------------------------------------
# leave()
#
# The leave() method is called when the document has finished
# processing itself.  This removes the entry from the BLKSTACK list
# that was added visit() above.  For persistence of BLOCK definitions,
# the process() method (i.e. the PROCESS directive) does some extra
# magic to copy BLOCKs into a shared hash.
#------------------------------------------------------------------------

sub leave {
    my $self = shift;
    shift(@{ $self->{ BLKSTACK } });
}


#------------------------------------------------------------------------
# define_block($name, $block)
#
# Adds a new BLOCK definition to the local BLOCKS cache.  $block may
# be specified as a reference to a sub-routine or Template::Document
# object or as text which is compiled into a template.  Returns a true
# value (the $block reference or compiled block reference) if
# successful or undef on failure.  Call error() to retrieve the
# relevent error message (i.e. compilation failure).
#------------------------------------------------------------------------

sub define_block {
    my ($self, $name, $block) = @_;
    $block = $self->template(\$block)
    || return undef
        unless ref $block;
    $self->{ BLOCKS }->{ $name } = $block;
}


#------------------------------------------------------------------------
# define_filter($name, $filter, $is_dynamic)
#
# Adds a new FILTER definition to the local FILTER_CACHE.
#------------------------------------------------------------------------

sub define_filter {
    my ($self, $name, $filter, $is_dynamic) = @_;
    my ($result, $error);
    $filter = [ $filter, 1 ] if $is_dynamic;

    foreach my $provider (@{ $self->{ LOAD_FILTERS } }) {
    ($result, $error) = $provider->store($name, $filter);
    return 1 unless $error;
    $self->throw(&Template::Constants::ERROR_FILTER, $result)
        if $error == &Template::Constants::STATUS_ERROR;
    }
    $self->throw(&Template::Constants::ERROR_FILTER, 
         "FILTER providers declined to store filter $name");
}

sub define_view {
    my ($self, $name, $params) = @_;
    my $base;

    if (defined $params->{ base }) {
        my $base = $self->{ STASH }->get($params->{ base });

        return $self->throw(
            &Template::Constants::ERROR_VIEW, 
            "view base is not defined: $params->{ base }"
        ) unless $base;

        return $self->throw(
            &Template::Constants::ERROR_VIEW, 
            "view base is not a $VIEW_CLASS object: $params->{ base } => $base"
        ) unless blessed($base) && $base->isa($VIEW_CLASS);
        
        $params->{ base } = $base;
    }
    my $view = $self->view($params);
    $view->seal();
    $self->{ STASH }->set($name, $view);
}

sub define_views {
    my ($self, $views) = @_;
    
    # a list reference is better because the order is deterministic (and so
    # allows an earlier VIEW to be the base for a later VIEW), but we'll 
    # accept a hash reference and assume that the user knows the order of
    # processing is undefined
    $views = [ %$views ] 
        if ref $views eq 'HASH';
    
    # make of copy so we don't destroy the original list reference
    my @items = @$views;
    my ($name, $view);
    
    while (@items) {
        $self->define_view(splice(@items, 0, 2));
    }
}


#------------------------------------------------------------------------
# reset()
# 
# Reset the state of the internal BLOCKS hash to clear any BLOCK 
# definitions imported via the PROCESS directive.  Any original 
# BLOCKS definitions passed to the constructor will be restored.
#------------------------------------------------------------------------

sub reset {
    my ($self, $blocks) = @_;
    $self->{ BLKSTACK } = [ ];
    $self->{ BLOCKS   } = { %{ $self->{ INIT_BLOCKS } } };
}


#------------------------------------------------------------------------
# stash()
#
# Simple accessor methods to return the STASH values.  This is likely
# to be called quite often so we provide a direct method rather than
# relying on the slower AUTOLOAD.
#------------------------------------------------------------------------

sub stash {
    return $_[0]->{ STASH };
}


#------------------------------------------------------------------------
# define_vmethod($type, $name, \&sub)
#
# Passes $type, $name, and &sub on to stash->define_vmethod().
#------------------------------------------------------------------------
sub define_vmethod {
    my $self = shift;
    $self->stash->define_vmethod(@_);
}


#------------------------------------------------------------------------
# debugging($command, @args, \%params)
#
# Method for controlling the debugging status of the context.  The first
# argument can be 'on' or 'off' to enable/disable debugging, 'format'
# to define the format of the debug message, or 'msg' to generate a 
# debugging message reporting the file, line, message text, etc., 
# according to the current debug format.
#------------------------------------------------------------------------

sub debugging {
    my $self = shift;
    my $hash = ref $_[-1] eq 'HASH' ? pop : { };
    my @args = @_;

#    print "*** debug(@args)\n";
    if (@args) {
    if ($args[0] =~ /^on|1$/i) {
        $self->{ DEBUG_DIRS } = 1;
        shift(@args);
    }
    elsif ($args[0] =~ /^off|0$/i) {
        $self->{ DEBUG_DIRS } = 0;
        shift(@args);
    }
    }

    if (@args) {
    if ($args[0] =~ /^msg$/i) {
            return unless $self->{ DEBUG_DIRS };
        my $format = $self->{ DEBUG_FORMAT };
        $format = $DEBUG_FORMAT unless defined $format;
        $format =~ s/\$(\w+)/$hash->{ $1 }/ge;
        return $format;
    }
    elsif ($args[0] =~ /^format$/i) {
        $self->{ DEBUG_FORMAT } = $args[1];
    }
    # else ignore
    }

    return '';
}


#------------------------------------------------------------------------
# AUTOLOAD
#
# Provides pseudo-methods for read-only access to various internal 
# members.  For example, templates(), plugins(), filters(),
# eval_perl(), load_perl(), etc.  These aren't called very often, or
# may never be called at all.
#------------------------------------------------------------------------

sub AUTOLOAD {
    my $self   = shift;
    my $method = $AUTOLOAD;
    my $result;

    $method =~ s/.*:://;
    return if $method eq 'DESTROY';

    warn "no such context method/member: $method\n"
    unless defined ($result = $self->{ uc $method });

    return $result;
}


#------------------------------------------------------------------------
# DESTROY
#
# Stash may contain references back to the Context via macro closures,
# etc.  This breaks the circular references. 
#------------------------------------------------------------------------

sub DESTROY {
    my $self = shift;
    undef $self->{ STASH };
}



#========================================================================
#                     -- PRIVATE METHODS --
#========================================================================

#------------------------------------------------------------------------
# _init(\%config)
#
# Initialisation method called by Template::Base::new()
#------------------------------------------------------------------------

sub _init {
    my ($self, $config) = @_;
    my ($name, $item, $method, $block, $blocks);
    my @itemlut = ( 
        LOAD_TEMPLATES => 'provider',
        LOAD_PLUGINS   => 'plugins',
        LOAD_FILTERS   => 'filters' 
    );

    # LOAD_TEMPLATE, LOAD_PLUGINS, LOAD_FILTERS - lists of providers
    while (($name, $method) = splice(@itemlut, 0, 2)) {
        $item = $config->{ $name } 
            || Template::Config->$method($config)
            || return $self->error($Template::Config::ERROR);
        $self->{ $name } = ref $item eq 'ARRAY' ? $item : [ $item ];
    }

    my $providers  = $self->{ LOAD_TEMPLATES };
    my $prefix_map = $self->{ PREFIX_MAP } = $config->{ PREFIX_MAP } || { };
    while (my ($key, $val) = each %$prefix_map) {
        $prefix_map->{ $key } = [ ref $val ? $val : 
                                  map { $providers->[$_] } split(/\D+/, $val) ]
                                  unless ref $val eq 'ARRAY';
    }

    # STASH
    $self->{ STASH } = $config->{ STASH } || do {
        my $predefs  = $config->{ VARIABLES } 
            || $config->{ PRE_DEFINE } 
            || { };

        # hack to get stash to know about debug mode
        $predefs->{ _DEBUG } = ( ($config->{ DEBUG } || 0)
                                 & &Template::Constants::DEBUG_UNDEF ) ? 1 : 0
                                 unless defined $predefs->{ _DEBUG };
        $predefs->{ _STRICT } = $config->{ STRICT };
        
        Template::Config->stash($predefs)
            || return $self->error($Template::Config::ERROR);
    };
    
    # compile any template BLOCKS specified as text
    $blocks = $config->{ BLOCKS } || { };
    $self->{ INIT_BLOCKS } = $self->{ BLOCKS } = { 
        map {
            $block = $blocks->{ $_ };
            $block = $self->template(\$block)
                || return undef
                unless ref $block;
            ($_ => $block);
        } 
        keys %$blocks
    };

    # define any VIEWS
    $self->define_views( $config->{ VIEWS } )
        if $config->{ VIEWS };

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # RECURSION - flag indicating is recursion into templates is supported
    # EVAL_PERL - flag indicating if PERL blocks should be processed
    # TRIM      - flag to remove leading and trailing whitespace from output
    # BLKSTACK  - list of hashes of BLOCKs defined in current template(s)
    # CONFIG    - original configuration hash
    # EXPOSE_BLOCKS - make blocks visible as pseudo-files
    # DEBUG_FORMAT  - format for generating template runtime debugging messages
    # DEBUG         - format for generating template runtime debugging messages

    $self->{ RECURSION } = $config->{ RECURSION } || 0;
    $self->{ EVAL_PERL } = $config->{ EVAL_PERL } || 0;
    $self->{ TRIM      } = $config->{ TRIM } || 0;
    $self->{ BLKSTACK  } = [ ];
    $self->{ CONFIG    } = $config;
    $self->{ EXPOSE_BLOCKS } = defined $config->{ EXPOSE_BLOCKS }
                                     ? $config->{ EXPOSE_BLOCKS } 
                                     : 0;

    $self->{ DEBUG_FORMAT  } =  $config->{ DEBUG_FORMAT };
    $self->{ DEBUG_DIRS    } = ($config->{ DEBUG } || 0) 
                               & Template::Constants::DEBUG_DIRS;
    $self->{ DEBUG } = defined $config->{ DEBUG } 
        ? $config->{ DEBUG } & ( Template::Constants::DEBUG_CONTEXT
                               | Template::Constants::DEBUG_FLAGS )
        : $DEBUG;

    return $self;
}


#------------------------------------------------------------------------
# _dump()
#
# Debug method which returns a string representing the internal state
# of the context object.
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    my $output = "[Template::Context] {\n";
    my $format = "    %-16s => %s\n";
    my $key;

    foreach $key (qw( RECURSION EVAL_PERL TRIM )) {
    $output .= sprintf($format, $key, $self->{ $key });
    }
    foreach my $pname (qw( LOAD_TEMPLATES LOAD_PLUGINS LOAD_FILTERS )) {
    my $provtext = "[\n";
    foreach my $prov (@{ $self->{ $pname } }) {
        $provtext .= $prov->_dump();
#       $provtext .= ",\n";
    }
    $provtext =~ s/\n/\n        /g;
    $provtext =~ s/\s+$//;
    $provtext .= ",\n    ]";
    $output .= sprintf($format, $pname, $provtext);
    }
    $output .= sprintf($format, STASH => $self->{ STASH }->_dump());
    $output .= '}';
    return $output;
}


1;

__END__

=head1 NAME

Template::Context - Runtime context in which templates are processed

=head1 SYNOPSIS

    use Template::Context;
    
    # constructor
    $context = Template::Context->new(\%config)
        || die $Template::Context::ERROR;
    
    # fetch (load and compile) a template
    $template = $context->template($template_name);
    
    # fetch (load and instantiate) a plugin object
    $plugin = $context->plugin($name, \@args);
    
    # fetch (return or create) a filter subroutine
    $filter = $context->filter($name, \@args, $alias);
    
    # process/include a template, errors are thrown via die()
    $output = $context->process($template, \%vars);
    $output = $context->include($template, \%vars);
    
    # raise an exception via die()
    $context->throw($error_type, $error_message, \$output_buffer);
    
    # catch an exception, clean it up and fix output buffer
    $exception = $context->catch($exception, \$output_buffer);
    
    # save/restore the stash to effect variable localisation
    $new_stash = $context->localise(\%vars);
    $old_stash = $context->delocalise();
    
    # add new BLOCK or FILTER definitions
    $context->define_block($name, $block);
    $context->define_filter($name, \&filtersub, $is_dynamic);
    
    # reset context, clearing any imported BLOCK definitions
    $context->reset();
    
    # methods for accessing internal items
    $stash     = $context->stash();
    $tflag     = $context->trim();
    $epflag    = $context->eval_perl();
    $providers = $context->templates();
    $providers = $context->plugins();
    $providers = $context->filters();
    ...

=head1 DESCRIPTION

The C<Template::Context> module defines an object class for representing
a runtime context in which templates are processed.  It provides an
interface to the fundamental operations of the Template Toolkit
processing engine through which compiled templates (i.e. Perl code
constructed from the template source) can process templates, load
plugins and filters, raise exceptions and so on.

A default C<Template::Context> object is created by the L<Template> module.
Any C<Template::Context> options may be passed to the L<Template>
L<new()|Template#new()> constructor method and will be forwarded to the
C<Template::Context> constructor.

    use Template;
    
    my $template = Template->new({
        TRIM      => 1,
        EVAL_PERL => 1,
        BLOCKS    => {
            header => 'This is the header',
            footer => 'This is the footer',
        },
    });

Similarly, the C<Template::Context> constructor will forward all configuration
parameters onto other default objects (e.g. L<Template::Provider>,
L<Template::Plugins>, L<Template::Filters>, etc.) that it may need to
instantiate.

    $context = Template::Context->new({
        INCLUDE_PATH => '/home/abw/templates', # provider option
        TAG_STYLE    => 'html',                # parser option
    });

A C<Template::Context> object (or subclass) can be explicitly instantiated and
passed to the L<Template> L<new()|Template#new()> constructor method as the
C<CONTEXT> configuration item.

    use Template;
    use Template::Context;
    
    my $context  = Template::Context->new({ TRIM => 1 });
    my $template = Template->new({ CONTEXT => $context });

The L<Template> module uses the L<Template::Config>
L<context()|Template::Config#context()> factory method to create a default
context object when required. The C<$Template::Config::CONTEXT> package
variable may be set to specify an alternate context module. This will be
loaded automatically and its L<new()> constructor method called by the
L<context()|Template::Config#context()> factory method when a default context
object is required.

    use Template;
    
    $Template::Config::CONTEXT = 'MyOrg::Template::Context';
    
    my $template = Template->new({
        EVAL_PERL   => 1,
        EXTRA_MAGIC => 'red hot',  # your extra config items
        ...
    });

=head1 METHODS

=head2 new(\%params) 

The C<new()> constructor method is called to instantiate a
C<Template::Context> object. Configuration parameters may be specified as a
HASH reference or as a list of C<name =E<gt> value> pairs.

    my $context = Template::Context->new({
        INCLUDE_PATH => 'header',
        POST_PROCESS => 'footer',
    });
    
    my $context = Template::Context->new( EVAL_PERL => 1 );

The C<new()> method returns a C<Template::Context> object or C<undef> on
error. In the latter case, a relevant error message can be retrieved by the
L<error()|Template::Base#error()> class method or directly from the
C<$Template::Context::ERROR> package variable.

    my $context = Template::Context->new(\%config)
        || die Template::Context->error();
    
    my $context = Template::Context->new(\%config)
        || die $Template::Context::ERROR;

The following configuration items may be specified.  Please see 
L<Template::Manual::Config> for further details.

=head3 VARIABLES

The L<VARIABLES|Template::Manual::Config#VARIABLES> option can be used to
specify a hash array of template variables.

    my $context = Template::Context->new({
        VARIABLES => {
            title   => 'A Demo Page',
            author  => 'Joe Random Hacker',
            version => 3.14,
        },
    };

=head3 BLOCKS

The L<BLOCKS|Template::Manual::Config#BLOCKS> option can be used to pre-define
a default set of template blocks.

    my $context = Template::Context->new({
        BLOCKS => {
            header  => 'The Header.  [% title %]',
            footer  => sub { return $some_output_text },
            another => Template::Document->new({ ... }),
        },
    }); 

=head3 VIEWS

The L<VIEWS|Template::Manual::Config#VIEWS> option can be used to pre-define 
one or more L<Template::View> objects.

    my $context = Template::Context->new({
        VIEWS => [
            bottom => { prefix => 'bottom/' },
            middle => { prefix => 'middle/', base => 'bottom' },
            top    => { prefix => 'top/',    base => 'middle' },
        ],
    });

=head3 TRIM

The L<TRIM|Template::Manual::Config#TRIM> option can be set to have any
leading and trailing whitespace automatically removed from the output of all
template files and C<BLOCK>s.

example:

    [% BLOCK foo %]
    
    Line 1 of foo
    
    [% END %]
    
    before 
    [% INCLUDE foo %]
    after

output:

    before
    Line 1 of foo
    after

=head3 EVAL_PERL

The L<EVAL_PERL|Template::Manual::Config#EVAL_PERL> is used to indicate if
C<PERL> and/or C<RAWPERL> blocks should be evaluated. It is disabled by
default.

=head3 RECURSION

The L<RECURSION|Template::Manual::Config#RECURSION> can be set to 
allow templates to recursively process themselves, either directly
(e.g. template C<foo> calls C<INCLUDE foo>) or indirectly (e.g. 
C<foo> calls C<INCLUDE bar> which calls C<INCLUDE foo>).

=head3 LOAD_TEMPLATES

The L<LOAD_TEMPLATES|Template::Manual::Config#LOAD_TEMPLATES> option can be
used to provide a reference to a list of L<Template::Provider> objects or
sub-classes thereof which will take responsibility for loading and compiling
templates.

    my $context = Template::Context->new({
        LOAD_TEMPLATES => [
            MyOrg::Template::Provider->new({ ... }),
            Template::Provider->new({ ... }),
        ],
    });

=head3 LOAD_PLUGINS

The L<LOAD_PLUGINS|Template::Manual::Config#LOAD_PLUGINS> options can be used
to specify a list of provider objects responsible for loading and
instantiating template plugin objects.

    my $context = Template::Context->new({
        LOAD_PLUGINS => [
            MyOrg::Template::Plugins->new({ ... }),
            Template::Plugins->new({ ... }),
        ],
    });

=head3 LOAD_FILTERS

The L<LOAD_FILTERS|Template::Manual::Config#LOAD_FILTERS> option can be used
to specify a list of provider objects for returning and/or creating filter
subroutines.

    my $context = Template::Context->new({
        LOAD_FILTERS => [
            MyTemplate::Filters->new(),
            Template::Filters->new(),
        ],
    });

=head3 STASH

The L<STASH|Template::Manual::Config#STASH> option can be used to 
specify a L<Template::Stash> object or sub-class which will take
responsibility for managing template variables.  

    my $stash = MyOrg::Template::Stash->new({ ... });
    my $context = Template::Context->new({
        STASH => $stash,
    });

=head3 DEBUG

The L<DEBUG|Template::Manual::Config#DEBUG> option can be used to enable
various debugging features of the L<Template::Context> module.

    use Template::Constants qw( :debug );
    
    my $template = Template->new({
        DEBUG => DEBUG_CONTEXT | DEBUG_DIRS,
    });

=head2 template($name) 

Returns a compiled template by querying each of the L<LOAD_TEMPLATES> providers
(instances of L<Template::Provider>, or sub-class) in turn.  

    $template = $context->template('header');

On error, a L<Template::Exception> object of type 'C<file>' is thrown via
C<die()>.  This can be caught by enclosing the call to C<template()> in an
C<eval> block and examining C<$@>.

    eval { $template = $context->template('header') };
    if ($@) {
        print "failed to fetch template: $@\n";
    }

=head2 plugin($name, \@args)

Instantiates a plugin object by querying each of the L<LOAD_PLUGINS>
providers. The default L<LOAD_PLUGINS> provider is a L<Template::Plugins>
object which attempts to load plugin modules, according the various
configuration items such as L<PLUGIN_BASE|Template::Plugins#PLUGIN_BASE>,
L<LOAD_PERL|Template::Plugins#LOAD_PERL>, etc., and then instantiate an object
via L<new()|Template::Plugin#new()>. A reference to a list of constructor
arguments may be passed as the second parameter. These are forwarded to the
plugin constructor.

Returns a reference to a plugin (which is generally an object, but
doesn't have to be).  Errors are thrown as L<Template::Exception> objects
with the type set to 'C<plugin>'.

    $plugin = $context->plugin('DBI', 'dbi:msql:mydbname');

=head2 filter($name, \@args, $alias)

Instantiates a filter subroutine by querying the L<LOAD_FILTERS> providers.
The default L<LOAD_FILTERS> provider is a L<Template::Filters> object.

Additional arguments may be passed by list reference along with an optional
alias under which the filter will be cached for subsequent use. The filter is
cached under its own C<$name> if C<$alias> is undefined. Subsequent calls to
C<filter($name)> will return the cached entry, if defined. Specifying arguments
bypasses the caching mechanism and always creates a new filter. Errors are
thrown as L<Template::Exception> objects with the type set to 'C<filter>'.

    # static filter (no args)
    $filter = $context->filter('html');
    
    # dynamic filter (args) aliased to 'padright'
    $filter = $context->filter('format', '%60s', 'padright');
    
    # retrieve previous filter via 'padright' alias
    $filter = $context->filter('padright');

=head2 process($template, \%vars)

Processes a template named or referenced by the first parameter and returns
the output generated.  An optional reference to a hash array may be passed
as the second parameter, containing variable definitions which will be set
before the template is processed.  The template is processed in the current
context, with no localisation of variables performed.   Errors are thrown
as L<Template::Exception> objects via C<die()>.  

    $output = $context->process('header', { title => 'Hello World' });

=head2 include($template, \%vars)

Similar to L<process()>, but using localised variables.  Changes made to
any variables will only persist until the C<include()> method completes.

    $output = $context->include('header', { title => 'Hello World' });

=head2 throw($error_type, $error_message, \$output)

Raises an exception in the form of a L<Template::Exception> object by calling
C<die()>. This method may be passed a reference to an existing
L<Template::Exception> object; a single value containing an error message
which is used to instantiate a L<Template::Exception> of type 'C<undef>'; or a
pair of values representing the exception C<type> and C<info> from which a
L<Template::Exception> object is instantiated. e.g.

    $context->throw($exception);
    $context->throw("I'm sorry Dave, I can't do that");
    $context->throw('denied', "I'm sorry Dave, I can't do that");

The optional third parameter may be a reference to the current output
buffer.  This is then stored in the exception object when created,
allowing the catcher to examine and use the output up to the point at
which the exception was raised.

    $output .= 'blah blah blah';
    $output .= 'more rhubarb';
    $context->throw('yack', 'Too much yacking', \$output);

=head2 catch($exception, \$output)

Catches an exception thrown, either as a reference to a L<Template::Exception>
object or some other value. In the latter case, the error string is promoted
to a L<Template::Exception> object of 'C<undef>' type. This method also
accepts a reference to the current output buffer which is passed to the
L<Template::Exception> constructor, or is appended to the output buffer stored
in an existing L<Template::Exception> object, if unique (i.e. not the same
reference). By this process, the correct state of the output buffer can be
reconstructed for simple or nested throws.

=head2 define_block($name, $block)

Adds a new block definition to the internal L<BLOCKS> cache.  The first 
argument should contain the name of the block and the second a reference
to a L<Template::Document> object or template sub-routine, or template text
which is automatically compiled into a template sub-routine.  

Returns a true value (the sub-routine or L<Template::Document> reference) on
success or undef on failure. The relevant error message can be retrieved by
calling the L<error()|Template::Base#error()> method.

=head2 define_filter($name, \&filter, $is_dynamic)

Adds a new filter definition by calling the
L<store()|Template::Filters#store()> method on each of the L<LOAD_FILTERS>
providers until accepted (in the usual case, this is accepted straight away by
the one and only L<Template::Filters> provider). The first argument should
contain the name of the filter and the second a reference to a filter
subroutine. The optional third argument can be set to any true value to
indicate that the subroutine is a dynamic filter factory. 

Returns a true value or throws a 'C<filter>' exception on error.

=head2 define_view($name, \%params)

This method allows you to define a named L<view|Template::View>.

    $context->define_view( 
        my_view => { 
            prefix => 'my_templates/' 
        } 
    );

The view is then accessible as a template variable.

    [% my_view.print(some_data) %]

=head2 define_views($views)

This method allows you to define multiple named L<views|Template::View>.
A reference to a hash array or list reference should be passed as an argument.

    $context->define_view({     # hash reference
        my_view_one => { 
            prefix => 'my_templates_one/' 
        },
        my_view_two => { 
            prefix => 'my_templates_two/' 
        } 
    });

If you're defining multiple views of which one or more are based on other 
views in the same definition then you should pass them as a list reference.
This ensures that they get created in the right order (Perl does not preserve
the order of items defined in a hash reference so you can't guarantee that
your base class view will be defined before your subclass view).

    $context->define_view([     # list referenence
        my_view_one => {
            prefix => 'my_templates_one/' 
        },
        my_view_two => { 
            prefix => 'my_templates_two/' ,
            base   => 'my_view_one',
        } 
    ]);

The views are then accessible as template variables.

    [% my_view_one.print(some_data) %]
    [% my_view_two.print(some_data) %]

See also the L<VIEWS> option.

=head2 localise(\%vars)

Clones the stash to create a context with localised variables.  Returns a 
reference to the newly cloned stash object which is also stored
internally.

    $stash = $context->localise();

=head2 delocalise()

Restore the stash to its state prior to localisation.

    $stash = $context->delocalise();

=head2 visit(\%blocks)

This method is called by L<Template::Document> objects immediately before
they process their content.  It is called to register any local C<BLOCK>
definitions with the context object so that they may be subsequently
delivered on request.

=head2 leave()

Compliment to the L<visit()> method. Called by L<Template::Document> objects
immediately after they process their content.

=head2 reset()

Clears the local L<BLOCKS> cache of any C<BLOCK> definitions.  Any initial set of
L<BLOCKS> specified as a configuration item to the constructor will be reinstated.

=head2 AUTOLOAD

An C<AUTOLOAD> method provides access to context configuration items.

    $stash     = $context->stash();
    $tflag     = $context->trim();
    $epflag    = $context->eval_perl();
    ...

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template>, L<Template::Document>, L<Template::Exception>,
L<Template::Filters>, L<Template::Plugins>, L<Template::Provider>,
L<Template::Service>, L<Template::Stash>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
