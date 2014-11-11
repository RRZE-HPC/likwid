#============================================================= -*-Perl-*-
#
# Template::View
#
# DESCRIPTION
#   A custom view of a template processing context.  Can be used to 
#   implement custom "skins".
#
# AUTHOR
#   Andy Wardley   <abw@kfs.org>
#
# COPYRIGHT
#   Copyright (C) 2000 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
# TODO
#  * allowing print to have a hash ref as final args will cause problems
#    if you do this: [% view.print(hash1, hash2, hash3) %].  Current
#    work-around is to do [% view.print(hash1); view.print(hash2); 
#    view.print(hash3) %] or [% view.print(hash1, hash2, hash3, { }) %]
#
#============================================================================

package Template::View;

use strict;
use warnings;
use base 'Template::Base';

our $VERSION  = 2.91;
our $DEBUG    = 0 unless defined $DEBUG;
our @BASEARGS = qw( context );
our $AUTOLOAD;
our $MAP = {
    HASH    => 'hash',
    ARRAY   => 'list',
    TEXT    => 'text',
    default => '',
};


#------------------------------------------------------------------------
# _init(\%config)
#
# Initialisation method called by the Template::Base class new() 
# constructor.  $self->{ context } has already been set, by virtue of
# being named in @BASEARGS.  Remaining config arguments are presented 
# as a hash reference.
#------------------------------------------------------------------------

sub _init {
    my ($self, $config) = @_;

    # move 'context' somewhere more private
    $self->{ _CONTEXT } = $self->{ context };
    delete $self->{ context };
    
    # generate table mapping object types to templates
    my $map = $config->{ map } || { };
    $map->{ default } = $config->{ default } unless defined $map->{ default };
    $self->{ map } = {
        %$MAP,
        %$map,
    };

    # local BLOCKs definition table
    $self->{ _BLOCKS } = $config->{ blocks } || { };
    
    # name of presentation method which printed objects might provide
    $self->{ method } = defined $config->{ method } 
                              ? $config->{ method } : 'present';
    
    # view is sealed by default preventing variable update after 
    # definition, however we don't actually seal a view until the 
    # END of the view definition
    my $sealed = $config->{ sealed };
    $sealed = 1 unless defined $sealed;
    $self->{ sealed } = $sealed ? 1 : 0;

    # copy remaining config items from $config or set defaults
    foreach my $arg (qw( base prefix suffix notfound silent )) {
        $self->{ $arg } = $config->{ $arg } || '';
    }

    # name of data item used by view()
    $self->{ item } = $config->{ item } || 'item';

    # map methods of form ${include_prefix}_foobar() to include('foobar')?
    $self->{ include_prefix } = $config->{ include_prefix } || 'include_';
    # what about mapping foobar() to include('foobar')?
    $self->{ include_naked  } = defined $config->{ include_naked } 
                                      ? $config->{ include_naked } : 1;

    # map methods of form ${view_prefix}_foobar() to include('foobar')?
    $self->{ view_prefix } = $config->{ view_prefix } || 'view_';
    # what about mapping foobar() to view('foobar')?
    $self->{ view_naked  } = $config->{ view_naked  } || 0;

    # the view is initially unsealed, allowing directives in the initial 
    # view template to create data items via the AUTOLOAD; once sealed via
    # call to seal(), the AUTOLOAD will not update any internal items.
    delete @$config{ qw( base method map default prefix suffix notfound item 
                         include_prefix include_naked silent sealed
                         view_prefix view_naked blocks ) };
    $config = { %{ $self->{ base }->{ data } }, %$config }
        if $self->{ base };
    $self->{ data   } = $config;
    $self->{ SEALED } = 0;

    return $self;
}


#------------------------------------------------------------------------
# seal()
# unseal()
#
# Seal or unseal the view to allow/prevent new datat items from being
# automatically created by the AUTOLOAD method.
#------------------------------------------------------------------------

sub seal {
    my $self = shift;
    $self->{ SEALED } = $self->{ sealed };
}

sub unseal {
    my $self = shift;
    $self->{ SEALED } = 0;
}


#------------------------------------------------------------------------
# clone(\%config)
#
# Cloning method which takes a copy of $self and then applies to it any 
# modifications specified in the $config hash passed as an argument.
# Configuration items may also be specified as a list of "name => $value"
# arguments.  Returns a reference to the cloned Template::View object.
#
# NOTE: may need to copy BLOCKS???
#------------------------------------------------------------------------

sub clone {
    my $self   = shift;
    my $clone  = bless { %$self }, ref $self;
    my $config = ref $_[0] eq 'HASH' ? shift : { @_ };

    # merge maps
    $clone->{ map } = {
        %{ $self->{ map } },
        %{ $config->{ map } || { } },
    };

    # "map => { default=>'xxx' }" can be specified as "default => 'xxx'"
    $clone->{ map }->{ default } = $config->{ default }
        if defined $config->{ default };

    # update any remaining config items
    my @args = qw( base prefix suffix notfound item method include_prefix 
                   include_naked view_prefix view_naked );
    foreach my $arg (@args) {
        $clone->{ $arg } = $config->{ $arg } if defined $config->{ $arg };
    }
    push(@args, qw( default map ));
    delete @$config{ @args };

    # anything left is data
    my $data = $clone->{ data } = { %{ $self->{ data } } };
    @$data{ keys %$config } = values %$config;

    return $clone;
}


#------------------------------------------------------------------------
# print(@items, ..., \%config)
#
# Prints @items in turn by mapping each to an approriate template using 
# the internal 'map' hash.  If an entry isn't found and the item is an 
# object that implements the method named in the internal 'method' item,
# (default: 'present'), then the method will be called passing a reference
# to $self, against which the presenter method may make callbacks (e.g. 
# to view_item()).  If the presenter method isn't implemented, then the 
# 'default' map entry is consulted and used if defined.  The final argument 
# may be a reference to a hash array providing local overrides to the internal
# defaults for various items (prefix, suffix, etc).  In the presence
# of this parameter, a clone of the current object is first made, applying
# any configuration updates, and control is then delegated to it.
#------------------------------------------------------------------------

sub print {
    my $self = shift;

    # if final config hash is specified then create a clone and delegate to it
    # NOTE: potential problem when called print(\%data_hash1, \%data_hash2);
    if ((scalar @_ > 1) && (ref $_[-1] eq 'HASH')) {
        my $cfg = pop @_;
        my $clone = $self->clone($cfg)
            || return;
        return $clone->print(@_) 
            || $self->error($clone->error());
    }
    my ($item, $type, $template, $present);
    my $method = $self->{ method };
    my $map = $self->{ map };
    my $output = '';
    
    # print each argument
    foreach $item (@_) {
        my $newtype;
        
        if (! ($type = ref $item)) {
            # non-references are TEXT
            $type = 'TEXT';
            $template = $map->{ $type };
        }
        elsif (! defined ($template = $map->{ $type })) {
            # no specific map entry for object, maybe it implements a 
            # 'present' (or other) method?
            if ( $method && UNIVERSAL::can($item, $method) ) {
                $present = $item->$method($self);       ## call item method
                # undef returned indicates error, note that we expect 
                # $item to have called error() on the view
                return unless defined $present;
                $output .= $present;
                next;                                   ## NEXT
            }   
            elsif ( ref($item) eq 'HASH' 
                    && defined($newtype = $item->{$method})
                    && defined($template = $map->{"$method=>$newtype"})) {
            }
            elsif ( defined($newtype)
                    && defined($template = $map->{"$method=>*"}) ) {
                $template =~ s/\*/$newtype/;
            }    
            elsif (! ($template = $map->{ default }) ) {
                # default not defined, so construct template name from type
                ($template = $type) =~ s/\W+/_/g;
            }
        }
#       else {
#           $self->DEBUG("defined map type for $type: $template\n");
#       }
        $self->DEBUG("printing view '", $template || '', "', $item\n") if $DEBUG;
        $output .= $self->view($template, $item)
            if $template;
    }
    return $output;
}


#------------------------------------------------------------------------
# view($template, $item, \%vars)
#
# Wrapper around include() which expects a template name, $template,
# followed by a data item, $item, and optionally, a further hash array
# of template variables.  The $item is added as an entry to the $vars
# hash (which is created empty if not passed as an argument) under the
# name specified by the internal 'item' member, which is appropriately
# 'item' by default.  Thus an external object present() method can
# callback against this object method, simply passing a data item to
# be displayed.  The external object doesn't have to know what the
# view expects the item to be called in the $vars hash.
#------------------------------------------------------------------------

sub view {
    my ($self, $template, $item) = splice(@_, 0, 3);
    my $vars = ref $_[0] eq 'HASH' ? shift : { @_ };
    $vars->{ $self->{ item } } = $item if defined $item;
    $self->include($template, $vars);
}


#------------------------------------------------------------------------
# include($template, \%vars)
#
# INCLUDE a template, $template, mapped according to the current prefix,
# suffix, default, etc., where $vars is an optional hash reference 
# containing template variable definitions.  If the template isn't found
# then the method will default to any 'notfound' template, if defined 
# as an internal item.
#------------------------------------------------------------------------

sub include {
    my ($self, $template, $vars) = @_;
    my $context = $self->{ _CONTEXT };

    $template = $self->template($template);

    $vars = { } unless ref $vars eq 'HASH';
    $vars->{ view } ||= $self;

    $context->include( $template, $vars );

# DEBUGGING
#    my $out = $context->include( $template, $vars );
#    print STDERR "VIEW return [$out]\n";
#    return $out;
}


#------------------------------------------------------------------------
# template($template)
#
# Returns a compiled template for the specified template name, according
# to the current configuration parameters.
#------------------------------------------------------------------------

sub template {
    my ($self, $name) = @_;
    my $context = $self->{ _CONTEXT };
    return $context->throw(Template::Constants::ERROR_VIEW,
                           "no view template specified")
        unless $name;

    my $notfound = $self->{ notfound };
    my $base = $self->{ base };
    my ($template, $block, $error);

    return $block
        if ($block = $self->{ _BLOCKS }->{ $name });
    
    # try the named template
    $template = $self->template_name($name);
    $self->DEBUG("looking for $template\n") if $DEBUG;
    eval { $template = $context->template($template) };

    # try asking the base view if not found
    if (($error = $@) && $base) {
        $self->DEBUG("asking base for $name\n") if $DEBUG;
        eval { $template = $base->template($name) };
    }

    # try the 'notfound' template (if defined) if that failed
    if (($error = $@) && $notfound) {
        unless ($template = $self->{ _BLOCKS }->{ $notfound }) {
            $notfound = $self->template_name($notfound);
            $self->DEBUG("not found, looking for $notfound\n") if $DEBUG;
            eval { $template = $context->template($notfound) };

            return $context->throw(Template::Constants::ERROR_VIEW, $error)
                if $@;  # return first error
        }
    }
    elsif ($error) {
        $self->DEBUG("no 'notfound'\n") 
            if $DEBUG;
        return $context->throw(Template::Constants::ERROR_VIEW, $error);
    }
    return $template;
}

    
#------------------------------------------------------------------------
# template_name($template)
#
# Returns the name of the specified template with any appropriate prefix
# and/or suffix added.
#------------------------------------------------------------------------

sub template_name {
    my ($self, $template) = @_;
    $template = $self->{ prefix } . $template . $self->{ suffix }
        if $template;

    $self->DEBUG("template name: $template\n") if $DEBUG;
    return $template;
}


#------------------------------------------------------------------------
# default($val)
#
# Special case accessor to retrieve/update 'default' as an alias for 
# '$map->{ default }'.
#------------------------------------------------------------------------

sub default {
    my $self = shift;
    return @_ ? ($self->{ map }->{ default } = shift) 
              :  $self->{ map }->{ default };
}


#------------------------------------------------------------------------
# AUTOLOAD
#

# Returns/updates public internal data items (i.e. not prefixed '_' or
# '.') or presents a view if the method matches the view_prefix item,
# e.g. view_foo(...) => view('foo', ...).  Similarly, the
# include_prefix is used, if defined, to map include_foo(...) to
# include('foo', ...).  If that fails then the entire method name will
# be used as the name of a template to include iff the include_named
# parameter is set (default: 1).  Last attempt is to match the entire
# method name to a view() call, iff view_naked is set.  Otherwise, a
# 'view' exception is raised reporting the error "no such view member:
# $method".
#------------------------------------------------------------------------

sub AUTOLOAD {
    my $self = shift;
    my $item = $AUTOLOAD;
    $item =~ s/.*:://;
    return if $item eq 'DESTROY';

    if ($item =~ /^[\._]/) {
        return $self->{ _CONTEXT }->throw(Template::Constants::ERROR_VIEW,
                            "attempt to view private member: $item");
    }
    elsif (exists $self->{ $item }) {
        # update existing config item (e.g. 'prefix') if unsealed
        return $self->{ _CONTEXT }->throw(Template::Constants::ERROR_VIEW,
                            "cannot update config item in sealed view: $item")
            if @_ && $self->{ SEALED };
        $self->DEBUG("accessing item: $item\n") if $DEBUG;
        return @_ ? ($self->{ $item } = shift) : $self->{ $item };
    }
    elsif (exists $self->{ data }->{ $item }) {
        # get/update existing data item (must be unsealed to update)
        if (@_ && $self->{ SEALED }) {
            return $self->{ _CONTEXT }->throw(Template::Constants::ERROR_VIEW,
                                  "cannot update item in sealed view: $item")
                unless $self->{ silent };
            # ignore args if silent
            @_ = ();
        }
        $self->DEBUG(@_ ? "updating data item: $item <= $_[0]\n" 
                        : "returning data item: $item\n") if $DEBUG;
        return @_ ? ($self->{ data }->{ $item } = shift) 
                  :  $self->{ data }->{ $item };
    }
    elsif (@_ && ! $self->{ SEALED }) {
        # set data item if unsealed
        $self->DEBUG("setting unsealed data: $item => @_\n") if $DEBUG;
        $self->{ data }->{ $item } = shift;
    }
    elsif ($item =~ s/^$self->{ view_prefix }//) {
        $self->DEBUG("returning view($item)\n") if $DEBUG;
        return $self->view($item, @_);
    }
    elsif ($item =~ s/^$self->{ include_prefix }//) {
        $self->DEBUG("returning include($item)\n") if $DEBUG;
        return $self->include($item, @_);
    }
    elsif ($self->{ include_naked }) {
        $self->DEBUG("returning naked include($item)\n") if $DEBUG;
        return $self->include($item, @_);
    }
    elsif ($self->{ view_naked }) {
        $self->DEBUG("returning naked view($item)\n") if $DEBUG;
        return $self->view($item, @_);
    }
    else {
        return $self->{ _CONTEXT }->throw(Template::Constants::ERROR_VIEW,
                                         "no such view member: $item");
    }
}


1;


__END__

=head1 NAME

Template::View - customised view of a template processing context

=head1 SYNOPSIS

    # define a view
    [% VIEW view
            # some standard args
            prefix        => 'my_', 
            suffix        => '.tt2',
            notfound      => 'no_such_file'
            ...

            # any other data
            title         => 'My View title'
            other_item    => 'Joe Random Data'
            ...
    %]
       # add new data definitions, via 'my' self reference
       [% my.author = "$abw.name <$abw.email>" %]
       [% my.copy   = "&copy; Copyright 2000 $my.author" %]

       # define a local block
       [% BLOCK header %]
       This is the header block, title: [% title or my.title %]
       [% END %]

    [% END %]

    # access data items for view
    [% view.title %]
    [% view.other_item %]

    # access blocks directly ('include_naked' option, set by default)
    [% view.header %]
    [% view.header(title => 'New Title') %]

    # non-local templates have prefix/suffix attached
    [% view.footer %]           # => [% INCLUDE my_footer.tt2 %]

    # more verbose form of block access
    [% view.include( 'header', title => 'The Header Title' ) %]
    [% view.include_header( title => 'The Header Title' ) %]

    # very short form of above ('include_naked' option, set by default)
    [% view.header( title => 'The Header Title' ) %]

    # non-local templates have prefix/suffix attached
    [% view.footer %]           # => [% INCLUDE my_footer.tt2 %]

    # fallback on the 'notfound' template ('my_no_such_file.tt2')
    # if template not found 
    [% view.include('missing') %]
    [% view.include_missing %]
    [% view.missing %]

    # print() includes a template relevant to argument type
    [% view.print("some text") %]     # type=TEXT, template='text'

    [% BLOCK my_text.tt2 %]           # 'text' with prefix/suffix
       Text: [% item %]
    [% END %]

    # now print() a hash ref, mapped to 'hash' template
    [% view.print(some_hash_ref) %]   # type=HASH, template='hash'

    [% BLOCK my_hash.tt2 %]           # 'hash' with prefix/suffix
       hash keys: [% item.keys.sort.join(', ')
    [% END %]

    # now print() a list ref, mapped to 'list' template
    [% view.print(my_list_ref) %]     # type=ARRAY, template='list'

    [% BLOCK my_list.tt2 %]           # 'list' with prefix/suffix
       list: [% item.join(', ') %]
    [% END %]

    # print() maps 'My::Object' to 'My_Object'
    [% view.print(myobj) %]

    [% BLOCK my_My_Object.tt2 %]
       [% item.this %], [% item.that %]
    [% END %]

    # update mapping table
    [% view.map.ARRAY = 'my_list_template' %]
    [% view.map.TEXT  = 'my_text_block'    %]


    # change prefix, suffix, item name, etc.
    [% view.prefix = 'your_' %]
    [% view.default = 'anyobj' %]
    ...

=head1 DESCRIPTION

TODO

=head1 METHODS

=head2 new($context, \%config)

Creates a new Template::View presenting a custom view of the specified 
$context object.

A reference to a hash array of configuration options may be passed as the 
second argument.

=over 4

=item prefix

Prefix added to all template names.

    [% USE view(prefix => 'my_') %]
    [% view.view('foo', a => 20) %]     # => my_foo

=item suffix

Suffix added to all template names.

    [% USE view(suffix => '.tt2') %]
    [% view.view('foo', a => 20) %]     # => foo.tt2

=item map 

Hash array mapping reference types to template names.  The print() 
method uses this to determine which template to use to present any
particular item.  The TEXT, HASH and ARRAY items default to 'test', 
'hash' and 'list' appropriately.

    [% USE view(map => { ARRAY   => 'my_list', 
                         HASH    => 'your_hash',
                         My::Foo => 'my_foo', } ) %]

    [% view.print(some_text) %]         # => text
    [% view.print(a_list) %]            # => my_list
    [% view.print(a_hash) %]            # => your_hash
    [% view.print(a_foo) %]             # => my_foo

    [% BLOCK text %]
       Text: [% item %]
    [% END %]

    [% BLOCK my_list %]
       list: [% item.join(', ') %]
    [% END %]

    [% BLOCK your_hash %]
       hash keys: [% item.keys.sort.join(', ')
    [% END %]

    [% BLOCK my_foo %] 
       Foo: [% item.this %], [% item.that %]
    [% END %]

=item method

Name of a method which objects passed to print() may provide for presenting
themselves to the view.  If a specific map entry can't be found for an 
object reference and it supports the method (default: 'present') then 
the method will be called, passing the view as an argument.  The object 
can then make callbacks against the view to present itself.

    package Foo;

    sub present {
        my ($self, $view) = @_;
        return "a regular view of a Foo\n";
    }

    sub debug {
        my ($self, $view) = @_;
        return "a debug view of a Foo\n";
    }

In a template:

    [% USE view %]
    [% view.print(my_foo_object) %]     # a regular view of a Foo

    [% USE view(method => 'debug') %]
    [% view.print(my_foo_object) %]     # a debug view of a Foo

=item default

Default template to use if no specific map entry is found for an item.

    [% USE view(default => 'my_object') %]

    [% view.print(objref) %]            # => my_object

If no map entry or default is provided then the view will attempt to 
construct a template name from the object class, substituting any 
sequence of non-word characters to single underscores, e.g.

    # 'fubar' is an object of class Foo::Bar
    [% view.print(fubar) %]             # => Foo_Bar

Any current prefix and suffix will be added to both the default template 
name and any name constructed from the object class.

=item notfound

Fallback template to use if any other isn't found.

=item item

Name of the template variable to which the print() method assigns the current
item.  Defaults to 'item'.

    [% USE view %]
    [% BLOCK list %] 
       [% item.join(', ') %] 
    [% END %]
    [% view.print(a_list) %]

    [% USE view(item => 'thing') %]
    [% BLOCK list %] 
       [% thing.join(', ') %] 
    [% END %]
    [% view.print(a_list) %]

=item view_prefix

Prefix of methods which should be mapped to view() by AUTOLOAD.  Defaults
to 'view_'.

    [% USE view %]
    [% view.view_header() %]                    # => view('header')

    [% USE view(view_prefix => 'show_me_the_' %]
    [% view.show_me_the_header() %]             # => view('header')

=item view_naked

Flag to indcate if any attempt should be made to map method names to 
template names where they don't match the view_prefix.  Defaults to 0.

    [% USE view(view_naked => 1) %]

    [% view.header() %]                 # => view('header')

=back

=head2 print( $obj1, $obj2, ... \%config)

TODO

=head2 view( $template, \%vars, \%config );

TODO

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 2000-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>

=cut





