#============================================================= -*-Perl-*-
#
# Template::Stash::Context
#
# DESCRIPTION
#   This is an alternate stash object which includes a patch from 
#   Craig Barratt to implement various new virtual methods to allow
#   dotted template variable to denote if object methods and subroutines
#   should be called in scalar or list context.  It adds a little overhead
#   to each stash call and I'm a little wary of doing that.  So for now,
#   it's implemented as a separate stash module which will allow us to 
#   test it out, benchmark it and switch it in or out as we require.
#
#   This is what Craig has to say about it:
#   
#   Here's a better set of features for the core.  Attached is a new version
#   of Stash.pm (based on TT2.02) that:
#   
#     - supports the special op "scalar" that forces scalar context on
#       function calls, eg:
#   
#           cgi.param("foo").scalar
#   
#       calls cgi.param("foo") in scalar context (unlike my wimpy
#       scalar op from last night).  Array context is the default.
#   
#       With non-function operands, scalar behaves like the perl
#       version (eg: no-op for scalar, size for arrays, etc).
#   
#     - supports the special op "ref" that behaves like the perl ref.
#       If applied to a function the function is not called.  Eg:
#   
#           cgi.param("foo").ref
#   
#       does *not* call cgi.param and evaluates to "CODE".  Similarly,
#       HASH.ref, ARRAY.ref return what you expect.
#   
#     - adds a new scalar and list op called "array" that is a no-op for
#       arrays and promotes scalars to one-element arrays.
#   
#     - allows scalar ops to be applied to arrays and hashes in place,
#       eg: ARRAY.repeat(3) repeats each element in place.
#   
#     - allows list ops to be applied to scalars by promoting the scalars
#       to one-element arrays (like an implicit "array").  So you can
#       do things like SCALAR.size, SCALAR.join and get a useful result.
#   
#       This also means you can now use x.0 to safely get the first element
#       whether x is an array or scalar.
#   
#   The new Stash.pm passes the TT2.02 test suite.  But I haven't tested the
#   new features very much.  One nagging implementation problem is that the
#   "scalar" and "ref" ops have higher precedence than user variable names.
#   
# AUTHORS
#   Andy Wardley  <abw@kfs.org>
#   Craig Barratt <craig@arraycomm.com>
#
# COPYRIGHT
#   Copyright (C) 1996-2001 Andy Wardley.  All Rights Reserved.
#   Copyright (C) 1998-2001 Canon Research Centre Europe Ltd.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Stash::Context;

use strict;
use warnings;
use base 'Template::Stash';

our $VERSION = 1.63;
our $DEBUG   = 0 unless defined $DEBUG;


#========================================================================
#                    -- PACKAGE VARIABLES AND SUBS --
#========================================================================

#------------------------------------------------------------------------
# copy virtual methods from those in the regular Template::Stash
#------------------------------------------------------------------------

our $ROOT_OPS = { 
    %$Template::Stash::ROOT_OPS,
    defined $ROOT_OPS ? %$ROOT_OPS : (),
};

our $SCALAR_OPS = { 
    %$Template::Stash::SCALAR_OPS,
    'array' => sub { return [$_[0]] },
    defined $SCALAR_OPS ? %$SCALAR_OPS : (),
};

our $LIST_OPS = { 
    %$Template::Stash::LIST_OPS,
    'array' => sub { return $_[0] },
    defined $LIST_OPS ? %$LIST_OPS : (),
};
                    
our $HASH_OPS = { 
    %$Template::Stash::HASH_OPS,
    defined $HASH_OPS ? %$HASH_OPS : (),
};
 


#========================================================================
#                      -----  CLASS METHODS -----
#========================================================================

#------------------------------------------------------------------------
# new(\%params)
#
# Constructor method which creates a new Template::Stash object.
# An optional hash reference may be passed containing variable 
# definitions that will be used to initialise the stash.
#
# Returns a reference to a newly created Template::Stash.
#------------------------------------------------------------------------

sub new {
    my $class  = shift;
    my $params = ref $_[0] eq 'HASH' ? shift(@_) : { @_ };

    my $self   = {
        global  => { },
        %$params,
        %$ROOT_OPS,
        '_PARENT' => undef,
        '_CLASS'  => $class,
    };

    bless $self, $class;
}


#========================================================================
#                   -----  PUBLIC OBJECT METHODS -----
#========================================================================

#------------------------------------------------------------------------
# clone(\%params)
#
# Creates a copy of the current stash object to effect localisation 
# of variables.  The new stash is blessed into the same class as the 
# parent (which may be a derived class) and has a '_PARENT' member added
# which contains a reference to the parent stash that created it
# ($self).  This member is used in a successive declone() method call to
# return the reference to the parent.
# 
# A parameter may be provided which should reference a hash of 
# variable/values which should be defined in the new stash.  The 
# update() method is called to define these new variables in the cloned
# stash.
#
# Returns a reference to a cloned Template::Stash.
#------------------------------------------------------------------------

sub clone {
    my ($self, $params) = @_;
    $params ||= { };

    # look out for magical 'import' argument which imports another hash
    my $import = $params->{ import };
    if (defined $import && UNIVERSAL::isa($import, 'HASH')) {
        delete $params->{ import };
    }
    else {
        undef $import;
    }

    my $clone = bless { 
        %$self,                 # copy all parent members
        %$params,               # copy all new data
        '_PARENT' => $self,     # link to parent
    }, ref $self;
    
    # perform hash import if defined
    &{ $HASH_OPS->{ import }}($clone, $import)
        if defined $import;

    return $clone;
}

        
#------------------------------------------------------------------------
# declone($export) 
#
# Returns a reference to the PARENT stash.  When called in the following
# manner:
#    $stash = $stash->declone();
# the reference count on the current stash will drop to 0 and be "freed"
# and the caller will be left with a reference to the parent.  This 
# contains the state of the stash before it was cloned.  
#------------------------------------------------------------------------

sub declone {
    my $self = shift;
    $self->{ _PARENT } || $self;
}


#------------------------------------------------------------------------
# get($ident)
# 
# Returns the value for an variable stored in the stash.  The variable
# may be specified as a simple string, e.g. 'foo', or as an array 
# reference representing compound variables.  In the latter case, each
# pair of successive elements in the list represent a node in the 
# compound variable.  The first is the variable name, the second a 
# list reference of arguments or 0 if undefined.  So, the compound 
# variable [% foo.bar('foo').baz %] would be represented as the list
# [ 'foo', 0, 'bar', ['foo'], 'baz', 0 ].  Returns the value of the
# identifier or an empty string if undefined.  Errors are thrown via
# die().
#------------------------------------------------------------------------

sub get {
    my ($self, $ident, $args) = @_;
    my ($root, $result);
    $root = $self;

    if (ref $ident eq 'ARRAY'
        || ($ident =~ /\./) 
        && ($ident = [ map { s/\(.*$//; ($_, 0) } split(/\./, $ident) ])) {
        my $size = $#$ident;

        # if $ident is a list reference, then we evaluate each item in the 
        # identifier against the previous result, using the root stash 
        # ($self) as the first implicit 'result'...

        foreach (my $i = 0; $i <= $size; $i += 2) {
            if ( $i + 2 <= $size && ($ident->[$i+2] eq "scalar"
                                    || $ident->[$i+2] eq "ref") ) {
                $result = $self->_dotop($root, @$ident[$i, $i+1], 0,
                                        $ident->[$i+2]);
                $i += 2;
            } else {
                $result = $self->_dotop($root, @$ident[$i, $i+1]);
            }
            last unless defined $result;
            $root = $result;
        }
    }
    else {
        $result = $self->_dotop($root, $ident, $args);
    }

    return defined $result 
        ? $result 
        : $self->undefined($ident, $args);
}


#------------------------------------------------------------------------
# set($ident, $value, $default)
#
# Updates the value for a variable in the stash.  The first parameter
# should be the variable name or array, as per get().  The second 
# parameter should be the intended value for the variable.  The third,
# optional parameter is a flag which may be set to indicate 'default'
# mode.  When set true, the variable will only be updated if it is
# currently undefined or has a false value.  The magical 'IMPORT'
# variable identifier may be used to indicate that $value is a hash
# reference whose values should be imported.  Returns the value set,
# or an empty string if not set (e.g. default mode).  In the case of 
# IMPORT, returns the number of items imported from the hash.
#------------------------------------------------------------------------

sub set {
    my ($self, $ident, $value, $default) = @_;
    my ($root, $result, $error);

    $root = $self;

    ELEMENT: {
        if (ref $ident eq 'ARRAY'
            || ($ident =~ /\./) 
            && ($ident = [ map { s/\(.*$//; ($_, 0) }
                           split(/\./, $ident) ])) {

            # a compound identifier may contain multiple elements (e.g. 
            # foo.bar.baz) and we must first resolve all but the last, 
            # using _dotop() with the $lvalue flag set which will create 
            # intermediate hashes if necessary...
            my $size = $#$ident;
            foreach (my $i = 0; $i < $size - 2; $i += 2) {
                $result = $self->_dotop($root, @$ident[$i, $i+1], 1);
                last ELEMENT unless defined $result;
                $root = $result;
            }

            # then we call _assign() to assign the value to the last element
            $result = $self->_assign($root, @$ident[$size-1, $size], 
                                     $value, $default);
        }
        else {
            $result = $self->_assign($root, $ident, 0, $value, $default);
        }
    }

    return defined $result ? $result : '';
}


#------------------------------------------------------------------------
# getref($ident)
# 
# Returns a "reference" to a particular item.  This is represented as a 
# closure which will return the actual stash item when called.  
# WARNING: still experimental!
#------------------------------------------------------------------------

sub getref {
    my ($self, $ident, $args) = @_;
    my ($root, $item, $result);
    $root = $self;

    if (ref $ident eq 'ARRAY') {
        my $size = $#$ident;

        foreach (my $i = 0; $i <= $size; $i += 2) {
            ($item, $args) = @$ident[$i, $i + 1]; 
            last if $i >= $size - 2;  # don't evaluate last node
            last unless defined 
                ($root = $self->_dotop($root, $item, $args));
        }
    }
    else {
        $item = $ident;
    }

    if (defined $root) {
        return sub { my @args = (@{$args||[]}, @_);
                     $self->_dotop($root, $item, \@args);
                 }
    }
    else {
        return sub { '' };
    }
}




#------------------------------------------------------------------------
# update(\%params)
#
# Update multiple variables en masse.  No magic is performed.  Simple
# variable names only.
#------------------------------------------------------------------------

sub update {
    my ($self, $params) = @_;

    # look out for magical 'import' argument to import another hash
    my $import = $params->{ import };
    if (defined $import && UNIVERSAL::isa($import, 'HASH')) {
        @$self{ keys %$import } = values %$import;
        delete $params->{ import };
    }

    @$self{ keys %$params } = values %$params;
}


#========================================================================
#                  -----  PRIVATE OBJECT METHODS -----
#========================================================================

#------------------------------------------------------------------------
# _dotop($root, $item, \@args, $lvalue, $nextItem)
#
# This is the core 'dot' operation method which evaluates elements of 
# variables against their root.  All variables have an implicit root 
# which is the stash object itself (a hash).  Thus, a non-compound 
# variable 'foo' is actually '(stash.)foo', the compound 'foo.bar' is
# '(stash.)foo.bar'.  The first parameter is a reference to the current
# root, initially the stash itself.  The second parameter contains the 
# name of the variable element, e.g. 'foo'.  The third optional
# parameter is a reference to a list of any parenthesised arguments 
# specified for the variable, which are passed to sub-routines, object 
# methods, etc.  The final parameter is an optional flag to indicate 
# if this variable is being evaluated on the left side of an assignment
# (e.g. foo.bar.baz = 10).  When set true, intermediated hashes will 
# be created (e.g. bar) if necessary.  
#
# Returns the result of evaluating the item against the root, having
# performed any variable "magic".  The value returned can then be used
# as the root of the next _dotop() in a compound sequence.  Returns
# undef if the variable is undefined.
#------------------------------------------------------------------------

sub _dotop {
    my ($self, $root, $item, $args, $lvalue, $nextItem) = @_;
    my $rootref = ref $root;
    my ($value, @result, $ret, $retVal);
    $nextItem ||= "";
    my $scalarContext = 1 if ( $nextItem eq "scalar" );
    my $returnRef = 1     if ( $nextItem eq "ref" );

    $args ||= [ ];
    $lvalue ||= 0;

#    print STDERR "_dotop(root=$root, item=$item, args=[@$args])\n"
#       if $DEBUG;

    # return undef without an error if either side of the dot is unviable
    # or if an attempt is made to access a private member, starting _ or .
    return undef
        unless defined($root) and defined($item) and $item !~ /^[\._]/;

    if (ref(\$root) eq "SCALAR" && !$lvalue &&
            (($value = $LIST_OPS->{ $item }) || $item =~ /^-?\d+$/) ) {
        #
        # Promote scalar to one element list, to be processed below.
        #
        $rootref = 'ARRAY';
        $root = [$root];
    }
    if ($rootref eq $self->{_CLASS} || $rootref eq 'HASH') {

        # if $root is a regular HASH or a Template::Stash kinda HASH (the 
        # *real* root of everything).  We first lookup the named key 
        # in the hash, or create an empty hash in its place if undefined
        # and the $lvalue flag is set.  Otherwise, we check the HASH_OPS
        # pseudo-methods table, calling the code if found, or return undef.

        if (defined($value = $root->{ $item })) {
            ($ret, $retVal, @result) = _dotop_return($value, $args, $returnRef,
                                                     $scalarContext);
            return $retVal if ( $ret );                     ## RETURN
        }
        elsif ($lvalue) {
            # we create an intermediate hash if this is an lvalue
            return $root->{ $item } = { };                  ## RETURN
        }
        elsif ($value = $HASH_OPS->{ $item }) {
            @result = &$value($root, @$args);               ## @result
        }
        elsif (ref $item eq 'ARRAY') {
            # hash slice
            return [@$root{@$item}];                       ## RETURN
        }
        elsif ($value = $SCALAR_OPS->{ $item }) {
            #
            # Apply scalar ops to every hash element, in place.
            #
            foreach my $key ( keys %$root ) {
                $root->{$key} = &$value($root->{$key}, @$args);
            }
        }
    }
    elsif ($rootref eq 'ARRAY') {

        # if root is an ARRAY then we check for a LIST_OPS pseudo-method 
        # (except for l-values for which it doesn't make any sense)
        # or return the numerical index into the array, or undef

        if (($value = $LIST_OPS->{ $item }) && ! $lvalue) {
            @result = &$value($root, @$args);               ## @result
        }
        elsif (($value = $SCALAR_OPS->{ $item }) && ! $lvalue) {
            #
            # Apply scalar ops to every array element, in place.
            #
            for ( my $i = 0 ; $i < @$root ; $i++ ) {
                $root->[$i] = &$value($root->[$i], @$args); ## @result
            }
        }
        elsif ($item =~ /^-?\d+$/) {
            $value = $root->[$item];
            ($ret, $retVal, @result) = _dotop_return($value, $args, $returnRef,
                                                     $scalarContext);
            return $retVal if ( $ret );                     ## RETURN
        }
        elsif (ref $item eq 'ARRAY' ) {
            # array slice
            return [@$root[@$item]];                        ## RETURN
        }
    }

    # NOTE: we do the can-can because UNIVSERAL::isa($something, 'UNIVERSAL')
    # doesn't appear to work with CGI, returning true for the first call
    # and false for all subsequent calls. 

    elsif (ref($root) && UNIVERSAL::can($root, 'can')) {

        # if $root is a blessed reference (i.e. inherits from the 
        # UNIVERSAL object base class) then we call the item as a method.
        # If that fails then we try to fallback on HASH behaviour if 
        # possible.
        return ref $root->can($item) if ( $returnRef );       ## RETURN
        eval {
            @result = $scalarContext ? scalar $root->$item(@$args)
                                     : $root->$item(@$args);  ## @result
        };

        if ($@) {
            # failed to call object method, so try some fallbacks
            if (UNIVERSAL::isa($root, 'HASH')
                    && defined($value = $root->{ $item })) {
                ($ret, $retVal, @result) = _dotop_return($value, $args,
                                                    $returnRef, $scalarContext);
                return $retVal if ( $ret );                     ## RETURN
            }
            elsif (UNIVERSAL::isa($root, 'ARRAY') 
                   && ($value = $LIST_OPS->{ $item })) {
                @result = &$value($root, @$args);
            }
            else {
                @result = (undef, $@);
            }
        }
    }
    elsif (($value = $SCALAR_OPS->{ $item }) && ! $lvalue) {

        # at this point, it doesn't look like we've got a reference to
        # anything we know about, so we try the SCALAR_OPS pseudo-methods
        # table (but not for l-values)

        @result = &$value($root, @$args);                   ## @result
    }
    elsif ($self->{ _DEBUG }) {
        die "don't know how to access [ $root ].$item\n";   ## DIE
    }
    else {
        @result = ();
    }

    # fold multiple return items into a list unless first item is undef
    if (defined $result[0]) {
        return ref(@result > 1 ? [ @result ] : $result[0])
                                            if ( $returnRef );  ## RETURN
        if ( $scalarContext ) {
            return scalar @result if ( @result > 1 );           ## RETURN
            return scalar(@{$result[0]}) if ( ref $result[0] eq "ARRAY" );
            return scalar(%{$result[0]}) if ( ref $result[0] eq "HASH" );
            return $result[0];                                  ## RETURN
        } else {
            return @result > 1 ? [ @result ] : $result[0];      ## RETURN
        }
    }
    elsif (defined $result[1]) {
        die $result[1];                                     ## DIE
    }
    elsif ($self->{ _DEBUG }) {
        die "$item is undefined\n";                         ## DIE
    }

    return undef;
}

#------------------------------------------------------------------------
# ($ret, $retVal, @result) = _dotop_return($value, $args, $returnRef,
#                                          $scalarContext);
#
# Handle the various return processing for _dotop
#------------------------------------------------------------------------

sub _dotop_return
{
    my($value, $args, $returnRef, $scalarContext) = @_;
    my(@result);

    return (1, ref $value) if ( $returnRef );                     ## RETURN
    if ( $scalarContext ) {
        return (1, scalar(@$value)) if ref $value eq 'ARRAY';     ## RETURN
        return (1, scalar(%$value)) if ref $value eq 'HASH';      ## RETURN
        return (1, scalar($value))  unless ref $value eq 'CODE';  ## RETURN;
        @result = scalar &$value(@$args)                          ## @result;
    } else {
        return (1, $value) unless ref $value eq 'CODE';           ## RETURN
        @result = &$value(@$args);                                ## @result
    }
    return (0, undef, @result);
}


#------------------------------------------------------------------------
# _assign($root, $item, \@args, $value, $default)
#
# Similar to _dotop() above, but assigns a value to the given variable
# instead of simply returning it.  The first three parameters are the
# root item, the item and arguments, as per _dotop(), followed by the 
# value to which the variable should be set and an optional $default
# flag.  If set true, the variable will only be set if currently false
# (undefined/zero)
#------------------------------------------------------------------------

sub _assign {
    my ($self, $root, $item, $args, $value, $default) = @_;
    my $rootref = ref $root;
    my $result;
    $args ||= [ ];
    $default ||= 0;

#    print(STDERR "_assign(root=$root, item=$item, args=[@$args], \n",
#                         "value=$value, default=$default)\n")
#       if $DEBUG;

    # return undef without an error if either side of the dot is unviable
    # or if an attempt is made to update a private member, starting _ or .
    return undef                                                ## RETURN
        unless $root and defined $item and $item !~ /^[\._]/;
    
    if ($rootref eq 'HASH' || $rootref eq $self->{_CLASS}) {
        # if the root is a hash we set the named key
        return ($root->{ $item } = $value)                      ## RETURN
            unless $default && $root->{ $item };
    }
    elsif ($rootref eq 'ARRAY' && $item =~ /^-?\d+$/) {
            # or set a list item by index number
            return ($root->[$item] = $value)                    ## RETURN
                unless $default && $root->{ $item };
    }
    elsif (UNIVERSAL::isa($root, 'UNIVERSAL')) {
        # try to call the item as a method of an object
        return $root->$item(@$args, $value);                    ## RETURN
    }
    else {
        die "don't know how to assign to [$root].[$item]\n";    ## DIE
    }

    return undef;
}


#------------------------------------------------------------------------
# _dump()
#
# Debug method which returns a string representing the internal state
# of the object.  The method calls itself recursively to dump sub-hashes.
#------------------------------------------------------------------------

sub _dump {
    my $self   = shift;
    my $indent = shift || 1;
    my $buffer = '    ';
    my $pad    = $buffer x $indent;
    my $text   = '';
    local $" = ', ';

    my ($key, $value);


    return $text . "...excessive recursion, terminating\n"
        if $indent > 32;

    foreach $key (keys %$self) {

        $value = $self->{ $key };
        $value = '<undef>' unless defined $value;

        if (ref($value) eq 'ARRAY') {
            $value = "$value [@$value]";
        }
        $text .= sprintf("$pad%-8s => $value\n", $key);
        next if $key =~ /^\./;
        if (UNIVERSAL::isa($value, 'HASH')) {
            $text .= _dump($value, $indent + 1);
        }
    }
    $text;
}


1;

__END__

=head1 NAME

Template::Stash::Context - Experimetal stash allowing list/scalar context definition

=head1 SYNOPSIS

    use Template;
    use Template::Stash::Context;

    my $stash = Template::Stash::Context->new(\%vars);
    my $tt2   = Template->new({ STASH => $stash });

=head1 DESCRIPTION

This is an alternate stash object which includes a patch from 
Craig Barratt to implement various new virtual methods to allow
dotted template variable to denote if object methods and subroutines
should be called in scalar or list context.  It adds a little overhead
to each stash call and I'm a little wary of applying that to the core
default stash without investigating the effects first. So for now,
it's implemented as a separate stash module which will allow us to 
test it out, benchmark it and switch it in or out as we require.

This is what Craig has to say about it:

Here's a better set of features for the core.  Attached is a new version
of Stash.pm (based on TT2.02) that:

* supports the special op "scalar" that forces scalar context on
function calls, eg:

    cgi.param("foo").scalar

calls cgi.param("foo") in scalar context (unlike my wimpy
scalar op from last night).  Array context is the default.

With non-function operands, scalar behaves like the perl
version (eg: no-op for scalar, size for arrays, etc).

* supports the special op "ref" that behaves like the perl ref.
If applied to a function the function is not called.  Eg:

    cgi.param("foo").ref

does *not* call cgi.param and evaluates to "CODE".  Similarly,
HASH.ref, ARRAY.ref return what you expect.

* adds a new scalar and list op called "array" that is a no-op for
arrays and promotes scalars to one-element arrays.

* allows scalar ops to be applied to arrays and hashes in place,
eg: ARRAY.repeat(3) repeats each element in place.

* allows list ops to be applied to scalars by promoting the scalars
to one-element arrays (like an implicit "array").  So you can
do things like SCALAR.size, SCALAR.join and get a useful result.

This also means you can now use x.0 to safely get the first element
whether x is an array or scalar.

The new Stash.pm passes the TT2.02 test suite.  But I haven't tested the
new features very much.  One nagging implementation problem is that the
"scalar" and "ref" ops have higher precedence than user variable names.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt>

L<http://wardley.org/|http://wardley.org/>




=head1 VERSION

1.63, distributed as part of the
Template Toolkit version 2.19, released on 27 April 2007.

=head1 COPYRIGHT

  Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.


This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Stash|Template::Stash>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
