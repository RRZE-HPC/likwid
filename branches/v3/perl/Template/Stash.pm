#============================================================= -*-Perl-*-
#
# Template::Stash
#
# DESCRIPTION
#   Definition of an object class which stores and manages access to 
#   variables for the Template Toolkit. 
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

package Template::Stash;

use strict;
use warnings;
use Template::VMethods;
use Template::Exception;
use Scalar::Util qw( blessed reftype );

our $VERSION    = 2.91;
our $DEBUG      = 0 unless defined $DEBUG;
our $PRIVATE    = qr/^[_.]/;
our $UNDEF_TYPE = 'var.undef';
our $UNDEF_INFO = 'undefined variable: %s';

# alias _dotop() to dotop() so that we have a consistent method name
# between the Perl and XS stash implementations
*dotop = \&_dotop;


#------------------------------------------------------------------------
# Virtual Methods
#
# If any of $ROOT_OPS, $SCALAR_OPS, $HASH_OPS or $LIST_OPS are already
# defined then we merge their contents with the default virtual methods
# define by Template::VMethods.  Otherwise we can directly alias the 
# corresponding Template::VMethod package vars.
#------------------------------------------------------------------------

our $ROOT_OPS = defined $ROOT_OPS 
    ? { %{$Template::VMethods::ROOT_VMETHODS}, %$ROOT_OPS }
    : $Template::VMethods::ROOT_VMETHODS;

our $SCALAR_OPS = defined $SCALAR_OPS 
    ? { %{$Template::VMethods::TEXT_VMETHODS}, %$SCALAR_OPS }
    : $Template::VMethods::TEXT_VMETHODS;

our $HASH_OPS = defined $HASH_OPS 
    ? { %{$Template::VMethods::HASH_VMETHODS}, %$HASH_OPS }
    : $Template::VMethods::HASH_VMETHODS;

our $LIST_OPS = defined $LIST_OPS 
    ? { %{$Template::VMethods::LIST_VMETHODS}, %$LIST_OPS }
    : $Template::VMethods::LIST_VMETHODS;


#------------------------------------------------------------------------
# define_vmethod($type, $name, \&sub)
#
# Defines a virtual method of type $type (SCALAR, HASH, or LIST), with
# name $name, that invokes &sub when called.  It is expected that &sub
# be able to handle the type that it will be called upon.
#------------------------------------------------------------------------

sub define_vmethod {
    my ($class, $type, $name, $sub) = @_;
    my $op;
    $type = lc $type;

    if ($type =~ /^scalar|item$/) {
        $op = $SCALAR_OPS;
    }
    elsif ($type eq 'hash') {
        $op = $HASH_OPS;
    }
    elsif ($type =~ /^list|array$/) {
        $op = $LIST_OPS;
    }
    else {
        die "invalid vmethod type: $type\n";
    }

    $op->{ $name } = $sub;

    return 1;
}


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
    if (defined $import && ref $import eq 'HASH') {
        delete $params->{ import };
    }
    else {
        undef $import;
    }

    my $clone = bless { 
        %$self,         # copy all parent members
        %$params,       # copy all new data
        '_PARENT' => $self,     # link to parent
    }, ref $self;
    
    # perform hash import if defined
    &{ $HASH_OPS->{ import } }($clone, $import)
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
            $result = $self->_dotop($root, @$ident[$i, $i+1]);
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
    if (defined $import && ref $import eq 'HASH') {
        @$self{ keys %$import } = values %$import;
        delete $params->{ import };
    }

    @$self{ keys %$params } = values %$params;
}


#------------------------------------------------------------------------
# undefined($ident, $args)
#
# Method called when a get() returns an undefined value.  Can be redefined
# in a subclass to implement alternate handling.
#------------------------------------------------------------------------

sub undefined {
    my ($self, $ident, $args) = @_;

    if ($self->{ _STRICT }) {
        # Sorry, but we can't provide a sensible source file and line without
        # re-designing the whole architecure of TT (see TT3)
        die Template::Exception->new(
            $UNDEF_TYPE, 
            sprintf(
                $UNDEF_INFO, 
                $self->_reconstruct_ident($ident)
            )
        ) if $self->{ _STRICT };
    }
    else {
        # There was a time when I thought this was a good idea. But it's not.
        return '';
    }
}

sub _reconstruct_ident {
    my ($self, $ident) = @_;
    my ($name, $args, @output);
    my @input = ref $ident eq 'ARRAY' ? @$ident : ($ident);

    while (@input) {
        $name = shift @input;
        $args = shift @input || 0;
        $name .= '(' . join(', ', map { /^\d+$/ ? $_ : "'$_'" } @$args) . ')'
            if $args && ref $args eq 'ARRAY';
        push(@output, $name);
    }
    
    return join('.', @output);
}


#========================================================================
#                  -----  PRIVATE OBJECT METHODS -----
#========================================================================

#------------------------------------------------------------------------
# _dotop($root, $item, \@args, $lvalue)
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
    my ($self, $root, $item, $args, $lvalue) = @_;
    my $rootref = ref $root;
    my $atroot  = (blessed $root && $root->isa(ref $self));
    my ($value, @result);

    $args ||= [ ];
    $lvalue ||= 0;

#    print STDERR "_dotop(root=$root, item=$item, args=[@$args])\n"
#   if $DEBUG;

    # return undef without an error if either side of the dot is unviable
    return undef unless defined($root) and defined($item);

    # or if an attempt is made to access a private member, starting _ or .
    return undef if $PRIVATE && $item =~ /$PRIVATE/;

    if ($atroot || $rootref eq 'HASH') {
        # if $root is a regular HASH or a Template::Stash kinda HASH (the 
        # *real* root of everything).  We first lookup the named key 
        # in the hash, or create an empty hash in its place if undefined
        # and the $lvalue flag is set.  Otherwise, we check the HASH_OPS
        # pseudo-methods table, calling the code if found, or return undef.
        
        if (defined($value = $root->{ $item })) {
            return $value unless ref $value eq 'CODE';      ## RETURN
            @result = &$value(@$args);                      ## @result
        }
        elsif ($lvalue) {
            # we create an intermediate hash if this is an lvalue
            return $root->{ $item } = { };                  ## RETURN
        }
        # ugly hack: only allow import vmeth to be called on root stash
        elsif (($value = $HASH_OPS->{ $item })
               && ! $atroot || $item eq 'import') {
            @result = &$value($root, @$args);               ## @result
        }
        elsif ( ref $item eq 'ARRAY' ) {
            # hash slice
            return [@$root{@$item}];                        ## RETURN
        }
    }
    elsif ($rootref eq 'ARRAY') {    
        # if root is an ARRAY then we check for a LIST_OPS pseudo-method 
        # or return the numerical index into the array, or undef
        if ($value = $LIST_OPS->{ $item }) {
            @result = &$value($root, @$args);               ## @result
        }
        elsif ($item =~ /^-?\d+$/) {
            $value = $root->[$item];
            return $value unless ref $value eq 'CODE';      ## RETURN
            @result = &$value(@$args);                      ## @result
        }
        elsif ( ref $item eq 'ARRAY' ) {
            # array slice
            return [@$root[@$item]];                        ## RETURN
        }
    }
    
    # NOTE: we do the can-can because UNIVSERAL::isa($something, 'UNIVERSAL')
    # doesn't appear to work with CGI, returning true for the first call
    # and false for all subsequent calls. 
    
    # UPDATE: that doesn't appear to be the case any more
    
    elsif (blessed($root) && $root->can('can')) {

        # if $root is a blessed reference (i.e. inherits from the 
        # UNIVERSAL object base class) then we call the item as a method.
        # If that fails then we try to fallback on HASH behaviour if 
        # possible.
        eval { @result = $root->$item(@$args); };       
        
        if ($@) {
            # temporary hack - required to propogate errors thrown
            # by views; if $@ is a ref (e.g. Template::Exception
            # object then we assume it's a real error that needs
            # real throwing

            my $class = ref($root) || $root;
            die $@ if ref($@) || ($@ !~ /Can't locate object method "\Q$item\E" via package "\Q$class\E"/);

            # failed to call object method, so try some fallbacks
            if (reftype $root eq 'HASH') {
                if( defined($value = $root->{ $item })) {
                    return $value unless ref $value eq 'CODE';      ## RETURN
                    @result = &$value(@$args);
                }
                elsif ($value = $HASH_OPS->{ $item }) {
                    @result = &$value($root, @$args);
                }
                elsif ($value = $LIST_OPS->{ $item }) {
                    @result = &$value([$root], @$args);
                }
            }
            elsif (reftype $root eq 'ARRAY') {
                if( $value = $LIST_OPS->{ $item }) {
                   @result = &$value($root, @$args);
                }
                elsif( $item =~ /^-?\d+$/ ) {
                   $value = $root->[$item];
                   return $value unless ref $value eq 'CODE';      ## RETURN
                   @result = &$value(@$args);                      ## @result
                }
                elsif ( ref $item eq 'ARRAY' ) {
                    # array slice
                    return [@$root[@$item]];                        ## RETURN
                }
            }
            elsif ($value = $SCALAR_OPS->{ $item }) {
                @result = &$value($root, @$args);
            }
            elsif ($value = $LIST_OPS->{ $item }) {
                @result = &$value([$root], @$args);
            }
            elsif ($self->{ _DEBUG }) {
                @result = (undef, $@);
            }
        }
    }
    elsif (($value = $SCALAR_OPS->{ $item }) && ! $lvalue) {
        # at this point, it doesn't look like we've got a reference to
        # anything we know about, so we try the SCALAR_OPS pseudo-methods
        # table (but not for l-values)
        @result = &$value($root, @$args);           ## @result
    }
    elsif (($value = $LIST_OPS->{ $item }) && ! $lvalue) {
        # last-ditch: can we promote a scalar to a one-element
        # list and apply a LIST_OPS virtual method?
        @result = &$value([$root], @$args);
    }
    elsif ($self->{ _DEBUG }) {
        die "don't know how to access [ $root ].$item\n";   ## DIE
    }
    else {
        @result = ();
    }

    # fold multiple return items into a list unless first item is undef
    if (defined $result[0]) {
        return                              ## RETURN
        scalar @result > 1 ? [ @result ] : $result[0];
    }
    elsif (defined $result[1]) {
        die $result[1];                     ## DIE
    }
    elsif ($self->{ _DEBUG }) {
        die "$item is undefined\n";         ## DIE
    }

    return undef;
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
    my $atroot  = ($root eq $self);
    my $result;
    $args ||= [ ];
    $default ||= 0;

    # return undef without an error if either side of the dot is unviable
    return undef unless $root and defined $item;

    # or if an attempt is made to update a private member, starting _ or .
    return undef if $PRIVATE && $item =~ /$PRIVATE/;
    
    if ($rootref eq 'HASH' || $atroot) {
        # if the root is a hash we set the named key
        return ($root->{ $item } = $value)          ## RETURN
            unless $default && $root->{ $item };
    }
    elsif ($rootref eq 'ARRAY' && $item =~ /^-?\d+$/) {
        # or set a list item by index number
        return ($root->[$item] = $value)            ## RETURN
            unless $default && $root->{ $item };
    }
    elsif (blessed($root)) {
        # try to call the item as a method of an object
        
        return $root->$item(@$args, $value)         ## RETURN
            unless $default && $root->$item();
        
# 2 issues:
#   - method call should be wrapped in eval { }
#   - fallback on hash methods if object method not found
#
#     eval { $result = $root->$item(@$args, $value); };     
# 
#     if ($@) {
#         die $@ if ref($@) || ($@ !~ /Can't locate object method/);
# 
#         # failed to call object method, so try some fallbacks
#         if (UNIVERSAL::isa($root, 'HASH') && exists $root->{ $item }) {
#         $result = ($root->{ $item } = $value)
#             unless $default && $root->{ $item };
#         }
#     }
#     return $result;                       ## RETURN
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
    return "[Template::Stash] " . $self->_dump_frame(2);
}

sub _dump_frame {
    my ($self, $indent) = @_;
    $indent ||= 1;
    my $buffer = '    ';
    my $pad    = $buffer x $indent;
    my $text   = "{\n";
    local $" = ', ';

    my ($key, $value);

    return $text . "...excessive recursion, terminating\n"
        if $indent > 32;
    
    foreach $key (keys %$self) {
        $value = $self->{ $key };
        $value = '<undef>' unless defined $value;
        next if $key =~ /^\./;
        if (ref($value) eq 'ARRAY') {
            $value = '[ ' . join(', ', map { defined $_ ? $_ : '<undef>' }
                                 @$value) . ' ]';
        }
        elsif (ref $value eq 'HASH') {
            $value = _dump_frame($value, $indent + 1);
        }
        
        $text .= sprintf("$pad%-16s => $value\n", $key);
    }
    $text .= $buffer x ($indent - 1) . '}';
    return $text;
}


1;

__END__

=head1 NAME

Template::Stash - Magical storage for template variables

=head1 SYNOPSIS

    use Template::Stash;
    
    my $stash = Template::Stash->new(\%vars);
    
    # get variable values
    $value = $stash->get($variable);
    $value = $stash->get(\@compound);
    
    # set variable value
    $stash->set($variable, $value);
    $stash->set(\@compound, $value);
    
    # default variable value
    $stash->set($variable, $value, 1);
    $stash->set(\@compound, $value, 1);
    
    # set variable values en masse
    $stash->update(\%new_vars)
    
    # methods for (de-)localising variables
    $stash = $stash->clone(\%new_vars);
    $stash = $stash->declone();

=head1 DESCRIPTION

The C<Template::Stash> module defines an object class which is used to store
variable values for the runtime use of the template processor.  Variable
values are stored internally in a hash reference (which itself is blessed 
to create the object) and are accessible via the L<get()> and L<set()> methods.

Variables may reference hash arrays, lists, subroutines and objects
as well as simple values.  The stash automatically performs the right
magic when dealing with variables, calling code or object methods,
indexing into lists, hashes, etc.

The stash has L<clone()> and L<declone()> methods which are used by the
template processor to make temporary copies of the stash for
localising changes made to variables.

=head1 PUBLIC METHODS

=head2 new(\%params)

The C<new()> constructor method creates and returns a reference to a new
C<Template::Stash> object.  

    my $stash = Template::Stash->new();

A hash reference may be passed to provide variables and values which
should be used to initialise the stash.

    my $stash = Template::Stash->new({ var1 => 'value1', 
                                       var2 => 'value2' });

=head2 get($variable)

The C<get()> method retrieves the variable named by the first parameter.

    $value = $stash->get('var1');

Dotted compound variables can be retrieved by specifying the variable
elements by reference to a list.  Each node in the variable occupies
two entries in the list.  The first gives the name of the variable
element, the second is a reference to a list of arguments for that 
element, or C<0> if none.

    [% foo.bar(10).baz(20) %]
    
    $stash->get([ 'foo', 0, 'bar', [ 10 ], 'baz', [ 20 ] ]);

=head2 set($variable, $value, $default)

The C<set()> method sets the variable name in the first parameter to the 
value specified in the second.

    $stash->set('var1', 'value1');

If the third parameter evaluates to a true value, the variable is
set only if it did not have a true value before.

    $stash->set('var2', 'default_value', 1);

Dotted compound variables may be specified as per L<get()> above.

    [% foo.bar = 30 %]
    
    $stash->set([ 'foo', 0, 'bar', 0 ], 30);

The magical variable 'C<IMPORT>' can be specified whose corresponding
value should be a hash reference.  The contents of the hash array are
copied (i.e. imported) into the current namespace.

    # foo.bar = baz, foo.wiz = waz
    $stash->set('foo', { 'bar' => 'baz', 'wiz' => 'waz' });
    
    # import 'foo' into main namespace: bar = baz, wiz = waz
    $stash->set('IMPORT', $stash->get('foo'));

=head2 clone(\%params)

The C<clone()> method creates and returns a new C<Template::Stash> object
which represents a localised copy of the parent stash. Variables can be freely
updated in the cloned stash and when L<declone()> is called, the original stash
is returned with all its members intact and in the same state as they were
before C<clone()> was called.

For convenience, a hash of parameters may be passed into C<clone()> which 
is used to update any simple variable (i.e. those that don't contain any 
namespace elements like C<foo> and C<bar> but not C<foo.bar>) variables while 
cloning the stash.  For adding and updating complex variables, the L<set()> 
method should be used after calling C<clone().>  This will correctly resolve
and/or create any necessary namespace hashes.

A cloned stash maintains a reference to the stash that it was copied 
from in its C<_PARENT> member.

=head2 declone()

The C<declone()> method returns the C<_PARENT> reference and can be used to
restore the state of a stash as described above.

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
