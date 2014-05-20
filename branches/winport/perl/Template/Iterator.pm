#============================================================= -*-Perl-*-
#
# Template::Iterator
#
# DESCRIPTION
#
#   Module defining an iterator class which is used by the FOREACH
#   directive for iterating through data sets.  This may be
#   sub-classed to define more specific iterator types.
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

package Template::Iterator;

use strict;
use warnings;
use base 'Template::Base';
use Template::Constants;
use Template::Exception;
use Scalar::Util qw(blessed);

use constant ODD  => 'odd';
use constant EVEN => 'even';

our $VERSION = 2.68;
our $DEBUG   = 0 unless defined $DEBUG;
our $AUTOLOAD;

#========================================================================
#                      -----  CLASS METHODS -----
#========================================================================

#------------------------------------------------------------------------
# new(\@target, \%options)
#
# Constructor method which creates and returns a reference to a new 
# Template::Iterator object.  A reference to the target data (array
# or hash) may be passed for the object to iterate through.
#------------------------------------------------------------------------

sub new {
    my $class  = shift;
    my $data   = shift || [ ];
    my $params = shift || { };

    if (ref $data eq 'HASH') {
        # map a hash into a list of { key => ???, value => ??? } hashes,
        # one for each key, sorted by keys
        $data = [ map { { key => $_, value => $data->{ $_ } } }
                  sort keys %$data ];
    }
    elsif (blessed($data) && $data->can('as_list')) {
        $data = $data->as_list();
    }
    elsif (ref $data ne 'ARRAY') {
        # coerce any non-list data into an array reference
        $data  = [ $data ] ;
    }

    bless {
        _DATA  => $data,
        _ERROR => '',
    }, $class;
}


#========================================================================
#                   -----  PUBLIC OBJECT METHODS -----
#========================================================================

#------------------------------------------------------------------------
# get_first()
#
# Initialises the object for iterating through the target data set.  The 
# first record is returned, if defined, along with the STATUS_OK value.
# If there is no target data, or the data is an empty set, then undef 
# is returned with the STATUS_DONE value.  
#------------------------------------------------------------------------

sub get_first {
    my $self  = shift;
    my $data  = $self->{ _DATA };

    $self->{ _DATASET } = $self->{ _DATA };
    my $size = scalar @$data;
    my $index = 0;
    
    return (undef, Template::Constants::STATUS_DONE) unless $size;

    # initialise various counters, flags, etc.
    @$self{ qw( SIZE MAX INDEX COUNT FIRST LAST ) } 
            = ( $size, $size - 1, $index, 1, 1, $size > 1 ? 0 : 1, undef );
    @$self{ qw( PREV NEXT ) } = ( undef, $self->{ _DATASET }->[ $index + 1 ]);

    return $self->{ _DATASET }->[ $index ];
}



#------------------------------------------------------------------------
# get_next()
#
# Called repeatedly to access successive elements in the data set.
# Should only be called after calling get_first() or a warning will 
# be raised and (undef, STATUS_DONE) returned.
#------------------------------------------------------------------------

sub get_next {
    my $self = shift;
    my ($max, $index) = @$self{ qw( MAX INDEX ) };
    my $data = $self->{ _DATASET };

    # warn about incorrect usage
    unless (defined $index) {
        my ($pack, $file, $line) = caller();
        warn("iterator get_next() called before get_first() at $file line $line\n");
        return (undef, Template::Constants::STATUS_DONE);   ## RETURN ##
    }

    # if there's still some data to go...
    if ($index < $max) {
        # update counters and flags
        $index++;
        @$self{ qw( INDEX COUNT FIRST LAST ) }
        = ( $index, $index + 1, 0, $index == $max ? 1 : 0 );
        @$self{ qw( PREV NEXT ) } = @$data[ $index - 1, $index + 1 ];
        return $data->[ $index ];                           ## RETURN ##
    }
    else {
        return (undef, Template::Constants::STATUS_DONE);   ## RETURN ##
    }
}


#------------------------------------------------------------------------
# get_all()
#
# Method which returns all remaining items in the iterator as a Perl list
# reference.  May be called at any time in the life-cycle of the iterator.
# The get_first() method will be called automatically if necessary, and
# then subsequent get_next() calls are made, storing each returned 
# result until the list is exhausted.  
#------------------------------------------------------------------------

sub get_all {
    my $self = shift;
    my ($max, $index) = @$self{ qw( MAX INDEX ) };
    my @data;

    # handle cases where get_first() has yet to be called.
    unless (defined $index) {
        my ($first, $status) = $self->get_first;

        # refresh $max and $index, after get_first updates MAX and INDEX
        ($max, $index) = @$self{ qw( MAX INDEX ) };

        # empty lists are handled here.
        if ($status && $status == Template::Constants::STATUS_DONE) {
            return (undef, Template::Constants::STATUS_DONE);   ## RETURN ##
        }

        push @data, $first;

        ## if there's nothing left in the iterator, return the single value.
        unless ($index < $max) {
            return \@data;
        }
    }

    # if there's still some data to go...
    if ($index < $max) {
        $index++;
        push @data, @{ $self->{ _DATASET } } [ $index..$max ];
        
        # update counters and flags
        @$self{ qw( INDEX COUNT FIRST LAST ) }
        = ( $max, $max + 1, 0, 1 );

        return \@data;                                      ## RETURN ##
    }
    else {
        return (undef, Template::Constants::STATUS_DONE);   ## RETURN ##
    }
}

sub odd {
    shift->{ COUNT } % 2 ? 1 : 0
}

sub even {
    shift->{ COUNT } % 2 ? 0 : 1
}

sub parity {
    shift->{ COUNT } % 2 ? ODD : EVEN;
}


#------------------------------------------------------------------------
# AUTOLOAD
#
# Provides access to internal fields (e.g. size, first, last, max, etc)
#------------------------------------------------------------------------

sub AUTOLOAD {
    my $self = shift;
    my $item = $AUTOLOAD;
    $item =~ s/.*:://;
    return if $item eq 'DESTROY';

    # alias NUMBER to COUNT for backwards compatability
    $item = 'COUNT' if $item =~ /NUMBER/i;

    return $self->{ uc $item };
}


#========================================================================
#                   -----  PRIVATE DEBUG METHODS -----
#========================================================================

#------------------------------------------------------------------------
# _dump()
#
# Debug method which returns a string detailing the internal state of 
# the iterator object.
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    join('',
         "  Data: ", $self->{ _DATA  }, "\n",
         " Index: ", $self->{ INDEX  }, "\n",
         "Number: ", $self->{ NUMBER }, "\n",
         "   Max: ", $self->{ MAX    }, "\n",
         "  Size: ", $self->{ SIZE   }, "\n",
         " First: ", $self->{ FIRST  }, "\n",
         "  Last: ", $self->{ LAST   }, "\n",
         "\n"
     );
}


1;

__END__

=head1 NAME

Template::Iterator - Data iterator used by the FOREACH directive

=head1 SYNOPSIS

    my $iter = Template::Iterator->new(\@data, \%options);

=head1 DESCRIPTION

The C<Template::Iterator> module defines a generic data iterator for use 
by the C<FOREACH> directive.  

It may be used as the base class for custom iterators.

=head1 PUBLIC METHODS

=head2 new($data) 

Constructor method.  A reference to a list of values is passed as the
first parameter.  Subsequent calls to L<get_first()> and L<get_next()> calls 
will return each element from the list.

    my $iter = Template::Iterator->new([ 'foo', 'bar', 'baz' ]);

The constructor will also accept a reference to a hash array and will 
expand it into a list in which each entry is a hash array containing
a 'C<key>' and 'C<value>' item, sorted according to the hash keys.

    my $iter = Template::Iterator->new({ 
        foo => 'Foo Item',
        bar => 'Bar Item',
    });

This is equivalent to:

    my $iter = Template::Iterator->new([
        { key => 'bar', value => 'Bar Item' },
        { key => 'foo', value => 'Foo Item' },
    ]);

When passed a single item which is not an array reference, the constructor
will automatically create a list containing that single item.

    my $iter = Template::Iterator->new('foo');

This is equivalent to:

    my $iter = Template::Iterator->new([ 'foo' ]);

Note that a single item which is an object based on a blessed ARRAY 
references will NOT be treated as an array and will be folded into 
a list containing that one object reference.

    my $list = bless [ 'foo', 'bar' ], 'MyListClass';
    my $iter = Template::Iterator->new($list);

equivalent to:

    my $iter = Template::Iterator->new([ $list ]);

If the object provides an C<as_list()> method then the L<Template::Iterator>
constructor will call that method to return the list of data.  For example:

    package MyListObject;
    
    sub new {
        my $class = shift;
        bless [ @_ ], $class;
    }

    package main;
    
    my $list = MyListObject->new('foo', 'bar');
    my $iter = Template::Iterator->new($list);

This is then functionally equivalent to:

    my $iter = Template::Iterator->new([ $list ]);

The iterator will return only one item, a reference to the C<MyListObject>
object, C<$list>.

By adding an C<as_list()> method to the C<MyListObject> class, we can force
the C<Template::Iterator> constructor to treat the object as a list and 
use the data contained within.

    package MyListObject;
    
    ...
    
    sub as_list {
        my $self = shift;
        return $self;
    }
    
    package main;
    
    my $list = MyListObject->new('foo', 'bar');
    my $iter = Template::Iterator->new($list);

The iterator will now return the two items, 'C<foo>' and 'C<bar>', which the 
C<MyObjectList> encapsulates.

=head2 get_first()

Returns a C<($value, $error)> pair for the first item in the iterator set.
The C<$error> returned may be zero or undefined to indicate a valid datum
was successfully returned.  Returns an error of C<STATUS_DONE> if the list 
is empty.

=head2 get_next()

Returns a C<($value, $error)> pair for the next item in the iterator set.
Returns an error of C<STATUS_DONE> if all items in the list have been 
visited.

=head2 get_all()

Returns a C<(\@values, $error)> pair for all remaining items in the iterator 
set.  Returns an error of C<STATUS_DONE> if all items in the list have been 
visited.

=head2 size()

Returns the size of the data set or undef if unknown.

=head2 max()

Returns the maximum index number (i.e. the index of the last element) 
which is equivalent to L<size()> - C<1>.

=head2 index()

Returns the current index number which is in the range C<0> to L<max()>.

=head2 count()

Returns the current iteration count in the range C<1> to L<size()>.  This is
equivalent to L<index()> + C<1>.  

=head2 first()

Returns a boolean value to indicate if the iterator is currently on 
the first iteration of the set.

=head2 last()

Returns a boolean value to indicate if the iterator is currently on
the last iteration of the set.

=head2 prev()

Returns the previous item in the data set, or C<undef> if the iterator is
on the first item.

=head2 next()

Returns the next item in the data set or C<undef> if the iterator is on the 
last item.

=head2 parity()

Returns the text string C<even> or C<odd> to indicate the parity of the 
current iteration count (starting at 1).  This is typically used to create
striped I<zebra tables>.

    <table>
    [% FOREACH name IN ['Arthur', 'Ford', 'Trillian'] -%]
      <tr class="[% loop.parity %]">
        <td>[% name %]</td>
      </tr>
    [% END %]
    </table>

This will produce the following output:

    <table>
      <tr class="odd">
        <td>Arthur</td>
      </tr>
      <tr class="even">
        <td>Ford</td>
      </tr>
      <tr class="odd">
        <td>Trillian</td>
      </tr>
    </table>

You can then style the C<tr.odd> and C<tr.even> elements using CSS:

    tr.odd td {
        background-color: black;
        color: white;
    }
    
    tr.even td {
        background-color: white;
        color: black;
    }

=head2 odd()

Returns a boolean (0/1) value to indicate if the current iterator count
(starting at 1) is an odd number. In other words, this will return a true
value for the first iterator, the third, fifth, and so on.

=head2 even()

Returns a boolean (0/1) value to indicate if the current iterator count
(starting at 1) is an even number. In other words, this will return a true
value for the second iteration, the fourth, sixth, and so on.

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
