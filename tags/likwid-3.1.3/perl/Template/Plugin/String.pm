#============================================================= -*-Perl-*-
#
# Template::Plugin::String
#
# DESCRIPTION
#   Template Toolkit plugin to implement a basic String object.
#
# AUTHOR
#   Andy Wardley   <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 2001-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin::String;

use strict;
use warnings;
use base 'Template::Plugin';
use Template::Exception;

use overload q|""| => "text",
             fallback => 1;

our $VERSION = 2.40;
our $ERROR   = '';

*centre  = \*center;
*append  = \*push;
*prepend = \*unshift; 

#------------------------------------------------------------------------

sub new {
    my ($class, @args) = @_;
    my $context = ref $class ? undef : shift(@args);
    my $config = @args && ref $args[-1] eq 'HASH' ? pop(@args) : { };

    $class = ref($class) || $class;

    my $text = defined $config->{ text } 
        ? $config->{ text }
        : (@args ? shift(@args) : '');

#    print STDERR "text: [$text]\n";
#    print STDERR "class: [$class]\n";
    
    my $self = bless {
        text     => $text,
        filters  => [ ],
        _CONTEXT => $context,
    }, $class;

    my $filter = $config->{ filter } || $config->{ filters };

    # install any output filters specified as 'filter' or 'filters' option
    $self->output_filter($filter)
        if $filter;

    return $self;
}


sub text {
    my $self = shift;
    return $self->{ text } unless @{ $self->{ filters } };

    my $text = $self->{ text };
    my $context = $self->{ _CONTEXT };

    foreach my $dispatch (@{ $self->{ filters } }) {
        my ($name, $args) = @$dispatch;
        my $code = $context->filter($name, $args)
            || $self->throw($context->error());
        $text = &$code($text);
    }
    return $text;
}


sub copy {
    my $self = shift;
    $self->new($self->{ text });
}


sub throw {
    my $self = shift;

    die (Template::Exception->new('String', join('', @_)));
}


#------------------------------------------------------------------------
# output_filter($filter)
#
# Install automatic output filter(s) for the string.  $filter can a list:
# [ 'name1', 'name2' => [ ..args.. ], name4 => { ..args.. } ] or a hash
# { name1 => '', name2 => [ args ], name3 => { args } }
#------------------------------------------------------------------------

sub output_filter {
    my ($self, $filter) = @_;
    my ($name, $args, $dispatch);
    my $filters = $self->{ filters };
    my $count = 0;

    if (ref $filter eq 'HASH') {
        $filter = [ %$filter ];
    }
    elsif (ref $filter ne 'ARRAY') {
        $filter = [ split(/\s*\W+\s*/, $filter) ];
    }

    while (@$filter) {
        $name = shift @$filter;

        # args may follow as a reference (or empty string, e.g. { foo => '' }
        if (@$filter && (ref($filter->[0]) || ! length $filter->[0])) {
            $args = shift @$filter;
            if ($args) {
                $args = [ $args ] unless ref $args eq 'ARRAY';
            }
            else {
                $args = [ ];
            }
        }
        else {
            $args = [ ];
        }

#       $self->DEBUG("adding output filter $name(@$args)\n");

        push(@$filters, [ $name, $args ]);
        $count++;
    }

    return '';
}


#------------------------------------------------------------------------

sub push {
    my $self = shift;
    $self->{ text } .= join('', @_);
    return $self;
}


sub unshift {
    my $self = shift;
    $self->{ text } = join('', @_) . $self->{ text };
    return $self;
}


sub pop {
    my $self = shift;
    my $strip = shift || return $self;
    $self->{ text } =~ s/$strip$//;
    return $self;
}


sub shift {
    my $self = shift;
    my $strip = shift || return $self;
    $self->{ text } =~ s/^$strip//;
    return $self;
}

#------------------------------------------------------------------------

sub center {
    my ($self, $width) = @_;
    my $text = $self->{ text };
    my $len = length $text;
    $width ||= 0;

    if ($len < $width) {
        my $lpad = int(($width - $len) / 2);
        my $rpad = $width - $len - $lpad;
        $self->{ text } = (' ' x $lpad) . $self->{ text } . (' ' x $rpad);
    }

    return $self;
}


sub left {
    my ($self, $width) = @_;
    my $len = length $self->{ text };
    $width ||= 0;

    $self->{ text } .= (' ' x ($width - $len))
        if $width > $len;

    return $self;
}


sub right {
    my ($self, $width) = @_;
    my $len = length $self->{ text };
    $width ||= 0;

    $self->{ text } = (' ' x ($width - $len)) . $self->{ text }
        if $width > $len;

    return $self;
}


sub format {
    my ($self, $format) = @_;
    $format = '%s' unless defined $format;
    $self->{ text } = sprintf($format, $self->{ text });
    return $self;
}


sub filter {
    my ($self, $name, @args) = @_;

    my $context = $self->{ _CONTEXT };

    my $code = $context->filter($name, \@args)
        || $self->throw($context->error());
    return &$code($self->{ text });
}


#------------------------------------------------------------------------

sub upper {
    my $self = CORE::shift;
    $self->{ text } = uc $self->{ text };
    return $self;
}


sub lower {
    my $self = CORE::shift;
    $self->{ text } = lc $self->{ text };
    return $self;    
}


sub capital {
    my $self = CORE::shift;
    $self->{ text } =~ s/^(.)/\U$1/;
    return $self;    
}

#------------------------------------------------------------------------

sub chop {
    my $self = CORE::shift;
    chop $self->{ text };
    return $self;
}


sub chomp {
    my $self = CORE::shift;
    chomp $self->{ text };
    return $self;
}


sub trim {
    my $self = CORE::shift;
    for ($self->{ text }) {
        s/^\s+//; 
        s/\s+$//; 
    }
    return $self;    
}


sub collapse {
    my $self = CORE::shift;
    for ($self->{ text }) {
        s/^\s+//; 
        s/\s+$//; 
        s/\s+/ /g 
    }
    return $self;    

}

#------------------------------------------------------------------------

sub length {
    my $self = CORE::shift;
    return length $self->{ text };
}


sub truncate {
    my ($self, $length, $suffix) = @_;
    return $self unless defined $length;
    $suffix ||= '';
    return $self if CORE::length $self->{ text } <= $length;
    $self->{ text } = CORE::substr($self->{ text }, 0, 
                             $length - CORE::length($suffix)) . $suffix;
    return $self;
}


sub substr {
    my ($self, $offset, $length, $replacement) = @_;
    $offset ||= 0;

    if(defined $length) {
        if (defined $replacement) {
            my $removed = CORE::substr( $self->{text}, $offset, $length );
            CORE::substr( $self->{text}, $offset, $length ) = $replacement;
            return $removed;
        }
        else {
            return CORE::substr( $self->{text}, $offset, $length );
        }
    } 
    else {
        return CORE::substr( $self->{text}, $offset );
    }
}


sub repeat {
    my ($self, $n) = @_;
    return $self unless defined $n;
    $self->{ text } = $self->{ text } x $n;
    return $self;
}


sub replace {
    my ($self, $search, $replace) = @_;
    return $self unless defined $search;
    $replace = '' unless defined $replace;
    $self->{ text } =~ s/$search/$replace/g;
    return $self;
}


sub remove {
    my ($self, $search) = @_;
    $search = '' unless defined $search;
    $self->{ text } =~ s/$search//g;
    return $self;
}


sub split {
    my $self  = CORE::shift;
    my $split = CORE::shift;
    my $limit = CORE::shift || 0;
    $split = '\s+' unless defined $split;
    return [ split($split, $self->{ text }, $limit) ];
}


sub search {
    my ($self, $pattern) = @_;
    return $self->{ text } =~ /$pattern/;
}


sub equals {
    my ($self, $comparison) = @_;
    return $self->{ text } eq $comparison;
}


1;

__END__

=head1 NAME

Template::Plugin::String - Object oriented interface for string manipulation

=head1 SYNOPSIS

    # create String objects via USE directive
    [% USE String %]
    [% USE String 'initial text' %]
    [% USE String text => 'initial text' %]

    # or from an existing String via new()
    [% newstring = String.new %]
    [% newstring = String.new('newstring text') %]
    [% newstring = String.new( text => 'newstring text' ) %]

    # or from an existing String via copy()
    [% newstring = String.copy %]

    # append text to string
    [% String.append('text to append') %]

    # format left, right or center/centre padded
    [% String.left(20) %]
    [% String.right(20) %]
    [% String.center(20) %]   # American spelling
    [% String.centre(20) %]   # European spelling

    # and various other methods...

=head1 DESCRIPTION

This module implements a C<String> class for doing stringy things to
text in an object-oriented way. 

You can create a C<String> object via the C<USE> directive, adding any 
initial text value as an argument or as the named parameter C<text>.

    [% USE String %]
    [% USE String 'initial text' %]
    [% USE String text='initial text' %]

The object created will be referenced as C<String> by default, but you
can provide a different variable name for the object to be assigned
to:

    [% USE greeting = String 'Hello World' %]

Once you've got a C<String> object, you can use it as a prototype to 
create other C<String> objects with the C<new()> method.

    [% USE String %]
    [% greeting = String.new('Hello World') %]

The C<new()> method also accepts an initial text string as an argument
or the named parameter C<text>.

    [% greeting = String.new( text => 'Hello World' ) %]

You can also call C<copy()> to create a new C<String> as a copy of the 
original.

    [% greet2 = greeting.copy %]

The C<String> object has a C<text()> method to return the content of the 
string.

    [% greeting.text %]

However, it is sufficient to simply print the string and let the
overloaded stringification operator call the C<text()> method
automatically for you.

    [% greeting %]

Thus, you can treat C<String> objects pretty much like any regular piece
of text, interpolating it into other strings, for example:

    [% msg = "It printed '$greeting' and then dumped core\n" %]

You also have the benefit of numerous other methods for manipulating
the string.  

    [% msg.append("PS  Don't eat the yellow snow") %]

Note that all methods operate on and mutate the contents of the string
itself.  If you want to operate on a copy of the string then simply
take a copy first:

    [% msg.copy.append("PS  Don't eat the yellow snow") %]

These methods return a reference to the C<String> object itself.  This
allows you to chain multiple methods together.

    [% msg.copy.append('foo').right(72) %]

It also means that in the above examples, the C<String> is returned which
causes the C<text()> method to be called, which results in the new value of
the string being printed.  To suppress printing of the string, you can
use the C<CALL> directive.

    [% foo = String.new('foo') %]
    
    [% foo.append('bar') %]         # prints "foobar"
    
    [% CALL foo.append('bar') %]    # nothing

=head1 CONSTRUCTOR METHODS

These methods are used to create new C<String> objects.

=head2 new()

Creates a new string using an initial value passed as a positional
argument or the named parameter C<text>.

    [% USE String %]
    [% msg = String.new('Hello World') %]
    [% msg = String.new( text => 'Hello World' ) %]

=head2 copy()

Creates a new C<String> object which contains a copy of the original string.

    [% msg2 = msg.copy %]

=head1 INSPECTOR METHODS

These methods are used to examine the string.

=head2 text()

Returns the internal text value of the string.  The stringification
operator is overloaded to call this method.  Thus the following are
equivalent:

    [% msg.text %]
    [% msg %]

=head2 length()

Returns the length of the string.

    [% USE String("foo") %]
    [% String.length %]   # => 3

=head2 search($pattern)

Searches the string for the regular expression specified in C<$pattern>
returning true if found or false otherwise.

    [% item = String.new('foo bar baz wiz waz woz') %]
    [% item.search('wiz') ? 'WIZZY! :-)' : 'not wizzy :-(' %]

=head2 split($pattern, $limit)

Splits the string based on the delimiter C<$pattern> and optional C<$limit>.
Delegates to Perl's internal C<split()> so the parameters are exactly the same.

    [% FOREACH item.split %]
         ...
    [% END %]

    [% FOREACH item.split('baz|waz') %]
         ...
    [% END %]

=head1 MUTATOR METHODS

These methods modify the internal value of the string.  For example:

    [% USE str=String('foobar') %]
    [% str.append('.html') %]   # str => 'foobar.html'

The value of C<str> is now 'C<foobar.html>'.  If you don't want
to modify the string then simply take a copy first.

    [% str.copy.append('.html') %]

These methods all return a reference to the C<String> object itself.  This
has two important benefits.  The first is that when used as above, the 
C<String> object 'C<str>' returned by the C<append()> method will be stringified
with a call to its C<text()> method.  This will return the newly modified 
string content.  In other words, a directive like:

    [% str.append('.html') %]

will update the string and also print the new value.  If you just want
to update the string but not print the new value then use C<CALL>.

    [% CALL str.append('.html') %]

The other benefit of these methods returning a reference to the C<String>
is that you can chain as many different method calls together as you
like.  For example:

    [% String.append('.html').trim.format(href) %]

Here are the methods:

=head2 push($suffix, ...) / append($suffix, ...)

Appends all arguments to the end of the string.  The 
C<append()> method is provided as an alias for C<push()>.

    [% msg.push('foo', 'bar') %]
    [% msg.append('foo', 'bar') %]

=head2 pop($suffix)

Removes the suffix passed as an argument from the end of the String.

    [% USE String 'foo bar' %]
    [% String.pop(' bar')   %]   # => 'foo'

=head2 unshift($prefix, ...) / prepend($prefix, ...)

Prepends all arguments to the beginning of the string.  The
C<prepend()> method is provided as an alias for C<unshift()>.

    [% msg.unshift('foo ', 'bar ') %]
    [% msg.prepend('foo ', 'bar ') %]

=head2 shift($prefix)

Removes the prefix passed as an argument from the start of the String.

    [% USE String 'foo bar' %]
    [% String.shift('foo ') %]   # => 'bar'

=head2 left($pad)

If the length of the string is less than C<$pad> then the string is left
formatted and padded with spaces to C<$pad> length.

    [% msg.left(20) %]

=head2 right($pad)

As per L<left()> but right padding the C<String> to a length of C<$pad>.

    [% msg.right(20) %]

=head2 center($pad) / centre($pad)

As per L<left()> and L<right()> but formatting the C<String> to be centered within 
a space padded string of length C<$pad>.  The C<centre()> method is provided as 
an alias for C<center()>.

    [% msg.center(20) %]    # American spelling
    [% msg.centre(20) %]    # European spelling

=head2 format($format)

Apply a format in the style of C<sprintf()> to the string.

    [% USE String("world") %]
    [% String.format("Hello %s\n") %]  # => "Hello World\n"

=head2 upper()

Converts the string to upper case.

    [% USE String("foo") %]
    [% String.upper %]  # => 'FOO'

=head2 lower()

Converts the string to lower case

    [% USE String("FOO") %]
    [% String.lower %]  # => 'foo'

=head2 capital()

Converts the first character of the string to upper case.  

    [% USE String("foo") %]
    [% String.capital %]  # => 'Foo'

The remainder of the string is left untouched.  To force the string to
be all lower case with only the first letter capitalised, you can do 
something like this:

    [% USE String("FOO") %]
    [% String.lower.capital %]  # => 'Foo'

=head2 chop()

Removes the last character from the string.

    [% USE String("foop") %]
    [% String.chop %]   # => 'foo'

=head2 chomp()

Removes the trailing newline from the string.

    [% USE String("foo\n") %]
    [% String.chomp %]  # => 'foo'

=head2 trim()

Removes all leading and trailing whitespace from the string

    [% USE String("   foo   \n\n ") %]
    [% String.trim %]   # => 'foo'

=head2 collapse()

Removes all leading and trailing whitespace and collapses any sequences
of multiple whitespace to a single space.

    [% USE String(" \n\r  \t  foo   \n \n bar  \n") %]
    [% String.collapse %]   # => "foo bar"

=head2 truncate($length, $suffix)

Truncates the string to C<$length> characters.

    [% USE String('long string') %]
    [% String.truncate(4) %]  # => 'long'

If C<$suffix> is specified then it will be appended to the truncated
string.  In this case, the string will be further shortened by the 
length of the suffix to ensure that the newly constructed string
complete with suffix is exactly C<$length> characters long.

    [% USE msg = String('Hello World') %]
    [% msg.truncate(8, '...') %]   # => 'Hello...'

=head2 replace($search, $replace)

Replaces all occurences of C<$search> in the string with C<$replace>.

    [% USE String('foo bar foo baz') %]
    [% String.replace('foo', 'wiz')  %]  # => 'wiz bar wiz baz'

=head2 remove($search)

Remove all occurences of C<$search> in the string.

    [% USE String('foo bar foo baz') %]
    [% String.remove('foo ')  %]  # => 'bar baz'

=head2 repeat($count)

Repeats the string C<$count> times.

    [% USE String('foo ') %]
    [% String.repeat(3)  %]  # => 'foo foo foo '

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

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
