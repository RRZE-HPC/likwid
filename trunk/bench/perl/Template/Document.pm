##============================================================= -*-Perl-*-
#
# Template::Document
#
# DESCRIPTION
#   Module defining a class of objects which encapsulate compiled
#   templates, storing additional block definitions and metadata 
#   as well as the compiled Perl sub-routine representing the main
#   template content.
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

package Template::Document;

use strict;
use warnings;
use base 'Template::Base';
use Template::Constants;

our $VERSION = 2.79;
our $DEBUG   = 0 unless defined $DEBUG;
our $ERROR   = '';
our ($COMPERR, $AUTOLOAD, $UNICODE);

BEGIN {
    # UNICODE is supported in versions of Perl from 5.008 onwards
    if ($UNICODE = $] > 5.007 ? 1 : 0) {
        if ($] > 5.008) {
            # utf8::is_utf8() available from Perl 5.8.1 onwards
            *is_utf8 = \&utf8::is_utf8;
        }
        elsif ($] == 5.008) {
            # use Encode::is_utf8() for Perl 5.8.0
            require Encode;
            *is_utf8 = \&Encode::is_utf8;
        }
    }
}


#========================================================================
#                     -----  PUBLIC METHODS -----
#========================================================================

#------------------------------------------------------------------------
# new(\%document)
#
# Creates a new self-contained Template::Document object which 
# encapsulates a compiled Perl sub-routine, $block, any additional 
# BLOCKs defined within the document ($defblocks, also Perl sub-routines)
# and additional $metadata about the document.
#------------------------------------------------------------------------

sub new {
    my ($class, $doc) = @_;
    my ($block, $defblocks, $metadata) = @$doc{ qw( BLOCK DEFBLOCKS METADATA ) };
    $defblocks ||= { };
    $metadata  ||= { };

    # evaluate Perl code in $block to create sub-routine reference if necessary
    unless (ref $block) {
        local $SIG{__WARN__} = \&catch_warnings;
        $COMPERR = '';

        # DON'T LOOK NOW! - blindly untainting can make you go blind!
        $block =~ /(.*)/s;
        $block = $1;
        
        $block = eval $block;
        return $class->error($@)
            unless defined $block;
    }

    # same for any additional BLOCK definitions
    @$defblocks{ keys %$defblocks } = 
        # MORE BLIND UNTAINTING - turn away if you're squeamish
        map { 
            ref($_) 
                ? $_ 
                : ( /(.*)/s && eval($1) or return $class->error($@) )
            } values %$defblocks;
    
    bless {
        %$metadata,
        _BLOCK     => $block,
        _DEFBLOCKS => $defblocks,
        _HOT       => 0,
    }, $class;
}


#------------------------------------------------------------------------
# block()
#
# Returns a reference to the internal sub-routine reference, _BLOCK, 
# that constitutes the main document template.
#------------------------------------------------------------------------

sub block {
    return $_[0]->{ _BLOCK };
}


#------------------------------------------------------------------------
# blocks()
#
# Returns a reference to a hash array containing any BLOCK definitions 
# from the template.  The hash keys are the BLOCK nameand the values
# are references to Template::Document objects.  Returns 0 (# an empty hash)
# if no blocks are defined.
#------------------------------------------------------------------------

sub blocks {
    return $_[0]->{ _DEFBLOCKS };
}


#------------------------------------------------------------------------
# process($context)
#
# Process the document in a particular context.  Checks for recursion,
# registers the document with the context via visit(), processes itself,
# and then unwinds with a large gin and tonic.
#------------------------------------------------------------------------

sub process {
    my ($self, $context) = @_;
    my $defblocks = $self->{ _DEFBLOCKS };
    my $output;


    # check we're not already visiting this template
    return $context->throw(Template::Constants::ERROR_FILE, 
                           "recursion into '$self->{ name }'")
        if $self->{ _HOT } && ! $context->{ RECURSION };   ## RETURN ##

    $context->visit($self, $defblocks);

    $self->{ _HOT } = 1;
    eval {
        my $block = $self->{ _BLOCK };
        $output = &$block($context);
    };
    $self->{ _HOT } = 0;

    $context->leave();

    die $context->catch($@)
        if $@;
        
    return $output;
}


#------------------------------------------------------------------------
# AUTOLOAD
#
# Provides pseudo-methods for read-only access to various internal 
# members. 
#------------------------------------------------------------------------

sub AUTOLOAD {
    my $self   = shift;
    my $method = $AUTOLOAD;

    $method =~ s/.*:://;
    return if $method eq 'DESTROY';
#    my ($pkg, $file, $line) = caller();
#    print STDERR "called $self->AUTOLOAD($method) from $file line $line\n";
    return $self->{ $method };
}


#========================================================================
#                     -----  PRIVATE METHODS -----
#========================================================================


#------------------------------------------------------------------------
# _dump()
#
# Debug method which returns a string representing the internal state
# of the object.
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    my $dblks;
    my $output = "$self : $self->{ name }\n";

    $output .= "BLOCK: $self->{ _BLOCK }\nDEFBLOCKS:\n";

    if ($dblks = $self->{ _DEFBLOCKS }) {
        foreach my $b (keys %$dblks) {
            $output .= "    $b: $dblks->{ $b }\n";
        }
    }

    return $output;
}


#========================================================================
#                      ----- CLASS METHODS -----
#========================================================================

#------------------------------------------------------------------------
# as_perl($content)
#
# This method expects a reference to a hash passed as the first argument
# containing 3 items:
#     METADATA   # a hash of template metadata
#     BLOCK      # string containing Perl sub definition for main block
#     DEFBLOCKS  # hash containing further subs for addional BLOCK defs
# It returns a string containing Perl code which, when evaluated and 
# executed, will instantiate a new Template::Document object with the 
# above data.  On error, it returns undef with an appropriate error
# message set in $ERROR.
#------------------------------------------------------------------------

sub as_perl {
    my ($class, $content) = @_;
    my ($block, $defblocks, $metadata) = @$content{ qw( BLOCK DEFBLOCKS METADATA ) };

    $block =~ s/\n(?!#line)/\n    /g;
    $block =~ s/\s+$//;

    $defblocks = join('', map {
        my $code = $defblocks->{ $_ };
        $code =~ s/\n(?!#line)/\n        /g;
        $code =~ s/\s*$//;
        "        '$_' => $code,\n";
    } keys %$defblocks);
    $defblocks =~ s/\s+$//;

    $metadata = join('', map { 
        my $x = $metadata->{ $_ }; 
        $x =~ s/(['\\])/\\$1/g; 
        "        '$_' => '$x',\n";
    } keys %$metadata);
    $metadata =~ s/\s+$//;

    return <<EOF
#------------------------------------------------------------------------
# Compiled template generated by the Template Toolkit version $Template::VERSION
#------------------------------------------------------------------------

$class->new({
    METADATA => {
$metadata
    },
    BLOCK => $block,
    DEFBLOCKS => {
$defblocks
    },
});
EOF
}


#------------------------------------------------------------------------
# write_perl_file($filename, \%content)
#
# This method calls as_perl() to generate the Perl code to represent a
# compiled template with the content passed as the second argument.
# It then writes this to the file denoted by the first argument.
#
# Returns 1 on success.  On error, sets the $ERROR package variable
# to contain an error message and returns undef.
#------------------------------------------------------------------------

sub write_perl_file {
    my ($class, $file, $content) = @_;
    my ($fh, $tmpfile);
    
    return $class->error("invalid filename: $file")
        unless $file =~ /^(.+)$/s;

    eval {
        require File::Temp;
        require File::Basename;
        ($fh, $tmpfile) = File::Temp::tempfile( 
            DIR => File::Basename::dirname($file) 
        );
        my $perlcode = $class->as_perl($content) || die $!;
        
        if ($UNICODE && is_utf8($perlcode)) {
            $perlcode = "use utf8;\n\n$perlcode";
            binmode $fh, ":utf8";
        }
        print $fh $perlcode;
        close($fh);
    };
    return $class->error($@) if $@;
    return rename($tmpfile, $file)
        || $class->error($!);
}


#------------------------------------------------------------------------
# catch_warnings($msg)
#
# Installed as
#------------------------------------------------------------------------

sub catch_warnings {
    $COMPERR .= join('', @_); 
}

    
1;

__END__

=head1 NAME

Template::Document - Compiled template document object

=head1 SYNOPSIS

    use Template::Document;
    
    $doc = Template::Document->new({
        BLOCK => sub { # some perl code; return $some_text },
        DEFBLOCKS => {
            header => sub { # more perl code; return $some_text },
            footer => sub { # blah blah blah; return $some_text },
        },
        METADATA => {
            author  => 'Andy Wardley',
            version => 3.14,
        }
    }) || die $Template::Document::ERROR;
    
    print $doc->process($context);

=head1 DESCRIPTION

This module defines an object class whose instances represent compiled
template documents.  The L<Template::Parser> module creates a
C<Template::Document> instance to encapsulate a template as it is compiled
into Perl code.

The constructor method, L<new()>, expects a reference to a hash array
containing the C<BLOCK>, C<DEFBLOCKS> and C<METADATA> items.  

The C<BLOCK> item should contain a reference to a Perl subroutine or a textual
representation of Perl code, as generated by the L<Template::Parser> module.
This is then evaluated into a subroutine reference using C<eval()>. 

The C<DEFLOCKS> item should reference a hash array containing further named
C<BLOCK>s which may be defined in the template. The keys represent C<BLOCK>
names and the values should be subroutine references or text strings of Perl
code as per the main C<BLOCK> item. 

The C<METADATA> item should reference a hash array of metadata items relevant
to the document.

The L<process()> method can then be called on the instantiated
C<Template::Document> object, passing a reference to a L<Template::Context>
object as the first parameter. This will install any locally defined blocks
(C<DEFBLOCKS>) in the C<BLOCKS> cache in the context (via a call to
L<visit()|Template::Context#visit()>) so that they may be subsequently
resolved by the context. The main C<BLOCK> subroutine is then executed,
passing the context reference on as a parameter. The text returned from the
template subroutine is then returned by the L<process()> method, after calling
the context L<leave()|Template::Context#leave()> method to permit cleanup and
de-registration of named C<BLOCKS> previously installed.

An C<AUTOLOAD> method provides access to the C<METADATA> items for the
document. The L<Template::Service> module installs a reference to the main
C<Template::Document> object in the stash as the C<template> variable. This allows
metadata items to be accessed from within templates, including C<PRE_PROCESS>
templates.

header:

    <html>
    <head>
    <title>[% template.title %]
    </head>
    ...

C<Template::Document> objects are usually created by the L<Template::Parser>
but can be manually instantiated or sub-classed to provide custom
template components.

=head1 METHODS

=head2 new(\%config)

Constructor method which accept a reference to a hash array containing the
structure as shown in this example:

    $doc = Template::Document->new({
        BLOCK => sub { # some perl code; return $some_text },
        DEFBLOCKS => {
            header => sub { # more perl code; return $some_text },
            footer => sub { # blah blah blah; return $some_text },
        },
        METADATA => {
            author  => 'Andy Wardley',
            version => 3.14,
        }
    }) || die $Template::Document::ERROR;

C<BLOCK> and C<DEFBLOCKS> items may be expressed as references to Perl subroutines
or as text strings containing Perl subroutine definitions, as is generated
by the L<Template::Parser> module.  These are evaluated into subroutine references
using C<eval()>.

Returns a new C<Template::Document> object or C<undef> on error. The
L<error()|Template::Base#error()> class method can be called, or the C<$ERROR>
package variable inspected to retrieve the relevant error message.

=head2 process($context)

Main processing routine for the compiled template document. A reference to a
L<Template::Context> object should be passed as the first parameter. The
method installs any locally defined blocks via a call to the context
L<visit()|Template::Context#visit()> method, processes its own template,
(passing the context reference as a parameter) and then calls
L<leave()|Template::Context#leave()> in the context to allow cleanup.

    print $doc->process($context);

Returns a text string representing the generated output for the template.
Errors are thrown via C<die()>.

=head2 block()

Returns a reference to the main C<BLOCK> subroutine.

=head2 blocks()

Returns a reference to the hash array of named C<DEFBLOCKS> subroutines.

=head2 AUTOLOAD

An autoload method returns C<METADATA> items.

    print $doc->author();

=head1 PACKAGE SUB-ROUTINES

=head2 write_perl_file(\%config)

This package subroutine is provided to effect persistence of compiled
templates.  If the C<COMPILE_EXT> option (to indicate a file extension
for saving compiled templates) then the L<Template::Parser> module calls
this subroutine before calling the L<new()> constructor.  At this stage,
the parser has a representation of the template as text strings
containing Perl code.  We can write that to a file, enclosed in a
small wrapper which will allow us to susequently C<require()> the file
and have Perl parse and compile it into a C<Template::Document>.  Thus we
have persistence of compiled templates.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template>, L<Template::Parser>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
