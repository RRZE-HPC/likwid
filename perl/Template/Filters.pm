#============================================================= -*-Perl-*-
#
# Template::Filters
#
# DESCRIPTION
#   Defines filter plugins as used by the FILTER directive.
#
# AUTHORS
#   Andy Wardley <abw@wardley.org>, with a number of filters contributed
#   by Leslie Michael Orchard <deus_x@nijacode.com>
#
# COPYRIGHT
#   Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Filters;

use strict;
use warnings;
use locale;
use base 'Template::Base';
use Template::Constants;
use Scalar::Util 'blessed';

our $VERSION         = 2.87;
our $AVAILABLE       = { };
our $TRUNCATE_LENGTH = 32;
our $TRUNCATE_ADDON  = '...';


#------------------------------------------------------------------------
# standard filters, defined in one of the following forms:
#   name =>   \&static_filter
#   name => [ \&subref, $is_dynamic ]
# If the $is_dynamic flag is set then the sub-routine reference 
# is called to create a new filter each time it is requested;  if
# not set, then it is a single, static sub-routine which is returned
# for every filter request for that name.
#------------------------------------------------------------------------

our $FILTERS = {
    # static filters 
    'html'            => \&html_filter,
    'html_para'       => \&html_paragraph,
    'html_break'      => \&html_para_break,
    'html_para_break' => \&html_para_break,
    'html_line_break' => \&html_line_break,
    'xml'             => \&xml_filter,
    'uri'             => \&uri_filter,
    'url'             => \&url_filter,
    'upper'           => sub { uc $_[0] },
    'lower'           => sub { lc $_[0] },
    'ucfirst'         => sub { ucfirst $_[0] },
    'lcfirst'         => sub { lcfirst $_[0] },
    'stderr'          => sub { print STDERR @_; return '' },
    'trim'            => sub { for ($_[0]) { s/^\s+//; s/\s+$// }; $_[0] },
    'null'            => sub { return '' },
    'collapse'        => sub { for ($_[0]) { s/^\s+//; s/\s+$//; s/\s+/ /g };
                               $_[0] },

    # dynamic filters
    'html_entity' => [ \&html_entity_filter_factory, 1 ],
    'indent'      => [ \&indent_filter_factory,      1 ],
    'format'      => [ \&format_filter_factory,      1 ],
    'truncate'    => [ \&truncate_filter_factory,    1 ],
    'repeat'      => [ \&repeat_filter_factory,      1 ],
    'replace'     => [ \&replace_filter_factory,     1 ],
    'remove'      => [ \&remove_filter_factory,      1 ],
    'eval'        => [ \&eval_filter_factory,        1 ],
    'evaltt'      => [ \&eval_filter_factory,        1 ],  # alias
    'perl'        => [ \&perl_filter_factory,        1 ],
    'evalperl'    => [ \&perl_filter_factory,        1 ],  # alias
    'redirect'    => [ \&redirect_filter_factory,    1 ],
    'file'        => [ \&redirect_filter_factory,    1 ],  # alias
    'stdout'      => [ \&stdout_filter_factory,      1 ],
};

# name of module implementing plugin filters
our $PLUGIN_FILTER = 'Template::Plugin::Filter';



#========================================================================
#                         -- PUBLIC METHODS --
#========================================================================

#------------------------------------------------------------------------
# fetch($name, \@args, $context)
#
# Attempts to instantiate or return a reference to a filter sub-routine 
# named by the first parameter, $name, with additional constructor 
# arguments passed by reference to a list as the second parameter, 
# $args.  A reference to the calling Template::Context object is 
# passed as the third paramter.
#
# Returns a reference to a filter sub-routine or a pair of values
# (undef, STATUS_DECLINED) or ($error, STATUS_ERROR) to decline to
# deliver the filter or to indicate an error.
#------------------------------------------------------------------------

sub fetch {
    my ($self, $name, $args, $context) = @_;
    my ($factory, $is_dynamic, $filter, $error);

    $self->debug("fetch($name, ", 
                 defined $args ? ('[ ', join(', ', @$args), ' ]') : '<no args>', ', ',
                 defined $context ? $context : '<no context>', 
                 ')') if $self->{ DEBUG };

    # allow $name to be specified as a reference to 
    # a plugin filter object;  any other ref is 
    # assumed to be a coderef and hence already a filter;
    # non-refs are assumed to be regular name lookups

    if (ref $name) {
        if (blessed($name) && $name->isa($PLUGIN_FILTER)) {
            $factory = $name->factory()
                || return $self->error($name->error());
        }
        else {
            return $name;
        }
    }
    else {
        return (undef, Template::Constants::STATUS_DECLINED)
            unless ($factory = $self->{ FILTERS }->{ $name }
                    || $FILTERS->{ $name });
    }

    # factory can be an [ $code, $dynamic ] or just $code
    if (ref $factory eq 'ARRAY') {
        ($factory, $is_dynamic) = @$factory;
    }
    else {
        $is_dynamic = 0;
    }

    if (ref $factory eq 'CODE') {
        if ($is_dynamic) {
            # if the dynamic flag is set then the sub-routine is a 
            # factory which should be called to create the actual 
            # filter...
            eval {
                ($filter, $error) = &$factory($context, $args ? @$args : ());
            };
            $error ||= $@;
            $error = "invalid FILTER for '$name' (not a CODE ref)"
                unless $error || ref($filter) eq 'CODE';
        }
        else {
            # ...otherwise, it's a static filter sub-routine
            $filter = $factory;
        }
    }
    else {
        $error = "invalid FILTER entry for '$name' (not a CODE ref)";
    }

    if ($error) {
        return $self->{ TOLERANT } 
               ? (undef,  Template::Constants::STATUS_DECLINED) 
               : ($error, Template::Constants::STATUS_ERROR) ;
    }
    else {
        return $filter;
    }
}


#------------------------------------------------------------------------
# store($name, \&filter)
#
# Stores a new filter in the internal FILTERS hash.  The first parameter
# is the filter name, the second a reference to a subroutine or 
# array, as per the standard $FILTERS entries.
#------------------------------------------------------------------------

sub store {
    my ($self, $name, $filter) = @_;

    $self->debug("store($name, $filter)") if $self->{ DEBUG };

    $self->{ FILTERS }->{ $name } = $filter;
    return 1;
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

    $self->{ FILTERS  } = $params->{ FILTERS } || { };
    $self->{ TOLERANT } = $params->{ TOLERANT }  || 0;
    $self->{ DEBUG    } = ( $params->{ DEBUG } || 0 )
                          & Template::Constants::DEBUG_FILTERS;


    return $self;
}



#------------------------------------------------------------------------
# _dump()
# 
# Debug method
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    my $output = "[Template::Filters] {\n";
    my $format = "    %-16s => %s\n";
    my $key;

    foreach $key (qw( TOLERANT )) {
        my $val = $self->{ $key };
        $val = '<undef>' unless defined $val;
        $output .= sprintf($format, $key, $val);
    }

    my $filters = $self->{ FILTERS };
    $filters = join('', map { 
        sprintf("    $format", $_, $filters->{ $_ });
    } keys %$filters);
    $filters = "{\n$filters    }";
    
    $output .= sprintf($format, 'FILTERS (local)' => $filters);

    $filters = $FILTERS;
    $filters = join('', map { 
        my $f = $filters->{ $_ };
        my ($ref, $dynamic) = ref $f eq 'ARRAY' ? @$f : ($f, 0);
        sprintf("    $format", $_, $dynamic ? 'dynamic' : 'static');
    } sort keys %$filters);
    $filters = "{\n$filters    }";
    
    $output .= sprintf($format, 'FILTERS (global)' => $filters);

    $output .= '}';
    return $output;
}


#========================================================================
#                         -- STATIC FILTER SUBS --
#========================================================================

#------------------------------------------------------------------------
# uri_filter()                                           [% FILTER uri %]
#
# URI escape a string.  This code is borrowed from Gisle Aas' URI::Escape
# module, copyright 1995-2004.  See RFC2396 for details.
#-----------------------------------------------------------------------

# cache of escaped characters
our $URI_ESCAPES;

sub uri_filter {
    my $text = shift;

    $URI_ESCAPES ||= {
        map { ( chr($_), sprintf("%%%02X", $_) ) } (0..255),
    };

    if ($] >= 5.008 && utf8::is_utf8($text)) {
        utf8::encode($text);
    }
    
    $text =~ s/([^A-Za-z0-9\-_.!~*'()])/$URI_ESCAPES->{$1}/eg;
    $text;
}

#------------------------------------------------------------------------
# url_filter()                                           [% FILTER uri %]
#
# NOTE: the difference: url vs uri. 
# This implements the old-style, non-strict behaviour of the uri filter 
# which allows any valid URL characters to pass through so that 
# http://example.com/blah.html does not get the ':' and '/' characters 
# munged. 
#-----------------------------------------------------------------------

sub url_filter {
    my $text = shift;

    $URI_ESCAPES ||= {
        map { ( chr($_), sprintf("%%%02X", $_) ) } (0..255),
    };

    if ($] >= 5.008 && utf8::is_utf8($text)) {
        utf8::encode($text);
    }
    
    $text =~ s/([^;\/?:@&=+\$,A-Za-z0-9\-_.!~*'()])/$URI_ESCAPES->{$1}/eg;
    $text;
}


#------------------------------------------------------------------------
# html_filter()                                         [% FILTER html %]
#
# Convert any '<', '>' or '&' characters to the HTML equivalents, '&lt;',
# '&gt;' and '&amp;', respectively. 
#------------------------------------------------------------------------

sub html_filter {
    my $text = shift;
    for ($text) {
        s/&/&amp;/g;
        s/</&lt;/g;
        s/>/&gt;/g;
        s/"/&quot;/g;
    }
    return $text;
}


#------------------------------------------------------------------------
# xml_filter()                                           [% FILTER xml %]
#
# Same as the html filter, but adds the conversion of ' to &apos; which
# is native to XML.
#------------------------------------------------------------------------

sub xml_filter {
    my $text = shift;
    for ($text) {
        s/&/&amp;/g;
        s/</&lt;/g;
        s/>/&gt;/g;
        s/"/&quot;/g;
        s/'/&apos;/g;
    }
    return $text;
}


#------------------------------------------------------------------------
# html_paragraph()                                 [% FILTER html_para %]
#
# Wrap each paragraph of text (delimited by two or more newlines) in the
# <p>...</p> HTML tags.
#------------------------------------------------------------------------

sub html_paragraph  {
    my $text = shift;
    return "<p>\n" 
           . join("\n</p>\n\n<p>\n", split(/(?:\r?\n){2,}/, $text))
           . "</p>\n";
}


#------------------------------------------------------------------------
# html_para_break()                          [% FILTER html_para_break %]
#                                               
# Join each paragraph of text (delimited by two or more newlines) with
# <br><br> HTML tags.
#------------------------------------------------------------------------

sub html_para_break  {
    my $text = shift;
    $text =~ s|(\r?\n){2,}|$1<br />$1<br />$1|g;
    return $text;
}

#------------------------------------------------------------------------
# html_line_break()                          [% FILTER html_line_break %]
#
# replaces any newlines with <br> HTML tags.
#------------------------------------------------------------------------

sub html_line_break  {
    my $text = shift;
    $text =~ s|(\r?\n)|<br />$1|g;
    return $text;
}

#========================================================================
#                    -- DYNAMIC FILTER FACTORIES --
#========================================================================

#------------------------------------------------------------------------
# html_entity_filter_factory(\%options)                 [% FILTER html %]
#
# Dynamic version of the static html filter which attempts to locate the
# Apache::Util or HTML::Entities modules to perform full entity encoding
# of the text passed.  Returns an exception if one or other of the 
# modules can't be located.
#------------------------------------------------------------------------

sub use_html_entities {
    require HTML::Entities;
    return ($AVAILABLE->{ HTML_ENTITY } = \&HTML::Entities::encode_entities);
}

sub use_apache_util {
    require Apache::Util;
    Apache::Util::escape_html('');      # TODO: explain this
    return ($AVAILABLE->{ HTML_ENTITY } = \&Apache::Util::escape_html);
}

sub html_entity_filter_factory {
    my $context = shift;
    my $haz;
    
    # if Apache::Util is installed then we use escape_html
    $haz = $AVAILABLE->{ HTML_ENTITY } 
       ||  eval { use_apache_util()   }
       ||  eval { use_html_entities() }
       ||  -1;      # we use -1 for "not available" because it's a true value

    return ref $haz eq 'CODE'
        ? $haz
        : (undef, Template::Exception->new( 
            html_entity => 'cannot locate Apache::Util or HTML::Entities' )
          );
}


#------------------------------------------------------------------------
# indent_filter_factory($pad)                    [% FILTER indent(pad) %]
#
# Create a filter to indent text by a fixed pad string or when $pad is
# numerical, a number of space. 
#------------------------------------------------------------------------

sub indent_filter_factory {
    my ($context, $pad) = @_;
    $pad = 4 unless defined $pad;
    $pad = ' ' x $pad if $pad =~ /^\d+$/;

    return sub {
        my $text = shift;
        $text = '' unless defined $text;
        $text =~ s/^/$pad/mg;
        return $text;
    }
}

#------------------------------------------------------------------------
# format_filter_factory()                     [% FILTER format(format) %]
#
# Create a filter to format text according to a printf()-like format
# string.
#------------------------------------------------------------------------

sub format_filter_factory {
    my ($context, $format) = @_;
    $format = '%s' unless defined $format;

    return sub {
        my $text = shift;
        $text = '' unless defined $text;
        return join("\n", map{ sprintf($format, $_) } split(/\n/, $text));
    }
}


#------------------------------------------------------------------------
# repeat_filter_factory($n)                        [% FILTER repeat(n) %]
#
# Create a filter to repeat text n times.
#------------------------------------------------------------------------

sub repeat_filter_factory {
    my ($context, $iter) = @_;
    $iter = 1 unless defined $iter and length $iter;

    return sub {
        my $text = shift;
        $text = '' unless defined $text;
        return join('\n', $text) x $iter;
    }
}


#------------------------------------------------------------------------
# replace_filter_factory($s, $r)    [% FILTER replace(search, replace) %]
#
# Create a filter to replace 'search' text with 'replace'
#------------------------------------------------------------------------

sub replace_filter_factory {
    my ($context, $search, $replace) = @_;
    $search = '' unless defined $search;
    $replace = '' unless defined $replace;

    return sub {
        my $text = shift;
        $text = '' unless defined $text;
        $text =~ s/$search/$replace/g;
        return $text;
    }
}


#------------------------------------------------------------------------
# remove_filter_factory($text)                  [% FILTER remove(text) %]
#
# Create a filter to remove 'search' string from the input text.
#------------------------------------------------------------------------

sub remove_filter_factory {
    my ($context, $search) = @_;

    return sub {
        my $text = shift;
        $text = '' unless defined $text;
        $text =~ s/$search//g;
        return $text;
    }
}


#------------------------------------------------------------------------
# truncate_filter_factory($n)                    [% FILTER truncate(n) %]
#
# Create a filter to truncate text after n characters.
#------------------------------------------------------------------------

sub truncate_filter_factory {
    my ($context, $len, $char) = @_;
    $len  = $TRUNCATE_LENGTH unless defined $len;
    $char = $TRUNCATE_ADDON  unless defined $char;

    # Length of char is the minimum length
    my $lchar = length $char;
    if ($len < $lchar) {
        $char  = substr($char, 0, $len);
        $lchar = $len;
    }

    return sub {
        my $text = shift;
        return $text if length $text <= $len;
        return substr($text, 0, $len - $lchar) . $char;


    }
}


#------------------------------------------------------------------------
# eval_filter_factory                                   [% FILTER eval %]
# 
# Create a filter to evaluate template text.
#------------------------------------------------------------------------

sub eval_filter_factory {
    my $context = shift;

    return sub {
        my $text = shift;
        $context->process(\$text);
    }
}


#------------------------------------------------------------------------
# perl_filter_factory                                   [% FILTER perl %]
# 
# Create a filter to process Perl text iff the context EVAL_PERL flag 
# is set.
#------------------------------------------------------------------------

sub perl_filter_factory {
    my $context = shift;
    my $stash = $context->stash;

    return (undef, Template::Exception->new('perl', 'EVAL_PERL is not set'))
        unless $context->eval_perl();

    return sub {
        my $text = shift;
        local($Template::Perl::context) = $context;
        local($Template::Perl::stash)   = $stash;
        my $out = eval <<EOF;
package Template::Perl; 
\$stash = \$context->stash(); 
$text
EOF
        $context->throw($@) if $@;
        return $out;
    }
}


#------------------------------------------------------------------------
# redirect_filter_factory($context, $file)    [% FILTER redirect(file) %]
#
# Create a filter to redirect the block text to a file.
#------------------------------------------------------------------------

sub redirect_filter_factory {
    my ($context, $file, $options) = @_;
    my $outpath = $context->config->{ OUTPUT_PATH };

    return (undef, Template::Exception->new('redirect', 
                                            'OUTPUT_PATH is not set'))
        unless $outpath;

    $context->throw('redirect', "relative filenames are not supported: $file")
        if $file =~ m{(^|/)\.\./};

    $options = { binmode => $options } unless ref $options;

    sub {
        my $text = shift;
        my $outpath = $context->config->{ OUTPUT_PATH }
            || return '';
        $outpath .= "/$file";
        my $error = Template::_output($outpath, \$text, $options);
        die Template::Exception->new('redirect', $error)
            if $error;
        return '';
    }
}


#------------------------------------------------------------------------
# stdout_filter_factory($context, $binmode)    [% FILTER stdout(binmode) %]
#
# Create a filter to print a block to stdout, with an optional binmode.
#------------------------------------------------------------------------

sub stdout_filter_factory {
    my ($context, $options) = @_;

    $options = { binmode => $options } unless ref $options;

    sub {
        my $text = shift;
        binmode(STDOUT) if $options->{ binmode };
        print STDOUT $text;
        return '';
    }
}


1;

__END__

=head1 NAME

Template::Filters - Post-processing filters for template blocks

=head1 SYNOPSIS

    use Template::Filters;
    
    $filters = Template::Filters->new(\%config);
    
    ($filter, $error) = $filters->fetch($name, \@args, $context);
    
    if ($filter) {
        print &$filter("some text");
    }
    else {
        print "Could not fetch $name filter: $error\n";
    }

=head1 DESCRIPTION

The C<Template::Filters> module implements a provider for creating subroutines
that implement the standard filters. Additional custom filters may be provided
via the L<FILTERS> configuration option.

=head1 METHODS

=head2 new(\%params) 

Constructor method which instantiates and returns a reference to a
C<Template::Filters> object.  A reference to a hash array of configuration
items may be passed as a parameter.  These are described below.  

    my $filters = Template::Filters->new({
        FILTERS => { ... },
    });
    
    my $template = Template->new({
        LOAD_FILTERS => [ $filters ],
    });

A default C<Template::Filters> module is created by the L<Template> module
if the L<LOAD_FILTERS> option isn't specified.  All configuration parameters
are forwarded to the constructor.

    $template = Template->new({
        FILTERS => { ... },
    });

=head2 fetch($name, \@args, $context)

Called to request that a filter of a given name be provided.  The name
of the filter should be specified as the first parameter.  This should
be one of the standard filters or one specified in the L<FILTERS>
configuration hash.  The second argument should be a reference to an
array containing configuration parameters for the filter.  This may be
specified as 0, or undef where no parameters are provided.  The third
argument should be a reference to the current L<Template::Context>
object.

The method returns a reference to a filter sub-routine on success.  It
may also return C<(undef, STATUS_DECLINE)> to decline the request, to allow
delegation onto other filter providers in the L<LOAD_FILTERS> chain of 
responsibility.  On error, C<($error, STATUS_ERROR)> is returned where $error
is an error message or L<Template::Exception> object indicating the error
that occurred. 

When the C<TOLERANT> option is set, errors are automatically downgraded to
a C<STATUS_DECLINE> response.

=head2 use_html_entities()

This class method can be called to configure the C<html_entity> filter to use
the L<HTML::Entities> module. An error will be raised if it is not installed
on your system.

    use Template::Filters;
    Template::Filters->use_html_entities();

=head2 use_apache_util()

This class method can be called to configure the C<html_entity> filter to use
the L<Apache::Util> module. An error will be raised if it is not installed on
your system.

    use Template::Filters;
    Template::Filters->use_apache_util();

=head1 CONFIGURATION OPTIONS

The following list summarises the configuration options that can be provided
to the C<Template::Filters> L<new()> constructor. Please see
L<Template::Manual::Config> for further information about each option.

=head2 FILTERS

The L<FILTERS|Template::Manual::Config#FILTERS> option can be used to specify
custom filters which can then be used with the
L<FILTER|Template::Manual::Directives#FILTER> directive like any other. These
are added to the standard filters which are available by default.

    $filters = Template::Filters->new({
        FILTERS => {
            'sfilt1' =>   \&static_filter,
            'dfilt1' => [ \&dyanamic_filter_factory, 1 ],
        },
    });

=head2 TOLERANT

The L<TOLERANT|Template::Manual::Config#TOLERANT> flag can be set to indicate
that the C<Template::Filters> module should ignore any errors and instead
return C<STATUS_DECLINED>.

=head2 DEBUG

The L<DEBUG|Template::Manual::Config#DEBUG> option can be used to enable
debugging messages for the Template::Filters module by setting it to include
the C<DEBUG_FILTERS> value.

    use Template::Constants qw( :debug );
    
    my $template = Template->new({
        DEBUG => DEBUG_FILTERS | DEBUG_PLUGINS,
    });

=head1 STANDARD FILTERS

Please see L<Template::Manual::Filters> for a list of the filters provided
with the Template Toolkit, complete with examples of use.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Manual::Filters>, L<Template>, L<Template::Context>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
