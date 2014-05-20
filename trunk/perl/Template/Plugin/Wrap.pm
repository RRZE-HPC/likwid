#============================================================= -*-Perl-*-
#
# Template::Plugin::Wrap
#
# DESCRIPTION
#   Plugin for wrapping text via the Text::Wrap module.
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

package Template::Plugin::Wrap;

use strict;
use warnings;
use base 'Template::Plugin';
use Text::Wrap;

our $VERSION = 2.68;

sub new {
    my ($class, $context, $format) = @_;;
    $context->define_filter('wrap', [ \&wrap_filter_factory => 1 ]);
    return \&tt_wrap;
}

sub tt_wrap {
    my $text  = shift;
    my $width = shift || 72;
    my $itab  = shift;
    my $ntab  = shift;
    $itab = '' unless defined $itab;
    $ntab = '' unless defined $ntab;
    $Text::Wrap::columns = $width;
    Text::Wrap::wrap($itab, $ntab, $text);
}

sub wrap_filter_factory {
    my ($context, @args) = @_;
    return sub {
        my $text = shift;
        tt_wrap($text, @args);
    }
}


1;

__END__

=head1 NAME

Template::Plugin::Wrap - Plugin interface to Text::Wrap

=head1 SYNOPSIS

    [% USE wrap %]
    
    # call wrap subroutine
    [% wrap(mytext, width, initial_tab,  subsequent_tab) %]
    
    # or use wrap FILTER
    [% mytext FILTER wrap(width, initital_tab, subsequent_tab) %]

=head1 DESCRIPTION

This plugin provides an interface to the L<Text::Wrap> module which 
provides simple paragraph formatting.

It defines a C<wrap> subroutine which can be called, passing the input
text and further optional parameters to specify the page width (default:
72), and tab characters for the first and subsequent lines (no defaults).

    [% USE wrap %]
    
    [% text = BLOCK %]
    First, attach the transmutex multiplier to the cross-wired 
    quantum homogeniser.
    [% END %]
    
    [% wrap(text, 40, '* ', '  ') %]

Output:

    * First, attach the transmutex
      multiplier to the cross-wired quantum
      homogeniser.

It also registers a C<wrap> filter which accepts the same three optional 
arguments but takes the input text directly via the filter input.

Example 1:

    [% FILTER bullet = wrap(40, '* ', '  ') -%]
    First, attach the transmutex multiplier to the cross-wired quantum
    homogeniser.
    [%- END %]

Output:

    * First, attach the transmutex
      multiplier to the cross-wired quantum
      homogeniser.

Example 2:

    [% FILTER bullet -%]
    Then remodulate the shield to match the harmonic frequency, taking 
    care to correct the phase difference.
    [% END %]

Output:

    * Then remodulate the shield to match
      the harmonic frequency, taking 
      care to correct the phase difference.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

The L<Text::Wrap> module was written by David Muir Sharnoff
with help from Tim Pierce and many others.

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<Text::Wrap>

