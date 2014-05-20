#============================================================= -*-Perl-*-
#
# Template::Constants.pm
#
# DESCRIPTION
#   Definition of constants for the Template Toolkit.
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
 
package Template::Constants;

require Exporter;
use strict;
use warnings;
use Exporter;
# Perl::MinimumVersion seems to think this is a Perl 5.008ism...
# use base qw( Exporter );
use vars qw( @EXPORT_OK %EXPORT_TAGS );
use vars qw( $DEBUG_OPTIONS @STATUS @ERROR @CHOMP @DEBUG @ISA );
# ... so we'll do it the Old Skool way just to keep it quiet
@ISA = qw( Exporter );

our $VERSION = 2.75;


#========================================================================
#                         ----- EXPORTER -----
#========================================================================

# STATUS constants returned by directives
use constant STATUS_OK       =>   0;      # ok
use constant STATUS_RETURN   =>   1;      # ok, block ended by RETURN
use constant STATUS_STOP     =>   2;      # ok, stoppped by STOP 
use constant STATUS_DONE     =>   3;      # ok, iterator done
use constant STATUS_DECLINED =>   4;      # ok, declined to service request
use constant STATUS_ERROR    => 255;      # error condition

# ERROR constants for indicating exception types
use constant ERROR_RETURN    =>  'return'; # return a status code
use constant ERROR_FILE      =>  'file';   # file error: I/O, parse, recursion
use constant ERROR_VIEW      =>  'view';   # view error
use constant ERROR_UNDEF     =>  'undef';  # undefined variable value used
use constant ERROR_PERL      =>  'perl';   # error in [% PERL %] block
use constant ERROR_FILTER    =>  'filter'; # filter error
use constant ERROR_PLUGIN    =>  'plugin'; # plugin error

# CHOMP constants for PRE_CHOMP and POST_CHOMP
use constant CHOMP_NONE      => 0; # do not remove whitespace
use constant CHOMP_ALL       => 1; # remove whitespace up to newline
use constant CHOMP_ONE       => 1; # new name for CHOMP_ALL
use constant CHOMP_COLLAPSE  => 2; # collapse whitespace to a single space
use constant CHOMP_GREEDY    => 3; # remove all whitespace including newlines

# DEBUG constants to enable various debugging options
use constant DEBUG_OFF       =>    0; # do nothing
use constant DEBUG_ON        =>    1; # basic debugging flag
use constant DEBUG_UNDEF     =>    2; # throw undef on undefined variables
use constant DEBUG_VARS      =>    4; # general variable debugging
use constant DEBUG_DIRS      =>    8; # directive debugging
use constant DEBUG_STASH     =>   16; # general stash debugging
use constant DEBUG_CONTEXT   =>   32; # context debugging
use constant DEBUG_PARSER    =>   64; # parser debugging
use constant DEBUG_PROVIDER  =>  128; # provider debugging
use constant DEBUG_PLUGINS   =>  256; # plugins debugging
use constant DEBUG_FILTERS   =>  512; # filters debugging
use constant DEBUG_SERVICE   => 1024; # context debugging
use constant DEBUG_ALL       => 2047; # everything

# extra debugging flags
use constant DEBUG_CALLER    => 4096; # add caller file/line
use constant DEBUG_FLAGS     => 4096; # bitmask to extraxt flags

$DEBUG_OPTIONS  = {
    &DEBUG_OFF      => off      => off      => &DEBUG_OFF,
    &DEBUG_ON       => on       => on       => &DEBUG_ON,
    &DEBUG_UNDEF    => undef    => undef    => &DEBUG_UNDEF,
    &DEBUG_VARS     => vars     => vars     => &DEBUG_VARS,
    &DEBUG_DIRS     => dirs     => dirs     => &DEBUG_DIRS,
    &DEBUG_STASH    => stash    => stash    => &DEBUG_STASH,
    &DEBUG_CONTEXT  => context  => context  => &DEBUG_CONTEXT,
    &DEBUG_PARSER   => parser   => parser   => &DEBUG_PARSER,
    &DEBUG_PROVIDER => provider => provider => &DEBUG_PROVIDER,
    &DEBUG_PLUGINS  => plugins  => plugins  => &DEBUG_PLUGINS,
    &DEBUG_FILTERS  => filters  => filters  => &DEBUG_FILTERS,
    &DEBUG_SERVICE  => service  => service  => &DEBUG_SERVICE,
    &DEBUG_ALL      => all      => all      => &DEBUG_ALL,
    &DEBUG_CALLER   => caller   => caller   => &DEBUG_CALLER,
};

@STATUS  = qw( STATUS_OK STATUS_RETURN STATUS_STOP STATUS_DONE
               STATUS_DECLINED STATUS_ERROR );
@ERROR   = qw( ERROR_FILE ERROR_VIEW ERROR_UNDEF ERROR_PERL 
               ERROR_RETURN ERROR_FILTER ERROR_PLUGIN );
@CHOMP   = qw( CHOMP_NONE CHOMP_ALL CHOMP_ONE CHOMP_COLLAPSE CHOMP_GREEDY );
@DEBUG   = qw( DEBUG_OFF DEBUG_ON DEBUG_UNDEF DEBUG_VARS 
               DEBUG_DIRS DEBUG_STASH DEBUG_CONTEXT DEBUG_PARSER
               DEBUG_PROVIDER DEBUG_PLUGINS DEBUG_FILTERS DEBUG_SERVICE
               DEBUG_ALL DEBUG_CALLER DEBUG_FLAGS );

@EXPORT_OK   = ( @STATUS, @ERROR, @CHOMP, @DEBUG );
%EXPORT_TAGS = (
    'all'      => [ @EXPORT_OK ],
    'status'   => [ @STATUS    ],
    'error'    => [ @ERROR     ],
    'chomp'    => [ @CHOMP     ],
    'debug'    => [ @DEBUG     ],
);


sub debug_flags {
    my ($self, $debug) = @_;
    my (@flags, $flag, $value);
    $debug = $self unless defined($debug) || ref($self);
    
    if ($debug =~ /^\d+$/) {
        foreach $flag (@DEBUG) {
            next if $flag =~ /^DEBUG_(OFF|ALL|FLAGS)$/;

            # don't trash the original
            my $copy = $flag;
            $flag =~ s/^DEBUG_//;
            $flag = lc $flag;
            return $self->error("no value for flag: $flag")
                unless defined($value = $DEBUG_OPTIONS->{ $flag });
            $flag = $value;

            if ($debug & $flag) {
                $value = $DEBUG_OPTIONS->{ $flag };
                return $self->error("no value for flag: $flag") unless defined $value;
                push(@flags, $value);
            }
        }
        return wantarray ? @flags : join(', ', @flags);
    }
    else {
        @flags = split(/\W+/, $debug);
        $debug = 0;
        foreach $flag (@flags) {
            $value = $DEBUG_OPTIONS->{ $flag };
            return $self->error("unknown debug flag: $flag") unless defined $value;
            $debug |= $value;
        }
        return $debug;
    }
}


1;

__END__

=head1 NAME

Template::Constants - Defines constants for the Template Toolkit

=head1 SYNOPSIS

    use Template::Constants qw( :status :error :all );

=head1 DESCRIPTION

The C<Template::Constants> modules defines, and optionally exports into the
caller's namespace, a number of constants used by the L<Template> package.

Constants may be used by specifying the C<Template::Constants> package 
explicitly:

    use Template::Constants;
    print Template::Constants::STATUS_DECLINED;

Constants may be imported into the caller's namespace by naming them as 
options to the C<use Template::Constants> statement:

    use Template::Constants qw( STATUS_DECLINED );
    print STATUS_DECLINED;

Alternatively, one of the following tagset identifiers may be specified
to import sets of constants: 'C<:status>', 'C<:error>', 'C<:all>'.

    use Template::Constants qw( :status );
    print STATUS_DECLINED;

Consult the documentation for the C<Exporter> module for more information 
on exporting variables.

=head1 EXPORTABLE TAG SETS

The following tag sets and associated constants are defined: 

    :status
        STATUS_OK             # no problem, continue
        STATUS_RETURN         # ended current block then continue (ok)
        STATUS_STOP           # controlled stop (ok) 
        STATUS_DONE           # iterator is all done (ok)
        STATUS_DECLINED       # provider declined to service request (ok)
        STATUS_ERROR          # general error condition (not ok)

    :error
        ERROR_RETURN          # return a status code (e.g. 'stop')
        ERROR_FILE            # file error: I/O, parse, recursion
        ERROR_UNDEF           # undefined variable value used
        ERROR_PERL            # error in [% PERL %] block
        ERROR_FILTER          # filter error
        ERROR_PLUGIN          # plugin error

    :chomp                  # for PRE_CHOMP and POST_CHOMP
        CHOMP_NONE            # do not remove whitespace
        CHOMP_ONE             # remove whitespace to newline
        CHOMP_ALL             # old name for CHOMP_ONE (deprecated)
        CHOMP_COLLAPSE        # collapse whitespace to a single space
        CHOMP_GREEDY          # remove all whitespace including newlines

    :debug
        DEBUG_OFF             # do nothing
        DEBUG_ON              # basic debugging flag
        DEBUG_UNDEF           # throw undef on undefined variables
        DEBUG_VARS            # general variable debugging
        DEBUG_DIRS            # directive debugging
        DEBUG_STASH           # general stash debugging
        DEBUG_CONTEXT         # context debugging
        DEBUG_PARSER          # parser debugging
        DEBUG_PROVIDER        # provider debugging
        DEBUG_PLUGINS         # plugins debugging
        DEBUG_FILTERS         # filters debugging
        DEBUG_SERVICE         # context debugging
        DEBUG_ALL             # everything
        DEBUG_CALLER          # add caller file/line info
        DEBUG_FLAGS           # bitmap used internally

    :all
        All the above constants.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template>, C<Exporter>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:
