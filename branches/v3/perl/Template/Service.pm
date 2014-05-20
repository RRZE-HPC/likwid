#============================================================= -*-Perl-*-
#
# Template::Service
#
# DESCRIPTION
#   Module implementing a template processing service which wraps a
#   template within PRE_PROCESS and POST_PROCESS templates and offers 
#   ERROR recovery.
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

package Template::Service;

use strict;
use warnings;
use base 'Template::Base';
use Template::Config;
use Template::Exception;
use Template::Constants;
use Scalar::Util 'blessed';

use constant EXCEPTION => 'Template::Exception';

our $VERSION = 2.80;
our $DEBUG   = 0 unless defined $DEBUG;
our $ERROR   = '';


#========================================================================
#                     -----  PUBLIC METHODS -----
#========================================================================

#------------------------------------------------------------------------
# process($template, \%params)
#
# Process a template within a service framework.  A service may encompass
# PRE_PROCESS and POST_PROCESS templates and an ERROR hash which names
# templates to be substituted for the main template document in case of
# error.  Each service invocation begins by resetting the state of the 
# context object via a call to reset().  The AUTO_RESET option may be set 
# to 0 (default: 1) to bypass this step.
#------------------------------------------------------------------------

sub process {
    my ($self, $template, $params) = @_;
    my $context = $self->{ CONTEXT };
    my ($name, $output, $procout, $error);
    $output = '';

    $self->debug("process($template, ", 
                 defined $params ? $params : '<no params>',
                 ')') if $self->{ DEBUG };

    $context->reset()
        if $self->{ AUTO_RESET };

    # pre-request compiled template from context so that we can alias it 
    # in the stash for pre-processed templates to reference
    eval { $template = $context->template($template) };
    return $self->error($@)
        if $@;

    # localise the variable stash with any parameters passed
    # and set the 'template' variable
    $params ||= { };
    # TODO: change this to C<||=> so we can use a template parameter
    $params->{ template } = $template 
        unless ref $template eq 'CODE';
    $context->localise($params);

    SERVICE: {
        # PRE_PROCESS
        eval {
            foreach $name (@{ $self->{ PRE_PROCESS } }) {
                $self->debug("PRE_PROCESS: $name") if $self->{ DEBUG };
                $output .= $context->process($name);
            }
        };
        last SERVICE if ($error = $@);

        # PROCESS
        eval {
            foreach $name (@{ $self->{ PROCESS } || [ $template ] }) {
                $self->debug("PROCESS: $name") if $self->{ DEBUG };
                $procout .= $context->process($name);
            }
        };
        if ($error = $@) {
            last SERVICE
                unless defined ($procout = $self->_recover(\$error));
        }
        
        if (defined $procout) {
            # WRAPPER
            eval {
                foreach $name (reverse @{ $self->{ WRAPPER } }) {
                    $self->debug("WRAPPER: $name") if $self->{ DEBUG };
                    $procout = $context->process($name, { content => $procout });
                }
            };
            last SERVICE if ($error = $@);
            $output .= $procout;
        }
        
        # POST_PROCESS
        eval {
            foreach $name (@{ $self->{ POST_PROCESS } }) {
                $self->debug("POST_PROCESS: $name") if $self->{ DEBUG };
                $output .= $context->process($name);
            }
        };
        last SERVICE if ($error = $@);
    }

    $context->delocalise();
    delete $params->{ template };

    if ($error) {
    #   $error = $error->as_string if ref $error;
        return $self->error($error);
    }

    return $output;
}


#------------------------------------------------------------------------
# context()
# 
# Returns the internal CONTEXT reference.
#------------------------------------------------------------------------

sub context {
    return $_[0]->{ CONTEXT };
}


#========================================================================
#                     -- PRIVATE METHODS --
#========================================================================

sub _init {
    my ($self, $config) = @_;
    my ($item, $data, $context, $block, $blocks);
    my $delim = $config->{ DELIMITER };
    $delim = ':' unless defined $delim;

    # coerce PRE_PROCESS, PROCESS and POST_PROCESS to arrays if necessary, 
    # by splitting on non-word characters
    foreach $item (qw( PRE_PROCESS PROCESS POST_PROCESS WRAPPER )) {
        $data = $config->{ $item };
        $self->{ $item } = [ ], next unless (defined $data);
        $data = [ split($delim, $data || '') ]
            unless ref $data eq 'ARRAY';
        $self->{ $item } = $data;
    }
    # unset PROCESS option unless explicitly specified in config
    $self->{ PROCESS } = undef
        unless defined $config->{ PROCESS };
    
    $self->{ ERROR      } = $config->{ ERROR } || $config->{ ERRORS };
    $self->{ AUTO_RESET } = defined $config->{ AUTO_RESET }
                            ? $config->{ AUTO_RESET } : 1;
    $self->{ DEBUG      } = ( $config->{ DEBUG } || 0 )
                            & Template::Constants::DEBUG_SERVICE;
    
    $context = $self->{ CONTEXT } = $config->{ CONTEXT }
        || Template::Config->context($config)
        || return $self->error(Template::Config->error);
    
    return $self;
}


#------------------------------------------------------------------------
# _recover(\$exception)
#
# Examines the internal ERROR hash array to find a handler suitable 
# for the exception object passed by reference.  Selecting the handler
# is done by delegation to the exception's select_handler() method, 
# passing the set of handler keys as arguments.  A 'default' handler 
# may also be provided.  The handler value represents the name of a 
# template which should be processed. 
#------------------------------------------------------------------------

sub _recover {
    my ($self, $error) = @_;
    my $context = $self->{ CONTEXT };
    my ($hkey, $handler, $output);

    # there shouldn't ever be a non-exception object received at this
    # point... unless a module like CGI::Carp messes around with the 
    # DIE handler. 
    return undef
        unless blessed($$error) && $$error->isa(EXCEPTION);

    # a 'stop' exception is thrown by [% STOP %] - we return the output
    # buffer stored in the exception object
    return $$error->text()
        if $$error->type() eq 'stop';

    my $handlers = $self->{ ERROR }
        || return undef;                    ## RETURN

    if (ref $handlers eq 'HASH') {
        if ($hkey = $$error->select_handler(keys %$handlers)) {
            $handler = $handlers->{ $hkey };
            $self->debug("using error handler for $hkey") if $self->{ DEBUG };
        }
        elsif ($handler = $handlers->{ default }) {
            # use default handler
            $self->debug("using default error handler") if $self->{ DEBUG };
        }
        else {
            return undef;                   ## RETURN
        }
    }
    else {
        $handler = $handlers;
        $self->debug("using default error handler") if $self->{ DEBUG };
    }
    
    eval { $handler = $context->template($handler) };
    if ($@) {
        $$error = $@;
        return undef;                       ## RETURN
    };
    
    $context->stash->set('error', $$error);
    eval {
        $output .= $context->process($handler);
    };
    if ($@) {
        $$error = $@;
        return undef;                       ## RETURN
    }

    return $output;
}



#------------------------------------------------------------------------
# _dump()
#
# Debug method which return a string representing the internal object
# state. 
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    my $context = $self->{ CONTEXT }->_dump();
    $context =~ s/\n/\n    /gm;

    my $error = $self->{ ERROR };
    $error = join('', 
          "{\n",
          (map { "    $_ => $error->{ $_ }\n" }
           keys %$error),
          "}\n")
    if ref $error;
    
    local $" = ', ';
    return <<EOF;
$self
PRE_PROCESS  => [ @{ $self->{ PRE_PROCESS } } ]
POST_PROCESS => [ @{ $self->{ POST_PROCESS } } ]
ERROR        => $error
CONTEXT      => $context
EOF
}


1;

__END__

=head1 NAME

Template::Service - General purpose template processing service

=head1 SYNOPSIS

    use Template::Service;
    
    my $service = Template::Service->new({
        PRE_PROCESS  => [ 'config', 'header' ],
        POST_PROCESS => 'footer',
        ERROR        => {
            user     => 'user/index.html', 
            dbi      => 'error/database',
            default  => 'error/default',
        },
    });
    
    my $output = $service->process($template_name, \%replace)
        || die $service->error(), "\n";

=head1 DESCRIPTION

The C<Template::Service> module implements an object class for providing
a consistent template processing service. 

Standard header (L<PRE_PROCESS|PRE_PROCESS_POST_PROCESS>) and footer
(L<POST_PROCESS|PRE_PROCESS_POST_PROCESS>) templates may be specified which
are prepended and appended to all templates processed by the service (but not
any other templates or blocks C<INCLUDE>d or C<PROCESS>ed from within). An
L<ERROR> hash may be specified which redirects the service to an alternate
template file in the case of uncaught exceptions being thrown. This allows
errors to be automatically handled by the service and a guaranteed valid
response to be generated regardless of any processing problems encountered.

A default C<Template::Service> object is created by the L<Template> module.
Any C<Template::Service> options may be passed to the L<Template>
L<new()|Template#new()> constructor method and will be forwarded to the
L<Template::Service> constructor.

    use Template;
    
    my $template = Template->new({
        PRE_PROCESS  => 'header',
        POST_PROCESS => 'footer',
    });

Similarly, the C<Template::Service> constructor will forward all configuration
parameters onto other default objects (e.g. L<Template::Context>) that it may
need to instantiate.

A C<Template::Service> object (or subclass) can be explicitly instantiated and
passed to the L<Template> L<new()|Template#new()> constructor method as the
L<SERVICE> item.

    use Template;
    use Template::Service;
    
    my $service = Template::Service->new({
        PRE_PROCESS  => 'header',
        POST_PROCESS => 'footer',
    });
    
    my $template = Template->new({
        SERVICE => $service,
    });

The C<Template::Service> module can be sub-classed to create custom service
handlers.

    use Template;
    use MyOrg::Template::Service;
    
    my $service = MyOrg::Template::Service->new({
        PRE_PROCESS  => 'header',
        POST_PROCESS => 'footer',
        COOL_OPTION  => 'enabled in spades',
    });
    
    my $template = Template->new({
        SERVICE => $service,
    });

The L<Template> module uses the L<Template::Config>
L<service()|Template::Config#service()> factory method to create a default
service object when required. The C<$Template::Config::SERVICE> package
variable may be set to specify an alternate service module. This will be
loaded automatically and its L<new()> constructor method called by the
L<service()|Template::Config#service()> factory method when a default service
object is required. Thus the previous example could be written as:

    use Template;
    
    $Template::Config::SERVICE = 'MyOrg::Template::Service';
    
    my $template = Template->new({
        PRE_PROCESS  => 'header',
        POST_PROCESS => 'footer',
        COOL_OPTION  => 'enabled in spades',
    });

=head1 METHODS

=head2 new(\%config)

The C<new()> constructor method is called to instantiate a C<Template::Service>
object.  Configuration parameters may be specified as a HASH reference or
as a list of C<name =E<gt> value> pairs.

    my $service1 = Template::Service->new({
        PRE_PROCESS  => 'header',
        POST_PROCESS => 'footer',
    });
    
    my $service2 = Template::Service->new( ERROR => 'error.html' );

The C<new()> method returns a C<Template::Service> object or C<undef> on
error. In the latter case, a relevant error message can be retrieved by the
L<error()|Template::Base#error()> class method or directly from the
C<$Template::Service::ERROR> package variable.

    my $service = Template::Service->new(\%config)
        || die Template::Service->error();
        
    my $service = Template::Service->new(\%config)
        || die $Template::Service::ERROR;

=head2 process($input, \%replace)

The C<process()> method is called to process a template specified as the first
parameter, C<$input>. This may be a file name, file handle (e.g. C<GLOB> or
C<IO::Handle>) or a reference to a text string containing the template text. An
additional hash reference may be passed containing template variable
definitions.

The method processes the template, adding any
L<PRE_PROCESS|PRE_PROCESS_POST_PROCESS> or
L<POST_PROCESS|PRE_PROCESS_POST_PROCESS> templates defined, and returns the
output text. An uncaught exception thrown by the template will be handled by a
relevant L<ERROR> handler if defined. Errors that occur in the
L<PRE_PROCESS|PRE_PROCESS_POST_PROCESS> or
L<POST_PROCESS|PRE_PROCESS_POST_PROCESS> templates, or those that occur in the
main input template and aren't handled, cause the method to return C<undef> to
indicate failure. The appropriate error message can be retrieved via the
L<error()|Template::Base#error()> method.

    $service->process('myfile.html', { title => 'My Test File' })
        || die $service->error();

=head2 context()

Returns a reference to the internal context object which is, by default, an
instance of the L<Template::Context> class.

=head1 CONFIGURATION OPTIONS

The following list summarises the configuration options that can be provided
to the C<Template::Service> L<new()> constructor. Please consult
L<Template::Manual::Config> for further details and examples of each
configuration option in use.

=head2 PRE_PROCESS, POST_PROCESS

The L<PRE_PROCESS|Template::Manual::Config#PRE_PROCESS_POST_PROCESS> and
L<POST_PROCESS|Template::Manual::Config#PRE_PROCESS_POST_PROCESS> options may
be set to contain the name(s) of template files which should be processed
immediately before and/or after each template. These do not get added to
templates processed into a document via directives such as C<INCLUDE>
C<PROCESS>, C<WRAPPER>, etc.

    my $service = Template::Service->new({
        PRE_PROCESS  => 'header',
        POST_PROCESS => 'footer',
    };

Multiple templates may be specified as a reference to a list.  Each is 
processed in the order defined.

    my $service = Template::Service->new({
        PRE_PROCESS  => [ 'config', 'header' ],
        POST_PROCESS => 'footer',
    };

=head2 PROCESS

The L<PROCESS|Template::Manual::Config#PROCESS> option may be set to contain
the name(s) of template files which should be processed instead of the main
template passed to the C<Template::Service> L<process()> method. This can be used to
apply consistent wrappers around all templates, similar to the use of
L<PRE_PROCESS|PRE_PROCESS_POST_PROCESS> and 
L<POST_PROCESS|PRE_PROCESS_POST_PROCESS> templates.

    my $service = Template::Service->new({
        PROCESS  => 'content',
    };
    
    # processes 'content' instead of 'foo.html'
    $service->process('foo.html');

A reference to the original template is available in the C<template>
variable.  Metadata items can be inspected and the template can be
processed by specifying it as a variable reference (i.e. prefixed by
'C<$>') to an C<INCLUDE>, C<PROCESS> or C<WRAPPER> directive.

Example C<PROCESS> template:

    <html>
      <head>
        <title>[% template.title %]</title>
      </head>
      <body>
      [% PROCESS $template %]
      </body>
    </html>

=head2 ERROR

The L<ERROR|Template::Manual::Config#ERROR> (or C<ERRORS> if you prefer)
configuration item can be used to name a single template or specify a hash
array mapping exception types to templates which should be used for error
handling. If an uncaught exception is raised from within a template then the
appropriate error template will instead be processed.

If specified as a single value then that template will be processed 
for all uncaught exceptions. 

    my $service = Template::Service->new({
        ERROR => 'error.html'
    });

If the L<ERROR/ERRORS|Template::Manual::Config#ERROR> item is a hash reference
the keys are assumed to be exception types and the relevant template for a
given exception will be selected. A C<default> template may be provided for
the general case.

    my $service = Template::Service->new({
        ERRORS => {
            user     => 'user/index.html',
            dbi      => 'error/database',
            default  => 'error/default',
        },
    });

=head2 AUTO_RESET

The L<AUTO_RESET|Template::Manual::Config#AUTO_RESET> option is set by default
and causes the local C<BLOCKS> cache for the L<Template::Context> object to be
reset on each call to the L<Template> L<process()|Template#process()> method.
This ensures that any C<BLOCK>s defined within a template will only persist until
that template is finished processing. 

=head2 DEBUG

The L<DEBUG|Template::Manual::Config#DEBUG> option can be used to enable
debugging messages from the C<Template::Service> module by setting it to include
the C<DEBUG_SERVICE> value.

    use Template::Constants qw( :debug );
    
    my $template = Template->new({
        DEBUG => DEBUG_SERVICE,
    });

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
