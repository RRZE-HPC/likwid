#================================================================= -*-Perl-*- 
#
# Template::Directive
#
# DESCRIPTION
#   Factory module for constructing templates from Perl code.
#
# AUTHOR
#   Andy Wardley   <abw@wardley.org>
#
# WARNING
#   Much of this module is hairy, even furry in places.  It needs
#   a lot of tidying up and may even be moved into a different place 
#   altogether.  The generator code is often inefficient, particulary in 
#   being very anal about pretty-printing the Perl code all neatly, but 
#   at the moment, that's still high priority for the sake of easier
#   debugging.
#
# COPYRIGHT
#   Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Directive;

use strict;
use warnings;
use base 'Template::Base';
use Template::Constants;
use Template::Exception;

our $VERSION   = 2.20;
our $DEBUG     = 0 unless defined $DEBUG;
our $WHILE_MAX = 1000 unless defined $WHILE_MAX;
our $PRETTY    = 0 unless defined $PRETTY;
our $OUTPUT    = '$output .= ';


sub _init {
    my ($self, $config) = @_;
    $self->{ NAMESPACE } = $config->{ NAMESPACE };
    return $self;
}


sub pad {
    my ($text, $pad) = @_;
    $pad = ' ' x ($pad * 4);
    $text =~ s/^(?!#line)/$pad/gm;
    $text;
}

#========================================================================
# FACTORY METHODS
#
# These methods are called by the parser to construct directive instances.
#========================================================================

#------------------------------------------------------------------------
# template($block)
#------------------------------------------------------------------------

sub template {
    my ($class, $block) = @_;
    $block = pad($block, 2) if $PRETTY;

    return "sub { return '' }" unless $block =~ /\S/;

    return <<EOF;
sub {
    my \$context = shift || die "template sub called without context\\n";
    my \$stash   = \$context->stash;
    my \$output  = '';
    my \$_tt_error;
    
    eval { BLOCK: {
$block
    } };
    if (\$@) {
        \$_tt_error = \$context->catch(\$@, \\\$output);
        die \$_tt_error unless \$_tt_error->type eq 'return';
    }

    return \$output;
}
EOF
}


#------------------------------------------------------------------------
# anon_block($block)                            [% BLOCK %] ... [% END %]
#------------------------------------------------------------------------

sub anon_block {
    my ($class, $block) = @_;
    $block = pad($block, 2) if $PRETTY;

    return <<EOF;

# BLOCK
$OUTPUT do {
    my \$output  = '';
    my \$_tt_error;
    
    eval { BLOCK: {
$block
    } };
    if (\$@) {
        \$_tt_error = \$context->catch(\$@, \\\$output);
        die \$_tt_error unless \$_tt_error->type eq 'return';
    }

    \$output;
};
EOF
}


#------------------------------------------------------------------------
# block($blocktext)
#------------------------------------------------------------------------

sub block {
    my ($class, $block) = @_;
    return join("\n", @{ $block || [] });
}


#------------------------------------------------------------------------
# textblock($text)
#------------------------------------------------------------------------

sub textblock {
    my ($class, $text) = @_;
    return "$OUTPUT " . &text($class, $text) . ';';
}


#------------------------------------------------------------------------
# text($text)
#------------------------------------------------------------------------

sub text {
    my ($class, $text) = @_;
    for ($text) {
        s/(["\$\@\\])/\\$1/g;
        s/\n/\\n/g;
    }
    return '"' . $text . '"';
}


#------------------------------------------------------------------------
# quoted(\@items)                                               "foo$bar"
#------------------------------------------------------------------------

sub quoted {
    my ($class, $items) = @_;
    return '' unless @$items;
    return ("('' . " . $items->[0] . ')') if scalar @$items == 1;
    return '(' . join(' . ', @$items) . ')';
#    my $r = '(' . join(' . ', @$items) . ' . "")';
#    print STDERR "[$r]\n";
#    return $r;
}


#------------------------------------------------------------------------
# ident(\@ident)                                             foo.bar(baz)
#------------------------------------------------------------------------

sub ident {
    my ($class, $ident) = @_;
    return "''" unless @$ident;
    my $ns;

    # does the first element of the identifier have a NAMESPACE
    # handler defined?
    if (ref $class && @$ident > 2 && ($ns = $class->{ NAMESPACE })) {
        my $key = $ident->[0];
        $key =~ s/^'(.+)'$/$1/s;
        if ($ns = $ns->{ $key }) {
            return $ns->ident($ident);
        }
    }
        
    if (scalar @$ident <= 2 && ! $ident->[1]) {
        $ident = $ident->[0];
    }
    else {
        $ident = '[' . join(', ', @$ident) . ']';
    }
    return "\$stash->get($ident)";
}

#------------------------------------------------------------------------
# identref(\@ident)                                         \foo.bar(baz)
#------------------------------------------------------------------------

sub identref {
    my ($class, $ident) = @_;
    return "''" unless @$ident;
    if (scalar @$ident <= 2 && ! $ident->[1]) {
        $ident = $ident->[0];
    }
    else {
        $ident = '[' . join(', ', @$ident) . ']';
    }
    return "\$stash->getref($ident)";
}


#------------------------------------------------------------------------
# assign(\@ident, $value, $default)                             foo = bar
#------------------------------------------------------------------------

sub assign {
    my ($class, $var, $val, $default) = @_;

    if (ref $var) {
        if (scalar @$var == 2 && ! $var->[1]) {
            $var = $var->[0];
        }
        else {
            $var = '[' . join(', ', @$var) . ']';
        }
    }
    $val .= ', 1' if $default;
    return "\$stash->set($var, $val)";
}


#------------------------------------------------------------------------
# args(\@args)                                        foo, bar, baz = qux
#------------------------------------------------------------------------

sub args {
    my ($class, $args) = @_;
    my $hash = shift @$args;
    push(@$args, '{ ' . join(', ', @$hash) . ' }')
        if @$hash;

    return '0' unless @$args;
    return '[ ' . join(', ', @$args) . ' ]';
}

#------------------------------------------------------------------------
# filenames(\@names)
#------------------------------------------------------------------------

sub filenames {
    my ($class, $names) = @_;
    if (@$names > 1) {
        $names = '[ ' . join(', ', @$names) . ' ]';
    }
    else {
        $names = shift @$names;
    }
    return $names;
}


#------------------------------------------------------------------------
# get($expr)                                                    [% foo %]
#------------------------------------------------------------------------

sub get {
    my ($class, $expr) = @_;  
    return "$OUTPUT $expr;";
}


#------------------------------------------------------------------------
# call($expr)                                              [% CALL bar %]
#------------------------------------------------------------------------

sub call {
    my ($class, $expr) = @_;  
    $expr .= ';';
    return $expr;
}


#------------------------------------------------------------------------
# set(\@setlist)                               [% foo = bar, baz = qux %]
#------------------------------------------------------------------------

sub set {
    my ($class, $setlist) = @_;
    my $output;
    while (my ($var, $val) = splice(@$setlist, 0, 2)) {
        $output .= &assign($class, $var, $val) . ";\n";
    }
    chomp $output;
    return $output;
}


#------------------------------------------------------------------------
# default(\@setlist)                   [% DEFAULT foo = bar, baz = qux %]
#------------------------------------------------------------------------

sub default {
    my ($class, $setlist) = @_;  
    my $output;
    while (my ($var, $val) = splice(@$setlist, 0, 2)) {
        $output .= &assign($class, $var, $val, 1) . ";\n";
    }
    chomp $output;
    return $output;
}


#------------------------------------------------------------------------
# insert(\@nameargs)                                    [% INSERT file %] 
#         # => [ [ $file, ... ], \@args ]
#------------------------------------------------------------------------

sub insert {
    my ($class, $nameargs) = @_;
    my ($file, $args) = @$nameargs;
    $file = $class->filenames($file);
    return "$OUTPUT \$context->insert($file);"; 
}


#------------------------------------------------------------------------
# include(\@nameargs)                    [% INCLUDE template foo = bar %] 
#          # => [ [ $file, ... ], \@args ]    
#------------------------------------------------------------------------

sub include {
    my ($class, $nameargs) = @_;
    my ($file, $args) = @$nameargs;
    my $hash = shift @$args;
    $file = $class->filenames($file);
    $file .= @$hash ? ', { ' . join(', ', @$hash) . ' }' : '';
    return "$OUTPUT \$context->include($file);"; 
}


#------------------------------------------------------------------------
# process(\@nameargs)                    [% PROCESS template foo = bar %] 
#         # => [ [ $file, ... ], \@args ]
#------------------------------------------------------------------------

sub process {
    my ($class, $nameargs) = @_;
    my ($file, $args) = @$nameargs;
    my $hash = shift @$args;
    $file = $class->filenames($file);
    $file .= @$hash ? ', { ' . join(', ', @$hash) . ' }' : '';
    return "$OUTPUT \$context->process($file);"; 
}


#------------------------------------------------------------------------
# if($expr, $block, $else)                             [% IF foo < bar %]
#                                                         ...
#                                                      [% ELSE %]
#                                                         ...
#                                                      [% END %]
#------------------------------------------------------------------------

sub if {
    my ($class, $expr, $block, $else) = @_;
    my @else = $else ? @$else : ();
    $else = pop @else;
    $block = pad($block, 1) if $PRETTY;

    my $output = "if ($expr) {\n$block\n}\n";

    foreach my $elsif (@else) {
        ($expr, $block) = @$elsif;
        $block = pad($block, 1) if $PRETTY;
        $output .= "elsif ($expr) {\n$block\n}\n";
    }
    if (defined $else) {
        $else = pad($else, 1) if $PRETTY;
        $output .= "else {\n$else\n}\n";
    }

    return $output;
}


#------------------------------------------------------------------------
# foreach($target, $list, $args, $block)    [% FOREACH x = [ foo bar ] %]
#                                              ...
#                                           [% END %]
#------------------------------------------------------------------------

sub foreach {
    my ($class, $target, $list, $args, $block, $label) = @_;
    $args  = shift @$args;
    $args  = @$args ? ', { ' . join(', ', @$args) . ' }' : '';
    $label ||= 'LOOP';

    my ($loop_save, $loop_set, $loop_restore, $setiter);
    if ($target) {
        $loop_save    = 'eval { $_tt_oldloop = ' . &ident($class, ["'loop'"]) . ' }';
        $loop_set     = "\$stash->{'$target'} = \$_tt_value";
        $loop_restore = "\$stash->set('loop', \$_tt_oldloop)";
    }
    else {
        $loop_save    = '$stash = $context->localise()';
#       $loop_set     = "\$stash->set('import', \$_tt_value) "
#                       . "if ref \$value eq 'HASH'";
        $loop_set     = "\$stash->get(['import', [\$_tt_value]]) "
                        . "if ref \$_tt_value eq 'HASH'";
        $loop_restore = '$stash = $context->delocalise()';
    }
    $block = pad($block, 3) if $PRETTY;

    return <<EOF;

# FOREACH 
do {
    my (\$_tt_value, \$_tt_error, \$_tt_oldloop);
    my \$_tt_list = $list;
    
    unless (UNIVERSAL::isa(\$_tt_list, 'Template::Iterator')) {
        \$_tt_list = Template::Config->iterator(\$_tt_list)
            || die \$Template::Config::ERROR, "\\n"; 
    }

    (\$_tt_value, \$_tt_error) = \$_tt_list->get_first();
    $loop_save;
    \$stash->set('loop', \$_tt_list);
    eval {
$label:   while (! \$_tt_error) {
            $loop_set;
$block;
            (\$_tt_value, \$_tt_error) = \$_tt_list->get_next();
        }
    };
    $loop_restore;
    die \$@ if \$@;
    \$_tt_error = 0 if \$_tt_error && \$_tt_error eq Template::Constants::STATUS_DONE;
    die \$_tt_error if \$_tt_error;
};
EOF
}

#------------------------------------------------------------------------
# next()                                                       [% NEXT %]
#
# Next iteration of a FOREACH loop (experimental)
#------------------------------------------------------------------------

sub next {
    my ($class, $label) = @_;
    $label ||= 'LOOP';
    return <<EOF;
(\$_tt_value, \$_tt_error) = \$_tt_list->get_next();
next $label;
EOF
}


#------------------------------------------------------------------------
# wrapper(\@nameargs, $block)            [% WRAPPER template foo = bar %] 
#          # => [ [$file,...], \@args ]    
#------------------------------------------------------------------------

sub wrapper {
    my ($class, $nameargs, $block) = @_;
    my ($file, $args) = @$nameargs;
    my $hash = shift @$args;

    local $" = ', ';
#    print STDERR "wrapper([@$file], { @$hash })\n";

    return $class->multi_wrapper($file, $hash, $block)
        if @$file > 1;
    $file = shift @$file;

    $block = pad($block, 1) if $PRETTY;
    push(@$hash, "'content'", '$output');
    $file .= @$hash ? ', { ' . join(', ', @$hash) . ' }' : '';

    return <<EOF;

# WRAPPER
$OUTPUT do {
    my \$output = '';
$block
    \$context->include($file); 
};
EOF
}


sub multi_wrapper {
    my ($class, $file, $hash, $block) = @_;
    $block = pad($block, 1) if $PRETTY;

    push(@$hash, "'content'", '$output');
    $hash = @$hash ? ', { ' . join(', ', @$hash) . ' }' : '';

    $file = join(', ', reverse @$file);
#    print STDERR "multi wrapper: $file\n";

    return <<EOF;

# WRAPPER
$OUTPUT do {
    my \$output = '';
$block
    foreach ($file) {
        \$output = \$context->include(\$_$hash); 
    }
    \$output;
};
EOF
}


#------------------------------------------------------------------------
# while($expr, $block)                                 [% WHILE x < 10 %]
#                                                         ...
#                                                      [% END %]
#------------------------------------------------------------------------

sub while {
    my ($class, $expr, $block, $label) = @_;
    $block = pad($block, 2) if $PRETTY;
    $label ||= 'LOOP';

    return <<EOF;

# WHILE
do {
    my \$_tt_failsafe = $WHILE_MAX;
$label:
    while (--\$_tt_failsafe && ($expr)) {
$block
    }
    die "WHILE loop terminated (> $WHILE_MAX iterations)\\n"
        unless \$_tt_failsafe;
};
EOF
}


#------------------------------------------------------------------------
# switch($expr, \@case)                                    [% SWITCH %]
#                                                          [% CASE foo %]
#                                                             ...
#                                                          [% END %]
#------------------------------------------------------------------------

sub switch {
    my ($class, $expr, $case) = @_;
    my @case = @$case;
    my ($match, $block, $default);
    my $caseblock = '';

    $default = pop @case;

    foreach $case (@case) {
        $match = $case->[0];
        $block = $case->[1];
        $block = pad($block, 1) if $PRETTY;
        $caseblock .= <<EOF;
\$_tt_match = $match;
\$_tt_match = [ \$_tt_match ] unless ref \$_tt_match eq 'ARRAY';
if (grep(/^\\Q\$_tt_result\\E\$/, \@\$_tt_match)) {
$block
    last SWITCH;
}
EOF
    }

    $caseblock .= $default
        if defined $default;
    $caseblock = pad($caseblock, 2) if $PRETTY;

return <<EOF;

# SWITCH
do {
    my \$_tt_result = $expr;
    my \$_tt_match;
    SWITCH: {
$caseblock
    }
};
EOF
}


#------------------------------------------------------------------------
# try($block, \@catch)                                        [% TRY %]
#                                                                ...
#                                                             [% CATCH %] 
#                                                                ...
#                                                             [% END %]
#------------------------------------------------------------------------

sub try {
    my ($class, $block, $catch) = @_;
    my @catch = @$catch;
    my ($match, $mblock, $default, $final, $n);
    my $catchblock = '';
    my $handlers = [];

    $block = pad($block, 2) if $PRETTY;
    $final = pop @catch;
    $final = "# FINAL\n" . ($final ? "$final\n" : '')
           . 'die $_tt_error if $_tt_error;' . "\n" . '$output;';
    $final = pad($final, 1) if $PRETTY;

    $n = 0;
    foreach $catch (@catch) {
        $match = $catch->[0] || do {
            $default ||= $catch->[1];
            next;
        };
        $mblock = $catch->[1];
        $mblock = pad($mblock, 1) if $PRETTY;
        push(@$handlers, "'$match'");
        $catchblock .= $n++ 
            ? "elsif (\$_tt_handler eq '$match') {\n$mblock\n}\n" 
               : "if (\$_tt_handler eq '$match') {\n$mblock\n}\n";
    }
    $catchblock .= "\$_tt_error = 0;";
    $catchblock = pad($catchblock, 3) if $PRETTY;
    if ($default) {
        $default = pad($default, 1) if $PRETTY;
        $default = "else {\n    # DEFAULT\n$default\n    \$_tt_error = '';\n}";
    }
    else {
        $default = '# NO DEFAULT';
    }
    $default = pad($default, 2) if $PRETTY;

    $handlers = join(', ', @$handlers);
return <<EOF;

# TRY
$OUTPUT do {
    my \$output = '';
    my (\$_tt_error, \$_tt_handler);
    eval {
$block
    };
    if (\$@) {
        \$_tt_error = \$context->catch(\$@, \\\$output);
        die \$_tt_error if \$_tt_error->type =~ /^return|stop\$/;
        \$stash->set('error', \$_tt_error);
        \$stash->set('e', \$_tt_error);
        if (defined (\$_tt_handler = \$_tt_error->select_handler($handlers))) {
$catchblock
        }
$default
    }
$final
};
EOF
}


#------------------------------------------------------------------------
# throw(\@nameargs)                           [% THROW foo "bar error" %]
#       # => [ [$type], \@args ]
#------------------------------------------------------------------------

sub throw {
    my ($class, $nameargs) = @_;
    my ($type, $args) = @$nameargs;
    my $hash = shift(@$args);
    my $info = shift(@$args);
    $type = shift @$type;           # uses same parser production as INCLUDE
                                    # etc., which allow multiple names
                                    # e.g. INCLUDE foo+bar+baz

    if (! $info) {
        $args = "$type, undef";
    }
    elsif (@$hash || @$args) {
        local $" = ', ';
        my $i = 0;
        $args = "$type, { args => [ " 
              . join(', ', $info, @$args) 
              . ' ], '
              . join(', ', 
                     (map { "'" . $i++ . "' => $_" } ($info, @$args)),
                     @$hash)
              . ' }';
    }
    else {
        $args = "$type, $info";
    }
    
    return "\$context->throw($args, \\\$output);";
}


#------------------------------------------------------------------------
# clear()                                                     [% CLEAR %]
#
# NOTE: this is redundant, being hard-coded (for now) into Parser.yp
#------------------------------------------------------------------------

sub clear {
    return "\$output = '';";
}

#------------------------------------------------------------------------
# break()                                                     [% BREAK %]
#
# NOTE: this is redundant, being hard-coded (for now) into Parser.yp
#------------------------------------------------------------------------

sub OLD_break {
    return 'last LOOP;';
}

#------------------------------------------------------------------------
# return()                                                   [% RETURN %]
#------------------------------------------------------------------------

sub return {
    return "\$context->throw('return', '', \\\$output);";
}

#------------------------------------------------------------------------
# stop()                                                       [% STOP %]
#------------------------------------------------------------------------

sub stop {
    return "\$context->throw('stop', '', \\\$output);";
}


#------------------------------------------------------------------------
# use(\@lnameargs)                         [% USE alias = plugin(args) %]
#     # => [ [$file, ...], \@args, $alias ]
#------------------------------------------------------------------------

sub use {
    my ($class, $lnameargs) = @_;
    my ($file, $args, $alias) = @$lnameargs;
    $file = shift @$file;       # same production rule as INCLUDE
    $alias ||= $file;
    $args = &args($class, $args);
    $file .= ", $args" if $args;
#    my $set = &assign($class, $alias, '$plugin'); 
    return "# USE\n"
         . "\$stash->set($alias,\n"
         . "            \$context->plugin($file));";
}

#------------------------------------------------------------------------
# view(\@nameargs, $block)                           [% VIEW name args %]
#     # => [ [$file, ... ], \@args ]
#------------------------------------------------------------------------

sub view {
    my ($class, $nameargs, $block, $defblocks) = @_;
    my ($name, $args) = @$nameargs;
    my $hash = shift @$args;
    $name = shift @$name;       # same production rule as INCLUDE
    $block = pad($block, 1) if $PRETTY;

    if (%$defblocks) {
        $defblocks = join(",\n", map { "'$_' => $defblocks->{ $_ }" }
                                keys %$defblocks);
        $defblocks = pad($defblocks, 1) if $PRETTY;
        $defblocks = "{\n$defblocks\n}";
        push(@$hash, "'blocks'", $defblocks);
    }
    $hash = @$hash ? '{ ' . join(', ', @$hash) . ' }' : '';

    return <<EOF;
# VIEW
do {
    my \$output = '';
    my \$_tt_oldv = \$stash->get('view');
    my \$_tt_view = \$context->view($hash);
    \$stash->set($name, \$_tt_view);
    \$stash->set('view', \$_tt_view);

$block

    \$stash->set('view', \$_tt_oldv);
    \$_tt_view->seal();
#    \$output;     # not used - commented out to avoid warning
};
EOF
}


#------------------------------------------------------------------------
# perl($block)
#------------------------------------------------------------------------

sub perl {
    my ($class, $block) = @_;
    $block = pad($block, 1) if $PRETTY;

    return <<EOF;

# PERL
\$context->throw('perl', 'EVAL_PERL not set')
    unless \$context->eval_perl();

$OUTPUT do {
    my \$output = "package Template::Perl;\\n";

$block

    local(\$Template::Perl::context) = \$context;
    local(\$Template::Perl::stash)   = \$stash;

    my \$_tt_result = '';
    tie *Template::Perl::PERLOUT, 'Template::TieString', \\\$_tt_result;
    my \$_tt_save_stdout = select *Template::Perl::PERLOUT;

    eval \$output;
    select \$_tt_save_stdout;
    \$context->throw(\$@) if \$@;
    \$_tt_result;
};
EOF
}


#------------------------------------------------------------------------
# no_perl()
#------------------------------------------------------------------------

sub no_perl {
    my $class = shift;
    return "\$context->throw('perl', 'EVAL_PERL not set');";
}


#------------------------------------------------------------------------
# rawperl($block)
#
# NOTE: perhaps test context EVAL_PERL switch at compile time rather than
# runtime?
#------------------------------------------------------------------------

sub rawperl {
    my ($class, $block, $line) = @_;
    for ($block) {
        s/^\n+//;
        s/\n+$//;
    }
    $block = pad($block, 1) if $PRETTY;
    $line = $line ? " (starting line $line)" : '';

    return <<EOF;
# RAWPERL
#line 1 "RAWPERL block$line"
$block
EOF
}



#------------------------------------------------------------------------
# filter()
#------------------------------------------------------------------------

sub filter {
    my ($class, $lnameargs, $block) = @_;
    my ($name, $args, $alias) = @$lnameargs;
    $name = shift @$name;
    $args = &args($class, $args);
    $args = $args ? "$args, $alias" : ", undef, $alias"
        if $alias;
    $name .= ", $args" if $args;
    $block = pad($block, 1) if $PRETTY;
 
    return <<EOF;

# FILTER
$OUTPUT do {
    my \$output = '';
    my \$_tt_filter = \$context->filter($name)
              || \$context->throw(\$context->error);

$block
    
    &\$_tt_filter(\$output);
};
EOF
}


#------------------------------------------------------------------------
# capture($name, $block)
#------------------------------------------------------------------------

sub capture {
    my ($class, $name, $block) = @_;

    if (ref $name) {
        if (scalar @$name == 2 && ! $name->[1]) {
            $name = $name->[0];
        }
        else {
            $name = '[' . join(', ', @$name) . ']';
        }
    }
    $block = pad($block, 1) if $PRETTY;

    return <<EOF;

# CAPTURE
\$stash->set($name, do {
    my \$output = '';
$block
    \$output;
});
EOF

}


#------------------------------------------------------------------------
# macro($name, $block, \@args)
#------------------------------------------------------------------------

sub macro {
    my ($class, $ident, $block, $args) = @_;
    $block = pad($block, 2) if $PRETTY;

    if ($args) {
        my $nargs = scalar @$args;
        $args = join(', ', map { "'$_'" } @$args);
        $args = $nargs > 1 
            ? "\@_tt_args{ $args } = splice(\@_, 0, $nargs)"
            : "\$_tt_args{ $args } = shift";

        return <<EOF;

# MACRO
\$stash->set('$ident', sub {
    my \$output = '';
    my (%_tt_args, \$_tt_params);
    $args;
    \$_tt_params = shift;
    \$_tt_params = { } unless ref(\$_tt_params) eq 'HASH';
    \$_tt_params = { \%_tt_args, %\$_tt_params };

    my \$stash = \$context->localise(\$_tt_params);
    eval {
$block
    };
    \$stash = \$context->delocalise();
    die \$@ if \$@;
    return \$output;
});
EOF

    }
    else {
        return <<EOF;

# MACRO
\$stash->set('$ident', sub {
    my \$_tt_params = \$_[0] if ref(\$_[0]) eq 'HASH';
    my \$output = '';

    my \$stash = \$context->localise(\$_tt_params);
    eval {
$block
    };
    \$stash = \$context->delocalise();
    die \$@ if \$@;
    return \$output;
});
EOF
    }
}


sub debug {
    my ($class, $nameargs) = @_;
    my ($file, $args) = @$nameargs;
    my $hash = shift @$args;
    $args  = join(', ', @$file, @$args);
    $args .= @$hash ? ', { ' . join(', ', @$hash) . ' }' : '';
    return "$OUTPUT \$context->debugging($args); ## DEBUG ##"; 
}


1;

__END__

=head1 NAME

Template::Directive - Perl code generator for template directives

=head1 SYNOPSIS

    # no user serviceable parts inside

=head1 DESCRIPTION

The C<Template::Directive> module defines a number of methods that
generate Perl code for the runtime representation of the various 
Template Toolkit directives.

It is used internally by the L<Template::Parser> module.

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Parser>

=cut

# Local Variables:
# mode: perl
# perl-indent-level: 4
# indent-tabs-mode: nil
# End:
#
# vim: expandtab shiftwidth=4:

