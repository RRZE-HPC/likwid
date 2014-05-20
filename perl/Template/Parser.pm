#============================================================= -*-Perl-*-
#
# Template::Parser
#
# DESCRIPTION
#   This module implements a LALR(1) parser and assocated support 
#   methods to parse template documents into the appropriate "compiled"
#   format.  Much of the parser DFA code (see _parse() method) is based 
#   on Francois Desarmenien's Parse::Yapp module.  Kudos to him.
# 
# AUTHOR
#   Andy Wardley <abw@wardley.org>
#
# COPYRIGHT
#   Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#   The following copyright notice appears in the Parse::Yapp 
#   documentation.  
#
#      The Parse::Yapp module and its related modules and shell
#      scripts are copyright (c) 1998 Francois Desarmenien,
#      France. All rights reserved.
#
#      You may use and distribute them under the terms of either
#      the GNU General Public License or the Artistic License, as
#      specified in the Perl README file.
# 
#============================================================================

package Template::Parser;

use strict;
use warnings;
use base 'Template::Base';

use Template::Constants qw( :status :chomp );
use Template::Directive;
use Template::Grammar;

# parser state constants
use constant CONTINUE => 0;
use constant ACCEPT   => 1;
use constant ERROR    => 2;
use constant ABORT    => 3;

our $VERSION = 2.89;
our $DEBUG   = 0 unless defined $DEBUG;
our $ERROR   = '';


#========================================================================
#                        -- COMMON TAG STYLES --
#========================================================================

our $TAG_STYLE   = {
    'default'   => [ '\[%',    '%\]'    ],
    'template1' => [ '[\[%]%', '%[\]%]' ],
    'metatext'  => [ '%%',     '%%'     ],
    'html'      => [ '<!--',   '-->'    ],
    'mason'     => [ '<%',     '>'      ],
    'asp'       => [ '<%',     '%>'     ],
    'php'       => [ '<\?',    '\?>'    ],
    'star'      => [ '\[\*',   '\*\]'   ],
};
$TAG_STYLE->{ template } = $TAG_STYLE->{ tt2 } = $TAG_STYLE->{ default };


our $DEFAULT_STYLE = {
    START_TAG   => $TAG_STYLE->{ default }->[0],
    END_TAG     => $TAG_STYLE->{ default }->[1],
#    TAG_STYLE   => 'default',
    ANYCASE     => 0,
    INTERPOLATE => 0,
    PRE_CHOMP   => 0,
    POST_CHOMP  => 0,
    V1DOLLAR    => 0,
    EVAL_PERL   => 0,
};

our $QUOTED_ESCAPES = {
        n => "\n",
        r => "\r",
        t => "\t",
};

# note that '-' must come first so Perl doesn't think it denotes a range
our $CHOMP_FLAGS  = qr/[-=~+]/;



#========================================================================
#                      -----  PUBLIC METHODS -----
#========================================================================

#------------------------------------------------------------------------
# new(\%config)
#
# Constructor method. 
#------------------------------------------------------------------------

sub new {
    my $class  = shift;
    my $config = $_[0] && ref($_[0]) eq 'HASH' ? shift(@_) : { @_ };
    my ($tagstyle, $debug, $start, $end, $defaults, $grammar, $hash, $key, $udef);

    my $self = bless { 
        START_TAG   => undef,
        END_TAG     => undef,
        TAG_STYLE   => 'default',
        ANYCASE     => 0,
        INTERPOLATE => 0,
        PRE_CHOMP   => 0,
        POST_CHOMP  => 0,
        V1DOLLAR    => 0,
        EVAL_PERL   => 0,
        FILE_INFO   => 1,
        GRAMMAR     => undef,
        _ERROR      => '',
        IN_BLOCK    => [ ],
        FACTORY     => $config->{ FACTORY } || 'Template::Directive',
    }, $class;

    # update self with any relevant keys in config
    foreach $key (keys %$self) {
        $self->{ $key } = $config->{ $key } if defined $config->{ $key };
    }
    $self->{ FILEINFO } = [ ];
    
    # DEBUG config item can be a bitmask
    if (defined ($debug = $config->{ DEBUG })) {
        $self->{ DEBUG } = $debug & ( Template::Constants::DEBUG_PARSER
                                    | Template::Constants::DEBUG_FLAGS );
        $self->{ DEBUG_DIRS } = $debug & Template::Constants::DEBUG_DIRS;
    }
    # package variable can be set to 1 to support previous behaviour
    elsif ($DEBUG == 1) {
        $self->{ DEBUG } = Template::Constants::DEBUG_PARSER;
        $self->{ DEBUG_DIRS } = 0;
    }
    # otherwise let $DEBUG be a bitmask
    else {
        $self->{ DEBUG } = $DEBUG & ( Template::Constants::DEBUG_PARSER
                                    | Template::Constants::DEBUG_FLAGS );
        $self->{ DEBUG_DIRS } = $DEBUG & Template::Constants::DEBUG_DIRS;
    }

    $grammar = $self->{ GRAMMAR } ||= do {
        require Template::Grammar;
        Template::Grammar->new();
    };

    # build a FACTORY object to include any NAMESPACE definitions,
    # but only if FACTORY isn't already an object
    if ($config->{ NAMESPACE } && ! ref $self->{ FACTORY }) {
        my $fclass = $self->{ FACTORY };
        $self->{ FACTORY } = $fclass->new( NAMESPACE => $config->{ NAMESPACE } )
            || return $class->error($fclass->error());
    }
    
    # load grammar rules, states and lex table
    @$self{ qw( LEXTABLE STATES RULES ) } 
        = @$grammar{ qw( LEXTABLE STATES RULES ) };
    
    $self->new_style($config)
        || return $class->error($self->error());
        
    return $self;
}

#-----------------------------------------------------------------------
# These methods are used to track nested IF and WHILE blocks.  Each 
# generated if/while block is given a label indicating the directive 
# type and nesting depth, e.g. FOR0, WHILE1, FOR2, WHILE3, etc.  The
# NEXT and LAST directives use the innermost label, e.g. last WHILE3;
#-----------------------------------------------------------------------

sub enter_block {
    my ($self, $name) = @_;
    my $blocks = $self->{ IN_BLOCK };
    push(@{ $self->{ IN_BLOCK } }, $name);
}

sub leave_block {
    my $self = shift;
    my $label = $self->block_label;
    pop(@{ $self->{ IN_BLOCK } });
    return $label;
}

sub in_block {
    my ($self, $name) = @_;
    my $blocks = $self->{ IN_BLOCK };
    return @$blocks && $blocks->[-1] eq $name;
}

sub block_label {
    my ($self, $prefix, $suffix) = @_;
    my $blocks = $self->{ IN_BLOCK };
    my $name   = @$blocks 
        ? $blocks->[-1] . scalar @$blocks 
        : undef;
    return join('', grep { defined $_ } $prefix, $name, $suffix);
}



#------------------------------------------------------------------------
# new_style(\%config)
# 
# Install a new (stacked) parser style.  This feature is currently 
# experimental but should mimic the previous behaviour with regard to 
# TAG_STYLE, START_TAG, END_TAG, etc.
#------------------------------------------------------------------------

sub new_style {
    my ($self, $config) = @_;
    my $styles = $self->{ STYLE } ||= [ ];
    my ($tagstyle, $tags, $start, $end, $key);

    # clone new style from previous or default style
    my $style  = { %{ $styles->[-1] || $DEFAULT_STYLE } };

    # expand START_TAG and END_TAG from specified TAG_STYLE
    if ($tagstyle = $config->{ TAG_STYLE }) {
        return $self->error("Invalid tag style: $tagstyle")
            unless defined ($tags = $TAG_STYLE->{ $tagstyle });
        ($start, $end) = @$tags;
        $config->{ START_TAG } ||= $start;
        $config->{   END_TAG } ||= $end;
    }

    foreach $key (keys %$DEFAULT_STYLE) {
        $style->{ $key } = $config->{ $key } if defined $config->{ $key };
    }
    push(@$styles, $style);
    return $style;
}


#------------------------------------------------------------------------
# old_style()
#
# Pop the current parser style and revert to the previous one.  See 
# new_style().   ** experimental **
#------------------------------------------------------------------------

sub old_style {
    my $self = shift;
    my $styles = $self->{ STYLE };
    return $self->error('only 1 parser style remaining')
        unless (@$styles > 1);
    pop @$styles;
    return $styles->[-1];
}


#------------------------------------------------------------------------
# parse($text, $data)
#
# Parses the text string, $text and returns a hash array representing
# the compiled template block(s) as Perl code, in the format expected
# by Template::Document.
#------------------------------------------------------------------------

sub parse {
    my ($self, $text, $info) = @_;
    my ($tokens, $block);

    $info->{ DEBUG } = $self->{ DEBUG_DIRS }
        unless defined $info->{ DEBUG };

#    print "info: { ", join(', ', map { "$_ => $info->{ $_ }" } keys %$info), " }\n";

    # store for blocks defined in the template (see define_block())
    my $defblock = $self->{ DEFBLOCK } = { };
    my $metadata = $self->{ METADATA } = [ ];
    $self->{ DEFBLOCKS } = [ ];

    $self->{ _ERROR } = '';

    # split file into TEXT/DIRECTIVE chunks
    $tokens = $self->split_text($text)
        || return undef;                                    ## RETURN ##

    push(@{ $self->{ FILEINFO } }, $info);

    # parse chunks
    $block = $self->_parse($tokens, $info);

    pop(@{ $self->{ FILEINFO } });

    return undef unless $block;                             ## RETURN ##

    $self->debug("compiled main template document block:\n$block")
        if $self->{ DEBUG } & Template::Constants::DEBUG_PARSER;

    return {
        BLOCK     => $block,
        DEFBLOCKS => $defblock,
        METADATA  => { @$metadata },
    };
}



#------------------------------------------------------------------------
# split_text($text)
#
# Split input template text into directives and raw text chunks.
#------------------------------------------------------------------------

sub split_text {
    my ($self, $text) = @_;
    my ($pre, $dir, $prelines, $dirlines, $postlines, $chomp, $tags, @tags);
    my $style = $self->{ STYLE }->[-1];
    my ($start, $end, $prechomp, $postchomp, $interp ) = 
        @$style{ qw( START_TAG END_TAG PRE_CHOMP POST_CHOMP INTERPOLATE ) };
    my $tags_dir = $self->{ANYCASE} ? qr<TAGS>i : qr<TAGS>;

    my @tokens = ();
    my $line = 1;

    return \@tokens                                         ## RETURN ##
        unless defined $text && length $text;

    # extract all directives from the text
    while ($text =~ s/
           ^(.*?)               # $1 - start of line up to directive
           (?:
            $start          # start of tag
            (.*?)           # $2 - tag contents
            $end            # end of tag
            )
           //sx) {
        
        ($pre, $dir) = ($1, $2);
        $pre = '' unless defined $pre;
        $dir = '' unless defined $dir;
        
        $prelines  = ($pre =~ tr/\n//);  # newlines in preceeding text
        $dirlines  = ($dir =~ tr/\n//);  # newlines in directive tag
        $postlines = 0;                  # newlines chomped after tag
        
        for ($dir) {
            if (/^\#/) {
                # comment out entire directive except for any end chomp flag
                $dir = ($dir =~ /($CHOMP_FLAGS)$/o) ? $1 : '';
            }
            else {
                s/^($CHOMP_FLAGS)?\s*//so;
                # PRE_CHOMP: process whitespace before tag
                $chomp = $1 ? $1 : $prechomp;
                $chomp =~ tr/-=~+/1230/;
                if ($chomp && $pre) {
                    # chomp off whitespace and newline preceding directive
                    if ($chomp == CHOMP_ALL) { 
                        $pre =~ s{ (\r?\n|^) [^\S\n]* \z }{}mx;
                    }
                    elsif ($chomp == CHOMP_COLLAPSE) { 
                        $pre =~ s{ (\s+) \z }{ }x;
                    }
                    elsif ($chomp == CHOMP_GREEDY) { 
                        $pre =~ s{ (\s+) \z }{}x;
                    }
                }
            }
            
            # POST_CHOMP: process whitespace after tag
            s/\s*($CHOMP_FLAGS)?\s*$//so;
            $chomp = $1 ? $1 : $postchomp;
            $chomp =~ tr/-=~+/1230/;
            if ($chomp) {
                if ($chomp == CHOMP_ALL) { 
                    $text =~ s{ ^ ([^\S\n]* \n) }{}x  
                        && $postlines++;
                }
                elsif ($chomp == CHOMP_COLLAPSE) { 
                    $text =~ s{ ^ (\s+) }{ }x  
                        && ($postlines += $1=~y/\n//);
                }
                # any trailing whitespace
                elsif ($chomp == CHOMP_GREEDY) { 
                    $text =~ s{ ^ (\s+) }{}x  
                        && ($postlines += $1=~y/\n//);
                }
            }
        }
            
        # any text preceding the directive can now be added
        if (length $pre) {
            push(@tokens, $interp
                 ? [ $pre, $line, 'ITEXT' ]
                 : ('TEXT', $pre) );
        }
        $line += $prelines;
            
        # and now the directive, along with line number information
        if (length $dir) {
            # the TAGS directive is a compile-time switch
            if ($dir =~ /^$tags_dir\s+(.*)/) {
                my @tags = split(/\s+/, $1);
                if (scalar @tags > 1) {
                    ($start, $end) = map { quotemeta($_) } @tags;
                }
                elsif ($tags = $TAG_STYLE->{ $tags[0] }) {
                    ($start, $end) = @$tags;
                }
                else {
                    warn "invalid TAGS style: $tags[0]\n";
                }
            }
            else {
                # DIRECTIVE is pushed as:
                #   [ $dirtext, $line_no(s), \@tokens ]
                push(@tokens, 
                     [ $dir, 
                       ($dirlines 
                        ? sprintf("%d-%d", $line, $line + $dirlines)
                        : $line),
                       $self->tokenise_directive($dir) ]);
            }
        }
            
        # update line counter to include directive lines and any extra
        # newline chomped off the start of the following text
        $line += $dirlines + $postlines;
    }
        
    # anything remaining in the string is plain text 
    push(@tokens, $interp 
         ? [ $text, $line, 'ITEXT' ]
         : ( 'TEXT', $text) )
        if length $text;
        
    return \@tokens;                                        ## RETURN ##
}
    


#------------------------------------------------------------------------
# interpolate_text($text, $line)
#
# Examines $text looking for any variable references embedded like
# $this or like ${ this }.
#------------------------------------------------------------------------

sub interpolate_text {
    my ($self, $text, $line) = @_;
    my @tokens  = ();
    my ($pre, $var, $dir);


   while ($text =~
           /
           ( (?: \\. | [^\$] ){1,3000} ) # escaped or non-'$' character [$1]
           |
           ( \$ (?:                 # embedded variable            [$2]
             (?: \{ ([^\}]*) \} )   # ${ ... }                     [$3]
             |
             ([\w\.]+)              # $word                        [$4]
             )
           )
        /gx) {

        ($pre, $var, $dir) = ($1, $3 || $4, $2);

        # preceding text
        if (defined($pre) && length($pre)) {
            $line += $pre =~ tr/\n//;
            $pre =~ s/\\\$/\$/g;
            push(@tokens, 'TEXT', $pre);
        }
        # $variable reference
        if ($var) {
            $line += $dir =~ tr/\n/ /;
            push(@tokens, [ $dir, $line, $self->tokenise_directive($var) ]);
        }
        # other '$' reference - treated as text
        elsif ($dir) {
            $line += $dir =~ tr/\n//;
            push(@tokens, 'TEXT', $dir);
        }
    }

    return \@tokens;
}



#------------------------------------------------------------------------
# tokenise_directive($text)
#
# Called by the private _parse() method when it encounters a DIRECTIVE
# token in the list provided by the split_text() or interpolate_text()
# methods.  The directive text is passed by parameter.
#
# The method splits the directive into individual tokens as recognised
# by the parser grammar (see Template::Grammar for details).  It
# constructs a list of tokens each represented by 2 elements, as per
# split_text() et al.  The first element contains the token type, the
# second the token itself.
#
# The method tokenises the string using a complex (but fast) regex.
# For a deeper understanding of the regex magic at work here, see
# Jeffrey Friedl's excellent book "Mastering Regular Expressions",
# from O'Reilly, ISBN 1-56592-257-3
#
# Returns a reference to the list of chunks (each one being 2 elements) 
# identified in the directive text.  On error, the internal _ERROR string 
# is set and undef is returned.
#------------------------------------------------------------------------

sub tokenise_directive {
    my ($self, $text, $line) = @_;
    my ($token, $uctoken, $type, $lookup);
    my $lextable = $self->{ LEXTABLE };
    my $style    = $self->{ STYLE }->[-1];
    my ($anycase, $start, $end) = @$style{ qw( ANYCASE START_TAG END_TAG ) };
    my @tokens = ( );

    while ($text =~ 
            / 
                # strip out any comments
                (\#[^\n]*)
           |
                # a quoted phrase matches in $3
                (["'])                   # $2 - opening quote, ' or "
                (                        # $3 - quoted text buffer
                    (?:                  # repeat group (no backreference)
                        \\\\             # an escaped backslash \\
                    |                    # ...or...
                        \\\2             # an escaped quote \" or \' (match $1)
                    |                    # ...or...
                        .                # any other character
                    |   \n
                    )*?                  # non-greedy repeat
                )                        # end of $3
                \2                       # match opening quote
            |
                # an unquoted number matches in $4
                (-?\d+(?:\.\d+)?)       # numbers
            |
                # filename matches in $5
                ( \/?\w+(?:(?:\/|::?)\w*)+ | \/\w+)
            |
                # an identifier matches in $6
                (\w+)                    # variable identifier
            |   
                # an unquoted word or symbol matches in $7
                (   [(){}\[\]:;,\/\\]    # misc parenthesis and symbols
#               |   \->                  # arrow operator (for future?)
                |   [+\-*]               # math operations
                |   \$\{?                # dollar with option left brace
                |   =>                   # like '='
                |   [=!<>]?= | [!<>]     # eqality tests
                |   &&? | \|\|?          # boolean ops
                |   \.\.?                # n..n sequence
                |   \S+                  # something unquoted
                )                        # end of $7
            /gmxo) {

        # ignore comments to EOL
        next if $1;

        # quoted string
        if (defined ($token = $3)) {
            # double-quoted string may include $variable references
            if ($2 eq '"') {
                if ($token =~ /[\$\\]/) {
                    $type = 'QUOTED';
                    # unescape " and \ but leave \$ escaped so that 
                        # interpolate_text() doesn't incorrectly treat it
                    # as a variable reference
#                   $token =~ s/\\([\\"])/$1/g;
                        for ($token) {
                                s/\\([^\$nrt])/$1/g;
                                s/\\([nrt])/$QUOTED_ESCAPES->{ $1 }/ge;
                        }
                    push(@tokens, ('"') x 2,
                                  @{ $self->interpolate_text($token) },
                                  ('"') x 2);
                    next;
                }
                else {
                    $type = 'LITERAL';
                    $token =~ s['][\\']g;
                    $token = "'$token'";
                }
            } 
            else {
                $type = 'LITERAL';
                $token = "'$token'";
            }
        }
        # number
        elsif (defined ($token = $4)) {
            $type = 'NUMBER';
        }
        elsif (defined($token = $5)) {
            $type = 'FILENAME';
        }
        elsif (defined($token = $6)) {
            # Fold potential keywords to UPPER CASE if the ANYCASE option is
            # set, unless (we've got some preceeding tokens and) the previous
            # token is a DOT op.  This prevents the 'last' in 'data.last'
            # from being interpreted as the LAST keyword.
            $uctoken = 
                ($anycase && (! @tokens || $tokens[-2] ne 'DOT'))
                    ? uc $token
                    :    $token;
            if (defined ($type = $lextable->{ $uctoken })) {
                $token = $uctoken;
            }
            else {
                $type = 'IDENT';
            }
        }
        elsif (defined ($token = $7)) {
            # reserved words may be in lower case unless case sensitive
            $uctoken = $anycase ? uc $token : $token;
            unless (defined ($type = $lextable->{ $uctoken })) {
                $type = 'UNQUOTED';
            }
        }

        push(@tokens, $type, $token);

#       print(STDERR " +[ $type, $token ]\n")
#           if $DEBUG;
    }

#    print STDERR "tokenise directive() returning:\n  [ @tokens ]\n"
#       if $DEBUG;

    return \@tokens;                                        ## RETURN ##
}


#------------------------------------------------------------------------
# define_block($name, $block)
#
# Called by the parser 'defblock' rule when a BLOCK definition is 
# encountered in the template.  The name of the block is passed in the 
# first parameter and a reference to the compiled block is passed in
# the second.  This method stores the block in the $self->{ DEFBLOCK }
# hash which has been initialised by parse() and will later be used 
# by the same method to call the store() method on the calling cache
# to define the block "externally".
#------------------------------------------------------------------------

sub define_block {
    my ($self, $name, $block) = @_;
    my $defblock = $self->{ DEFBLOCK } 
        || return undef;

    $self->debug("compiled block '$name':\n$block")
        if $self->{ DEBUG } & Template::Constants::DEBUG_PARSER;

    $defblock->{ $name } = $block;
    
    return undef;
}

sub push_defblock {
    my $self = shift;
    my $stack = $self->{ DEFBLOCK_STACK } ||= [];
    push(@$stack, $self->{ DEFBLOCK } );
    $self->{ DEFBLOCK } = { };
}

sub pop_defblock {
    my $self  = shift;
    my $defs  = $self->{ DEFBLOCK };
    my $stack = $self->{ DEFBLOCK_STACK } || return $defs;
    return $defs unless @$stack;
    $self->{ DEFBLOCK } = pop @$stack;
    return $defs;
}


#------------------------------------------------------------------------
# add_metadata(\@setlist)
#------------------------------------------------------------------------

sub add_metadata {
    my ($self, $setlist) = @_;
    my $metadata = $self->{ METADATA } 
        || return undef;

    push(@$metadata, @$setlist);
    
    return undef;
}


#------------------------------------------------------------------------
# location()
#
# Return Perl comment indicating current parser file and line
#------------------------------------------------------------------------

sub location {
    my $self = shift;
    return "\n" unless $self->{ FILE_INFO };
    my $line = ${ $self->{ LINE } };
    my $info = $self->{ FILEINFO }->[-1];
    my $file = $info->{ path } || $info->{ name } 
        || '(unknown template)';
    $line =~ s/\-.*$//; # might be 'n-n'
    $line ||= 1;
    return "#line $line \"$file\"\n";
}


#========================================================================
#                     -----  PRIVATE METHODS -----
#========================================================================

#------------------------------------------------------------------------
# _parse(\@tokens, \@info)
#
# Parses the list of input tokens passed by reference and returns a 
# Template::Directive::Block object which contains the compiled 
# representation of the template. 
#
# This is the main parser DFA loop.  See embedded comments for 
# further details.
#
# On error, undef is returned and the internal _ERROR field is set to 
# indicate the error.  This can be retrieved by calling the error() 
# method.
#------------------------------------------------------------------------

sub _parse {
    my ($self, $tokens, $info) = @_;
    my ($token, $value, $text, $line, $inperl);
    my ($state, $stateno, $status, $action, $lookup, $coderet, @codevars);
    my ($lhs, $len, $code);         # rule contents
    my $stack = [ [ 0, undef ] ];   # DFA stack

# DEBUG
#   local $" = ', ';

    # retrieve internal rule and state tables
    my ($states, $rules) = @$self{ qw( STATES RULES ) };

    # call the grammar set_factory method to install emitter factory
    $self->{ GRAMMAR }->install_factory($self->{ FACTORY });

    $line = $inperl = 0;
    $self->{ LINE   } = \$line;
    $self->{ FILE   } = $info->{ name };
    $self->{ INPERL } = \$inperl;

    $status = CONTINUE;
    my $in_string = 0;

    while(1) {
        # get state number and state
        $stateno =  $stack->[-1]->[0];
        $state   = $states->[$stateno];

        # see if any lookaheads exist for the current state
        if (exists $state->{'ACTIONS'}) {

            # get next token and expand any directives (i.e. token is an 
            # array ref) onto the front of the token list
            while (! defined $token && @$tokens) {
                $token = shift(@$tokens);
                if (ref $token) {
                    ($text, $line, $token) = @$token;
                    if (ref $token) {
                        if ($info->{ DEBUG } && ! $in_string) {
                            # - - - - - - - - - - - - - - - - - - - - - - - - -
                            # This is gnarly.  Look away now if you're easily
                            # frightened.  We're pushing parse tokens onto the
                            # pending list to simulate a DEBUG directive like so:
                            # [% DEBUG msg line='20' text='INCLUDE foo' %]
                            # - - - - - - - - - - - - - - - - - - - - - - - - -
                            my $dtext = $text;
                            $dtext =~ s[(['\\])][\\$1]g;
                            unshift(@$tokens, 
                                    DEBUG   => 'DEBUG',
                                    IDENT   => 'msg',
                                    IDENT   => 'line',
                                    ASSIGN  => '=',
                                    LITERAL => "'$line'",
                                    IDENT   => 'text',
                                    ASSIGN  => '=',
                                    LITERAL => "'$dtext'",
                                    IDENT   => 'file',
                                    ASSIGN  => '=',
                                    LITERAL => "'$info->{ name }'",
                                    (';') x 2,
                                    @$token, 
                                    (';') x 2);
                        }
                        else {
                            unshift(@$tokens, @$token, (';') x 2);
                        }
                        $token = undef;  # force redo
                    }
                    elsif ($token eq 'ITEXT') {
                        if ($inperl) {
                            # don't perform interpolation in PERL blocks
                            $token = 'TEXT';
                            $value = $text;
                        }
                        else {
                            unshift(@$tokens, 
                                    @{ $self->interpolate_text($text, $line) });
                            $token = undef; # force redo
                        }
                    }
                }
                else {
                    # toggle string flag to indicate if we're crossing
                    # a string boundary
                    $in_string = ! $in_string if $token eq '"';
                    $value = shift(@$tokens);
                }
            };
            # clear undefined token to avoid 'undefined variable blah blah'
            # warnings and let the parser logic pick it up in a minute
            $token = '' unless defined $token;

            # get the next state for the current lookahead token
            $action = defined ($lookup = $state->{'ACTIONS'}->{ $token })
                      ? $lookup
                      : defined ($lookup = $state->{'DEFAULT'})
                        ? $lookup
                        : undef;
        }
        else {
            # no lookahead actions
            $action = $state->{'DEFAULT'};
        }

        # ERROR: no ACTION
        last unless defined $action;

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # shift (+ive ACTION)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if ($action > 0) {
            push(@$stack, [ $action, $value ]);
            $token = $value = undef;
            redo;
        };

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # reduce (-ive ACTION)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ($lhs, $len, $code) = @{ $rules->[ -$action ] };

        # no action imples ACCEPTance
        $action
            or $status = ACCEPT;

        # use dummy sub if code ref doesn't exist
        $code = sub { $_[1] }
            unless $code;

        @codevars = $len
                ?   map { $_->[1] } @$stack[ -$len .. -1 ]
                :   ();

        eval {
            $coderet = &$code( $self, @codevars );
        };
        if ($@) {
            my $err = $@;
            chomp $err;
            return $self->_parse_error($err);
        }

        # reduce stack by $len
        splice(@$stack, -$len, $len);

        # ACCEPT
        return $coderet                                     ## RETURN ##
            if $status == ACCEPT;

        # ABORT
        return undef                                        ## RETURN ##
            if $status == ABORT;

        # ERROR
        last 
            if $status == ERROR;
    }
    continue {
        push(@$stack, [ $states->[ $stack->[-1][0] ]->{'GOTOS'}->{ $lhs }, 
              $coderet ]), 
    }

    # ERROR                                                 ## RETURN ##
    return $self->_parse_error('unexpected end of input')
        unless defined $value;

    # munge text of last directive to make it readable
#    $text =~ s/\n/\\n/g;

    return $self->_parse_error("unexpected end of directive", $text)
        if $value eq ';';   # end of directive SEPARATOR

    return $self->_parse_error("unexpected token ($value)", $text);
}



#------------------------------------------------------------------------
# _parse_error($msg, $dirtext)
#
# Method used to handle errors encountered during the parse process
# in the _parse() method.  
#------------------------------------------------------------------------

sub _parse_error {
    my ($self, $msg, $text) = @_;
    my $line = $self->{ LINE };
    $line = ref($line) ? $$line : $line;
    $line = 'unknown' unless $line;

    $msg .= "\n  [% $text %]"
        if defined $text;

    return $self->error("line $line: $msg");
}


#------------------------------------------------------------------------
# _dump()
# 
# Debug method returns a string representing the internal state of the 
# object.
#------------------------------------------------------------------------

sub _dump {
    my $self = shift;
    my $output = "[Template::Parser] {\n";
    my $format = "    %-16s => %s\n";
    my $key;

    foreach $key (qw( START_TAG END_TAG TAG_STYLE ANYCASE INTERPOLATE 
                      PRE_CHOMP POST_CHOMP V1DOLLAR )) {
        my $val = $self->{ $key };
        $val = '<undef>' unless defined $val;
        $output .= sprintf($format, $key, $val);
    }

    $output .= '}';
    return $output;
}


1;

__END__

=head1 NAME

Template::Parser - LALR(1) parser for compiling template documents

=head1 SYNOPSIS

    use Template::Parser;
    
    $parser   = Template::Parser->new(\%config);
    $template = $parser->parse($text)
        || die $parser->error(), "\n";

=head1 DESCRIPTION

The C<Template::Parser> module implements a LALR(1) parser and associated
methods for parsing template documents into Perl code.

=head1 PUBLIC METHODS

=head2 new(\%params)

The C<new()> constructor creates and returns a reference to a new 
C<Template::Parser> object.  

A reference to a hash may be supplied as a parameter to provide configuration values.  
See L<CONFIGURATION OPTIONS> below for a summary of these options and 
L<Template::Manual::Config> for full details.

    my $parser = Template::Parser->new({
        START_TAG => quotemeta('<+'),
        END_TAG   => quotemeta('+>'),
    });

=head2 parse($text)

The C<parse()> method parses the text passed in the first parameter and
returns a reference to a hash array of data defining the compiled
representation of the template text, suitable for passing to the
L<Template::Document> L<new()|Template::Document#new()> constructor method. On
error, undef is returned.

    $data = $parser->parse($text)
        || die $parser->error();

The C<$data> hash reference returned contains a C<BLOCK> item containing the
compiled Perl code for the template, a C<DEFBLOCKS> item containing a
reference to a hash array of sub-template C<BLOCK>s defined within in the
template, and a C<METADATA> item containing a reference to a hash array
of metadata values defined in C<META> tags.

=head1 CONFIGURATION OPTIONS

The C<Template::Parser> module accepts the following configuration 
options.  Please see L<Template::Manual::Config> for futher details
on each option.

=head2 START_TAG, END_TAG

The L<START_TAG|Template::Manual::Config#START_TAG_END_TAG> and
L<END_TAG|Template::Manual::Config#START_TAG_END_TAG> options are used to
specify character sequences or regular expressions that mark the start and end
of a template directive.

    my $parser = Template::Parser->new({ 
        START_TAG => quotemeta('<+'),
        END_TAG   => quotemeta('+>'),
    });

=head2 TAG_STYLE

The L<TAG_STYLE|Template::Manual::Config#TAG_STYLE> option can be used to set
both L<START_TAG> and L<END_TAG> according to pre-defined tag styles.

    my $parser = Template::Parser->new({ 
        TAG_STYLE => 'star',     # [* ... *]
    });

=head2 PRE_CHOMP, POST_CHOMP

The L<PRE_CHOMP|Template::Manual::Config#PRE_CHOMP_POST_CHOMP> and
L<POST_CHOMP|Template::Manual::Config#PRE_CHOMP_POST_CHOMP> can be set to remove
any whitespace before or after a directive tag, respectively.

    my $parser = Template::Parser-E<gt>new({
        PRE_CHOMP  => 1,
        POST_CHOMP => 1,
    });

=head2 INTERPOLATE

The L<INTERPOLATE|Template::Manual::Config#INTERPOLATE> flag can be set
to allow variables to be embedded in plain text blocks.

    my $parser = Template::Parser->new({ 
        INTERPOLATE => 1,
    });

Variables should be prefixed by a C<$> to identify them, using curly braces
to explicitly scope the variable name where necessary.

    Hello ${name},
    
    The day today is ${day.today}.

=head2 ANYCASE

The L<ANYCASE|Template::Manual::Config#ANYCASE> option can be set
to allow directive keywords to be specified in any case.

    # with ANYCASE set to 1
    [% INCLUDE foobar %]    # OK
    [% include foobar %]    # OK
    [% include = 10   %]    # ERROR, 'include' is a reserved word

=head2 GRAMMAR

The L<GRAMMAR|Template::Manual::Config#GRAMMAR> configuration item can be used
to specify an alternate grammar for the parser. This allows a modified or
entirely new template language to be constructed and used by the Template
Toolkit.

    use MyOrg::Template::Grammar;
    
    my $parser = Template::Parser->new({ 
        GRAMMAR = MyOrg::Template::Grammar->new();
    });

By default, an instance of the default L<Template::Grammar> will be
created and used automatically if a C<GRAMMAR> item isn't specified.

=head2 DEBUG

The L<DEBUG|Template::Manual::Config#DEBUG> option can be used to enable
various debugging features of the C<Template::Parser> module.

    use Template::Constants qw( :debug );
    
    my $template = Template->new({
        DEBUG => DEBUG_PARSER | DEBUG_DIRS,
    });

=head1 AUTHOR

Andy Wardley E<lt>abw@wardley.orgE<gt> L<http://wardley.org/>

=head1 COPYRIGHT

Copyright (C) 1996-2007 Andy Wardley.  All Rights Reserved.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

The main parsing loop of the C<Template::Parser> module was derived from a
standalone parser generated by version 0.16 of the C<Parse::Yapp> module. The
following copyright notice appears in the C<Parse::Yapp> documentation.

    The Parse::Yapp module and its related modules and shell
    scripts are copyright (c) 1998 Francois Desarmenien,
    France. All rights reserved.
    
    You may use and distribute them under the terms of either
    the GNU General Public License or the Artistic License, as
    specified in the Perl README file.

=head1 SEE ALSO

L<Template>, L<Template::Grammar>, L<Template::Directive>

