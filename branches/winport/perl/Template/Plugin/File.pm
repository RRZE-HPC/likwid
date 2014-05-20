#============================================================= -*-Perl-*-
#
# Template::Plugin::File
#
# DESCRIPTION
#  Plugin for encapsulating information about a system file.
#
# AUTHOR
#   Originally written by Michael Stevens <michael@etla.org> as the
#   Directory plugin, then mutilated by Andy Wardley <abw@kfs.org> 
#   into separate File and Directory plugins, with some additional 
#   code for working with views, etc.
#
# COPYRIGHT
#   Copyright 2000-2007 Michael Stevens, Andy Wardley.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin::File;

use strict;
use warnings;
use Cwd;
use File::Spec;
use File::Basename;
use base 'Template::Plugin';

our $VERSION = 2.71;

our @STAT_KEYS = qw( dev ino mode nlink uid gid rdev size 
                     atime mtime ctime blksize blocks );


#------------------------------------------------------------------------
# new($context, $file, \%config)
#
# Create a new File object.  Takes the pathname of the file as
# the argument following the context and an optional 
# hash reference of configuration parameters.
#------------------------------------------------------------------------

sub new {
    my $config = ref($_[-1]) eq 'HASH' ? pop(@_) : { };
    my ($class, $context, $path) = @_;
    my ($root, $home, @stat, $abs);

    return $class->throw('no file specified')
        unless defined $path and length $path;

    # path, dir, name, root, home

    if (File::Spec->file_name_is_absolute($path)) {
        $root = '';
    }
    elsif (($root = $config->{ root })) {
        # strip any trailing '/' from root
        $root =~ s[/$][];
    }
    else {
        $root = '';
    }

    my ($name, $dir, $ext) = fileparse($path, '\.\w+');
    # fixup various items
    $dir  =~ s[/$][];
    $dir  = '' if $dir eq '.';
    $name = $name . $ext;
    $ext  =~ s/^\.//g;

    my @fields = File::Spec->splitdir($dir);
    shift @fields if @fields && ! length $fields[0];
    $home = join('/', ('..') x @fields);
    $abs = File::Spec->catfile($root ? $root : (), $path);

    my $self = { 
        path  => $path,
        name  => $name,
        root  => $root,
        home  => $home,
        dir   => $dir,
        ext   => $ext,
        abs   => $abs,
        user  => '',
        group => '',
        isdir => '',
        stat  => defined $config->{ stat } 
                       ? $config->{ stat } 
                       : ! $config->{ nostat },
        map { ($_ => '') } @STAT_KEYS,
    };

    if ($self->{ stat }) {
        (@stat = stat( $abs ))
            || return $class->throw("$abs: $!");

        @$self{ @STAT_KEYS } = @stat;

        unless ($config->{ noid }) {
            $self->{ user  } = eval { getpwuid( $self->{ uid }) || $self->{ uid } };
            $self->{ group } = eval { getgrgid( $self->{ gid }) || $self->{ gid } };
        }
        $self->{ isdir } = -d $abs;
    }

    bless $self, $class;
}


#-------------------------------------------------------------------------
# rel($file)
#
# Generate a relative filename for some other file relative to this one.
#------------------------------------------------------------------------

sub rel {
    my ($self, $path) = @_;
    $path = $path->{ path } if ref $path eq ref $self;  # assumes same root
    return $path if $path =~ m[^/];
    return $path unless $self->{ home };
    return $self->{ home } . '/' . $path;
}


#------------------------------------------------------------------------
# present($view)
#
# Present self to a Template::View.
#------------------------------------------------------------------------

sub present {
    my ($self, $view) = @_;
    $view->view_file($self);
}


sub throw {
    my ($self, $error) = @_;
    die (Template::Exception->new('File', $error));
}

1;

__END__

=head1 NAME

Template::Plugin::File - Plugin providing information about files

=head1 SYNOPSIS

    [% USE File(filepath) %]
    [% File.path %]         # full path
    [% File.name %]         # filename
    [% File.dir %]          # directory

=head1 DESCRIPTION

This plugin provides an abstraction of a file.  It can be used to 
fetch details about files from the file system, or to represent abstract
files (e.g. when creating an index page) that may or may not exist on 
a file system.

A file name or path should be specified as a constructor argument.  e.g.

    [% USE File('foo.html') %]
    [% USE File('foo/bar/baz.html') %]
    [% USE File('/foo/bar/baz.html') %]

The file should exist on the current file system (unless C<nostat>
option set, see below) as an absolute file when specified with as
leading 'C</>' as per 'C</foo/bar/baz.html>', or otherwise as one relative
to the current working directory.  The constructor performs a C<stat()>
on the file and makes the 13 elements returned available as the plugin
items:

    dev ino mode nlink uid gid rdev size 
    atime mtime ctime blksize blocks

e.g.

    [% USE File('/foo/bar/baz.html') %]
    
    [% File.mtime %]
    [% File.mode %]
    ...

In addition, the C<user> and C<group> items are set to contain the user
and group names as returned by calls to C<getpwuid()> and C<getgrgid()> for
the file C<uid> and C<gid> elements, respectively.  On Win32 platforms
on which C<getpwuid()> and C<getgrid()> are not available, these values are
undefined.

    [% USE File('/tmp/foo.html') %]
    [% File.uid %]      # e.g. 500
    [% File.user %]     # e.g. abw

This user/group lookup can be disabled by setting the C<noid> option.

    [% USE File('/tmp/foo.html', noid=1) %]
    [% File.uid %]      # e.g. 500
    [% File.user %]     # nothing

The C<isdir> flag will be set if the file is a directory.

    [% USE File('/tmp') %]
    [% File.isdir %]    # 1

If the C<stat()> on the file fails (e.g. file doesn't exists, bad
permission, etc) then the constructor will throw a C<File> exception.
This can be caught within a C<TRY...CATCH> block.

    [% TRY %]
       [% USE File('/tmp/myfile') %]
       File exists!
    [% CATCH File %]
       File error: [% error.info %]
    [% END %]

Note the capitalisation of the exception type, 'C<File>', to indicate an
error thrown by the C<File> plugin, to distinguish it from a regular
C<file> exception thrown by the Template Toolkit.

Note that the C<File> plugin can also be referenced by the lower case
name 'C<file>'.  However, exceptions are always thrown of the C<File>
type, regardless of the capitalisation of the plugin named used.

    [% USE file('foo.html') %]
    [% file.mtime %]

As with any other Template Toolkit plugin, an alternate name can be 
specified for the object created.

    [% USE foo = file('foo.html') %]
    [% foo.mtime %]

The C<nostat> option can be specified to prevent the plugin constructor
from performing a C<stat()> on the file specified.  In this case, the
file does not have to exist in the file system, no attempt will be made
to verify that it does, and no error will be thrown if it doesn't.
The entries for the items usually returned by C<stat()> will be set 
empty.

    [% USE file('/some/where/over/the/rainbow.html', nostat=1) 
    [% file.mtime %]     # nothing

=head1 METHODS

All C<File> plugins, regardless of the C<nostat> option, have set a number
of items relating to the original path specified.

=head2 path

The full, original file path specified to the constructor.

    [% USE file('/foo/bar.html') %]
    [% file.path %]     # /foo/bar.html

=head2 name

The name of the file without any leading directories.

    [% USE file('/foo/bar.html') %]
    [% file.name %]     # bar.html

=head2 dir

The directory element of the path with the filename removed.

    [% USE file('/foo/bar.html') %]
    [% file.name %]     # /foo

=head2 ext

The file extension, if any, appearing at the end of the path following 
a 'C<.>' (not included in the extension).

    [% USE file('/foo/bar.html') %]
    [% file.ext %]      # html

=head2 home

This contains a string of the form 'C<../..>' to represent the upward path
from a file to its root directory.

    [% USE file('bar.html') %]
    [% file.home %]     # nothing
    
    [% USE file('foo/bar.html') %]
    [% file.home %]     # ..
    
    [% USE file('foo/bar/baz.html') %]
    [% file.home %]     # ../..

=head2 root

The C<root> item can be specified as a constructor argument, indicating
a root directory in which the named file resides.  This is otherwise
set empty.

    [% USE file('foo/bar.html', root='/tmp') %]
    [% file.root %]     # /tmp

=head2 abs

This returns the absolute file path by constructing a path from the 
C<root> and C<path> options.

    [% USE file('foo/bar.html', root='/tmp') %]
    [% file.path %]     # foo/bar.html
    [% file.root %]     # /tmp
    [% file.abs %]      # /tmp/foo/bar.html

=head2 rel(path)

This returns a relative path from the current file to another path specified
as an argument.  It is constructed by appending the path to the 'C<home>' 
item.

    [% USE file('foo/bar/baz.html') %]
    [% file.rel('wiz/waz.html') %]      # ../../wiz/waz.html

=head1 EXAMPLES

    [% USE file('/foo/bar/baz.html') %]
    
    [% file.path  %]      # /foo/bar/baz.html
    [% file.dir   %]      # /foo/bar
    [% file.name  %]      # baz.html
    [% file.home  %]      # ../..
    [% file.root  %]      # ''
    [% file.abs   %]      # /foo/bar/baz.html
    [% file.ext   %]      # html
    [% file.mtime %]      # 987654321
    [% file.atime %]      # 987654321
    [% file.uid   %]      # 500
    [% file.user  %]      # abw

    [% USE file('foo.html') %]
    
    [% file.path %]           # foo.html
    [% file.dir  %]       # ''
    [% file.name %]           # foo.html
    [% file.root %]       # ''
    [% file.home %]       # ''
    [% file.abs  %]       # foo.html

    [% USE file('foo/bar/baz.html') %]
    
    [% file.path %]           # foo/bar/baz.html
    [% file.dir  %]       # foo/bar
    [% file.name %]           # baz.html
    [% file.root %]       # ''
    [% file.home %]       # ../..
    [% file.abs  %]       # foo/bar/baz.html

    [% USE file('foo/bar/baz.html', root='/tmp') %]
    
    [% file.path %]           # foo/bar/baz.html
    [% file.dir  %]       # foo/bar
    [% file.name %]           # baz.html
    [% file.root %]       # /tmp
    [% file.home %]       # ../..
    [% file.abs  %]       # /tmp/foo/bar/baz.html

    # calculate other file paths relative to this file and its root
    [% USE file('foo/bar/baz.html', root => '/tmp/tt2') %]
    
    [% file.path('baz/qux.html') %]         # ../../baz/qux.html
    [% file.dir('wiz/woz.html')  %]     # ../../wiz/woz.html

=head1 AUTHORS

Michael Stevens wrote the original C<Directory> plugin on which this is based.
Andy Wardley split it into separate C<File> and C<Directory> plugins, added
some extra code and documentation for C<VIEW> support, and made a few other
minor tweaks.

=head1 COPYRIGHT

Copyright 2000-2007 Michael Stevens, Andy Wardley.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<Template::Plugin::Directory>, L<Template::View>

