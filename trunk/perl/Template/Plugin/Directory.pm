#============================================================= -*-Perl-*-
#
# Template::Plugin::Directory
#
# DESCRIPTION
#   Plugin for encapsulating information about a file system directory.
#
# AUTHORS
#   Michael Stevens <michael@etla.org>, with some mutilations from 
#   Andy Wardley <abw@wardley.org>.
#
# COPYRIGHT
#   Copyright (C) 2000-2007 Michael Stevens, Andy Wardley.
#
#   This module is free software; you can redistribute it and/or
#   modify it under the same terms as Perl itself.
#
#============================================================================

package Template::Plugin::Directory;

use strict;
use warnings;
use Cwd;
use File::Spec;
use Template::Plugin::File;
use base 'Template::Plugin::File';

our $VERSION = 2.70;


#------------------------------------------------------------------------
# new(\%config)
#
# Constructor method.
#------------------------------------------------------------------------

sub new {
    my $config = ref($_[-1]) eq 'HASH' ? pop(@_) : { };
    my ($class, $context, $path) = @_;

    return $class->throw('no directory specified')
        unless defined $path and length $path;

    my $self = $class->SUPER::new($context, $path, $config);
    my ($dir, @files, $name, $item, $abs, $rel, $check);
    $self->{ files } = [ ];
    $self->{ dirs  } = [ ];
    $self->{ list  } = [ ];
    $self->{ _dir  } = { };

    # don't read directory if 'nostat' or 'noscan' set
    return $self if $config->{ nostat } || $config->{ noscan };

    $self->throw("$path: not a directory")
        unless $self->{ isdir };

    $self->scan($config);

    return $self;
}


#------------------------------------------------------------------------
# scan(\%config)
#
# Scan directory for files and sub-directories.
#------------------------------------------------------------------------

sub scan {
    my ($self, $config) = @_;
    $config ||= { };
    local *DH;
    my ($dir, @files, $name, $abs, $rel, $item);
    
    # set 'noscan' in config if recurse isn't set, to ensure Directories
    # created don't try to scan deeper
    $config->{ noscan } = 1 unless $config->{ recurse };

    $dir = $self->{ abs };
    opendir(DH, $dir) or return $self->throw("$dir: $!");

    @files = readdir DH;
    closedir(DH) 
        or return $self->throw("$dir close: $!");

    my ($path, $files, $dirs, $list) = @$self{ qw( path files dirs list ) };
    @$files = @$dirs = @$list = ();

    foreach $name (sort @files) {
        next if $name =~ /^\./;
        $abs = File::Spec->catfile($dir, $name);
        $rel = File::Spec->catfile($path, $name);

        if (-d $abs) {
            $item = Template::Plugin::Directory->new(undef, $rel, $config);
            push(@$dirs, $item);
        }
        else {
            $item = Template::Plugin::File->new(undef, $rel, $config);
            push(@$files, $item);
        }
        push(@$list, $item);
        $self->{ _dir }->{ $name } = $item;
    }

    return '';
}


#------------------------------------------------------------------------
# file($filename)
#
# Fetch a named file from this directory.
#------------------------------------------------------------------------

sub file {
    my ($self, $name) = @_;
    return $self->{ _dir }->{ $name };
}


#------------------------------------------------------------------------
# present($view)
#
# Present self to a Template::View
#------------------------------------------------------------------------

sub present {
    my ($self, $view) = @_;
    $view->view_directory($self);
}


#------------------------------------------------------------------------
# content($view)
# 
# Present directory content to a Template::View.
#------------------------------------------------------------------------

sub content {
    my ($self, $view) = @_;
    return $self->{ list } unless $view;
    my $output = '';
    foreach my $file (@{ $self->{ list } }) {
        $output .= $file->present($view);
    }
    return $output;
}


#------------------------------------------------------------------------
# throw($msg)
#
# Throw a 'Directory' exception.
#------------------------------------------------------------------------

sub throw {
    my ($self, $error) = @_;
    die (Template::Exception->new('Directory', $error));
}

1;

__END__

=head1 NAME

Template::Plugin::Directory - Plugin for generating directory listings

=head1 SYNOPSIS

    [% USE dir = Directory(dirpath) %]
    
    # files returns list of regular files
    [% FOREACH file = dir.files %]
       [% file.name %] [% file.path %] ...
    [% END %]
    
    # dirs returns list of sub-directories
    [% FOREACH subdir = dir.dirs %]
       [% subdir.name %] [% subdir.path %] ...
    [% END %]
    
    # list returns both interleaved in order
    [% FOREACH item = dir.list %]
       [% IF item.isdir %]
          Directory: [% item.name %]
       [% ELSE %]
          File: [% item.name %]
       [% END %]
    [% END %]
    
    # define a VIEW to display dirs/files
    [% VIEW myview %]
       [% BLOCK file %]
       File: [% item.name %]
       [% END %]
       
       [% BLOCK directory %]
       Directory: [% item.name %] 
       [% item.content(myview) | indent -%]
       [% END %]
    [% END %]
    
    # display directory content using view
    [% myview.print(dir) %]

=head1 DESCRIPTION

This Template Toolkit plugin provides a simple interface to directory
listings.  It is derived from the L<Template::Plugin::File> module and
uses L<Template::Plugin::File> object instances to represent files within
a directory.  Sub-directories within a directory are represented by
further C<Template::Plugin::Directory> instances.

The constructor expects a directory name as an argument.

    [% USE dir = Directory('/tmp') %]

It then provides access to the files and sub-directories contained within 
the directory.

    # regular files (not directories)
    [% FOREACH file IN dir.files %]
       [% file.name %]
    [% END %]

    # directories only
    [% FOREACH file IN dir.dirs %]
       [% file.name %]
    [% END %]

    # files and/or directories
    [% FOREACH file IN dir.list %]
       [% file.name %] ([% file.isdir ? 'directory' : 'file' %])
    [% END %]

The plugin constructor will throw a C<Directory> error if the specified
path does not exist, is not a directory or fails to C<stat()> (see
L<Template::Plugin::File>).  Otherwise, it will scan the directory and
create lists named 'C<files>' containing files, 'C<dirs>' containing
directories and 'C<list>' containing both files and directories combined.
The C<nostat> option can be set to disable all file/directory checks
and directory scanning.

Each file in the directory will be represented by a
L<Template::Plugin::File> object instance, and each directory by another
C<Template::Plugin::Directory>.  If the C<recurse> flag is set, then those
directories will contain further nested entries, and so on.  With the
C<recurse> flag unset, as it is by default, then each is just a place
marker for the directory and does not contain any further content
unless its C<scan()> method is explicitly called.  The C<isdir> flag can
be tested against files and/or directories, returning true if the item
is a directory or false if it is a regular file.

    [% FOREACH file = dir.list %]
       [% IF file.isdir %]
          * Directory: [% file.name %]
       [% ELSE %]
          * File: [% file.name %]
       [% END %]
    [% END %]

This example shows how you might walk down a directory tree, displaying 
content as you go.  With the recurse flag disabled, as is the default, 
we need to explicitly call the C<scan()> method on each directory, to force
it to lookup files and further sub-directories contained within. 

    [% USE dir = Directory(dirpath) %]
    * [% dir.path %]
    [% INCLUDE showdir %]
    
    [% BLOCK showdir -%]
      [% FOREACH file = dir.list -%]
        [% IF file.isdir -%]
        * [% file.name %]
          [% file.scan -%]
          [% INCLUDE showdir dir=file FILTER indent(4) -%]
        [% ELSE -%]
        - [% f.name %]
        [% END -%]
      [% END -%]
     [% END %]

This example is adapted (with some re-formatting for clarity) from
a test in F<t/directry.t> which produces the following output:

    * test/dir
        - file1
        - file2
        * sub_one
            - bar
            - foo
        * sub_two
            - waz.html
            - wiz.html
        - xyzfile

The C<recurse> flag can be set (disabled by default) to cause the
constructor to automatically recurse down into all sub-directories,
creating a new C<Template::Plugin::Directory> object for each one and 
filling it with any further content.  In this case there is no need
to explicitly call the C<scan()> method.

    [% USE dir = Directory(dirpath, recurse=1) %]
       ...
       
        [% IF file.isdir -%]
        * [% file.name %]
          [% INCLUDE showdir dir=file FILTER indent(4) -%]
        [% ELSE -%]
           ...

The directory plugin also provides support for views. A view can be defined as
a C<VIEW ... END> block and should contain C<BLOCK> definitions for files
('C<file>') and directories ('C<directory>').

    [% VIEW myview %]
    [% BLOCK file %]
       - [% item.name %]
    [% END %]
    
    [% BLOCK directory %]
       * [% item.name %]
         [% item.content(myview) FILTER indent %]
    [% END %]
    [% END %]

The view C<print()> method can then be called, passing the
C<Directory> object as an argument.

    [% USE dir = Directory(dirpath, recurse=1) %]
    [% myview.print(dir) %]

When a directory is presented to a view, either as C<[% myview.print(dir) %]>
or C<[% dir.present(view) %]>, then the C<directory> C<BLOCK> within the
C<myview> C<VIEW> is processed. The C<item> variable will be set to alias the
C<Directory> object.

    [% BLOCK directory %]
       * [% item.name %]
         [% item.content(myview) FILTER indent %]
    [% END %]

In this example, the directory name is first printed and the content(view)
method is then called to present each item within the directory to the view.
Further directories will be mapped to the C<directory> block, and files will be
mapped to the C<file> block.

With the recurse option disabled, as it is by default, the C<directory>
block should explicitly call a C<scan()> on each directory.

    [% VIEW myview %]
    [% BLOCK file %]
       - [% item.name %]
    [% END %]
    
    [% BLOCK directory %]
       * [% item.name %]
         [% item.scan %]
         [% item.content(myview) FILTER indent %]
    [% END %]
    [% END %]
    
    [% USE dir = Directory(dirpath) %]
    [% myview.print(dir) %]

=head1 AUTHORS

Michael Stevens wrote the original Directory plugin on which this is based.
Andy Wardley split it into separate L<File|Template::Plugin::File> and
L<Directory|Template::Plugin::Directory> plugins, added some extra code and
documentation for C<VIEW> support, and made a few other minor tweaks.

=head1 COPYRIGHT

Copyright (C) 2000-2007 Michael Stevens, Andy Wardley.

This module is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

=head1 SEE ALSO

L<Template::Plugin>, L<Template::Plugin::File>, L<Template::View>

