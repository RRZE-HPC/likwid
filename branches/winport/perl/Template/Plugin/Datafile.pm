#============================================================= -*-Perl-*-
#
# Template::Plugin::Datafile
#
# DESCRIPTION
#   Template Toolkit Plugin which reads a datafile and constructs a 
#   list object containing hashes representing records in the file.
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

package Template::Plugin::Datafile;

use strict;
use warnings;
use base 'Template::Plugin';

our $VERSION = 2.72;

sub new {
    my ($class, $context, $filename, $params) = @_;
    my ($delim, $line, @fields, @data, @results);
    my $self = [ ];
    local *FD;
    local $/ = "\n";

    $params ||= { };
    $delim = $params->{'delim'} || ':';
    $delim = quotemeta($delim);

    return $class->fail("No filename specified")
        unless $filename;

    open(FD, $filename)
        || return $class->fail("$filename: $!");

    # first line of file should contain field definitions
    while (! $line || $line =~ /^#/) {
        $line = <FD>;
        chomp $line;
        $line =~ s/\r$//;
    }

    (@fields = split(/\s*$delim\s*/, $line)) 
        || return $class->fail("first line of file must contain field names");

    # read each line of the file
    while (<FD>) {
        chomp;
        s/\r$//;

        # ignore comments and blank lines
        next if /^#/ || /^\s*$/;

        # split line into fields
        @data = split(/\s*$delim\s*/);

        # create hash record to represent data
        my %record;
        @record{ @fields } = @data;

        push(@$self, \%record);
    }

#    return $self;
    bless $self, $class;
}       


sub as_list {
    return $_[0];
}


1;

__END__

=head1 NAME

Template::Plugin::Datafile - Plugin to construct records from a simple data file

=head1 SYNOPSIS

    [% USE mydata = datafile('/path/to/datafile') %]
    [% USE mydata = datafile('/path/to/datafile', delim = '|') %]
    
    [% FOREACH record = mydata %]
       [% record.this %]  [% record.that %]
    [% END %]

=head1 DESCRIPTION

This plugin provides a simple facility to construct a list of hash 
references, each of which represents a data record of known structure,
from a data file.

    [% USE datafile(filename) %]

A absolute filename must be specified (for this initial implementation at 
least - in a future version it might also use the C<INCLUDE_PATH>).  An 
optional C<delim> parameter may also be provided to specify an alternate
delimiter character.

    [% USE userlist = datafile('/path/to/file/users')     %]
    [% USE things   = datafile('items', delim = '|') %]

The format of the file is intentionally simple.  The first line
defines the field names, delimited by colons with optional surrounding
whitespace.  Subsequent lines then defines records containing data
items, also delimited by colons.  e.g.

    id : name : email : tel
    abw : Andy Wardley : abw@tt2.org : 555-1234
    sam : Simon Matthews : sam@tt2.org : 555-9876

Each line is read, split into composite fields, and then used to 
initialise a hash array containing the field names as relevant keys.
The plugin returns a blessed list reference containing the hash 
references in the order as defined in the file.

    [% FOREACH user = userlist %]
       [% user.id %]: [% user.name %]
    [% END %]

The first line of the file B<must> contain the field definitions.
After the first line, blank lines will be ignored, along with comment
line which start with a 'C<#>'.

=head1 BUGS

Should handle file names relative to C<INCLUDE_PATH>.
Doesn't permit use of 'C<:>' in a field.  Some escaping mechanism is required.

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
