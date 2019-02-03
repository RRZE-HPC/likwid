import os
import fileinput

def parse_likwid_file(filename, last_line = ''):
    '''Parse terminal output of LIKWID

    Arguments:
    filename -- the path to the file
    last_line -- the last line of non-LIKWID output. Starting from the line after this
        the parser will be active

    Return:
    parsed data stored as:
        dictionary (region)
        dictionary (table name i.e. Event, Event_Sum, Metric, Metric_Sum or alike
        dictionary of rows.

    That is, one can get the row as result['vmult']['Metric_Sum']['MFLOP/s']
    '''
    result = {}
    fin = open(filename, 'r')
    debug_output = False

    row_separator = '---------'

    found_start = (last_line == "")
    region = ''
    separator_counter = 0
    table_name = ''
    table_ind = 0

    for line in fin:
        line = line.strip()

        # skip empty lines
        if line == "":
            continue

        if found_start:
            #
            # Main logic to parse
            #

            # Check if we found one of the regions:
            if 'Region:' in line:
                region = line[8:]
                if debug_output:
                    print '-- Region: {0}'.format(region)
                # regions should be unique
                assert region not in result
                result[region] = {}
                separator_counter = 0
                table_name = ''
                continue

            if 'Group:' in line:
                # FIXME: read in groups as well?
                continue

            # If we are in LIKWID part, we should have some region always around
            assert region != ''

            # At this point we have only 3 options: we are on one of the separators,
            # inside the header or inside the core of the table.
            if row_separator in line:
                separator_counter = separator_counter + 1
                # reset the counter if we are at the end of the current table
                if separator_counter == 3:
                    separator_counter = 0
                continue

            # get the columns, disregard empty first and last
            columns = [s.strip() for s in line.split('|')][1:-1]
            if separator_counter == 1:
                # should be reading the header, get the name:
                table_name = columns[0]
                if 'Sum' in line and 'Min' in line and 'Max' in line and 'Avg' in line:
                    table_name = table_name + '_Sum'

                # table names for a given region should be unique
                assert table_name not in result[region]
                result[region][table_name] = {}
                if debug_output:
                    print '   Name  : {0}'.format(table_name)
            else:
                # we should have table name around already
                assert table_name != ''
                # we should have non-empty list of columns
                assert len(columns) > 1
                # index should be either 0 or 1
                assert table_ind == 0 or table_ind == 1
                key = columns[0]
                val = columns[1:]

                # finally put the data
                result[region][table_name][key] = val
                if debug_output:
                    print '      {0}'.format(key)
        else:
            # If we have not found LIKWID part yet
            if last_line in line:
                found_start = True

    return result

