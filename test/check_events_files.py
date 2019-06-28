#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================
#
#      Filename:  check_events_files.py
#
#      Description:  Basic checks for performance event list files
#
#      Author:   Mikhail Terekhov, termim@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
#
#      This program is free software: you can redistribute it and/or modify it under
#      the terms of the GNU General Public License as published by the Free Software
#      Foundation, either version 3 of the License, or (at your option) any later
#      version.
#
#      This program is distributed in the hope that it will be useful, but WITHOUT ANY
#      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License along with
#      this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =======================================================================================
from collections import OrderedDict
import json
from pathlib import Path
import sys

from pyparsing import *
ParserElement.setDefaultWhitespaceChars(' \t')



def err(fmt, *args, **kw):
    print(fmt.format(*args, **kw), file=sys.stderr)



class EventParser():
    """
    The expected event list EBNF gramma is:

    eol = '\n' ;
    letter = 'A' - 'Z' | 'a' - 'z' ;
    digit = '0' - '9'
    hexnum = ['0x'], { digit | 'a'-'f' | 'A'-'F' } ;
    number = ( digit, { digit } ) | hexnum ;

    group_name = name ;
    event_num = hexnum ;
    subsystem = letter, { letter | digit } ;
    event_name = group_name, '_', name ;
    option = name ;
    event_option = "EVENT_OPTION_", name, '=', number ;
    name = letter, { letter | digit | '_' } ;

    event_list = event_group, { event_group } ;
    event_group = event_group_header, event, { event } ;
    event_group_header = "EVENT_", group_name, event_num, subsystem, { '|', subsystem }, eol ;
    event = [ options ], [ default_options ], umask ;
    options = "OPTIONS_", event_name, option, { '|', option }, eol ;
    default_options = "DEFAULT_OPTIONS_", event_name, event_option, { ',', event_option }, eol ;
    umask = "UMASK_", event_name, hexnum, [ hexnum, [ hexnum ] ], eol ;

    """


    def __init__(self):

        Comment = pythonStyleComment.suppress()
        Hexnum = Literal("0x").suppress() + Word(hexnums).setParseAction(tokenMap(int,16))
        Number = (Optional(Literal("0x")) + Word(hexnums)).setParseAction(
                    lambda s, loc, toks:
                        int(toks[-1], 10 if len(toks) == 1 else 16))
        LinePrefix = lambda pref: Literal(pref) + Literal('_').suppress()
        Name = Word(alphanums + '_.')
        EventName = Name.setResultsName("name")

        self.Evt = (
            LinePrefix("EVENT") + EventName
            - Hexnum
                .setResultsName("number")
                .setName("event code (hex)")
            - delimitedList(Word(alphanums + '_')
                .setName("pipe (|) separated list of subsystem names (PMC, PBOX0|PBOX1 etc.)"),
                delim='|').setResultsName("subsystem")
            ).setParseAction(self._set_event)

        self.Options = (
            LinePrefix("OPTIONS") + EventName
            - delimitedList(
                        Literal("EVENT_OPTION").suppress() +
                        Literal('_').suppress() +
                        Name.setName("option name"),
                    delim='|')
                .setResultsName("options")
                .setName("pipe (|) separated list of option names")
            - LineEnd()
            ).setParseAction(self._set_options)

        self.DefaultOptions = (
            LinePrefix("DEFAULT_OPTIONS") + EventName
            - delimitedList(
                        Group(
                              Literal("EVENT_OPTION_").suppress() +
                              Name.setName("option name") +
                              Literal('=').suppress() +
                              Number.setName("option value (integer)")
                              ),
                    delim=',')
                .setResultsName("options")
                .setName("option list (comma separated list of NAME=VALUE pairs)")
            - LineEnd()
            ).setParseAction(self._set_default_options)

        self.Umask = (
            LinePrefix("UMASK") + EventName
            - OneOrMore(Hexnum)
                .setResultsName("umask")
                .setName("UMASK value (integer)")
            - LineEnd()
            ).setParseAction(self._parse_umask)

        self.Event = ( self.Umask | self.Evt | self.Options | self.DefaultOptions | Comment ) + StringEnd()

        self.event_head, self.event_count, self.events = None, 0, OrderedDict()
        self.options, self.default_options = None, None


    def _set_event(self, s, loc, token):

        event_head = token.asDict()
        for key in ('number', 'subsystem'):
            value = event_head[key]
            event_head[key] = tuple(value) if len(value) > 1 else value[0]
        event_head.update(line_num = self.line_num)

        if self.event_head:
            if event_head['name'] == self.event_head['name'] and event_head['subsystem'] == self.event_head['subsystem']:
                self.nerrors += 1
                err("\n{}:{}:\nDuplicate event name: {}"
                    "\nPrevious declaration is here:\n{}:{}:\n",
                    self.fname, self.line_num, event_head['name'], self.fname, self.event_head['line_num'])
            if self.event_count == 0:
                if event_head['name'].startswith(self.event_head['name']):
                    self.nerrors += 1
                    err('\n{}:{}:', self.fname, self.line_num)
                    err("Expected: 'UMASK_{}'", event_head['name'])
                    err("Found:    'EVENT_{}'", event_head['name'])
                else:
                    self.nerrors += 1
                    err('\n{}:{}:', self.fname, self.line_num)
                    err("Expected: 'UMASK_{}_....'", self.event_head['name'])
                    err("Found:    'EVENT_{}'", event_head['name'])

        self.event_head, self.event_count = event_head, 0


    def _set_options(self, s, loc, token):

        if self.options is not None:
            self.nerrors += 1
            err("\n{}:{}:\nUnused OPTIONS: {}", self.fname, self.line_num, token['name'])
        if self.opts.naming:
            if not token['name'].startswith(self.event_head['name']):
                self.nerrors += 1
                err("\n{}:{}:\nExpected OPTIONS name starting with: {}", self.fname, self.line_num, self.event_head['name'])
        self.options = token.asDict()['options']


    def _set_default_options(self, s, loc, token):

        if self.default_options is not None:
            self.nerrors += 1
            err("\n{}:{}:\nUnused DEFAULT_OPTIONS: {}", self.fname, self.line_num, token['name'])
        if self.opts.naming:
            if not token['name'].startswith(self.event_head['name']):
                self.nerrors += 1
                err("\n{}:{}:\nExpected DEFAULT_OPTIONS name starting with: {}", self.fname, self.line_num, self.event_head['name'])
        self.default_options = dict(token.asDict()['options'])


    def _parse_umask(self, s, loc, token):

        token = token.asDict()
        if self.opts.naming:
            # TODO: check for naming consistency
            if not token['name'].startswith(self.event_head['name']):
                self.nerrors += 1
                err("\n{}:{}:\nExpected UMASK name starting with: {}", self.fname, self.line_num, self.event_head['name'])

        event = self.event_head.copy()
        event.update(name=token['name'],
                     umask = token['umask'] if len(token['umask']) > 1 else token['umask'][0],
                     line_num = self.line_num
                     )

        key = event['subsystem'], event['name']
        if key in self.events:
            self.nerrors += 1
            err("\n{}:{}:\nDuplicate event name: {}"
                "\nPrevious declaration is here:\n{}:{}:\n",
                self.fname, self.line_num, event['name'], self.fname, self.events[key]['line_num'])

        if self.options is not None:
            event['options'], self.options = self.options, None
        if self.default_options is not None:
            event['default_options'], self.default_options = self.default_options, None

        self.events[key] = event
        self.event_count += 1


    def reset(self, fname, opts):

        self.fname = fname
        self.event_head, self.event_count, self.events = None, 0, OrderedDict()
        self.options, self.default_options = None, None
        self.opts = opts
        self.nerrors = 0


    def load(self, afile, opts):
        import copy
        lopts = copy.deepcopy(opts)
        lopts.naming = False
        self.check(afile, lopts)
        return self.events


    def check(self, afile, opts):

        self.reset(afile.name, opts)
        try:
            self.parse_lines(afile)
        except Exception as ex:
            self.nerrors += 1
            err("Internal Error: {}", ex)
        return self.nerrors


    def parse_lines(self, afile):

        self.line_num = 0
        for l in afile:
            self.line_num += 1
            l = l.strip()
            if not l: continue
            try:
                self.Event.parseString(l)
            except (ParseException, ParseSyntaxException) as pex:
                self.nerrors += 1
                err('\n{}:{}:{}:', afile.name, self.line_num, pex.col)
                err('{}:', pex.msg)
                err('{}', pex.line)
                err("{}^", '-'*pex.col)
        return self.nerrors


    def to_json(self, f, indent=4, sort_keys=True):

        json.dump([ dict([ (k, v) for k, v in V.items() if k != 'line_num' ])
                    for V in self.events.values() ],
                  f, indent=indent, sort_keys=sort_keys)



def test_EventParser(opts):

    import unittest

    class TestEventParser(unittest.TestCase):

        def setUp(self):
            self.parser = EventParser()
            self.reset_parser("", naming=False)

        def reset_parser(self, name, **kw):
            class Opts:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)
            opts = Opts(**kw)
            self.parser.reset(name, opts)

        def test_event(self):
            res = self.parser.Evt.parseString("EVENT_C_LO_AD_CREDITS_EMPTY             0x22 RBOX0C0|RBOX0C1|RBOX1C0|RBOX1C1").asDict()
            self.assertEqual(res["name"], "C_LO_AD_CREDITS_EMPTY")
            self.assertEqual(res["number"], [0x22])
            self.assertEqual(res["subsystem"], ['RBOX0C0', 'RBOX0C1', 'RBOX1C0', 'RBOX1C1'])


        def test_bad_event(self):
            with self.assertRaises(ParseSyntaxException):
                self.parser.Evt.parseString("EVENT_C_LO_AD_CREDITS_EMPTY 0x22")
            with self.assertRaises(ParseSyntaxException):
                self.parser.Evt.parseString("EVENT_C_LO_AD_CREDITS_EMPTY PMC")


        def test_options(self):
            res = self.parser.Options.parseString("OPTIONS_OFFCORE_RESPONSE_0_OPTIONS                  EVENT_OPTION_MATCH0_MASK|EVENT_OPTION_MATCH1_MASK").asDict()
            self.assertEqual(res["name"], "OFFCORE_RESPONSE_0_OPTIONS")
            self.assertEqual(res["options"], ['MATCH0_MASK', 'MATCH1_MASK'])


        def test_bad_options(self):
            with self.assertRaises(ParseSyntaxException):
                self.parser.Options.parseString("OPTIONS_OFFCORE")
            with self.assertRaises(ParseSyntaxException):
                self.parser.Options.parseString("OPTIONS_OFFCORE EVENT_OPTION_MATCH0_MASK,EVENT_OPTION_MATCH1_MASK")


        def test_default_options(self):
            res = self.parser.DefaultOptions.parseString("DEFAULT_OPTIONS_UOPS_ISSUED_CORE_TOTAL_CYCLES EVENT_OPTION_THRESHOLD=0xA,EVENT_OPTION_INVERT=1,EVENT_OPTION_ANYTHREAD=1").asDict()
            self.assertEqual(res["name"], "UOPS_ISSUED_CORE_TOTAL_CYCLES")
            self.assertEqual(res["options"], [['THRESHOLD', 10], ['INVERT', 1], ['ANYTHREAD', 1]])


        def test_bad_default_options(self):
            with self.assertRaises(ParseSyntaxException):
                self.parser.DefaultOptions.parseString("DEFAULT_OPTIONS_OFFCORE")
            with self.assertRaises(ParseSyntaxException):
                self.parser.DefaultOptions.parseString("DEFAULT_OPTIONS_OFFCORE EVENT_OPTION_MATCH0_MASK|EVENT_OPTION_MATCH1_MASK")


        def test_umask(self):
            self.parser.Umask.setParseAction(lambda s, loc, token: None)
            res = self.parser.Umask.parseString("UMASK_C_LO_AD_CREDITS_EMPTY             0x22 0x33").asDict()
            self.assertEqual(res["name"], "C_LO_AD_CREDITS_EMPTY")
            self.assertEqual(res["umask"], [0x22, 0x33])


        def test_umask_name(self):
            self.reset_parser("test_umask_name", naming=True)
            res = self.parser.parse_lines([
                "EVENT_SIMD_SAT_UOPS_EXEC   0xB1  PMC",
                "UMASK_SIMD_SAT_UOP_EXEC_AR 0x80"])
            self.assertEqual(res, 1)


        def test_options_name(self):
            self.reset_parser("test_options_name", naming=True)
            res = self.parser.parse_lines([
                "EVENT_SIMD_SAT_UOPS_EXEC   0xB1  PMC",
                "OPTIONS_SIMD_SAT_UOP_EXEC_AR EVENT_OPTION_OPCODE_MASK"])
            self.assertEqual(res, 1)


        def test_default_options_name(self):
            self.reset_parser("test_options_name", naming=True)
            res = self.parser.parse_lines([
                "EVENT_SIMD_SAT_UOPS_EXEC   0xB1  PMC",
                "DEFAULT_OPTIONS_SIMD_SAT_UOP_EXEC_AR EVENT_OPTION_DONAME=3"])
            self.assertEqual(res, 1)


    res = unittest.TextTestRunner().run(unittest.TestSuite((
                unittest.makeSuite(TestEventParser),
            )))
    return len(res.errors) + len(res.failures)



def resolve_events_files(opts):

    if not opts.files:
        files = opts.input_dir.glob("perfmon_*_events.txt")
    else:
        files = []
        for f in opts.files:
            f = Path(f).expanduser()
            if f.is_file():
                files.append(f.resolve())
            elif (opts.input_dir / f).is_file():
                files.append((opts.input_dir / f).resolve())
            elif (opts.input_dir / "perfmon_{}_events.txt".format(f)).is_file():
                files.append((opts.input_dir / "perfmon_{}_events.txt".format(f)).resolve())
            else:
                raise Exception("Can't find data file '{}'".format(f))
    return files


def check_events(opts):

    files = resolve_events_files(opts)
    ep = EventParser()
    nerrors = 0
    for fn in files:
        if opts.verbose:
            print("checking: {}".format(fn))
        with fn.open() as f:
            nerrors += ep.check(f, opts)
        if opts.json:
            o = opts.output_dir / fn.name.replace(fn.suffix, '.json')
            ep.to_json(o.open('w'), indent=4)
    if opts.verbose and nerrors:
        print("\nFound {} errors".format(nerrors))
    return nerrors > 0



if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser(description='Check data files.')
    ap.set_defaults(func=lambda _: ap.print_help())

    subparsers = ap.add_subparsers(title='subcommands',
                                        description="use '<subcmd> --help' for "
                                                    "help on subcommands",
                                        #help='valid subcommands'
                                        )

    etest = subparsers.add_parser('events', help="test events data files")
    etest.set_defaults(func=check_events)
    etest.add_argument('--verbose', '-v', action='store_true', default=False,
                       help="more verbose output"
                       )
    default = Path(__file__).parent / "../src/includes"
    default = default.resolve()
    etest.add_argument('--input-dir', '-d', type=Path, default=default,
                help="path to the directory with event data files, "
                     "default: {}".format(default)
                )
    etest.add_argument('files', metavar='FILE', type=str, nargs='*',
                help="event data file to check; "
                     "will be looked relative to current directory first, "
                     "then relative to the INPUT_DIR and then "
                     "INPUT_DIR/perfmon_{FILE}_events.txt filre will be tried."
                )
    etest.add_argument('--json', '-j', action='store_true', default=False,
                help="dump event descriptions in JSON format"
                )
    etest.add_argument('--output-dir', '-o', type=Path, default=Path('.'),
                help="path to the directory where to put output files, "
                     "default: current directory"
                )
    etest.add_argument('--naming', '-n', action='store_true', default=False,
                help="check for naming consistency, i.e. name in the "
                     "UMASK, OPTIONS and DEFAULT_OPTIONS lines should "
                     "have the name from the EVENT line as a prefix."
                )

    self_test = subparsers.add_parser('self', help="self test for the checker")
    self_test.set_defaults(func=test_EventParser)

    args = ap.parse_args()

    try:
        rc = args.func(args)
    except Exception as ex:
        err("{}", ex)
        sys.exit(1)
    sys.exit(rc)
