#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================================================================================
#
#      Filename:  check_data_files.py
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
import logging as log
from pathlib import Path
from pprint import pprint, pformat
import sys

from pyparsing import *
ParserElement.setDefaultWhitespaceChars(' \t')



def err(fmt, *args, **kw):
    log.error(fmt.format(*args, **kw))



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
        self.Evt.ignore(Comment)

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
        self.Options.ignore(Comment)

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
        self.DefaultOptions.ignore(Comment)

        self.Umask = (
            LinePrefix("UMASK") + EventName
            - OneOrMore(Hexnum)
                .setResultsName("umask")
                .setName("UMASK value (integer)")
            - LineEnd()
            ).setParseAction(self._parse_umask)
        self.Umask.ignore(Comment)

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
        self.line_num = 0


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
        files = list(opts.input_dir.glob("perfmon_*_events.txt"))
        if not files:
            raise Exception("Couldn't find any events files in '{}', "
                            "please use --input-dir option.\n"
                            "Use `{} events --help` for details."
                            .format(opts.input_dir, sys.argv[0]))
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

    start_logging(args)
    files = resolve_events_files(opts)
    if opts.json:
        opts.output_dir.mkdir(parents=True, exist_ok=True)
    ep = EventParser()
    nerrors = 0
    for fn in files:
        log.info("checking: {}".format(fn))
        with fn.open() as f:
            nerrors += ep.check(f, opts)
        if opts.json:
            o = opts.output_dir / fn.name.replace(fn.suffix, '.json')
            with o.open('w') as jf:
                ep.to_json(jf, indent=4)
    if nerrors:
        log.info("\nFound {} errors".format(nerrors))
    return nerrors > 0



UNITS = [
    's',
    'MHz',
    'MFLOP/s',
    'MUOPS/s',
    'MBytes/s',
    'GBytes',
    '%',
    'J',
    'W',
    'C',
    ]

def miss_brackets_in_units(s):

    for dim in UNITS:
        if ' {} '.format(dim) in s:
            return dim
        if s.endswith(' {}'.format(dim)):
            return dim
    return None



def extract_units(s):

    l = s.split()
    if len(l) > 1 and l[-1].startswith('[') and l[-1].endswith(']'):
        return ' '.join(l[:-1]), l[-1][1:-1]
    else:
        return s, None



class GroupParser():
    """
    eol = '\n' ;
    letter = 'A' - 'Z' | 'a' - 'z' ;
    digit = '0' - '9'
    name = letter, { letter | digit | '_' } ;
    word = -eol, { -eol } ;
    line = word, { word }, eol ;
    op = '*' | '/' | '+' | '-' ;

    short = "SHORT", short_title, eol ;
    noht = "REQUIRE_NOHT", eol ;
    counter_name = name ;
    event_name = name ;
    event = counter_name, event_name, eol ;
    eventset = "EVENTSET", eol, event, { event }, eol ;
    metric_name = letter, { letter | digit | '_' | '(' | ')' | ' ' } ;
    units = UNITS
    metric = metric_name, [ units ], counter_name, {  op, counter_name }, eol ;
    metrics = "METRICS", eol, metric, { metric }, eol ;
    formula = metric_name, [ units ], '=', event_name, {  op,  event_name}, eol ;
    formulas = "Formulas:", eol, formula, { formula } ;
    sep = '--' | '-', eol ;
    long = line, { line } ;
    group = short, [ noht ], eventset, metrics, "LONG", eol, formulas, sep, long
    """

    def __init__(self):

        EOL = LineEnd().suppress()
        Line = LineStart() + SkipTo(LineEnd(), failOn=LineStart()+LineEnd()) + EOL

        short = Keyword('SHORT').suppress() + SkipTo(LineEnd(), failOn=LineStart()+LineEnd()) + EOL
        short.setParseAction(
            lambda s, loc, toks:
                self.group.update(short=toks[0])
        )
        self.parser = short + OneOrMore(EOL)

        noht = Keyword('REQUIRE_NOHT')
        noht.setParseAction(
            lambda s, loc, toks:
                self.group.update(require_noht=True)
        )
        self.parser += Optional(noht() + OneOrMore(EOL))

        eventset = Keyword('EVENTSET').suppress() + EOL + \
        Group(OneOrMore(
            LineStart() + Group(Word(alphanums) + SkipTo(LineEnd(), failOn=LineStart()+LineEnd())
                            ).setParseAction(self.add_event) + EOL
        ))
        self.parser += eventset + OneOrMore(EOL)

        metrics = Keyword('METRICS').suppress() + EOL + \
        Group(OneOrMore(
            LineStart() + Group(OneOrMore(Word(alphanums+'.[]()-+*/')) + SkipTo(LineEnd(), failOn=LineStart()+LineEnd())
                            ).setParseAction(self.add_metric) + EOL
        ))
        self.parser += metrics + OneOrMore(EOL)

        long = Keyword('LONG').suppress() + EOL
        self.parser += long

        Formula = LineStart() + SkipTo(LineEnd(), failOn=LineStart()+(Keyword('-') ^ Keyword('--'))+LineEnd()) + EOL
        formulae = Keyword('Formulas:') + EOL + OneOrMore(Formula().setParseAction(self.add_formula))
        self.parser += formulae

        descr = (Keyword('-') ^ Keyword('--')).suppress() + EOL + Group(OneOrMore(Line()))
        descr.setParseAction(
            lambda s, loc, toks:
                self.group.update(long=' '.join(toks[0]))
        )
        self.parser += descr


    def add_event(self, s, loc, toks):
        log.debug("add_event: |{}|".format(toks[0]))
        self.group['events'].append(dict(counter=toks[0][0], event=toks[0][1]))


    def add_metric(self, s, loc, toks):
        log.debug("add_metric: |{}|".format(toks[0]))
        metric = ' '.join(toks[0][0:-2])
        dim = miss_brackets_in_units(metric)
        if dim:
            raise ParseSyntaxException(s, loc, "expected '[{}]', found '{}'".format(dim, dim))
        metric, units = extract_units(metric)
        formula = toks[0][-2]
        self.group['metrics'].append(dict(metric=metric, formula=formula, units=units))


    def add_formula(self, s, loc, toks):
        if not '=' in toks[0]:
            i = [toks[0].find(x['event']) for x in self.group['events'] if toks[0].find(x['event'])>=0]
            pos = min(i) if i else 0
            while pos > 0 and toks[0][pos] != ' ':
                pos -= 1
            raise ParseSyntaxException(s,loc + pos, "expected '=' sign")
        metric, formula = [ x.strip() for x in toks[0].split('=', 1) ]
        dim = miss_brackets_in_units(metric)
        if dim:
            raise ParseSyntaxException(s, loc, "expected '[{}]', found '{}'".format(dim, dim))
        metric, units = extract_units(metric)
        log.debug("add_formula: |{}| = |{}|".format(metric, formula))
        #if metric in self.group['formulae']:
            #self.nerrors += 1
            #err("\n{}:{}:\nDuplicate metric formula: {}", self.fname, self.line_num, token['name'])
        self.group['formulae'].append(dict(metric=metric, formula=formula, units=units))


    def reset(self, fname, opts):
        self.fname = fname
        self.opts = opts
        self.nerrors = 0
        self.group = dict(
                          short=None,
                          events=[],
                          metrics=[],
                          formulae=[],
                          long=[]
                          )
        self.nerrors = 0


    def check(self, afile, opts):
        self.reset(afile.name, opts)
        try:
            res =self.parser.parseFile(afile)
            log.debug(pformat(self.group))
            log.debug('\ngroup=\n{}'.format(json.dumps(self.group, indent=2)))
        except Exception as pex:
            self.nerrors += 1
            err('\n{}:{}:{}:', self.fname, pex.lineno, pex.col)
            err('{}:', pex.msg)
            err('{}', pex.line)
            err("{}^", '-'*(pex.col-1))
        return self.nerrors


    def to_json(self, f, indent=4, sort_keys=True):

        json.dump(self.group, f, indent=indent, sort_keys=sort_keys)



def resolve_group_files(opts):

    if not opts.files:
        files = list(opts.input_dir.glob("**/*.txt"))
        if not files:
            raise Exception("Couldn't find any group files in '{}', "
                            "please use --input-dir option.\n"
                            "Use `{} groups --help` for details."
                            .format(opts.input_dir, sys.argv[0]))
    else:
        files = []
        for f in opts.files:
            f = Path(f).expanduser()
            _files = []
            if f.is_file():
                _files.append(f.resolve())
            elif (opts.input_dir / f).is_file():
                _files.append((opts.input_dir / f).resolve())
            elif (opts.input_dir / "{}.txt".format(f)).is_file():
                _files.append((opts.input_dir / "{}.txt".format(f)).resolve())
            elif (opts.input_dir / f).is_dir():
                _files = [x.resolve() for x in (opts.input_dir / f).glob("*.txt")]
            if not _files:
                trials = [f, (opts.input_dir / f), opts.input_dir / "{}.txt".format(f), (opts.input_dir / f / '*.txt')]
                raise Exception("Can't find data file:\n\t   '{}'".format("'\n\tor '".join([str(f) for f in trials])))
            files.extend(_files)
    return files



def check_groups(opts):

    start_logging(args)
    files = resolve_group_files(opts)
    gp = GroupParser()
    nerrors = 0
    for fn in files:
        log.info("checking: {}".format(fn))
        with fn.open() as f:
            nerrors += gp.check(f, opts)
        if opts.json:
            o = opts.output_dir / fn.relative_to(opts.input_dir).with_suffix('.json')
            o.parent.mkdir(parents=True, exist_ok=True)
            with o.open('w') as jf:
                gp.to_json(jf, indent=4)
    if nerrors:
        log.info("\nFound {} errors".format(nerrors))
    return nerrors > 0



def start_logging(args):

    log_levels = log.ERROR, log.INFO, log.DEBUG,
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    log.basicConfig(
                    #filename=str(Path(__file__).with_suffix('.log')),
                    #filemode='w',
                    format='%(message)s',
                    level=log_level)



def abs_path(path):
    return Path(path).expanduser().resolve()

if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser(description='Check data files.')
    ap.set_defaults(func=lambda _: ap.print_help())

    subparsers = ap.add_subparsers(title='subcommands',
                                        description="use '<subcmd> --help' for "
                                                    "help on subcommands",
                                        )

    etest = subparsers.add_parser('events', help="test events data files")
    etest.set_defaults(func=check_events)
    etest.add_argument('--verbose', '-v', action='store_true', default=False,
                       help="more verbose output"
                       )
    default = abs_path(__file__).parent / "../src/includes"
    default = default.resolve()
    etest.add_argument('--input-dir', '-d', type=abs_path, default=default,
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
    etest.add_argument('--output-dir', '-o', type=abs_path, default=abs_path('.'),
                help="directory where to put output JSON files, "
                     "default: current directory"
                )
    etest.add_argument('--naming', '-n', action='store_true', default=False,
                help="check for naming consistency, i.e. name in the "
                     "UMASK, OPTIONS and DEFAULT_OPTIONS lines should "
                     "have the name from the EVENT line as a prefix."
                )

    self_test = subparsers.add_parser('self', help="self test for the events checker")
    self_test.set_defaults(func=test_EventParser)

    gtest = subparsers.add_parser('groups', help="test group ddescriptions")
    gtest.set_defaults(func=check_groups)
    gtest.add_argument('--verbose', '-v', action='count', default=0)
    default = abs_path(__file__).parent / "../groups"
    if default.exists():
        default = default.resolve()
    else:
        default = abs_path(__file__).parent
    gtest.add_argument('--input-dir', '-d', type=abs_path,
                        default=default,
                        help='path to the directory with groups data files, default: {}'.format(default))
    gtest.add_argument('files', metavar='FILE', type=str, nargs='*',
                    help='group data file to check')
    gtest.add_argument('--json', '-j', action='store_true', default=False,
                    help="dump group descriptions in JSON format")
    gtest.add_argument('--output-dir', '-o', type=abs_path,
                        default=abs_path('.'),
                        help='directory where to put output JSON files, default: current directory')

    args = ap.parse_args()

    try:
        rc = args.func(args)
    except Exception as ex:
        err("{}", ex)
        sys.exit(1)
    sys.exit(rc)
