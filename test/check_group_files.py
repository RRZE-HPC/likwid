#!/usr/bin/env python

# =======================================================================================
#
#      Filename:  check_group_files.py
#
#      Description:  Basic checks for performance group files
#
#      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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

import sys, os, re, glob, os.path, subprocess

#SRCPATH="/home/rrze/unrz/unrz139/Work/likwid"
SRCPATH="/home/tom/Work/likwid-branch-4.3.0"
GROUPPATH=os.path.join(SRCPATH, "groups")
EVENTHEADERPATH=os.path.join(SRCPATH, "src/includes")
PERFCTR="likwid-perfctr"

def get_valid_shorts():
    l = os.listdir(GROUPPATH)
    return l


def get_local_arch():
    p = subprocess.Popen("%s -i" % PERFCTR, stdout=subprocess.PIPE, shell=True)
    stdoutdata, stderrdata = p.communicate()
    if p.returncode == 0:
        for l in stdoutdata.split("\n"):
            if l.startswith("CPU short:"):
                return re.split("\s+", l)[-1]


def get_all_events_local_arch():
    events = {}
    p = subprocess.Popen("%s -e" % PERFCTR, stdout=subprocess.PIPE, shell=True)
    stdoutdata, stderrdata = p.communicate()
    if p.returncode == 0:
        parse = False

        for l in stdoutdata.split("\n"):
            if l.startswith("Event tags"):
                parse = True
                continue
            if parse:
                if not l.strip(): continue
                llist = re.split(",\s+", l)
                events[llist[0]] = llist[-1]
    return events

def get_all_events_given_arch(arch):
    events = {}
    efile = os.path.join(EVENTHEADERPATH, "perfmon_%s_events.txt" % arch)
    if os.path.exists(efile):
        f = open(efile, "r")
        raw = f.read().strip().split("\n")
        f.close()
        limit = None
        for l in raw:
            if l.startswith("EVENT_"):
                limit = re.split("\s+", l)[-1]
            if l.startswith("UMASK_"):
                e = re.match("UMASK_([\w\d_]+)", l)
                if e:
                    events[e.group(1)] = limit
    return events

def get_all_counters_local_arch():
    counters = []
    p = subprocess.Popen("%s -e" % PERFCTR, stdout=subprocess.PIPE, shell=True)
    stdoutdata, stderrdata = p.communicate()
    if p.returncode == 0:
        parse = False

        for l in stdoutdata.split("\n"):
            if l.startswith("Counter tags"):
                parse = True
                continue
            if parse:
                if not l.strip(): continue
                llist = re.split(",\s+", l)
                counters.append(llist[0])
    return counters

def get_all_counters_given_arch(arch):
    counters = []
    cfile = os.path.join(EVENTHEADERPATH, "perfmon_%s_counters.h" % arch)
    if os.path.exists(cfile):
        f = open(cfile, "r")
        raw = f.read().strip().split("\n")
        f.close()
        parse = False
        for l in raw:
            if l.startswith("static RegisterMap"):
                parse = True
                continue
            if l.startswith("};"):
                parse = False
            if parse:
                c = re.match("^\s+{\"([\w\d]+)\"", l)
                if c:
                    counters.append(c.group(1))
    return counters

def get_all_groupfiles(arch):
    path=os.path.join(GROUPPATH, arch)
    if not os.path.exists(path):
        print("Cannot find group path %s" % path)
        return []
    grouplist = glob.glob(path+"/*")
    return grouplist

def check_short(gfile):
    if not os.path.exists(gfile):
        print("Cannot find group file %s" % gfile)
        return False
    with open(gfile) as f:
        inf = f.read().split("\n")
        for l in inf:
            if l.startswith("SHORT"):
                if len(re.split("\s+", l)) > 1:
                    return True
                else:
                    return False
    return False

def check_eventset(gfile, allevents, allcounters):
    events = {}
    noht = False
    if not os.path.exists(gfile):
        print("Cannot find group file %s" % gfile)
        return False
    with open(gfile) as f:
        parse = False
        inf = f.read().split("\n")
        for l in inf:
            if l.startswith("REQUIRE_NOHT"):
                noht = True
            if l.startswith("EVENTSET"):
                parse = True
                continue
            if l.startswith("METRICS"):
                parse = False
                continue
            if parse:
                if not l.strip(): continue
                elist = re.split("\s+", l)
                o = None
                c = elist[0]
                e = elist[1]
                if ":" in c:
                    tmp = c.split(":")
                    c = tmp[0]
                    o = ":".join(tmp[1:])
                if not events.has_key(c):
                    events[c] = e
                else:
                    print("Counter register used twice: %s and %s" % (e, events[c]))
                    return False
    for c in events.keys():
        if c not in allcounters and not noht:
            print("Counter register %s does not exist" % c)
            return False
        elif c not in allcounters:
            print("Group requires HyperThreading to be off!")
            return True
        if events[c] not in allevents.keys():
            print("Event %s unknown" % events[c])
            return False
    return True

def check_metrics(gfile):
    if not os.path.exists(gfile):
        print("Cannot find group file %s" % gfile)
        return False
    metrics = {}
    with open(gfile) as f:
        parse = False
        inf = f.read().split("\n")
        for l in inf:
            if l.startswith("METRICS"):
                parse = True
                continue
            if l.startswith("LONG"):
                parse = False
                continue
            if parse and len(l) > 0:
                llist = re.split("\s+", l)
                name = " ".join(llist[:-1])
                if "[G" in l and not re.match("1[\.\d]*E[-+][\d]*9", llist[-1]):
                    print("Wrong unit? %s" % l)
                if "[M" in l and not "[MHz]" in l and not re.match("1[\.\d]*E[-+][\d]*6", llist[-1]):
                    print("Wrong unit? %s" % l)
                if "[%]" in l and not ("100*" in l or "*100" in l):
                    print("Scaling factor missing? %s" % l)
                metrics[" ".join(llist[:-1])] = llist[-1]
    if len(metrics.keys()) > 0:
        return True
    return False

def check_long(gfile):
    if not os.path.exists(gfile):
        print("Cannot find group file %s" % gfile)
        return False
    longlines = []
    with open(gfile) as f:
        parse = False
        inf = f.read().split("\n")
        for l in inf:
            if l.startswith("LONG"):
                parse = True
                continue
            if parse and len(l) > 0:
                longlines.append(l)
    if len(longlines) > 0:
        return True
    return False

arch = None
if len(sys.argv) == 2:
    arch = sys.argv[1]
    if arch == "-h" or arch == "--help":
        print("Checks performance group files for some basic problems:")
        print("\t- Is a short description defined?")
        print("\t- Is an eventset defined?")
        print("\t- Are all counters in the eventset available?")
        print("\t- Are all events in the eventset available?")
        print("\t- Is the scaling factor correct accoring to the unit in metric name")
        print("\t- Is there a long description?")
        print("")
        print("If no command line argument is given, all architectures are tested.")
        print("")
        print("If the first argument is a valid architecture string, the lists are")
        print("filled from definitions from source.")
        print("")
        print("If first argument is 'local' the current system is checked. This can")
        print("throw more errors as the other modes as the output of likwid-perfctr -e")
        print("contains only counters and events that are accessible on the current system.")
        print("")
        print("Available architectures:")
        print(", ".join(get_valid_shorts()))
        sys.exit(0)
    if arch != "local" and arch not in get_valid_shorts():
        print("Given arch not available")
        sys.exit(1)

alist = []
if arch == "local":
    events = get_all_events_local_arch()
    counters = get_all_counters_local_arch()
    alist = [get_local_arch()]
elif arch != None:
    ea = arch
    ca = arch
    if arch == "broadwellD":
        ea = "broadwelld"
        ca = "broadwelld"
    if arch == "pentiumm":
        ea = "pm"
        ca = "pm"
    if arch == "westmere":
        ca = "nehalem"
    if arch == "atom":
        ca = "core2"
    if arch == "k8":
        ca = "k10"
    events = get_all_events_given_arch(ea)
    counters = get_all_counters_given_arch(ca)
    alist = [arch]
else:
    alist = get_valid_shorts()

for a in alist:
    print("Checking architecture %s" % a)
    glist = get_all_groupfiles(a)
    if len(alist) > 1:
        ea = a
        ca = a
        if a == "broadwellD":
            ea = "broadwelld"
            ca = "broadwelld"
        if a == "pentiumm":
            ea = "pm"
            ca = "pm"
        if a == "westmere":
            ca = "nehalem"
        if a == "atom":
            ca = "core2"
        if a == "k8":
            ca = "k10"
        events = get_all_events_given_arch(ea)
        counters = get_all_counters_given_arch(ca)
    for f in glist:
        g = os.path.basename(f).split(".")[0]
        print("Checking group %s" % g)
        if not check_short(f):
            print("Short failure in group file %s" % f)
        if not check_eventset(f, events, counters):
            print("Eventset failure in group file %s" % f)
        if not check_metrics(f):
            print("Metrics failure in group file %s" % f)
        if not check_long(f):
            print("Long failure in group file %s" % f)
    print("")
