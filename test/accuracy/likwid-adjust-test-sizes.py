#!/usr/bin/env python


import os, sys, os.path, re, subprocess

topology_exec = "/home/hpc/unrz/unrz139/Apps-arm/bin/likwid-topology"
topology_re_size = re.compile("^Size:\s+(.*)")
re_size_unit = re.compile("(\d+)\s(\w+)")

cachesizes = []

def get_caches():
    level = 0
    print("here")
    p = subprocess.Popen(topology_exec, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    sout = p.stdout.read()
    if p.returncode != 0:
        print("err", p.returncode )
        return level
    for line in sout.split("\n"):
        if line.startswith("Size:"):
            print(line)
            string = topology_re_size.match(line).group(1).strip()
            size, unit = re_size_unit.match(string).groups()
            if unit == "kB":
                size = int(size)*1024
            elif unit == "MB":
                size = int(size)*1024*1024
            cachesizes.append(size)
            level += 1
    fp = open("/proc/meminfo")
    f = fp.read().strip().split("\n")
    fp.close()
    for line in f:
        if line.startswith("MemTotal:"):
            linelist = re.split("\s+", line)
            size = int(linelist[1])
            if linelist[2] == "kB":
                size *= 1024
            elif linelist[2] == "MB":
                size *= 1024*1024
            if size > 1024*1024*1024:
                size = 1024*1024*1024
            cachesizes.append(size)
    return level

def get_important_tests():
    important = ["L2", "L3", "MEM", "CLOCK", "UOPS", "HA"]
    adjust = []
    regular = []
    fp = open("SET.txt")
    f = fp.read().strip().split("\n")
    fp.close()
    for line in f:
        found = False
        for imp in important:
            if imp in line:
                adjust.append(line)
                found = True
        if not found:
            regular.append(line)
    return adjust, regular

def adjust_tests(testgroup):
    fp = open("TESTS/"+testgroup+".txt", "r")
    f = fp.read().strip().split("\n")
    fp.close()
    newdata = []
    level = re.match("L(\d+)", testgroup)
    if level:
        level = int(level.group(1))-1
    else:
        level = len(cachesizes)-1
    print(level, cachesizes)
    min_size = int((cachesizes[level-1] + (0.3*cachesizes[level-1]))/1024)
    max_size = int((cachesizes[level] - (0.2*cachesizes[level]))/1024)
    diff = (cachesizes[level] - cachesizes[level-1])/1024
    step = diff/5
    i = 0
    while i < len(f):
        if not f[i].startswith("VARIANT"):
            newdata.append(f[i]+"\n")
            i+=1
        else:
            count = 0
            for j in range(i,i+4):
                if f[j].startswith("VARIANT"):
                    count += 1
                else: break
            i += count
            newdata.append("VARIANT %dkB 1000\n" % (int(min_size+step),))
            newdata.append("VARIANT %dkB 1000\n" % (int(min_size+(2*step)),))
            newdata.append("VARIANT %dkB 1000\n" % (int(min_size+(3*step)),))
            newdata.append("VARIANT %dkB 1000\n" % (int(min_size+(4*step)),))
    fp = open("TESTS/"+testgroup+".txt", "w")
    for line in newdata:
        fp.write(line)
    fp.close()

level = get_caches()
adjust, regular = get_important_tests()
for testgroup in adjust:
    print("Adjusting "+testgroup)
    adjust_tests(testgroup)
if len(regular) > 0:
    print("Not adjusting:")
    print(regular)


