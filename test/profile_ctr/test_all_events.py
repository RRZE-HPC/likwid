#!/usr/bin/env python


import sys, os, re, subprocess, time

perfctr = "../../likwid-perfctr"
testcommand = "hostname"
outfilename = "/tmp/profile/ctr_read_write_cyc"
outfilename_suffix =".txt"
counters = []
eventdict = {}
parse_events = 0

LOOPS= 100

good_counters = ["PMC", "MBOX", "PWR", "TMP"]

selected_counters = []
# Read available counters and events
p = subprocess.Popen("../../likwid-perfctr -e", shell=True,bufsize=0, stdout=subprocess.PIPE, close_fds=True, stderr=subprocess.PIPE)
child_stdout = p.stdout.read()

for line in child_stdout.split('\n'):
    if not line.strip(): continue
    if "Counters names" in line:
        counters = re.split("\s+",line)
        for counter in counters:
            take_it = False
            for gcounter in good_counters:
                if counter.startswith(gcounter): take_it = True
            if take_it:
                selected_counters.append(counter)
        continue
    if "Event tags" in line:
        parse_events = 1
        continue
    if parse_events:
        linelist = line.split(",")
        eventdict[linelist[0].strip()] = {}
        eventdict[linelist[0].strip()]["counter"] = []
        for counter in selected_counters:
            if linelist[3].strip() in counter:
                eventdict[linelist[0].strip()]["counter"].append(counter)


if eventdict.has_key("INSTR_RETIRED_ANY"):
    eventdict["INSTR_RETIRED_ANY"]["counter"] += ["FIXC0","FIXC1","FIXC2"]
else:
    eventdict["INSTR_RETIRED_ANY"] = {}
    eventdict["INSTR_RETIRED_ANY"]["counter"] = ["FIXC0","FIXC1","FIXC2"]
    
if eventdict.has_key("CPU_CLK_UNHALTED_CORE"):
    eventdict["CPU_CLK_UNHALTED_CORE"]["counter"] += ["FIXC0","FIXC1","FIXC2"]
else:
    eventdict["CPU_CLK_UNHALTED_CORE"] = {}
    eventdict["CPU_CLK_UNHALTED_CORE"]["counter"] = ["FIXC0","FIXC1","FIXC2"]
    
if eventdict.has_key("CPU_CLK_UNHALTED_REF"):
    eventdict["CPU_CLK_UNHALTED_REF"]["counter"] += ["FIXC0","FIXC1","FIXC2"]
else:
    eventdict["CPU_CLK_UNHALTED_REF"] = {}
    eventdict["CPU_CLK_UNHALTED_REF"]["counter"] = ["FIXC0","FIXC1","FIXC2"]


used_group = []
check_events = []
for event in sorted(eventdict.keys()):
    for counter in eventdict[event]["counter"]:
        if counter not in used_group:
            check_events.append(event)
            used_group.append(counter)

check_events = list(set(check_events))

#print used_group
#print check_events
#for event in check_events:
#    print eventdict[event]["counter"]    


            
for event in check_events:
    if len(eventdict[event]["counter"]) == 0: continue
    for counter in eventdict[event]["counter"]:
        command = perfctr
        command += "  -M 0 -c 0 "
        command += '-g '+event+':'+counter+' '
        command += testcommand
        print "Testing read/write cycles for counter "+event+":"+counter
        outfile = open(outfilename+"_"+event+"_"+counter+outfilename_suffix,'w')
        outfile.write("# "+event+":"+counter+"\n")
        outfile.write("# Iteration Init Setup Read Write\n")
        for loop in range(0,LOOPS):

            p = subprocess.Popen(command, shell=True,bufsize=0, close_fds=True, \
                stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            current_stdout = p.stdout.read()
            eventdict[event][loop] = {}
            eventdict[event][loop]["write"] = 0
            eventdict[event][loop]["read"] = 0
            eventdict[event][loop]["init"] = 0
            eventdict[event][loop]["setup"] = 0
            for line in current_stdout.split("\n"):
	            if "Stop Init" in line:
	                eventdict[event][loop]["init"] = int(line.split(" ")[-1])
	            elif "Stop Setup" in line:
	                eventdict[event][loop]["setup"] = int(line.split(" ")[-1])
	            elif "Start Reading" in line:
	                parse_events = 1
	            elif "Stop Reading" in line:
	                parse_events = 0
	            if parse_events and "_Write" in line:
	                eventdict[event][loop]["write"] += int(line.split(" ")[3])
	            if parse_events and "_Read" in line:
	                eventdict[event][loop]["read"] += int(line.split(" ")[3])
            outfile.write("%d %d %d %d %d\n" % (loop, eventdict[event][loop]["init"], \
                eventdict[event][loop]["setup"], eventdict[event][loop]["read"], eventdict[event][loop]["write"],))
        outfile.close()
        
for event in sorted(eventdict.keys()):
    if len(eventdict[event]["counter"]) == 0: continue
    if event in check_events: continue
    counter = eventdict[event]["counter"][0]
    command = perfctr
    command += "  -M 0 -c 0 "
    command += '-g '+event+':'+counter+' '
    command += testcommand
    print "Testing read/write cycles for counter "+event+":"+counter
    outfile = open(outfilename+"_"+event+"_"+counter+outfilename_suffix,'w')
    outfile.write("# "+event+":"+counter+"\n")
    outfile.write("# Iteration Init Setup Read Write\n")
    for loop in range(0,LOOPS):

        p = subprocess.Popen(command, shell=True,bufsize=0, close_fds=True, \
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        current_stdout = p.stdout.read()
        eventdict[event][loop] = {}
        eventdict[event][loop]["write"] = 0
        eventdict[event][loop]["read"] = 0
        eventdict[event][loop]["init"] = 0
        eventdict[event][loop]["setup"] = 0
        for line in current_stdout.split("\n"):
            if "Stop Init" in line:
                eventdict[event][loop]["init"] = int(line.split(" ")[-1])
            elif "Stop Setup" in line:
                eventdict[event][loop]["setup"] = int(line.split(" ")[-1])
            elif "Start Reading" in line:
                parse_events = 1
            elif "Stop Reading" in line:
                parse_events = 0
            if parse_events and "_Write" in line:
                eventdict[event][loop]["write"] += int(line.split(" ")[3])
            if parse_events and "_Read" in line:
                eventdict[event][loop]["read"] += int(line.split(" ")[3])
        outfile.write("%d %d %d %d %d\n" % (loop, eventdict[event][loop]["init"], \
            eventdict[event][loop]["setup"], eventdict[event][loop]["read"], eventdict[event][loop]["write"],))
    outfile.close()

