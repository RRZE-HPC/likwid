#!/usr/bin/python

import sys, os, re, glob, os.path
import statistics

outfilename = "ctr_event_"
outfilename_suffix = ".txt"
outfolder = "aggregated"

picfilename_suffix = ".png"
picfolder = "pics"

scriptname = "ctr_event_"
scriptname_suffix = ".plot"
scriptfolder = "g_scripts"

if not os.path.exists(outfolder): os.mkdir(outfolder)
if not os.path.exists(outfolder): os.mkdir(picfolder)
if not os.path.exists(outfolder): os.mkdir(scriptfolder)

avail_regs = ["PMC", "MBOX", "PWR", "TMP","FIXC"]

if len(sys.argv) != 2:
    print "Please give a counter name as cmd parameter"
    sys.exit(1)



counter = sys.argv[1]
register = ""
filelist = glob.glob("./ctr_read_write*"+counter+"*")


ids = ["init","setup","read","write"]
amount = 0
all = [[],[],[],[]]
for file in filelist:
    infile = open(file,'r')
    for line in infile.read().split("\n"):
        if line.startswith("#") or line == "": continue
        linelist = re.split("\s+", line)
        for i in range(0,4):
            if int(linelist[i+1]) != 0:
                all[i].append(int(linelist[i+1]))
        amount += 1



for i in range(0,4): all[i].sort()
mini = []
maxi = []
medi = []
avg = []
var = []

#first_index = (amount * 5) /100
#last_index = (amount * 95) / 100
first_index = 0
last_index = amount

for i in range(0,4):
    mini.append(min(all[i][first_index:last_index]))
    maxi.append(max(all[i][first_index:last_index]))
    medi.append(statistics.median(all[i][first_index:last_index]))
    avg.append(statistics.mean(all[i][first_index:last_index]))
    var.append(statistics.variance(all[i][first_index:last_index]))

#print "Minima: "+ str(mini)
#print "Maxima: "+ str(maxi)
#print "Median: "+ str(medi)
#print "Average: "+ str(avg)
#print "Variance: "+ str(var)

outfile = open(os.path.join(outfolder,outfilename+counter+outfilename_suffix),'w')
outfile.write("# ID Min Max Median Average Variance\n")
for i in range(0,4):
    outfile.write("%s %d %d %d %.1f %.2f %.2f\n" % (ids[i], i, mini[i],maxi[i], medi[i], avg[i], var[i],))
outfile.close()

script = open(os.path.join(scriptfolder,scriptname+counter+scriptname_suffix), 'w')
for id in avail_regs:
    if (filelist[0].split("_")[-1].split(".")[0][:-1].startswith(id)): reg = id;
    
script.write("set title 'Event %s using %s registers';\n" % (counter, reg ,))
script.write("set terminal png;\n")
script.write("set output '../%s';\n" % (os.path.join(picfolder,outfilename+counter+picfilename_suffix),))
script.write("set xtics 1;\n")
script.write("set logscale y;\n")
script.write("set xrange [-0.5:3.5];\n")
script.write("set yrange [1:10000000];\n")
script.write("set ylabel 'Cycles';\n")
script.write("set xtics ('Init' 0, 'Setup' 1, 'Read' 2, 'Write' 3);\n")
script.write("plot '../%s/%s' using 2:6:3:4 title '%s' with errorbars\n" % (outfolder,outfilename+counter+outfilename_suffix,counter,))
script.close()
