#!/usr/bin/env python

import sys, os, re, glob, os.path
import statistics

outfilename = "ctr_counter_"
outfilename_suffix = ".txt"
outfolder = "aggregated"

picfilename_suffix = ".png"
picfolder = "pics"

scriptname = "ctr_counter_"
scriptname_suffix = ".plot"
scriptfolder = "g_scripts"

avail_regs = ["PMC", "MBOX", "PWR", "TMP", "FIXC"]

if len(sys.argv) != 2:
    print "Please give a counter name as cmd parameter"
    sys.exit(1)
    
if not os.path.exists(outfolder): os.mkdir(outfolder)
if not os.path.exists(picfolder): os.mkdir(picfolder)
if not os.path.exists(scriptfolder): os.mkdir(scriptfolder)
    
event = sys.argv[1]
filelist = glob.glob("./ctr_read_write*"+event+"_*")
cur_regs = []
data = {}
for file in filelist:
    for reg in avail_regs:
        match = None
        match = re.match("./ctr_read_write_cyc_"+event+"_[\w_]*(\w*"+reg+"[\d\w]*).txt", file)
        if match:
            cur_regs.append(match.group(1))
            data[match.group(1)] = {}
            data[match.group(1)]["read"] = []
            data[match.group(1)]["write"] = []

    fobj = open(file,'r')
    for line in fobj.read().split("\n"):
        if line.startswith("#") or not line.strip(): continue
        this_key = ""
        for key in data.keys():
            if key in file: this_key = key
        linelist = re.split("\s+",line)
        data[this_key]["read"].append(int(linelist[3]))
        data[this_key]["write"].append(int(linelist[4]))
    for key in data.keys():
        if sum(data[key]["read"]) == 0 or sum(data[key]["write"]) == 0:
            del data[key]
    fobj.close()

        
outfile = open(os.path.join(outfolder,outfilename+event+outfilename_suffix),'w')

header = "# Reg Read_Min Read_Max Read_Mean Read_Median Read_Variance Write_Min Write_Max Write_Mean Write_Median Write_Variance"
outfile.write(header+"\n")
x_index = 0
for key in sorted(data.keys()):
    outlist = [key, str(x_index)]
    for subkey in sorted(data[key]):
        if not data.has_key(key): continue
        elif len(data[key][subkey]) == 0: 
            del data[key]
            x_index -= 1
            continue
        
        outlist.append(str(min(data[key][subkey])))
        outlist.append(str(max(data[key][subkey])))
        outlist.append(str(statistics.mean(data[key][subkey])))
        outlist.append(str(statistics.median(data[key][subkey])))
        outlist.append(str(statistics.variance(data[key][subkey])))
    outfile.write(" ".join(outlist)+"\n")
    x_index += 1
outfile.close()


script = open(os.path.join(scriptfolder,scriptname+event+scriptname_suffix),'w')
script.write("set title 'Comparision of counter access times'\n")
script.write("set terminal png;\n")
script.write("set output '../%s';\n" % (os.path.join(picfolder,outfilename+event+picfilename_suffix),))
script.write("set logscale y;\n")
script.write("set ylabel 'Cycles';\n")
script.write("set yrange [1:1000000];\n")
xtics_string = "set xtics ("
for i,key in enumerate(sorted(data.keys())):
    xtics_string += "'"+key+"' "+str(i+0.25)+","
script.write(xtics_string[:-1]+") rotate\n")
script.write("plot [-0.5:"+str(len(data.keys()))+"] '../"+outfolder+"/"+outfilename+event+outfilename_suffix+"'  using ($2+0.1):6:3:4 title 'Performance counter read' with errorbars, '../"+outfolder+"/"+outfilename+event+outfilename_suffix+"'  using ($2+0.4):11:8:9 title 'Performance counter write' with errorbars\n")
script.close()

