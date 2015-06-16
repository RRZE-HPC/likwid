#!/usr/bin/env python

import os, sys, os.path
import re
import subprocess
import socket
import stat
import getopt

# Needed for Wiki page output
import glob
import statistics

bench_plain = "./likwid-bench-plain"
bench_marker = "./likwid-bench-marker"
bench_papi = "./likwid-bench-papi"
perfctr = "../../likwid-perfctr"
topology = "../../likwid-topology"
topology_type = re.compile("^CPU type:\s+(.*)")
topology_sockets = re.compile("^Sockets:\s+(\d+)")
topology_corespersocket = re.compile("^Cores per socket:\s+(\d+)")
topology_threadspercore = re.compile("^Threads per core:\s+(\d+)")
testlist = "SET.txt"
testfolder = "TESTS"
resultfolder = "RESULTS"
hostname = socket.gethostname()
picture_base = ".."

gnu_colors = ["red","blue","green","black"]#,"brown", "gray","violet", "cyan", "magenta","orange","#4B0082","#800000","turquoise","#006400","yellow"]
gnu_marks = [5,13,9,2]#,3,4,6,7,8,9,10,11,12,14,15]

wiki = False
papi = False
only_wiki = False
sets = []
out_pgf = False
out_gnuplot = False
out_grace = False
scriptfilename = "create_plots.sh"
out_script = False
test_set = {}
plain_set = {}
corrected_set = {}
marker_set = {}
papi_set = {}

if not os.path.exists(bench_plain) or not os.path.exists(bench_marker):
    print "Please run make before using likwid-accuracy.py"
    sys.exit(1)
if not os.path.exists(perfctr):
    print "Cannot find likwid-perfctr"
    sys.exit(1)


def usage():
    print "Execute and evaluate accuracy tests for LIKWID with likwid-bench and likwid-perfctr"
    print
    print "-h/--help:\tPrint this help text"
    print "-s/--sets:\tSpecifiy testgroups (comma separated). Can also be set in SET.txt"
#    print "--wiki:\t\tBesides testing write out results in Google code wiki syntax"
#    print "--only_wiki:\tDo not run benchmarks, read results from file and write out results in Google code wiki syntax"
    print "Picture options:"
    print "--pgf:\t\tCreate TeX document for each test with PGFPlot"
    print "--gnuplot:\tCreate GNUPlot script for each test"
    print "--grace:\tCreate Grace script that can be evaluated with gracebat"
    print "--script:\tActivate recording of commands in a bash script"
    print "--scriptname:\tRecord commands to create pictures in file (default: %s)" % (os.path.join(os.path.join(resultfolder,hostname),scriptfilename))

def get_system_info():
    name = None
    sockets = 0
    corespersocket = 0
    threadspercore = 0
    
    p = subprocess.Popen(topology, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode != 0:
        name = "Unknown system"
        return
    for line in p.stdout.read().split("\n"):
        if not line.strip() or line.startswith("*") or line.startswith("-"): continue
        if line.startswith("CPU type"):
            name = topology_type.match(line).group(1).strip()
        if line.startswith("Sockets"):
            sockets = int(topology_sockets.match(line).group(1))
        if line.startswith("Cores per socket"):
            corespersocket = int(topology_corespersocket.match(line).group(1))
        if line.startswith("Threads per core"):
            threadspercore = int(topology_threadspercore.match(line).group(1))
        if name and sockets > 0 and corespersocket > 0 and threadspercore > 0:
            break
    return name, sockets, corespersocket, threadspercore

def get_groups():
    groups = {}
    p = subprocess.Popen(perfctr+" -a", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    if p.returncode != 0:
        return groups
    for line in p.stdout.read().split("\n"):
        if line.startswith("-") or not line.strip(): continue
        if line.startswith("Available"): continue
        print(re.split("\s*", line.strip()))
        linelist = re.split("\s+", line.strip())
        name = linelist[0]
        description = " ".join(linelist[1:])
        groups[name] = description
    return groups

def get_test_groups(groupdict):
    groups = {}
    if len(sets) > 0:
        setlist = sets
    else:
        setfp = open("SET.txt",'r')
        setlist = setfp.read().strip().split("\n")
        setfp.close()
    
    filelist = glob.glob(testfolder+"/*.txt")
    for name in setlist:
        tests = []
        file = os.path.join(testfolder, name) + ".txt"
        if not os.path.exists(file): continue
        fp = open(file,'r')
        finput = fp.read().strip().split("\n")
        fp.close()    
        for line in finput:
            if line.startswith("TEST"):
                tests.append(line.split(" ")[1])
        groups[name] = tests
                
            
    return groups
    
def get_values_from_file(file, lineoffset, linecount):
    results = []
    fp = open(file,'r')
    finput = fp.read().strip().split("\n")
    fp.close()
    try:
        for line in finput[lineoffset:lineoffset+linecount]:
            results.append(float(line.split(" ")[1]))
    except:
        print "Cannot read file %s from %d to %d" % (file, lineoffset,lineoffset+linecount, )
        for line in finput[lineoffset:lineoffset+linecount]:
            print line
    return results

def write_pgf(group, test, plain_file, correct_file, marker_file, papi_file=None, execute=False, script=None):
    filename = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+".tex")
    fp = open(filename,'w')
    fp.write("\documentclass{article}\n")
    fp.write("\usepackage{pgfplots}\n")
    fp.write("\\begin{document}\n")
    fp.write("% cut from here\n")
    fp.write("\\begin{tikzpicture}\n")
    fp.write("\\begin{axis}[xlabel={Run}, ylabel={MFlops/s / MBytes/s},title={%s\_%s},legend pos=south east,xtick=data,width=.75\\textwidth]\n" % (group.replace("_","\_"),test.replace("_","\_"),))
    fp.write("\\addplot+[red,mark=square*,mark options={draw=red, fill=red}] table {%s};\n" % (os.path.basename(plain_file),))
    fp.write("\\addlegendentry{plain};\n")
    fp.write("\\addplot+[blue,mark=*,mark options={draw=blue, fill=blue}] table {%s};\n" % (os.path.basename(correct_file),))
    fp.write("\\addlegendentry{corrected};\n")
    fp.write("\\addplot+[green,mark=diamond*,mark options={draw=green, fill=green}] table {%s};\n" % (os.path.basename(marker_file),))
    fp.write("\\addlegendentry{marker};\n")
    if papi and papi_file:
        fp.write("\\addplot+[black,mark=triangle*,mark options={draw=black, fill=black}] table {%s};\n" % (os.path.basename(papi_file),))
        fp.write("\\addlegendentry{papi};\n")
    fp.write("\\end{axis}\n")
    fp.write("\\end{tikzpicture}\n")
    fp.write("% stop cutting here\n")
    fp.write("\\end{document}\n")
    fp.close()
    if execute:
        cmd = "cd %s && pdflatex %s && cd -" % (os.path.dirname(filename), os.path.basename(filename),)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        if p.returncode != 0:
            print p.stdout.read()
        p.stdout.close()
    if script:
        script.write("pdflatex %s\n" % (os.path.basename(filename),))
    return filename
    
def write_gnuplot(group, test, plain_file, correct_file, marker_file, papi_file=None, execute=False, script=None):
    filename = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+".plot")
    fp = open(filename,'w')
    for i,color in enumerate(gnu_colors):
        fp.write("set style line %d linetype 1 linecolor rgb '%s' lw 2 pt %s\n" % (i+1, color,gnu_marks[i]))
    fp.write("set terminal jpeg\n")
    fp.write("set title '%s_%s'\n" % (group, test,))
    fp.write("set output '%s'\n" % (os.path.basename(os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+".jpg")),))
    fp.write("set xlabel 'Size - %d runs each'\n" % (test_set[group][test]["RUNS"],))
    fp.write("set ylabel 'MFlops/s / MBytes/s'\n")
    #fp.write("set xtics 0,%d,%d\n" % (test_set[group][test]["RUNS"], test_set[group][test]["RUNS"]*len(test_set[group][test]["variants"]),))
    fp.write("set xtics %d\n" % (test_set[group][test]["RUNS"]*len(test_set[group][test]["variants"]),))
    for i,variant in enumerate(test_set[group][test]["variants"]):
        fp.write("set xtics add (\"%s\" %f)\n" % (variant, (i*test_set[group][test]["RUNS"])+(0.5*test_set[group][test]["RUNS"]),))
    plot_string = "plot '%s' using 1:2 title 'plain' with linespoints ls 1, \\\n '%s' using 1:2 title 'corrected' with linespoints ls 2, \\\n '%s' using 1:2 title 'marker' with linespoints ls 3" % (os.path.basename(plain_file), os.path.basename(correct_file), os.path.basename(marker_file),)
    if papi and papi_file:
        plot_string += ", \\\n '%s' using 1:2 title 'papi' with linespoints ls 4\n" % (os.path.basename(papi_file),)
    fp.write(plot_string+"\n")
    fp.close()
    if execute:
        cmd = "cd %s && gnuplot %s && cd -" % (os.path.dirname(filename), os.path.basename(filename),)
        print cmd
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        if p.returncode != 0:
            print p.stdout.read()
        p.stdout.close()
    if script:
        script.write("gnuplot %s\n" % (os.path.basename(filename),))
    return filename

def write_grace(group, test, plain_file, correct_file, marker_file, papi_file=None, execute=False, script=None):
    filename = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+".bat")
    agrname = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+".agr")
    pngname = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+".png")
    if execute or script:
        plain_file = os.path.basename(plain_file)
        marker_file = os.path.basename(marker_file)
        correct_file = os.path.basename(correct_file)
        if papi_file: papi_file = os.path.basename(papi_file)
        pngname = os.path.basename(pngname)
        agrname = os.path.basename(agrname)
    cmd_options = "-autoscale xy -nxy %s -nxy %s -nxy %s " % (plain_file, correct_file, marker_file,)
    if papi and papi_file:
        cmd_options += "-nxy %s " % (papi_file,)
    out_options = "-hdevice PNG -printfile %s " % (pngname,)
    out_options += "-saveall %s" % (agrname,)
    fp = open(filename,'w')
    fp.write("title \"%s_%s\"\n" % (group, test,))
    fp.write("xaxis label \"Run\"\n")
    fp.write("xaxis label char size 1.2\n")
    fp.write("xaxis ticklabel char size 1.2\n")
    fp.write("yaxis label \"MFlops/s / MBytes/s\"\n")
    fp.write("yaxis label char size 1.2\n")
    fp.write("yaxis ticklabel char size 1.2\n")
    fp.write("legend 0.8,0.7\n")
    fp.write("s0 legend \"plain\"\n")
    fp.write("s0 symbol 2\n")
    fp.write("s0 symbol size 1\n")
    fp.write("s0 symbol color 2\n")
    fp.write("s0 symbol pattern 1\n")
    fp.write("s0 symbol fill color 2\n")
    fp.write("s0 symbol fill pattern 1\n")
    fp.write("s0 symbol linewidth 2\n")
    fp.write("s0 symbol linestyle 1\n")
    fp.write("s0 line type 1\n")
    fp.write("s0 line color 2\n")
    fp.write("s0 line linestyle 1\n")
    fp.write("s0 line linewidth 2\n")
    fp.write("s0 line pattern 1\n")
    fp.write("s1 legend \"corrected\"\n")
    fp.write("s1 symbol 3\n")
    fp.write("s1 symbol size 1\n")
    fp.write("s1 symbol color 4\n")
    fp.write("s1 symbol pattern 1\n")
    fp.write("s1 symbol fill color 4\n")
    fp.write("s1 symbol fill pattern 1\n")
    fp.write("s1 symbol linewidth 2\n")
    fp.write("s1 symbol linestyle 1\n")
    fp.write("s1 line type 1\n")
    fp.write("s1 line color 4\n")
    fp.write("s1 line linestyle 1\n")
    fp.write("s1 line linewidth 2\n")
    fp.write("s1 line pattern 1\n")
    fp.write("s2 legend \"marker\"\n")
    fp.write("s2 symbol 4\n")
    fp.write("s2 symbol size 1\n")
    fp.write("s2 symbol color 3\n")
    fp.write("s2 symbol pattern 1\n")
    fp.write("s2 symbol fill color 3\n")
    fp.write("s2 symbol fill pattern 1\n")
    fp.write("s2 symbol linewidth 2\n")
    fp.write("s2 symbol linestyle 1\n")
    fp.write("s2 line type 1\n")
    fp.write("s2 line color 3\n")
    fp.write("s2 line linestyle 1\n")
    fp.write("s2 line linewidth 2\n")
    fp.write("s2 line pattern 1\n")
    if papi and papi_file:
        fp.write("s3 legend \"papi\"\n")
        fp.write("s3 symbol 5\n")
        fp.write("s3 symbol size 1\n")
        fp.write("s3 symbol color \"black\"\n")
        fp.write("s3 symbol pattern 1\n")
        fp.write("s3 symbol fill color \"black\"\n")
        fp.write("s3 symbol fill pattern 1\n")
        fp.write("s3 symbol linewidth 2\n")
        fp.write("s3 symbol linestyle 1\n")
        fp.write("s3 line type 1\n")
        fp.write("s3 line color \"black\"\n")
        fp.write("s3 line linestyle 1\n")
        fp.write("s3 line linewidth 2\n")
        fp.write("s3 line pattern 1\n")
    fp.close()
    if execute:
        cmd = "cd %s && gracebat %s -param %s %s && cd -" % (os.path.dirname(filename), cmd_options, os.path.basename(filename),out_options,)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        if p.returncode != 0:
            print p.stdout.read()
        p.stdout.close()
    if script:
        script.write("gracebat %s -param %s %s\n" % (cmd_options, os.path.basename(filename),out_options,))
    return filename

try:
    opts, args = getopt.getopt(sys.argv[1:], "hs:", ["help", "sets=","script","scriptname=","wiki","only_wiki","pgf","gnuplot","grace","papi"])
except getopt.GetoptError as err:
    print str(err)
    usage()
    sys.exit(2)

if len(opts) == 0:
    usage()
    sys.exit(1)

for o, a in opts:
    if o in ("-h","--help"):
        usage()
        sys.exit(0)
    if o == "--wiki":
        wiki = True
    if o == "--only_wiki":
        only_wiki = True
    if o == "--papi":
        papi = True
    if o == "--pgf":
        out_pgf = True
    if o == "--gnuplot":
        out_gnuplot = True
    if o == "--grace":
        out_grace = True
    if o in ("-s","--sets"):
        sets = a.split(",")
    if o == "--script":
        out_script = True
    if o == "--scriptname":
        scriptfilename = a

if len(sets) == 0 and not os.path.exists(testlist):
    print "Cannot find file %s containing list of testgroups" % (testlist,)
    sys.exit(1)
if not os.path.exists(testfolder):
    print "Cannot find folder %s containing the testgroups" % (testfolder,)
    sys.exit(1)


if len(sets) == 0:
    fp = open(testlist,'r')
    tmp = fp.read().split("\n")
    for item in tmp:
        if not item.strip() or item.startswith("#"): continue
        sets.append(item)
    fp.close()
for line in sets:
    if not line.strip() or line.startswith("#"): continue
    if os.path.exists("%s/%s.txt" % (testfolder,line.strip(),)):
        test_set[line.strip()] = {}
        plain_set[line.strip()] = {}
        corrected_set[line.strip()] = {}
        marker_set[line.strip()] = {}
        papi_set[line.strip()] = {}
        testfp = open("%s/%s.txt" % (testfolder,line.strip(),),'r')
        test = None
        for i,testline in enumerate(testfp.read().split("\n")):
            if test and not testline.strip(): test = None
            if testline.startswith("REGEX_BENCH"):
                test_set[line.strip()]["REGEX_BENCH"] = re.compile(" ".join(testline.split(" ")[1:]))
            if testline.startswith("REGEX_PERF"):
                test_set[line.strip()]["REGEX_PERF"] = re.compile(" ".join(testline.split(" ")[1:]))
            if testline.startswith("REGEX_PAPI"):
                test_set[line.strip()]["REGEX_PAPI"] = re.compile(" ".join(testline.split(" ")[1:]))
            if testline.startswith("TEST"):
                test = testline.split(" ")[1]
                test_set[line.strip()][test] = {}
                test_set[line.strip()][test]["WA_FACTOR"] = 1.0
                plain_set[line.strip()][test] = {}
                corrected_set[line.strip()][test] = {}
                marker_set[line.strip()][test] = {}
                papi_set[line.strip()][test] = {}
            if testline.startswith("RUNS") and test:
                test_set[line.strip()][test]["RUNS"] = int(testline.split(" ")[1])
            if testline.startswith("WA_FACTOR") and test:
                test_set[line.strip()][test]["WA_FACTOR"] = float(testline.split(" ")[1])
            if testline.startswith("VARIANT") and test:
                linelist = re.split("\s+",testline);
                variant = linelist[1]
                if not test_set[line.strip()][test].has_key("variants"):
                    test_set[line.strip()][test]["variants"] = []
                test_set[line.strip()][test][variant] = linelist[2]
                test_set[line.strip()][test]["variants"].append(linelist[1])
                plain_set[line.strip()][test][variant] = []
                corrected_set[line.strip()][test][variant] = []
                marker_set[line.strip()][test][variant] = []
                papi_set[line.strip()][test][variant] = []
        testfp.close()



if len(test_set.keys()) == 0:
    print "Cannot find any group in %s" % (testlist)
    sys.exit(1)

if not os.path.exists(resultfolder):
    os.mkdir(resultfolder)
if not os.path.exists(os.path.join(resultfolder,hostname)):
    os.mkdir(os.path.join(resultfolder,hostname))

if not only_wiki:
    scriptfile = os.path.join(os.path.join(resultfolder,hostname),scriptfilename)
    script = open(scriptfile,'w')
    script.write("#!/bin/bash\n")

    for group in test_set.keys():
        perfctr_string = "%s -c S0:0 -g %s -m " % (perfctr,group,)
        for test in test_set[group].keys():
            if test.startswith("REGEX"): continue
            file_plain = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_plain.dat")
            raw_plain = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_plain.raw")
            file_correct = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_correct.dat")
            file_marker = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_marker.dat")
            raw_marker = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_marker.raw")
            outfp_plain = open(file_plain,'w')
            rawfp_plain = open(raw_plain,'w')
            outfp_correct = open(file_correct,'w')
            outfp_marker = open(file_marker,'w')
            rawfp_marker = open(raw_marker,'w')
            if papi:
                file_papi = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_papi.dat")
                raw_papi = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_papi.raw")
                outfp_papi = open(file_papi,'w')
                rawfp_papi = open(raw_papi,'w')
            else:
                file_papi = None
                raw_papi = None
            counter = 1
            for size in test_set[group][test]["variants"]:
                if size.startswith("RUNS"): continue
                bench_options = "-t %s -w S0:%s:1" % (test, size,)
                for i in range(0,test_set[group][test]["RUNS"]):
                    # Run with plain likwid-bench
                    print "*",
                    sys.stdout.flush()
                    p = subprocess.Popen(bench_plain+" "+bench_options, shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    try:
                        p.wait()
                        stdout = p.stdout.read()
                        p.stdout.close()
                    except:
                        sys.exit(1)
                    for line in stdout.split("\n"):
                        #if p.returncode != 0: print line
                        match = test_set[group]["REGEX_BENCH"].match(line)
                        if match:
                            value = float(match.group(1)) * test_set[group][test]["WA_FACTOR"]
                            plain_set[group][test][size].append(match.group(1))
                            corrected_set[group][test][size].append(str(value))
                            outfp_plain.write(str(counter)+" "+match.group(1)+"\n")
                            outfp_correct.write(str(counter)+" "+str(value)+"\n")
                        rawfp_plain.write(line+"\n")
                    # Run with papi instrumented likwid-bench
                    if papi:
                        os.environ["PAPI_BENCH"] = str(group)
                        p = subprocess.Popen(bench_papi+" "+bench_options, shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                        try:
                            p.wait()
                            stdout = p.stdout.read()
                            p.stdout.close()
                        except:
                            sys.exit(1)
                        for line in stdout.split("\n"):
                            #if p.returncode != 0: print line
                            match = test_set[group]["REGEX_PAPI"].match(line)
                            if match:
                                papi_set[group][test][size].append(match.group(1))
                                outfp_papi.write(str(counter)+" "+match.group(1)+"\n")
                            rawfp_papi.write(line+"\n")
                    # Run with LIKWID instrumented likwid-bench and likwid-perfctr
                    p = subprocess.Popen(perfctr_string+" "+bench_marker+" "+bench_options, shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                    stdout = ""
                    try:
                        p.wait()
                        stdout = p.stdout.read()
                        p.stdout.close()
                    except:
                        sys.exit(1)
                    for line in stdout.split("\n"):
                        #if p.returncode != 0: print line
                        match = test_set[group]["REGEX_PERF"].match(line)
                        if match:
                            marker_set[group][test][size].append(float(match.group(1)))
                            outfp_marker.write(str(counter)+" "+str(float(match.group(1)))+"\n")
                        rawfp_marker.write(line+"\n")
                    counter += 1
                print("")
            outfp_plain.close()
            rawfp_plain.close()
            outfp_correct.close()
            outfp_marker.close()
            rawfp_marker.close()
            if papi:
                outfp_papi.close()
                rawfp_papi.close()
            if out_pgf: pgf_file = write_pgf(group, test, file_plain, file_correct, file_marker, file_papi, script=script)
            if out_gnuplot: plot_file = write_gnuplot(group, test, file_plain, file_correct, file_marker, file_papi, script=script)
            if out_grace: grace_file = write_grace(group, test, file_plain, file_correct, file_marker, file_papi, script=script)


    script.close()
    os.chmod(scriptfile, stat.S_IRWXU)
if only_wiki:
    for group in test_set.keys():
        for test in test_set[group].keys():
            if test.startswith("REGEX"): continue
            filename = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_plain.dat")
            for i,size in enumerate(test_set[group][test]["variants"]):
                start = i*test_set[group][test]["RUNS"]
                end = (i+1)*test_set[group][test]["RUNS"]
                runs = test_set[group][test]["RUNS"]
                print "Read file %s for size %s from %d to %d" % (filename,size, start, end,)
                plain_set[group][test][size] = get_values_from_file(filename, start, runs)
                if len(plain_set[group][test][size]) == 0: plain_set[group][test][size].append(0)
            filename = os.path.join(os.path.join(resultfolder,hostname),group+"_"+test+"_marker.dat")
            for i,size in enumerate(test_set[group][test]["variants"]):
                start = i*test_set[group][test]["RUNS"]
                end = (i+1)*test_set[group][test]["RUNS"]
                runs = test_set[group][test]["RUNS"]
                print "Read file %s for size %s from %d to %d" % (filename,size, start, end,)
                marker_set[group][test][size] = get_values_from_file(filename, start, runs)
                if len(marker_set[group][test][size]) == 0: marker_set[group][test][size].append(0)


if wiki or only_wiki:
    name, sockets, corespersocket, threadspercore = get_system_info();
    groups = get_groups()
    testable_groups = get_test_groups(groups)
    if testable_groups.has_key("FLOPS_DP"): del testable_groups["FLOPS_DP"]

    print "# Accuracy Tests for %s\n" % (name,)
    print "## Hardware description"
    print "Sockets: %d<br>" % (sockets,)
    print "Cores per socket: %d<br>" % (corespersocket,)
    print "Threads per core: %d<br>" % (threadspercore,)
    print "Total number of processing units: %d<br>" % (sockets * corespersocket * threadspercore)
    print
    print "## Available groups"
    print "Each architecture defines a different set of performance groups. These groups help users to measure their derived metrics. Besides the event and counter defintion, a performance groups contains derived metrics that are calculated based on the measured data.<br>Here all the groups available for the %s are listed:<br>\n" % (name,)
    print "| Name | Description |"
    print "| ---- | ----------- |"
    for grp in groups.keys():
        print "| %s | %s |" % (grp, groups[grp],)
    print
    print "## Available verification tests"
    print "Not all groups can be tested for accuracy. We don't have a test application for each performance group. Here only the groups are listed that can be verified. Each group is followed by the low-level benchmarks that are performed for comparison.<br>\n"
    #print testable_groups
    print "| Group | Tests |"
    print "|-------|-------|"
    for grp in testable_groups.keys():
        print "| %s | %s |" % (grp, ", ".join (testable_groups[grp]))
    print
    print "## Accuracy comparison"
    print "For each varification group, the tests are performed twice. Once in a plain manner without measuring but calculating the resulting values and once through an instumented code with LIKWID.<br>\n"
    
    
    for grp in testable_groups.keys():
        print "### Verification of Group %s" % (grp,)
        for test in testable_groups[grp]:
            #print grp, test, test_set[grp][test]
            print "#### Verification of Group %s with Test %s\n" % (grp, test,)
            print "| *Stream size* | *Iterations* |"
            print "|---------------|--------------|"
            for variant in test_set[grp][test]["variants"]:
                print "| %s | %s |" % (variant, test_set[grp][test][variant], )
            print 
            print "Each data size is tested %d times, hence the first %d entries on the x-axis correspond to the %d runs for the first data size of %s and so on.<br>\n" % (test_set[grp][test]["RUNS"],test_set[grp][test]["RUNS"],test_set[grp][test]["RUNS"],test_set[grp][test]["variants"][0],)
            print "%s/images/%s/%s_%s.png" % (picture_base,hostname, grp, test,)
            print
            file_plain = os.path.join(os.path.join(resultfolder,hostname),grp+"_"+test+"_plain.dat")
            file_marker = os.path.join(os.path.join(resultfolder,hostname),grp+"_"+test+"_marker.dat")
            print "| Variant | Plain (Min) | LIKWID (Min) | Plain (Max) | LIKWID (Max) | Plain (Avg) | LIKWID (Avg) |"
            print "| ------- | ------- | ------- | ------- | ------- | ------- | ------- |"
            for i, variant in enumerate(test_set[grp][test]["variants"]):
                results_plain = get_values_from_file(file_plain, i*test_set[grp][test]["RUNS"], test_set[grp][test]["RUNS"])
                results_correct = get_values_from_file(file_correct, i*test_set[grp][test]["RUNS"], test_set[grp][test]["RUNS"])
                results_marker = get_values_from_file(file_marker, i*test_set[grp][test]["RUNS"], test_set[grp][test]["RUNS"])
                if results_plain == []: results_plain.append(0)
                if results_marker == []: results_marker.append(0)
                if results_correct == []: results_correct.append(0)
                print "| %s | %d | %d | %d | %d | %d | %d |" % (variant, min(results_correct), min(results_marker), max(results_correct), max(results_marker), int(statistics.mean(results_correct)), int(statistics.mean(results_marker)),)
            print
            print
