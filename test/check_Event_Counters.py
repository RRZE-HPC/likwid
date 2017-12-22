#!/usr/bin/env/python3

# ===========================================================================================================
#
#      Filename:  check_Event_Counters.py
#
#      Description:  Check an event for each counter
#
# ===========================================================================================================

from subprocess import *
import os, re, csv, argparse, sys


event_counter = []
counters = []
eventlist = []
cwd = os.getcwd()
core = ['-c 1','-C 1','-c M:scatter','-C M:scatter','-c E:N:2','-C E:N:2','-c E:N:2:1:2','-C E:N:2:1:2']


# ===========================================================================================================
# Getting execution options

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help = "Location of the program")
parser.add_argument("-E", "--executable", help = "The executable to test")
args = parser.parse_args()

if args.executable:
	exe = args.executable
else:
	exe = "hostname"

if args.directory:
	if os.path.isdir(args.directory):
		PATH = args.directory
		if PATH[-1] != '/':
			PATH = PATH + '/'
	else:
		sys.exit("Exit with NON-Zero code: The given directory is not valid")
else:
	PATH = ""


# ===========================================================================================================
# Executing "likwid-perfctr -e" and storing the results in "outstr" and errors in "outerr"	

try:
	EventList = PATH + "likwid-perfctr -e"
	p = Popen(EventList, stdout = PIPE, stderr = PIPE, universal_newlines = True, shell = True)
	outstr, outerr = p.communicate()
	if p.returncode != 0:
		raise Exception
except Exception:
	print("unable to execute: %s" %EventList)
	outerr = "unable to execute: %s" %EventList

	
# ===========================================================================================================	
#	store the events and corresponding counters in a list of tuples named "event_counter"

lines = outstr.split('\n')
idx_counter = lines.index("")

for c in lines[2:idx_counter]:
	args = c.split(',')
	counters.append(args[0].strip())
for e in lines[idx_counter + 5:-1]:
	args = e.split(',')
	temp = args[3].split('|')
	if len(temp) > 1:
		for i in range(0,len(temp)):
			event_counter.append((args[0].strip(), temp[i].strip()))
	else:
		event_counter.append((args[0].strip(), temp[0].strip()))	

# ===========================================================================================================		
#	create a list of counters (one of each kind) in a list named "count_set"

count_set = [item for item in counters if "FIXC" in item]
for item in counters:
	if item[:3] not in ''.join(count_set):
		count_set.append(item)
temp = [entry for entry in counters if 'FIX' in entry and 'FIXC' not in entry]
count_set = count_set + temp

# ===========================================================================================================		
#	create a list containing commands to be called with "-g" in a list named "eventlist"
			
for count in count_set:
	for e in event_counter:
		if count.startswith(e[1]):
			eventlist.append(e[0]+':'+count.strip())
			break

# ===========================================================================================================
#	Extracting Header 

cmd = PATH + "likwid-perfctr -c 0 -g BRANCH -V 2 -f hostname"
p = Popen(cmd, stdout = PIPE, stderr = PIPE, universal_newlines = True, shell = True)
outstr, outerr = p.communicate()
if p.returncode == 0:
	divider = '-'*80
	header = re.search(r'%s\n.*?%s' %(divider, divider), outstr, re.DOTALL).group()
else:
	print("ERROR: Likwid-perfctr is not loaded.")
	exit(1)
# ===========================================================================================================
#	Execute Likwid commands in commandline and store the result in "out.txt"
#	and error messages in "err.txt"


with open(cwd + '/out.csv', 'w', newline = '') as out,\
	open(cwd + '/err.csv','w', newline = '') as err:
	outwr = csv.writer(out, delimiter = '\t',quoting = csv.QUOTE_NONE, quotechar = '', escapechar = '\t')
	errwr = csv.writer(err, quoting = csv.QUOTE_NONE, quotechar = '',escapechar = '\t')
	outwr.writerow([header, '\n'*3])
	errwr.writerow([header, '\n'*3])

	counter = 0
	
	for c in core[0:1]:
		EventList = PATH + "likwid-pin %s -p"%c
		p = Popen(EventList, stdout = PIPE, stderr = PIPE, universal_newlines = True, shell = True)
		pinOut, pinErr = p.communicate()

		for e in eventlist:
			cmd = PATH + "likwid-perfctr %s\t -g %s -V 2 -f %s" %(c, e, exe)
			print(cmd)
			p = Popen(cmd, stdout = PIPE, stderr = PIPE, universal_newlines = True, shell = True)
			outstr, outeImmarr = p.communicate()
			if p.returncode == 0:
				for i in outstr.split("\n"):
					if "Runtime" in i:
						outwr.writerow(['%-35s' %e,'%-10s' %c,'Exit 0','\n'])
						pinPrint = PATH + "likwid-pin %s -p:"%c
						out.write('%-35s %-10s\n'%(pinPrint, (pinOut + pinErr)))
						out.write(outstr[outstr.find('+'):outstr.rfind('+') + 1] + '\n\n')
						out.write(('#'*80 + '\n')*3 + '\n')
						break					
			else:
				outwr.writerow(['%-35s' %e,'%-10s' %c, 'Exit 1', '\n'])
				pinPrint = PATH + "likwid-pin %s -p:" %c
				err.write('%-35s %-10s\n' %(pinPrint, str(pinOut+pinErr)))
				errwr.writerow([outerr + '\n\n'])
				out.write(('#'*80 + '\n')*3 + '\n')


				counter += 1

# ===========================================================================================================
# Delete Error file if There exists no Error

if counter == 0:
	os.remove(cwd + "/err.csv", dir_fd = None)
