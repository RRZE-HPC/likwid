#!/home/rrze/unrz/unrz139/Work/likwid/trunk/ext/lua/lua

--[[
 * =======================================================================================
 *
 *      Filename:  likwid-perfctr.lua
 *
 *      Description:  An application to read out performance counter registers
 *                    on x86 processors
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Thomas Roehl
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================]]

VERSION = 4
RELEASE = 0
LIBLIKWIDPIN = "/usr/local/lib/liblikwidpin.so"
P6_FAMILY = 6

require("liblikwid")
local likwid = require("likwid")

HLINE = string.rep("-",80)
local SLINE = string.rep("*",80)

local function version()
    print(string.format("likwid-perfctr.lua --  Version %d.%d",VERSION,RELEASE))
end

local function examples()
    print("Examples:")
    print("Run command on CPU 2 and measure performance group TEST:")
    print("likwid-perfctr.lua -C 2 -g TEST ./a.out")
end

local function usage()
    version()
    print("A tool to read out performance counter registers on x86 processors\n")
    print("Options:")
    print("-h\t\t Help message")
    print("-v\t\t Version information")
    print("-V <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details)")
    print("-c <list>\t Processor ids to measure (required), e.g. 1,2-4,8")
    print("-C <list>\t Processor ids to pin threads and measure, e.g. 1,2-4,8")
    print("-g <string>\t Performance group or custom event set string")
    print("-H\t\t Get group help (together with -g switch)")
    print("-s <hex>\t Bitmask with threads to skip")
    print("-M <0|1>\t Set how MSR registers are accessed, 0=direct, 1=msrd")
    print("-a\t\t List available performance groups")
    print("-e\t\t List available events and counter registers")
    print("-i\t\t Print CPU info")
    print("Modes:")
    print("-S <time>\t Stethoscope mode with duration in s, ms or us, e.g 20ms")
    print("-t <time>\t Timeline mode with frequency in s, ms or us, e.g. 300ms")
    print("-m\t\t Use Marker API inside code")
    print("Output options:")
    print("-o <file>\t Store output to file. (Optional: Apply text filter according to filename suffix)")
    print("-O\t\t Output easily parseable CSV instead of fancy tables")
    print("\n")
    examples()
end


local config = likwid_getConfiguration();
verbose = 0
print_groups = false
print_events = false
print_info = false
cpulist = nil
num_cpus = 0
pin_cpus = false
group_string = nil
event_string = nil
event_string_list = {}
group_list = {}
group_ids = {}
print_group_help = false
skip_mask = "0x0"
counter_mask = {}
if config["daemonMode"] < 0 then
    access_mode = 1
else
    access_mode = config["daemonMode"]
end
use_marker = false
use_stethoscope = false
use_timeline = false
daemon_run = 0
use_wrapper = false
duration = 100000
use_switch_interval = false
switch_interval = 5
output = ""
use_csv = false
execString = nil
markerFile = string.format("/tmp/likwid_%d.txt",likwid_getpid("pid"))


for opt,arg in likwid.getopt(arg, "ac:C:eg:hHimM:o:OPs:S:t:vV:") do
    if (opt == "h") then
        usage()
        os.exit(0)
    elseif (opt == "v") then
        version()
        os.exit(0)
    elseif (opt == "V") then
        verbose = tonumber(arg)
        likwid_setVerbosity(verbose)
    elseif (opt == "c") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
    elseif (opt == "C") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
        pin_cpus = true
    elseif (opt == "a") then
        print_groups = true
    elseif (opt == "e") then
        print_events = true
    elseif (opt == "g") then
        table.insert(event_string_list, arg)
    elseif (opt == "H") then
        print_group_help = true
    elseif (opt == "s") then
        skip_mask = arg
    elseif (opt == "M") then
        access_mode = tonumber(arg)
        if (access_mode < 0 and access_mode > 1) then
            print("Access mode must be 0 for direct access and 1 for access daemon")
            os.exit(1)
        end
    elseif (opt == "i") then
        print_info = true
        verbose = true
    elseif (opt == "m") then
        use_marker = true
        use_wrapper = true
    elseif (opt == "S") then
        use_stethoscope = true
        duration = likwid.parse_time(arg)
    elseif (opt == "t") then
        use_timeline = true
        duration = likwid.parse_time(arg)
    elseif (opt == "o") then
        output = arg
        io.output(arg)
    elseif (opt == "O") then
        use_csv = true
    end
end

io.stdout:setvbuf("no")
cpuinfo = likwid_getCpuInfo()



if print_events == true then
    local tab = likwid_getEventsAndCounters()
    print(string.format("This architecture has %d counters.", #tab["Counters"]))
    local outstr = "Counters names: "
    for _, counter in pairs(tab["Counters"]) do
        outstr = outstr .. counter["Name"] .. " "
    end
    print(outstr)
    print(string.format("This architecture has %d events.",#tab["Events"]))
    print("Event tags (tag, id, umask, counters):")
    for _, eventTab in pairs(tab["Events"]) do
        outstr = eventTab["Name"] .. ", "
        outstr = outstr .. string.format("0x%X, 0x%X, ",eventTab["ID"],eventTab["UMask"])
        outstr = outstr .. eventTab["Limit"]
        if eventTab["Options"] > 0 then
            outstr = outstr .. ", " .. string.format("0x%X",eventTab["Options"])
        end
        print(outstr)
    end
    os.exit(0)
end

if print_groups == true then
    local num_groups, grouplist = likwid.get_groups(cpuinfo["short_name"])
    for i,g in pairs(grouplist) do
        local gdata = likwid.get_groupdata(cpuinfo["short_name"], g)
        print(string.format("%10s\t%s",g,gdata["ShortDescription"]))
    end
    os.exit(0)
end

if print_group_help == true then
    if #event_string_list == 0 then
        print("Group(s) must be given on commandline to get group help")
        os.exit(1)
    end
    for i,event_string in pairs(event_string_list) do
        local s,e = event_string:find(":")
        if s ~= nil then
            print("Given string is no group")
            os.exit(1)
        end
        local gdata = likwid.get_groupdata(cpuinfo["short_name"],event_string)
        print(string.format("Group %s:",event_string))
        print(gdata["LongDescription"])
    end
    os.exit(0)
end

print(HLINE)
print(string.format("CPU type:\t%s",cpuinfo["name"]))
if (cpuinfo["clock"] > 0) then
    print(string.format("CPU clock:\t%3.2f GHz",cpuinfo["clock"] * 1.E-09))
else
    print(string.format("CPU clock:\t%3.2f GHz",likwid_getCpuClock() * 1.E-09))
end
io.stdout:flush()

if verbose > 0 then
    print(string.format("CPU family:\t%u", cpuinfo["family"]))
    print(string.format("CPU model:\t%u", cpuinfo["model"]))
    print(string.format("CPU stepping:\t%u", cpuinfo["stepping"]))
    print(string.format("CPU features:\t%s", cpuinfo["features"]))
    if cpuinfo["family"] == P6_FAMILY and cpuinfo["perf_version"] > 0 then
        print(HLINE)
        print(string.format("PERFMON version:\t%u",cpuinfo["perf_version"]))
        print(string.format("PERFMON number of counters:\t%u",cpuinfo["perf_num_ctr"]))
        print(string.format("PERFMON width of counters:\t%u",cpuinfo["perf_width_ctr"]))
        print(string.format("PERFMON number of fixed counters:\t%u",cpuinfo["perf_num_fixed_ctr"]))
    end
end

if print_info then
    os.exit(0)
end

if num_cpus == 0 then
    print("Option -c <list> or -C <list> must be given on commandline")
    usage()
    os.exit(1)
end

if #event_string_list == 0 then
    print("Option(s) -g <string> must be given on commandline")
    usage()
    os.exit(1)
end

if num_cpus > 0 then
    for i,cpu1 in pairs(cpulist) do
        for j, cpu2 in pairs(cpulist) do
            if i ~= j and cpu1 == cpu2 then
                print("List of CPUs is not unique, got two times CPU " .. tostring(cpu1))
                os.exit(1)
            end
        end
    end
end

if use_stethoscope == false and use_timeline == false then
    use_wrapper = true
end

if use_wrapper == true and use_timeline == false and #event_string_list > 1 then
    use_timeline = true
end

if use_wrapper and likwid.tablelength(arg)-2 == 0 and print_info == false then
    print("No Executable can be found on commandline")
    usage()
    os.exit(0)
end



if pin_cpus then
    local omp_threads = os.getenv("OMP_NUM_THREADS")
    if omp_threads == nil then
        likwid_setenv("OMP_NUM_THREADS",tostring(num_cpus))
    end
    
    if num_cpus > 1 then
        local preload = os.getenv("LD_PRELOAD")
        local pinString = tostring(cpulist[2])
        for i=3,likwid.tablelength(cpulist) do
            pinString = pinString .. "," .. cpulist[i]
        end
        pinString = pinString .. "," .. cpulist[1]
        skipString = skip_mask

        likwid_setenv("KMP_AFFINITY","disabled")
        likwid_setenv("LIKWID_PIN", pinString)
        likwid_setenv("LIKWID_SKIP",skipString)
        likwid_setenv("LIKWID_SILENT","true")
        if preload == nil then
            likwid_setenv("LD_PRELOAD",LIBLIKWIDPIN)
        else
            likwid_setenv("LD_PRELOAD",LIBLIKWIDPIN .. ":" .. preload)
        end
    end
    likwid_pinProcess(cpulist[1], 1)
end



for i, event_string in pairs(event_string_list) do
    local s,e = event_string:find(":")
    gdata = nil
    if s == nil then
        table.insert(group_list,likwid.get_groupdata(cpuinfo["short_name"], event_string))
        event_string_list[i] = group_list[i]["EventString"]
    else
        table.insert(group_list, likwid.new_groupdata(event_string))
    end
end



if likwid_setAccessClientMode(access_mode) ~= 0 then
    os.exit(1)
end
if likwid_init(num_cpus, cpulist) < 0 then
    os.exit(1)
end

for i, event_string in pairs(event_string_list) do
    local gid = likwid_addEventSet(event_string)
    if gid < 0 then
        os.exit(1)
    end
    table.insert(group_ids, gid)
end

likwid_setupCounters(group_ids[1])
print(HLINE)

if use_timeline == true then
    local cores_string = "CORES: "
    for i, cpu in pairs(cpulist) do
        cores_string = cores_string .. tostring(cpu) .. " "
    end
    print(cores_string:sub(1,cores_string:len()-1))
    likwid_startDaemon(duration, switch_interval);
end

if use_wrapper or use_timeline then
    execString = table.concat(arg," ",1, likwid.tablelength(arg)-2)
    if verbose == true then
        print(string.format("Executing: %s",execString))
    end
end

if use_marker == true then
    likwid_setenv("LIKWID_FILEPATH", markerFile)
    likwid_setenv("LIKWID_MODE", tostring(access_mode))
    likwid_setenv("LIKWID_MASK", likwid.createBitMask(gdata))
    likwid_setenv("LIKWID_GROUPS", tostring(likwid_getNumberOfGroups()))
    local str = event_string
    likwid_setenv("LIKWID_EVENTS", str)
end

if use_wrapper or use_stethoscope then
    local ret = likwid_startCounters()
    if ret < 0 then
        print(string.format("Error starting counters for thread %d.",ret * (-1))
        os.exit(1)
    end
end
io.stdout:flush()
if use_wrapper or use_timeline then
    local err = os.execute(execString)
    if (err == false) then
        print("Failed to execute command: ".. exec)
    end
else
    usleep(duration)
end
io.stdout:flush()
if use_wrapper or use_stethoscope then
    local ret = likwid_stopCounters()
    if ret < 0 then
         print(string.format("Error stopping counters for thread %d.",ret * (-1))
        os.exit(1)
    end
elseif use_timeline then
    likwid_stopDaemon(9)
end

if use_marker == true then
    groups, results = likwid.getMarkerResults(markerFile)
    likwid.print_markerOutput(groups, results, gdata, cpulist)
elseif use_wrapper or use_stethoscope then
    for i, group in pairs(group_ids) do
        print(string.format("Group %d: %s", group, group_list[group]["GroupString"]))
        likwid.print_output(group, group_list[group], cpulist)
    end
end


likwid_finalize()

