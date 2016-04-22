#!<INSTALLED_BINPREFIX>/likwid-lua
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
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
 * =======================================================================================
]]

package.path = '<INSTALLED_PREFIX>/share/lua/?.lua;' .. package.path

local likwid = require("likwid")

local function version()
    print(string.format("likwid-perfctr --  Version %d.%d",likwid.version,likwid.release))
end

local function examples()
    print("Examples:")
    print("Run command on CPU 2 and measure performance group TEST:")
    print("likwid-perfctr -C 2 -g TEST ./a.out")
end

local function usage()
    version()
    print("A tool to read out performance counter registers on x86 processors\n")
    print("Options:")
    print("-h, --help\t\t Help message")
    print("-v, --version\t\t Version information")
    print("-V, --verbose <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print("-c <list>\t\t Processor ids to measure (required), e.g. 1,2-4,8")
    print("-C <list>\t\t Processor ids to pin threads and measure, e.g. 1,2-4,8")
    print("\t\t\t For information about the <list> syntax, see likwid-pin")
    print("-g, --group <string>\t Performance group or custom event set string")
    print("-H\t\t\t Get group help (together with -g switch)")
    print("-s, --skip <hex>\t Bitmask with threads to skip")
    print("-M <0|1>\t\t Set how MSR registers are accessed, 0=direct, 1=accessDaemon")
    print("-a\t\t\t List available performance groups")
    print("-e\t\t\t List available events and counter registers")
    print("-E <string>\t\t List available events and corresponding counters that match <string>")
    print("-i, --info\t\t Print CPU info")
    print("-T <time>\t\t Switch eventsets with given frequency")
    print("-f, --force\t\t Force overwrite of registers if they are in use")
    print("Modes:")
    print("-S <time>\t\t Stethoscope mode with duration in s, ms or us, e.g 20ms")
    print("-t <time>\t\t Timeline mode with frequency in s, ms or us, e.g. 300ms")
    print("-m, --marker\t\t Use Marker API inside code")
    print("Output options:")
    print("-o, --output <file>\t Store output to file. (Optional: Apply text filter according to filename suffix)")
    print("-O\t\t\t Output easily parseable CSV instead of fancy tables")
    print("--stats\t\t\t Always print statistics table")
    print("\n")
    examples()
end


local config = likwid.getConfiguration()
verbose = 0
print_groups = false
print_events = false
print_event = nil
print_info = false
cpulist = nil
num_cpus = 0
pin_cpus = false
group_string = nil
event_string = nil
event_string_list = {}
avail_groups = {}
num_avail_groups = 0
group_list = {}
group_ids = {}
activeGroup = 0
print_group_help = false
skip_mask = nil
counter_mask = {}
access_flags = "e"
if config["daemonMode"] < 0 then
    access_mode = 1
else
    access_mode = config["daemonMode"]
    if access_mode == 0 then
        access_flags = "rw"
    end
end
set_access_modes = false
use_marker = false
use_stethoscope = false
use_timeline = false
daemon_run = 0
use_wrapper = false
duration = 2.E06
switch_interval = 5
output = ""
use_csv = false
print_stats = false
execString = nil
outfile = nil
forceOverwrite = 0
gotC = false
markerFile = string.format("/tmp/likwid_%d.txt",likwid.getpid())
print_stdout = print
cpuClock = 1
likwid.catchSignal()

if #arg == 0 then
    usage()
    os.exit(0)
end

for opt,arg in likwid.getopt(arg, {"a", "c:", "C:", "e", "E:", "g:", "h", "H", "i", "m", "M:", "o:", "O", "P", "s:", "S:", "t:", "v", "V:", "T:", "f", "group:", "help", "info", "version", "verbose:", "output:", "skip:", "marker", "force", "stats"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-");
        if s == 1 then
            print_stdout(string.format("Argmument %s to option -%s starts with invalid character -.", arg, opt))
            print_stdout("Did you forget an argument to an option?")
            os.exit(1)
        end
    end
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
        likwid.setVerbosity(verbose)
    elseif (opt == "c") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
        gotC = true
    elseif (opt == "C") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
        pin_cpus = true
        gotC = true
    elseif (opt == "a") then
        print_groups = true
    elseif (opt == "e") then
        print_events = true
    elseif (opt == "E") then
        print_event = arg
    elseif opt == "f" or opt == "force" then
        forceOverwrite = 1
    elseif opt == "g" or opt == "group" then
        table.insert(event_string_list, arg)
    elseif (opt == "H") then
        print_group_help = true
    elseif opt == "s" or opt == "skip" then
        if arg:match("0x[0-9A-F]") then
            skip_mask = arg
        else
            if arg:match("[0-9A-F]") then
                print("Given skip mask looks like hex, sanitizing arg to 0x"..arg)
                skip_mask = "0x"..arg
            else
                print("Skip mask must be given in hex")
            end
        end
    elseif (opt == "M") then
        access_mode = tonumber(arg)
        set_access_modes = true
        if access_mode == 0 then
            access_flags = "rw"
        else
            access_flags = "e"
        end
        if (access_mode < 0 and access_mode > 1) then
            print_stdout("Access mode must be 0 for direct access and 1 for access daemon")
            os.exit(1)
        end
    elseif opt == "i" or opt == "info" then
        print_info = true
        verbose = true
    elseif opt == "m" or opt == "marker" then
        use_marker = true
        use_wrapper = true
    elseif (opt == "S") then
        use_stethoscope = true
        duration = likwid.parse_time(arg)
    elseif (opt == "t") then
        use_timeline = true
        duration = likwid.parse_time(arg)
    elseif (opt == "T") then
        duration = likwid.parse_time(arg)
    elseif opt == "o" or opt == "output" then
        local suffix = ""
        if string.match(arg, "%.") then
            suffix = string.match(arg, ".-[^\\/]-%.?([^%.\\/]*)$")
        end
        if suffix ~= "txt" then
            use_csv = true
        end
        outfile = arg:gsub("%%h", likwid.gethostname())
        outfile = outfile:gsub("%%p", likwid.getpid())
        outfile = outfile:gsub("%%j", likwid.getjid())
        outfile = outfile:gsub("%%r", likwid.getMPIrank())
        io.output(outfile..".tmp")
        print = function(...) for k,v in pairs({...}) do io.write(v .. "\n") end end
    elseif (opt == "O") then
        use_csv = true
    elseif (opt == "stats") then
        print_stats = true
    elseif opt == "?" then
        print("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print("Option requires an argument")
        os.exit(1)
    end
end

io.stdout:setvbuf("no")
cpuinfo = likwid.getCpuInfo()
cputopo = likwid.getCpuTopology()

if not likwid.msr_available(access_flags) then
    if access_mode == 1 then
        print_stdout("MSR device files not available")
        print_stdout("Please load msr kernel module before retrying")
        os.exit(1)
    else
        print_stdout("MSR device files not readable and writeable")
        print_stdout("Be sure that you have enough permissions to access the MSR files directly")
        os.exit(1)
    end
end

if num_cpus == 0 and
   not gotC and
   not print_events and
   print_event == nil and
   not print_groups and
   not print_group_help and
   not print_info then
    print_stdout("Option -c <list> or -C <list> must be given on commandline")
    usage()
    os.exit(1)
elseif num_cpus == 0 and
       gotC and
       not print_events and
       print_event == nil and
       not print_groups and
       not print_group_help and
       not print_info then
    print_stdout("CPUs given on commandline are not valid in current environment, maybe it's limited by a cpuset.")
    os.exit(1)
end


if num_cpus > 0 then
    for i,cpu1 in pairs(cpulist) do
        for j, cpu2 in pairs(cpulist) do
            if i ~= j and cpu1 == cpu2 then
                print_stdout("List of CPUs is not unique, got two times CPU " .. tostring(cpu1))
                os.exit(1)
            end
        end
    end
end



if print_events == true then
    local tab = likwid.getEventsAndCounters()
    print_stdout(string.format("This architecture has %d counters.", #tab["Counters"]))
    local outstr = "Counters names: "
    print_stdout("Counter tags(name, type<, options>):")
    for _, counter in pairs(tab["Counters"]) do
        outstr = string.format("%s, %s", counter["Name"], counter["TypeName"]);
        if counter["Options"]:len() > 0 then
            outstr = outstr .. string.format(", %s",counter["Options"])
        end
        print_stdout(outstr)
    end
    print_stdout("\n\n")
    print_stdout(string.format("This architecture has %d events.",#tab["Events"]))
    print_stdout("Event tags (tag, id, umask, counters<, options>):")
    for _, eventTab in pairs(tab["Events"]) do
        outstr = eventTab["Name"] .. ", "
        outstr = outstr .. string.format("0x%X, 0x%X, ",eventTab["ID"],eventTab["UMask"])
        outstr = outstr .. eventTab["Limit"]
        if #eventTab["Options"] > 0 then
            outstr = outstr .. string.format(", %s",eventTab["Options"])
        end
        print_stdout(outstr)
    end
    os.exit(0)
end

if print_event ~= nil then
    function case_insensitive_pattern(pattern)
        local p = pattern:gsub("(%%?)(.)", function(percent, letter)
            if percent ~= "" or not letter:match("%a") then
              return percent .. letter
            else
                return string.format("[%s%s]", letter:lower(), letter:upper())
            end
        end)
        return p
    end
    local tab = likwid.getEventsAndCounters()
    local events = {}
    local counters = {}
    local outstr = ""
    for _, eventTab in pairs(tab["Events"]) do
        if eventTab["Name"]:match(case_insensitive_pattern(print_event)) then
            table.insert(events, eventTab)
        end
    end
    for _, counter in pairs(tab["Counters"]) do
        for _, event in pairs(events) do
            if counter["Name"]:match(event["Limit"]) then
                counters[counter["Name"]] = counter
            end
        end
    end
    print_stdout(string.format("Found %d event(s) with search key %s:", #events, print_event))
    for _, eventTab in pairs(events) do
        outstr = eventTab["Name"] .. ", "
        outstr = outstr .. string.format("0x%X, 0x%X, ",eventTab["ID"],eventTab["UMask"])
        outstr = outstr .. eventTab["Limit"]
        if #eventTab["Options"] > 0 then
            outstr = outstr .. string.format(", %s",eventTab["Options"])
        end
        print_stdout(outstr)
    end
    print_stdout("\nUsable counter(s) for above event(s):")
    for i, counter in pairs(counters) do
        outstr = string.format("%s, %s", counter["Name"], counter["TypeName"]);
        if counter["Options"]:len() > 0 then
            outstr = outstr .. string.format(", %s",counter["Options"])
        end
        print_stdout(outstr)
    end
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(0)
end

avail_groups = likwid.getGroups()
if print_groups == true then
    print_stdout(string.format("%11s\t%s","Group name", "Description"))
    print_stdout(likwid.hline)
    for i,g in pairs(avail_groups) do
        print_stdout(string.format("%11s\t%s",g["Name"], g["Info"]))
    end
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(0)
end

if print_group_help == true then
    if #event_string_list == 0 then
        print_stdout("Group(s) must be given on commandline to get group help")
        os.exit(1)
    end
    for i,event_string in pairs(event_string_list) do
        local s,e = event_string:find(":")
        if s ~= nil then
            print_stdout("Given string is no group")
            os.exit(1)
        end
        for i,g in pairs(avail_groups) do
            if event_string == g["Name"] then
                print_stdout(string.format("Group %s:",g["Name"]))
                print_stdout(g["Long"])
            end
        end
    end
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(0)
end

if #event_string_list == 0 and not print_info then
    print_stdout("Option(s) -g <string> must be given on commandline")
    usage()
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(1)
end

if (cpuinfo["clock"] > 0) then
    cpuClock = cpuinfo["clock"]
else
    cpuClock = likwid.getCpuClock()
end

if outfile == nil then
    print_stdout(likwid.hline)
    print_stdout(string.format("CPU name:\t%s",cpuinfo["osname"]))
    print_stdout(string.format("CPU type:\t%s",cpuinfo["name"]))
    print_stdout(string.format("CPU clock:\t%3.2f GHz",cpuClock * 1.E-09))
end

if print_info or verbose > 0 then
    print_stdout(string.format("CPU family:\t%u", cpuinfo["family"]))
    print_stdout(string.format("CPU model:\t%u", cpuinfo["model"]))
    print_stdout(string.format("CPU short:\t%s", cpuinfo["short_name"]))
    print_stdout(string.format("CPU stepping:\t%u", cpuinfo["stepping"]))
    print_stdout(string.format("CPU features:\t%s", cpuinfo["features"]))
    P6_FAMILY = 6
    if cpuinfo["family"] == P6_FAMILY and cpuinfo["perf_version"] > 0 then
        print_stdout(likwid.hline)
        print_stdout(string.format("PERFMON version:\t%u",cpuinfo["perf_version"]))
        print_stdout(string.format("PERFMON number of counters:\t%u",cpuinfo["perf_num_ctr"]))
        print_stdout(string.format("PERFMON width of counters:\t%u",cpuinfo["perf_width_ctr"]))
        print_stdout(string.format("PERFMON number of fixed counters:\t%u",cpuinfo["perf_num_fixed_ctr"]))
    end
    print_stdout(likwid.hline)
    if print_info then
        likwid.printSupportedCPUs()
        likwid.putTopology()
        likwid.putConfiguration()
        os.exit(0)
    end
end

if use_stethoscope == false and use_timeline == false and use_marker == false then
    use_wrapper = true
end

if use_wrapper and likwid.tablelength(arg)-2 == 0 and print_info == false then
    print_stdout("No Executable can be found on commandline")
    usage()
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(0)
end

if use_marker then
    if likwid.access(markerFile, "rw") ~= -1 then
        print_stdout(string.format("ERROR: MarkerAPI file %s not accessible. Maybe a remaining file of another user.", markerFile))
        print_stdout("Please purge all MarkerAPI files from /tmp.")
        os.exit(1)
    end
    if not pin_cpus then
        print_stdout("Warning: The Marker API requires the application to run on the selected CPUs.")
        print_stdout("Warning: likwid-perfctr pins the application only when using the -C command line option.")
        print_stdout("Warning: LIKWID assumes that the application does it before the first instrumented code region is started.")
        print_stdout("Warning: You can use the string in the environment variable LIKWID_THREADS to pin you application to")
        print_stdout("Warning: to the CPUs specified after the -c command line option.")
    end
end

if verbose == 0 then
    likwid.setenv("LIKWID_SILENT","true")
end

if pin_cpus then
    local omp_threads = os.getenv("OMP_NUM_THREADS")
    if omp_threads == nil then
        likwid.setenv("OMP_NUM_THREADS",tostring(math.tointeger(num_cpus)))
    elseif num_cpus > tonumber(omp_threads) then
        print_stdout(string.format("Environment variable OMP_NUM_THREADS already set to %s but %d cpus required", omp_threads,num_cpus))
    end
    if os.getenv("CILK_NWORKERS") == nil then
        likwid.setenv("CILK_NWORKERS", tostring(math.tointeger(num_cpus)))
    end
    if skip_mask then
        likwid.setenv("LIKWID_SKIP",skip_mask)
    end
    likwid.setenv("KMP_AFFINITY","disabled")

    if num_cpus > 1 then
        local pinString = tostring(math.tointeger(cpulist[2]))
        for i=3,likwid.tablelength(cpulist) do
            pinString = pinString .. "," .. tostring(math.tointeger(cpulist[i]))
        end
        pinString = pinString .. "," .. tostring(math.tointeger(cpulist[1]))
        likwid.setenv("LIKWID_PIN", pinString)

        local preload = os.getenv("LD_PRELOAD")
        if preload == nil then
            likwid.setenv("LD_PRELOAD",likwid.pinlibpath)
        else
            likwid.setenv("LD_PRELOAD",likwid.pinlibpath .. ":" .. preload)
        end
    elseif num_cpus == 1 then
        likwid.setenv("LIKWID_PIN", tostring(math.tointeger(cpulist[1])))
        if verbose > 0 then
            likwid.pinProcess(cpulist[1], 0)
        else
            likwid.pinProcess(cpulist[1], 1)
        end
    end
end



--[[for i, event_string in pairs(event_string_list) do
    local groupdata = likwid.get_groupdata(event_string)
    if groupdata == nil then
        print_stdout("Cannot read event string, it's neither a performance group nor a proper event string <event>:<counter>:<options>,...")
        usage()
        likwid.putTopology()
        likwid.putConfiguration()
        os.exit(1)
    end
    table.insert(group_list, groupdata)
    event_string_list[i] = groupdata["EventString"]
end]]


if set_access_modes then
    if likwid.setAccessClientMode(access_mode) ~= 0 then
        likwid.putTopology()
        likwid.putConfiguration()
        os.exit(1)
    end
end
if likwid.init(num_cpus, cpulist) < 0 then
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(1)
end

likwid.setenv("LIKWID_FORCE", tostring(forceOverwrite))
for i, event_string in pairs(event_string_list) do
    if event_string:len() > 0 then
        local gid = likwid.addEventSet(event_string)
        if gid < 0 then
            likwid.putTopology()
            likwid.putConfiguration()
            likwid.finalize()
            os.exit(1)
        end
        table.insert(group_ids, gid)
    end
end
if #group_ids == 0 then
    print("ERROR: No valid eventset given on commandline. Exiting...")
    likwid.putTopology()
    likwid.putConfiguration()
    likwid.finalize()
    os.exit(1)
end

activeGroup = group_ids[1]
likwid.setupCounters(activeGroup)
if outfile == nil then
    print_stdout(likwid.hline)
end

if use_marker == true then
    likwid.setenv("LIKWID_FILEPATH", markerFile)
    likwid.setenv("LIKWID_MODE", tostring(access_mode))
    likwid.setenv("LIKWID_DEBUG", tostring(verbose))
    local str = table.concat(event_string_list, "|")
    likwid.setenv("LIKWID_EVENTS", str)
    likwid.setenv("LIKWID_THREADS", table.concat(cpulist,","))
    likwid.setenv("LIKWID_FORCE", "-1")
end

execString = table.concat(arg," ",1, likwid.tablelength(arg)-2)
if verbose == true then
    print_stdout(string.format("Executing: %s",execString))
end
local ldpath = os.getenv("LD_LIBRARY_PATH")
local libpath = likwid.pinlibpath:match("([/%g]+)/%g+.so")
if ldpath == nil then
    likwid.setenv("LD_LIBRARY_PATH", libpath)
elseif not ldpath:match(libpath) then
    likwid.setenv("LD_LIBRARY_PATH", libpath..":"..ldpath)
end


if use_timeline == true then
    local cores_string = "CORES: "
    for i, cpu in pairs(cpulist) do
        cores_string = cores_string .. tostring(cpu) .. "|"
    end
    io.stderr:write("# "..cores_string:sub(1,cores_string:len()-1).."\n")
    for gid, group in pairs(group_list) do
        local strlist = {}
        if group["Metrics"] == nil then
            for i,e in pairs(group["Events"]) do
                table.insert(strlist, e["Event"])
            end
        else
            for i,e in pairs(group["Metrics"]) do
                table.insert(strlist, e["description"])
            end
        end
        io.stderr:write("# "..table.concat(strlist, "|").."\n")
    end
end



io.stdout:flush()
local groupTime = {}
if use_wrapper or use_timeline then
    local start = likwid.startClock()
    local stop = 0
    local alltime = 0
    local nr_events = likwid.getNumberOfEvents(activeGroup)
    local nr_threads = likwid.getNumberOfThreads()
    local firstrun = true
    
    if use_wrapper and #group_ids == 1 then
        duration = 30.E06
    end

    local ret = likwid.startCounters()
    if ret < 0 then
        print_stdout(string.format("Error starting counters for cpu %d.",cpulist[ret * (-1)]))
        os.exit(1)
    end

    local pid = nil
    if pin_cpus then
        pid = likwid.startProgram(execString, #cpulist, cpulist)
    else
        pid = likwid.startProgram(execString, 0, cpulist)
    end

    if not pid then
        print_stdout("Failed to execute command: ".. execString)
    end
    start = likwid.startClock()
    groupTime[activeGroup] = 0
    while true do
        if likwid.getSignalState() ~= 0 then
            likwid.killProgram()
            break
        end
        local remain = likwid.sleep(duration)
        if remain > 0 or not likwid.checkProgram() then
            io.stdout:flush()
            break
        end
        if use_timeline == true then
            stop = likwid.stopClock()
            likwid.stopCounters()
            
            local time = likwid.getClock(start, stop)
            if likwid.getNumberOfMetrics(activeGroup) == 0 then
                results = likwid.getLastResults()
            else
                results = likwid.getLastMetrics()
            end
            str = tostring(math.tointeger(activeGroup)) .. " "..tostring(#results[activeGroup]).." "..tostring(#cpulist).." "..tostring(time)
            for i,l1 in pairs(results[activeGroup]) do
                for j, value in pairs(l1) do
                    str = str .. " " .. tostring(value)
                end
            end
            io.stderr:write(str.."\n")
            groupTime[activeGroup] = time
            likwid.startCounters()
        else
            likwid.readCounters()
        end
        if #group_ids > 1 then
            likwid.switchGroup(activeGroup + 1)
            activeGroup = likwid.getIdOfActiveGroup()
            if groupTime[activeGroup] == nil then
                groupTime[activeGroup] = 0
            end
            nr_events = likwid.getNumberOfEvents(activeGroup)
        end
        
    end
    stop = likwid.stopClock()
elseif use_stethoscope then
    local ret = likwid.startCounters()
    if ret < 0 then
        print_stdout(string.format("Error starting counters for cpu %d.",cpulist[ret * (-1)]))
        os.exit(1)
    end
    likwid.sleep(duration)
elseif use_marker then
    local ret = likwid.startCounters()
    if ret < 0 then
        print_stdout(string.format("Error starting counters for cpu %d.",cpulist[ret * (-1)]))
        os.exit(1)
    end
    local ret = os.execute(execString)
    if ret == nil then
        print_stdout("Failed to execute command: ".. execString)
    end
end

local ret = likwid.stopCounters()
if ret < 0 then
    print_stdout(string.format("Error stopping counters for thread %d.",ret * (-1)))
    likwid.finalize()
    likwid.putTopology()
    likwid.putConfiguration()
    os.exit(1)
end
io.stdout:flush()
if outfile == nil then
    print_stdout(likwid.hline)
end


if use_marker == true then
    results, metrics = likwid.getMarkerResults(markerFile, cpulist)
    if #results == 0 then
        print_stdout("No regions could be found in Marker API result file")
    else
        for r=1, #results do
            likwid.printOutput(results[r], metrics[r], cpulist, r, print_stats)
        end
    end
    os.remove(markerFile)
elseif use_timeline == false then
    results = likwid.getResults()
    metrics = likwid.getMetrics()
    likwid.printOutput(results, metrics, cpulist, nil, print_stats)
end

if outfile then
    local suffix = ""
    if string.match(outfile,"%.") then
        suffix = string.match(outfile, ".-[^\\/]-%.?([^%.\\/]*)$")
    end
    local command = "<INSTALLED_PREFIX>/share/likwid/filter/" .. suffix
    local tmpfile = outfile..".tmp"
    if suffix == "" then
        os.rename(tmpfile, outfile)
    elseif suffix ~= "txt" and suffix ~= "csv" and likwid.access(command, "x") then
        print_stdout("Cannot find filter script, save output in CSV format to file "..outfile)
        os.rename(tmpfile, outfile)
    else
        if suffix ~= "txt" and suffix ~= "csv" then
            command = command .." ".. tmpfile .. " perfctr"
            local f = assert(io.popen(command))
            if f ~= nil then
                local o = f:read("*a")
                if o:len() > 0 then
                    print_stdout(string.format("Failed to executed filter script %s.",command))
                end
            else
                print_stdout("Failed to call filter script, save output in CSV format to file "..outfile)
                os.rename(tmpfile, outfile)
                os.remove(tmpfile)
            end
        else
            os.rename(tmpfile, outfile)
            os.remove(tmpfile)
        end
    end
end

likwid.finalize()
likwid.putTopology()
likwid.putNumaInfo()
likwid.putConfiguration()
os.exit(0)
