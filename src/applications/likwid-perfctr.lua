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
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

print_stdout = print
print_stderr = function(...)
    for k, v in pairs({ ... }) do io.stderr:write(v .. "\n") end
    io.stderr:flush()
end

local function version()
    print_stdout(string.format("likwid-perfctr -- Version %d.%d.%d (commit: %s)", likwid.version, likwid.release,
        likwid.minor, likwid.commit))
end

local function examples()
    print("Examples:")
    print("  List all performance groups:")
    print("    $ likwid-perfctr -a")
    print()
    print("  List all events and counters:")
    print("    $ likwid-perfctr -e")
    print()
    print("  List all events and suitable counters for events with 'L2' in them:")
    print("    $ likwid-perfctr -E L2")
    print()
    print("  Run command on CPU 2 and measure performance group CLOCK:")
    print("    $ likwid-perfctr -C 2 -g CLOCK ./a.out")
    if likwid.nvSupported() then
        print("  Run command and measure on GPU 1 the performance group FLOPS_DP (Only with NVMarkerAPI):")
        print("     $ likwid-perfctr -G 1 -W FLOPS_DP -m ./a.out")
        print()
        print("  It is possible to combine CPU and GPU measurements (with MarkerAPI and NVMarkerAPI):")
        print("    $ likwid-perfctr -C 2 -g CLOCK -G 1 -W FLOPS_DP -m ./a.out")
    end
    if likwid.rocmSupported() then
        print("  Run command and measure on GPU 1 the performance group PCI (Only with ROCmMarkerAPI):")
        print("    $ likwid-perfctr -I 1 -R PCI -m ./a.out")
        print("  It is possible to combine CPU and GPU measurements (with MarkerAPI and ROCmMarkerAPI):")
        print("    $ likwid-perfctr -C 2 -g CLOCK -I 1 -R PCI -m ./a.out")
    end
end

local function usage(config)
    version()
    print()
    print("A tool to read out performance counter registers on x86, ARM and POWER processors")
    print()
    print("General Options:")
    print("  -h, --help               Help message")
    print("  -v, --version            Version information")
    print("  -V, --verbose <level>    Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print("  -c <list>                Processor ids to measure (required), e.g. 1,2-4,8")
    print("  -C <list>                Processor ids to pin threads and measure, e.g. 1,2-4,8")
    print("                           For information about the <list> syntax, see likwid-pin")
    if likwid.nvSupported() then
        print("  -G, --gpus <list>        List of CUDA GPUs to monitor")
    end
    if likwid.rocmSupported() then
        print("  -I <list>                List of ROCm GPUs to monitor")
    end
    print("  -g, --group <string>     Performance group or custom event set string for CPU monitoring")
    if likwid.nvSupported() then
        print("  -W, --cudagroup <string> Performance group or custom event set string for GPU monitoring")
    end
    if likwid.rocmSupported() then
        print("  -R, --rocmgroup <string> Performance group or custom event set string for ROCm GPU monitoring")
    end
    print("  -H                       Get group help (together with -g switch)")
    print("  -s, --skip <hex>         Bitmask with threads to skip")
    print("  -M <0|1>                 Set how MSR registers are accessed, 0=direct, 1=accessDaemon")
    print("  -a                       List available performance groups")
    print("  -e                       List available events and counter registers")
    print("  -E <string>              List available events and corresponding counters that match <string>")
    print("  -i, --info               Print CPU info")
    print("  -T <time>                Switch eventsets with given frequency")
    print("  -f, --force              Force overwrite of registers if they are in use")
    print()
    print("Mode Options:")
    print("  -S <time>                Stethoscope mode with duration in s, ms or us, e.g 20ms")
    print("  -t <time>                Timeline mode with frequency in s, ms or us, e.g. 300ms")
    print("                           The output format (to stderr) is:")
    print("                             <groupID> <nrEvents> <nrThreads> <Timestamp> <Event1_Thread1> <Event1_Thread2> ... <EventN_ThreadN>")
    print("                           or")
    print("                             <groupID> <nrEvents> <nrThreads> <Timestamp> <Metric1_Thread1> <Metric1_Thread2> ... <MetricN_ThreadN>")
    print("  -m, --marker             Use Marker API inside code")
    print()
    print("Output options:")
    print("  -o, --output <file>      Store output to file. (Optional: Apply text filter according to filename suffix)")
    print("  -O                       Output easily parseable CSV instead of fancy tables")
    print("  --stats                  Always print statistics table")
    if config and config["daemonMode"] == -1 then
        print()
        print("perf_event specific options:")
        print("  --perfpid <pid>          Measure given PID")
        print("  --execpid                Use the PID of wrapped application for measurements")
    end
    print("")
    examples()
end

local function file_exists(path)
    local file = io.open(path, "r")
    if file ~= nil then
        io.close(file)
        return true
    else
        return false
    end
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
if config["daemonMode"] == 1 then
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
overflow_interval = 2.E06
output = ""
use_csv = false
print_stats = false
execString = nil
outfile = nil
outfile_orig = nil
outprefix = ""
forceOverwrite = 0
gotC = false
markerFolder = "/tmp"
markerFile = string.format("%s/likwid_%d.txt", markerFolder, likwid.getpid())
cpuClock = 1
execpid = false
local perf_paranoid = likwid.perf_event_paranoid()
if config["daemonMode"] == -1 then
    if perf_paranoid > 2 then
        print_stderr(string.format("Cannot use performance monitoring with perf_event_paranoid = %d", perf_paranoid))
        os.exit(1)
    elseif perf_paranoid > 0 then
        execpid = true
    end
end
perfflags = nil
perfpid = nil
nan2value = '-'
cputopo = nil
cpuinfo = nil
cliopts = { "a", "c:", "C:", "e", "E:", "g:", "h", "H", "i", "m", "M:", "o:", "O", "P", "s:", "S:", "t:", "v", "V:",
    "T:", "f", "group:", "help", "info", "version", "verbose:", "output:", "skip:", "marker", "force", "stats",
    "execpid", "perfflags:", "perfpid:", "Z", "outprefix:" }


---------------------------
nvSupported = likwid.nvSupported()
num_cuda_gpus = 0
gpulist_cuda = {}
cuda_event_string_list = {}
nvMarkerFile = string.format("%s/likwid_gpu_%d.txt", markerFolder, likwid.getpid())
gotCudaG = false
cudagroups = {}
cudatopo = nil
if nvSupported then
    table.insert(cliopts, "W:")
    table.insert(cliopts, "G:")
    table.insert(cliopts, "cudagroup:")
end
---------------------------
rocmSupported = likwid.rocmSupported()
gpulist_rocm = {}
rocm_event_string_list = {}
rocmMarkerFile = string.format("%s/likwid_rocm_%d.txt", markerFolder, likwid.getpid())
gotRocmG = false
rocmgroups = {}
rocmInitialized = false
rocmtopo = nil
if rocmSupported then
    table.insert(cliopts, "I:")
    table.insert(cliopts, "R:")
    table.insert(cliopts, "rocmgroup:")
end
---------------------------

likwid.catchSignal()

local function perfctr_exit(exitcode)
    if likwid.access(nvMarkerFile, "e") == 0 then
        os.remove(nvMarkerFile)
    end
    if likwid.access(markerFile, "e") == 0 then
        os.remove(markerFile)
    end
    if rocmSupported then
        if rocmInitialized then
            likwid.finalize_rocm()
            rocmInitialized = false
            rocmgroups = {}
            rocm_event_string_list = {}
            gpulist_rocm = {}
        end
        if likwid.access(rocmMarkerFile, "e") == 0 then
            os.remove(rocmMarkerFile)
        end
    end
    if cputopo then
        likwid.putTopology()
        cputopo = nil
        cpuinfo = nil
    end
    if cudatopo then
        likwid.putCudaTopology()
        cudatopo = nil
    end
    if rocmtopo then
        likwid.putRocmTopology()
        rocmtopo = nil
    end
    if config then
        likwid.putConfiguration()
        config = nil
    end
    os.exit(exitcode)
end

if #arg == 0 then
    usage(config)
    perfctr_exit(0)
end

for opt, arg in likwid.getopt(arg, cliopts) do
    if (type(arg) == "string") then
        local s, e = arg:find("-");
        if s == 1 then
            print_stderr(string.format("Argument %s to option -%s starts with invalid character -.", arg, opt))
            print_stderr("Did you forget an argument to an option?")
            perfctr_exit(1)
        end
    end
    if opt == "h" or opt == "help" then
        usage(config)
        perfctr_exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        perfctr_exit(0)
    elseif opt == "V" or opt == "verbose" then
        if arg ~= nil and tonumber(arg) ~= nil then
            verbose = tonumber(arg)
            likwid.setVerbosity(verbose)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
    elseif (opt == "c") then
        if arg ~= nil then
            num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
        gotC = true
    elseif (opt == "C") then
        if arg ~= nil then
            num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
        pin_cpus = true
        gotC = true
    elseif (opt == "a") then
        print_groups = true
    elseif (opt == "Z") then
        nan2value = 0
    elseif (opt == "e") then
        print_events = true
    elseif (opt == "execpid") then
        execpid = true
    elseif (opt == "perfflags") then
        perfflags = arg
    elseif (opt == "perfpid") then
        if likwid.perfctr_checkpid(arg) then
            perfpid = arg
            execpid = false
        end
    elseif (opt == "outprefix") then
        outprefix = arg
    elseif (opt == "E") then
        if arg ~= nil then
            print_event = arg
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
    elseif opt == "f" or opt == "force" then
        forceOverwrite = 1
    elseif opt == "g" or opt == "group" then
        if arg ~= nil then
            table.insert(event_string_list, arg)
        end
    elseif (opt == "H") then
        print_group_help = true
    elseif opt == "s" or opt == "skip" then
        if arg:match("0x[0-9A-Fa-f]") then
            skip_mask = arg
        else
            if arg:match("[0-9A-Fa-f]") then
                print_stderr("Given skip mask looks like hex, sanitizing arg to 0x"..arg)
                skip_mask = "0x"..arg
            else
                print_stderr("Skip mask must be given in hex")
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
            perfctr_exit(1)
        end
    elseif opt == "i" or opt == "info" then
        print_info = true
        verbose = true
    elseif opt == "m" or opt == "marker" then
        use_marker = true
        --use_wrapper = true
    elseif (opt == "S") then
        use_stethoscope = true
        if arg ~= nil and arg:match("%d+%a?s") then
            duration = likwid.parse_time(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
    elseif (opt == "t") then
        use_timeline = true
        if arg ~= nil and arg:match("%d+%a?s") then
            duration = likwid.parse_time(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
    elseif (opt == "T") then
        if arg ~= nil and arg:match("%d+%a?s") then
            duration = likwid.parse_time(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
    elseif opt == "o" or opt == "output" then
        local suffix = ""
        if string.match(arg, "%.") then
            suffix = string.match(arg, ".-[^\\/]-%.?([^%.\\/]*)$")
        end
        if suffix ~= "txt" then
            use_csv = true
        end
        outfile_orig = arg
        outfile = arg:gsub("%%h", likwid.gethostname())
        outfile = outfile:gsub("%%p", likwid.getpid())
        outfile = outfile:gsub("%%j", likwid.getjid())
        outfile = outfile:gsub("%%r", likwid.getMPIrank())
        io.output(outfile)
        print = function(...) for k, v in pairs({ ... }) do io.write(v .. "\n") end end
    elseif (opt == "O") then
        use_csv = true
    elseif (opt == "stats") then
        print_stats = true
        ---------------------------
    elseif nvSupported and (opt == "G") then
        if arg ~= nil then
            num_cuda_gpus, gpulist_cuda = likwid.gpustr_to_gpulist_cuda(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
        gotCudaG = true
    elseif nvSupported and (opt == "W" or opt == "cudagroup") then
        if arg ~= nil then
            table.insert(cuda_event_string_list, arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
        ---------------------------
    elseif rocmSupported and (opt == "I") then
        if arg ~= nil then
            gpulist_rocm = likwid.gpustr_to_gpulist_rocm(arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
        gotCudaG = true
    elseif rocmSupported and (opt == "R" or opt == "rocmgroup") then
        if arg ~= nil then
            table.insert(rocm_event_string_list, arg)
        else
            print_stderr("Option requires an argument")
            perfctr_exit(1)
        end
        ---------------------------
    elseif opt == "?" then
        print_stderr("Invalid commandline option -" .. arg)
        perfctr_exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        perfctr_exit(1)
    end
end
local execList = {}
for i = 1, likwid.tablelength(arg) - 2 do
    if string.find(arg[i], " ") then
        table.insert(execList, "\"" .. arg[i] .. "\"")
    else
        table.insert(execList, arg[i])
    end
end

if perfpid and (not execpid) and (not cpulist) then
    local rawlist = likwid.perfctr_pid_cpulist(perfpid)
    if rawlist then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(rawlist)
    end
end

io.stdout:setvbuf("no")
cpuinfo = likwid.getCpuInfo()
cputopo = likwid.getCpuTopology()
---------------------------
if nvSupported then
    cudatopo = likwid.getCudaTopology()
end
---------------------------
if rocmSupported then
    rocmtopo = likwid.getRocmTopology()
end

if num_cpus == 0 and
    not gotC and
    not print_events and
    print_event == nil and
    not print_groups and
    not print_group_help and
    not print_info then
    cpulist = {}
    pin_cpus = false
    for cntr = 0, cputopo["numHWThreads"] - 1 do
        if cputopo["threadPool"][cntr]["inCpuSet"] == 1 then
            num_cpus = num_cpus + 1
            table.insert(cpulist, cputopo["threadPool"][cntr]["apicId"])
        end
    end
elseif num_cpus == 0 and
    gotC and
    not print_events and
    print_event == nil and
    not print_groups and
    not print_group_help and
    not print_info then
    print_stderr("CPUs given on commandline are not valid in current environment, maybe it's limited by a cpuset.")
    perfctr_exit(1)
end

if use_timeline and outfile then
    print_stderr("Redirecting output in timeline mode not supported")
    perfctr_exit(1)
end

---------------------------
if nvSupported and
    num_cuda_gpus == 0 and
    not gotCudaG and
    cudatopo and
    not print_events and
    print_event == nil and
    not print_groups and
    not print_group_help and
    not print_info then
    newgpulist = {}
    for g = 1, cudatopo["numDevices"] do
        num_cuda_gpus = num_cuda_gpus + 1
        table.insert(newgpulist, cudatopo["devices"][g]["id"])
    end
    gpulist_cuda = newgpulist
end
---------------------------
if rocmSupported and
    not gotRocmG and
    rocmtopo and
    not print_events and
    print_event == nil and
    not print_groups and
    not print_group_help and
    not print_info then
    newrocmlist = {}
    for _, device in pairs(rocmtopo.devices) do
        table.insert(newrocmlist, device.id)
    end
    gpulist_rocm = newrocmlist
end
---------------------------

if num_cpus > 0 then
    for i, cpu1 in pairs(cpulist) do
        for j, cpu2 in pairs(cpulist) do
            if i ~= j and cpu1 == cpu2 then
                print_stderr("List of CPUs is not unique, got two times CPU " .. tostring(cpu1))
                perfctr_exit(1)
            end
        end
    end
end

---------------------------
if nvSupported and cudatopo and num_cuda_gpus > 0 then
    for i, gpu1 in pairs(gpulist_cuda) do
        for j, gpu2 in pairs(gpulist_cuda) do
            if i ~= j and gpu1 == gpu2 then
                print_stderr("List of GPUs is not unique, got two times GPU " .. tostring(gpu1))
                perfctr_exit(1)
            end
        end
    end
end
---------------------------
if rocmSupported and rocmtopo and #gpulist_rocm > 0 then
    for i, gpu1 in pairs(gpulist_rocm) do
        for j, gpu2 in pairs(gpulist_rocm) do
            if i ~= j and gpu1 == gpu2 then
                print_stderr("List of GPUs is not unique, got two times GPU " .. tostring(gpu1))
                perfctr_exit(1)
            end
        end
    end
end
---------------------------

if print_events == true then
    local tab = likwid.getEventsAndCounters()
    print_stdout(string.format("This architecture has %d counters.", #tab["Counters"]))
    local outstr = "Counters names: "
    print_stdout("Counter tags(name, type<, options>):")
    for _, counter in pairs(tab["Counters"]) do
        outstr = string.format("%s, %s", counter["Name"], counter["TypeName"]);
        if counter["Options"]:len() > 0 then
            outstr = outstr .. string.format(", %s", counter["Options"])
        end
        print_stdout(outstr)
    end
    print_stdout("\n\n")
    print_stdout(string.format("This architecture has %d events.", #tab["Events"]))
    print_stdout("Event tags (tag, id, umask, counters<, options>):")
    for _, eventTab in pairs(tab["Events"]) do
        outstr = eventTab["Name"] .. ", "
        outstr = outstr .. string.format("0x%X, 0x%X, ", eventTab["ID"], eventTab["UMask"])
        outstr = outstr .. eventTab["Limit"]
        if #eventTab["Options"] > 0 then
            outstr = outstr .. string.format(", %s", eventTab["Options"])
        end
        print_stdout(outstr)
    end
    ---------------------------
    if nvSupported and cudatopo then
        local cudahome = os.getenv("CUDA_HOME")
        if cudahome and cudahome:len() > 0 then
            ldpath = os.getenv("LD_LIBRARY_PATH")
            local cuptilib = string.format("%s/extras/CUPTI/lib64", cudahome)
            likwid.setenv("LD_LIBRARY_PATH", cuptilib .. ":" .. ldpath)
        end
        -- we need the gpulist to initialize nvmon
        newgpulist = {}
        for g = 1, cudatopo["numDevices"] do
            num_cuda_gpus = num_cuda_gpus + 1
            table.insert(newgpulist, cudatopo["devices"][g]["id"])
        end
        gpulist_cuda = newgpulist
        -- nvmon has be initialized to initialize nvml which provides the smi events
        if likwid.nvInit(num_cuda_gpus, gpulist_cuda) < 0 then
            perfctr_exit(1)
        end
        tab = likwid.getCudaEventsAndCounters()
        for d = 0, tab["numDevices"], 1 do
            if tab["devices"][d] then
                print_stdout("\n\n")
                print_stdout(string.format("The GPUs %d provides %d events.", d, #tab["devices"][d]))
                print_stdout("You can use as many GPUx counters until you get an error.")
                print_stdout("Event tags (tag, counters)")
                for _, e in pairs(tab["devices"][d]) do
                    outstr = string.format("%s, %s", e["Name"], e["Limit"])
                    print_stdout(outstr)
                end
            end
        end
    end
    ---------------------------
    if rocmSupported and rocmtopo then
        local rocmhome = os.getenv("ROCM_HOME")
        if rocmhome and rocmhome:len() > 0 then
            ldpath = os.getenv("LD_LIBRARY_PATH")
            local libpathlist = {}
            table.insert(libpathlist, string.format("%s/hip/lib", rocmhome))
            table.insert(libpathlist, string.format("%s/hsa/lib", rocmhome))
            table.insert(libpathlist, string.format("%s/lib/rocprofiler", rocmhome))
            table.insert(libpathlist, string.format("%s/rocprofiler/lib", rocmhome))
            table.insert(libpathlist, string.format("%s/lib", rocmhome))
            table.insert(libpathlist, ldpath)
            likwid.setenv("LD_LIBRARY_PATH", table.concat(libpathlist, ":"))
            print_stdout(os.getenv("LD_LIBRARY_PATH"))
            --likwid.setenv("HSA_TOOLS_LIB", "librocprofiler64.so")
            likwid.setenv("HSA_TOOLS_LIB", "librocprof-tool.so")
        end
        likwid.init_rocm({})
        rocmInitialized = true
        tab = likwid.getRocmEventsAndCounters()
        for _, device in pairs(tab.devices) do
            print_stdout("\n\n")
            print_stdout(string.format("The ROCM GPU '%d' provides %d events.", device.gpuId, #device.events))
            print_stdout("You can use as many ROCMx counters until you get an error.")
            print_stdout("Event names")
            for _, event in pairs(device.events) do
                print_stdout(event.name)
            end
        end
    end
    ---------------------------
    perfctr_exit(0)
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
    ---------------------------
    if nvSupported and cudatopo then
        local cudahome = os.getenv("CUDA_HOME")
        if cudahome and cudahome:len() > 0 then
            ldpath = os.getenv("LD_LIBRARY_PATH")
            local cuptilib = string.format("%s/extras/CUPTI/lib64", cudahome)
            likwid.setenv("LD_LIBRARY_PATH", cuptilib .. ":" .. ldpath)
        end
        if cudahome then
            tab = likwid.getGpuEventsAndCounters()
            for d = 0, tab["numDevices"] - 1, 1 do
                for _, e in pairs(tab["devices"][d]) do
                    if e["Name"]:match(case_insensitive_pattern(print_event)) then
                        local f = false
                        for _, x in pairs(events) do
                            if e["Name"] == x["Name"] then
                                f = true
                                break
                            end
                        end
                        if not f then
                            table.insert(events, e)
                            counters["GPU"] = { ["Name"] = "GPU", ["TypeName"] = "Nvidia GPU counters" }
                        end
                    end
                end
            end
        end
    end
    ---------------------------
    if rocmSupported and rocmtopo then
        local rocmhome = os.getenv("ROCM_HOME")
        if rocmhome and rocmhome:len() > 0 then
            ldpath = os.getenv("LD_LIBRARY_PATH")
            local hiplib = string.format("%s/hip/lib", rocmhome)
            local hsalib = string.format("%s/hsa/lib", rocmhome)
            local rocproflib = string.format("%s/lib/rocprofiler", rocmhome)
            likwid.setenv("LD_LIBRARY_PATH", hiplib .. ":" .. hsalib .. ":" .. rocproflib .. ":" .. ldpath)
            likwid.setenv("HSA_TOOLS_LIB", "librocprofiler64.so")
        end
        if rocmhome then
            tab = likwid.getGpuEventsAndCounters_rocm()
            for d = 0, tab["numDevices"] - 1, 1 do
                for _, e in pairs(tab["devices"][d]) do
                    if e["Name"]:match(case_insensitive_pattern(print_event)) then
                        local f = false
                        for _, x in pairs(events) do
                            if e["Name"] == x["Name"] then
                                f = true
                                break
                            end
                        end
                        if not f then
                            table.insert(events, e)
                            counters["ROCM"] = { ["Name"] = "ROCM", ["TypeName"] = "ROCM GPU counters" }
                        end
                    end
                end
            end
        end
    end
    ---------------------------
    print_stdout(string.format("Found %d event(s) with search key %s:", #events, print_event))
    for _, eventTab in pairs(events) do
        outstr = eventTab["Name"] .. ", "
        if (eventTab["ID"] and eventTab["UMask"]) then
            outstr = outstr .. string.format("0x%X, 0x%X, ", eventTab["ID"], eventTab["UMask"])
        end
        outstr = outstr .. eventTab["Limit"]
        if eventTab["Options"] and #eventTab["Options"] > 0 then
            outstr = outstr .. string.format(", %s", eventTab["Options"])
        end
        print_stdout(outstr)
    end
    print_stdout("\nUsable counter(s) for above event(s):")
    for i, counter in pairs(counters) do
        outstr = string.format("%s, %s", counter["Name"], counter["TypeName"]);
        if counter["Options"] and counter["Options"]:len() > 0 then
            outstr = outstr .. string.format(", %s", counter["Options"])
        end
        print_stdout(outstr)
    end
    likwid.putTopology()
    likwid.putConfiguration()
    perfctr_exit(0)
end

avail_groups = likwid.getGroups()
if print_groups == true then
    if avail_groups then
        local max_len = 0
        for i, g in pairs(avail_groups) do
            if g["Name"]:len() > max_len then max_len = g["Name"]:len() end
        end
        local s = string.format("%%%ds\t%%s", max_len)
        if nvSupported and cudatopo then
            print_stdout(string.format(s, "PerfMon group name", "Description"))
        else
            print_stdout(string.format(s, "Group name", "Description"))
        end
        print_stdout(likwid.hline)
        for i, g in pairs(avail_groups) do
            print_stdout(string.format(s, g["Name"], g["Info"]))
        end
    else
        print_stdout(string.format("No groups defined for %s", cpuinfo["name"]))
    end
    ---------------------------
    if nvSupported and cudatopo then
        avail_groups = likwid.getCudaGroups(0)
        if avail_groups then
            local max_len = 0
            for i, g in pairs(avail_groups) do
                if g["Name"]:len() > max_len then max_len = g["Name"]:len() end
            end
            local s = string.format("%%%ds\t%%s", max_len)
            print_stdout(string.format(s, "\nNvMon group name", "Description"))
            print_stdout(likwid.hline)
            for i, g in pairs(avail_groups) do
                print_stdout(string.format(s, g["Name"], g["Info"]))
            end
        else
            print_stdout(string.format("No groups defined for %s", cudatopo["devices"][1]["name"]))
        end
    end
    ---------------------------
    if rocmSupported and rocmtopo then
        avail_groups = likwid.getRocmGroups()
        if #avail_groups > 0 then
            local max_len = 0
            for i, g in pairs(avail_groups) do
                max_len = math.max(max_len, g.name:len())
            end
            local s = string.format("%%%ds\t%%s", max_len)
            print_stdout(string.format(s, "\nRocMon group name", "Description"))
            print_stdout(likwid.hline)
            for i, g in pairs(avail_groups) do
                print_stdout(string.format(s, g.name, g.shortInfo))
            end
        else
            print_stdout(string.format("No groups defined for %s", rocmtopo["devices"][1]["name"]))
        end
    end
    ---------------------------
    perfctr_exit(0)
end

if print_group_help == true then
    if #event_string_list == 0 then
        print_stdout("Group(s) must be given on commandline to get group help")
        perfctr_exit(1)
    end
    for i, event_string in pairs(event_string_list) do
        local s, e = event_string:find(":")
        if s ~= nil then
            print_stdout("Given string is no group")
            perfctr_exit(1)
        end
        for i, g in pairs(avail_groups) do
            if event_string == g["Name"] then
                print_stdout(string.format("Group %s:", g["Name"]))
                print_stdout(g["Long"])
            end
        end
    end
    perfctr_exit(0)
end

if #event_string_list == 0 and #cuda_event_string_list == 0 and #rocm_event_string_list == 0 and not print_info then
    print_stderr("Option(s) -g <string>, -W <string> (Nvidia) or -R <string> (AMD) must be given on commandline")
    usage()
    perfctr_exit(1)
end

if (cpuinfo["clock"] > 0) then
    cpuClock = cpuinfo["clock"]
else
    cpuClock = likwid.getCpuClock()
end

if outfile == nil then
    print_stdout(likwid.hline)
    print_stdout(string.format("CPU name:\t%s", cpuinfo["osname"]))
    print_stdout(string.format("CPU type:\t%s", cpuinfo["name"]))
    print_stdout(string.format("CPU clock:\t%3.2f GHz", cpuClock * 1.E-09))
end

if print_info or verbose > 0 then
    print_stdout(string.format("CPU family:\t%u", cpuinfo["family"]))
    print_stdout(string.format("CPU model:\t%u", cpuinfo["model"]))
    print_stdout(string.format("CPU vendor:\t%u", cpuinfo["vendor"]))
    print_stdout(string.format("CPU part:\t%u", cpuinfo["part"]))
    print_stdout(string.format("CPU short:\t%s", cpuinfo["short_name"]))
    print_stdout(string.format("CPU stepping:\t%u", cpuinfo["stepping"]))
    print_stdout(string.format("CPU features:\t%s", cpuinfo["features"]))
    print_stdout(string.format("CPU arch:\t%s", cpuinfo["architecture"]))
    P6_FAMILY = 6
    if cpuinfo["family"] == P6_FAMILY and cpuinfo["perf_version"] > 0 then
        print_stdout(likwid.hline)
        print_stdout(string.format("PERFMON version:\t\t\t%u", cpuinfo["perf_version"]))
        print_stdout(string.format("PERFMON number of counters:\t\t%u", cpuinfo["perf_num_ctr"]))
        print_stdout(string.format("PERFMON width of counters:\t\t%u", cpuinfo["perf_width_ctr"]))
        print_stdout(string.format("PERFMON number of fixed counters:\t%u", cpuinfo["perf_num_fixed_ctr"]))
        print_stdout(string.format("PERFMON supports desktop memory controllers:\t%u", cpuinfo["supportClientmem"]))
    end
    print_stdout(string.format("PERFMON supports Uncore:\t%u", cpuinfo["supportUncore"]))
    ---------------------------
    if nvSupported and cudatopo then
        print_stdout(likwid.hline)
        for i = 1, cudatopo["numDevices"] do
            gpu = cudatopo["devices"][i]
            print_stdout(string.format("NVMON GPU %d compute capability:\t%d.%d", gpu["id"], gpu["ccapMajor"],
                gpu["ccapMinor"]))
            print_stdout(string.format("NVMON GPU %d short:\t\t%s", gpu["id"], gpu["short"]))
        end
    end
    ---------------------------
    if rocmSupported and rocmtopo then
        print_stdout(likwid.hline)
        for i = 1, rocmtopo["numDevices"] do
            gpu = rocmtopo["devices"][i]
            print_stdout(string.format("ROCMON GPU %d compute capability:\t%d.%d", gpu["id"], gpu["ccapMajor"],
                gpu["ccapMinor"]))
            print_stdout(string.format("ROCMON GPU %d short:\t\t\t%s", gpu["id"], gpu["short"]))
        end
    end
    ---------------------------
    print_stdout(likwid.hline)
    if print_info then
        likwid.printSupportedCPUs()
        perfctr_exit(0)
    end
    --[[if nvSupported and cudatopo then
        print("Supported NVIDIA GPUs processors:")
        print("\tCompute capability < 7.0")
        print("\tCompute capability >= 7.0")
    end
    if rocmSupported and rocmtopo then
        print("Supported AMD ROCM GPUs processors:")
        print("\tall variants")
    end]]
end

if use_marker == true and use_timeline == true then
    print_stderr("Cannot run Marker API and Timeline mode simultaneously")
    perfctr_exit(0)
elseif use_marker == true and use_stethoscope == true then
    print_stderr("Cannot run Marker API and Stethoscope mode simultaneously")
    perfctr_exit(0)
elseif use_timeline == true and use_stethoscope == true then
    print_stderr("Cannot run Timeline and Stethoscope mode simultaneously")
    perfctr_exit(0)
end

if use_stethoscope == false and use_timeline == false and use_marker == false then
    use_wrapper = true
end

if use_wrapper and likwid.tablelength(arg) - 2 == 0 and print_info == false then
    print_stderr("No Executable can be found on commandline")
    usage(config)
    perfctr_exit(0)
end

if use_marker then
    if likwid.access(markerFile, "rw") ~= -1 then
        print_stderr(string.format("ERROR: MarkerAPI file %s not accessible. Maybe a remaining file of another user.",
            markerFile))
        print_stderr(string.format("Please purge all MarkerAPI files from %s.", markerFolder))
        perfctr_exit(1)
    end
    if nvSupported and #gpulist_cuda and likwid.access(nvMarkerFile, "rw") ~= -1 then
        print_stderr(string.format("ERROR: GPUMarkerAPI file %s not accessible. Maybe a remaining file of another user.",
            nvMarkerFile))
        print_stderr(string.format("Please purge all GPUMarkerAPI files from %s.", markerFolder))
        perfctr_exit(1)
    end
    if rocmSupported and #gpulist_rocm and likwid.access(rocmMarkerFile, "rw") ~= -1 then
        print_stderr(string.format("ERROR: GPUMarkerAPI file %s not accessible. Maybe a remaining file of another user.",
            rocmMarkerFile))
        print_stderr(string.format("Please purge all GPUMarkerAPI files from %s.", markerFolder))
        perfctr_exit(1)
    end
    if not pin_cpus and #cpulist > 0 and #event_string_list > 0 then
        print_stderr("Warning: The Marker API requires the application to run on the selected CPUs.")
        print_stderr("Warning: likwid-perfctr pins the application only when using the -C command line option.")
        print_stderr(
        "Warning: LIKWID assumes that the application does it before the first instrumented code region is started.")
        print_stderr(
        "Warning: You can use the string in the environment variable LIKWID_THREADS to pin you application to")
        print_stderr("Warning: to the CPUs specified after the -c command line option.")
    end
end

if verbose == 0 then
    likwid.setenv("LIKWID_SILENT", "true")
end

if pin_cpus then
    local omp_threads = os.getenv("OMP_NUM_THREADS")
    if omp_threads == nil then
        likwid.setenv("OMP_NUM_THREADS", tostring(math.tointeger(num_cpus)))
    elseif num_cpus > tonumber(omp_threads) then
        print_stderr(string.format("Environment variable OMP_NUM_THREADS already set to %s but %d cpus required",
            omp_threads, num_cpus))
    end
    if omp_threads and tonumber(omp_threads) < num_cpus then
        num_cpus = tonumber(omp_threads)
        for i = #cpulist, num_cpus + 1, -1 do
            cpulist[i] = nil
        end
    end
    if os.getenv("CILK_NWORKERS") == nil then
        likwid.setenv("CILK_NWORKERS", tostring(math.tointeger(num_cpus)))
    end
    if os.getenv("TBB_MAX_NUM_THREADS") == nil then
        likwid.setenv("TBB_MAX_NUM_THREADS", tostring(math.tointeger(num_cpus)))
    end
    if skip_mask then
        likwid.setenv("LIKWID_SKIP", skip_mask)
    end
    likwid.setenv("KMP_AFFINITY", "disabled")

    if num_cpus > 1 then
        local pinString = tostring(math.tointeger(cpulist[2]))
        for i = 3, likwid.tablelength(cpulist) do
            pinString = pinString .. "," .. tostring(math.tointeger(cpulist[i]))
        end
        pinString = pinString .. "," .. tostring(math.tointeger(cpulist[1]))
        likwid.setenv("LIKWID_PIN", pinString)

        local preload = os.getenv("LD_PRELOAD")
        if preload == nil then
            likwid.setenv("LD_PRELOAD", likwid.pinlibpath)
        else
            likwid.setenv("LD_PRELOAD", likwid.pinlibpath .. ":" .. preload)
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

if nvSupported and #gpulist_cuda > 0 and #cuda_event_string_list > 0 then
    likwid.setenv("LIKWID_NVMON_GPUS", table.concat(gpulist_cuda, ","))
    str = table.concat(cuda_event_string_list, "|")
    likwid.setenv("LIKWID_NVMON_EVENTS", str)
    if verbose > 0 then
        likwid.setenv("LIKWID_NVMON_VERBOSITY", tostring(verbose))
    end
end

if rocmSupported and #gpulist_rocm > 0 and #rocm_event_string_list > 0 then
    likwid.setenv("LIKWID_ROCMON_GPUS", table.concat(gpulist_rocm, ","))
    str = table.concat(rocm_event_string_list, "|")
    likwid.setenv("LIKWID_ROCMON_EVENTS", str)
    if verbose > 0 then
        likwid.setenv("LIKWID_ROCMON_VERBOSITY", tostring(verbose))
    end
end

if use_marker == true then
    likwid.setenv("LIKWID_FILEPATH", markerFile)
    likwid.setenv("LIKWID_MODE", tostring(access_mode))
    likwid.setenv("LIKWID_DEBUG", tostring(verbose))
    local str = table.concat(event_string_list, "|")
    likwid.setenv("LIKWID_EVENTS", str)
    likwid.setenv("LIKWID_THREADS", table.concat(cpulist, ","))
    likwid.setenv("LIKWID_FORCE", "-1")
    likwid.setenv("KMP_INIT_AT_FORK", "FALSE")
    if nvSupported and #gpulist_cuda > 0 and #cuda_event_string_list > 0 then
        likwid.setenv("LIKWID_NVMON_FILEPATH", nvMarkerFile)
    end
    if rocmSupported and #gpulist_rocm > 0 and #rocm_event_string_list > 0 then
        likwid.setenv("LIKWID_ROCMON_FILEPATH", rocmMarkerFile)
        if verbose > 0 then
            likwid.setenv("LIKWID_ROCMON_VERBOSITY", tostring(verbose))
        end
    end
end

if use_timeline == true or use_wrapper == true then
    if nvSupported and #gpulist_cuda > 0 and #cuda_event_string_list > 0 then
        if outfile then
            likwid.setenv("LIKWID_OUTPUTFILE", outfile)
            likwid.setenv("LIKWID_NVMON_OUTPUTFILE", outfile)
        end
        if use_timeline == true then
            --likwid.setenv("LIKWID_INTERVAL", duration / 1000)
            --likwid.setenv("LIKWID_NVMON_INTERVAL", duration / 1000)
            print_stderr("ERROR: Timeline mode for Nvidia GPUs not supported")
            perfctr_exit(1)
        else
            likwid.setenv("LIKWID_NVMON_MARKER_FORMAT", 1)
            likwid.setenv("LIKWID_NVMON_OUTPUTFILE", nvMarkerFile)
        end
    end
    if rocmSupported and #gpulist_rocm > 0 and #rocm_event_string_list > 0 then
        if outfile then
            likwid.setenv("LIKWID_OUTPUTFILE", outfile)
            likwid.setenv("LIKWID_ROCMON_OUTPUTFILE", outfile)
        end
        if use_timeline == true then
            --likwid.setenv("LIKWID_INTERVAL", duration / 1000)
            --likwid.setenv("LIKWID_ROCMON_INTERVAL", duration / 1000)
            print_stderr("ERROR: Timeline mode for AMD GPUs not supported")
            perfctr_exit(1)
        else
            likwid.setenv("LIKWID_ROCMON_MARKER_FORMAT", 1)
            likwid.setenv("LIKWID_ROCMON_OUTPUTFILE", rocmMarkerFile)
        end
    end
end

local likwid_hwthreads = {}
for i = 1, #cpulist do
    table.insert(likwid_hwthreads, tostring(math.tointeger(cpulist[i])))
end
likwid.setenv("LIKWID_HWTHREADS", table.concat(likwid_hwthreads, ","))

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

if #event_string_list > 0 then
    if set_access_modes then
        if likwid.setAccessClientMode(access_mode) ~= 0 then
            perfctr_exit(1)
        end
    end

    if likwid.init(num_cpus, cpulist) < 0 then
        perfctr_exit(1)
    end
end
---------------------------
if nvSupported and #cuda_event_string_list > 0 then
    if likwid.nvInit(num_cuda_gpus, gpulist_cuda) < 0 then
        perfctr_exit(1)
    end
    local preload = os.getenv("LD_PRELOAD")
    if preload == nil then
        likwid.setenv("LD_PRELOAD", "likwid-appDaemon.so")
    else
        likwid.setenv("LD_PRELOAD", "likwid-appDaemon.so" .. ":" .. preload)
    end
    if verbose > 0 then
        print_stdout("LD_PRELOAD=" .. os.getenv("LD_PRELOAD"))
    end
    local devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if devices == nil then
        likwid.setenv("CUDA_VISIBLE_DEVICES", table.concat(gpulist_cuda, ","))
    else
        print_stderr(string.format("ERROR: Cannot restrict CUDA devices to '%s', CUDA_VISIBLE_DEVICES already set", table.concat(gpulist_cuda, ",")))
    end
end
---------------------------
if rocmSupported and #rocm_event_string_list > 0 then
    --likwid.init_rocm(gpulist_rocm)
    --rocmInitialized = true
    local preload = os.getenv("LD_PRELOAD")
    if preload == nil then
        likwid.setenv("LD_PRELOAD", "likwid-appDaemon.so")
    else
        likwid.setenv("LD_PRELOAD", preload .. ":" .. "likwid-appDaemon.so")
    end
    local devices = os.getenv("ROCR_VISIBLE_DEVICES")
    if devices == nil then
        likwid.setenv("ROCR_VISIBLE_DEVICES", table.concat(gpulist_rocm, ","))
    else
        print_stderr(string.format("ERROR: Cannot restrict ROCm devices to '%s', ROCR_VISIBLE_DEVICES already set", table.concat(gpulist_rocm, ",")))
    end
end
---------------------------

if verbose > 0 then
    print_stdout(string.format("Executing: %s", table.concat(execList, " ")))
end
local ldpath = os.getenv("LD_LIBRARY_PATH")
local libpath = string.match(likwid.pinlibpath, "([/%a%d]+)/[%a%s%d]*")
if ldpath == nil then
    likwid.setenv("LD_LIBRARY_PATH", libpath)
elseif not ldpath:match(libpath) then
    likwid.setenv("LD_LIBRARY_PATH", libpath .. ":" .. ldpath)
end
---------------------------
if nvSupported then
    local cudahome = os.getenv("CUDA_HOME")
    if cudahome then
        ldpath = os.getenv("LD_LIBRARY_PATH")
        local cuptilib = string.format("%s/extras/CUPTI/lib64", cudahome)
        local likwidlib = "<INSTALLED_LIBPREFIX>"
        if not ldpath:match(cuptilib) then
            likwid.setenv("LD_LIBRARY_PATH", cuptilib .. ":" .. likwidlib .. ":" .. ldpath)
        else
            likwid.setenv("LD_LIBRARY_PATH", likwidlib .. ":" .. ldpath)
        end
    end
end
---------------------------
if rocmSupported then
    local rocmhome = os.getenv("ROCM_HOME")
    if rocmhome and rocmhome:len() > 0 then
        ldpath = os.getenv("LD_LIBRARY_PATH")
        local hiplib = string.format("%s/hip/lib", rocmhome)
        local hsalib = string.format("%s/hsa/lib", rocmhome)
        local rocproflib = string.format("%s/lib/rocprofiler", rocmhome)
        local likwidlib = "<INSTALLED_LIBPREFIX>"
        likwid.setenv("LD_LIBRARY_PATH", hiplib .. ":" .. hsalib .. ":" .. rocproflib .. ":" .. likwidlib .. ":" ..
        ldpath)
        likwid.setenv("HSA_TOOLS_LIB", "librocprofiler64.so")
    end
end
---------------------------


local pid = nil
if #execList > 0 then
    local execString = table.concat(execList, " ")
    if execpid then
        likwid.setenv("LIKWID_PERF_EXECPID", "1")
    end
    if pin_cpus then
        pid = likwid.startProgram(execString, #cpulist, cpulist)
    else
        pid = likwid.startProgram(execString, 0, cpulist)
    end
    if execpid then
        perfpid = pid
    end
end
if not pid and #execList > 0 then
    print_stderr(string.format("Failed to execute command: %s", table.concat(execList, " ")))
    perfctr_exit(1)
elseif #execList > 0 then
    likwid.sendSignal(pid, 19)
end


if forceOverwrite == 1 and os.getenv("LIKWID_FORCE") ~= tostring(forceOverwrite) then
    likwid.setenv("LIKWID_FORCE", tostring(forceOverwrite))
end
for i, event_string in pairs(event_string_list) do
    if event_string:len() > 0 then
        if perfpid ~= nil then
            likwid.setenv("LIKWID_PERF_PID", tostring(perfpid))
        end
        if perfpid ~= nil and perfflags ~= nil then
            likwid.setenv("LIKWID_PERF_FLAGS", tostring(perfflags))
        end
        local gid = likwid.addEventSet(event_string)
        if gid < 0 then
            likwid.finalize()
            perfctr_exit(1)
        end
        table.insert(group_ids, gid)
    end
end
---------------------------
if nvSupported and use_timeline == false then
    for i, event_string in pairs(cuda_event_string_list) do
        if event_string:len() > 0 then
            local gid = likwid.nvAddEventSet(event_string)
            if gid < 0 then
                likwid.nvFinalize()
                perfctr_exit(1)
            end
            table.insert(cudagroups, gid)
        end
    end
end
---------------------------
-- ROCM no longer requires adding of event sets, since they are
-- not used at all in the parent process.
---------------------------
if #group_ids == 0 and #cudagroups == 0 and #rocm_event_string_list == 0 and use_timeline == false then
    print_stderr("ERROR: No valid eventset given on commandline. Exiting...")
    likwid.finalize()
    perfctr_exit(1)
end

if #event_string_list > 0 then
    activeGroup = group_ids[1]
    ret = likwid.setupCounters(activeGroup)
    if ret < 0 then
        likwid.killProgram(pid)
        perfctr_exit(1)
    end
    if outfile == nil then
        print_stdout(likwid.hline)
    end
end
---------------------------
if nvSupported and #cuda_event_string_list > 0 then
    activeNvGroup = cudagroups[1]
    if outfile == nil then
        print_stdout(likwid.hline)
    end
end
---------------------------
if rocmSupported and #rocm_event_string_list > 0 then
    -- What is this print useful for?
    if outfile == nil then
        print_stdout(likwid.hline)
    end
end
---------------------------


if #event_string_list > 0 then
    timeline_delim = " "
    if use_csv then
        timeline_delim = ","
    end
    if use_timeline == true then
        local delim = "|"
        local word_delim = ": "
        if outfile_orig ~= nil then
            io.output(outfile)
            delim = timeline_delim
            word_delim = timeline_delim
        end
        local clist = {}
        for i, cpu in pairs(cpulist) do
            table.insert(clist, tostring(cpu))
        end
        print(outprefix .. "# HWThreads" .. word_delim .. table.concat(clist, delim))
        for i, gid in pairs(group_ids) do
            local strlist = { "GID" }
            if likwid.getNumberOfMetrics(gid) == 0 then
                table.insert(strlist, "EventCount")
                table.insert(strlist, "CpuCount")
                table.insert(strlist, "Total runtime [s]")
                for e = 1, likwid.getNumberOfEvents(gid) do
                    table.insert(strlist, likwid.getNameOfEvent(gid, e))
                end
            else
                table.insert(strlist, "MetricsCount")
                table.insert(strlist, "CpuCount")
                table.insert(strlist, "Total runtime [s]")
                for m = 1, likwid.getNumberOfMetrics(gid) do
                    table.insert(strlist, likwid.getNameOfMetric(gid, m))
                end
            end
            print(outprefix .. "# " .. table.concat(strlist, delim))
        end
    end
end


io.stdout:flush()
io.stderr:flush()
local groupTime = {}
local exitvalue = 0
local twork = 0
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

    if #event_string_list > 0 then
        local ret = likwid.startCounters()
        if ret < 0 then
            print_stderr(string.format("Error starting counters for cpu %d.", cpulist[ret * (-1)]))
            perfctr_exit(1)
        end
    end

    likwid.sendSignal(pid, 18)

    start = likwid.startClock()
    groupTime[activeGroup] = 0

    while true do
        local state = likwid.getSignalState()
        if state ~= 0 then
            if #execList > 0 then
                likwid.killProgram(pid)
            end
            break
        end
	exitvalue, exited = likwid.checkProgram(pid)
        if exited then
            io.stdout:flush()
            if #execList > 0 then
                break
            end
        end
        local sleeptime = duration - (twork * 1E6)
	local remain = 0
        if sleeptime > 0 then
           remain = likwid.sleep(math.floor(duration - (twork * 1E6)))
        end

        exitvalue, exited = likwid.checkProgram(pid)
        if exited then
            io.stdout:flush()
            if #execList > 0 then
                break
            end
        end

        if use_timeline == true and #event_string_list > 0 then
            stop = likwid.stopClock()
            xstart = likwid.startClock()
            if #event_string_list > 0 then
                likwid.readCounters()
            end

            local time = likwid.getClock(start, stop)
            if likwid.getNumberOfMetrics(activeGroup) == 0 then
                results = likwid.getLastResults(nan2value)
            else
                results = likwid.getLastMetrics(nan2value)
            end
            local outList = {}
            table.insert(outList, tostring(math.tointeger(activeGroup)))
            table.insert(outList, tostring(#results[activeGroup]))
            table.insert(outList, tostring(#cpulist))
            table.insert(outList, tostring(time))
            for i, l1 in pairs(results[activeGroup]) do
                for j, value in pairs(l1) do
                    table.insert(outList, tostring(value))
                end
            end
            if not outfile then
                print_stderr(outprefix .. table.concat(outList, timeline_delim))
            else
                print(outprefix .. table.concat(outList, timeline_delim))
                io.flush()
            end
            groupTime[activeGroup] = time
            xstop = likwid.stopClock()
            twork = likwid.getClock(xstart, xstop)
        else
            xstart = likwid.startClock()
            if #event_string_list > 0 then
                likwid.readCounters()
            end
            xstop = likwid.stopClock()
            twork = likwid.getClock(xstart, xstop)
        end
        if #group_ids > 1 then
            likwid.switchGroup(activeGroup + 1)
            activeGroup = likwid.getIdOfActiveGroup()
            if groupTime[activeGroup] == nil then
                groupTime[activeGroup] = 0
            end
            nr_events = likwid.getNumberOfEvents(activeGroup)
        end

        stop = likwid.stopClock()
    end
elseif use_stethoscope then
    local ret = likwid.startCounters()
    if ret < 0 then
        print_stderr(string.format("Error starting counters for cpu %d.", cpulist[ret * (-1)]))
        perfctr_exit(1)
    end
    likwid.sleep(duration)
elseif use_marker then
    likwid.sendSignal(pid, 18)
    exitvalue = likwid.waitpid(pid)
end

if not use_marker then
    if #event_string_list > 0 then
        local ret = likwid.stopCounters()
        if ret < 0 then
            print_stderr(string.format("Error stopping counters for thread %d.", ret * (-1)))
            likwid.finalize()
            perfctr_exit(exitvalue)
        end
    end
end
io.stdout:flush()
if outfile == nil then
    print_stdout(likwid.hline)
end


if use_marker == true then
    if #event_string_list > 0 then
        if likwid.access(markerFile, "e") >= 0 then
            results, metrics = likwid.getMarkerResults(markerFile, cpulist, nan2value)
            if not results then
                print_stderr("Failure reading Marker API result file.")
            elseif #results == 0 then
                print_stderr("No regions could be found in Marker API result file.")
            else
                for r = 1, #results do
                    likwid.printOutput(results[r], metrics[r], cpulist, r, print_stats)
                end
            end
            os.remove(markerFile)
        else
            print_stderr(
            "MMarker API result file does not exist. This may happen if the application was not compiled with LIKWID_PERFMON macro or the application has not called LIKWID_MARKER_CLOSE.")
        end
    end
    ---------------------------
    if nvSupported and #cuda_event_string_list > 0 then
        if likwid.access(nvMarkerFile, "e") >= 0 then
            results, metrics = likwid.getMarkerResultsCuda(nvMarkerFile, gpulist_cuda, nan2value)
            if not results then
                print_stderr("Failure reading Nv Marker API result file.")
            elseif #results == 0 then
                print_stderr("No regions could be found in Nv Marker API result file.")
            else
                for r = 1, #results do
                    likwid.printOutputCuda(results[r], metrics[r], gpulist_cuda, r, print_stats)
                end
            end
            likwid.destroyNvMarkerFile()
            os.remove(nvMarkerFile)
        else
            print_stderr(
            "GPU Marker API result file does not exist. This may happen if the application was not compiled with LIKWID_NVMON macro or the application has not called LIKWID_GPUMARKER_CLOSE.")
        end
    end
    ---------------------------
    if rocmSupported and #rocm_event_string_list > 0 then
        if likwid.access(rocmMarkerFile, "e") >= 0 then
            results, metrics = likwid.getMarkerResultsRocm(rocmMarkerFile, gpulist_rocm, nan2value)
            if not results then
                print_stderr("Failure reading ROCM Marker API result file.")
            elseif #results == 0 then
                print_stderr("No regions could be found in ROCM Marker API result file.")
            else
                for r = 1, #results do
                    likwid.printOutputRocm(results[r], metrics[r], gpulist_rocm, r, print_stats)
                end
            end
            os.remove(rocmMarkerFile)
        else
            print_stderr(
            "ROCM Marker API result file does not exist. This may happen if the application has not called LIKWID_ROCMMARKER_CLOSE.")
        end
    end
    ---------------------------
elseif use_timeline == false then
    if #event_string_list > 0 then
        results = likwid.getResults(nan2value)
        metrics = likwid.getMetrics(nan2value)
        likwid.printOutput(results, metrics, cpulist, nil, print_stats)
    end
    if nvSupported and #cuda_event_string_list > 0 then
        if likwid.access(nvMarkerFile, "e") >= 0 then
            results, metrics = likwid.getMarkerResultsCuda(nvMarkerFile, gpulist_cuda, nan2value)
            if not results then
                print_stderr("Failure reading appdaemon result file.")
            elseif #results == 0 then
                print_stderr("No regions could be found in appdaemon result file.")
            else
                for r = 1, #results do
                    likwid.printOutputCuda(results[r], metrics[r], gpulist_cuda, r, print_stats)
                end
            end
            likwid.destroyNvMarkerFile()
            os.remove(nvMarkerFile)
        else
            print_stderr(
            "Appdaemon result file does not exist. This may happen if the application segfault and the exit handlers weren't executed.")
        end
    end
    if rocmSupported and #rocm_event_string_list > 0 then
        if likwid.access(rocmMarkerFile, "e") >= 0 then
            results = likwid.markerInitResultsFromFileRocm(rocmMarkerFile)
            likwid.printOutputRocm(results, nan2value)
            os.remove(rocmMarkerFile)
        else
            print_stderr(
            "Appdaemon result file does not exist. This may happen if the application segfault and the exit handlers weren't executed.")
        end
    end
end

if outfile and not use_timeline then
    local suffix = ""
    if string.match(outfile, ".-[^\\/]-%.?([^%.\\/]*)$") then
        suffix = string.match(outfile, ".-[^\\/]-%.?([^%.\\/]*)$")
    end
    local command = "<INSTALLED_PREFIX>/share/likwid/filter/" .. suffix
    if suffix:len() > 0 and suffix ~= "csv" and suffix ~= "txt" then
        if likwid.access(command, "x") == 0 then
            local tmpfile = outfile .. ".tmp"
            os.rename(outfile, tmpfile)
            local cmd = command .. " " .. tmpfile .. " perfctr"
            local f = assert(io.popen(cmd), "r")
            if f ~= nil then
                local o = f:read("*a")
                if o:len() > 0 then
                    print_stderr(string.format("Failed to executed filter script %s. Output file %s in CSV format.",
                        command, outfile))
                    if likwid.access(tmpfile, "e") == 0 then
                        os.rename(tmpfile, outfile)
                    end
                end
            end
            if likwid.access(tmpfile, "e") == 0 then
                os.remove(tmpfile)
            end
        elseif likwid.access(command, "e") == 0 then
            print_stderr(string.format("Filter script %s not executable. Output file %s in CSV format.", command, outfile))
        else
            print_stderr(string.format("Filter script %s not available. Output file %s in CSV format.", command, outfile))
        end
    end
end

likwid.finalize()
perfctr_exit(exitvalue)
