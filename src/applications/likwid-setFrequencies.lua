#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-setFrequencies.lua
 *
 *      Description:  A application to set the CPU frequency of CPU cores and domains.
 *
 *      Version:   4.3.3
 *      Released:  09.11.2018
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end

sys_base_path = "/sys/devices/system/cpu"
set_command = "<INSTALLED_PREFIX>/sbin/likwid-setFreq"


function version()
    print_stdout(string.format("likwid-setFrequencies -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

function usage()
    version()
    print_stdout("A tool to adjust frequencies and governors on x86 CPUs.\n")
    print_stdout("Options:")
    print_stdout("-h\t\t Help message")
    print_stdout("-v\t\t Version information")
    print_stdout("-V <0-3>\t Verbosity (0=only_error, 3=developer)")
    print_stdout("-c dom\t\t Likwid thread domain which to apply settings (default are all CPUs)")
    print_stdout("\t\t See likwid-pin -h for details")
    print_stdout("-g gov\t\t Set governor (" .. table.concat(likwid.getAvailGovs(0), ", ") .. ")")
    print_stdout("-f/--freq freq\t Set minimal and maximal CPU frequency")
    print_stdout("-p\t\t Print current frequencies (CPUs + Uncore)")
    print_stdout("-l\t\t List available CPU frequencies")
    print_stdout("-m\t\t List available CPU governors")
    print_stdout("-t/--turbo <0|1> De/Activate turbo mode")
    print_stdout("")
    print_stdout("-x/--min freq\t Set minimal CPU frequency")
    print_stdout("-y/--max freq\t Set maximal CPU frequency")
    print_stdout("--umin freq\t Set minimal Uncore frequency")
    print_stdout("--umax freq\t Set maximal Uncore frequency")
    print_stdout("")
    print_stdout("-reset\tSet governor 'performance', set minimal and maximal frequency to")
    print_stdout("\tthe CPU limits and deactivate turbo")
    print_stdout("-ureset\tSet Uncore frequencies to its min and max limits")
    print_stdout("")
    print_stdout("For the options -f, -x and -y:")
    print_stdout("\t acpi-cpufreq driver: set the userspace governor implicitly")
    print_stdout("\t intel_pstate driver: keep current governor")
    print_stdout("")
    print_stdout("The min/max frequencies can be slightly off with the intel_pstate driver as")
    print_stdout("the value is calculated while the current frequency is read from sysfs.")
    print_stdout("")
    print_stdout("In general the min/max Uncore frequency can be set freely, even to 0 or 1E20")
    print_stdout("but the hardware stays inside its limits. LIKWID reduces the range of possible")
    print_stdout("frequencies to the minimal core frequency (likwid-setFrequencies -l) and the ")
    print_stdout("maximally achievable turbo frequency with a single core (C0 value at")
    print_stdout("likwid-powermeter -i output).")
    print_stdout("To check whether the set Uncore frequency is really set and fixed, use")
    print_stdout("likwid-perfctr -g UNCORE_CLOCK:UBOXFIX -C 0 <exec_doing_work>.")
    print_stdout("Sleeping is commonly not sufficient.")
    print_stdout("")
    print_stdout("If you switch governors with the intel_pstate driver, it might be that the driver")
    print_stdout("changes the frequency settings, please check afterwards and re-set frequencies if")
    print_stdout("needed.")
end

function round(x)
    if (type(x) ~= "number") then
        x = tonumber(x)
    end

    s = string.format("%f", x)
    if not s:match("%d+%.%d+") then
        s = s.. ".0"
    end
    slen = s:len()
    while slen > 3 do
        if s:sub(slen,slen) ~= "0" then break end
        slen = slen - 1
    end
    if slen > 5 then
        slen = 5
    end
    return s:sub(1, slen)
end

verbosity = 0
governor = nil
frequency = nil
min_freq = nil
max_freq = nil
min_u_freq = nil
max_u_freq = nil
domain = nil
printCurFreq = false
printAvailFreq = false
printAvailGovs = false
do_reset = false
do_ureset = false
set_turbo = false
turbo = 0

if #arg == 0 then
    usage()
    os.exit(0)
end


for opt,arg in likwid.getopt(arg, {"V:", "g:", "c:", "f:", "l", "p", "h", "v", "m", "x:", "y:", "t:", "help","version","freq:", "min:", "max:", "umin:", "umax:", "reset", "turbo:", "ureset"}) do
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif (opt == "c") then
        domain = arg
    elseif (opt == "g") then
        governor = arg
    elseif opt == "f" or opt == "freq" then
        frequency = arg
        min_freq = arg
        max_freq = arg
    elseif opt == "x" or opt == "min" then
        min_freq = arg
    elseif opt == "y" or opt == "max" then
        max_freq = arg
    elseif opt == "t" or opt == "turbo" then
        set_turbo = true
        local t = tonumber(arg)
        if (t >= 0 and t <= 1) then
            turbo = t
        else
            print_stderr(string.format("ERROR: Value %s for turbo not valid: 1=active turbo, 0=disabled turbo", arg))
        end
    elseif opt == "V" then
        local s = tonumber(arg)
        if (s >= 0 and s <= 3) then
            verbosity = s
        else
            print_stderr(string.format("ERROR: Value %s for verbosity not valid", arg))
        end
    elseif opt == "reset" then
        do_reset = true
    elseif opt == "ureset" then
        do_ureset = true
    elseif opt == "umin" then
        min_u_freq = arg
    elseif opt == "umax" then
        max_u_freq = arg
    elseif (opt == "p") then
        printCurFreq = true
    elseif (opt == "l") then
        printAvailFreq = true
    elseif (opt == "m") then
        printAvailGovs = true
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end

topo = likwid.getCpuTopology()
affinity = likwid.getAffinityInfo()
if not domain or domain == "N" then
    domain = "N:0-" .. tostring(topo["numHWThreads"]-1)
end
if domain:match("[SCM]%d") then
    for i, dom in pairs(affinity["domains"]) do
        if dom["tag"]:match(domain) then
            domain = domain..":0-"..tostring(dom["numberOfProcessors"]-1)
        end
    end
end
cpulist = {}
socklist = {}
numthreads, cpulist = likwid.cpustr_to_cpulist(domain)
for i, dom in pairs(affinity["domains"]) do
    if dom["tag"]:match("S%d") then
        for k, d in pairs(dom["processorList"]) do
            local found = false
            for j, c in pairs(cpulist) do
                if c == d then
                    
                    found = true
                    break
                end
            end
            if found then
                s = tonumber(dom["tag"]:match("S(%d)"))
                found = false
                for j, c in pairs(socklist) do
                    if c == s then found = true end
                end
                if not found then
                    table.insert(socklist, s)
                end
            end
        end
    end
end

if verbosity == 3 then
    print_stdout(string.format("DEBUG: Given CPU expression expands to %d CPU cores:", numthreads))
    local str = "DEBUG: " .. tostring(cpulist[1])
    for i=2, numthreads  do
        str = str .. "," .. tostring(cpulist[i])
    end
    print_stdout(str)
    print_stdout(string.format("DEBUG: Given CPU expression expands to %d CPU sockets:", #socklist))
    str = "DEBUG: " .. tostring(socklist[1])
    for i=2, #socklist do
        str = str .. "," .. tostring(socklist[i])
    end
    print_stdout(str)
end


if printAvailGovs then
    local govs = likwid.getAvailGovs(0)
    if govs and #govs > 0 then
        print_stdout("Available governors:")
        print_stdout(string.format("%s", table.concat(govs, " ")))
    else
        print_stdout("Cannot get governors from cpufreq module")
    end
end

if printAvailFreq then
    local freqs = likwid.getAvailFreq(0)
    if #freqs > 0 then
        print_stdout("Available frequencies:")
        print_stdout(string.format("%s", table.concat(freqs, " ")))
    else
        print_stdout("Cannot get frequencies from cpufreq module")
    end
end

if printCurFreq then
    str = {"Current CPU frequencies:"}
    local processed = 0
    for i=1,#cpulist do
        gov = likwid.getGovernor(cpulist[i])
        if not gov then break end
        freq = tonumber(likwid.getCpuClockCurrent(cpulist[i]))/1E9
        min = tonumber(likwid.getCpuClockMin(cpulist[i]))/1E9
        max = tonumber(likwid.getCpuClockMax(cpulist[i]))/1E9
        t = tonumber(likwid.getTurbo(cpulist[i]));
        if gov and freq and min and max and t >= 0 then
            processed = processed + 1
            table.insert(str, string.format("CPU %d: governor %12s min/cur/max %s/%s/%s GHz Turbo %d",cpulist[i], gov, round(min), round(freq), round(max), t))
        end
    end
    table.insert(str, "")
    if processed > 0 then
        print_stdout(table.concat(str, "\n"))
    else
        print_stdout("Cannot read frequency data from cpufreq module\n")
        os.exit(1)
    end
    test = likwid.getUncoreFreqMin(socklist[i])
    if test ~= 0 then
        print_stdout("Current Uncore frequencies:")
        for i=1,#socklist do
            min = tonumber(likwid.getUncoreFreqMin(socklist[i]))/1000.0
            max = tonumber(likwid.getUncoreFreqMax(socklist[i]))/1000.0
            print_stdout(string.format("Socket %d: min/max %s/%s GHz", socklist[i], round(min), round(max)))
        end
    else
        print("No support for Uncore frequencies")
    end
end

if printAvailGovs or printAvailFreq or printCurFreq then
    os.exit(0)
end

if do_reset then
    local f = likwid.setTurbo(cpulist[1], 0)
    local availfreqs = likwid.getAvailFreq(cpulist[1])
    local availgovs = likwid.getAvailGovs(cpulist[1])
    if not min_freq then
        min_freq = availfreqs[1]
    end
    if not (set_turbo or max_freq) then
        set_turbo = true
        turbo = 0
        max_freq = availfreqs[#availfreqs]
    end
    if not governor then
        governor = availgovs[#availgovs]
    end
    if min_freq and governor then
        print_stdout(string.format("Reset to governor %s with min freq. %g GHz and deactivate turbo mode", governor, min_freq))
    end
end

if do_ureset then
    local availfreqs = likwid.getAvailFreq(cpulist[1])
    local power = likwid.getPowerInfo()
    local minf = tonumber(availfreqs[1])
    local maxf = tonumber(power["turbo"]["steps"][1]) / 1000
    min_u_freq = minf
    max_u_freq = maxf
end

if numthreads > 0 and not (frequency or min_freq or max_freq or governor or min_u_freq or max_u_freq or set_turbo) then
    print_stderr("ERROR: You need to set either a frequency or governor for the selected CPUs on commandline")
    os.exit(1)
end

if min_freq and max_freq and min_freq > max_freq then
    print_stderr("ERROR: Minimal CPU frequency higher than maximal frequency.")
    os.exit(1)
end
if min_freq and max_freq and max_freq < min_freq then
    print_stderr("ERROR: Maximal CPU frequency lower than minimal frequency.")
    os.exit(1)
end
if min_u_freq and max_u_freq and max_u_freq < min_u_freq then
    print_stderr("ERROR: Maximal Uncore frequency lower than minimal frequency.")
    os.exit(1)
end


local availfreqs = likwid.getAvailFreq(cpulist[i])
if (frequency or min_freq or max_freq) and #availfreqs == 0 then
    print_stdout("Cannot set CPU frequency, cpufreq module not properly loaded")
    os.exit(1)
end
local savailfreqs = {}
for i,f in pairs(availfreqs) do
    savailfreqs[i] = round(f)
end
if verbosity == 3 then
    print_stdout("DEBUG Available freq.: "..table.concat(availfreqs, ", "))
end


for x=1,2 do
    if min_freq then
        for i=1,#cpulist do
            local valid_freq = false
            for k,v in pairs(savailfreqs) do
                if (tonumber(min_freq) == tonumber(v)) then
                    if verbosity == 3 then
                        print_stdout(string.format("DEBUG: Min frequency %g valid", min_freq))
                    end
                    valid_freq = true
                    break
                end
            end
            if min_freq == availturbo then
                valid_freq = true
            end
            if not valid_freq then
                print_stderr(string.format("ERROR: Selected min. frequency %s not available for CPU %d! Please select one of\n%s", min_freq, cpulist[i], table.concat(savailfreqs, ", ")))
                os.exit(1)
            end
            if verbosity == 3 then
                print_stdout(string.format("DEBUG: Set min. frequency for CPU %d to %d", cpulist[i], tonumber(min_freq)*1E6))
            end
            local f = likwid.setCpuClockMin(cpulist[i], tonumber(min_freq)*1E6)
            if (f ~= tonumber(min_freq)*1E6) then os.exit(0) end
        end
    end


    if set_turbo then
        for i=1,#cpulist do
            if verbosity == 3 then
                print_stdout(string.format("DEBUG: Set turbo mode for CPU %d to %d", cpulist[i], turbo))
            end
            local f = likwid.setTurbo(cpulist[i], turbo)
        end
    end
    if max_freq then
        for i=1,#cpulist do
            local valid_freq = false
            for k,v in pairs(savailfreqs) do
                if (tonumber(max_freq) == tonumber(v)) then
                    if verbosity == 3 then
                        print_stdout(string.format("DEBUG: Max frequency %g valid", max_freq))
                    end
                    valid_freq = true
                    break
                end
            end
            if max_freq == availturbo then
                valid_freq = true
            end
            if not valid_freq then
                print_stderr(string.format("ERROR: Selected max. frequency %s not available for CPU %d! Please select one of\n%s", max_freq, cpulist[i], table.concat(savailfreqs, ", ")))
                os.exit(1)
            end
            if verbosity == 3 then
                print_stdout(string.format("DEBUG: Set max. frequency for CPU %d to %d", cpulist[i], tonumber(max_freq)*1E6))
            end
            local f = likwid.setCpuClockMax(cpulist[i], tonumber(max_freq)*1E6)
            if (f ~= tonumber(max_freq)*1E6) then os.exit(0) end
        end
    end
end

if min_u_freq then
    for s=1,#socklist do
        socket = socklist[s]
        if verbosity == 3 then
            print_stdout(string.format("DEBUG: Set min. uncore frequency for socket %d to %d MHz", socket, min_u_freq*1000))
        end
        local err = likwid.setUncoreFreqMin(socket, min_u_freq*1000);
        if err ~= 0 then
            print_stderr(string.format("Setting of minimal Uncore frequency %f failed on socket %d", tonumber(min_u_freq)*1000, socket))
        end
    end
end

if max_u_freq then
    for s=1,#socklist do
        socket = socklist[s]
        if verbosity == 3 then
            print_stdout(string.format("DEBUG: Set max. uncore frequency for socket %d to %d MHz", socket, max_u_freq*1000))
        end
        local err = likwid.setUncoreFreqMax(socket, max_u_freq*1000);
        if err ~= 0 then
            print_stderr(string.format("Setting of maximal Uncore frequency %d failed on socket %d", tonumber(max_u_freq)*1000, socket))
        end
    end
end

if governor then
    if verbosity == 3 then
        print_stdout(string.format("DEBUG: Set governor %s", governor))
    end
    local govs = likwid.getAvailGovs(cpulist[1])
    local cur_govs = {}
    local cur_min = {}
    local cur_max = {}
    for i,c in pairs(cpulist) do
        cur_govs[i] = likwid.getGovernor(c)
        cur_min[i] = likwid.getCpuClockMin(c)
        cur_max[i] = likwid.getCpuClockMax(c)
    end
    
    local valid_gov = false
    for k,v in pairs(govs) do
        if (governor == v) then
            valid_gov = true
            break
        end
    end
    local cur_freqs = {}
    if not valid_gov then
        print_stderr(string.format("ERROR: Governor %s not available! Please select one of\n%s", governor, table.concat(govs, ", ")))
        os.exit(1)
    end
    for i=1,#cpulist do
        if verbosity == 3 then
            print_stdout(string.format("DEBUG: Set governor for CPU %d to %s", cpulist[i], governor))
        end
        local f = likwid.setGovernor(cpulist[i], governor)
        if f == 0 then os.exit(0) end
        if do_reset then
            likwid.setCpuClockMin(cpulist[i], cur_min[i])
            likwid.setCpuClockMax(cpulist[i], cur_max[i])
        end
    end
end
likwid.putAffinityInfo()
likwid.putTopology()
os.exit(0)
