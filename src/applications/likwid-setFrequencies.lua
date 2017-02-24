#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-setFrequencies.lua
 *
 *      Description:  A application to set the CPU frequency of CPU cores and domains.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
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
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end

sys_base_path = "/sys/devices/system/cpu"
set_command = "<INSTALLED_PREFIX>/sbin/likwid-setFreq"


function version()
    print_stdout(string.format("likwid-setFrequencies --  Version %d.%d",likwid.version,likwid.release))
end

function usage()
    version()
    print_stdout("A tool to adjust frequencies and governors on x86 CPUs.\n")
    print_stdout("Options:")
    print_stdout("-h\t Help message")
    print_stdout("-v\t Version information")
    print_stdout("-c dom\t Likwid thread domain which to apply settings (default are all CPUs)")
    print_stdout("\t See likwid-pin -h for details")
    print_stdout("-g gov\t Set governor (" .. table.concat(likwid.getAvailGovs(nil), ", ") .. ") (set to ondemand if omitted)")
    print_stdout("-f/--freq freq\t Set current CPU frequency, implicitly sets userspace governor")
    print_stdout("-x/--min freq\t Set minimal CPU frequency")
    print_stdout("-y/--max freq\t Set maximal CPU frequency")
    print_stdout("--umin freq\t Set minimal Uncore frequency")
    print_stdout("--umax freq\t Set maximal Uncore frequency")
    print_stdout("-p\t Print current frequencies (CPUs + Uncore)")
    print_stdout("-l\t List available CPU frequencies")
    print_stdout("-m\t List available CPU governors")
    print_stdout("")
    print_stdout("In order to set the highest frequency, use the governor 'turbo'. This sets the")
    print_stdout("minimal frequency to the available minimum, the maximal and current frequency")
    print_stdout("to the turbo related frequency. The governor is set to 'performance'.")
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

if #arg == 0 then
    usage()
    os.exit(0)
end


for opt,arg in likwid.getopt(arg, {"g:", "c:", "f:", "l", "p", "h", "v", "m", "x:", "y:", "help","version","freq:", "min:", "max:", "umin:", "umax:"}) do
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
    elseif opt == "x" or opt == "min" then
        min_freq = arg
    elseif opt == "y" or opt == "max" then
        max_freq = arg
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
if likwid.getDriver() ~= "acpi-cpufreq" then
    print_stderr("The system does not use the acpi-cpufreq driver, other drivers are not usable with likwid-setFrequencies.")
    os.exit(1)
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
    print_stdout(string.format("Given CPU expression expands to %d CPU cores:", numthreads))
    local str = tostring(cpulist[1])
    for i=2, numthreads  do
        str = str .. "," .. tostring(cpulist[i])
    end
    print_stdout(str)
    print_stdout(string.format("Given CPU expression expands to %d CPU sockets:", #socklist))
    str = tostring(socklist[1])
    for i=2, #socklist do
        str = str .. "," .. tostring(socklist[i])
    end
    print_stdout(str)
end


if printAvailGovs then
    local govs = likwid.getAvailGovs(0)
    print_stdout("Available governors:")
    print_stdout(string.format("%s %s", table.concat(govs, " "), "turbo"))
end

if printAvailFreq then
    local freqs, turbo = likwid.getAvailFreq(0)
    print_stdout("Available frequencies:")
    print_stdout(string.format("%s %s", turbo, table.concat(freqs, " ")))
end

if printCurFreq then
    print_stdout("Current CPU frequencies:")
    for i=1,#cpulist do
        gov = likwid.getGovernor(cpulist[i])
        freq = tonumber(likwid.getCpuClockCurrent(cpulist[i]))/1E9
        min = tonumber(likwid.getCpuClockMin(cpulist[i]))/1E9
        max = tonumber(likwid.getCpuClockMax(cpulist[i]))/1E9
        print_stdout(string.format("CPU %d: governor %12s min/cur/max %s/%s/%s GHz",cpulist[i], gov, min, freq, max))
    end
    print_stdout("")
    print_stdout("Current Uncore frequencies:")
    for i=1,#socklist do
        min = tonumber(likwid.getUncoreFreqMin(socklist[i]))/1000.0
        max = tonumber(likwid.getUncoreFreqMax(socklist[i]))/1000.0
        print_stdout(string.format("Socket %d: min/max %s/%s GHz", socklist[i], min, max))
    end
end

if printAvailGovs or printAvailFreq or printCurFreq then
    os.exit(0)
end

if numthreads > 0 and not (frequency or min_freq or max_freq or governor or min_u_freq) then
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


local availfreqs, availturbo = likwid.getAvailFreq(cpulist[i])
if governor == "turbo" then
    if not min_freq then
        min_freq = availfreqs[#availfreqs]
    end
    if not max_freq or max_freq < availturbo then
        max_freq = availturbo
    end
    frequency = availturbo
end

if min_freq then
    for i=1,#cpulist do
        local valid_freq = false
        for k,v in pairs(availfreqs) do
            if (min_freq == v) then
                valid_freq = true
                break
            end
        end
        if min_freq == availturbo then
            valid_freq = true
        end
        if not valid_freq then
            print_stderr(string.format("ERROR: Selected min. frequency %s not available for CPU %d! Please select one of\n%s", min_freq, cpulist[i], table.concat(availfreqs, ", ")))
            os.exit(1)
        end
        local f = likwid.setCpuClockMin(cpulist[i], tonumber(min_freq)*1E6)
    end
end

if max_freq then
    for i=1,#cpulist do
        local valid_freq = false
        for k,v in pairs(availfreqs) do
            if (max_freq == v) then
                valid_freq = true
                break
            end
        end
        if max_freq == availturbo then
            valid_freq = true
        end
        if not valid_freq then
            print_stderr(string.format("ERROR: Selected max. frequency %s not available for CPU %d! Please select one of\n%s", max_freq, cpulist[i], table.concat(availfreqs, ", ")))
            os.exit(1)
        end
        local f = likwid.setCpuClockMax(cpulist[i], tonumber(max_freq)*1E6)
    end
end

if min_u_freq then
    socket = 0
    local err = likwid.setUncoreFreqMin(socket, min_u_freq*1000);
    if err ~= 0 then
        print_stderr(string.format("Setting of minimal Uncore frequency %f failed on socket %d\n", min_u_freq, socket))
        os.exit(1)
    end
end

if max_u_freq then
    socket = 0
    local err = likwid.setUncoreFreqMax(socket, min_u_freq*1000);
    if err ~= 0 then
        print_stderr(string.format("Setting of minimal Uncore frequency %d failed on socket %d\n", min_u_freq, socket))
        os.exit(1)
    end
end

if frequency then
    for i=1,#cpulist do
        
        local valid_freq = false
        for k,v in pairs(availfreqs) do
            if (frequency == v) then
                valid_freq = true
                break
            end
        end
        if frequency == availturbo then
            valid_freq = true
        end
        if not valid_freq then
            print_stderr(string.format("ERROR: Selected frequency %s not available for CPU %d! Please select one of\n%s", frequency, cpulist[i], table.concat(availfreqs, ", ")))
            os.exit(1)
        end
        local f = likwid.setCpuClockCurrent(cpulist[i], tonumber(frequency)*1E6)
    end
end

if governor then
    local govs = likwid.getAvailGovs(cpulist[1])
    local cur_govs = {}
    for i,c in pairs(cpulist) do
        table.insert(cur_govs, likwid.getGovernor(cpulist[1]))
    end
    
    local valid_gov = false
    for k,v in pairs(govs) do
        if (governor == v) then
            valid_gov = true
            break
        end
    end
    local cur_freqs = {}
    if governor == "turbo" and availturbo ~= "0" then
        valid_gov = true
        governor = "performance"
        for i=1,#cpulist do
            cur_freqs[cpulist[i]] = availturbo
        end
    end
    if not valid_gov then
        print_stderr(string.format("ERROR: Governor %s not available! Please select one of\n%s", governor, table.concat(govs, ", ")))
        os.exit(1)
    end
    for i=1,#cpulist do
        if governor ~= cur_govs[i] then
            local f = likwid.setGovernor(cpulist[i], governor)
        end
    end
end
likwid.putAffinityInfo()
likwid.putTopology()
os.exit(0)
