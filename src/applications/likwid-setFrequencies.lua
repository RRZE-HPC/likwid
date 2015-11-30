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

sys_base_path = "/sys/devices/system/cpu"
set_command = "<INSTALLED_PREFIX>/sbin/likwid-setFreq"


function version()
    print(string.format("likwid-setFrequencies --  Version %d.%d",likwid.version,likwid.release))
end

function usage()
    version()
    print("A tool to adjust frequencies and governors on x86 CPUs.\n")
    print("Options:")
    print("-h\t Help message")
    print("-v\t Version information")
    print("-c dom\t Likwid thread domain which to apply settings (default are all CPUs)")
    print("\t See likwid-pin -h for details")
    print("-g gov\t Set governor (" .. table.concat(getAvailGovs(nil), ", ") .. ") (set to ondemand if omitted)")
    print("-f freq\t Set fixed frequency, implicitly sets userspace governor")
    print("-p\t Print current frequencies")
    print("-l\t List available frequencies")
    print("-m\t List available governors")
end

function getCurrentMinFreq(cpuid)
    local min = 10000000
    if cpuid == nil or cpuid < 0 then
        for cpuid=0,topo["numHWThreads"]-1 do
            fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_min_freq")
            if verbosity == 3 then
                print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_min_freq" )
            end
            line = fp:read("*l")
            if tonumber(line)/1E6 < min then
                min = tonumber(line)/1E6
            end
            fp:close()
        end
    else
        fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_min_freq")
        if verbosity == 3 then
            print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_min_freq" )
        end
        line = fp:read("*l")
        if tonumber(line)/1E6 < min then
            min = tonumber(line)/1E6
        end
        fp:close()
    end
    return min
end

function getCurrentMaxFreq(cpuid)
    local max = 0
    if cpuid == nil or cpuid < 0 then
        for cpuid=0,topo["numHWThreads"]-1 do
            fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_max_freq")
            if verbosity == 3 then
                print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_max_freq" )
            end
            line = fp:read("*l")
            if tonumber(line)/1E6 > max then
                max = tonumber(line)/1E6
            end
            fp:close()
        end
    else
        fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_max_freq")
        if verbosity == 3 then
            print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_max_freq" )
        end
        line = fp:read("*l")
        if tonumber(line)/1E6 > max then
            max = tonumber(line)/1E6
        end
        fp:close()
    end
    return max
end


function getAvailFreq(cpuid)
    if cpuid == nil then
        cpuid = 0
    end
    if cpuid < 0 then
        cpuid = 0
    end
    fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_available_frequencies")
    if verbosity == 3 then
        print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_available_frequencies" )
    end
    line = fp:read("*l")
    fp:close()
    
    local tmp = likwid.stringsplit(line:gsub("^%s*(.-)%s*$", "%1"), " ", nil, " ")
    local avail = {}
    local turbo = tonumber(tmp[1])/1E6
    local j = 1
    for i=2,#tmp do
        local freq = tonumber(tmp[i])/1E6
        avail[j] = tostring(freq)
        if not avail[j]:match("%d+.%d+") then
            avail[j] = avail[j] ..".0"
        end
        j = j + 1
    end
    if verbosity == 1 then
        print(string.format("The system provides %d scaling frequencies, frequency %s is taken as turbo mode", #avail,turbo))
    end
    return avail, tostring(turbo)
end

function getCurFreq()
    local freqs = {}
    local govs = {}
    for cpuid=0,topo["numHWThreads"]-1 do
        local fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_cur_freq")
        if verbosity == 3 then
            print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_cur_freq" )
        end
        local line = fp:read("*l")
        fp:close()
        freqs[cpuid] = tostring(tonumber(line)/1E6)
        if not freqs[cpuid]:match("%d.%d") then
            freqs[cpuid] = freqs[cpuid] ..".0"
        end
        local fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_governor")
        if verbosity == 3 then
            print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_governor" )
        end
        local line = fp:read("*l")
        fp:close()
        govs[cpuid] = line
    end
    return freqs, govs
end

function getAvailGovs(cpuid)
    if (cpuid == nil) or (cpuid < 1) then
        cpuid = 0
    end
    local fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_available_governors")
    if verbosity == 3 then
        print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_available_governors" )
    end
    local line = fp:read("*l")
    fp:close()
    local avail = likwid.stringsplit(line:gsub("^%s*(.-)%s*$", "%1"), "%s+", nil, "%s+")
    for i=1,#avail do
        if avail[i] == "userspace" then
            table.remove(avail, i)
            break
        end
    end
    table.insert(avail, "turbo")
    if verbosity == 1 then
        print(string.format("The system provides %d scaling governors", #avail))
    end
    return avail
end

local function testDriver()
    local fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",0) .. "/cpufreq/scaling_driver")
    if verbosity == 3 then
        print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",0) .. "/cpufreq/scaling_driver" )
    end
    local line = fp:read("*l")
    fp:close()
    if line == "acpi-cpufreq" then
        return true
    end
    return false
end

verbosity = 0
governor = nil
frequency = nil
domain = nil
printCurFreq = false
printAvailFreq = false
printAvailGovs = false

if #arg == 0 then
    usage()
    os.exit(0)
end


for opt,arg in likwid.getopt(arg, {"g:", "c:", "f:", "l", "p", "h", "v", "m", "help","version","freq:"}) do
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
    elseif (opt == "p") then
        printCurFreq = true
    elseif (opt == "l") then
        printAvailFreq = true
    elseif (opt == "m") then
        printAvailGovs = true
    elseif opt == "?" then
        print("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print("Option requires an argument")
        os.exit(1)
    end
end
if not testDriver() then
    print("The system does not use the acpi-cpufreq driver, other drivers are not usable with likwid-setFrequencies.")
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
numthreads, cpulist = likwid.cpustr_to_cpulist(domain)
if verbosity == 3 then
    print(string.format("Given CPU expression expands to %d CPU cores:", numthreads))
    local str = tostring(cpulist[1])
    for i=2, numthreads  do
        str = str .. "," .. tostring(cpulist[i])
    end
    print(str)
end


if printAvailGovs then
    local govs = getAvailGovs(nil)
    print("Available governors:")
    print(table.concat(govs, ", "))
end

if printAvailFreq then
    print("Available frequencies:")
    local out = {}
    local i = 1;
    local freqs, turbo = getAvailFreq(nil)
    if turbo ~= "0" then
        table.insert(out, turbo)
    end
    for i=1,#freqs do
        table.insert(out, freqs[i])
    end

    print(table.concat(out, " "))
end

if printCurFreq then
    print("Current frequencies:")
    local freqs = {}
    local govs = {}
    freqs, govs = getCurFreq()
    for i=1,#cpulist do
        print(string.format("CPU %d: governor %12s frequency %5s GHz",cpulist[i],govs[cpulist[i]], freqs[cpulist[i]]))
    end
end

if printAvailGovs or printAvailFreq or printCurFreq then
    os.exit(0)
end

if numthreads > 0 and not (frequency or governor) then
    print("You need to set either a frequency or governor for the selected CPUs on commandline")
    os.exit(1)
end

if frequency then
    for i=1,#cpulist do
        local freqs, turbo = getAvailFreq(cpulist[i])
        local valid_freq = false
        for k,v in pairs(freqs) do
            if (frequency == v) then
                valid_freq = true
                break
            end
        end
        if frequency == turbo then
            valid_freq = true
        end
        if not valid_freq then
            print(string.format("Frequency %s not available for CPU %d! Please select one of\n%s", frequency, cpulist[i], table.concat(freqs, ", ")))
            os.exit(1)
        end
    
        local cmd = set_command .. " " .. tostring(cpulist[i]) .. " " .. tostring(tonumber(frequency)*1E6)
        if governor then
            cmd = cmd .. " " .. governor
        end
        if verbosity == 3 then
            print("Execute: ".. cmd)
        end
        local err = os.execute(cmd)
        if err == false or err == nil then
            print("Failed to set frequency for CPU "..tostring(cpulist[i]))
        end
    end
    if governor then
        governor = nil
    end
end

if governor then
    local govs = getAvailGovs(nil)
    local freqs, turbo = getAvailFreq(nil)
    local cur_freqs, cur_govs = getCurFreq()
    local valid_gov = false
    for k,v in pairs(govs) do
        if (governor == v) then
            valid_gov = true
            break
        end
    end
    if governor == "turbo" and turbo ~= "0" then
        valid_gov = true
        for i=1,#cpulist do
            cur_freqs[cpulist[i]] = turbo
        end
    end
    if not valid_gov then
        print(string.format("Governor %s not available! Please select one of\n%s", governor, table.concat(govs, ", ")))
        os.exit(1)
    end
    for i=1,#cpulist do
        if governor ~= cur_govs[cpulist[i]] then
            local cmd = set_command .. " " .. tostring(cpulist[i]) .. " "
            if governor == "turbo" then
                cmd = cmd .. tostring(tonumber(turbo)*1E6)
            else
                cmd = cmd .. tostring(tonumber(cur_freqs[cpulist[i]])*1E6) .. " " .. governor
            end
            if verbosity == 3 then
                print("Execute: ".. cmd)
            end
            local err = os.execute(cmd)
            if err == false or err == nil then
                print("Failed to set governor for CPU "..tostring(cpulist[i]))
            end
        end
    end
end
likwid.putAffinityInfo()
likwid.putTopology()
os.exit(0)
