#!<PREFIX>/bin/likwid-lua

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

package.path = package.path .. ';<PREFIX>/share/lua/?.lua'

local likwid = require("likwid")

sys_base_path = "/sys/devices/system/cpu"
set_command = "<PREFIX>/sbin/likwid-setFreq"


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

function getAvailFreq(cpuid)
    if (cpuid == nil) or (cpuid < 1) then
        cpuid = 0
    end
    local fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_available_frequencies")
    if verbosity == 3 then
        print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_available_frequencies" )
    end
    local line = fp:read("*l")
    fp:close()
    local tmp = likwid.stringsplit(line:gsub("^%s*(.-)%s*$", "%1"), " ", nil, " ")
    local avail = {}
    local turbo = tostring(tonumber(tmp[1])/1E6)
    for i=2,#tmp do
        avail[i-1] = tostring(tonumber(tmp[i])/1E6)
        if not avail[i-1]:match("%d.%d") then
            avail[i-1] = avail[i-1] ..".0"
        end
    end
    if verbosity == 1 then
        print(string.format("The system provides %d scaling frequencies, frequency %s is taken as turbo mode", #avail,turbo))
    end
    return avail, turbo
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
    local fp = io.open(sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_driver")
    if verbosity == 3 then
        print("Reading "..sys_base_path .. "/" .. string.format("cpu%d",cpuid) .. "/cpufreq/scaling_driver" )
    end
    local line = fp:read("*l")
    fp:close()
    if line:match("acpi-cpufreq") then
        return true
    end
    return false
end

verbosity = 3
governor = nil
frequency = nil
domain = nil
printCurFreq = false
printAvailFreq = false
printAvailGovs = false


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
    end
end
if not testDriver() then
    print("The system does not use the acpi-cpufreq driver, other drivers are not usable with likwid-setFrequencies.")
    os.exit(1)
end

topo = likwid.getCpuTopology()
affinity = likwid.getAffinityInfo()
if not domain then
    domain = "N:1-" .. tostring(topo["numHWThreads"])
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
    local freqs, turbo = getAvailFreq(nil)
    print("Available frequencies:")
    print(turbo .. ", " .. table.concat(freqs, ", "))
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

if frequency then
    local freqs, turbo = getAvailFreq(nil)
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
        print(string.format("Frequency %s not available! Please select one of\n%s", frequency, table.concat(freqs, ", ")))
        os.exit(1)
    end
    for i=1,#cpulist do
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
    if (governor == "turbo") then
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
