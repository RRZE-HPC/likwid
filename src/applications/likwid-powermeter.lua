#!/usr/bin/env lua

--[[
 * =======================================================================================
 *
 *      Filename:  likwid-powermeter.lua
 *
 *      Description:  An application to get information about power 
 *      consumption on architectures implementing the RAPL interface.
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

require("liblikwid")
local likwid = require("likwid")

local HLINE = string.rep("-",80)
local SLINE = string.rep("*",80)

local function version()
    print(string.format("likwid-powermeter --  Version %d.%d",VERSION,RELEASE))
end

local function examples()
    print("Examples:")
    print("Measure the power consumption for 4 seconds on socket 1")
    print("likwid-powermeter -s 4 -c 1")
    print("")
    print("Use it as wrapper for an application to measure the energy for the whole execution")
    print("likwid-powermeter -c 1 ./a.out")
end

local function usage()
    version()
    print("A tool to print power and clocking information on x86 CPUs.\n")
    print("Options:")
    print("-h\t\t Help message")
    print("-v\t\t Version information")
    print("-M <0|1>\t Set how MSR registers are accessed, 0=direct, 1=msrd")
    print("-c <list>\t Specify sockets to measure")
    print("-i\t\t Print information from MSR_PKG_POWER_INFO register and Turbo mode")
    print("-s <duration>\t Set measure duration in sec. (default 2s)")
    print("-p\t\t Print dynamic clocking and CPI values")
    print("")
    examples()
end


print_info = false
use_perfctr = false
stethoscope = false
fahrenheit = false
access_mode = 1
time_interval = 2
sockets = {}
eventString = "PWR_PKG_ENERGY:PWR0,PWR_PP0_ENERGY:PWR1,PWR_DRAM_ENERGY:PWR3"

cpuinfo = likwid_getCpuInfo()
cputopo = likwid_getCpuTopology()
numatopo = likwid_getNumaInfo()

for opt,arg in likwid.getopt(arg, "c:hiM:ps:vf") do
    if (opt == "h") then
        usage()
        os.exit(0)
    elseif (opt == "v") then
        version()
        os.exit(0)
    elseif (opt == "c") then
        if (arg:find(",") ~= nil) then
            sockets = arg:split(",")
            for i,socket in pairs(sockets) do
                sockets[i] = tonumber(socket)
                if (sockets[i] == nil) then
                    print("All entries of the socket list must be numbers, entry " .. socket .. " is no number.")
                    usage()
                    os.exit(1)
                elseif (sockets[i] < 0) then
                    print("Only entries greater than 0 are allowed")
                    usage()
                    os.exit(1)
                elseif (sockets[i] >= cputopo["numSockets"]) then
                    print("Socket " .. sockets[i] .. " does not exist")
                    usage()
                    os.exit(1)
                end
            end
        else
            local socket = tonumber(arg)
            if (socket == nil) then
                print("All entries of the socket list must be numbers, entry " .. socket .. " is no number.")
                usage()
                os.exit(1)
            elseif (socket < 0) then
                print("Only entries greater than 0 are allowed")
                usage()
                os.exit(1)
            elseif (socket >= cputopo["numSockets"]) then
                print("Socket " .. socket .. " does not exist")
                os.exit(1)
            end
            table.insert(sockets,socket)
        end
    elseif (opt == "M") then
        access_mode = tonumber(arg)
        if (access_mode == nil) then
            print("Access mode (-M) must be an number")
            usage()
            os.exit(1)
        elseif (access_mode < 0) or (access_mode > 1) then
            print(string.format("Access mode (-M) %d not valid.",access_mode))
            usage()
            os.exit(1)
        end
        
    elseif (opt == "i") then
        print_info = true
    elseif (opt == "p") then
        use_perfctr = true
    elseif (opt == "f") then
        fahrenheit = true
    elseif (opt == "s") then
        time_interval = tonumber(arg)
        stethoscope = true
        if (time_interval == nil) then
            print("Time interval (-s) must be an number")
            usage()
            os.exit(1)
        elseif (time_interval <= 0) then
            print("Time interval (-s) must be greater than 0")
            usage()
            os.exit(1)
        end
    end
end
accessClient_setaccessmode(access_mode)

cpulist = {}
for i,socketId in pairs(sockets) do
    for j, cpuId in pairs(numatopo[socketId]["processors"]) do
        table.insert(cpulist,cpuId)
    end
end




power = likwid_getPowerInfo()

print(HLINE);
print(string.format("CPU name:\t%s", cpuinfo["name"]))
print(string.format("CPU clock:\t%3.2f GHz",likwid_getCpuClock() *  1.E-09))
print(HLINE);

if (print_info) then
    if (power["turbo"]["numSteps"] > 0) then
        print(string.format("Base clock:\t%.2f MHz", power["baseFrequency"]))
        print(string.format("Minimal clock:\t%.2f MHz", power["minFrequency"]))
        print("Turbo Boost Steps:")
        for i,step in pairs(power["turbo"]["steps"]) do
            print(string.format("C%d %.2f MHz",i,power["turbo"]["steps"][i]))
        end
    end
    print(HLINE)
end

hasDRAM = 0

if (cpuinfo["model"] == 45 or cpuinfo["model"] == 62) then
    hasDRAM = 1
elseif (cpuinfo["model"] ~= 45 and
        cpuinfo["model"] ~= 42 and
        cpuinfo["model"] ~= 58 and
        cpuinfo["model"] ~= 62 and
        cpuinfo["model"] ~= 60) then
    print("RAPL not supported for this processor")
    os.exit(1)
end

if (print_info) then
    print(string.format("Thermal Spec Power: %g Watts",power["tdp"]))
    print(string.format("Minimum  Power: %g Watts",power["minPower"]))
    print(string.format("Maximum  Power: %g Watts",power["maxPower"]))
    print(string.format("Maximum  Time Window: %g micro sec",power["maxTimeWindow"]))
    print(HLINE)
    os.exit(0)
end

if (use_perfctr) then
    affinity = likwid_getAffinityInfo
    argString = ""
    for i,socket in pairs(sockets) do
        argString = argString .. string.format("S%u:0-%u",socket,(cputopo["numCoresPerSocket"]*cputopo["numThreadsPerCore"])-1)
        if (i < #sockets) then
            argString = argString .. "@"
        end
    end
    likwid_init(likwid.tablelength(cpulist), cpulist)
    group = likwid_addEventSet(eventString)
    likwid_setupCounters(group)
end

for i,socket in pairs(sockets) do
    cpu = numatopo[socket]["processors"][0]
    print(string.format("Measure for socket %d on cpu %d", socket,cpu ))
    if (stethoscope) then
        if (use_perfctr) then
            likwid_startCounters()
        end
        
        if (hasDRAM) then before3 = likwid_startPower(cpu, 3) end
        before0 = likwid_startPower(cpu, 0)
        sleep(time_interval);
        after0 = likwid_stopPower(cpu, 0)
        if (hasDRAM) then 
            after3 = likwid_stopPower(cpu, 3) 
        end
        
        if (use_perfctr) then
            likwid_stopCounters()
            likwid_finalize()
        end
        runtime = time_interval;
    else
        execString = ""
        for i, str in pairs(arg) do
            if (str == "lua") then break end
            execString = execString .. str .. " "
        end

        if (use_perfctr) then
            likwid_startCounters()
        end
        
        if (hasDRAM) then before3 = likwid_startPower(cpu, 3) end
        before0 = likwid_startPower(cpu, 0)
        time_before = likwid_startClock()

        err = os.execute(execString)

        if (err == false) then
            print(string.format("Failed to execute %s!",execString))
            likwid_stopCounters()
            likwid_finalize()
            likwid_putNumaInfo()
            likwid_putAffinityInfo()
            likwid_putTopology()
            os.exit(1)
        end
        after0 = likwid_stopPower(cpu, 0)
        if (hasDRAM) then 
            after3 = likwid_stopPower(cpu, 3)
        end
        if (use_perfctr) then
            likwid_stopCounters()
            likwid_finalize()
        end
        time_after = likwid_stopClock()
        runtime = likwid_getClock(time_before, time_after)
    end
    print(string.format("Runtime: %g s",runtime))
    print("Domain: PKG")
    energy = (after0 - before0) * power["energyUnit"]
    print(string.format("Energy consumed: %g Joules",energy))
    print(string.format("Power consumed: %g Watts",energy/runtime))
    if (hasDRAM) then
        print("Domain: DRAM")
        energy = (after3 - before3) * power["energyUnit"]
        print(string.format("Energy consumed: %g Joules",energy));
        print(string.format("Power consumed: %g Watts",energy/runtime))
    end
    if (string.find(cpuinfo["features"],"TM2") ~= nil) then
        print(HLINE)
        likwid_initTemp(cpu);
        print("Current core temperatures:");
        for i=0,(cputopo["numCoresPerSocket"]*cputopo["numThreadsPerCore"])-1 do
            cpuid = numatopo[socket]["processors"][i]
            if (fahrenheit) then
                print(string.format("Core %d: %u °F",cpuid, 1.8*likwid_readTemp(cpuid)+32));
            else
                print(string.format("Core %d: %u °C",cpuid, likwid_readTemp(cpuid)));
            end
        end
    end
    
end

--likwid_finalize();
likwid_putNumaInfo()
likwid_putAffinityInfo()
likwid_putTopology()

