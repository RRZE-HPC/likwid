#!<PREFIX>/bin/likwid-lua

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
package.path = package.path .. ';<PREFIX>/share/lua/?.lua'

local likwid = require("likwid")

local function version()
    print(string.format("likwid-powermeter --  Version %d.%d",likwid.version,likwid.release))
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
    print("-h, --help\t Help message")
    print("-v, --version\t Version information")
    print("-V, --verbose <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print("-M <0|1>\t\t Set how MSR registers are accessed, 0=direct, 1=accessDaemon")
    print("-c <list>\t\t Specify sockets to measure")
    print("-i, --info\t Print information from MSR_PKG_POWER_INFO register and Turbo mode")
    print("-s <duration>\t Set measure duration in us, ms or s. (default 2s)")
    print("-p\t\t Print dynamic clocking and CPI values, uses likwid-perfctr")
    print("-t\t\t Print current temperatures of all CPU cores")
    print("-f\t\t Print current temperatures in Fahrenheit")
    print("")
    examples()
end

local config = likwid.getConfiguration();

print_info = false
use_perfctr = false
stethoscope = false
fahrenheit = false
print_temp = false
verbose = 0
if config["daemonMode"] < 0 then
    access_mode = 1
else
    access_mode = config["daemonMode"]
end
time_interval = 2.E06
sockets = {0}
eventString = "PWR_PKG_ENERGY:PWR0,PWR_PP0_ENERGY:PWR1,PWR_PP1_ENERGY:PWR2,PWR_DRAM_ENERGY:PWR3"
domainList = {"PKG", "PP0", "PP1", "DRAM"}

cpuinfo = likwid.getCpuInfo()
cputopo = likwid.getCpuTopology()
numatopo = likwid.getNumaInfo()
affinity = likwid_getAffinityInfo()

for opt,arg in likwid.getopt(arg, {"V:", "c:", "h", "i", "M:", "p", "s:", "v", "f", "t", "help", "info", "version", "verbose:"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-");
        if s == 1 then
            print(string.format("Argmument %s to option -%s starts with invalid character -.", arg, opt))
            print("Did you forget an argument to an option?")
            os.exit(1)
        end
    end
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif (opt == "c") then
        num_sockets, sockets = likwid.sockstr_to_socklist(arg)
        if num_sockets == 0 then
            os.exit(1)
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
        
    elseif opt == "i" or opt == "info" then
        print_info = true
    elseif (opt == "p") then
        use_perfctr = true
    elseif (opt == "f") then
        fahrenheit = true
    elseif (opt == "t") then
        print_temp = true
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
        likwid.setVerbosity(verbose)
    elseif (opt == "s") then
        time_interval = likwid.parse_time(arg)
        stethoscope = true
    end
end



cpulist = {}
before = {}
after = {}
for i,socketId in pairs(sockets) do
    local affinityID = "S"..tostring(socketId)
    for j, domain in pairs(affinity["domains"]) do
        if domain["tag"] == affinityID then
            table.insert(cpulist,domain["processorList"][1])
            before[domain["processorList"][1]] = {}
            after[domain["processorList"][1]] = {}
            for _, id in pairs(domainList) do
                before[domain["processorList"][1]][id] = 0
                after[domain["processorList"][1]][id] = 0
            end
        end
    end
end


if likwid.setAccessClientMode(access_mode) ~= 0 then
    os.exit(1)
end

power = likwid.getPowerInfo()
if not power then
    print(string.format("The %s does not support reading power data",cpuinfo["name"]))
    os.exit(1)
end



print(likwid.hline);
print(string.format("CPU name:\t%s",cpuinfo["osname"]))
print(string.format("CPU type:\t%s",cpuinfo["name"]))
if cpuinfo["clock"] > 0 then
    print(string.format("CPU clock:\t%3.2f GHz",cpuinfo["clock"] *  1.E-09))
else
    print(string.format("CPU clock:\t%3.2f GHz",likwid.getCpuClock() *  1.E-09))
end
print(likwid.hline);

if print_info or verbose > 0 then
    if (power["turbo"]["numSteps"] > 0) then
        print(string.format("Base clock:\t%.2f MHz", power["baseFrequency"]))
        print(string.format("Minimal clock:\t%.2f MHz", power["minFrequency"]))
        print("Turbo Boost Steps:")
        for i,step in pairs(power["turbo"]["steps"]) do
            print(string.format("C%d %.2f MHz",i-1,power["turbo"]["steps"][i]))
        end
    end
    print(likwid.hline)
end

if power["hasRAPL"] == 0 then
    print("Measuring power is not supported on this machine")
    os.exit(1)
end

if (print_info) then
    for i, dname in pairs(domainList) do
        local domain = power["domains"][dname]
        if domain["supportInfo"] then
            print(string.format("Info for RAPL domain %s:", dname));
            print(string.format("Thermal Spec Power: %g Watts",domain["tdp"]))
            print(string.format("Minimum Power: %g Watts",domain["minPower"]))
            print(string.format("Maximum Power: %g Watts",domain["maxPower"]))
            print(string.format("Maximum Time Window: %g micro sec",domain["maxTimeWindow"]))
            print()
        end
    end
    print(likwid.hline)
    os.exit(0)
end

if (stethoscope) and (time_interval < power["timeUnit"]*1.E06) then
    print("Time interval too short, minimum measurement time is "..tostring(power["timeUnit"]*1.E06).. " us")
    os.exit(1)
end

if (use_perfctr) then
    affinity = likwid.getAffinityInfo()
    argString = ""
    for i,socket in pairs(sockets) do
        argString = argString .. string.format("S%u:0-%u",socket,(cputopo["numCoresPerSocket"]*cputopo["numThreadsPerCore"])-1)
        if (i < #sockets) then
            argString = argString .. "@"
        end
    end
    likwid.init(likwid.tablelength(cpulist), cpulist)
    group = likwid.addEventSet(eventString)
    likwid.setupCounters(group)
end

local execString = ""
if #arg == 0 then
    stethoscope = true
else
    for i=1,#arg do
        execString = execString .. arg[i] .. " "
    end
end

if (use_perfctr) then
    likwid.startCounters()
else
    for i,socket in pairs(sockets) do
        cpu = cpulist[i]
        for idx, dom in pairs(domainList) do
            if (power["domains"][dom]["supportStatus"]) then before[cpu][dom] = likwid.startPower(cpu, idx) end
        end
    end
end
time_before = likwid.startClock()
if (stethoscope) then
    if time_interval >= 1.E06 then
        sleep(time_interval/1.E06)
    else
        usleep(time_interval)
    end
else
    err = os.execute(execString)
    if (err == false) then
        print(string.format("Failed to execute %s!",execString))
        likwid.stopCounters()
        likwid.finalize()
        likwid.putNumaInfo()
        likwid.putAffinityInfo()
        likwid.putTopology()
        os.exit(1)
    end
end
time_after = likwid.stopClock()

if (use_perfctr) then
    likwid.stopCounters()
    likwid.finalize()
else
    for i,socket in pairs(sockets) do
        cpu = cpulist[i]
        for idx, dom in pairs(domainList) do
            if (power["domains"][dom]["supportStatus"]) then after[cpu][dom] = likwid.stopPower(cpu, idx) end
        end
    end
end
runtime = likwid.getClock(time_before, time_after)
print(string.format("Runtime: %g s",runtime))

for i,socket in pairs(sockets) do
    cpu = cpulist[i]
    print(string.format("Measure for socket %d on CPU %d", socket,cpu ))
    for _, dom in pairs(domainList) do
        if power["domains"][dom]["supportStatus"] then
            print("Domain: "..dom)
            local energy = likwid.calcPower(before[cpu][dom], after[cpu][dom], 0)
            print(string.format("Energy consumed: %g Joules",energy))
            print(string.format("Power consumed: %g Watts",energy/runtime))
        end
    end
end

if print_temp and (string.find(cpuinfo["features"],"TM2") ~= nil) then
    print(likwid.hline)
    likwid.initTemp(cpulist[i]);
    print("Current core temperatures:");
    for i=1,cputopo["numSockets"] do
        local tag = "S" .. tostring(i-1)
        for _, domain in pairs(affinity["domains"]) do
            if domain["tag"] == tag then
                for j=1,#domain["processorList"] do
                    local cpuid = domain["processorList"][j]
                    if (fahrenheit) then
                        print(string.format("Socket %d Core %d: %u F",i-1,cpuid, 1.8*likwid.readTemp(cpuid)+32));
                    else
                        print(string.format("Socket %d Core %d: %u C",i-1,cpuid, likwid.readTemp(cpuid)));
                    end
                end
            end
        end
    end
end
    


likwid.putPowerInfo()
likwid.putNumaInfo()
likwid.putAffinityInfo()
likwid.putTopology()
likwid.putConfiguration()
