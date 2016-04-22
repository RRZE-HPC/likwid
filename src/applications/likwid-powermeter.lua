#!<INSTALLED_BINPREFIX>/likwid-lua
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
time_orig = "2s"
read_interval = 30.E06
sockets = {}
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
        print_temp = true
    elseif (opt == "t") then
        print_temp = true
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
        likwid.setVerbosity(verbose)
    elseif (opt == "s") then
        time_interval = likwid.parse_time(arg)
        time_orig = arg
        stethoscope = true
    elseif opt == "?" then
        print("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print("Option requires an argument")
        os.exit(1)
    end
end



cpulist = {}
before = {}
after = {}
if #sockets > 0 then
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
else
    for j, domain in pairs(affinity["domains"]) do
        if domain["tag"]:match("S%d+") then
            table.insert(cpulist,domain["processorList"][1])
            table.insert(sockets, domain["tag"]:match("S(%d+)"))
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


if not use_perfctr then
    print(likwid.hline);
    print(string.format("CPU name:\t%s",cpuinfo["osname"]))
    print(string.format("CPU type:\t%s",cpuinfo["name"]))
    if cpuinfo["clock"] > 0 then
        print(string.format("CPU clock:\t%3.2f GHz",cpuinfo["clock"] *  1.E-09))
    else
        print(string.format("CPU clock:\t%3.2f GHz",likwid.getCpuClock() *  1.E-09))
    end
    print(likwid.hline)
end

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
            print(string.format("Thermal Spec Power: %g Watt",domain["tdp"]*1E-6))
            print(string.format("Minimum Power: %g Watt",domain["minPower"]*1E-6))
            print(string.format("Maximum Power: %g Watt",domain["maxPower"]*1E-6))
            print(string.format("Maximum Time Window: %g micro sec",domain["maxTimeWindow"]))
            print()
        end
    end
    print(likwid.hline)
end

if (stethoscope) and (time_interval < power["timeUnit"]) then
    print("Time interval too short, minimum measurement time is "..tostring(power["timeUnit"]).. " us")
    os.exit(1)
end

local execString = ""
if use_perfctr then
    affinity = likwid.getAffinityInfo()
    argString = ""
    for i,socket in pairs(sockets) do
        argString = argString .. string.format("S%u:0-%u",socket,(cputopo["numCoresPerSocket"]*cputopo["numThreadsPerCore"])-1)
        if (i < #sockets) then
            argString = argString .. "@"
        end
    end
    execString = string.format("<INSTALLED_PREFIX>/bin/likwid-perfctr -C %s -f -g CLOCK ",argString)
end


if #arg == 0 then
    if use_perfctr then
        execString = execString .. string.format(" -S %s ", time_orig)
        stethoscope = false
    else
        stethoscope = true
    end
else
    if use_perfctr then
        execString = execString .. table.concat(arg," ",1, likwid.tablelength(arg)-2)
    else
        execString = table.concat(arg," ",1, likwid.tablelength(arg)-2)
    end
end

if not print_info and not print_temp then
    if stethoscope or (#arg > 0 and not use_perfctr) then
        for i,socket in pairs(sockets) do
            cpu = cpulist[i]
            for idx, dom in pairs(domainList) do
                if (power["domains"][dom]["supportStatus"]) then before[cpu][dom] = likwid.startPower(cpu, idx) end
            end
        end

        time_before = likwid.startClock()
        if stethoscope then
            if read_interval < time_interval then
                while ((read_interval <= time_interval) and (time_interval > 0)) do
                    likwid.sleep(read_interval)
                    for i,socket in pairs(sockets) do
                        cpu = cpulist[i]
                        for idx, dom in pairs(domainList) do
                            if (power["domains"][dom]["supportStatus"]) then after[cpu][dom] = likwid.stopPower(cpu, idx) end
                        end
                    end
                    time_interval = time_interval - read_interval
                    if time_interval < read_interval then
                        read_interval = time_interval
                    end
                end
            else
                likwid.sleep(time_interval)
            end
        else
            local pid = likwid.startProgram(execString, 0, {})
            if not pid then
                print(string.format("Failed to execute %s!",execString))
                likwid.finalize()
                os.exit(1)
            end
            while true do
                if likwid.getSignalState() ~= 0 then
                    likwid.killProgram()
                    break
                end
                local remain = likwid.sleep(read_interval)
                for i,socket in pairs(sockets) do
                    cpu = cpulist[i]
                    for idx, dom in pairs(domainList) do
                        if (power["domains"][dom]["supportStatus"]) then after[cpu][dom] = likwid.stopPower(cpu, idx) end
                    end
                end
                if remain > 0 or not likwid.checkProgram() then
                    io.stdout:flush()
                    break
                end
            end
        end
        time_after = likwid.stopClock()

        for i,socket in pairs(sockets) do
            cpu = cpulist[i]
            for idx, dom in pairs(domainList) do
                if (power["domains"][dom]["supportStatus"]) then after[cpu][dom] = likwid.stopPower(cpu, idx) end
            end
        end
        runtime = likwid.getClock(time_before, time_after)

        print(likwid.hline)
        print(string.format("Runtime: %g s",runtime))

        for i,socket in pairs(sockets) do
            cpu = cpulist[i]
            print(string.format("Measure for socket %d on CPU %d", socket,cpu ))
            for j, dom in pairs(domainList) do
                if power["domains"][dom]["supportStatus"] then
                    local energy = likwid.calcPower(before[cpu][dom], after[cpu][dom], 0)
                    print(string.format("Domain %s:", dom))
                    print(string.format("Energy consumed: %g Joules",energy))
                    print(string.format("Power consumed: %g Watt",energy/runtime))
                end
            end
            if i < #sockets then print("") end
        end
        print(likwid.hline)
    else
        err = os.execute(execString)
        if err == false then
            print(string.format("Failed to execute %s!",execString))
            likwid.putPowerInfo()
            likwid.finalize()
            os.exit(1)
        end
    end
end

if print_temp and (string.find(cpuinfo["features"],"TM2") ~= nil) then
    print(likwid.hline)
    print("Current core temperatures:");
    for i=1,cputopo["numSockets"] do
        local tag = "S" .. tostring(i-1)
        for _, domain in pairs(affinity["domains"]) do
            if domain["tag"] == tag then
                for j=1,#domain["processorList"] do
                    local cpuid = domain["processorList"][j]
                    likwid.initTemp(cpuid);
                    if (fahrenheit) then
                        local f = 1.8*tonumber(likwid.readTemp(cpuid))+32
                        print(string.format("Socket %d Core %d: %.0f F",i-1,cpuid, f));
                    else
                        print(string.format("Socket %d Core %d: %.0f C",i-1,cpuid, tonumber(likwid.readTemp(cpuid))));
                    end
                end
            end
        end
    end
    print(likwid.hline)
end

likwid.putPowerInfo()
likwid.finalize()
