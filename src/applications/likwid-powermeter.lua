#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-powermeter.lua
 *
 *      Description:  An application to get information about power
 *      consumption on architectures implementing the RAPL interface.
 *
 *      Version:   5.1
 *      Released:  16.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
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

print_stdout = print
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(v .. "\n") end end

local function version()
    print_stdout(string.format("likwid-powermeter -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

local function examples()
    print_stdout("Examples:")
    print_stdout("Measure the power consumption for 4 seconds on socket 1")
    print_stdout("likwid-powermeter -s 4 -c 1")
    print_stdout("")
    print_stdout("Use it as wrapper for an application to measure the energy for the whole execution")
    print_stdout("likwid-powermeter -c 1 ./a.out")
end

local function usage()
    version()
    print_stdout("A tool to print power and clocking information on x86 CPUs.\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t Help message")
    print_stdout("-v, --version\t Version information")
    print_stdout("-V, --verbose <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print_stdout("-M <0|1>\t\t Set how MSR registers are accessed, 0=direct, 1=accessDaemon")
    print_stdout("-c <list>\t\t Specify sockets to measure")
    print_stdout("-i, --info\t Print information from MSR_PKG_POWER_INFO register and Turbo mode")
    print_stdout("-s <duration>\t Set measure duration in us, ms or s. (default 2s)")
    print_stdout("-p\t\t Print dynamic clocking and CPI values, uses likwid-perfctr")
    print_stdout("-t\t\t Print current temperatures of all CPU cores")
    print_stdout("-f\t\t Print current temperatures in Fahrenheit")
    print_stdout("")
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
raw_selection = nil
cpuinfo = likwid.getCpuInfo()
if cpuinfo["isIntel"] == 1 then
    domainList = {"PKG", "PP0", "PP1", "DRAM", "PLATFORM"}
else
    domainList = {"CORE", "PKG"}
end
cputopo = likwid.getCpuTopology()
numatopo = likwid.getNumaInfo()
affinity = likwid_getAffinityInfo()

for opt,arg in likwid.getopt(arg, {"V:", "c:", "h", "i", "M:", "p", "s:", "v", "f", "t", "help", "info", "version", "verbose:"}) do
    if (type(arg) == "string") then
        local s,e = arg:find("-");
        if s == 1 then
            print_stderr(string.format("Argument %s to option -%s starts with invalid character -.", arg, opt))
            print_stderr("Did you forget an argument to an option?")
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
        raw_selection = arg
    elseif (opt == "M") then
        access_mode = tonumber(arg)
        if (access_mode == nil) then
            print_stderr("Access mode (-M) must be an number")
            usage()
            os.exit(1)
        elseif (access_mode < 0) or (access_mode > 1) then
            print_stderr(string.format("Access mode (-M) %d not valid.",access_mode))
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
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
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
                if #domain["processorList"] > 0 then
                    local c = domain["processorList"][1]
                    table.insert(cpulist,domain["processorList"][1])
                    before[domain["processorList"][1]] = {}
                    after[domain["processorList"][1]] = {}
                    for _, id in pairs(domainList) do
                        before[domain["processorList"][1]][id] = 0
                        after[domain["processorList"][1]][id] = 0
                    end
                else
                    print_stderr(string.format("No CPU available in domain %s", domain["tag"]))
                end
            end
        end
    end
else
    for j, domain in pairs(affinity["domains"]) do
        if domain["tag"]:match("S%d+") then
            if #domain["processorList"] > 0 then
                table.insert(cpulist,domain["processorList"][1])
                table.insert(sockets, domain["tag"]:match("S(%d+)"))
                before[domain["processorList"][1]] = {}
                after[domain["processorList"][1]] = {}
                for _, id in pairs(domainList) do
                    before[domain["processorList"][1]][id] = 0
                    after[domain["processorList"][1]][id] = 0
                end
            else
                print_stderr(string.format("No CPU available in domain %s", domain["tag"]))
            end
        end
    end
end


if likwid.setAccessClientMode(access_mode) ~= 0 then
    os.exit(1)
end

if #sockets == 0 or #cpulist == 0 then
    if raw_selection then
        print_stderr(string.format("No CPU accessible for selection -c %s", raw_selection))
    else
        print_stderr(string.format("No CPU accessible in all sockets"))
    end
    os.exit(1)
end

power = likwid.getPowerInfo()
if not power then
    print_stderr(string.format("The %s does not support reading power data or access is locked",cpuinfo["name"]))
    os.exit(1)
end


if not use_perfctr then
    print_stdout(likwid.hline);
    print_stdout(string.format("CPU name:\t%s",cpuinfo["osname"]))
    print_stdout(string.format("CPU type:\t%s",cpuinfo["name"]))
    if cpuinfo["clock"] > 0 then
        print_stdout(string.format("CPU clock:\t%3.2f GHz",cpuinfo["clock"] *  1.E-09))
    else
        print_stdout(string.format("CPU clock:\t%3.2f GHz",likwid.getCpuClock() *  1.E-09))
    end
    print_stdout(likwid.hline)
end

if print_info or verbose > 0 then
    if (power["turbo"]["numSteps"] > 0) then
        print_stdout(string.format("Base clock:\t%.2f MHz", power["baseFrequency"]))
        print_stdout(string.format("Minimal clock:\t%.2f MHz", power["minFrequency"]))
        print_stdout("Turbo Boost Steps:")
        for i,step in pairs(power["turbo"]["steps"]) do
            print_stdout(string.format("C%d %.2f MHz",i-1,power["turbo"]["steps"][i]))
        end
    end
    print_stdout(likwid.hline)
end

if power["hasRAPL"] == 0 then
    print_stderr("Measuring power is not supported on this machine")
    os.exit(1)
end

if (print_info) then
    for i, dname in pairs(domainList) do
        local domain = power["domains"][dname]
        if domain and domain["supportInfo"] then
            print_stdout(string.format("Info for RAPL domain %s:", dname));
            print_stdout(string.format("Thermal Spec Power: %g Watt",domain["tdp"]*1E-6))
            print_stdout(string.format("Minimum Power: %g Watt",domain["minPower"]*1E-6))
            print_stdout(string.format("Maximum Power: %g Watt",domain["maxPower"]*1E-6))
            print_stdout(string.format("Maximum Time Window: %g micro sec",domain["maxTimeWindow"]))
            print_stdout()
        end
    end
    if power["minUncoreFreq"] > 0 and power["maxUncoreFreq"] > 0 then
        print_stdout("Info about Uncore:")
        print_stdout(string.format("Minimal Uncore frequency: %g MHz", power["minUncoreFreq"]))
        print_stdout(string.format("Maximal Uncore frequency: %g MHz", power["maxUncoreFreq"]))
        print_stdout()
    end
    if power["perfBias"] then
        print_stdout(string.format("Performance energy bias: %.0f (0=highest performance, 15 = lowest energy)", power["perfBias"]))
        print_stdout()
    end
    print_stdout(likwid.hline)
end

if (stethoscope) and (time_interval < power["timeUnit"]) then
    print_stderr("Time interval too short, minimum measurement time is "..tostring(power["timeUnit"]).. " us")
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

local execList = {}
if #arg == 0 then
    if use_perfctr then
        execString = execString .. string.format(" -S %s ", time_orig)
        stethoscope = false
    else
        stethoscope = true
    end
else
    for i=1, likwid.tablelength(arg)-2 do
        table.insert(execList, arg[i])
    end
    if use_perfctr then
        execString = execString .. table.concat(execList," ")
    else
        execString = table.concat(execList," ")
    end
end

local exitvalue = 0
if not print_info and not print_temp then
    if stethoscope or (#arg > 0 and not use_perfctr) then
        for i,socket in pairs(sockets) do
            cpu = cpulist[i]
            for idx, dom in pairs(domainList) do
                if (power["domains"][dom] and power["domains"][dom]["supportStatus"]) then
                    before[cpu][dom] = likwid.startPower(cpu, idx)
                end
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
                            if (power["domains"][dom] and power["domains"][dom]["supportStatus"]) then
                                after[cpu][dom] = likwid.stopPower(cpu, idx)
                            end
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
            local pid = likwid.startProgram(table.concat(execList," "), 0, {})
            if not pid then
                print_stderr(string.format("Failed to execute %s!",table.concat(execList," ")))
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
                        if (power["domains"][dom] and power["domains"][dom]["supportStatus"]) then
                            after[cpu][dom] = likwid.stopPower(cpu, idx)
                        end
                    end
                end
                exitvalue, exited = likwid.checkProgram(pid)
                if exited then
                    io.stdout:flush()
                    break
                end
            end
        end
        time_after = likwid.stopClock()

        for i,socket in pairs(sockets) do
            cpu = cpulist[i]
            for idx, dom in pairs(domainList) do
                if (power["domains"][dom] and power["domains"][dom]["supportStatus"]) then
                    after[cpu][dom] = likwid.stopPower(cpu, idx)
                end
            end
        end
        runtime = likwid.getClock(time_before, time_after)

        print_stdout(likwid.hline)
        print_stdout(string.format("Runtime: %g s",runtime))

        for i,socket in pairs(sockets) do
            cpu = cpulist[i]
            print_stdout(string.format("Measure for socket %d on CPU %d", socket,cpu ))
            table.concat(domainList, ",")
            for j, dom in pairs(domainList) do
                if power["domains"][dom] and power["domains"][dom]["supportStatus"] then
                    local energy = likwid.calcPower(before[cpu][dom], after[cpu][dom], j-1)
                    print_stdout(string.format("Domain %s:", dom))
                    print_stdout(string.format("Energy consumed: %g Joules",energy))
                    print_stdout(string.format("Power consumed: %g Watt",energy/runtime))
                end
            end
            if i < #sockets then print_stdout("") end
        end
        print_stdout(likwid.hline)
    else
        err = os.execute(execString)
        if err == false then
            print_stderr(string.format("Failed to execute %s!", execString))
            likwid.putPowerInfo()
            likwid.finalize()
            os.exit(1)
        end
    end
end

if print_temp and (string.find(cpuinfo["features"],"TM2") ~= nil) then
    print_stdout(likwid.hline)
    print_stdout("Current core temperatures:");
    for i=1,cputopo["numSockets"] do
        local tag = "S" .. tostring(i-1)
        for _, domain in pairs(affinity["domains"]) do
            if domain["tag"] == tag then
                for j=1,#domain["processorList"] do
                    local cpuid = domain["processorList"][j]
                    likwid.initTemp(cpuid);
                    if (fahrenheit) then
                        local f = 1.8*tonumber(likwid.readTemp(cpuid))+32
                        print_stdout(string.format("Socket %d Core %d: %.0f F",i-1,cpuid, f));
                    else
                        print_stdout(string.format("Socket %d Core %d: %.0f C",i-1,cpuid, tonumber(likwid.readTemp(cpuid))));
                    end
                end
            end
        end
    end
    print_stdout(likwid.hline)
elseif print_temp then
    print_stdout("Architecture does not support temperature reading")
end

likwid.putPowerInfo()
likwid.finalize()
os.exit(exitvalue)
