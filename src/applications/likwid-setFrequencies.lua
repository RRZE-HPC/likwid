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
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
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
print_stderr = function(...) for k,v in pairs({...}) do io.stderr:write(tostring(v) .. "\n") end end

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
    print_stdout("-c dom\t\t CPU selection or LIKWID thread domain")
    print_stdout("\t\t Default behavior is to apply the frequencies to all CPUs.")
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
    if s:sub(slen,slen) == "." then slen = slen - 1 end
    return s:sub(1, slen)
end

function valid_freq(freq, freq_list, turbofreq)
    local valid_freq = false
    for k,v in pairs(freq_list) do
        if (freq == v) then
            valid_freq = true
            break
        end
    end
    if (not valid_freq) and freq == turbofreq then
        valid_freq = true
    end
    return valid_freq
end

function get_base_freq()
    freq = nil
    f = io.open("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency", "r")
    if f ~= nil then
        out = f:read("*a")
        freq = tonumber(out)
        f:close()
    end
    return freq
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
driver = nil
base_freq = nil

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
        domain = tostring(arg)
    elseif (opt == "g") then
        governor = tostring(arg)
    elseif opt == "f" or opt == "freq" then
        if arg then
            frequency = tonumber(arg)*1E6
            min_freq = tonumber(arg)*1E6
            max_freq = tonumber(arg)*1E6
        end
    elseif opt == "x" or opt == "min" then
        if arg then min_freq = tonumber(arg)*1E6 end
    elseif opt == "y" or opt == "max" then
        if arg then max_freq = tonumber(arg)*1E6 end
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
            likwid.setVerbosity(s)
        else
            print_stderr(string.format("ERROR: Value %s for verbosity not valid", arg))
        end
    elseif opt == "reset" then
        do_reset = true
    elseif opt == "ureset" then
        do_ureset = true
    elseif opt == "umin" then
        if arg then min_u_freq = tostring(arg) end
    elseif opt == "umax" then
        if arg then max_u_freq = tostring(arg) end
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

cpuinfo = likwid.getCpuInfo()
topo = likwid.getCpuTopology()
affinity = likwid.getAffinityInfo()
if not domain or domain == "N" then
    domain = "N:0-" .. tostring(topo["activeHWThreads"]-1)
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
        if #dom["processorList"] > 0 then
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
end

likwid.initFreq()
driver = likwid.getFreqDriver(cpulist[1])

base_freq = likwid.getCpuClockBase(cpulist[1])
if base_freq == 0 then
    if get_base_freq() ~= nil then
        base_freq = get_base_freq()
    end
end

if verbosity == 3 then
    print_stdout(string.format("DEBUG: Given CPU expression expands to %d HW Threads:", numthreads))
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
    govs = likwid.getAvailGovs(0)
    if #govs > 0 then
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
        outfreqs = {}
        for _, f in pairs(freqs) do
            table.insert(outfreqs, round(tonumber(f)/1.E6))
        end
        print_stdout(string.format("%s", table.concat(outfreqs, " ")))
    else
        print_stdout("Cannot get frequencies from cpufreq module")
        if driver == "intel_pstate" then
            freqs = {}
            min = tonumber(likwid.getConfCpuClockMin(cpulist[1]))/1E6
            max = tonumber(likwid.getConfCpuClockMax(cpulist[1]))/1E6
            print_stdout("The intel_pstate module allows free selection of frequencies in the available range")
            print_stdout(string.format("Minimal CPU frequency %s", round(min)))
            print_stdout(string.format("Maximal CPU frequency %s", round(max)))
            os.exit(0)
        end
    end
end

if printCurFreq then
    str = {"Current CPU frequencies:"}
    local processed = 0
    for i=1,#cpulist do
        gov = likwid.getGovernor(cpulist[i])
        freq = tonumber(likwid.getCpuClockCurrent(cpulist[i]))/1E6
        min = tonumber(likwid.getCpuClockMin(cpulist[i]))/1E6
        max = tonumber(likwid.getCpuClockMax(cpulist[i]))/1E6
        t = tonumber(likwid.getTurbo(cpulist[i]));
        if gov and freq and min and max and t >= 0 then
            processed = processed + 1
            table.insert(str, string.format("HWThread %d: governor %12s min/cur/max %s/%s/%s GHz Turbo %d",cpulist[i], gov, round(min), round(freq), round(max), t))
        end
    end
    table.insert(str, "")
    if processed > 0 then
        print_stdout(table.concat(str, "\n"))
    else
        print_stdout("Cannot read frequency data from cpufreq module\n")
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
    likwid.finalizeFreq()
    os.exit(0)
end

if do_reset then
    local availfreqs = likwid.getAvailFreq(cpulist[1])
    local availgovs = likwid.getAvailGovs(cpulist[1])
    if driver == "intel_pstate" then
        availfreqs = {likwid.getConfCpuClockMin(cpulist[1]), likwid.getConfCpuClockMax(cpulist[1])}
    end
    if not min_freq then
        min_freq = availfreqs[1]
    end
    if min_freq >= availfreqs[#availfreqs] then
        min_freq = availfreqs[#availfreqs]
    end
    if not (set_turbo or max_freq) then
        set_turbo = true
        turbo = 0
        max_freq = availfreqs[#availfreqs]
        if max_freq <= availfreqs[1] then
            max_freq = availfreqs[1]
        end
    end
    if not governor then
        governor = nil
        for i, g in pairs(availgovs) do
            if g:match("^performance") then
                governor = g
                break
            end
        end
        if not governor then
            for i, g in pairs(availgovs) do
                if g:match("^conservative") then
                    governor = g
                    break
                end
            end
            if not governor then
                governor = availgovs[#availgovs]
            end
        end
    end
    if min_freq and governor then
        print_stdout(string.format("Reset to governor %s with min freq. %s GHz and deactivate turbo mode", governor, round(min_freq/1E6)))
    end
end

if do_ureset then
    if cpuinfo["isIntel"] == 1 then
        local availfreqs = likwid.getAvailFreq(cpulist[1])
        if #availfreqs == 0 then
            availfreqs = {likwid.getConfCpuClockMin(cpulist[1]), likwid.getConfCpuClockMax(cpulist[1])}
        end
        local power = likwid.getPowerInfo()
        local minf = tonumber(availfreqs[1]/1E6)
        if (minf > tonumber(availfreqs[#availfreqs]/1E6)) then
            minf = tonumber(availfreqs[#availfreqs]/1E6)
        end
        local maxf = tonumber(power["turbo"]["steps"][1]) / 1000
        if (minf > maxf) then
            local s = minf
            minf = maxf
            maxf = s
        end
        min_u_freq = minf
        max_u_freq = maxf
    else
        print_stderr("ERROR: AMD CPUs provide no interface to manipulate the Uncore frequency.")
        likwid.finalizeFreq()
        os.exit(1)
    end
end

if numthreads > 0 and not (frequency or min_freq or max_freq or governor or min_u_freq or max_u_freq or set_turbo) then
    print_stderr("ERROR: You need to set either a frequency or governor for the selected CPUs on commandline")
    likwid.finalizeFreq()
    os.exit(1)
end

if min_freq and max_freq and min_freq > max_freq then
    print_stderr("ERROR: Minimal CPU frequency higher than maximal frequency.")
    likwid.finalizeFreq()
    os.exit(1)
end
if min_freq and max_freq and max_freq < min_freq then
    print_stderr("ERROR: Maximal CPU frequency lower than minimal frequency.")
    likwid.finalizeFreq()
    os.exit(1)
end
if min_u_freq and max_u_freq and max_u_freq < min_u_freq then
    print_stderr("ERROR: Maximal Uncore frequency lower than minimal frequency.")
    likwid.finalizeFreq()
    os.exit(1)
end

if max_freq and (likwid.getTurbo(cpulist[1]) == 1 or (set_turbo and turbo == 1)) and tonumber(max_freq) < base_freq then
    print_stderr("ERROR: Setting maximal CPU frequency below base CPU frequency with activated Turbo mode is not supported.")
    likwid.finalizeFreq()
    os.exit(1)
end
if (likwid.getTurbo(cpulist[1]) == 0 and not set_turbo) then
    local do_exit = false
    if max_freq and tonumber(max_freq) > base_freq then
        print_stderr("ERROR: Setting maximal CPU frequency above base CPU frequency with deactivated Turbo mode is not supported.")
        do_exit = true
    elseif min_freq and tonumber(min_freq) > base_freq then
        print_stderr("ERROR: Setting minimal CPU frequency above base CPU frequency with deactivated Turbo mode is not supported.")
        do_exit = true
    end
    if do_exit then
        likwid.finalizeFreq()
        os.exit(1)
    end
end


local availfreqs = likwid.getAvailFreq(cpulist[1])
if driver == "intel_pstate" then
    availfreqs = {likwid.getConfCpuClockMin(cpulist[1])/1E6, likwid.getConfCpuClockMax(cpulist[1])/1E6}
end
if (frequency or min_freq or max_freq) and #availfreqs == 0 and likwid.getFreqDriver(cpulist[1]) ~= "intel_pstate" then
    print_stdout("Cannot set CPU frequency, cpufreq module not properly loaded")
    likwid.finalizeFreq()
    os.exit(1)
end
local savailfreqs = {}
for i,f in pairs(availfreqs) do
    savailfreqs[i] = round(f)
end
if verbosity == 3 then
    print_stdout("DEBUG Available freq.: "..table.concat(savailfreqs, ", "))
end
if driver ~= "intel_pstate" then
    if max_freq then
        local test_freq = round(tonumber(max_freq))
        if not valid_freq(test_freq, savailfreqs, availturbo) then
            print_stderr(string.format("ERROR: Selected max. frequency %s not available! Please select one of\n%s", test_freq, table.concat(savailfreqs, ", ")))
            likwid.finalizeFreq()
            os.exit(1)
        end
    end
    if min_freq then
        local test_freq = round(tonumber(min_freq))
        if not valid_freq(test_freq, savailfreqs, availturbo) then
            print_stderr(string.format("ERROR: Selected min. frequency %s not available! Please select one of\n%s", test_freq, table.concat(savailfreqs, ", ")))
            likwid.finalizeFreq()
            os.exit(1)
        end
    end
end

min_first = false
max_first = false
if min_freq and tonumber(min_freq)/1E6 > tonumber(likwid.getCpuClockMax(cpulist[i]))/1E6 then
    max_first = true
end
if max_freq and tonumber(max_freq)/1E6 < tonumber(likwid.getCpuClockMin(cpulist[i]))/1E6 then
    min_first = true
end

if set_turbo then
    for i=1,#cpulist do
        if verbosity == 3 then
            print_stdout(string.format("DEBUG: Set turbo mode for CPU %d to %d", cpulist[i], turbo))
        end
        local f = likwid.setTurbo(cpulist[i], turbo)
    end
end


if max_first and max_freq then
    for i=1,#cpulist do
        local f = likwid.setCpuClockMax(cpulist[i], max_freq)
    end
    if min_freq then
        for i=1,#cpulist do
            local f = likwid.setCpuClockMin(cpulist[i], min_freq)
        end
    end
elseif min_first and min_freq then
    for i=1,#cpulist do
        local f = likwid.setCpuClockMin(cpulist[i], min_freq)
    end
    if max_freq then
        for i=1,#cpulist do
            local f = likwid.setCpuClockMax(cpulist[i], max_freq)
        end
    end
else
    for i=1,#cpulist do
        local f = likwid.setCpuClockMin(cpulist[i], min_freq)
        local f = likwid.setCpuClockMax(cpulist[i], max_freq)
    end
end

if min_u_freq then
    test = likwid.getUncoreFreqMin(socklist[1])
    if test == 0 then
        print_stderr("ERROR: This CPU does not provide an interface to manipulate the Uncore frequency.")
        min_u_freq = nil
        os.exit(1)
    end
end
if max_u_freq then
    test = likwid.getUncoreFreqMax(socklist[1])
    if test == 0 then
        print_stderr("ERROR: This CPU does not provide an interface to manipulate the Uncore frequency.")
        max_u_freq = nil
        os.exit(1)
    end
end


if min_u_freq then
    if cpuinfo["isIntel"] == 1 then
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
    else
        print_stderr("ERROR: AMD CPUs provide no interface to manipulate the Uncore frequency.")
    end
end

if max_u_freq then
    if cpuinfo["isIntel"] == 1 then
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
    else
        print_stderr("ERROR: AMD CPUs provide no interface to manipulate the Uncore frequency.")
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
    if valid_gov then
        for i=1,#cpulist do
            if verbosity == 3 then
                print_stdout(string.format("DEBUG: Set governor for CPU %d to %s", cpulist[i], governor))
            end
            local f = likwid.setGovernor(cpulist[i], governor)
            if do_reset then
                likwid.setCpuClockMin(cpulist[i], cur_min[i])
                likwid.setCpuClockMax(cpulist[i], cur_max[i])
            end
        end
    else
        print_stderr(string.format("ERROR: Governor %s not available! Please select one of\n%s", governor, table.concat(govs, ", ")))
    end
end
likwid.finalizeFreq()
likwid.putAffinityInfo()
likwid.putTopology()
os.exit(0)
