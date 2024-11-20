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
    print_stdout("-t\t\t Print current temperatures of all hardware threads")
    print_stdout("-f\t\t Print current temperatures in Fahrenheit")
    print_stdout("")
    examples()
end

local config = likwid.getConfiguration();

print_info = false
stethoscope = false
fahrenheit = false
print_temp = false
verbose = 0
if config["daemonMode"] < 0 then
    access_mode = 1
else
    access_mode = config["daemonMode"]
end
time_interval = 2
time_orig = "2s"
read_interval = 10
cpustr = nil
cpuinfo = likwid.getCpuInfo()
--if cpuinfo["isIntel"] == 1 then
--    domainList = {"PKG", "PP0", "PP1", "DRAM", "PLATFORM"}
--else
--    domainList = {"CORE", "PKG"}
--end

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
        error("-c is currently unsupported. Please run likwid-powermeter with likwid-pin")
        cpustr = arg
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
    elseif (opt == "f") then
        fahrenheit = true
        print_temp = true
    elseif (opt == "t") then
        print_temp = true
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
        likwid.setVerbosity(verbose)
    elseif (opt == "s") then
        time_interval = likwid.parse_time(arg) / 1000000
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

-- initialize sysfeatures module
local err = likwid.initSysFeatures()
if err < 0 then
    print_stderr("Cannot initialize LIKWID sysfeatures")
    os.exit(1)
end

-- enumerate available devices
local device_types = { likwid.hwthread, likwid.core, likwid.die, likwid.socket, likwid.node }
if likwid.nvSupported() then
    table.insert(device_types, likwid.nvidia_gpu)
end
if likwid_rocmSupported() then
    table.insert(device_types, likwid.amd_gpu)
end

local devices = {}
for _, devtype in pairs(device_types) do
    -- getAvailableDevices returns a list of integers IDs for a particular device type
    devices[devtype] = {}
    for _, devid in pairs(likwid.getAvailableDevices(devtype)) do
        devices[devtype][devid] = likwid.createDevice(devtype, devid)
    end
end

-- set client access mode
if likwid.setAccessClientMode(access_mode) ~= 0 then
    os.exit(1)
end

-- check if stethoscope mode is enabled simultaneous with a program parameter
if #arg ~= 0 and stethoscope then
    print_stderr("Stethoscope mode cannot be used to launch programs")
    os.exit(1)
end

-- print device information
if print_info then
    print_stdout(likwid.hline);
    print_stdout(string.format("CPU name:\t%s", cpuinfo["osname"]))
    print_stdout(string.format("CPU type:\t%s", cpuinfo["name"]))
    if cpuinfo["clock"] > 0 then
        print_stdout(string.format("CPU clock:\t%3.2f GHz", cpuinfo["clock"] * 1.E-09))
    else
        print_stdout(string.format("CPU clock:\t%3.2f GHz", likwid.getCpuClock() * 1.E-09))
    end
    print_stdout(likwid.hline)
end

-- TODO print clock speeds and power information when print_info is enabled

-- Check if time interval is not too short TODO, we currently do not have a way to retrieve
-- the timeUnit and it may also differ between metric types.
if time_interval < 1 then
    print_stderr("Warning: Time interval too short, measurement may be inaccurate")
end

-- TODO begin: do we still need this?
local pinList = {}
local execString = ""
if use_perfctr then
    execString = string.format("<INSTALLED_PREFIX>/bin/likwid-perfctr -C %s -f -g CLOCK ",table.concat(pinList, "@"))
elseif not stethoscope then
    execString = string.format("<INSTALLED_PREFIX>/bin/likwid-pin -c %s -q ",table.concat(pinList, "@"))
end
-- TODO end

-- TODO begin: what does this do?
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
        if string.find(arg[i], " ") then
            table.insert(execList, "\""..arg[i].."\"")
        else
            table.insert(execList, arg[i])
        end
    end
    execString = execString .. table.concat(execList," ")
end
-- TODO end

-- Read all metrics from power_metrics, but filter them by the features which are actually available
local power_features_wanted = {
    "rapl.pkg_energy", "rapl.dram_energy", "rapl.pp0_energy", "rapl.pp1_energy", "rapl.psys_energy",
    "nvml.energy",
}
local features_available = likwid.sysFeatures_list()
local power_features_available = {}
for _, wanted in pairs(power_features_wanted) do
    for _, available in pairs(features_available) do
        if wanted == available.FullName then
            table.insert(power_features_available, available)
            break
        end
    end
end

if #power_features_available == 0 then
    print_stderr("No power features available. Cannot measure power.")
    os.exit(1)
end

-- start measure time
local time_before = likwid.getTimeOfDay()
local prev_time = time_before
local energy = {} -- structure: energy[dev_type][dev_idx][metric_type][X] (X=1: raw measurement, X=2: corrected measurement)

next_power2 = function(x)
    local l2x = math.log(x) / math.log(2)
    return 2 ^ (math.ceil(l2x))
end

-- launch program (if any)
if not stethoscope then
    pid = likwid.startProgram(execString, 0, {})
    if not pid then
        print_stderr(string.format("Failed to execute %s!", execString))
        os.exit(1)
    end
    -- in order to avoid wait times when the program has exited, we cap the maximum wait time
    read_interval = math.min(read_interval, 1)
end

-- loop and measure until either the launched program exits or if the time elapses
local exitvalue = 0
local quit = false
while true do
    -- Iterate over all device types, devices, and features.
    for devtype, devlist in pairs(devices) do
        if energy[devtype] == nil then
            energy[devtype] = {}
        end
        energy_for_devtype = energy[devtype]
        for _, dev in pairs(devlist) do
            if energy_for_devtype[dev:id()] == nil then
                energy_for_devtype[dev:id()] = {}
            end
            energy_for_device = energy_for_devtype[dev:id()]
            for _, power_feature in pairs(power_features_available) do
                if power_feature.TypeID ~= devtype then
                    goto continue
                end
                if energy_for_device[power_feature.FullName] == nil then
                    energy_for_device[power_feature.FullName] = {}
                end
                energy_for_feature = energy_for_device[power_feature.FullName]
                -- Read old and new value.
                -- We have to keep track of raw and aggregated values separately in order to be able to notice
                -- if the counters overflow.
                local old_value_raw = energy_for_feature[1]
                local old_value_aggr = energy_for_feature[2]
                local new_value_raw, err = likwid.sysFeatures_get(power_feature.FullName, dev)
                new_value_raw = tonumber(new_value_raw)

                if new_value_raw ~= nil and old_value_aggr == nil then
                    -- if old value is not set, simply initialize
                    energy_for_feature[2] = 0
                elseif new_value_raw ~= nil and old_value_aggr ~= nil then
                    -- if old and new value are set, update energy, overflow adjusted
                    local delta = new_value_raw - old_value_raw
                    if new_value_raw < old_value_raw then
                        -- This case should only occur if the counter overflowed.
                        -- Get next power-of-two of old value in order to determine overflow point.
                        local overflow_point = next_power2(old_value)
                        delta = (new_value_raw + overflow_point) - old_value_raw
                    end
                    energy_for_feature[2] = energy_for_feature[2] + delta
                end

                -- unconditionally store the raw value, in order to do proper overflow detection
                energy_for_feature[1] = new_value_raw
                ::continue::
            end
        end
    end

    -- Instead of breaking out immediatley, we always delay breaking until here.
    -- This makes sure there is always a measurement taken after e.g. the program has exited.
    if quit then
        time_after = likwid.getTimeOfDay()
        break
    end

    -- Wait depending on the measurement mode
    if stethoscope then
        -- In stethoscope mode, we simply sleep periodically.
        local cur_time = likwid.getTimeOfDay()
        if cur_time > time_before + time_interval then
            quit = true
            sleep_duration = 0
        elseif cur_time - time_before + read_interval > time_interval then
            sleep_duration = time_interval - (cur_time - time_before)
        else
            sleep_duration = read_interval - (cur_time - prev_time)
        end
        sleep_duration = math.max(sleep_duration, 0)

        -- sleep accepts its parameters in microsecond range
        likwid.nanosleep(sleep_duration)
        prev_time = likwid.getTimeOfDay()
    else
        -- In execution wrapper mode, we simply check and wait
        if likwid.getSignalState() ~= 0 then
            likwid.killProgram()
            quit = true
        else
            exitvalue, exited = likwid.checkProgram(pid)
            if exited then
                io.stdout:flush()
                quit = true
            else
                likwid.sleep(read_interval)
            end
        end
    end
end

runtime = time_after - time_before

print_stdout(likwid.hline)
print_stdout(string.format("Runtime: %.2fs", runtime))

for dev_type, energy_for_devtype in pairs(energy) do
    -- First check if there actually are any features recorded for this device type.
    -- This avoids printing headers without any results.
    local has_measurement = false
    for _, energy_for_device in pairs(energy_for_devtype) do
        for _, energy_for_feature in pairs(energy_for_device) do
            if #energy_for_feature > 0 then
                has_measurement = true
                goto outer_break
            end
        end
    end
    ::outer_break::

    if not has_measurement then
        goto continue
    end

    -- At this point we know that devices[dev_type] contains at least one device.
    -- However, because dev_type is a string and doesn't always start at 0, we cannot simply index the first
    -- entry in devices[dev_type] to retrieve the device type name.
    -- To solve this, loop over devices[dev_type] and just remember the first entry.
    local first_dev_id = nil
    for dev_id, dev in pairs(devices[dev_type]) do
        first_dev_id = dev_id
        break
    end

    print_stdout(string.format("Device Type: %s", devices[dev_type][first_dev_id]:typeName()))
    for dev_id, energy_for_device in pairs(energy_for_devtype) do
        print_stdout(string.format("  ID: %s", dev_id))
        for feature_name, energy_for_feature in pairs(energy_for_device) do
            local joules = energy_for_feature[2]
            if energy_for_feature[2] == nil then
                joules = "ERR"
                watts = "ERR"
            else
                joules = string.format("%11.3f", joules)
                watts = string.format("%11.3f", joules / runtime)
            end

            print_stdout(string.format("    %s:", feature_name))
            print_stdout(string.format("      Energy consumed:    %s J", joules))
            print_stdout(string.format("      Average power draw: %s W", watts))
        end
    end
    ::continue::
end

print_stdout(likwid.hline)

if print_temp then
    print_stderr("Printing temperature is not yet implemented")
    os.exit(1)
end

likwid.finalizeSysFeatures()
likwid.finalize()
os.exit(exitvalue)
