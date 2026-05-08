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

--local config = {}
--config.stdout = print
--config.stderr = function(...) for k,v in pairs({...}) do io.stderr:write(tostring(v) .. "\n") end end

local function version(config)
    local version = likwid.getMajorVersion()
    local release = likwid.getMinorVersion()
    local bugfix = likwid.getBugfixVersion()
    config.stdout("likwid-setFrequencies -- Version %d.%d.%d (commit: %s)", version, release, bugfix, likwid.commit)
end

local function examples(config)
    config.stdout("Examples:")
    config.stdout("  # Check whether the set Uncore frequency is really set and fixed")
    config.stdout("  $ likwid-perfctr -g UNCORE_CLOCK:UBOXFIX -C 0 <exec_doing_work>")
    config.stdout()
end

local function usage(config)
    version(config)
    config.stdout()
    config.stdout("A tool to adjust frequencies and governors on x86 CPUs.")
    config.stdout()
    config.stdout("Options:")
    config.stdout("  -h                Help message")
    config.stdout("  -v                Version information")
    config.stdout("  -V <0-3>          Verbosity (0=only_error, 3=developer)")
    config.stdout("  -c dom            CPU selection or LIKWID thread domain")
    config.stdout("                    Default behavior is to apply the frequencies to all CPUs.")
    config.stdout("                    See likwid-pin -h for details")
    config.stdout("  -p                Print current frequencies (CPUs + Uncore)")
    config.stdout("  -l                List available CPU frequencies")
    config.stdout("  -m                List available CPU governors")
    config.stdout()
    config.stdout("  -f, --freq <freq> Set minimal and maximal CPU frequency")
    config.stdout("  -x, --min <freq>  Set minimal CPU frequency")
    config.stdout("  -y, --max <freq>  Set maximal CPU frequency")
    config.stdout("  -g gov            Set CPU frequency governor")
    config.stdout("  -t, --turbo <0|1> De/Activate turbo mode")
    if config.cpuinfo.architecture == "x86_64" then
        if config.cpuinfo.isIntel then
            config.stdout("  --umin <freq>     Set minimal Uncore frequency")
            config.stdout("  --umax <freq>     Set maximal Uncore frequency")
        else
            config.stdout("  --cmax <freq>     Set maximal core boost frequency")
            config.stdout("  --smax <freq>     Set maximal socket boost frequency")
        end
    end
    config.stdout()
    config.stdout("  -reset            Set governor 'performance', set minimal and maximal frequency to")
    config.stdout("                    the CPU limits and deactivate turbo.")
    if config.cpuinfo.architecture == "x86_64" and config.cpuinfo.isIntel then
        config.stdout("  -ureset           Set Uncore frequencies to its min and max limits")
    end
    config.stdout()
    config.stdout("Notes:")
    config.stdout("  For the options -f, -x and -y:")
    config.stdout("  - acpi-cpufreq driver: set the userspace governor implicitly")
    config.stdout("  - intel_pstate driver: keep current governor")
    config.stdout()
    config.stdout("  In general the min/max Uncore frequency can be set freely, even to 0 or 1E20")
    config.stdout("  but the hardware stays inside its limits. LIKWID reduces the range of possible")
    config.stdout("  frequencies to the minimal core frequency (likwid-setFrequencies -l) and the")
    config.stdout("  maximally achievable turbo frequency with a single core (C0 value at")
    config.stdout("  likwid-powermeter -i output).")
    config.stdout()
    config.stdout("  Sleeping is commonly not sufficient.")
    config.stdout()
    config.stdout("  If you switch governors with the intel_pstate driver, it might be that the driver")
    config.stdout("  changes the frequency settings, please check afterwards and re-set frequencies if")
    config.stdout("  needed.")
    config.stdout()
    examples(config)
end


-- Create an error
-- All function need to be on an error need to be called with ':'
-- Usage:
-- function myfunc()
--     local err = errors.new()
--     local errno = 42
--     err:add("This is error %d", errno)
--     return err:retval()
-- end

local errors = {}
errors.new = function()
    local e = {
        returncode = -1,
        entries = {},
    }

    -- Add a new formatted error
    -- Call err:add(format_string, format_arg1, format_arg2, ...)
    e.add = function(e, fmt, ...)
        local fmtargs = {...}
        if #fmtargs > 0 then
            local s = string.format("ERROR: " .. fmt, table.unpack(fmtargs))
            table.insert(e.entries, s)
        else
            table.insert(e.entries, fmt)
        end
        -- As soon as there is an error, this get 1
        e.returncode = 1
    end

    -- Format error to a string
    -- Call: err:tostring()
    e.tostring = function(e)
        if #e.entries == 1 then
            return e.entries[1] .. "\n"
        elseif #e.entries > 1 then
            return table.concat(e.entries, "\n")
        end
        return ""
    end

    -- Print error
    -- Call: err:print()
    e.print = function(e)
        local s = e:tostring()
        if #s > 0 then
            io.stderr:write(s .. "\n")
        end
    end

    -- Append the entries of another error
    -- Call: err:append(other_err)
    e.append = function(e, err)
        for _, line in pairs(err.entries) do
            e:add(line)
        end
    end

    -- Return the error or nil
    -- Calling function can check it like
    -- local err = func()
    -- if err then
    --     print(err:tostring())
    -- end
    e.retval = function(e)
        if #e.entries > 0 or e.returncode >= 0 then
            return e
        end
        return nil
    end

    -- Get the return code of the error
    -- Call: err:retcode()
    e.retcode = function(e)
        return e.returncode
    end

    -- Set the return code of the error
    -- Call: err:setretcode(new_return_code)
    e.setretcode = function(e, retcode)
        e.returncode = retcode
    end

    return e
end

local function unit_conv(inunit, outunit)
    local scale = 1.0

    if inunit== "kHz" and newunit == "MHz" then
        scale = 1.0/1E3
    elseif inunit == "kHz" and newunit == "GHz" then
        scale = 1.0/1E6
    elseif inunit == "MHz" and newunit == "kHz" then
        scale = 1000.0
    elseif inunit == "MHz" and newunit == "GHz" then
        scale = 1.0/1E3
    elseif inunit == "GHz" and newunit == "MHz" then
        scale = 1000
    elseif inunit == "GHz" and newunit == "kHz" then
        scale = 1000000
    elseif inunit == "Hz" and newunit == "kHz" then
        scale = 1.0/1E3
    elseif inunit == "Hz" and newunit == "MHz" then
        scale = 1.0/1E6
    elseif inunit == "Hz" and newunit == "GHz" then
        scale = 1.0/1E9
    end
    return scale
end

-- Container with scaling feature for frequency values
local freq = {}
-- Create a new freq value with freq.new(value_string, unit_string)
-- If no unit is given, 'Hz' is assumed.
freq.new = function(value, unit)
    if not unit then
        unit = "Hz"
    end
    local v = {
        value = value,
        unit = unit
    }

    -- Scale freq value to new unit
    -- Call value:scale(new_unit)
    v.scale = function(v, newunit)
        local scale = 1.0

        if v.unit == "kHz" and newunit == "MHz" then
            scale = 1.0/1E3
        elseif v.unit == "kHz" and newunit == "GHz" then
            scale = 1.0/1E6
        elseif v.unit == "MHz" and newunit == "kHz" then
            scale = 1000.0
        elseif v.unit == "MHz" and newunit == "GHz" then
            scale = 1.0/1E3
        elseif v.unit == "GHz" and newunit == "MHz" then
            scale = 1000
        elseif v.unit == "GHz" and newunit == "kHz" then
            scale = 1000000
        elseif v.unit == "Hz" and newunit == "kHz" then
            scale = 1.0/1E3
        elseif v.unit == "Hz" and newunit == "MHz" then
            scale = 1.0/1E6
        elseif v.unit == "Hz" and newunit == "GHz" then
            scale = 1.0/1E9
        end

        v.value = tostring(tonumber(v.value) * scale)
        v.unit = newunit
    end

    -- Get the freq value
    -- Call: value:getvalue()
    v.getvalue = function(v)
        return v.value
    end
    -- Get the freq unit
    -- Call: value:getunit()
    v.getunit = function(v)
        return v.unit
    end

    return v
end

local function sysft_name(category, feature)
    return string.format("%s.%s", category, feature)
end

local function get_feature_to_freq(config, category, feature, device)
    local feat = get_feature(config, category, feature)
    local val, sferr = sysFeatures_get(sysft_name(category, feature), device)
    if not val then
        return nil, sferr
    end
    return freq.new(val, feat.Unit)
end

-- Default command line options
local setFrequencies_cliopts = {
    "h", "help",
    "v", "version",
    "V:", "verbose:",
    "c:", "C:", "hwtselect:",
    "d:", "devices:",
    "g:", "governor:",
    "f:", "freq:",
    "x:", "min:",
    "y:", "max:",
    "t:", "turbo:",
    "p", "print",
    "l", "list",
    "m", "listgov",
    "reset",
    "unit:",
    "O",
}

-- Command line options for Intel systems (x86_64)
local setFrequencies_cliopts_x86_intel = {
    "ureset",
    "umin:",
    "umax:",
}

-- Command line options for AMD systems (x86_64)
local setFrequencies_cliopts_x86_amd = {
    "cmax:",
    "smax:",
}

-- Command line options for ARM systems
local setFrequencies_cliopts_arm = {
}

-- Command line options for Nvidia GPUs
local setFrequencies_cliopts_nvidia_gpu = {
}

-- Command line options for AMD GPUs
local setFrequencies_cliopts_amd_gpu = {
}

-- Parse command line options
local function read_cliargs(args)
    local err = errors.new()
    -- initialize config table with defaults
    local config = {}
    config.hwtselect = {}
    config.devices = {}
    config.verbose = -1
    config.printfreq = false
    config.listfreq = false
    config.listgov = false
    config.reset = false
    config.uncore_reset = false
    config.hwthread_set = false
    config.uncore_set = false
    config.cmax_set = false
    config.smax_set = false
    config.print_csv = false

    -- add output function for stdout
    -- Call config.stdout(format_string, format_arg1, format_arg2, ...)
    config.stdout = function(fmt, ...)
        if not fmt then
            print()
        else
            print(string.format(fmt, table.unpack({...})))
        end
    end
    -- add output function for stderr
    -- Call config.stderr(format_string, format_arg1, format_arg2, ...)
    config.stderr = function(fmt, ...)
        if not fmt then
            io.stderr:write("\n")
        else
            local v = string.format(fmt, table.unpack({...}))
            io.stderr:write(v .. "\n")
        end
    end
    -- add debug output function to stdout based on verbosity level (1-3)
    -- Call config:debug(debug_level, format_string, format_arg1, format_arg2, ...)
    -- Caution: In contrast to stdout() and stderr(), this function call required ':'!
    config.debug = function(config, level, fmt, ...)
        if level <= config.verbose then
            local newfmt = string.format("DEBUG<%d> %s", level, fmt)
            config.stdout(newfmt, table.unpack({...}))
        end
    end

    -- Get CPU information from LIKWID. It is used in usage()
    config.cpuinfo = likwid.getCpuInfo()

    if #arg == 0 then
        usage(config)
        err:setretcode(0)
        return nil, err:retval()
    end
    
    if config.cpuinfo.architecture == "x86_64" then
        if config.cpuinfo.isIntel then
            for _, o in pairs(setFrequencies_cliopts_x86_intel) do
                table.insert(setFrequencies_cliopts, o)
            end
        else
            for _, o in pairs(setFrequencies_cliopts_x86_amd) do
                table.insert(setFrequencies_cliopts, o)
            end
        end
    end

    local fix_hwthread_freq = nil
    for opt,arg in likwid.getopt(arg, setFrequencies_cliopts) do
        if opt == "h" or opt == "help" then
            usage(config)
            err:setretcode(0)
            return nil, err:retval()
        elseif opt == "v" or opt == "version" then
            version(config)
            err:setretcode(0)
            return nil, err:retval()
        elseif opt == "V" or opt == "verbose" then
            local verbose = tonumber(arg)
            if (not verbose) or (verbose and (verbose < 0 or verbose > 3)) then
                err:add("-V/--verbose requires number between 0 and 3 as argument")
            end
            config.verbose = tonumber(arg)
        elseif opt == "c" or opt == "C" or opt == "hwtselect" then
            config:debug(3, "Adding HWThread select: %s", tostring(arg))
            table.insert(config.hwtselect, tostring(arg))
        elseif opt == "d" or opt == "devices" then
            config:debug(3, "Adding device select: %s", tostring(arg))
            table.insert(config.devices, tostring(arg))
        elseif opt == "f" or opt == "freq" then
            config:debug(3, "Setting HWThread frequency to %s", tostring(arg))
            fix_hwthread_freq = tostring(arg)
            config.hwthread_set = true
        elseif opt == "x" or opt == "min" then
            config:debug(3, "Setting minimal HWThread frequency to %s", tostring(arg))
            config.min_freq = tostring(arg)
            config.hwthread_set = true
        elseif opt == "y" or opt == "max" then
            config:debug(3, "Setting maximal HWThread frequency to %s", tostring(arg))
            config.max_freq = tostring(arg)
            config.hwthread_set = true
        elseif opt == "t" or opt == "turbo" then
            config:debug(3, "Setting turbo mode for HWThread to %s", tostring(arg))
            config.set_turbo = tostring(arg)
        elseif opt == "g" or opt == "governor" then
            config:debug(3, "Setting HWThread governor to %s", tostring(arg))
            config.set_governor = tostring(arg)
        elseif opt == "l" or opt == "list" then
            config:debug(3, "Listing frequency limits of all given devices")
            config.listfreq = true
        elseif opt == "p" or opt == "print" then
            config:debug(3, "Print current frequencies of all given devices")
            config.printfreq = true
        elseif opt == "m" or opt == "listgov" then
            config:debug(3, "List available HWThread cpufreq governors")
            config.listgov = true
        elseif opt == "reset" then
            config.reset = true
        elseif opt == "ureset" then
            config.ureset = true
        elseif opt == "O" then
            config:debug(3, "Activate CSV output")
            config.print_csv = true
        elseif opt == "umin" then
            config:debug(3, "Setting minimal Intel Uncore frequency to %s", tostring(arg))
            config.umin = tostring(arg)
            config.uncore_set = true
        elseif opt == "umax" then
            config:debug(3, "Setting maximal Intel Uncore frequency to %s", tostring(arg))
            config.umax = tostring(arg)
            config.uncore_set = true
        elseif opt == "unit" then
            config:debug(3, "Set output unit to %s", tostring(arg))
            config.unit = tostring(arg)
        elseif opt == "?" then
            err:add("Invalid commandline option -%s", arg)
        elseif opt == "!" then
            if #arg == 1 then
                err:add("Option -%s requires an argument", arg)
            else
                err:add("Option --%s requires an argument", arg)
            end
        end
    end

    config:debug(3, "System uses %s architecture", config.cpuinfo.architecture)

    if fix_hwthread_freq and config.min_freq == nil and config.max_freq == nil then
        config.min_freq = fix_hwthread_freq
        config.max_freq = fix_hwthread_freq
    end
    --[[if #config.hwtselect == 0 then
        table.insert(config.hwtselect, 'N')
    end]]
    if config.verbose < 0 then
        config.verbose = 0
    end
    if config.unit then
        if not config.unit:match("[kMG]?Hz") then
            config.unit = nil
            err:add("Invalid unit given '%s'", config.unit)
        end
    end
    config:debug(3, "Configuration after parsing command line")
    for k,v in pairs(config) do
        if type(v) == "table" then
            local t = {}
            for tk,tv in pairs(v) do
                t[tk] = tostring(tv)
            end
            config:debug(3, " %s : [%s]", k, table.concat(t, ", "))
        elseif type(v) ~= "function" then
            config:debug(3, " %s : %s", k, tostring(v))
        end
    end

    return config, err:retval()
end


local function resolve_devices(config)
    local err = errors.new()
    -- Create output table but treat it like a set, insertion only if value
    -- is not in table
    __newindex_once = function(t, key, value)
        local found = false
        for _, v in pairs(t) do
            if v == value then
                found = true
            end
        end
        if not found then
            rawset(t, key, value)
        end
    end
    local parsedevs = setmetatable({}, {
        __newindex = __newindex_once
    })
    local extradevs = setmetatable({}, {
        __newindex = __newindex_once
    })

    -- Add HW threads given on command line
    if #config.hwtselect > 0 then
        config:debug(3, "Resolving HWThread sectors [%s]", table.concat(config.hwtselect, ", "))
        for _, c in pairs(config.hwtselect) do
            numthreads, cpulist = likwid.cpustr_to_cpulist(c)
            for k,v in pairs(cpulist) do
                table.insert(parsedevs, string.format("T:%d", v))
                if config.cpuinfo.isIntel and (config.uncore_set or config.listfreq or config.printfreq) then
                    for _, hwt in pairs(config.topology.threadPool) do
                        if hwt.apicId == v then
                            config:debug(3, "Adding extra device S:%d for Intel Uncore frequency", hwt.packageId)
                            table.insert(extradevs, string.format("S:%d", hwt.packageId))
                        end
                    end
                end
            end
        end
    end
    -- Add device strings given on command line
    for _, c in pairs(config.devices) do
        table.insert(parsedevs, c)
    end

    -- Add extra devices
    if #extradevs > 0 then
        config:debug(3, "Adding extra devices [%s]", table.concat(extradevs, ", "))
        for _, d in pairs(extradevs) do
            table.insert(parsedevs, d)
        end
    end

    if #parsedevs == 0 then
        err:add("No devices given on command line")
        return nil, err:retval()
    end

    config:debug(3, "Devices after parsing command line [%s]", table.concat(parsedevs, ", "))

    -- TODO: Make sure only valid devices are in parsedevs, otherwise
    --       createDevicesFromString will fail

    local devices = likwid.createDevicesFromString(table.concat(parsedevs, "@"))

    return devices, err:retval()
end


local function check_feature(config, name, permissions)
    local err = errors.new()
    if not permissions then
        permissions = "r"
    end
    for _, a in pairs(config.features) do
        local test = sysft_name(a.Category, a.Name)
        if test == name then
            if permissions == "rw" and a.ReadOnly then
                err:add("Feature %s is only read-only", name)
                return false, err:retval()
            end
            return true, err:retval()
        end
    end
    err:add("Feature %s not provided by the system", name)
    return false, err:retval()
end

local function get_feature(config, category, name)
    local err = errors.new()
    local fullname = sysft_name(category, name)
    if config.features[fullname] then
        return config.features[fullname], err:retval()
    else
        err:add("Feature %s not provided by the system", fullname)
    end
    return nil, err:retval()
--    for _, a in pairs(config.features) do
--        if category == a.Category and name == a.Name then
--            return a, nil
--        end
--    end
--    return nil, string.format("Feature %s not found", sysft_name(category, name))
end

local function test_intel_cpu(config)
    return config.cpuinfo.architecture == "x86_64" and config.cpuinfo.isIntel == 1
end

local function test_amd_cpu(config)
    return config.cpuinfo.architecture == "x86_64" and config.cpuinfo.isIntel == 0
end


local _print_frequencies_features = {
    {name = "HW Thread frequencies", category = "cpu_freq", devtype = likwid.hwthread, features = {
        {name = "Min", feature = "min_cpu_freq"},
        {name = "Current", feature = "cur_cpu_freq"},
        {name = "Max", feature = "max_cpu_freq"},
        {name = "Governor", feature = "governor"},
        {name = "Turbo", feature = "turbo"}
    }, test = function(config) return true end },
    {name = "Intel Uncore frequency", category = "uncore_freq", devtype = likwid.socket, features = {
        {name = "Min", feature = "min_uncore_freq"},
        {name = "Current", feature = "cur_uncore_freq"},
        {name = "Max", feature = "max_uncore_freq"},
    }, test = test_intel_cpu},
    {name = "Core boost limit", category = "hsmp", devtype = likwid.core, features = {
        {name = "Current", feature = "core_boost_limit_cur"},
    }, test = test_amd_cpu},
    {name = "AMD InfinityFabric frequency", category = "hsmp", devtype = likwid.socket, features = {
        {name = "Current", feature = "pkg_fclk"},
    }, test = test_amd_cpu},
    {name = "Memory Controller frequency", category = "hsmp", devtype = likwid.socket, features = {
        {name = "Current", feature = "pkg_mclk"},
    }, test = test_amd_cpu},
    {name = "Nvidia GPU clock", category = "nvml", devtype = likwid.nvidia_gpu, features = {
        {name = "Min", feature = "nvclockmin"},
        {name = "Current", feature = "nvclock"},
        {name = "Max", feature = "nvclockmax"},
    }, test = function(config) return true end},
}

local _list_frequencies_features = {
    {name = "HW Thread frequencies", category = "cpu_freq", devtype = likwid.hwthread, features = {
        {name = "Available", feature = "avail_freqs"},
    }, test = function(config) return config.features["cpu_freq.avail_freqs"] ~= nil end },
    {name = "HW Thread frequencies", category = "cpu_freq", devtype = likwid.hwthread, features = {
        {name = "Min", feature = "min_cpu_freq_limit"},
        {name = "Max", feature = "max_cpu_freq_limit"},
    }, test = function(config) return config.features["cpu_freq.avail_freqs"] == nil end },
    {name = "Intel Uncore frequency", category = "uncore_freq", devtype = likwid.socket, features = {
        {name = "Min", feature = "min_uncore_freq_limit"},
        {name = "Max", feature = "max_uncore_freq_limit"},
    }, test = test_intel_cpu},
    {name = "Nvidia GPU clock", category = "nvml", devtype = likwid.nvidia_gpu, features = {
        {name = "Min", feature = "nvclockmin"},
        {name = "Max", feature = "nvclockmax"},
    }, test = function(config) return true end},
}

local _list_governor_features = {
    {name = "HW Thread frequency governors", category = "cpu_freq", devtype = likwid.hwthread, features = {
        {name = "Available", feature = "avail_governors"},
    }, test = function(config) return true end},
}

local function get_best_unit(features)
    unitcounts = {}
    for _, f in pairs(features) do
        if f.feature.Unit then
            if not unitcounts[f.Unit] then
                unitcounts[f.feature.Unit] = 1
            else
                unitcounts[f.feature.Unit] = unitcounts[f.feature.Unit] + 1
            end
        end
    end
    max_unit_count = 0
    target_unit = nil
    for unit, count in pairs(unitcounts) do
        if count > max_unit_count then
            max_unit_count = count
            target_unit = unit
        end
    end
    return target_unit
end

local function getDeviceTypePrettyNames(devicetype)
    if devicetype == likwid.hwthread then
        return "HW Thread"
    elseif devicetype == likwid.hwthread then
        return "HW Thread"
    elseif devicetype == likwid.core then
        return "CPU core"
    elseif devicetype == likwid.socket then
        return "Socket"
    elseif devicetype == likwid.nvidia_gpu then
        return "Nvidia GPU"
    elseif devicetype == likwid.llc then
        return "Last-level cache"
    elseif devicetype == likwid.numa then
        return "NUMA domain"
    elseif devicetype == likwid.die then
        return "CPU die"
    elseif devicetype == likwid.amd_gpu then
        return "AMD GPU"
    elseif devicetype == intel_gpu then
        return "Intel GPU"
    end
    return "UNKNOWN"
end


local function gen_device_feature_table(config, devicetype, features)
    local err = errors.new()
    local first = setmetatable({getDeviceTypePrettyNames(devicetype)}, {
        align = "left",
    })
    if #features == 0 then
        return nil, "Empty feature table"
    end
    for _, dev in pairs(config.devices) do
        if dev:typeId() == devicetype then
            local devid = tostring(dev:id())
            config:debug(2, "Checking features for device %s", devid)
            table.insert(first, devid)
        end
    end
    local best_unit = config.unit
    if not best_unit then
        best_unit = get_best_unit(features)
    end
    local others = {}
    for _, f in pairs(features) do
        if f.feature.TypeID == devicetype then
            local h = f.header
            if f.feature.Unit and best_unit then
                h = string.format("%s [%s]", h, best_unit)
            end
            config:debug(2, "Filling column %s", h)
            local featstring = sysft_name(f.feature.Category, f.feature.Name)
            local tab = {h}
            for _, dev in pairs(config.devices) do
                if dev:typeId() == devicetype then
                    val, geterr = likwid.sysFeatures_get(featstring, dev)
                    if not val then
                        err:add(geterr)
                        table.insert(tab, "-")
                    else
                        if f.feature.Unit and best_unit then
                            local tmp = freq.new(val, f.feature.Unit)
                            tmp:scale(best_unit)
                            config:debug(3, "Unit conversion from %s to %s", f.feature.Unit, best_unit)
                            table.insert(tab, math.tointeger(math.floor(tmp:getvalue())))
                        else
                            table.insert(tab, val)
                        end
                    end
                end
            end
            table.insert(others, tab)
        else
            err:add("Feature %s is not provided for device type %d", sysft_name(f.Category, f.Name), devicetype)
        end
    end
    if not err:retval() then
        config:debug(2, "Preparing output table")
        local out = {first}
        for _, t in pairs(others) do
            table.insert(out, t)
        end
        return out, err:retval()
    end
    return nil, err:retval()
end

local function print_frequencies(config)
    local err = errors.new()
    config:debug(3, "Collecting frequencies of various devices for printing")
    for _, intab in pairs(_print_frequencies_features) do
        local cont = false
        if intab.test then
            cont = intab.test(config)
        end
        if cont == true then
            local feattab = {}
            for _, f in pairs(intab.features) do
                local feature, lerr = get_feature(config, intab.category, f.feature)
                if lerr then
                    err:append(lerr)
                else
                    if feature then
                        table.insert(feattab, {header=f.name, feature=feature})
                    end
                end
            end
            if #feattab > 0 then
                config:debug(3, "Generating table for device type %d with %d features", intab.devtype, #feattab)
                tab, lerr = gen_device_feature_table(config, intab.devtype, feattab)
                if tab then
                    config.stdout(intab.name)
                    if config.print_csv == false then
                        likwid.printtable(tab)
                    else
                        likwid.printcsv(tab, #feattab+1)
                    end
                else
                    if lerr then
                        err:append(lerr)
                    end
                end
            end
        end
    end
    return err:retval()
end


local function list_frequencies(config)
    local err = errors.new()
    config:debug(3, "Collecting frequencies of various devices for listing")
    for _, intab in pairs(_list_frequencies_features) do
        local cont = false
        if intab.test then
            cont = intab.test(config)
        end
        if cont == true then
            local feattab = {}
            for _, f in pairs(intab.features) do
                local feature, lerr = get_feature(config, intab.category, f.feature)
                if lerr then
                    err:append(lerr)
                else
                    if feature then
                        table.insert(feattab, {header=f.name, feature=feature})
                    end
                end
            end
            if #feattab > 0 then
                config:debug(3, "Generating table for device type %d with %d features", intab.devtype, #feattab)
                tab, lerr = gen_device_feature_table(config, intab.devtype, feattab)
                if tab then
                    config.stdout(intab.name)
                    if config.print_csv == false then
                        likwid.printtable(tab)
                    else
                        likwid.printcsv(tab, #feattab+1)
                    end
                else
                    if lerr then
                        err:append(lerr)
                    end
                end
            end
        end
    end
    return err:retval()
end


local function list_governors(config)
    local err = errors.new()
    config:debug(3, "Collecting governors of various devices for listing")
    for _, intab in pairs(_list_governor_features) do
        local feattab = {}
        for _, f in pairs(intab.features) do
            local feature, lerr = get_feature(config, intab.category, f.feature)
            if lerr then
                err:append(lerr)
            else
                if feature then
                    table.insert(feattab, {header=f.name, feature=feature})
                end
            end
        end
        if #feattab > 0 then
            config:debug(3, "Generating table for device type %d with %d features", intab.devtype, #feattab)
            tab, lerr = gen_device_feature_table(config, intab.devtype, feattab)
            if tab then
                config.stdout(intab.name)
                if config.print_csv == false then
                    likwid.printtable(tab)
                else
                    likwid.printcsv(tab, #feattab+1)
                end
            else
                if lerr then
                    err:append(lerr)
                end
            end
        end
    end
    return err:retval()
end


local _validate_freq_table = {
    {name = "HW Thread frequencies", config_min = "min_freq",
        config_max = "max_freq", devicetype = likwid.hwthread,
        test = function(config)
            return config.hwthread_set
        end,
        feature_min = {category = "cpu_freq", feature = "min_cpu_freq_limit"},
        feature_max = {category = "cpu_freq", feature = "max_cpu_freq_limit"},
        feature_list = {category = "cpu_freq", feature = "avail_freqs"}
    },
    {name = "Intel Uncore frequencies", config_min = "umin",
        config_max = "umax", devicetype = likwid.socket,
        test = function(config)
            return config.cpuinfo.isIntel and config.uncore_set
        end,
        feature_min = {category = "uncore_freq", feature = "min_uncore_freq_limit"},
        feature_max = {category = "uncore_freq", feature = "max_uncore_freq_limit"}
    },
    {name = "AMD core boost frequencies", config_max = "cmax", devicetype = likwid.core,
        test = function(config)
            return (config.cpuinfo.isIntel == false) and config.cmax_set
        end,
        feature_max = {category = "hsmp", feature = "core_boost_limit_cur"}
    },
    {name = "AMD socket boost frequencies", config_max = "smax", devicetype = likwid.socket,
        test = function(config)
            return (config.cpuinfo.isIntel == false) and config.smax_set
        end,
        feature_max = {category = "hsmp", feature = "pkg_boost_limit_cur"}
    },
    {name = "Nvidia GPU frequency", config_max = "nmax", devicetype = likwid.nvidia_gpu,
        test = function(config)
            return config.nvidia_set
        end,
        feature_min = {category = "nvml", feature = "nvclockmin"},
        feature_max = {category = "nvml", feature = "nvclockmax"},
    },
}

local function validate_frequencies(config)
    local err = errors.new()
    config:debug(1, "Validate frequencies")

    check_bounds = function(config, conf, testvalue, table_key_suffix)
        local err = errors.new()
        local config_key = string.format("config_%s", table_key_suffix)

        if conf[config_key] and config[conf[config_key]] and conf.test(config) then
            for _, dev in pairs(config.devices) do
                if dev:typeId() == conf.devicetype then
                    if conf.feature_min then
                        local min_limit_feat = get_feature(config, conf["feature_min"].category, conf["feature_min"].feature)
                        if min_limit_feat then
                            local min_limit_name = sysft_name(conf["feature_min"].category, conf["feature_min"].feature)
                            local min_limit = likwid.sysFeatures_get(min_limit_name, dev)
                            if tonumber(config[conf[config_key]]) < tonumber(min_limit) then
                                err:add("%s: Frequency %s below minimal allowed frequency of %s for device %s%d", conf.name, config[conf[config_key]], min_limit, dev:typeName(), dev:id())
                            end
                        end
                    end
                    if conf.feature_max then
                        local max_limit_feat = get_feature(config, conf["feature_max"].category, conf["feature_max"].feature)
                        if max_limit_feat then
                            local max_limit_name = sysft_name(conf["feature_max"].category, conf["feature_max"].feature)
                            local max_limit = likwid.sysFeatures_get(max_limit_name, dev)
                            if tonumber(config[conf[config_key]]) > tonumber(max_limit) then
                                err:add("%s: Frequency %s above maximal allowed frequency of %s for device %s%d", conf.name, config[conf[config_key]], max_limit, dev:typeName(), dev:id())
                            end
                        end
                    end
                    if conf.feature_list then
                        local limit_list_feat = get_feature(config, conf["feature_list"].category, conf["feature_list"].feature)
                        if limit_list_feat then
                            local limit_list_name = sysft_name(conf["feature_list"].category, conf["feature_list"].feature)
                            local limit_list = likwid.sysFeatures_get(limit_list_name, dev)
                            local valid = false
                            for f in string.gmatch(limit_list, "%d+") do
                                if f == config[conf[config_key]] then
                                    valid = true
                                    break
                                end
                            end
                            if valid == false then
                                err:add("%s: Frequency %s not provided for device %s%d", conf.name, config[conf[config_key]], dev:typeName(), dev:id())
                            end
                        end
                    end
                end
            end
        end


        return err:retval()
    end


    for _, conf in pairs(_validate_freq_table) do
        local lerr = check_bounds(config, conf, config[conf["config_min"]], "min")
        if lerr then
            err:append(lerr)
        end
        local lerr = check_bounds(config, conf, config[conf["config_max"]], "max")
        if lerr then
            err:append(lerr)
        end
    end

    return err:retval()
end


local function set_frequencies(config)
    local err = errors.new()
    config:debug(1, "Set frequencies")

    set_feature = function(config, category, name, value)
        local ferr = errors.new()
        feat = get_feature(config, category, name)
        for _, dev in pairs(config.devices) do
            if dev:typeId() == feat.TypeID then
                local featkey = sysft_name(category, name)
                config:debug(3, "Setting feature %s to %s for device %s%d", featkey, value, dev:typeName(), dev:id())
                ok, lerr = likwid.sysFeatures_set(featkey, dev, value)
                if not ok then
                    ferr:add("Failed to set feature %s for device %s%d: %s", featkey, dev:typeName(), dev:id(), lerr)
                end
            end
        end
        return ferr:retval()
    end

    if config.hwthread_set and config.min_freq then
        local lerr = set_feature(config, "cpu_freq", "min_cpu_freq", config.min_freq)
        if lerr then
            err:append(lerr)
        end
    end
    if config.hwthread_set and config.max_freq then
        local lerr = set_feature(config, "cpu_freq", "max_cpu_freq", config.max_freq)
        if lerr then
            err:append(lerr)
        end
    end

    if config.cpuinfo.isIntel and config.uncore_set then
        if config.umin then
            local lerr = set_feature(config, "uncore_freq", "min_uncore_freq", config.umin)
            if lerr then
                err:append(lerr)
            end
        end
        if config.umax then
            local lerr = set_feature(config, "uncore_freq", "max_uncore_freq", config.umax)
            if lerr then
                err:append(lerr)
            end
        end
    end

    if (config.cpuinfo.isIntel == false) and config.cmax_set then
        if config.cmax then
            local lerr = set_feature(config, "hsmp", "core_boost_limit_cur", config.cmax)
            if lerr then
                err:append(lerr)
            end
        end
    end

    if (config.cpuinfo.isIntel == false) and config.smax_set then
        if config.smax then
            local lerr = set_feature(config, "hsmp", "pkg_boost_limit_cur", config.smax)
            if lerr then
                err:append(lerr)
            end
        end
    end

    -- nvml
    -- rocm-smi

    return err:retval()
end

local _validate_governor_table = {
    {name = "HW Thread governors", config_gov = "set_governor",
        devicetype = likwid.hwthread,
        test = function(config)
            return true
        end,
        feature_list = {category = "cpu_freq", feature = "avail_governors"}
    },
}

local function validate_governors(config)
    local err = errors.new()
    config:debug(1, "Validate Governors")
    
    for _, conf in pairs(_validate_governor_table) do
        if conf.config_gov and config[conf.config_gov] then
            for _, dev in pairs(config.devices) do
                if conf.devicetype == dev:typeId() then
                    if conf.feature_list then
                        local limit_list_feat = get_feature(config, conf["feature_list"].category, conf["feature_list"].feature)
                        if limit_list_feat then
                            local limit_list_name = sysft_name(conf["feature_list"].category, conf["feature_list"].feature)
                            local limit_list = likwid.sysFeatures_get(limit_list_name, dev)
                            local valid = false
                            for f in string.gmatch(limit_list, "%a+") do
                                if f == config[conf.config_gov] then
                                    valid = true
                                    break
                                end
                            end
                            if valid == false then
                                err:add("%s: Governor %s not provided for device %s%d", conf.name, config[conf.config_gov], dev:typeName(), dev:id())
                            end
                        end
                    end
                end
            end
        end
    end
    return err:retval()
end

local function set_governors(config)
    local err = errors.new()
    config:debug(1, "Set Governors")

    set_feature = function(config, category, name, value)
        local ferr = errors.new()
        feat = get_feature(config, category, name)
        for _, dev in pairs(config.devices) do
            if dev:typeId() == feat.TypeID then
                local featkey = sysft_name(category, name)
                config:debug(3, "Setting feature %s to %s for device %s%d", featkey, value, dev:typeName(), dev:id())
                ok, lerr = likwid.sysFeatures_set(featkey, dev, value)
                if not ok then
                    ferr:add("Failed to set governor %s for device %s%d: %s", value, dev:typeName(), dev:id(), lerr)
                end
            end
        end
        return ferr:retval()
    end

    if config.set_governor then
        local lerr = set_feature(config, "cpu_freq", "governor", config.set_governor)
        if lerr then
            err:append(lerr)
        end
    end

    return err:retval()
end

local function get_sysfeatures_list(config)
    local tab = {}
    local err = errors.new()
    local sf_list = likwid.sysFeatures_list()
    if sf_list and #sf_list > 0 then
        for _, f in pairs(sf_list) do
            tab[sysft_name(f.Category, f.Name)] = f
        end
    else
        err:add("Failed to get list of provided features")
    end
    return tab, err:retval()
end

local function main(...)
    local err = errors.new()
    -- used temporarly for function calls to append it to err
    local lerr = nil
    local interr = 0
    local retcode = 0
    local cliargs = {...}

    local config, lerr = read_cliargs(cliargs)
    if lerr then
        err:append(lerr)
        return err:retval()
    end

    if (likwid.sysFeaturesSupported() ~= 1) then
        err:add("LIKWID installation does not support sysfeatures")
        return err:retval()
    end

    config.topology = likwid.getCpuTopology()

    interr = likwid.initSysFeatures()
    if interr ~= 0 then
        err:add("Failed to initialize LIKWID sysfeatures")
        return err:retval()
    end

    config.features, lerr = get_sysfeatures_list()
    if lerr then
        err:append(lerr)
        return err:retval()
    end

    config.devices, lerr = resolve_devices(config)
    if lerr then
        err:append(lerr)
        return err:retval()
    end

    if config.printfreq then
        local lerr = print_frequencies(config)
        if lerr then
            err:append(lerr)
        end
    end

    if config.listfreq then
        local lerr = list_frequencies(config)
        if lerr then
            err:append(lerr)
        end
    end

    if config.listgov then
        local lerr = list_governors(config)
        if lerr then
            err:append(lerr)
        end
    end


    lerr = validate_frequencies(config)
    if lerr then
        err:append(lerr)
    else
        lerr = set_frequencies(config)
        if lerr then
            err:append(lerr)
        end
    end

    lerr = validate_governors(config)
    if lerr then
        err:append(lerr)
    else
        lerr = set_governors(config)
        if lerr then
            err:append(lerr)
        end
    end

    likwid.finalizeSysFeatures()
    likwid.putTopology()
    return err:retval()
end

local err = main(table.unpack(arg))
if err then
    io.stderr:write(err:tostring())
    os.exit(err:retcode())
end
os.exit(0)

