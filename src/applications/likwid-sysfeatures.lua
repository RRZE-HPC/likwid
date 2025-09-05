#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-sysfeatures.lua
 *
 *      Description:  A application to retrieve and manipulate CPU features.
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
    print_stdout(string.format("likwid-features -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

local function example()
    print_stdout("Options:")
end

local function usage()
    version()
    print_stdout("A tool list and modify the states of system/OS features.\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t\t Help message")
    print_stdout("-v, --version\t\t Version information")
    print_stdout("-a, --all\t\t List all available features")
    print_stdout("-l, --list\t\t List features and state for given devices")
    print_stdout("-p, --print\t\t List available devices")
    print_stdout("-d, --devices <list>\t Perform operations on given devices")
    print_stdout("-g, --get <feature(s)>\t Get the value of the given feature(s)")
    print_stdout("                      \t feature format: <category>.<name> or just <name> if unique")
    print_stdout("                      \t can be a comma-separated list of features")
    print_stdout("-s, --set <feature(s)>\t Set feature(s) to the given value")
    print_stdout("                      \t format: <category>.<name>=<value> or just <name>=<value> if unique")
    print_stdout("                      \t can be a comma-separated list of features")
    print_stdout("    --saveall <file>  \t Save all available rw-features to file")
    print_stdout("    --loadall <file>  \t Load all features from file")
    print_stdout("-O\t\t\t Output results in CSV")
    print_stdout("-V, --verbose <level>\t Set verbosity\n")
end

local function toGetTable(devList, ft_list)
    -- prepare output table
    local all = {}
    -- first column contains the feature category and name
    local first = {}
    table.insert(first, "Feature/Dev")
    for _,f in pairs(ft_list) do
        table.insert(first, string.format("%s.%s", f.Category, f.Name))
    end
    setmetatable(first, {align = "left"})
    table.insert(all, first)
    -- create one column per given device with the current value of the feature
    for i, dev in pairs(devList) do
        local tab = {}
        table.insert(tab, string.format("%s %s", dev:typeName(), dev:id()))
        for _,f in pairs(ft_list) do
            if dev:typeId() == f.TypeID then
                if f.WriteOnly then
                    table.insert(tab, "(wronly)")
                else
                    local v, err = likwid.sysFeatures_get(string.format("%s.%s", f.Category, f.Name), dev)
                    if v == nil then
                        table.insert(tab, "fail")
                    else
                        table.insert(tab, v)
                    end
                end
            else
                table.insert(tab, "-")
            end
        end
        -- add the hw thread column to the table
        table.insert(all, tab)
    end
    return all
end

if #arg == 0 then
    usage()
    os.exit(0)
end

-- main variables with defaults
local listFeatures = false
local allFeatures = false
local printDevices = false
local devList = {}
local getList = {}
local setList = {}
local cpuinfo = likwid.getCpuInfo()
local cputopo = likwid.getCpuTopology()
local affinity = likwid.getAffinityInfo()
local verbose = 0
local output_csv = false

-- parse the command line
for opt,arg in likwid.getopt(arg, {"h","v","l","p","d:","g:","s:","a", "O","help","version","list", "print", "set:", "get:","all", "cpus:", "V:", "verbose:", "saveall:", "loadall:"}) do
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
    elseif opt == "d" or opt == "devices"then
        devList = likwid.createDevicesFromString(arg)
    elseif opt == "l" or opt == "list" then
        listFeatures = true
    elseif opt == "p" or opt == "print" then
        printDevices = true
    elseif opt == "O" then
        output_csv = true
    elseif opt == "V" or opt == "verbose" then
        verbose = tonumber(arg)
    elseif opt == "a" or opt == "all" then
        allFeatures = true
    elseif opt == "g" or opt == "get" then
        getList = likwid.stringsplit(arg, ",")
    elseif opt == "s" or opt == "set" then
        setList = likwid.stringsplit(arg, ",")
    elseif opt == "saveall" then
        saveFeatures = arg
    elseif opt == "loadall" then
        loadFeatures = arg
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end


-- validate command line input
if (not printDevices) and (not listFeatures) and (not allFeatures) and (not saveFeatures) and (not loadFeatures) and (#getList == 0) and (#setList == 0) then
    print_stderr("No operations specified, exiting...")
    os.exit(1)
end
if (printDevices or listFeatures or allFeatures or saveFeatures or loadFeatures) and (#getList > 0 or #setList > 0) then
    print_stderr("Cannot list features and get/set/load/save at the same time")
    os.exit(1)
end
if #devList == 0 then
    if listFeatures then
        print_stderr("Device selection (-d) required for listing the state of all features")
        os.exit(1)
    elseif #getList > 0 then
        print_stderr("Device selection (-d) required for getting the state of given features")
        os.exit(1)
    elseif #setList > 0 then
        print_stderr("Device selection (-d) required for setting the state of given features")
        os.exit(1)
    end
end

-- set verbosity
if verbose > 0 and verbose <= 3 then
    likwid.setVerbosity(verbose)
end

-- initialize the sysfeatures module
local err = likwid.initSysFeatures()
if err < 0 then
    print_stderr("Cannot initialize HW features module")
    os.exit(1)
end

-- get a list of all features for the system
local ft_list = likwid.sysFeatures_list()

-- print available devices
device_types = {}
device_types[likwid.hwthread] = "HWThread (T)"
device_types[likwid.core] = "Core (C)"
device_types[likwid.numa] = "NUMA (M)"
device_types[likwid.die] = "Die (D)"
device_types[likwid.socket] = "Socket (S)"
device_types[likwid.node] = "Node (N)"
if likwid.nvSupported() then
    device_types[likwid.nvidia_gpu] = "Nvidia GPU (GN)"
end
if likwid.rocmSupported() then
    device_types[likwid.amd_gpu] = "AMD GPU (GA)"
end

if printDevices then
    for devtype, name in pairs(device_types) do
        print(string.format("%s:", name))
        devices = likwid.getAvailableDevices(devtype)
        if #devices == 0 then
            print("\t<none>")
        else
            print("\t" .. table.concat(devices, ","))
        end
    end
end

-- print the list
if allFeatures then
    local all = {}
    local names = {}
    local types = {}
    local access = {}
    local descs = {}
    local cats = {}
    -- create a table of categories for sorting
    for _,f in pairs(ft_list) do
        found = false
        for _, c in pairs(cats) do
            if c == f.Category then
                found = true
            end
        end
        if not found then
            table.insert(cats, f.Category)
        end
    end
    table.sort(cats)

    -- prepare table for output
    -- first we create tables for the four columns
    table.insert(names, "Feature")
    table.insert(types, "Scope")
    table.insert(access, "Access")
    table.insert(descs, "Description")
    for _,c in pairs(cats) do
        for _,f in pairs(ft_list) do
            if f.Category == c then
                table.insert(names, string.format("%s.%s", f.Category, f.Name))
                table.insert(types, f.Type)
                table.insert(descs, f.Description)
                if f.ReadOnly then
                    table.insert(access, "rdonly")
                elseif f.WriteOnly then
                    table.insert(access, "wronly")
                else
                    table.insert(access, "rw")
                end
            end
        end
    end
    -- add all columns to the table
    setmetatable(names, {align = "left"})
    table.insert(all, names)
    table.insert(all, types)
    table.insert(all, access)
    setmetatable(descs, {align = "left"})
    table.insert(all, descs)

    -- print the table in selected format
    if output_csv then
        likwid.printcsv(all, #all)
    else
        print_stdout("Available features:")
        likwid.printtable(all)
    end

    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end

if listFeatures and #devList > 0 then
    -- prepare output table and get all features
    local all = toGetTable(devList, ft_list)

    -- print the table
    if output_csv then
        likwid.printcsv(all, #devList + 1)
    else
        likwid.printtable(all)
    end

    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end

if #getList > 0 and #devList > 0 then
    -- filter the full feature list to have only the selected ones
    -- the new list entry contains all entries of the original feature entry plus
    -- the user-given value
    local featList = {}
    for _, f in pairs(getList) do
        for _, l in pairs(ft_list) do
            if f == l.Name or f == string.format("%s.%s", l.Category, l.Name) then
                table.insert(featList, {
                    TypeID = l.TypeID,
                    Type = l.Type,
                    Name = l.Name,
                    Category = l.Category,
                    ReadOnly = l.ReadOnly,
                    WriteOnly = l.WriteOnly,
                })
            end
        end
    end

    -- get all features in the new list
    local tab = toGetTable(devList, featList)
    -- print the table
    if output_csv then
        likwid.printcsv(tab, #devList + 1)
    else
        likwid.printtable(tab)
    end

    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end

if #setList > 0 and #devList > 0 then
    -- filter the full feature list to have only the selected ones
    -- the new list entry contains all entries of the original feature entry plus
    -- the user-given value
    local featList = {}
    for _, f in pairs(setList) do
        local t = likwid.stringsplit(f, "=")
        if #t == 2 then
            featList[t[1]] = t[2]
        else
            print_stderr(string.format("Invalid format of '%s' in set list", f))
        end
    end

    -- set all features in the new list
    for i, dev in pairs(devList) do
        for featureName, featureValue in pairs(featList) do
            if verbose > 0 then
                print_stdout(string.format("Setting '%s' to '%s'", featureName, featureValue))
            end
            local v, err = likwid.sysFeatures_set(featureName, dev, featureValue)
            if not v then
                print_stderr(string.format("Failed to set feature '%s' to '%s'", featureName, featureValue))
                err:printStderr()
            end
        end
    end
    -- finalize sysfeatures module and exit
    likwid.finalizeSysFeatures()
    os.exit(0)
end

-- save all read/write features to file
if saveFeatures then
    local file = io.open(saveFeatures, "w")
    -- iterate over all device types
    for devtype, _ in pairs(device_types) do
        -- iterate over all features
        for _,f in pairs(ft_list) do
            local full_name = string.format("%s.%s", f.Category, f.Name)
            -- only allow matching device types and if feature is readable and writable
            if f.TypeID ~= devtype or f.ReadOnly or f.WriteOnly then
                goto next_feat
            end

            -- actually read the features
            for _, dev in pairs(likwid.getAllDevices(devtype)) do
                local lw_dev = likwid.createDevice(devtype, dev)
                local v, err = likwid.sysFeatures_get(full_name, lw_dev)
                if err then
                    print_stderr(string.format("Failed to get feature '%s' on device %s:%d: %s", full_name, lw_dev:typeName(), lw_dev:id(), err))
                    goto next_feat
                end
                file:write(string.format("%s.%s@%s=%s\n", f.Category, f.Name, lw_dev:id(), v))
            end

            ::next_feat::
        end
    end

    file:close()

    likwid.finalizeSysFeatures()
    os.exit(0)
end

-- load all features from file
if loadFeatures then
    for line in io.lines(loadFeatures) do
        -- split string like the following: cpu_freq.governor@5=schedutil
        local part1 = likwid.stringsplit(line, "=")
        if #part1 ~= 2 then
            print_stderr("Invalid line: " .. line)
            os.exit(1)
        end
        local part2 = likwid.stringsplit(part1[1], "@")
        if #part2 ~= 2 then
            print_stderr("Invalid line: " .. line)
            os.exit(1)
        end
        part3 = likwid.stringsplit(part2[1], ".")
        if #part3 ~= 2 then
            print_stderr("Invalid line: " .. line)
            os.exit(1)
        end
        local feat_cat = part3[1]
        local feat_name = part3[2]
        local dev_id = part2[2]
        local value = part1[2]

        local full_name = string.format("%s.%s", feat_cat, feat_name)

        -- get device type of this particular feature
        local devtype = nil
        for _, f in pairs(ft_list) do
            if f.Name == feat_name and f.Category == feat_cat then
                devtype = f.TypeID
                break
            end
        end
        if not devtype then
            print_stderr(string.format("Unknown feature: '%s'", full_name))
            os.exit(1)
        end
        local lw_dev = likwid.createDevice(devtype, dev_id)
        local success, err = likwid.sysFeatures_set(full_name, lw_dev, value)
        if not success then
            print_stderr(string.format("Failed to set feature '%s' on device %s:%d to %s: %s", full_name, lw_dev:typeName(), lw_dev:id(), value, err))
        end
    end

    likwid.finalizeSysFeatures()
    os.exit(0)
end

likwid.finalizeSysFeatures()
os.exit(0)
