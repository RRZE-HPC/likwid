#!/home/rrze/unrz/unrz139/Work/likwid/trunk/ext/lua/lua


require("liblikwid")
local likwid = require("likwid")

-- Read commandline arguments
if #arg ~= 2 then
    print("Need EventSet and measurement duration in seconds on commandline")
    os.exit(1)
end
local eventSet = arg[1]
local duration = arg[2]

-- Get architectural information for the current system
local cpuinfo = likwid_getCpuInfo()
local cputopo = likwid_getCpuTopology()
-- Read configuration file
local config = likwid_getConfiguration()

-- Add all cpus to the cpulist
local cpulist = {}
for i, thread in pairs(cputopo["threadPool"]) do
    table.insert(cpulist, thread["apicId"])
end

-- Select access mode to msr devices, try configuration file first
access_mode = config["daemonMode"]
if access_mode < 0 then
    access_mode = 1
end
if likwid_setAccessClientMode(access_mode) ~= 0 then
    os.exit(1)
end

-- Evaluate eventSet given on commandline. If it's a group, resolve to events
local s,e = eventSet:find(":")
gdata = nil
if s == nil then
    gdata = likwid.get_groupdata(cpuinfo["short_name"], eventSet)
    eventSet = gdata["EventString"]
else
    gdata = likwid.new_groupdata(eventSet)
end

-- Initialize likwid perfctr
likwid_init(cputopo["numHWThreads"], cpulist)
local groupID = likwid_addEventSet(eventSet)

likwid_setupCounters(groupID)

-- Perform the measurement
likwid_startCounters()
sleep(duration)
likwid_stopCounters()

-- Read all results and print them in CSV
local results = likwid.getResults()
for ig,g in pairs(results) do
    str = tostring(ig)..","..tostring(cputopo["numHWThreads"])
    for ie=0,#g do
        str = str .. "," ..gdata["Events"][ie]["Event"]
        for it=0,#g[ie] do
            str = str .. ","..tostring(g[ie][it])
        end
    end
    print(str)
end

-- Finalize likwid perfctr
likwid_finalize()
