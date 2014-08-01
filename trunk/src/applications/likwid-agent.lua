#!/home/rrze/unrz/unrz139/Work/likwid/trunk/ext/lua/lua


require("liblikwid")
local likwid = require("likwid")



if #arg ~= 2 then
    print("Need EventSet and measurement duration in seconds on commandline")
    os.exit(1)
end

local eventSet = arg[1]
local duration = arg[2]


local cputopo = likwid_getCpuTopology()
local config = likwid_getConfiguration()

local cpulist = {}

for i, thread in pairs(cputopo["threadPool"]) do
    table.insert(cpulist, thread["apicId"])
end

access_mode = config["daemonMode"]
if access_mode < 0 then
    access_mode = 1
end

if likwid_setAccessClientMode(access_mode) ~= 0 then
    os.exit(1)
end
likwid_init(cputopo["numHWThreads"], cpulist)

local groupID = likwid_addEventSet(eventSet)

likwid_setupCounters(groupID)

likwid_startCounters()
sleep(duration)
likwid_stopCounters()

local results = likwid.getResults()
for ig,g in pairs(results) do
    str = tostring(ig)..","..tostring(cputopo["numHWThreads"])
    for ie=0,#g do
        str = str .. "," ..tostring(ie)
        for it=0,#g[ie] do
            str = str .. ","..tostring(g[ie][it])
        end
    end
    print(str)
end

likwid_finalize()
