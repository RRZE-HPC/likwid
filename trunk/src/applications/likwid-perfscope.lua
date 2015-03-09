#!/home/rrze/unrz/unrz139/TMP/likwid-base/trunk/ext/lua/lua

package.path = package.path .. ';/home/rrze/unrz/unrz139/TMP/likwid-base/trunk/?.lua'

local likwid = require("likwid")

PERFCTR="/home/rrze/unrz/unrz139/TMP/likwid-base/trunk/likwid-perfctr"
FEEDGNUPLOT="/home/rrze/unrz/unrz139/TMP/likwid-base/trunk/perl/feedGnuplot"

local predefined_plots = {
    FLOPS_DP = {
        perfgroup = "FLOPS_DP",
        metricmatch = "DP MFlops/s",
        title = "Double Precision Flop Rate",
        ytitle = "MFlops/s",
        y2title = nil,
        xtitle = "Time"
    },
    L2 = {
        perfgroup = "L2",
        metricmatch = "L2 bandwidth [MBytes/s]",
        title = "L2 cache bandwidth",
        ytitle = "bandwidth [MB/s]",
        y2title = nil,
        xtitle = "Time"
    }
}

function copy_list(input)
    output = {}
    for i, item in pairs(input) do
        if type(item) == "table" then
            output[i] = copy_list(item)
        else
            output[i] = input[i]
        end
    end
    return output
end

local verbosity = 0
local eventStrings = {}
local terminal = "x11"
local num_cpus = 0
local cpulist = {}
local matchstring = nil
local group_list = {}
local timeline = "1s"
local outputfile = nil
local outputfilename = nil
local outputfilesuffix = nil

for opt,arg in likwid.getopt(arg, "hvV:g:C:t:o:") do
    if (opt == "h") then
        usage()
        os.exit(0)
    elseif (opt == "v") then
        version()
        os.exit(0)
    elseif (opt == "V") then
        verbosity = tonumber(arg)
    elseif (opt == "g") then
        table.insert(eventStrings, arg)
    elseif (opt == "C") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
    elseif (opt == "t") then
        timeline = arg
    elseif (opt == "o") then
        outputfile = arg
    end
end

if num_cpus == 0 then
    print("ERROR: CPU string must be given")
    os.exit(1)
end

if #arg == 0 then
    print("ERROR: Executable must be given on commandline")
    os.exit(1)
end

for i, event_def in pairs(eventStrings) do
    local eventlist = likwid.stringsplit(event_def,",")
    outopts = eventlist[#eventlist]
    table.remove(eventlist, #eventlist)
    event_string = table.concat(eventlist,",")

    local groupdata = nil
    groupdata = likwid.get_groupdata(event_string)
    if groupdata == nil then
        print("Cannot read event string, it's neither a performance group nor a proper event string <event>:<counter>:<options>,...")
        usage()
        os.exit(1)
    end
    if group_list[i] == nil then
        group_list[i] = {}
    end
    group_list[i]["gdata"] = groupdata

    formula = nil
    local title = nil
    local ytitle = nil
    local y2title = nil
    local xtitle = nil
    for j,estr in pairs(likwid.stringsplit(outopts, ":")) do
        if estr:match("formula=([%g]+)") then
            if not formula then formula = {} end
            table.insert(formula, estr:match("formula=([%g_]+)"))
        elseif estr:match("FORMULA=([%g]+)") then
            if not formula then formula = {} end
            table.insert(formula, estr:match("FORMULA=([%g_]+)"))
        elseif estr:match("^title=([%g%s]+)") and not title then
            title = estr:match("^title=([%g%s]+)")
        elseif estr:match("^TITLE=([%g%s]+)") and not title then
            title = estr:match("^TITLE=([%g%s]+)")
        elseif estr:match("ytitle=(%g+)") and not ytitle then
            ytitle = estr:match("ytitle=(%g+)")
        elseif estr:match("YTITLE=(%g+)") and not ytitle then
            ytitle = estr:match("YTITLE=(%g+)")
        elseif estr:match("y2title=(%g+)") and not y2title then
            y2title = estr:match("y2title=(%g+)")
        elseif estr:match("Y2TITLE=(%g+)") and not y2title then
            y2title = estr:match("Y2TITLE=(%g+)")
        elseif estr:match("xtitle=(%g+)") and not xtitle then
            xtitle = estr:match("xtitle=(%g+)")
        elseif estr:match("XTITLE=(%g+)") and not xtitle then
            xtitle = estr:match("XTITLE=(%g+)")
        end
    end
    group_list[i]["eventstring"] = event_string
    group_list[i]["counterlist"] = {}
    for k=1,#groupdata["Events"] do
        table.insert(group_list[i]["counterlist"], groupdata["Events"][k]["Counter"])
    end
    if title then
        group_list[i]["title"] = title
    end
    if ytitle then
        group_list[i]["ytitle"] = ytitle
    end
    if y2title then
        group_list[i]["y2title"] = y2title
    end
    if xtitle then
        group_list[i]["xtitle"] = xtitle
    end
    group_list[i]["formulas"] = formula
end

cmd = string.format("%s -C %s -t %s ",PERFCTR, table.concat(cpulist,","), timeline)
for i, group in pairs(group_list) do
    cmd = cmd .. " -g "..group["eventstring"]
end
cmd = cmd .. " "..table.concat(arg, " ")
cmd = cmd .. " 3>&1 1>&2 2>&3 3>&-"

f = assert (io.popen (cmd))


for i, group in pairs(group_list) do
    gnucmd = string.format("%s --stream --lines --domain --nodataid ", FEEDGNUPLOT)
    if group["title"] ~= nil then
        gnucmd = gnucmd .. string.format(" --title %q", "Group "..i..": "..group["title"])
    end
    if group["xtitle"] ~= nil then
        gnucmd = gnucmd .. string.format(" --xlabel %q", group["xtitle"])
    end
    if group["ytitle"] ~= nil then
        gnucmd = gnucmd .. string.format(" --ylabel %q", group["ytitle"])
    end
    if group["y2title"] ~= nil then
        gnucmd = gnucmd .. string.format(" --y2 --y2label %q", group["y2title"])
    end
    for f,formula in pairs(group["formulas"]) do
        gnucmd = gnucmd .. string.format(" --legend %d %q", f, formula)
    end
    gnucmd = gnucmd .. "1>/dev/null 2>&1"
    group_list[i]["output"] = assert(io.popen(gnucmd,"w"))
end

alldata = nil
olddata = nil
data = nil
while true do
    local l = f:read("*line")
    if l == nil or l:match("^%s*$") then break end

    linelist = likwid.stringsplit(l, ",")
    group = tonumber(linelist[1])
    nr_events = tonumber(linelist[2])
    nr_threads = tonumber(linelist[3])
    time = tonumber(linelist[4])
    table.remove(linelist, 1)
    table.remove(linelist, 1)
    table.remove(linelist, 1)
    table.remove(linelist, 1)

    if olddata == nil then
        olddata = {}
    end
    if alldata == nil then
        alldata = {}
    end
    if olddata[group] == nil then
        olddata[group] = {}
    end
    if alldata[group] == nil then
        alldata[group] = {}
    end
    if data ~= nil then olddata[group] = copy_list(data) end
    data = {}
    for j=1,nr_threads do
        data[j] = {}
        if alldata[group][j] == nil then
            alldata[group][j] = {}
        end
        if alldata[group][j]["time"] ~= nil then
            data[j]["time"] = time - alldata[group]["time"]
        else
            data[j]["time"] = time
        end
        alldata[group]["time"] = time
    end
    for i=1,nr_events do
        for j=1,nr_threads do
            if olddata[group][j] ~= nil then
                data[j][group_list[group]["counterlist"][i]] = tonumber(linelist[i]) - olddata[group][j][group_list[group]["counterlist"][i]]
            else
                data[j][group_list[group]["counterlist"][i]] = tonumber(linelist[i])
            end
            alldata[group][j][group_list[group]["counterlist"][i]] = tonumber(linelist[i])
            table.remove(linelist, 1)
        end
    end
    results = {}
    str = tostring(alldata[group]["time"])
    for i, thread in pairs(data) do
        results[i] = {}
        for j,formula in pairs(group_list[group]["formulas"]) do
            results[i][j] = likwid.calculate_metric(formula, thread)
            str = str .. " " ..tostring(results[i][j])
        end
    end
    group_list[group]["output"]:write(str.."\n")
    group_list[group]["output"]:flush()
end

io.close(f)
--for i, group in pairs(group_list) do
    --group["output"]:write("exit\n")
    --io.close(group["output"])
--end


