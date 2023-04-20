#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-perfscope.lua
 *
 *      Description:  An application to use the timeline mode of likwid-perfctr to generate
 *                    realtime plots using feedGnuplot
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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

PERFCTR="<INSTALLED_BINPREFIX>/likwid-perfctr"
FEEDGNUPLOT="<INSTALLED_BINPREFIX>/feedGnuplot"

local predefined_plots = {
    FLOPS_DP = {
        perfgroup = "FLOPS_DP",
        ymetricmatch = "MFlops/s",
        title = "Double Precision Flop Rate",
        ytitle = "MFlops/s",
        y2title = nil,
        xtitle = "Time"
    },
    FLOPS_SP = {
        perfgroup = "FLOPS_SP",
        ymetricmatch = "MFlops/s",
        title = "Single Precision Flop Rate",
        ytitle = "MFlops/s",
        y2title = nil,
        xtitle = "Time"
    },
    L2 = {
        perfgroup = "L2",
        ymetricmatch = "L2D load bandwidth [MBytes/s]",
        title = "L2 cache bandwidth",
        ytitle = "Load Bandwidth [MBytes/s]",
        y2metricmatch = "L2D evict bandwidth [MBytes/s]",
        y2title = "Evict Bandwidth [MBytes/s]",
        xtitle = "Time"
    },
    L3 = {
        perfgroup = "L3",
        ymetricmatch = "L3 load bandwidth [MBytes/s]",
        title = "L3 cache bandwidth",
        ytitle = "Load Bandwidth [MBytes/s]",
        y2title = "Evict Bandwidth [MBytes/s]",
        y2metricmatch = "L3 evict bandwidth [MBytes/s]",
        xtitle = "Time"
    },
    MEM = {
        perfgroup = "MEM",
        ymetricmatch = "Memory bandwidth [MBytes/s]",
        title = "Memory bandwidth",
        ytitle = "Bandwidth [MBytes/s]",
        y2title = nil,
        xtitle = "Time"
    },
    QPI = {
        perfgroup = "QPI",
        ymetricmatch = "QPI data bandwidth [MByte/s]",
        title = "QPI bandwidth",
        ytitle = "Bandwidth [MBytes/s]",
        y2title = nil,
        xtitle = "Time",
        y2metricmatch = "QPI link bandwidth [MByte/s]"
    },
    ENERGY = {
        perfgroup = "ENERGY",
        ymetricmatch = "Power [W]",
        title = "Consumed energy",
        ytitle = "Power [W]",
        y2title = "Power DRAM [W]",
        y2metricmatch = "Power DRAM [W]",
        xtitle = "Time"
    },
    TEMP = {
        perfgroup = "ENERGY",
        ymetricmatch = "Temperature [C]",
        title = "Temperature",
        ytitle = "Temperature [C]",
        y2title = nil,
        xtitle = "Time"
    },
    NUMA = {
        perfgroup = "NUMA",
        ymetricmatch = "Local DRAM bandwidth [MByte/s]",
        title = "NUMA separated memory bandwidth",
        ytitle = "Bandwidth [MBytes/s]",
        y2metricmatch = "Remote DRAM bandwidth [MByte/s]",
        y2title = nil,
        xtitle = "Time"
    },
}

local function version()
    print_stdout(string.format("likwid-perfscope -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

local function examples()
    print_stdout("Examples:")
    print_stdout("Run command on CPU 2 and measure performance group TEST:")
    print_stdout("likwid-perfscope -C 2 -g TEST -f 1s ./a.out")
end

local function usage()
    version()
    print_stdout("A tool to generate pictures on-the-fly from likwid-perfctr measurements\n")
    print_stdout("Options:")
    print_stdout("-h, --help\t\t Help message")
    print_stdout("-v, --version\t\t Version information")
    print_stdout("-V, --verbose <level>\t Verbose output, 0 (only errors), 1 (info), 2 (details), 3 (developer)")
    print_stdout("-a\t\t\t Print all preconfigured plot configurations for the current system.")
    print_stdout("-c <list>\t\t Processor ids to measure, e.g. 1,2-4,8")
    print_stdout("-C <list>\t\t Processor ids to pin threads and measure, e.g. 1,2-4,8")
    print_stdout("-g, --group <string>\t Preconfigured plot group or custom event set string with plot config. See man page for information.")
    print_stdout("-t, --time <time>\t Frequency in s, ms or us, e.g. 300ms, for the timeline mode of likwid-perfctr")
    print_stdout("-f, --force\t\t Overwrite counter configuration although already in use")
    print_stdout("-d, --dump\t\t Print output as it is send to feedGnuplot.")
    print_stdout("-p, --plotdump\t\t Use dump functionality of feedGnuplot. Plots out plot configurations plus data to directly submit to gnuplot")
    print_stdout("--host <host>\t\t Run likwid-perfctr on the selected host using SSH. Evaluation and plotting is done locally.")
    print_stdout("\t\t\t This can be used for machines that have no gnuplot installed. All paths must be similar to the local machine.")
    print_stdout("\n")
    examples()
end

local function test_gnuplot()
    cmd = "which gnuplot"
    f = io.popen(cmd)
    if f ~= nil then
        io.close(f)
        return true
    end
    return false
end

local eventStrings = {}
local terminal = "x11"
local num_cpus = 0
local cpulist = {}
local matchstring = nil
local group_list = {}
local timeline = "1s"
local print_configs = false
local pinning = false
local dump = false
local plotdump = false
local nrgroups, allgroups = likwid.get_groups()
local mfreq = 1.0
local plotrange = 0
local host = nil
local force = false
local verbose = nil

if #arg == 0 then
    usage()
    os.exit(0)
end

for opt,arg in likwid.getopt(arg, {"h","v","V:","g:","C:","c:","t:","r:","a","d","p","f","help", "version","group:","time:","dump","range:","plotdump","all", "host:", "force"}) do
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif opt == "g" or opt == "group" then
        table.insert(eventStrings, arg)
    elseif (opt == "c") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
    elseif (opt == "C") then
        num_cpus, cpulist = likwid.cpustr_to_cpulist(arg)
        pinning = true
    elseif opt == "t" or opt == "time" then
        timeline = arg
        mfreq = likwid.parse_time(timeline) * 1.E-6
    elseif opt == "d" or opt == "dump" then
        dump = true
    elseif opt == "p" or opt == "plotdump" then
        plotdump = true
    elseif opt == "r" or opt == "range" then
        plotrange = tonumber(arg)
    elseif opt == "a" or opt == "all" then
        print_configs = true
    elseif opt == "host" then
        host = arg
    elseif opt == "V" then
        local v = tonumber(arg)
        if v >= 0 and v <= 3 then
            verbose = v
        end
    elseif opt == "f" or opt == "force" then
        force = true
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end
local execList = {}
for i=1, likwid.tablelength(arg)-2 do
    if string.find(arg[i], " ") then
        table.insert(execList, "\""..arg[i].."\"")
    else
        table.insert(execList, arg[i])
    end
end

if print_configs then
    local num_groups, all_groups = likwid.get_groups()
    for name, config in pairs(predefined_plots) do
        for i,g in pairs(all_groups) do
            if g == config["perfgroup"] then
                print_stdout("Group "..name)
                print_stdout("\tPerfctr group: "..config["perfgroup"])
                print_stdout("\tMatch for metric: "..config["ymetricmatch"])
                print_stdout("\tTitle of plot: "..config["title"])
                print_stdout("\tTitle of x-axis: "..config["xtitle"])
                print_stdout("\tTitle of y-axis: "..config["ytitle"])
                if config["y2metricmatch"] then
                    print_stdout("\tMatch for second metric: "..config["y2metricmatch"])
                end
                if config["y2title"] then
                    print_stdout("\tTitle of y2-axis: "..config["y2title"])
                elseif config["y2metricmatch"] then
                    print_stdout("\tTitle of y2-axis: "..config["ytitle"])
                end
                print_stdout("")
                break
            end
        end
    end
    os.exit(0)
end

if not test_gnuplot() then
    print_stderr("GnuPlot not available")
    os.exit(1)
end

if num_cpus == 0 then
    print_stderr("ERROR: CPU string must be given")
    os.exit(1)
end

if #execList == 0 then
    print_stderr("ERROR: Executable must be given on commandline")
    os.exit(1)
end

for i, event_def in pairs(eventStrings) do
    local eventlist = likwid.stringsplit(event_def,",")

    event_string = nil
    plotgroup = nil
    plotgroupconfig = nil
    plotdefgroup = false
    for j, preconf in pairs(predefined_plots) do
        if eventlist[1] == j then
            for j,g in pairs(allgroups) do
                if g == preconf["perfgroup"] then
                    event_string = preconf["perfgroup"]
                    plotdefgroup = true
                    plotgroupconfig = preconf
                    plotgroup = j
                    break;
                end
            end
            break;
        end
    end
    if #eventlist > 1 then
        outopts = eventlist[#eventlist]
        table.remove(eventlist, #eventlist)
    end
    if event_string == nil then
        if plotdefgroup == false then
            event_string = table.concat(eventlist,",")
        end
    end

    local groupdata = nil
    groupdata = likwid.get_groupdata(event_string)
    if groupdata == nil then
        print_stderr("Cannot read event string, it's neither a performance group nor a proper event string <event>:<counter>:<options>,...")
        usage()
        os.exit(1)
    end
    if group_list[i] == nil then
        group_list[i] = {}
    end
    group_list[i]["gdata"] = groupdata

    formulalist = nil
    local title = nil
    local ytitle = nil
    local y2title = nil
    local y2funcindex = nil
    local xtitle = nil
    local output = nil
    if plotgroup ~= nil then
        title = plotgroupconfig["title"]
        ytitle = plotgroupconfig["ytitle"]
        xtitle = plotgroupconfig["xtitle"]
        if plotgroupconfig["y2title"] ~= nil then
            y2title = plotgroupconfig["y2title"]
        elseif plotgroupconfig["y2metricmatch"] ~= nil then
            y2title = plotgroupconfig["ytitle"]
        end
        for i,mconfig in pairs(groupdata["Metrics"]) do
            local mmatch = "%a*"..plotgroupconfig["ymetricmatch"]:gsub("%[","%%["):gsub("%]","%%]").."%a*"
            if mconfig["description"]:match(mmatch) then
                formulalist = {{name=mconfig["description"], index=i}}
            end
            if plotgroupconfig["y2metricmatch"] ~= nil then
                mmatch = "%a*"..plotgroupconfig["y2metricmatch"]:gsub("%[","%%["):gsub("%]","%%]").."%a*"
                if mconfig["description"]:match(mmatch) then
                    table.insert(formulalist, {name=mconfig["description"], index=i})
                end
            end
        end
    end

    --[[for j,estr in pairs(likwid.stringsplit(outopts, ":")) do
        if estr:match("^title=([%g%s]+)") then
            title = estr:match("^title=([%g%s]+)")
        elseif estr:match("^TITLE=([%g%s]+)") then
            title = estr:match("^TITLE=([%g%s]+)")
        elseif estr:match("ytitle=([%g%s]+)") then
            ytitle = estr:match("ytitle=([%g%s]+)")
        elseif estr:match("YTITLE=([%g%s]+)")then
            ytitle = estr:match("YTITLE=([%g%s]+)")
        elseif estr:match("y2title=(%d+)-([%g%s]+)") then
            y2funcindex, y2title = estr:match("y2title=(%d+)-([%g%s]+)")
        elseif estr:match("Y2TITLE=(%d+)-([%g%s]+)") then
            y2funcindex, y2title = estr:match("Y2TITLE=(%d+)-([%g%s]+)")
        elseif estr:match("y2title=([%g%s]+)") then
            y2title = estr:match("y2title=([%g%s]+)")
        elseif estr:match("Y2TITLE=([%g%s]+)") then
            y2title = estr:match("Y2TITLE=([%g%s]+)")
        elseif estr:match("xtitle=([%g%s]+)") then
            xtitle = estr:match("xtitle=([%g%s]+)")
        elseif estr:match("XTITLE=([%g%s]+)")then
            xtitle = estr:match("XTITLE=([%g%s]+)")
        elseif estr:match("[%g%s]+=[%g]+") then
            fname, form = estr:match("([%g%s]+)=([%g]+)")
            if formulalist == nil then
                formulalist = {}
            end
            if groupdata["Metrics"] ~= nil then
                for i,mconfig in pairs(groupdata["Metrics"]) do
                    if mconfig["description"]:match(fname) then
                        table.insert(formulalist, {name=fname, index=i})
                        break
                    end
                end
            else
                table.insert(formulalist, {name=fname, formula=form})
            end
        end
    end]]

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
    if y2funcindex then
        group_list[i]["y2funcindex"] = y2funcindex - 1
    else
        if formulalist ~= nil then
            group_list[i]["y2funcindex"] = #formulalist - 1
        end
    end
    if xtitle then
        group_list[i]["xtitle"] = xtitle
    end
    if formulalist ~= nil then
        group_list[i]["formulas"] = formulalist
    else
        group_list[i]["formulas"] = {}
    end
end

cmd = ""
if host ~= nil then
    cmd = cmd .. "ssh "..host.. " \"/bin/bash -c \\\" "
end
cmd = cmd .. " " ..PERFCTR
if pinning then
    cmd = cmd .. string.format(" -C %s",table.concat(cpulist,","))
else
    cmd = cmd .. string.format(" -c %s",table.concat(cpulist,","))
end
if force then
    cmd = cmd .. " -f"
end
if verbose then
    cmd = cmd .. string.format(" -V %d", verbose)
end
cmd = cmd .. string.format(" -t %s", timeline)

for i, group in pairs(group_list) do
    cmd = cmd .. " -g "..group["eventstring"]
end
cmd = cmd .. " ".. table.concat(execList, " ")
-- since io.popen can only read stdout we swap stdout and stderr
-- application output is written to stderr, we catch stdout
cmd = cmd .. " 3>&1 1>&2 2>&3 3>&-"
if host ~= nil then
    cmd = cmd .. " \\\" \" "
end



for i, group in pairs(group_list) do
    gnucmd = string.format("%s --stream %f --with linespoints --domain --nodataid", FEEDGNUPLOT, mfreq/#group_list)
    extracmds = {'set xtics font \",10\"', 'set ytics font \",10\"', 'set y2tics font \",10\"',
                 'set bmargin 10', 'set key font \",12\"' }
    gnucmd =  gnucmd .. " --geometry \"1280,1024\""
    if plotrange > 0 then
        gnucmd = gnucmd .. string.format(" --xlen %d", plotrange)
    else
        gnucmd = gnucmd .. " --xmin 0"
    end
    if group["title"] ~= nil then
        if #group_list > 1 then
            --gnucmd = gnucmd .. string.format(" --title %q", "Group "..i..": "..group["title"])
            table.insert(extracmds, string.format("set title 'Group %d: %s' font \",12\"", i, group["title"]))
        else
            --gnucmd = gnucmd .. string.format(" --title %q", group["title"])
            table.insert(extracmds, string.format("set title %q font \",12\"", group["title"]))
        end
    end
    if group["xtitle"] ~= nil then
        --gnucmd = gnucmd .. string.format(" --xlabel %q", group["xtitle"])
        table.insert(extracmds, string.format("set xlabel %q font \",12\"", group["xtitle"]))
    else
        --gnucmd = gnucmd .. string.format(" --xlabel %q", "Time")
        table.insert(extracmds, "set xlabel 'Time' font \",12\"")
    end
    if group["ytitle"] ~= nil then
        --gnucmd = gnucmd .. string.format(" --ylabel %q", group["ytitle"])
        table.insert(extracmds, string.format("set ylabel %q font \",12\"", group["ytitle"]))
    end
    if group["y2title"] ~= nil then
        for c,cpu in pairs(cpulist) do
            gnucmd = gnucmd .. string.format(" --y2 %d", (c-1) + (#cpulist * group["y2funcindex"]))
        end
        table.insert(extracmds, string.format("set y2label %q font \",12\"", group["y2title"]))
    end
    if group["formulas"] then
        if #cpulist == 1 then
            for f, fdesc in pairs(group["formulas"]) do
                gnucmd = gnucmd .. string.format(" --legend %d %q", f-1, fdesc["name"])
            end
        else
            local curveID = 0
            for f, fdesc in pairs(group["formulas"]) do
                for c,cpu in pairs(cpulist) do
                    gnucmd = gnucmd .. string.format(" --legend %d %q", curveID, "C"..cpu..": "..fdesc["name"])
                    curveID = curveID + 1
                end
            end
        end
    end
    gnucmd = gnucmd .. " --set 'key outside bmargin bottom'"
    gnucmd = gnucmd .. string.format(" --extracmds '%s'", table.concat(extracmds, ";"))
    if plotdump then
        gnucmd = gnucmd .. " --dump"
    else
        gnucmd = gnucmd .. " 1>/dev/null 2>&1"
    end
    group_list[i]["output"] = assert(io.popen(gnucmd,"w"))
end


likwid.catchSignal()
local mtime = {}
for i,g in pairs(group_list) do
    local str = "0 "
    for k,t in pairs(cpulist) do
        for j,c in pairs(g["formulas"]) do
            str = str .."0 "
        end
    end
    mtime[i] = nil
    g["output"]:write(str.."\n")
    g["output"]:flush()
    if dump then
        print_stdout(tostring(i).." ".. str)
    end
end

perfctr = assert (io.popen (cmd))
olddata = {}
oldmetric = {}
local perfctr_exited = false
local oldtime = 0
local clock = likwid.getCpuClock()
while true do
    local l = perfctr:read("*line")
    if l == nil then
        break
    end
    if l:match("^%d+ %d+ %d+ [%d.]+ %d+") then
        local data = {}
        local diff = {}
        linelist = likwid.stringsplit(l, " ")
        group = tonumber(linelist[1])
        nr_events = tonumber(linelist[2])
        nr_threads = tonumber(linelist[3])
        time = tonumber(linelist[4])
        table.remove(linelist, 1)
        table.remove(linelist, 1)
        table.remove(linelist, 1)
        table.remove(linelist, 1)

        for i=1,nr_events do
            if data[i] == nil then data[i] = {} end
            for j=1,nr_threads do
                data[i][j] = tonumber(linelist[1])
                table.remove(linelist, 1)
            end
        end

        str = tostring(time)
        for f, flist in pairs(group_list[group]["formulas"]) do
            if flist["index"] ~= nil then
                for i=1,nr_threads do
                    str = str .." ".. data[flist["index"]][i]
                end
            end
        end

        group_list[group]["output"]:write(str.."\n")
        group_list[group]["output"]:flush()
        if dump then
            print_stdout(tostring(group).." ".. str)
        end
        oldtime = time
    end
end

if perfctr_exited == false then
    while likwid.getSignalState() == 0 do
        likwid.sleep(1E6)
    end
end
for i, group in pairs(group_list) do
    group["output"]:write("exit\n")
    io.close(group["output"])
end
io.close(perfctr)



