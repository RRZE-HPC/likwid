#!<INSTALLED_BINPREFIX>/likwid-lua
--[[
 * =======================================================================================
 *
 *      Filename:  likwid-memsweeper.lua
 *
 *      Description:  An application to clean up NUMA memory domains.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
    print_stdout(string.format("likwid-memsweeper -- Version %d.%d.%d (commit: %s)",likwid.version,likwid.release,likwid.minor,likwid.commit))
end

local function examples()
    print_stdout("Examples:")
    print_stdout("To clean specific domain:")
    print_stdout("likwid-memsweeper -c 2")
    print_stdout("To clean a range of domains:")
    print_stdout("likwid-memsweeper -c 1-2")
    print_stdout("To clean specific domains:")
    print_stdout("likwid-memsweeper -c 0,1-2")

end

local function usage()
    version()
    print_stdout("A tool clean up NUMA memory domains.\n")
    print_stdout("Options:")
    print_stdout("-h\t\t Help message")
    print_stdout("-v\t\t Version information")
    print_stdout("-c <list>\t Specify NUMA domain ID to clean up")
    print_stdout("")
    examples()
end

numainfo = likwid.getNumaInfo()
nodes = {}
for i,_ in pairs(numainfo["nodes"]) do
    if tonumber(numainfo["nodes"][i]["id"]) ~= nil then
        table.insert(nodes,numainfo["nodes"][i]["id"])
    end
end

for opt,arg in likwid.getopt(arg, {"c:", "h", "v", "help", "version"}) do
    if opt == "h" or opt == "help" then
        usage()
        os.exit(0)
    elseif opt == "v" or opt == "version" then
        version()
        os.exit(0)
    elseif (opt == "c") then
        num_nodes, nodes = likwid.nodestr_to_nodelist(arg)
    elseif opt == "?" then
        print_stderr("Invalid commandline option -"..arg)
        os.exit(1)
    elseif opt == "!" then
        print_stderr("Option requires an argument")
        os.exit(1)
    end
end

for i,socket in pairs(nodes) do
    likwid.memSweepDomain(socket)
end
likwid.putNumaInfo()
