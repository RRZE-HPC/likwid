#!<PREFIX>/bin/likwid-lua

--[[
 * =======================================================================================
 *
 *      Filename:  likwid-memsweeper.lua
 *
 *      Description:  An application to clean up NUMA memory domains.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Thomas Roehl
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
 * =======================================================================================]]
package.path = package.path .. ';<PREFIX>/share/lua/?.lua'
local likwid = require("likwid")

local function version()
    print(string.format("likwid-memsweeper --  Version %d.%d",likwid.version,likwid.release))
end

local function examples()
    print("Examples:")
    print("To clean specific domain:")
    print("likwid-memsweeper.lua -c 2")
    print("To clean a range of domains:")
    print("likwid-memsweeper.lua -c 1-2")
    print("To clean specific domains:")
    print("likwid-memsweeper.lua -c 0,1-2")

end

local function usage()
    version()
    print("A tool clean up NUMA memory domains.\n")
    print("Options:")
    print("-h\t\t Help message")
    print("-v\t\t Version information")
    print("-c <list>\t Specify NUMA domain ID to clean up")
    print("\n")
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
    else
        print("Unknown option found on commandline")
        usage()
        os.exit(1)
    end
end

for i,socket in pairs(nodes) do
    likwid.memSweepDomain(socket)
end
likwid.putNumaInfo()
