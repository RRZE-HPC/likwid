#!<PREFIX>/bin/likwid-lua
--[[
 * =======================================================================================

 *
 *      Filename:  Lua-likwidAPI.lua
 *
 *      Description:  Example how to use the LIKWID API in Lua scripts
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

package.path = package.path .. ';<PREFIX>/share/lua/?.lua'

local likwid = require("likwid")

EVENTSET = "INSTR_RETIRED_ANY:FIXC0"

cpuinfo = likwid.getCpuInfo()
cputopo = likwid.getCpuTopology()

print(string.format("Likwid example on a %s with %d CPUs", cpuinfo.name, cputopo.numHWThreads))

local cpus = {}
for i, cpu in pairs(cputopo.threadPool) do
    table.insert(cpus, cpu.apicId)
end

if likwid.init(#cpus, cpus) ~= 0 then
    print("Failed to initialize LIKWID's performance monitoring module")
    likwid.putTopology()
    os.exit(1)
end

local gid = likwid.addEventSet(EVENTSET)
if gid <= 0 then
    print(string.format("Failed to add events %s to LIKWID's performance monitoring module", EVENTSET))
    likwid.finalize()
    likwid.putTopology()
    os.exit(1)
end


if likwid.setupCounters(gid) < 0 then
    printf(string.format("Failed to setup group %d in LIKWID's performance monitoring module\n", gid))
    likwid.finalize()
    likwid.putTopology()
    os.exit(1)
end
if likwid.startCounters() < 0 then
    printf(string.format("Failed to start group %d in LIKWID's performance monitoring module\n", gid))
    likwid.finalize()
    likwid.putTopology()
    os.exit(1)
end


-- Application code
likwid.sleep(2)


if likwid.stopCounters() < 0 then
    printf(string.format("Failed to stop group %d in LIKWID's performance monitoring module\n", gid))
    likwid.finalize()
    likwid.putTopology()
    os.exit(1)
end


for i,cpu in pairs(cpus) do
    result = likwid.getResult(gid, 1, i)
    print(string.format("Measurement result for event set %s at CPU %d: %f", EVENTSET, cpu, result))
end


likwid.finalize()
