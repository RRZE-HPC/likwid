--------------------------------------------------------------------------------
Examples
--------------------------------------------------------------------------------

This folder contains some examples showcasing the usage of LIKWID, either by
using the full LIKWID API or the MarkerAPI used for instrumenting user codes.

All compilable examples are also listed with `make help`. The command `make`
without target tries to build and run all compilable examples. It reuses the settings
of the LIKWID build from `../config.mk`.

- `C-markerAPI`: LIKWID's MarkerAPI in C/C++ applications
- `F-markerAPI`: LIKWID's MarkerAPI in Fortran applications
- `C-nvMarkerAPI.c`: LIKWID's Nvidia GPU related NvMarkerAPI in C/C++ applications
- `C-likwidAPI`: Using the full LIKWID API in C/C++ applications
- `Lua-likwidAPI`: Use the full LIKWID API in Lua
- `monitoring`: Showcase how to use LIKWID for system-wide monitoring
- `likwid-benchmark.sh`: Legacy version of [MachineState](https://github.com/RRZE-HPC/MachineState). Please use MachineState if you need something like this.

All examples can be built directly by `make <target>` (e.g. `make C-markerAPI`).

In order to run them, for each examples exists a `run` target: `make <target>-run`


