---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
* LIKWID command and/or API usage
* LIKWID version and download source (Github, FTP, package manger, ...)
* Operating system
* Does your application use libraries like MPI, OpenMP or Pthreads?
* In case of Nvidia GPUs, which CUDA version?
* Are you using the MarkerAPI (CPU code instrumentation) or the NvMarkerAPI (Nvidia GPU code instrumentation)?

**To Reproduce with a LIKWID command**
Please supply the output of the command with `-V 3` added to the command:
* likwid-topology
* likwid-pin
* likwid-perfctr
* likwid-setFrequencies
* likwid-powermeter
* likwid-perscope

Please supply the output of the command with `-d` added to the command line:
* likwid-mpirun

**Additional context**
Add any other context about the problem here.
