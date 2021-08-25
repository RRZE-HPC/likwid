# Gitlab CI at RRZE and NHR@FAU
The whole Github repository is synced with the [Gitlab services of RRZE](https://www.rrze.fau.de/serverdienste/infrastruktur/gitlab/) but no changes are done in the repository. For testing LIKWID builds at pull requests and before releases (new tags), some jobs are executed using the [Gitlab CI runner at NHR@FAU](https://hpc.fau.de/systems-services/systems-documentation-instructions/special-applications-and-tips-tricks/continuous-integration). The runner uses the [NHR@FAU Testcluster](https://hpc.fau.de/systems-services/systems-documentation-instructions/clusters/test-cluster/).

The main configuration is in the `.gitlab-ci.yml` file in the repository root. It uses some scripts to generate child pipelines. These scripts are located in `test/gitlab-ci/`.

# Simple jobs
The configuration uses the `.pre` phase to run some basic pre-build tests like checking the syntax of the event lists and the performance group files.

# Build jobs
At the moment, the NHR@FAU Testcluster contains only x86_64 and ARMv8 nodes. The configuration contains build jobs for the access modes `accessdaemon` and `perf_event` for x86_64 and `perf_event` for ARMv8. The `direct` access mode for x86_64 cannot be tested due to missing sudo permissions. With the same reason, the access daemon is not installed but the installations use a pre-installed daemon with proper permissions.

The builds are stored as artifacts and reused in the Architecture pipeline.

# Child pipelines
In order to avoid manually managing a list of available nodes and modules in the Testcluster, the current configuration contains two generator jobs for architecture-specific and Nvidia Cuda specific tests.

# Architecture pipeline
For each node in the Testcluster, the jobs are generated depending on the architecture and available access modes.

# CUDA pipeline
LIKWID supports the deprecated CUPTI Events API and the current CUpti Profiling API. Since the APIs on the Nvidia side change, a set of jobs is generated based on the availibility.

