#!/bin/bash -l

for L in $(sinfo -t idle -h --partition=work -o "%n"); do
    arch="x86"
    depend="build-x86-perf"
    if [ "$L" = "warmup" ]; then
        arch="arm8"
        depend="build-arm8"
    fi
    if [ "$L" = "medusa" ]; then
        depend="build-x86-perf-nv"
    fi

    cat <<EOF
test-$L-perf:
  stage: test
  variables:
    SLURM_NODELIST: $L
    SLURM_CONSTRAINT: hwperf
  needs:
    pipeline: \$PARENT_PIPELINE_ID
    job: $depend
  tags:
    - testcluster
  script:
    - export PATH=\$CI_PROJECT_DIR/$depend/bin:\$PATH
    - export LD_LIBRARY_PATH=\$CI_PROJECT_DIR/$depend/lib:\$LD_LIBRARY_PATH
    - likwid-topology
    - likwid-pin -p
    - likwid-perfctr -i
    - likwid-powermeter -i
EOF

    if [ "$arch" == "x86" ]; then
        depend="build-x86-daemon"
        if [ "$L" = "medusa" ]; then
            depend="build-x86-daemon-nv"
        fi
        echo
    cat <<EOF
test-$L-daemon:
  stage: test
  variables:
    SLURM_NODELIST: $L
    SLURM_CONSTRAINT: hwperf
  needs:
    pipeline: \$PARENT_PIPELINE_ID
    job: $depend
  tags:
    - testcluster
  script:
    - export PATH=\$CI_PROJECT_DIR/$depend/bin:\$PATH
    - export LD_LIBRARY_PATH=\$CI_PROJECT_DIR/$depend/lib:\$LD_LIBRARY_PATH
    - likwid-topology
    - likwid-pin -p
    - likwid-perfctr -i
    - likwid-powermeter -i
EOF
    fi
    echo
done
