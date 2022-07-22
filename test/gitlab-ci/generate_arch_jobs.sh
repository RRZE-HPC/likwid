#!/bin/bash -l

for L in $(sinfo -t idle -h --partition=work -o "%n %t" | grep "idle" | cut -d ' ' -f 1); do
    arch="x86"
    depend="build-x86-perf"
    if [ "$L" = "aurora1" ]; then
        continue
    fi
    if [ "$L" = "warmup" ]; then
        arch="arm8"
        depend="build-arm8-perf"
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
  before_script:
    - cp -r $depend /tmp/$depend
    - cd /tmp/$depend
    - export PATH=/tmp/$depend/bin:\$PATH
    - export LD_LIBRARY_PATH=/tmp/$depend/lib:\$LD_LIBRARY_PATH
  script:
    - likwid-topology
    - likwid-pin -p
    - likwid-perfctr -i
  after_script:
    - rm -rf /tmp/$depend
EOF

    if [ "$arch" == "x86" ]; then
        depend="build-x86-daemon"
        if [ "$L" = "aurora1" ]; then
            continue
        fi
        if [ "$L" = "milan1" ]; then
            continue
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
  before_script:
    - cp -r $depend /tmp/$depend
    - cd /tmp/$depend
    - export PATH=/tmp/$depend/bin:\$PATH
    - export LD_LIBRARY_PATH=/tmp/$depend/lib:\$LD_LIBRARY_PATH
  script:
    - likwid-topology
    - likwid-pin -p
    - likwid-perfctr -i
  after_script:
    - rm -rf /tmp/$depend
EOF
    fi
    echo
done
