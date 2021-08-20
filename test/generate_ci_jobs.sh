#!/bin/bash -l

for L in $(sinfo -t idle -h --partition=work -o "%n"); do
    arch="x86"
    depend="build-x86-perf"
    if [ "$L" = "warmup" ]; then
        arch="arm8"
        depend="build-arm8"
    fi

    cat <<EOF
test-$L-perf:
  stage: test
  variables:
    SLURM_NODELIST: $L
    SLURM_CONSTRAINT: hwperf
  dependencies:
    - $depend
  tags:
    - testcluster
  script:
    - export PATH=\$CI_PROJECT_DIR/likwid-$arch-perf/bin:\$PATH
    - export LD_LIBRARY_PATH=\$CI_PROJECT_DIR/likwid-$arch-perf/lib:\$LD_LIBRARY_PATH
EOF

    if [ "$arch" == "x86" ]; then
        echo
    cat <<EOF
test-$L-daemon:
  stage: test
  variables:
    SLURM_NODELIST: $L
    SLURM_CONSTRAINT: hwperf
  dependencies:
    - build-x86-daemon
  tags:
    - testcluster
  script:
    - export PATH=\$CI_PROJECT_DIR/likwid-$arch-daemon/bin:\$PATH
    - export LD_LIBRARY_PATH=\$CI_PROJECT_DIR/likwid-$arch-daemon/lib:\$LD_LIBRARY_PATH
EOF
    fi
    echo
done
