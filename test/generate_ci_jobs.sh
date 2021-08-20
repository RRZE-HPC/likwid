#!/bin/bash -l

cat <<EOF
build-x86-daemon:
    stage: build
    variables:
      LIKWID_COMPILER: "GCC"
      LIKWID_ACCESSMODE: "accessdaemon"
      LIKWID_MODULE: "likwid/5.2-dev"
    script:
      - mkdir likwid-x86-daemon
      - module load "$LIKWID_MODULE"
      - export LIKWID_PREFIX=\$(realpath \$(dirname \$(which likwid-topology))/..)
      - sed -e s+"ACCESSMODE = .*"+"ACCESSMODE=\$(echo "$LIKWID_ACCESSMODE")"+g -i config.mk
      - sed -e s+"COMPILER = .*"+"COMPILER=\$(echo "$LIKWID_COMPILER")"+g -i config.mk
      - sed -e s+"PREFIX ?= .*"+"PREFIX=\$(pwd)/likwid-x86-daemon"+g -i config.mk
      - sed -e s+"INSTRUMENT_BENCH = .*"+"INSTRUMENT_BENCH=true"+g -i config.mk
      - sed -e s+"INSTALLED_ACCESSDAEMON = .*"+"INSTALLED_ACCESSDAEMON=\$(echo "$LIKWID_PREFIX")/sbin/likwid-accessD"+g -i config.mk
      - sed -e s+"BUILDDAEMON = .*"+"BUILDDAEMON=false"+g -i config.mk
      - sed -e s+"BUILDFREQ = .*"+"BUILDFREQ=false"+g -i config.mk
      - make
      - make BUILDDAEMON=false BUILDFREQ=false install
    artifacts:
      paths:
        - likwid-x86-daemon/
    tags:
      - testcluster

build-x86-perf:
    stage: build
    variables:
      LIKWID_COMPILER: "GCC"
      LIKWID_ACCESSMODE: "perf_event"
      SLURM_NODELIST: broadep2
    script:
      - mkdir likwid-x86-perf
      - sed -e s+"ACCESSMODE = .*"+"ACCESSMODE=\$(echo "$LIKWID_ACCESSMODE")"+g -i config.mk
      - sed -e s+"COMPILER = .*"+"COMPILER=\$(echo "$LIKWID_COMPILER")"+g -i config.mk
      - sed -e s+"PREFIX ?= .*"+"PREFIX=\$(pwd)/likwid-x86-perf"+g -i config.mk
      - sed -e s+"INSTRUMENT_BENCH = .*"+"INSTRUMENT_BENCH=true"+g -i config.mk
      - sed -e s+"BUILDDAEMON = .*"+"BUILDDAEMON=false"+g -i config.mk
      - sed -e s+"BUILDFREQ = .*"+"BUILDFREQ=false"+g -i config.mk
      - make
      - make BUILDDAEMON=false BUILDFREQ=false install
    artifacts:
      paths:
        - likwid-x86-perf/
    tags:
      - testcluster

build-arm8:
    stage: build
    variables:
      LIKWID_COMPILER: "GCCARMv8"
      LIKWID_ACCESSMODE: "perf_event"
      SLURM_NODELIST: warmup
    script:
      - mkdir likwid-arm8-perf
      - sed -e s+"ACCESSMODE = .*"+"ACCESSMODE=\$(echo "$LIKWID_ACCESSMODE")"+g -i config.mk
      - sed -e s+"COMPILER = .*"+"COMPILER=\$(echo "$LIKWID_COMPILER")"+g -i config.mk
      - sed -e s+"PREFIX ?= .*"+"PREFIX=\$(pwd)/likwid-arm8-perf"+g -i config.mk
      - sed -e s+"INSTRUMENT_BENCH = .*"+"INSTRUMENT_BENCH=true"+g -i config.mk
      - sed -e s+"BUILDDAEMON = .*"+"BUILDDAEMON=false"+g -i config.mk
      - sed -e s+"BUILDFREQ = .*"+"BUILDFREQ=false"+g -i config.mk
      - make
      - make BUILDDAEMON=false BUILDFREQ=false install
    artifacts:
      paths:
        - likwid-arm8-perf/
    tags:
      - testcluster
EOF

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
