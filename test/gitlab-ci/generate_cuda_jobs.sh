#!/bin/bash -l

for VER in $(module avail -t cuda 2>&1 | grep -E "^cuda" | cut -d ' ' -f 1); do
    PVER=${VER/\//-}
cat <<EOF
build-$PVER:
    stage: build
    variables:
      LIKWID_COMPILER: "GCC"
      LIKWID_ACCESSMODE: "perf_event"
      SLURM_NODELIST: medusa
      CUDA_MODULE: $VER
    script:
      - module load "\$CUDA_MODULE"
      - echo "\$CUDA_HOME"
      - ls "\$CUDA_HOME/lib64"
      - ls "\$CUDA_HOME/extras/CUPTI/lib64"
      - sed -e s+"ACCESSMODE = .*"+"ACCESSMODE=\$LIKWID_ACCESSMODE"+g -i config.mk
      - sed -e s+"COMPILER = .*"+"COMPILER=\$LIKWID_COMPILER"+g -i config.mk
      - sed -e s+"NVIDIA_INTERFACE = .*"+"NVIDIA_INTERFACE=true"+g -i config.mk
      - sed -e s+"BUILDDAEMON = .*"+"BUILDDAEMON=false"+g -i config.mk
      - sed -e s+"BUILDFREQ = .*"+"BUILDFREQ=false"+g -i config.mk
      - make
      - make local
      - if [ -e likwid-accessD ]; then rm likwid-accessD; fi
      - if [ -e likwid-setFreq ]; then rm likwid-setFreq; fi
      - export LD_LIBRARY_PATH=\$(pwd):\$LD_LIBRARY_PATH
      - ./likwid-topology
    tags:
      - testcluster

EOF

done


