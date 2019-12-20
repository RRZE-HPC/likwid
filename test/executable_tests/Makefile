

all: topology pin markerAPI perfctr memsweeper powermeter bench genTopoCfg setFrequencies mpirun

topology:
	./tester.sh likwid-topology
pin:
	./tester.sh likwid-pin
perfctr: markerAPI
	./tester.sh likwid-perfctr
memsweeper:
	./tester.sh likwid-memsweeper
powermeter:
	./tester.sh likwid-powermeter
bench:
	./tester.sh likwid-bench
genTopoCfg:
	./tester.sh likwid-genTopoCfg
setFrequencies:
	./tester.sh likwid-setFrequencies
mpirun: mpipin markerAPI
	./tester.sh likwid-mpirun
markerAPI:
	make -s -C .. streamGCC
mpipin:
	make -s -C .. test-mpi
