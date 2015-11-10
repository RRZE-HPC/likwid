

all: topology pin markerAPI perfctr memsweeper powermeter bench genTopoCfg setFrequencies

topology:
	./tester.sh likwid-topology
pin:
	./tester.sh likwid-pin
perfctr:
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
markerAPI:
	make -s -C .. streamGCC
