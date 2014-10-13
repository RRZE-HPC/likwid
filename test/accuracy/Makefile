LIKWID_PATH=../..
LIKWID_APP=likwid-bench
HOST=$(shell hostname -s)


all: plain marker

plain:
	sed -i -e s/'INSTRUMENT_BENCH = .*#'/'INSTRUMENT_BENCH = false#'/g $(LIKWID_PATH)/config.mk
	sed -i -e s/'CPPFLAGS := -DPAPI '/'CPPFLAGS := '/g $(LIKWID_PATH)/Makefile
	cd $(LIKWID_PATH) && make distclean && make
	cp $(LIKWID_PATH)/$(LIKWID_APP) $(LIKWID_APP)-plain

marker:
	sed -i -e s/'INSTRUMENT_BENCH = .*#'/'INSTRUMENT_BENCH = true#'/g $(LIKWID_PATH)/config.mk
	sed -i -e s/'CPPFLAGS := -DPAPI '/'CPPFLAGS := '/g $(LIKWID_PATH)/Makefile
	cd $(LIKWID_PATH) && make distclean && make
	cp $(LIKWID_PATH)/$(LIKWID_APP) $(LIKWID_APP)-marker
papi:
	sed -i -e s/'INSTRUMENT_BENCH = .*#'/'INSTRUMENT_BENCH = false#'/g $(LIKWID_PATH)/config.mk
	cp $(LIKWID_PATH)/Makefile $(LIKWID_PATH)/Makefile.orig
	sed -i -e s/'CPPFLAGS := '/'CPPFLAGS := -DPAPI '/g $(LIKWID_PATH)/Makefile
	cd $(LIKWID_PATH) && make distclean && make
	cp $(LIKWID_PATH)/$(LIKWID_APP) $(LIKWID_APP)-papi
	mv $(LIKWID_PATH)/Makefile.orig $(LIKWID_PATH)/Makefile
