LIKWID_PATH=../..
LIKWID_BENCH_PATH=../../bench
LIKWID_APP=likwid-bench
HOST=$(shell hostname -s)


all: clean plain marker localize_likwid

plain:
	@echo "===> Building uninstrumented likwid-bench"
	@sed -i -e s/'INSTRUMENT_BENCH = .*#'/'INSTRUMENT_BENCH = false#'/g $(LIKWID_PATH)/config.mk
	@sed -i -e s/'CPPFLAGS := -DPAPI '/'CPPFLAGS := '/g $(LIKWID_PATH)/Makefile
	@cd $(LIKWID_PATH) && make distclean >/dev/null && make >/dev/null 2>/dev/null
	@cp $(LIKWID_BENCH_PATH)/$(LIKWID_APP) $(LIKWID_APP)-plain

marker:
	@echo "===> Building instrumented likwid-bench"
	@sed -i -e s/'INSTRUMENT_BENCH = .*#'/'INSTRUMENT_BENCH = true#'/g $(LIKWID_PATH)/config.mk
	@sed -i -e s/'CPPFLAGS := -DPAPI '/'CPPFLAGS := '/g $(LIKWID_PATH)/Makefile
	@cd $(LIKWID_PATH) && make distclean >/dev/null && make >/dev/null 2>/dev/null
	@cp $(LIKWID_BENCH_PATH)/$(LIKWID_APP) $(LIKWID_APP)-marker

papi:
	@echo "===> Building instrumented likwid-bench using PAPI API"
	@sed -i -e s/'INSTRUMENT_BENCH = .*#'/'INSTRUMENT_BENCH = false#'/g $(LIKWID_PATH)/config.mk
	@cp $(LIKWID_PATH)/Makefile $(LIKWID_PATH)/Makefile.orig
	@sed -i -e s/'CPPFLAGS := '/'CPPFLAGS := -DPAPI '/g $(LIKWID_PATH)/Makefile
	@cd $(LIKWID_PATH) && make distclean >/dev/null && make >/dev/null 2>/dev/null
	@cp $(LIKWID_BENCH_PATH)/$(LIKWID_APP) $(LIKWID_APP)-papi
	@mv $(LIKWID_PATH)/Makefile.orig $(LIKWID_PATH)/Makefile

localize_likwid:
	@cd $(LIKWID_PATH) && make local >/dev/null && cd - >/dev/null

clean:
	@echo "===> Cleaning old likwid-bench executables"
	@rm -f $(LIKWID_APP)-plain $(LIKWID_APP)-marker $(LIKWID_APP)-papi
