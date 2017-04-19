obj-m += enable_arm_pmu.o
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
CC ?= gcc

all:
	$(MAKE) -Wall -C $(KERNELDIR) M=$(PWD) modules

modules_install:
	install -m 666 enable_arm_pmu.ko /lib/modules/$(shell uname -r)/extra/

clean:
	rm -f *.ko *.o modules.order Module.symvers enable_arm_pmu.mod.c .enable_arm_pmu*.cmd
	rm -rf .tmp_versions
