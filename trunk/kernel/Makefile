obj-m := enable_rdpmc.o
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -Wpacked -C $(KERNELDIR) M=$(PWD) modules
	chmod 666 enable_rdpmc.ko

clean:
	rm -f *.ko *.o modules.order Module.symvers enable_rdpmc.mod.c
