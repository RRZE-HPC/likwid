obj-m := enable_rdpmc.o
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -Wpacked -C $(KERNELDIR) M=$(PWD) modules

modules_install:
	install -m 666 enable_rdpmc.ko /lib/modules/$(shell uname -r)/extra/

clean:
	rm -f *.ko *.o modules.order Module.symvers enable_rdpmc.mod.c .enable_rdpmc*.cmd
	rm -rf .tmp_versions
