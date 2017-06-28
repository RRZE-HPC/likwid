obj-m += enable_arm_pmu.o
KERNELDIR=/lib/modules/$(shell uname -r)/build
INSTALLDIR=/lib/modules/$(shell uname -r)/extra

all:
	make -C $(KERNELDIR) M=$(PWD) modules

modules_install:
	install -m 666 enable_arm_pmu.ko $(INSTALLDIR)

clean:
	make -C $(KERNELDIR) M=$(PWD) clean
