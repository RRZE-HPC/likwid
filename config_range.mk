CUDA_HOME ?= /usr/local/cuda

# CUPTI headers + CUPTI sample helper headers
CUPTI_INC := $(CUDA_HOME)/extras/CUPTI/include
CUPTI_SAMPLES_COMMON := $(CUDA_HOME)/extras/CUPTI/samples/common
CUPTI_SAMPLES_RANGE  := $(CUDA_HOME)/extras/CUPTI/samples/range_profiling

CXX ?= g++
CXXFLAGS += -fPIC -O2 -std=c++17 -I$(CUPTI_INC) -I$(CUPTI_SAMPLES_COMMON) -I$(CUPTI_SAMPLES_RANGE)
LDFLAGS  += -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64
LDLIBS   += -lcupti -lcuda

LIBS += -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -L$(CUDA_HOME)/lib64 -libcuda

CUDA_LIBDIR  ?= $(CUDA_HOME)/lib64
CUPTI_LIBDIR ?= $(CUDA_HOME)/extras/CUPTI/lib64

LIBS += -L$(CUDA_LIBDIR)  -lcuda -lcudart
LIBS += -L$(CUPTI_LIBDIR) -lcupti
LIBS += -ldl -lpthread

LDFLAGS += -Wl,-rpath,$(CUDA_LIBDIR) -Wl,-rpath,$(CUPTI_LIBDIR)
