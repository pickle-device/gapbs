CXX=g++

SERIAL?=1

CXX_FLAGS += -std=c++17 -O3 -g -Wall -no-pie -Wl,-E -rdynamic

PAR_FLAG = -fopenmp
LIBDIR = 
LDFLAGS =  -no-pie -Wl,-E

OUTPUT_SUFFIX=.hw
GEM5_SUFFIX=.m5
PICKLEDEVICE_SUFFIX=.pdev
SERIAL_SUFFIX=.ser

#ifeq ($(ENABLE_PICKLEDEVICE), 1)
CXX_FLAGS += -lpickledevice -Wfatal-errors
LDFLAGS   += -lpickledevice
ifeq ($(ENABLE_PICKLEDEVICE), 1)
  CXX_FLAGS += -DENABLE_PICKLEDEVICE=1
  LDFLAGS   += -DENABLE_PICKLEDEVICE=1
  TMP_OUTPUT_SUFFIX = $(OUTPUT_SUFFIX)
  OUTPUT_SUFFIX := $(TMP_OUTPUT_SUFFIX)$(PICKLEDEVICE_SUFFIX)
endif

ifeq ($(CHECK_NUM_EDGES), 1)
  CXX_FLAGS += -DCHECK_NUM_EDGES=1
  LDFLAGS   += -DCHECK_NUM_EDGES=1
endif

M5OPS_HEADER_PATH=/home/ubuntu/gem5/include
M5_BUILD_PATH=/home/ubuntu/gem5/util/m5/build/arm64/

ifeq ($(ENABLE_GEM5), 1)
  CXX_FLAGS += -I$(M5OPS_HEADER_PATH) -I$(M5OPS_HEADER_PATH)/../util/m5/src
  CXX_FLAGS += -DENABLE_GEM5=1
  LDFLAGS   += -lm5 -L$(M5_BUILD_PATH)/out/
  LDFLAGS   += -DENABLE_GEM5=1
  TMP_OUTPUT_SUFFIX = $(OUTPUT_SUFFIX)
  OUTPUT_SUFFIX := $(TMP_OUTPUT_SUFFIX)$(GEM5_SUFFIX)
endif

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif
ifeq ($(SERIAL), 1)
	OUTPUT_SUFFIX := $(TMP_OUTPUT_SUFFIX)$(SERIAL_SUFFIX)
endif

#KERNELS = bc bfs bfs_td cc cc_sv pr pr_spmv sssp tc
KERNELS = mcache mthread bfs bfs2 bfs_td cc pr
SUITE = $(KERNELS) converter

.PHONY: all
#all: $(SUITE); $(info $$CXX_FLAGS is [${CXX_FLAGS}])
all: $(SUITE); $(info $$CXX is [${CXX}])
	

% : src/%.cc src/*.h
	$(CXX) $(CXX_FLAGS) $< -o $@$(OUTPUT_SUFFIX) $(LDFLAGS) $(LDLIBS)

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/* *.hw *$(GEM5_SUFFIX) *$(PICKLEDEVICE_SUFFIX) *$(SERIAL_SUFFIX)
