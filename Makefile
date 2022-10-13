# /*
# * Copyright (c) 2021-2022, Virginia Tech.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *

ifeq ($(DEBUG), 1)
  OPTS=-g -DDEBUG -O0
else ifeq ($(DEBUG), 2)
  OPTS=-g -DDEBUG -O0 -DDEBUG_2
else
  OPTS=-O3
endif

CFLAGS := $(CFLAGS) -DDISABLE_DP_WEIGHT -DDISABLE_LIST -DDISABLE_DP_INDEX --std=c++17 $(OPTS)
LDFLAGS := $(LDFLAGS) --std=c++17
ifneq ($(NO_LIBSTDFS),1)
  SYCL_LD_FLAGS :=  $(SYCL_LD_FLAGS) -lstdc++fs
endif
SYCL_C_FLAGS := $(SYCL_C_FLAGS) $(CFLAGS)
SYCL_LD_FLAGS := $(SYCL_LD_FLAGS) $(LDFLAGS)
ifeq ($(COMPILER), HIPSYCL)
  ifeq ($(HIPSYCL_PATH),)
    HIPSYCL_PATH=/opt/hipSYCL
  endif
  ifeq ($(HIPSYCL_CLANG_PATH),)
    HIPSYCL_CLANG_PATH=$(HIPSYCL_PATH)/llvm
  endif
  ifeq ($(ROCPROFILER),1)
#  ROCPROFILER_INCL= -I /opt/rocm/include -I /opt/rocm/include/hsa
#  ROCPROFILER_LIB= -L /opt/rocm/lib -lrocprofiler64
    ROCPROFILER_C_FLAGS=-D ROCPROFILE
    ROCPROFILER_LD_FLAGS=-lrocprofiler64 -ldl
  endif
  ifeq ($(HIPSYCL_TARGETS),)
    HIPSYCL_TARGETS="omp"
  endif
  SYCL=$(HIPSYCL_PATH)/bin/syclcc
  ifneq ($(ROCM_PATH),)
    SYCL := $(SYCL) -isystem $(ROCM_PATH)/include
  endif
  ifneq ($(ROCM_DEVICE_LIB_PATH),)
    SYCL := $(SYCL) --rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH)
  endif
  ifneq ($(CUDA_PATH),)
    SYCL := $(SYCL) --cuda-path=$(CUDA_PATH)
  endif
  #SYCL_FLAGS=-isystem $(HIPSYCL_PATH) --hipsycl-targets="omp;hip:gfx900" -Wl,-rpath=$(HIPSYCL_PATH)/lib --hipsycl-explicit-multipass
  SYCL_C_FLAGS := $(SYCL_C_FLAGS) -isystem $(HIPSYCL_PATH) --hipsycl-targets=$(HIPSYCL_TARGETS) $(ROCPROFILER_C_FLAGS) -D HIPSYCL
  SYCL_LD_FLAGS := $(SYCL_LD_FLAGS) --hipsycl-targets=$(HIPSYCL_TARGETS) -Wl,-rpath=$(HIPSYCL_PATH)/lib,-rpath=$(HIPSYCL_CLANG_PATH)/lib $(OPTS) $(ROCPROFILER_LD_FLAGS) -fuse-ld=lld
endif
ifeq ($(COMPILER), ICX) #DPCPP in the HPC toolkit
  ONEAPI_PATH=/opt/intel/oneapi/compiler/2021.2.0/linux
#  LD_LIBRARY_PATH:=$(ONEAPI_PATH)/compiler/lib/intel64_lin:$(LD_LIBRARY_PATH)
  SYCL=$(ONEAPI_PATH)/bin/icx
  SYCL_C_FLAGS := $(SYCL_C_FLAGS) -fsycl -D SYCL_1_2_1 -D ICX -DEVENT_PROFILE -DNEEDS_NULL_DEVICE_PTR
  SYCL_LD_FLAGS := $(SYCL_LD_FLAGS) -fsycl -L $(ONEAPI_PATH)/lib -L$(ONEAPI_PATH)/compiler/lib/intel64_lin -Wl,-rpath=$(ONEAPI_PATH)/lib,-rpath=$(ONEAPI_PATH)/compiler/lib/intel64_lin $(OPTS) -lstdc++
  ifeq ($(FPGA), INTEL)
  SYCL_LD_FLAGS := $(SYCL_LD_FLAGS) -fintelfpga -Xshardware
  JACCARD_REUSE=-reuse-exe=jaccardSYCL
  COMPARE_REUSE=-reuse-exe=compareCoords
  SYCL_C_FLAGS := $(SYCL_C_FLAGS) -fintelfpga -Xshardware
  endif
  ifeq ($(FPGA), INTEL_EMU)
  SYCL_LD_FLAGS := $(SYCL_LD_FLAGS) -fintelfpga
  SYCL_C_FLAGS := $(SYCL_C_FLAGS) -fintelfpga
  endif
endif

.PHONY: all
all: jaccardSYCL compareCoords fileConvert readCSRHeader

compareCoords: compareCoords.o readMtxToCSR.o
	$(SYCL) -o compareCoords compareCoords.o readMtxToCSR.o $(COMPARE_REUSE) $(SYCL_LD_FLAGS)

fileConvert: fileConvert.o filetypes.o readMtxToCSR.o
	$(SYCL) -o fileConvert fileConvert.o filetypes.o readMtxToCSR.o $(SYCL_LD_FLAGS)

jaccardSYCL: jaccardSYCL.o readMtxToCSR.o main.o filetypes.o
	$(SYCL) -o jaccardSYCL jaccardSYCL.o readMtxToCSR.o main.o filetypes.o $(JACCARD_REUSE) $(SYCL_LD_FLAGS)

readCSRHeader: readCSRHeader.o readMtxToCSR.o
	g++ -o readCSRHeader readCSRHeader.o readMtxToCSR.o $(LDFLAGS)

compareCoords.o: compareCoords.cpp readMtxToCSR.hpp standalone_csr.hpp
	$(SYCL) -o compareCoords.o -c compareCoords.cpp $(SYCL_C_FLAGS)

fileConvert.o: fileConvert.cpp
	$(SYCL) -o fileConvert.o -c fileConvert.cpp $(SYCL_C_FLAGS)

filetypes.o: filetypes.cpp
	g++ -o filetypes.o -c filetypes.cpp $(CFLAGS)

jaccardSYCL.o: jaccard.cpp standalone_csr.hpp
	$(SYCL) $(SYCL_C_FLAGS) -o jaccardSYCL.o -c jaccard.cpp -D STANDALONE

main.o: main.cpp
	$(SYCL) $(SYCL_C_FLAGS) -o main.o -c main.cpp

readCSRHeader.o: readCSRHeader.cpp
	g++ -o readCSRHeader.o -c readCSRHeader.cpp $(CFLAGS)

readMtxToCSR.o: readMtxToCSR.cpp readMtxToCSR.hpp standalone_csr.hpp
	$(SYCL) $(SYCL_C_FLAGS) -o readMtxToCSR.o -c readMtxToCSR.cpp 

.PHONY: clean
clean:
	rm jaccardSYCL compareCoords fileConvert readCSRHeader *.o	
