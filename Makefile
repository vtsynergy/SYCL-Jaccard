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

HIPSYCL_PATH=`pwd`/../edge2-hipSYCL
HIPSYCL_CLANG_PATH=`pwd`/../../coreTSARWorkspace/SYCL-implementations/dependencies/llvm-github-srcInstall/
SYCL=$(HIPSYCL_PATH)/bin/syclcc
#SYCL_FLAGS=-isystem $(HIPSYCL_PATH) --hipsycl-targets="omp;hip:gfx900" -Wl,-rpath=$(HIPSYCL_PATH)/lib
SYCL_FLAGS=-isystem $(HIPSYCL_PATH) --hipsycl-targets="omp;hip:gfx900,gfx803" -Wl,-rpath=$(HIPSYCL_PATH)/lib,-rpath=$(HIPSYCL_CLANG_PATH)/lib -O3
ROCPROFILER_INCL= -I /opt/rocm/include -I /opt/rocm/include/hsa
ROCPROFILER_LIB= -L /opt/rocm/lib -lrocprofiler64

.PHONY: all
all: jaccardSYCL compareCoords

jaccardSYCL: jaccardSYCL.o readMtxToCSR.o main.o
	$(SYCL) $(SYCL_FLAGS) -o jaccardSYCL jaccardSYCL.o readMtxToCSR.o main.o -g $(ROCPROFILER_LIB)

main.o: main.cpp
	$(SYCL) $(SYCL_FLAGS) -o main.o -c main.cpp -g $(ROCPROFILER_INCL)

jaccardSYCL.o: jaccard.cpp standalone_csr.hpp
	$(SYCL) $(SYCL_FLAGS) -o jaccardSYCL.o -c jaccard.cpp -g -D STANDALONE -v

readMtxToCSR.o: readMtxToCSR.cpp readMtxToCSR.hpp standalone_csr.hpp
	$(SYCL) $(SYCL_FLAGS) -o readMtxToCSR.o -c readMtxToCSR.cpp -g 

compareCoords: compareCoords.o readMtxToCSR.o
	$(SYCL) $(SYCL_FLAGS) -o compareCoords compareCoords.o readMtxToCSR.o -g

compareCoords.o: compareCoords.cpp readMtxToCSR.hpp standalone_csr.hpp
	$(SYCL) $(SYCL_FLAGS) -o compareCoords.o -c compareCoords.cpp -g

.PHONY: clean
clean:
	rm jaccardSYCL *.o	
