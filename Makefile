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

.PHONY: all
all: jaccardCUDA compareCoords

jaccardCUDA: jaccardCUDA.o readMtxToCSR.o main.o
	nvcc -o jaccardCUDA jaccardCUDA.o readMtxToCSR.o main.o -g

main.o: main.cpp
	nvcc -o main.o -c main.cpp -g

jaccardCUDA.o: jaccard.cu standalone_csr.hpp
	nvcc jaccard.cu -o jaccardCUDA.o -D STANDALONE --system-include ./ -c -g

readMtxToCSR.o: readMtxToCSR.cpp readMtxToCSR.hpp standalone_csr.hpp
	g++ -o readMtxToCSR.o -c readMtxToCSR.cpp -g 

compareCoords: compareCoords.o readMtxToCSR.o
	g++ -o compareCoords compareCoords.o readMtxToCSR.o -g

compareCoords.o: compareCoords.cpp readMtxToCSR.hpp standalone_csr.hpp
	g++ -o compareCoords.o -c compareCoords.cpp -g

.PHONY: clean
clean:
	rm jaccardCUDA *.o	
