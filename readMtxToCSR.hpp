 /*
 * Copyright (c) 2021-2022, Virginia Tech.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <set>
#include <tuple>
#include "standalone_csr.hpp"

template <typename ET, typename VT, typename WT>
std::tuple<ET, VT, WT> readCoord(std::ifstream &fileIn, bool isWeighted = true);
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>>* fileToMTXSet(std::ifstream &fileIn, bool * hasWeights, bool * isDirected);
template <typename ET, typename VT, typename WT>
GraphCSRView<VT, ET, WT> * mtxSetToCSR(std::set<std::tuple<ET, VT, WT>> mtx, bool ignoreSelf = true, bool isZeroIndexed = false);
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>> * CSRToMtx(GraphCSRView<VT, ET, WT> &csr, bool isZeroIndexed = false);


typedef struct {
  int64_t numVerts;
  int64_t numEdges;
  struct alignas(alignof(int64_t)) {
    bool isWeighted : 1;
    bool isZeroIndexed : 1;
    bool isVertexT64 : 1;
    bool isEdgeT64 : 1;
    bool isWeightT64 : 1;
  } flags;
} CSRFileHeader;

template <typename ET, typename VT, typename WT>
void CSRToFile(std::ofstream &fileOut, GraphCSRView<VT, ET, WT> &csr, bool isZeroIndexed = false, bool isWeighted = false);

void * FileToCSR(std::ifstream &fileIn, CSRFileHeader * header);
