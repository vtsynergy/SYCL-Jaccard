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

template <typename WT>
std::tuple<int32_t, int32_t, WT> readCoord(std::ifstream &fileIn, bool isWeighted = true);
template <typename WT>
std::set<std::tuple<int32_t, int32_t, WT>>* readMtx(std::ifstream &fileIn, bool * hasWeights);
template <typename WT>
GraphCSRView<int32_t, int32_t, WT> * mtxSetToCSR(std::set<std::tuple<int32_t, int32_t, WT>> mtx, bool ignoreSelf = true, bool isZeroIndexed = false);
template <typename WT>
std::set<std::tuple<int32_t, int32_t, WT>> * CSRToMtx(GraphCSRView<int32_t, int32_t, WT> &csr, bool isZeroIndexed = false);

