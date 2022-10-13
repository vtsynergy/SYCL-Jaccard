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

#include "filetypes.hpp"
#include "readMtxToCSR.hpp"
#include <cmath>
#include <iostream>
#include <set>
#include <tuple>

std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *readCoordFile(std::ifstream &fileIn) {
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *ret_set =
      new std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>();
  // Keep reading data lines to EOF
  do {
    std::tuple<int32_t, int32_t, WEIGHT_TYPE> line =
        readCoord<int32_t, int32_t, WEIGHT_TYPE>(fileIn, true);
    if (!(fileIn.bad() || fileIn.fail() || fileIn.eof())) {
      ret_set->insert(line);
    }
  } while (!fileIn.eof());
  return ret_set;
}

void printNeighborsMTX(int32_t src, int32_t dest,
                       std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>::iterator goldItr) {
  // goldItr should be in the range of src, since we're only called during a mismatch, backtrack
  // until the first neighbor
  while (std::get<0>(*(goldItr)) == src)
    --goldItr;
  ++goldItr; // we went one past
  // print until we run out of neighbors
  std::cerr << "Begin neighbors of " << src << std::endl;
  while (std::get<0>(*goldItr) == src) {
    std::cerr << "(" << std::get<0>(*goldItr) << ", " << std::get<1>(*goldItr) << ")" << std::endl;
    ++goldItr;
  }
  std::cerr << "End neighbors of " << src << std::endl;
  // dest could be before or after
  if (dest < src) {
    while (std::get<0>(*(goldItr)) >= dest)
      --goldItr;
    ++goldItr; // we went one past
  } else {
    while (std::get<0>(*(goldItr)) < dest)
      ++goldItr;
  }
  std::cerr << "Begin neighbors of " << dest << std::endl;
  while (std::get<0>(*goldItr) == dest) {
    std::cerr << "(" << std::get<0>(*goldItr) << ", " << std::get<1>(*goldItr) << ")" << std::endl;
    ++goldItr;
  }
  std::cerr << "End neighbors of " << dest << std::endl;
}

void printNeighborsCSR(int32_t src, int32_t dest,
                       GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *graph) {

  // print until we run out of neighbors
  std::cerr << "Begin neighbors of " << src << std::endl;
  for (int i = graph->offsets[src]; i < graph->offsets[src + 1]; i++) {
    std::cerr << "(" << src << ", " << graph->indices[i] << ")" << std::endl;
  }
  std::cerr << "End neighbors of " << src << std::endl;
  // dest could be before or after
  std::cerr << "Begin neighbors of " << dest << std::endl;
  for (int i = graph->offsets[dest]; i < graph->offsets[dest + 1]; i++) {
    std::cerr << "(" << dest << ", " << graph->indices[i] << ")" << std::endl;
  }
  std::cerr << "End neighbors of " << dest << std::endl;
}

void readInputFiles(char *goldFile, char *testFile, graphFileType &type,
                    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> **goldSet,
                    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> **testSet,
                    GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> **goldCSR,
                    GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> **testCSR, int32_t &numVerts,
                    int32_t &numEdges) {
  graphFileType goldType, testType;
  std::ifstream gold;
  std::ifstream test;
  setupInFile(goldFile, gold, goldType);
  setupInFile(testFile, test, testType);
  type = goldType;
  // Read files and convert if necessary
  if (goldType == csr) {
    CSRFileHeader header;
    *goldCSR = static_cast<GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *>(FileToCSR(gold, &header));
    if (header.flags.isVertexT64 || header.flags.isEdgeT64 ||
        (header.flags.isWeighted &&
         ((header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, double>::value) ||
          (!header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, float>::value)))) {
      std::cerr << "Binary CSR Gold Input Header does not match required weight type" << std::endl;
      exit(3);
    }
    if (testType == csr) {
      *testCSR =
          static_cast<GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *>(FileToCSR(test, &header));
      if (header.flags.isVertexT64 || header.flags.isEdgeT64 ||
          (header.flags.isWeighted &&
           ((header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, double>::value) ||
            (!header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, float>::value)))) {
        std::cerr << "Binary CSR Test Input Header does not match required weight type"
                  << std::endl;
        exit(3);
      }

    } else if (testType == mtx) {
      bool hasWeights, isDirected;
      *testSet = fileToMTXSet<int32_t, int32_t, WEIGHT_TYPE>(test, &hasWeights, &isDirected,
                                                             &numVerts, &numEdges, false);
      *testCSR = mtxSetToCSR(**testSet, true, false);
      delete *testSet;

    } else {
      std::cerr << "File extension must be csr or mtx" << std::endl;
      exit(-1);
    }
    numVerts = (*goldCSR)->number_of_vertices;
    numEdges = (*goldCSR)->number_of_edges;
  } else if (goldType == mtx) {
    bool hasWeights, isDirected;
    *goldSet = fileToMTXSet<int32_t, int32_t, WEIGHT_TYPE>(gold, &hasWeights, &isDirected,
                                                           &numVerts, &numEdges, false);
    if (testType == csr) {
      CSRFileHeader header;
      *testCSR =
          static_cast<GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *>(FileToCSR(test, &header));
      if (header.flags.isVertexT64 || header.flags.isEdgeT64 ||
          (header.flags.isWeighted &&
           ((header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, double>::value) ||
            (!header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, float>::value)))) {
        std::cerr << "Binary CSR Gold Input Header does not match required weight type"
                  << std::endl;
        exit(3);
      }
      *testSet = CSRToMtx(**testCSR, header.flags.isZeroIndexed, true);
      delete (*testCSR)->offsets;
      delete (*testCSR)->indices;
      delete (*testCSR)->edge_data;
      delete *testCSR;

    } else if (testType == mtx) {
      *testSet = fileToMTXSet<int32_t, int32_t, WEIGHT_TYPE>(test, &hasWeights, &isDirected,
                                                             &numVerts, &numEdges, false);

    } else {
      std::cerr << "File extension must be csr or mtx" << std::endl;
      exit(-1);
    }
  } else {
    std::cerr << "File extension must be csr or mtx" << std::endl;
    exit(-1);
  }
  gold.close();
  test.close();
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Error, incorrect number of args, usage is:\n./compareCoords <goldFile> "
                 "<testFile> <DP_tolerance>"
              << std::endl;
  }
  // We will only populate the pointers we actually use
  graphFileType compType;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *goldSet;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *testSet;
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *goldCSR;
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *testCSR;
  int32_t numVerts = 0, numEdges = 0, count = 0;
  // This will use goldType, and convert testType if necessary
  readInputFiles(argv[1], argv[2], compType, &goldSet, &testSet, &goldCSR, &testCSR, numVerts,
                 numEdges);
  double tol = atof(argv[3]);
  bool keepReverseEdges = true, isWeighted = true;
  bool hasWeights = true, isDirected = false;
  int32_t warnCount = 0;

  std::cout << "Inputs successfully read, proceeding to test" << std::endl;
  int32_t onePct = numEdges / 100;

  if (compType == csr) {
    // If set sizes differ, report it and try to make due
    if (goldCSR->number_of_edges != testCSR->number_of_edges) {
      std::cerr << "Warning: sets differ in size! Gold: " << goldCSR->number_of_edges
                << " vs. Test: " << testCSR->number_of_edges << std::endl;
    }

    // Instead of iterators, we need a vertex and edge index for each CSR
    int32_t goldSrc = 0, goldDest = 0, testSrc = 0, testDest = 0;
    while ((goldSrc < goldCSR->number_of_vertices && goldDest < goldCSR->number_of_edges) &&
           (testSrc < testCSR->number_of_vertices && testDest < testCSR->number_of_edges)) {
      // If the coordinates match, then test for equality
      if ((goldSrc == testSrc) && (goldCSR->indices[goldDest] == testCSR->indices[testDest])) {
        // Check difference;
        if (fabs(goldCSR->edge_data[goldDest] - testCSR->edge_data[testDest]) > tol) {
          std::cerr << "Warning: zero-indexed elements at (" << goldSrc << ","
                    << goldCSR->indices[goldDest] << ") differ by more than " << tol
                    << " tolerance! Gold: " << goldCSR->edge_data[goldDest]
                    << " vs. Test: " << testCSR->edge_data[testDest] << std::endl;
          ++warnCount;
          printNeighborsCSR(goldSrc, goldCSR->indices[goldDest], goldCSR);
        }
        ++goldDest;
        ++testDest;
      } else { // Coordinate mismatch, shouldn't happen
        if (goldCSR->indices[goldDest] < testCSR->indices[testDest]) {
          std::cerr << "Element missing from test file: " << goldSrc << " "
                    << goldCSR->indices[goldDest] << " " << goldCSR->edge_data[goldDest]
                    << std::endl;
          ++warnCount;
          ++goldDest;
        } else {
          std::cerr << "Element added to test file: " << testSrc << " "
                    << testCSR->indices[testDest] << " " << testCSR->edge_data[testDest]
                    << std::endl;
          ++warnCount;
          ++testDest;
        }
      }
      if (goldDest == goldCSR->offsets[goldSrc + 1]) ++goldSrc;
      if (testDest == testCSR->offsets[testSrc + 1]) ++testSrc;
      count++;

      if ((count % onePct) == 0)
        std::cout << "Completed " << count / onePct << "\% of scan" << std::endl;
    }
    if (goldDest != goldCSR->number_of_edges) {
      std::cerr << "Warning: " << (goldCSR->number_of_edges - goldDest)
                << " elements unscanned from Gold input" << std::endl;
    }
    if (testDest != testCSR->number_of_edges) {
      std::cerr << "Warning: " << (testCSR->number_of_edges - testDest)
                << " elements unscanned from Test input" << std::endl;
    }
    delete goldCSR->offsets;
    delete goldCSR->indices;
    delete goldCSR->edge_data;
    delete goldCSR;
    delete testCSR->offsets;
    delete testCSR->indices;
    delete testCSR->edge_data;
    delete testCSR;
  } else if (compType == mtx) {
    // If the size of sets is different, they won't necessarily align, but they're sorted so we can
    // try to make due
    if (goldSet->size() != testSet->size()) {
      std::cerr << "Warning: sets differ in size! Gold: " << goldSet->size()
                << " vs. Test: " << testSet->size() << std::endl;
    }

    // Create our two iterators (rather than a loop, so we can advance the lessor on mismatch
    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>::iterator goldItr = goldSet->begin();
    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>::iterator testItr = testSet->begin();
    while (goldItr != goldSet->end() && testItr != testSet->end()) {

      // If the coordinates match, then test for equality
      if ((std::get<0>(*goldItr) == std::get<0>(*testItr)) &&
          (std::get<1>(*goldItr) == std::get<1>(*testItr))) {
        // Check difference;
        if (fabs(std::get<2>(*goldItr) - std::get<2>(*testItr)) > tol) {
          std::cerr << "Warning: one-indexed elements at (" << std::get<0>(*goldItr) << ","
                    << std::get<1>(*goldItr) << ") differ by more than " << tol
                    << " tolerance! Gold: " << std::get<2>(*goldItr)
                    << " vs. Test: " << std::get<2>(*testItr) << std::endl;
          ++warnCount;
          printNeighborsMTX(std::get<0>(*goldItr), std::get<1>(*goldItr), goldItr);
        }
        ++goldItr;
        ++testItr;
      } else { // Coordinate mismatch, shouldn't happen
        if (*goldItr < *testItr) {
          std::cerr << "Element missing from test file: " << std::get<0>(*goldItr) << " "
                    << std::get<1>(*goldItr) << " " << std::get<2>(*goldItr) << std::endl;
          ++warnCount;
          ++goldItr;
        } else {
          std::cerr << "Element added to test file: " << std::get<0>(*testItr) << " "
                    << std::get<1>(*testItr) << " " << std::get<2>(*testItr) << std::endl;
          ++warnCount;
          ++testItr;
        }
      }
      count++;

      if ((count % onePct) == 0)
        std::cout << "Completed " << count / onePct << "\% of scan" << std::endl;
    }
    // If we ran out of one set before the other
    if (goldItr != goldSet->end()) {
      std::cerr << "Warning: " << std::distance(goldItr, goldSet->end())
                << " elements unscanned from Gold input" << std::endl;
    }
    if (testItr != testSet->end()) {
      std::cerr << "Warning: " << std::distance(testItr, testSet->end())
                << " elements unscanned from Test input" << std::endl;
    }
    delete goldSet;
    delete testSet;
  }
  return warnCount;
}
