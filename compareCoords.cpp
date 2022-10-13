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

#include <iostream>
#include <set>
#include <tuple>
#include <cmath>
#include "readMtxToCSR.hpp"

std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> * readCoordFile(std::ifstream &fileIn) {
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> * ret_set = new std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>();
  //Keep reading data lines to EOF
  do {
    std::tuple<int32_t, int32_t, WEIGHT_TYPE> line = readCoord<int32_t, int32_t, WEIGHT_TYPE>(fileIn, true);
    if (!(fileIn.bad() || fileIn.fail() || fileIn.eof())) {
      ret_set->insert(line);
    } 
  } while (!fileIn.eof());
  return ret_set;
}


int main(int argc, char * argv[]) {
  if (argc != 4) {
    std::cerr << "Error, incorrect number of args, usage is:\n./compareCoords <goldFile> <testFile> <DP_tolerance>" << std::endl;
  }
  std::ifstream gold(argv[1]);
  std::ifstream test(argv[2]);
  double tol = atof(argv[3]);
  size_t warnCount = 0;

  std::set<std::tuple<int32_t, int32_t, double>> * goldSet = readCoordFile(gold);
  std::set<std::tuple<int32_t, int32_t, double>> * testSet = readCoordFile(test);
  gold.close();
  test.close();

  //If the size of sets is different, they won't necessarily align, but they're sorted so we can try to make due
  if (goldSet->size() != testSet->size()) {
    std::cerr << "Warning: sets differ in size! Gold: " << goldSet->size() << " vs. Test: " << testSet->size() << std::endl;
  }

  //Create our two iterators (rather than a loop, so we can advance the lessor on mismatch
  std::set<std::tuple<int32_t, int32_t, double>>::iterator goldItr = goldSet->begin();
  std::set<std::tuple<int32_t, int32_t, double>>::iterator testItr = testSet->begin();
  while (goldItr != goldSet->end() && testItr != testSet->end()) {
    //If the coordinates match, then test for equality
    if ((std::get<0>(*goldItr) == std::get<0>(*testItr)) && (std::get<1>(*goldItr) == std::get<1>(*testItr))) {
      //Check difference;
      if (fabs(std::get<2>(*goldItr) - std::get<2>(*testItr)) > tol) {
        std::cerr << "Warning: elements at (" << std::get<0>(*goldItr) << "," << std::get<1>(*goldItr) << ") differ by more than " << tol << " tolerance! Gold: " << std::get<2>(*goldItr) << " vs. Test: " << std::get<2>(*testItr) << std::endl;
        ++warnCount;
      }
      ++goldItr;
      ++testItr;
    } else { //Coordinate mismatch, shouldn't happen
      if (*goldItr < *testItr) {
        std::cerr << "Element missing from test file: " << std::get<0>(*goldItr) << " " << std::get<1>(*goldItr) << " " << std::get<2>(*goldItr) << std::endl;
        ++warnCount;
        ++goldItr;
      } else {
        std::cerr << "Element added to test file: " << std::get<0>(*testItr) << " " << std::get<1>(*testItr) << " " << std::get<2>(*testItr) << std::endl;
        ++warnCount;
        ++testItr;
      }
    }
  }
  //If we ran out of one set before the other
  if (goldItr != goldSet->end()) {
    std::cerr << "Warning: " << std::distance(goldItr, goldSet->end()) << " elements unscanned from Gold input" << std::endl;
  }
  if (testItr != testSet->end()) {
    std::cerr << "Warning: " << std::distance(testItr, testSet->end()) << " elements unscanned from Test input" << std::endl;
  }
  delete goldSet;
  delete testSet;

  return warnCount;
}
