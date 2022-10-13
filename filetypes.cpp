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

#if __GNUC__ == 7
  #include <experimental/filesystem>
  namespace std {
    namespace filesystem = experimental::filesystem;
  }
#else
  #include <filesystem>
#endif
#include "filetypes.hpp"

void setUpFiles(char * inFile, char * outFile, std::ifstream & retIFS, std::ofstream & retOFS, graphFileType & inType, graphFileType & outType) {
  std::filesystem::path inPath(inFile);
  std::filesystem::path outPath(outFile);
  if (inPath.extension() == ".mtx") {
    inType = mtx;
    retIFS = std::ifstream(inPath, std::ios_base::in);
  } else if (inPath.extension() == ".csr") {
    inType = csr;
    retIFS = std::ifstream(inPath, std::ios_base::in | std::ios_base::binary);
  } else {
    std::cerr << "Input File " << inPath << "has illegal extension, must be \".mtx\" (text) or \".csr\" (binary)" << std::endl;
    exit(1);
  }
  if (outPath.extension() == ".mtx") {
    outType = mtx;
    retOFS = std::ofstream(outPath, std::ios_base::out | std::ios_base::trunc);
  } else if (outPath.extension() == ".csr") {
    outType = csr;
    retOFS = std::ofstream(outPath, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
  } else {
    std::cerr << "Output File " << inPath << "has illegal extension, must be \".mtx\" (text) or \".csr\" (binary)" << std::endl;
    exit(2);
  }
}

