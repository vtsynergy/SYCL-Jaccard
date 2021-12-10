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

#ifndef WEIGHT_TYPE
 #ifndef DISABLE_DP_WEIGHT
  #define WEIGHT_TYPE double
 #else
  #define WEIGHT_TYPE float
 #endif
#endif

int main(int argc, char * argv[]) {
  if (argc != 4) {
    std::cerr << "Error, incorrect number of args, usage is:\n.fileConvert <input.[mtx|csr]> <output.[mtx|csr]> <keepReverseEdges (0 or 1)>" << std::endl;
  }
  std::ifstream fileIn;
  std::ofstream fileOut;
  graphFileType inType, outType, working;
  setUpFiles(argv[1], argv[2], fileIn, fileOut, inType, outType);
  bool keepReverseEdges = static_cast<bool>(atoi(argv[3]));
  bool isWeighted = false, isDirected = false, hasReverseEdges = false, isZeroIndexed = false;
  int64_t numVerts = 0, numEdges = 0;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>* mtx_in = nullptr;
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> * csr_in = nullptr;
  //Fetch the input
  switch (inType) {
    case (mtx): {
      working = mtx;
      //Header information comes with the MTX reader
      mtx_in = fileToMTXSet<int32_t, int32_t, WEIGHT_TYPE>(fileIn, &isWeighted, &isDirected, &numVerts, &numEdges);
      //By spec, MTX doesn't typically have reverse edges (It would have to be in general form, which we couldn't distinguish from a regular directed graph without exhaustively checking all the edge pairs)
    }
    break;

    case (csr): {
      working = csr;
      //Header information comes from the file
      CSRFileHeader header;
      csr_in = static_cast<GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *>(FileToCSR(fileIn, &header));
      isWeighted = header.flags.isWeighted;
      isDirected = header.flags.isDirected;
      isZeroIndexed = header.flags.isZeroIndexed;
      hasReverseEdges = header.flags.hasReverseEdges;
      numVerts = header.numVerts;
      numEdges = header.numEdges;
    }
    break;

    default: {
      std::cerr << "Unsupported input file type" << std::endl;
    }
    break;
  }
  fileIn.close();
  //Check that we can actually respect a reverseEdge request, if not emit a warning
  if (keepReverseEdges) {
    if (isDirected) {
      std::cerr << "Warning, Cannot retain reverseEdges of Directed input, could cause collisions" << std::endl;
      keepReverseEdges = false;
    }
    if (outType == mtx) {
      std::cerr << "Warning, Cannot retain reverseEdges with MTX output, would be indistinguishable from directed" << std::endl;
      keepReverseEdges = false;
    }
  }
  //Generate reverse edges if we need to, remove them if we need to
  if (keepReverseEdges && !hasReverseEdges) {
    //Generate them
    if (working == csr) {
      //Switch it to MTX to reverse them
      mtx_in = CSRToMtx(*csr_in, isZeroIndexed, isWeighted);
      working = mtx;
      isZeroIndexed = false;
      //Don't need to maintain it as CSR anymore
      delete csr_in;
    }
    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> * reverse = invertDirection(*mtx_in);
    mtx_in->insert(reverse->begin(), reverse->end());
    hasReverseEdges = true;
    numEdges *= 2;
    delete reverse; 
  } else if (hasReverseEdges && !keepReverseEdges) { 
    //Remove them
    if (working == csr) {
      //Convert it to MTX to dedup
      mtx_in = CSRToMtx(*csr_in, isZeroIndexed, isWeighted);
      working = mtx;
      isZeroIndexed = false;
      //Don't need to maintain it as CSR anymore
      delete csr_in;
    } else if (working != mtx) {
      //Future formats;
    }
    removeReverseEdges(*mtx_in);
    hasReverseEdges = false;
    numEdges /= 2;
  }
  //And write it
  switch (outType) {
    case (csr): {
      if (working == mtx) {
        //Promote it to CSR
        csr_in = mtxSetToCSR(*mtx_in, true, isZeroIndexed);
        working = csr;
        isZeroIndexed = true;
        delete mtx_in;
      } else if (working != csr) {
        //Future formats
      }
      //Write it
      CSRToFile(fileOut, *csr_in, isZeroIndexed, isWeighted, isDirected, hasReverseEdges);
    }
    break;

    case (mtx): {
      //Just write it
      if (working == csr) {
        //Convert it back to MTX
        mtx_in = CSRToMtx(*csr_in, isZeroIndexed, isWeighted);
        working = mtx;
        isZeroIndexed = false;
        //Don't need to maintain it as CSR anymore
        delete csr_in;
      } else if (working != mtx) {
        //Future formats
      }
      //Write it
      mtxSetToFile(fileOut, *mtx_in, numVerts, numEdges, isWeighted, isDirected);
    }
    break;
    default: {
      std::cerr << "Unsupported output file type" << std::endl;
    }
    break;
  }
  fileOut.close();
  if (working == csr) { 
    delete csr_in;
  }
  if (working == mtx) delete mtx_in;
}
