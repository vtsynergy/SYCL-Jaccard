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

#include <vector>
#include <iostream>
#include <string.h>
#include "readMtxToCSR.hpp"

//FIXME Producing a weight vector that we're not going to use inflates memory O(edges) unnecessarily
template <typename ET, typename VT, typename WT>
std::tuple<ET, VT, WT> readCoord(std::ifstream &fileIn, bool isWeighted) {
  std::tuple<ET, VT, WT> line;
  if (isWeighted) {
    fileIn >> std::get<0>(line) >> std::get<1>(line) >> std::get<2>(line);
  } else {
    fileIn >> std::get<0>(line) >> std::get<1>(line);
    std::get<2>(line) = (WT)1.0;
  }
  return line;
}
//This assumes we've already read the header
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>>* fileToMTXSet(std::ifstream &fileIn, bool * hasWeights, bool * isDirected) {
  std::set<std::tuple<ET, VT, WT>> * ret_edges = new std::set<std::tuple<ET, VT, WT>>();
  std::vector<std::tuple<ET, VT, WT>> * tmp_vec = new std::vector<std::tuple<ET, VT, WT>>();
  //TODO This should really do all the header parsing here, rather than relying on the caller to do it
  *hasWeights = false;
  char token[15] = {0};
  //Read the important header line(s)
  while(fileIn.peek() == '%') {
    fileIn.get(token, 15, ' '); //Grab the first token
    if (strstr(token, "MatrixMarket")) { //found the important line, don't care about the pointer
      std::string object, format, field, symmetry;
      fileIn >> object >> format >> field >> symmetry;
     // std::cout << " Read format: " << object << " " << format << " " << field << " " << symmetry << std::endl;
      if (object != "matrix") {
        std::cerr << "Error, can only process Matrix MTX files" << std::endl;
	exit(1);
      }
      if (format != "coordinate") {
        std::cerr << "Error, can only process Coordinate MTX files" << std::endl;
        exit(2);
      }
      //TODO treat the different types of weights differently
      if (field == "real" || field == "double" || field == "integer") {
        *hasWeights = true;
      }
      if (field == "complex") {
        std::cerr << "Error, cannot process complex weights from MTX files" << std::endl;
        exit(3);
      }
      //TODO, skew-symmetric would defacto not include self-edges, can we do anything with that information?
      if (symmetry == "general") {
        *isDirected = true;
      }
    } //else {
      fileIn.ignore(2048, '\n');
    //}
  }
  //Discard the rest of the header
  while(fileIn.peek() == '%') fileIn.ignore(2048, '\n');
  //Read the parameters line
  int64_t rows, columns, nnz;
  fileIn >> rows >> columns >> nnz;
  //std::cout << "read parameters: " << rows << " " << columns << " " << nnz << std::endl;


  //Keep reading data lines to EOF
  do {
    std::tuple<ET, VT, WT> line = readCoord<ET, VT, WT>(fileIn, *hasWeights);
    if (!(fileIn.bad() || fileIn.fail() || fileIn.eof())) {
#ifdef DEBUG_2
      std::cout << "Read Line: " << std::get<0>(line) << " " << std::get<1>(line) << " " << std::get<2>(line) << std::endl;
#endif //DEBUG_2
      tmp_vec->push_back(line);
    }
  } while (!fileIn.eof());
  ret_edges->insert(tmp_vec->begin(), tmp_vec->end());
  delete tmp_vec;
  if (!(*isDirected)) {
    std::set<std::tuple<ET, VT, WT>> * reverse = invertDirection(*ret_edges);
    ret_edges->insert(reverse->begin(), reverse->end());
    delete reverse;
  }
  return ret_edges;
}
template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>>* invertDirection(std::set<std::tuple<ET, VT, WT>> & mtx) {
  std::set<std::tuple<ET, VT, WT>> * ret_edges = new std::set<std::tuple<ET, VT, WT>>();
  std::vector<std::tuple<ET, VT, WT>> * tmp_vec = new std::vector<std::tuple<ET, VT, WT>>();
  for (std::tuple<ET, VT, WT> coord : mtx) {
    //Add a reverse edge
#ifdef DEBUG_2
    std::cout << "Adding reverse edge: " << std::get<1>(coord) << " " << std::get<0>(coord) << " " << std::get<2>(coord) << std::endl;
#endif //DEBUG_2
    tmp_vec->push_back(std::tuple<ET, VT, WT>(std::get<1>(coord), std::get<0>(coord), std::get<2>(coord)));
  }
  ret_edges->insert(tmp_vec->begin(), tmp_vec->end());
  delete tmp_vec;
}
template <typename ET, typename VT, typename WT>
void removeReverseEdges(std::set<std::tuple<ET, VT, WT>> &mtx) {
  for (std::tuple<ET, VT, WT> coord : mtx) {
    if (mtx.find(coord) != mtx.end()) {
      mtx.erase(std::tuple<ET, VT, WT>(std::get<1>(coord), std::get<0>(coord), std::get<2>(coord)));
    } else {
      std::cerr << "Warning, attempting to iterate already-removed reverse edge: " << std::get<0>(coord) << " " << std::get<1>(coord) << " " << std::get<2>(coord) << std::endl;
    }
  }
}


//FIXME This may break down with directed graphs, specifically if a vertex only has inbound, but not outbound graphs, it won't get an entry in row_bounds (which should have start == end) for such a case)
template <typename ET, typename VT, typename WT>
GraphCSRView<VT, ET, WT> * mtxSetToCSR(std::set<std::tuple<ET, VT, WT>> mtx, bool ignoreSelf, bool isZeroIndexed) {
  std::vector<WT> * weights = new std::vector<WT>();
  std::vector<VT> * columns = new std::vector<VT>();
  std::vector<ET> * row_bounds = new std::vector<ET>({0});
  int32_t prev_src = std::get<0>(*(mtx.begin())) - (isZeroIndexed ? 0 : 1);
  //std::set should mean we're iterating over them in sorted order
  for (std::tuple<ET, VT, WT> edge : mtx) {
    ET source = std::get<0>(edge) - (isZeroIndexed ? 0 : 1);
    VT destination = std::get<1>(edge) - (isZeroIndexed ? 0 : 1);
    WT weight = std::get<2>(edge);
#ifdef DEBUG_2
    std::cout << "CSR conversion of edge: " << source << " " << destination << " " << weight << std::endl;
    std::cout << "Previous source: " << prev_src << std::endl;
#endif //DEBUG_2
    while (source != prev_src) { // Needs to be a loop to skip empty rows and dropped self-only verts
      row_bounds->push_back(weights->size()); //Close the previous row's bounds
      ++prev_src;// = source;
    }
    if (source != destination || !ignoreSelf) { // Don't add a self reference
      weights->push_back(weight);
      columns->push_back(destination);
    }
  }
  row_bounds->push_back(weights->size()); //Close the final row's bounds
#ifdef DEBUG_2
  std::cout << "Final CSR" << std::endl;
  std::cout << "RowBounds ";
  for (auto row : *row_bounds) {
    std::cout << row << ", ";
  }
  std::cout << std::endl;
  std::cout << "Column ";
  for (auto col : *columns) {
    std::cout << col << ", ";
  }
  std::cout << std::endl;
  std::cout << "Weights ";
  for (auto weight : *weights) {
    std::cout << weight << ", ";
  }
  std::cout << std::endl;
#endif //DEBUG_2
  //The GraphCSRView *does not* maintain it's own copy, just pointers, so they must be dynamically allocated
  return new GraphCSRView<VT, ET, WT>(row_bounds->data(), columns->data(), weights->data(), row_bounds->size()-1, weights->size());
}

template <typename ET, typename VT, typename WT>
std::set<std::tuple<ET, VT, WT>> * CSRToMtx(GraphCSRView<VT, ET, WT> &csr, bool isZeroIndexed) {
std::set<std::tuple<ET, VT, WT>> * ret_set = new std::set<std::tuple<ET, VT, WT>>();
std::vector<std::tuple<ET, VT, WT>> * tmp_vec = new std::vector<std::tuple<ET, VT, WT>>();
  //TODO Is this legal to do?
  //cl::sycl::buffer<std::set<std::tuple<int32_t, int32_t, WT>>>(ret_set, csr.number_of_edges) ;

  //Submit a command group so we can use a host accessor
  //cl::sycl::queue q = cl::sycl::queue(cl::sycl::cpu_selector());
  //q.submit([&](cl::sycl::handler &cgh){
    //Host accessors to read the the CSR buffers
    //TODO Replace these with SYCL 2020 get_host_accessor, once supported
    auto offset_acc = csr.offsets.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(csr.number_of_vertices+1));
    auto indices_acc = csr.indices.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(csr.number_of_vertices));
    auto edge_acc = csr.edge_data.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(csr.number_of_edges));
    //TODO: Rework this to support parallel construction of the output set
    //FIXME: This probably has to actually be a task
    //cgh.single_task<class CSRToMTX_kern>([=]
    //Just iterate over all the rows
    for (int row = 0; row < csr.number_of_vertices; row++) {
      for (ET offset = offset_acc[row], end = offset_acc[row+1]; offset < end; offset++) {
        tmp_vec->push_back(std::tuple<ET, VT, WT>(row + (isZeroIndexed ? 0 : 1), indices_acc[offset] + (isZeroIndexed ? 0 : 1), edge_acc[offset]));
      }
    }
  //});
  ret_set->insert(tmp_vec->begin(), tmp_vec->end());
  delete tmp_vec;
  return ret_set;
}

template <typename ET, typename VT, typename WT>
void CSRToFile(std::ofstream &fileOut, GraphCSRView<VT, ET, WT> &csr, bool isZeroIndexed, bool isWeighted, bool isDirected, bool keepReverseEdges) {
  int64_t indexLeng, offsetLeng;
  VT * indices;
  ET * offsets;
  WT * weights;
  CSRFileHeader header = {CSR_BINARY_FORMAT_VERSION,
                          int64_t{csr.number_of_vertices},
                          int64_t{csr.number_of_edges},
                            {isWeighted,
                             isZeroIndexed,
                             isDirected,
                             keepReverseEdges,
                             std::is_same<VT, int64_t>::value,
                             std::is_same<ET, int64_t>::value,
                             std::is_same<WT, int64_t>::value
                            }
                          };
  std::set<std::tuple<ET, VT, WT>> * mtx = nullptr;
  GraphCSRView<VT, ET, WT> * halfGraph = nullptr;
  if (!keepReverseEdges){
    //FIXME: Add a step to convert the CSR to MTX and to remove duplicates if !isDirected
    mtx = CSRToMtx(csr);
    removeReverseEdges(*mtx);
    halfGraph = mtxSetToCSR(*mtx);
    offsets = halfGraph->offsets;
    offsetLeng = halfGraph->number_of_vertices+1;
    indices = halfGraph->indices;
    indexLeng = halfGraph->number_of_edges;
    weights = halfGraph->edge_data;
  } else {
    offsets = csr.offsets;
    offsetLeng = csr.number_of_vertices+1;
    indices = csr.indices;
    indexLeng = csr.number_of_edges;
    weights = csr.edge_data;
  }
  //Write header
  fileOut.write(reinterpret_cast<char*>(&header), sizeof(CSRFileHeader));
  //Write Vertex Offsets
  fileOut.write(reinterpret_cast<char*>(offsets), sizeof(ET)*offsetLeng);
  //Write Neighbor edges
  fileOut.write(reinterpret_cast<char*>(indices), sizeof(VT)*indexLeng);
  //Write Weights (If any)
  if (isWeighted) {
    fileOut.write(reinterpret_cast<char*>(weights), sizeof(WT)*indexLeng);
  }
  if (mtx != nullptr) delete mtx;
  if (halfGraph != nullptr) delete mtx;
}

template<typename ET, typename VT, typename WT>
inline GraphCSRView<VT, ET, WT> * CSRFileReader(std::ifstream &fileIn, CSRFileHeader header) {
  //Read Vertex Offsets
  std::vector<ET> * offsets = new std::vector<ET>(header.numVerts+1);
  fileIn.read(reinterpret_cast<char*>(offsets->data()),sizeof(ET)* (header.numVerts+1));
  //Read Neighbor Indices
  std::vector<VT> * indices = new std::vector<VT>(header.numEdges);
  fileIn.read(reinterpret_cast<char*>(indices->data()),sizeof(VT)* header.numEdges);
  //Read Weights (If any)
  WT * weightsPtr = nullptr;
  if (header.flags.isWeighted) {
    std::vector<WT> * weights = new std::vector<WT>(header.numEdges);
    fileIn.read(reinterpret_cast<char*>(weights->data()),sizeof(WT)* header.numEdges);
    weightsPtr = weights->data();
  }
  GraphCSRView<VT, ET, WT> * retGraph = new GraphCSRView<VT, ET, WT>(offsets->data(), indices->data(), weightsPtr, header.numVerts, header.numEdges);
  if (!header.flags.isDirected && !header.flags.hasReverseEdges) {
    //Undirected and need to add reverse edges
    std::set<std::tuple<ET, VT, WT>> * mtx = CSRToMtx(*retGraph, header.flags.isZeroIndexed);
    delete retGraph;
    std::set<std::tuple<ET, VT, WT>> * reverse = invertDirection(*mtx);
    mtx->insert(reverse->begin(), reverse->end());
    delete reverse;
    retGraph = mtxSetToCSR(*mtx, header.flags.isZeroIndexed);
    delete mtx;
  }
  //Construct and return the Graph
  return retGraph; 
}

//These inlines just progressively decode the header flags into specializations
template <typename VT, typename ET>
inline void * CSRFileReader(std::ifstream &fileIn, CSRFileHeader header) {
  if (header.flags.isWeightT64) return static_cast<void*>(CSRFileReader<VT, ET, double>(fileIn, header));
  else return static_cast<void*>(CSRFileReader<VT, ET, float>(fileIn, header));
}
template <typename VT>
inline void * CSRFileReader(std::ifstream &fileIn, CSRFileHeader header) {
  if (header.flags.isEdgeT64) return CSRFileReader<VT, int64_t>(fileIn, header);
  else return CSRFileReader<VT, int32_t>(fileIn, header);
}
inline void * CSRFileReader(std::ifstream &fileIn, CSRFileHeader header) {
  if (header.flags.isVertexT64) return CSRFileReader<int64_t>(fileIn, header);
  else return CSRFileReader<int32_t>(fileIn, header);
}

void * FileToCSR(std::ifstream &fileIn, CSRFileHeader * header){
  //Read the header (local)
  CSRFileHeader localHeader;
  fileIn.read(reinterpret_cast<char*>(&localHeader),sizeof(CSRFileHeader));
  //If header pointer is non-null, set it
  if (header != nullptr) *header = localHeader;
  return CSRFileReader(fileIn, localHeader);
}

//Explicit instantiations since separately compiling and linking
template std::tuple<int32_t, int32_t, WEIGHT_TYPE> readCoord(std::ifstream &fileIn, bool isWeighted);
template std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *fileToMTXSet(std::ifstream &fileIn, bool * hasWeights, bool * isDirected);
template GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> * mtxSetToCSR(std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> mtx, bool ignoreSelf, bool isZeroIndexed);
template std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *CSRToMtx(GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> &csr, bool isZeroIndexed);
template void CSRToFile<int32_t, int32_t, WEIGHT_TYPE>(std::ofstream &fileOut, GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> &csr, bool isZeroIndexed, bool isWeighted, bool isDirected, bool keepReverseEdges);
