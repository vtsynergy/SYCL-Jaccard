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

template <typename WT>
std::tuple<int32_t, int32_t, WT> readCoord(std::ifstream &fileIn, bool isWeighted) {
  std::tuple<int32_t, int32_t, WT> line;
  if (isWeighted) {
    fileIn >> std::get<0>(line) >> std::get<1>(line) >> std::get<2>(line);
  } else {
    fileIn >> std::get<0>(line) >> std::get<1>(line);
    std::get<2>(line) = (WT)1.0;
  }
  return line;
}
//This assumes we've already read the header
template <typename WT>
std::set<std::tuple<int32_t, int32_t, WT>>* readMtx(std::ifstream &fileIn, bool * hasWeights) {
  std::set<std::tuple<int32_t, int32_t, WT>> * ret_edges = new std::set<std::tuple<int32_t, int32_t, WT>>();
  //TODO This should really do all the header parsing here, rather than relying on the caller to do it
  *hasWeights = false;
  bool isDirected = false;
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
        isDirected = true;
      }
    } //else {
      fileIn.ignore(2048, '\n');
    //}
  }
  //Discard the rest of the header
  while(fileIn.peek() == '%') fileIn.ignore(2048, '\n');
  //Read the parameters line
  int32_t rows, columns, nnz;
  fileIn >> rows >> columns >> nnz;
  //std::cout << "read parameters: " << rows << " " << columns << " " << nnz << std::endl;


  //Keep reading data lines to EOF
  do {
    std::tuple<int32_t, int32_t, WT> line = readCoord<WT>(fileIn, *hasWeights);
    if (!(fileIn.bad() || fileIn.fail() || fileIn.eof())) {
//      std::cout << "Read Line: " << std::get<0>(line) << " " << std::get<1>(line) << " " << std::get<2>(line) << std::endl;
      ret_edges->insert(line);
      if (!isDirected) {
      //Add a reverse edge
      //std::cout << "Adding reverse edge: " << std::get<1>(line) << " " << std::get<0>(line) << " " << std::get<2>(line) << std::endl;
      ret_edges->insert(std::tuple<int32_t, int32_t, WT>(std::get<1>(line), std::get<0>(line), std::get<2>(line)));
      }
    }
  } while (!fileIn.eof());
  return ret_edges;
}


//FIXME This may break down with directed graphs, specifically if a vertex only has inbound, but not outbound graphs, it won't get an entry in row_bounds (which should have start == end) for such a case)
template <typename WT>
GraphCSRView<int32_t, int32_t, WT> * mtxSetToCSR(std::set<std::tuple<int32_t, int32_t, WT>> mtx, bool ignoreSelf, bool isZeroIndexed) {
  std::vector<WT> * weights = new std::vector<WT>();
  std::vector<int32_t> * columns = new std::vector<int32_t>();
  std::vector<int32_t> * row_bounds = new std::vector<int32_t>({0});
  int32_t prev_src = std::get<0>(*(mtx.begin())) - (isZeroIndexed ? 0 : 1);
  //std::set should mean we're iterating over them in sorted order
  for (std::tuple<int32_t, int32_t, WT> edge : mtx) {
    int32_t source = std::get<0>(edge) - (isZeroIndexed ? 0 : 1);
    int32_t destination = std::get<1>(edge) - (isZeroIndexed ? 0 : 1);
    WT weight = std::get<2>(edge);
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
  //The GraphCSRView *does not* maintain it's own copy, just pointers, so they must be dynamically allocated
  return new GraphCSRView<int32_t, int32_t, WT>(row_bounds->data(), columns->data(), weights->data(), row_bounds->size()-1, weights->size());
}

template <typename WT>
std::set<std::tuple<int32_t, int32_t, WT>> * CSRToMtx(GraphCSRView<int32_t, int32_t, WT> &csr, bool isZeroIndexed) {
std::set<std::tuple<int32_t, int32_t, WT>> * ret_set = new std::set<std::tuple<int32_t, int32_t, WT>>();
  //TODO Is this legal to do?
  //cl::sycl::buffer<std::set<std::tuple<int32_t, int32_t, WT>>>(ret_set, csr.number_of_edges) ;

  //Submit a command group so we can use a host accessor
  cl::sycl::queue q = cl::sycl::queue(cl::sycl::cpu_selector());
  q.submit([&](cl::sycl::handler &cgh){
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
      for (int32_t offset = offset_acc[row], end = offset_acc[row+1]; offset < end; offset++) {
        ret_set->insert(std::tuple<int32_t, int32_t, WT>(row + (isZeroIndexed ? 0 : 1), indices_acc[offset] + (isZeroIndexed ? 0 : 1), edge_acc[offset]));
      }
    }
  });
  return ret_set;
}

//Explicit instantiations since separately compiling and linking
template std::set<std::tuple<int32_t, int32_t, double>> *readMtx(std::ifstream &fileIn, bool * hasWeights);
template std::set<std::tuple<int32_t, int32_t, float>> *readMtx(std::ifstream &fileIn, bool * hasWeights);
template GraphCSRView<int32_t, int32_t, double> * mtxSetToCSR(std::set<std::tuple<int32_t, int32_t, double>> mtx, bool ignoreSelf = true, bool isZeroIndexed = false);
template GraphCSRView<int32_t, int32_t, float> * mtxSetToCSR(std::set<std::tuple<int32_t, int32_t, float>> mtx, bool ignoreSelf = true, bool isZeroIndexed = false);
template std::set<std::tuple<int32_t, int32_t, double>> *CSRToMtx(GraphCSRView<int32_t, int32_t, double> &csr, bool isZeroIndexed = false);
template std::set<std::tuple<int32_t, int32_t, float>> *CSRToMtx(GraphCSRView<int32_t, int32_t, float> &csr, bool isZeroIndexed = false);
template std::tuple<int32_t, int32_t, double> readCoord(std::ifstream &fileIn, bool isWeighted = true);
template std::tuple<int32_t, int32_t, float> readCoord(std::ifstream &fileIn, bool isWeighted = true);
