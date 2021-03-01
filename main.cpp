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
#include <cuda_runtime.h>
#include "readMtxToCSR.hpp" //implicitly includes standalone_csr.hpp
#include "standalone_algorithms.hpp"
#include "standalone_csr.hpp"


#ifndef WT
#define WT double
#endif

template <typename vertex_t, typename edge_t, typename weight_t>
void cudaMemcpyCSR(GraphCSRView<vertex_t, edge_t, weight_t> dst, GraphCSRView<vertex_t, edge_t, weight_t> src, enum cudaMemcpyKind dir) {
  if (dst.offsets != nullptr && src.offsets != nullptr) cudaMemcpy(dst.offsets, src.offsets, sizeof(edge_t) * (dst.number_of_vertices+1), dir);
  if (dst.indices != nullptr && src.indices != nullptr) cudaMemcpy(dst.indices, src.indices, sizeof(vertex_t) * dst.number_of_edges, dir);
  if (dst.edge_data != nullptr && src.edge_data != nullptr) cudaMemcpy(dst.edge_data, src.edge_data, sizeof(weight_t) * dst.number_of_edges, dir);
}

int main(int argc, char * argv[]) {

  //Open the specified file for reading
  //TODO arg bounds safety
  std::ifstream fileIn(argv[1]);
  bool isWeighted;
  std::set<std::tuple<int32_t, int32_t, WT>>* mtx_graph = readMtx<WT>(fileIn, &isWeighted);
  fileIn.close();

  //Convert it to a CSR
  GraphCSRView<int32_t, int32_t, WT> * graph = mtxSetToCSR(*mtx_graph);
  //Make a GPU graph
  int32_t * gpu_offsets, * gpu_columns;
  WT * gpu_weights;
  cudaMalloc(&gpu_offsets, sizeof(int32_t) * (graph->number_of_vertices +1));
  cudaMalloc(&gpu_columns, sizeof(int32_t) * graph->number_of_edges);
  cudaMalloc(&gpu_weights, sizeof(WT) * graph->number_of_edges);
  GraphCSRView<int32_t, int32_t, WT> gpu_graph(gpu_offsets, gpu_columns, gpu_weights, graph->number_of_vertices, graph->number_of_edges);
  //Copy data to it
  cudaMemcpyCSR<int32_t, int32_t, WT>(gpu_graph, *graph, cudaMemcpyHostToDevice);

  //Run the CPU implementation
  //TODO

  //Run the GPU implementation
  //Results buffer
  double * gpu_results, * gpu_results_d;
  gpu_results = (double*)malloc(sizeof(double*)*gpu_graph.number_of_edges);
  cudaMalloc(&gpu_results_d, gpu_graph.number_of_edges * sizeof(double));
  //This assume the graph's pointers are in GPU memory
  cugraph::jaccard<int32_t, int32_t, double>(gpu_graph, gpu_graph.edge_data, gpu_results_d);
  cudaMemcpy(gpu_results, gpu_results_d, sizeof(WT)* gpu_graph.number_of_edges, cudaMemcpyDeviceToHost);
  //Create a new results graph view, on the host side
  //Create a results graph (in which the weights are the jaccard similarity)
  GraphCSRView<int32_t, int32_t, WT> gpu_graph_results(graph->offsets, graph->indices, gpu_results, graph->number_of_vertices, graph->number_of_edges);
  //Compare results

  //Print formatted output data (i.e convert back to MTX)
  std::set<std::tuple<int32_t, int32_t, WT>> * gpu_results_mtx = CSRToMtx<WT>(gpu_graph_results);
  for (std::tuple<int32_t, int32_t, WT> edge : *gpu_results_mtx) {
    std::cout << std::get<0>(edge) << " " << std::get<1>(edge) << " " << std::get<2>(edge) << std::endl;
  } 
}
