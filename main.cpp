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
#include <CL/sycl.hpp>
#include "readMtxToCSR.hpp" //implicitly includes standalone_csr.hpp
#include "standalone_algorithms.hpp"
#include "standalone_csr.hpp"


#ifndef WT
#define WT double
#endif


int main(int argc, char * argv[]) {

  //Open the specified file for reading
  //TODO arg bounds safety
  std::ifstream fileIn(argv[1]);
  bool isWeighted;
  std::set<std::tuple<int32_t, int32_t, WT>>* mtx_graph = readMtx<WT>(fileIn, &isWeighted);
  fileIn.close();

  //Convert it to a CSR
  //FIXME this should read directly to buffers with writeback pointers
  GraphCSRView<int32_t, int32_t, WT> * graph = mtxSetToCSR(*mtx_graph);

  //Run the CPU implementation
  //TODO

  //Run the GPU implementation
  //Results buffer
  WT * gpu_results;
  gpu_results = (WT*)malloc(sizeof(WT*)*(graph->number_of_edges));
  //Pre-declare our results MTX set, since we'll need it after the device buffer scope
  std::set<std::tuple<int32_t, int32_t, WT>> * gpu_results_mtx;
  { //Results buffer scope, for implicit copyback
    cl::sycl::buffer<WT> results_buf(gpu_results, (graph->number_of_edges));
    sygraph::jaccard<int32_t, int32_t, double>(*graph, graph->edge_data, results_buf);
  
    //Create a new results graph view, using the a copy constructor to re-reference the results buffer
    GraphCSRView<int32_t, int32_t, WT> gpu_graph_results(graph->offsets, graph->indices, results_buf, graph->number_of_vertices, graph->number_of_edges);

    //Optionally explicit copy-back before the MTX conversion uses the data on the host
    #ifdef EXPLICIT_COPY
    cl::sycl::queue q = cl::sycl::queue(cl::sycl::cpu_selector());
    q.submit([&](cl::sycl::handler &cgh) {
      auto results_acc = results_buf.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(graph->number_of_edges));
      cgh.copy(results_acc, gpu_results);
    });
    #endif

    //convert back to MTX (implicit copyback if not done explicitly)
    gpu_results_mtx = CSRToMtx<WT>(gpu_graph_results, true);
  } //End Results buffer scope
  for (std::tuple<int32_t, int32_t, WT> edge : *gpu_results_mtx) {
    std::cout << std::get<0>(edge) << " " << std::get<1>(edge) << " " << std::get<2>(edge) << std::endl;
  } 
}
