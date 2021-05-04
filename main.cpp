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
#ifdef ROCPROFILE
#include </opt/rocm/rocprofiler/include/rocprofiler.h>
#endif

#ifndef WEIGHT_TYPE
#define WEIGHT_TYPE double
#endif


int main(int argc, char * argv[]) {

  //Open the specified file for reading
  //TODO arg bounds safety
  std::ifstream fileIn(argv[1]);
  std::ofstream fileOut(argv[2]);
  bool isWeighted;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>* mtx_graph = readMtx<WEIGHT_TYPE>(fileIn, &isWeighted);
  fileIn.close();

  //Go ahead and iterate over the SYCL devices and pick one if they've specified a device number
#ifdef EVENT_PROFILE
  cl::sycl::queue q{cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()}};
#else
  cl::sycl::queue q;
#endif
  std::vector<cl::sycl::device> all_devices;
  if (argc >= 4) {
    int count = 0;
    std::vector<cl::sycl::platform> plats = cl::sycl::platform::get_platforms();
    for (cl::sycl::platform plat : plats) {
      std::vector<cl::sycl::device> devs = plat.get_devices();
      for (cl::sycl::device dev : devs) {
        all_devices.push_back(dev);
        std::cerr << "SYCL Device [" << count << "]: " << dev.get_info<cl::sycl::info::device::name>() << std::endl;
        if (count == atoi(argv[3])) {
#ifdef EVENT_PROFILE
          q = cl::sycl::queue{dev, cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()}};
#else
          q = cl::sycl::queue{dev};
#endif
        }
        ++count;
      }
    }
  }

  //Convert it to a CSR
  //FIXME this should read directly to buffers with writeback pointers
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> * graph = mtxSetToCSR(*mtx_graph);

  //Run the CPU implementation
  //TODO

  //Run the GPU implementation
  //Results buffer
  WEIGHT_TYPE * gpu_results;
  gpu_results = (WEIGHT_TYPE*)malloc(sizeof(WEIGHT_TYPE*)*(graph->number_of_edges));
  //Pre-declare our results MTX set, since we'll need it after the device buffer scope
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> * gpu_results_mtx;
  { //Results buffer scope, for implicit copyback
    cl::sycl::buffer<WEIGHT_TYPE> results_buf(gpu_results, (graph->number_of_edges));
    sygraph::jaccard<int32_t, int32_t, WEIGHT_TYPE>(*graph, graph->edge_data, results_buf, q);
  
    //Create a new results graph view, using the a copy constructor to re-reference the results buffer
    GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> gpu_graph_results(graph->offsets, graph->indices, results_buf, graph->number_of_vertices, graph->number_of_edges);

    //Optionally explicit copy-back before the MTX conversion uses the data on the host
    #ifdef EXPLICIT_COPY
    cl::sycl::queue q = cl::sycl::queue(cl::sycl::cpu_selector());
    q.submit([&](cl::sycl::handler &cgh) {
      auto results_acc = results_buf.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>(graph->number_of_edges));
      cgh.copy(results_acc, gpu_results);
    });
    #endif

    //convert back to MTX (implicit copyback if not done explicitly)
    gpu_results_mtx = CSRToMtx<WEIGHT_TYPE>(gpu_graph_results, true);
  } //End Results buffer scope
  for (std::tuple<int32_t, int32_t, WEIGHT_TYPE> edge : *gpu_results_mtx) {
    fileOut << std::get<0>(edge) << " " << std::get<1>(edge) << " " << std::get<2>(edge) << std::endl;
  } 
  fileOut.close();
}
