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

#ifdef ICX
#define min(a, b) std::min((size_t)a, (size_t)b)
#endif

#ifndef WEIGHT_TYPE
 #ifndef DISABLE_DP_WEIGHT
  #define WEIGHT_TYPE double
 #else
  #define WEIGHT_TYPE float
 #endif
#endif


int main(int argc, char * argv[]) {

  //Open the specified file for reading
  //TODO arg bounds safety
  std::ifstream fileIn(argv[1]);
  std::ofstream fileOut(argv[2]);
  bool isWeighted;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>>* mtx_graph = readMtx<WEIGHT_TYPE>(fileIn, &isWeighted);
  fileIn.close();
  //We can't override weighting until here, or else the MTX will get confused about tokens per line if the file and override disagree on the presence of weight values.
  //Undef=defer to file, 1=Weighted, 0=Unweighted
  char * weighted_override = std::getenv("JACCARD_FORCE_WEIGHTS");
  if (weighted_override != NULL) {
    #ifdef DEBUG
      std::cerr << "Force Override of Weighted computation, current value is: " << isWeighted << " Override set to: " << weighted_override << std::endl;
    #endif
    if (std::strcmp(weighted_override, "1") == 0) isWeighted = true;
    if (std::strcmp(weighted_override, "0") == 0) isWeighted = false;
  }

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

    //FIXME Preprocess edge weights into vertex weights (simply sum them for now) IFF weighted
    //TODO Create the pseudo csrInd buffer
    cl::sycl::buffer<int32_t> presumInd(graph->number_of_edges);
    //TODO populate the pseudo buffer with a named lambda (that will be converted to a cuda kernel in a cherry-pick)
    cl::sycl::event presum_event = q.submit([&](cl::sycl::handler &cgh){
      cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::discard_write> presumInd_acc = presumInd.get_access<cl::sycl::access::mode::discard_write>(cgh, cl::sycl::range<1>{(size_t)graph->number_of_edges});
      cgh.parallel_for<class presumInd_kernel>(cl::sycl::range<1>{(size_t)graph->number_of_edges}, [=](cl::sycl::id<1> tid) {
        presumInd_acc[tid]=tid;
      });
    });
#ifdef DEBUG_2
    q.wait();
    std::cout << "DEBUG: Post-PreSum weight index vector of " << graph->number_of_edges << " elements" << std::endl;
    {
      auto debug_acc = presumInd.get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>{(size_t)graph->number_of_edges});
      for (int i = 0; i < graph->number_of_edges; i++) {
        std::cout << debug_acc[i] << std::endl;
      }
    }
#endif //DEBUG_2
    cl::sycl::range<2> vertSum_local{1, 32};
    //cl::sycl::range<2> sum_global{1 * sum_local.get(0), min((n + sum_local.get(1) -1) / sum_local.get(1), vertex_t{CUDA_MAX_BLOCKS}) * sum_local.get(1)};
    cl::sycl::range<2> vertSum_global{min((graph->number_of_vertices + vertSum_local.get(0) -1) / vertSum_local.get(0), int32_t{CUDA_MAX_BLOCKS}) * vertSum_local.get(0), vertSum_local.get(1)};

    // launch kernel
    //FIXME: SYCL-lambda the kernel launch
    //csrPtr should be reusable, that's just the start and end indicies for each row
    //csrInd is not going to be reusable. it needs to be an index into the weight structure, which should effectively just be [0, num_edges) in order
    //Work should be a new buffer of length num_verts
    //FIXME the vertex weight buffer may need to be a shareable pointer
    cl::sycl::buffer<WEIGHT_TYPE> vertWeights(graph->number_of_vertices);
    cl::sycl::event vertSum_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::read> csrPtr_acc = graph->offsets.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>{(size_t)graph->number_of_vertices+1});
      cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::read> csrInd_acc = presumInd.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>{(size_t)graph->number_of_edges});
      cl::sycl::accessor<WEIGHT_TYPE, 1, cl::sycl::access::mode::discard_write> work_acc = vertWeights.get_access<cl::sycl::access::mode::discard_write>(cgh, cl::sycl::range<1>{(size_t)graph->number_of_vertices});
      cl::sycl::accessor<WEIGHT_TYPE, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> shfl_temp(vertSum_local.get(0) * vertSum_local.get(1), cgh);
      cl::sycl::accessor<WEIGHT_TYPE, 1, cl::sycl::access::mode::read> weight_in_acc = graph->edge_data.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>{(size_t)graph->number_of_edges});
      sygraph::detail::Jaccard_RowSumKernel<true, int32_t, int32_t, WEIGHT_TYPE> vertSum_kernel(graph->number_of_vertices, csrPtr_acc, csrInd_acc, weight_in_acc, work_acc, shfl_temp);
      cgh.parallel_for(cl::sycl::nd_range<2>{vertSum_global, vertSum_local}, vertSum_kernel);
    });
#ifdef DEBUG_2
    q.wait();
    std::cout << "DEBUG: Post-VertSum weight vector of " << graph->number_of_vertices << " elements" << std::endl;
    {
      auto debug_acc = vertWeights.get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>{(size_t)graph->number_of_edges});
      for (int i = 0; i < graph->number_of_vertices; i++) {
        std::cout << debug_acc[i] << std::endl;
      }
    }
#endif //DEBUG_2
#ifdef EVENT_PROFILE
    wait_and_print(presum, "PreSum")
    wait_and_print(vertSum, "VertexSum")
#endif //EVENT_PROFILE
    

    //FIXME: adjust the data structure to use the new vertex weights, and existing column and row data
    //FIXME: refactor to handle unweighted MTX using native unweighted kernels, rather than 1 weights
    //sygraph::jaccard<int32_t, int32_t, WEIGHT_TYPE>(*graph, graph->edge_data, results_buf, q);
    sygraph::jaccard<int32_t, int32_t, WEIGHT_TYPE>(*graph, vertWeights, results_buf, q);
  
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
