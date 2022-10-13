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
#include "readMtxToCSR.hpp" //implicitly includes standalone_csr.hpp
#include "standalone_algorithms.hpp"
#include "standalone_csr.hpp"
#include "sycl_exceptions.hpp"
#include <CL/sycl.hpp>
#include <cstring>
#include <iostream>
#include <vector>
#ifdef ROCPROFILE
  #include </opt/rocm/rocprofiler/include/rocprofiler.h>
#endif

#ifndef WEIGHT_TYPE
  #ifndef DISABLE_DP_WEIGHT
    #define WEIGHT_TYPE double
  #else
    #define WEIGHT_TYPE float
  #endif
#endif


typedef enum {
  noChoice = 0,
  isForced = 1,
  ec_coarse = 2,
  vc_coarse = 4,
  undefined = -1
} implSelect;

implSelect selectImplementation() {
  implSelect retVal = noChoice;
  char *force_ec = std::getenv("JACCARD_FORCE_EDGE_CENTRIC");
  if (force_ec != nullptr) {
    std::cerr << "FORCE Edge-Centric Implementation" << std::endl;
    retVal = (implSelect)(ec_coarse | isForced);
  }
  char *force_vc = std::getenv("JACCARD_FORCE_VERTEX_CENTRIC");
  if (force_vc != nullptr) {
    std::cerr << "FORCE Vertex-Centric Implementation" << std::endl;
    retVal = (implSelect)(vc_coarse | isForced);
  }
  return retVal;
}

int main(int argc, char *argv[]) {

  // Open the specified file for reading
  // TODO arg bounds safety
  std::ifstream fileIn;
  std::ofstream fileOut;
  graphFileType inType, outType, working;
  setupInFile(argv[1], fileIn, inType);
  setupOutFile(argv[2], fileOut, outType);
  bool keepReverseEdges = true;
  bool isWeighted = false, isDirected = false, hasReverseEdges = false, isZeroIndexed = false;
  GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *graph;
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *mtx_graph;
  if (inType == mtx) { // IF extension is mtx, use the string r/w
    working = mtx;
    mtx_graph = fileToMTXSet<int32_t, int32_t, WEIGHT_TYPE>(fileIn, &isWeighted, &isDirected);
  } else if (inType == csr) { // IF extension is csr, use binary r/w
    working = csr;
    CSRFileHeader header;
    graph = static_cast<GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> *>(FileToCSR(fileIn, &header));
    isWeighted = header.flags.isWeighted;
    isDirected = header.flags.isDirected;
    hasReverseEdges = header.flags.hasReverseEdges;
    isZeroIndexed = header.flags.isZeroIndexed;
    if (header.flags.isVertexT64 || header.flags.isEdgeT64 ||
        (header.flags.isWeighted &&
         ((header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, double>::value) ||
          (!header.flags.isWeightT64 && !std::is_same<WEIGHT_TYPE, float>::value)))) {
      std::cerr << "Binary CSR Input Header does not match required data types" << std::endl;
      exit(3);
    }
  } else {
    std::cerr << "InputGraphType is" << inType << std::endl;
  }
  fileIn.close();
  // MTX needs to have the reverse edges generated
  if (!isDirected && !hasReverseEdges) {
    if (working == csr) {
      // Switch it to MTX to reverse them
      mtx_graph = CSRToMtx(*graph, isZeroIndexed, isWeighted);
      working = mtx;
      delete graph; // Don't need to maintain it as a CSR, as a new one will be generated later
    } else if (working != mtx) {
      // Future formats
    }
    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *reverse = invertDirection(*mtx_graph);
    mtx_graph->insert(reverse->begin(), reverse->end());
    hasReverseEdges = true;
    delete reverse;
  }
  // Convert it to a CSR
  if (working == mtx) {
    graph = mtxSetToCSR(*mtx_graph);
    working = csr;
    delete mtx_graph;
  } else if (working != csr) {
    // Future formats
  }

  // Add an environment variable to dump CSR for both input and output as a sideeffect of an
  // MTX-file run
  char *dump_csr = std::getenv("JACCARD_IN_CSR_DUMP_FILEPATH");
  if (dump_csr != nullptr) {
#ifdef DEBUG
    std::cerr << "Requested CSR Dump of input file \"" << argv[1] << "\" to \"" << dump_csr << "\""
              << std::endl;
#endif
    std::ofstream csrDumpFile(dump_csr,
                              std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    CSRToFile(csrDumpFile, (*graph), false, isWeighted);
    csrDumpFile.close();
    dump_csr = nullptr; // Reset it incase there is no output dump
  }
  // We can't override weighting until here, or else the MTX will get confused about tokens per line
  // if the file and override disagree on the presence of weight values. Undef=defer to file,
  // 1=Weighted, 0=Unweighted
  char *weighted_override = std::getenv("JACCARD_FORCE_WEIGHTED");
  if (weighted_override != NULL) {
#ifdef DEBUG
    std::cerr << "Force Override of Weighted computation, current value is: " << isWeighted
              << " Override set to: " << weighted_override << std::endl;
#endif
    bool hadWeights = isWeighted;
    if (std::strcmp(weighted_override, "1") == 0) isWeighted = true;
    if (std::strcmp(weighted_override, "0") == 0) isWeighted = false;
    // If the graph has null weights vector, we have to provide it something if it's being forced on
    if (isWeighted && !hadWeights) {
      std::vector<WEIGHT_TYPE> *forcedWeights =
          new std::vector<WEIGHT_TYPE>(graph->number_of_edges, WEIGHT_TYPE{1.0});
      graph->edge_data =
          cl::sycl::buffer<WEIGHT_TYPE>(forcedWeights->data(), graph->number_of_edges);
    }
  }

  // Go ahead and iterate over the SYCL devices and pick one if they've specified a device number
#ifdef EVENT_PROFILE
  cl::sycl::queue q{sycl_exception_handler,
                    cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()}};
#else
  cl::sycl::queue q{sycl_exception_handler};
#endif
  std::vector<cl::sycl::device> all_devices;
  if (argc >= 4) {
    int count = 0;
    std::vector<cl::sycl::platform> plats = cl::sycl::platform::get_platforms();
    for (cl::sycl::platform plat : plats) {
      std::vector<cl::sycl::device> devs = plat.get_devices();
      for (cl::sycl::device dev : devs) {
        all_devices.push_back(dev);
        std::cerr << "SYCL Device [" << count
                  << "]: " << dev.get_info<cl::sycl::info::device::name>() << std::endl;
        if (count == atoi(argv[3])) {
#ifdef EVENT_PROFILE
          q = cl::sycl::queue{
              dev, sycl_exception_handler,
              cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()}};
#else
          q = cl::sycl::queue{dev, sycl_exception_handler};
#endif
        }
        ++count;
      }
    }
  }
  std::cout << "Running on " << q.get_device().get_info<cl::sycl::info::device::name>() << "\n";


  // Run the CPU implementation
  // TODO

  // Run the GPU implementation
  // Results buffer
  WEIGHT_TYPE *gpu_results;
  gpu_results = (WEIGHT_TYPE *)malloc(sizeof(WEIGHT_TYPE *) * (graph->number_of_edges));
  ///< Pre-declare our results MTX set, since we'll need it after the device buffer scope
  std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *gpu_results_mtx;
  { // Results buffer scope, for implicit copyback
    cl::sycl::buffer<WEIGHT_TYPE> results_buf(gpu_results, (graph->number_of_edges));

    // Pick an implementation to use
    // TODO: Automatic selection based on graph properties
    implSelect implementation = selectImplementation();

    if (implementation & ec_coarse) {
      sygraph::jaccard<true, int32_t, int32_t, WEIGHT_TYPE>(*graph, results_buf, q);

    } else if (implementation & vc_coarse) {
#ifndef DISABLE_WEIGHTED
      if (isWeighted) {
        // Create the pseudo csrInd buffer
        cl::sycl::buffer<int32_t> presumInd(graph->number_of_edges);
        // populate the pseudo buffer with a named lambda
        try {
          cl::sycl::event presum_event = q.submit([&](cl::sycl::handler &cgh) {
            cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::discard_write> presumInd_acc =
                presumInd.get_access<cl::sycl::access::mode::discard_write>(
                    cgh, cl::sycl::range<1>{(size_t)graph->number_of_edges});
            cgh.parallel_for<class presumInd_kernel>(
                cl::sycl::range<1>{(size_t)graph->number_of_edges},
                [=](cl::sycl::id<1> tid) { presumInd_acc[tid] = tid.get(0); });
          });
  #ifdef DEBUG_2
          q.wait();
          std::cout << "DEBUG: Post-PreSum weight index vector of " << graph->number_of_edges
                    << " elements" << std::endl;
          {
            auto debug_acc = presumInd.get_access<cl::sycl::access::mode::read>(
                cl::sycl::range<1>{(size_t)graph->number_of_edges});
            for (int i = 0; i < graph->number_of_edges; i++) {
              std::cout << debug_acc[i] << std::endl;
            }
          }
  #endif // DEBUG_2
  #ifdef EVENT_PROFILE
          wait_and_print(presum, "PreSum")
  #endif // EVENT_PROFILE
        } catch (sycl::exception e) {
          std::cerr << "SYCL Exception during Vertex Weight Pre-Sum\n\t" << e.what() << std::endl;
        }
        cl::sycl::range<2> vertSum_local{1, 32};
        cl::sycl::range<2> vertSum_global{
            std::min((size_t)(graph->number_of_vertices + vertSum_local.get(0) - 1) /
                         vertSum_local.get(0),
                     size_t{CUDA_MAX_BLOCKS}) *
                vertSum_local.get(0),
            vertSum_local.get(1)};

        // Reuse the existing RowSum kernel with our pseudo indices to create synthetic vertex
        // weights (sum of edges)
        // FIXME the vertex weight buffer may need to be a shareable pointer
        cl::sycl::buffer<WEIGHT_TYPE> vertWeights(graph->number_of_vertices);
        try {
          cl::sycl::event vertSum_event = q.submit([&](cl::sycl::handler &cgh) {
            cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::read> csrPtr_acc =
                graph->offsets.get_access<cl::sycl::access::mode::read>(
                    cgh, cl::sycl::range<1>{(size_t)graph->number_of_vertices + 1});
            cl::sycl::accessor<int32_t, 1, cl::sycl::access::mode::read> csrInd_acc =
                presumInd.get_access<cl::sycl::access::mode::read>(
                    cgh, cl::sycl::range<1>{(size_t)graph->number_of_edges});
            cl::sycl::accessor<WEIGHT_TYPE, 1, cl::sycl::access::mode::discard_write> work_acc =
                vertWeights.get_access<cl::sycl::access::mode::discard_write>(
                    cgh, cl::sycl::range<1>{(size_t)graph->number_of_vertices});
            cl::sycl::accessor<WEIGHT_TYPE, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local>
                shfl_temp(vertSum_local.get(0) * vertSum_local.get(1), cgh);
            cl::sycl::accessor<WEIGHT_TYPE, 1, cl::sycl::access::mode::read> weight_in_acc =
                graph->edge_data.get_access<cl::sycl::access::mode::read>(
                    cgh, cl::sycl::range<1>{(size_t)graph->number_of_edges});
            sygraph::detail::Jaccard_RowSumKernel<true, int32_t, int32_t, WEIGHT_TYPE>
                vertSum_kernel(graph->number_of_vertices, csrPtr_acc, csrInd_acc, weight_in_acc,
                               work_acc, shfl_temp);
            cgh.parallel_for(cl::sycl::nd_range<2>{vertSum_global, vertSum_local}, vertSum_kernel);
          });
  #ifdef DEBUG_2
          q.wait();
          std::cout << "DEBUG: Post-VertSum weight vector of " << graph->number_of_vertices
                    << " elements" << std::endl;
          {
            auto debug_acc = vertWeights.get_access<cl::sycl::access::mode::read>(
                cl::sycl::range<1>{(size_t)graph->number_of_vertices});
            for (int i = 0; i < graph->number_of_vertices; i++) {
              std::cout << debug_acc[i] << std::endl;
            }
          }
  #endif // DEBUG_2
  #ifdef EVENT_PROFILE
          wait_and_print(vertSum, "EdgeSum")
  #endif // EVENT_PROFILE
        } catch (sycl::exception e) {
          std::cerr << "SYCL Exception during Vertex Weight Edge-Sum\n\t" << e.what() << std::endl;
        }

        sygraph::jaccard<false, int32_t, int32_t, WEIGHT_TYPE>(*graph, vertWeights, results_buf, q);
      } else { // Unweighted
#endif         // DISABLE_WEIGHTED
        sygraph::jaccard<false, int32_t, int32_t, WEIGHT_TYPE>(*graph, results_buf, q);
#ifndef DISABLE_WEIGHTED
      }
#endif // DISABLE_WEIGHTED
    }
    // Create a new results graph view, using the a copy constructor to re-reference the results
    // buffer
    GraphCSRView<int32_t, int32_t, WEIGHT_TYPE> gpu_graph_results(
        graph->offsets, graph->indices, results_buf, graph->number_of_vertices,
        graph->number_of_edges);

// Optionally explicit copy-back before the MTX conversion uses the data on the host
#ifdef EXPLICIT_COPY
    cl::sycl::queue q = cl::sycl::queue(cl::sycl::cpu_selector());
    q.submit([&](cl::sycl::handler &cgh) {
      auto results_acc = results_buf.get_access<cl::sycl::access::mode::read>(
          cgh, cl::sycl::range<1>(graph->number_of_edges));
      cgh.copy(results_acc, gpu_results);
    });
#endif

    // Don't need the inputs anymore
    // Need to get the host pointers from the buffers in some way
    delete graph;
    // Set isWeighted to true to retain the scors
    isWeighted = true;
    graph = nullptr;
    // Only remove edges if the formats disagree
    std::set<std::tuple<int32_t, int32_t, WEIGHT_TYPE>> *gpu_results_mtx = nullptr;
    if (hasReverseEdges && (!keepReverseEdges || (outType == mtx && !isDirected))) {
      gpu_results_mtx = CSRToMtx(gpu_graph_results, true, isWeighted);
      removeReverseEdges(*gpu_results_mtx);
      hasReverseEdges = false;
      working = mtx;
    }
    // Add an environment variable to dump CSR for output as a sideeffect of an MTX-file run
    dump_csr = std::getenv("JACCARD_OUT_CSR_DUMP_FILEPATH");
    if (dump_csr != nullptr) {
      if (working == mtx) {
        // The only reason it would be MTX at this point is if we had to delete reverse edges
        graph = mtxSetToCSR(*gpu_results_mtx, true, false);
        gpu_graph_results = *graph;
        delete graph; // Just holds pointers, don't need them anymore
        working = csr;
      }
#ifdef DEBUG
      std::cerr << "Requested CSR Dump of output file \"" << argv[2] << "\" to \"" << dump_csr
                << "\"" << std::endl;
#endif
      std::ofstream csrDumpFile(dump_csr,
                                std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
      CSRToFile(csrDumpFile, gpu_graph_results, isZeroIndexed, isWeighted, isDirected,
                keepReverseEdges);
      csrDumpFile.close();
    }
    if (outType == mtx) { // IF extension is mtx, use the string r/w
      // convert back to MTX (implicit copyback if not done explicitly)
      if (working == csr &&
          gpu_results_mtx ==
              nullptr) { // We have not had to generate the MTX yet (there were no reverse edges)
        gpu_results_mtx =
            CSRToMtx<int32_t, int32_t, WEIGHT_TYPE>(gpu_graph_results, true, isWeighted);
      }
      mtxSetToFile(fileOut, *gpu_results_mtx, gpu_graph_results.number_of_vertices,
                   gpu_graph_results.number_of_edges, isWeighted, isDirected);
    } else if (outType == csr) { // IF extension is csr, use binary r/w
      if (working ==
          mtx) { // We had to delete some reverse edges and haven't flipped back to CSR yet
        graph = mtxSetToCSR(*gpu_results_mtx);
        gpu_graph_results = *graph;
        delete graph; // Just holds pointers, don't need them anymore
        working = csr;
      }
      CSRToFile(fileOut, gpu_graph_results, isZeroIndexed, isWeighted, isDirected, hasReverseEdges);
    } // No else, but extensible if we need different outputs eventually
    fileOut.close();
    // Cleanup outputs. CSR is canonical form, so only delete pointers from there
    if (gpu_results_mtx != nullptr) {
      delete gpu_results_mtx;
      gpu_results_mtx = nullptr;
    }
  } // End Results buffer scope
  free(gpu_results);
}
