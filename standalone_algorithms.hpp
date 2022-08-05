/*
 * Original CUDA Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SYCL translation and edge-centric components Copyright (c) 2021-2022, Virginia Tech
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
 *
 * See NOTICE.md for detailed description of derivative portions and their origins
 */

#include "standalone_csr.hpp"
#ifdef INTEL_FPGA_EXT
  // Sometimes it's this path (2022.0.2)
  #include <sycl/ext/intel/fpga_extensions.hpp>
// Sometimes it's this path (2021.2.0)
//#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

#ifndef __STANDALONE_ALGORITHMS_HPP__
  #define __STANDALONE_ALGORITHMS_HPP__

  // Custom fixes for SYCL version inconsistencies
  #ifdef SYCL_1_2_1
    #define group_barrier(foo) tid_info.barrier()
  //#define group_barrier(foo) tid_info.barrier(cl::sycl::access::fence_space::local_space)
  #endif

  #ifdef EVENT_PROFILE
    #define wait_and_print(prefix, name)                                                           \
      {                                                                                            \
        prefix##_event.wait();                                                                     \
        auto end =                                                                                 \
            prefix##_event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();     \
        auto start =                                                                               \
            prefix##_event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();   \
        std::cerr << name << " kernel elapsed time: " << (end - start) << " (ns)" << std::endl;    \
      }
  #endif // EVENT_PROFILE

namespace sygraph {

/**
 * @brief     Compute jaccard similarity coefficient for all vertices
 *
 * Computes the Jaccard similarity coefficient for every pair of vertices in the graph
 * which are connected by an edge.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      device pointer to input vertex weights for weighted Jaccard
 * for unweighted Jaccard.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <bool edge_centric, typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> &graph, cl::sycl::buffer<WT> &weights,
             cl::sycl::buffer<WT> &result, cl::sycl::queue &q);

/**
 * @brief     Compute unweighted jaccard similarity coefficient for all vertices
 *
 * Computes the Jaccard similarity coefficient for every pair of vertices in the graph
 * which are connected by an edge.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * for unweighted Jaccard.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <bool edge_centric, typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> &graph, cl::sycl::buffer<WT> &result, cl::sycl::queue &q);

/**
 * @brief     Compute jaccard similarity coefficient for selected vertex pairs
 *
 * Computes the Jaccard similarity coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 *
 * @tparam edge_centric    Whether to use the edge-centric implementation (true) or vertex-centric
 * (false)
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      The input vertex weights for weighted Jaccard
 * @param[in] num_pairs    The number of vertex ID pairs specified
 * @param[in] first        Device pointer to first vertex ID of each pair
 * @param[in] second       Device pointer to second vertex ID of each pair
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> &graph, cl::sycl::buffer<WT> &weights, ET num_pairs,
                  cl::sycl::buffer<VT> &first, cl::sycl::buffer<VT> &second,
                  cl::sycl::buffer<WT> &result, cl::sycl::queue &q);

/**
 * @brief     Compute unweighted jaccard similarity coefficient for selected vertex pairs
 *
 * Computes the Jaccard similarity coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] num_pairs    The number of vertex ID pairs specified
 * @param[in] first        Device pointer to first vertex ID of each pair
 * @param[in] second       Device pointer to second vertex ID of each pair
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> &graph, ET num_pairs, cl::sycl::buffer<VT> &first,
                  cl::sycl::buffer<VT> &second, cl::sycl::buffer<WT> &result, cl::sycl::queue &q);

} // namespace sygraph

#endif
