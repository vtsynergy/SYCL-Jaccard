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

// From utilties/graph_utils.cuh
// FIXME Revisit the barriers and fences and local storage with subgroups
// FIXME revisit with SYCL group algorithms
template <typename count_t, typename index_t, typename value_t>
__inline__ value_t
parallel_prefix_sum(cl::sycl::nd_item<2> const &tid_info, count_t n,
                    cl::sycl::accessor<index_t, 1, cl::sycl::access::mode::read> ind,
                    count_t ind_off, cl::sycl::accessor<value_t, 1, cl::sycl::access::mode::read> w,
                    cl::sycl::accessor<value_t, 1, cl::sycl::access::mode::read_write,
                                       cl::sycl::access::target::local>
                        shfl_temp) {
  count_t i, j, mn;
  value_t v, last;
  value_t sum = 0.0;
  bool valid;

  // Parallel prefix sum (using __shfl)
  mn = (((n + tid_info.get_local_range(1) - 1) / tid_info.get_local_range(1)) *
        tid_info.get_local_range(1)); // n in multiple of blockDim.x
  for (i = tid_info.get_local_id(1); i < mn; i += tid_info.get_local_range(1)) {
    // All threads (especially the last one) must always participate
    // in the shfl instruction, otherwise their sum will be undefined.
    // So, the loop stopping condition is based on multiple of n in loop increments,
    // so that all threads enter into the loop and inside we make sure we do not
    // read out of bounds memory checking for the actual size n.

    // check if the thread is valid
    valid = i < n;

    // Notice that the last thread is used to propagate the prefix sum.
    // For all the threads, in the first iteration the last is 0, in the following
    // iterations it is the value at the last thread of the previous iterations.

    // get the value of the last thread
    // FIXME: __shfl_sync
    // FIXME make sure everybody is here
    group_barrier(tid_info.get_group());
    // write your current sum
    // This is a 2D block, use a linear ID
    shfl_temp[tid_info.get_local_linear_id()] = sum;
    // FIXME make sure everybody has read from the top thread in the same Y-dimensional subgroup
    group_barrier(tid_info.get_group());
    last = shfl_temp[tid_info.get_local_range(1) - 1 +
                     (tid_info.get_local_range(1) * tid_info.get_local_id(0))];
    // Move forward
    // last = __shfl_sync(warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

    // if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[ind_off + i]] : 0.0;

    // do prefix sum (of size warpSize=blockDim.x =< 32)
    for (j = 1; j < tid_info.get_local_range(1); j *= 2) {
      // FIXME: __shfl_up_warp
      // FIXME make sure everybody is here
      // Write your current sum
      group_barrier(tid_info.get_group());
      shfl_temp[tid_info.get_local_linear_id()] = sum;
      // FIXME Force writes to finish
      // read from tid-j
      // Using the x-dimension local id for the conditional protects from overflows to other
      // Y-subgroups Using the local_linear_id for the read saves us having to offset by x_range *
      // y_id
      group_barrier(tid_info.get_group());
      if (tid_info.get_local_id(1) >= j) v = shfl_temp[tid_info.get_local_linear_id() - j];
      // FIXME Force reads to finish
      // v = __shfl_up_sync(warp_full_mask(), sum, j, blockDim.x);
      if (tid_info.get_local_id(1) >= j) sum += v;
    }
    // shift by last
    sum += last;
    // notice that no __threadfence or __syncthreads are needed in this implementation
  }
  // get the value of the last thread (to all threads)
  // FIXME: __shfl_sync
  // FIXME make sure everybody is here
  // write your current sum
  // This is a 2D block, use a linear ID
  group_barrier(tid_info.get_group());
  shfl_temp[tid_info.get_local_linear_id()] = sum;
  // FIXME make sure everybody has read from the top thread in the same Y-dimensional group
  group_barrier(tid_info.get_group());
  last = shfl_temp[tid_info.get_local_range(1) - 1 +
                   (tid_info.get_local_range(1) * tid_info.get_local_id(0))];
  // Move forward
  // last = __shfl_sync(warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

  return last;
}
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

namespace detail {
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
class Jaccard_RowSumKernel {
  vertex_t n;
  cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd;
  // FIXME, with std::conditional_t we should be able to simplify out some of the code paths in the
  // other weight-branching kernels
  #ifdef NEEDS_NULL_DEVICE_PTR
  std::conditional_t<weighted, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read>,
                     cl::sycl::device_ptr<std::nullptr_t>>
      v;
  #else
  std::conditional_t<weighted, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read>,
                     std::nullptr_t>
      v;
  #endif
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> work;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::local>
      shfl_temp;

public:
  Jaccard_RowSumKernel<true>(
      vertex_t n, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> v,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> work,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          shfl_temp)
      : n{n}, csrInd{csrInd}, csrPtr{csrPtr}, v{v}, work{work}, shfl_temp{shfl_temp} {
  }
  // When not using weights, we don't care about v
  Jaccard_RowSumKernel<false>(
      vertex_t n, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> work,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          shfl_temp)
      : n{n}, csrInd{csrInd}, csrPtr{csrPtr}, work{work}, shfl_temp{shfl_temp} {
  }
  // Volume of neighboors (*weight_s)
  const void operator()(cl::sycl::nd_item<2> tid_info) const {
    vertex_t row;
    edge_t start, end, length;
    weight_t sum;

    vertex_t row_start = tid_info.get_global_id(0);
    vertex_t row_incr = tid_info.get_global_range(0);
    for (row = row_start; row < n; row += row_incr) {
      start = csrPtr[row];
      end = csrPtr[row + 1];
      length = end - start;

      // compute row sums
      // Must be if constexpr so it doesn't try to evaluate v when it's a nullptr_t
      if constexpr (weighted) {
        sum = parallel_prefix_sum(tid_info, length, csrInd, start, v, shfl_temp);
        if (tid_info.get_local_id(1) == 0) work[row] = sum;
      } else {
        work[row] = static_cast<weight_t>(length);
      }
    }
  }
};

} // namespace detail
} // namespace sygraph

#endif
