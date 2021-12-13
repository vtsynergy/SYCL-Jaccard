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

#ifndef __STANDALONE_CSR_HPP__
#define __STANDALONE_CSR_HPP__
//Need this on the old hipSYCL compiler we're using
#define HIPSYCL_EXT_FP_ATOMICS true
#include <CL/sycl.hpp>

//From utilities/graph_utils.cuh
#define CUDA_MAX_BLOCKS 65535
#define CUDA_MAX_KERNEL_THREADS 256
#define US
#ifndef WEIGHT_TYPE
 #ifndef DISABLE_DP_WEIGHT
  #define WEIGHT_TYPE double
 #else
  #define WEIGHT_TYPE float
 #endif
#endif
/**
 * @brief       A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t   Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCSRView {
 public:
  using vertex_type = vertex_t;
  using edge_type   = edge_t;
  using weight_type = weight_t;

  cl::sycl::buffer<weight_t> edge_data;  ///< edge weight


  vertex_t number_of_vertices;
  edge_t number_of_edges;
  cl::sycl::buffer<edge_t> offsets{nullptr};    ///< CSR offsets
  cl::sycl::buffer<vertex_t> indices;  ///< CSR indices

  /**
   * @brief      Default constructor
   */
  GraphCSRView()
    : GraphCSRView<vertex_t, edge_t, weight_t>(nullptr, nullptr, nullptr, 0, 0)
  {
  }
  GraphCSRView(weight_t *edge_data, vertex_t number_of_vertices, edge_t number_of_edges)
    : edge_data(edge_data, number_of_edges),
      number_of_vertices(number_of_vertices),
      number_of_edges(number_of_edges)
  {
  }
  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSRView(edge_t *offsets,
                                vertex_t *indices,
                                weight_t *edge_data,
                                vertex_t number_of_vertices,
                                edge_t number_of_edges)
    : offsets{offsets, number_of_vertices+1},
      indices{indices, number_of_edges},
      edge_data(edge_data, number_of_edges),
      number_of_vertices(number_of_vertices),
      number_of_edges(number_of_edges)
  {
  }

  /**
  * @brief Use copy constructors to re-reference existing SYCL buffers
  */
  GraphCSRView(cl::sycl::buffer<edge_t> offsets,
                                cl::sycl::buffer<vertex_t> indices,
                                cl::sycl::buffer<weight_t> edge_data,
                                vertex_t number_of_vertices,
                                edge_t number_of_edges)
    : offsets{offsets},
      indices{indices},
      edge_data(edge_data),
      number_of_vertices(number_of_vertices),
      number_of_edges(number_of_edges)
  {
  }
};

#endif
