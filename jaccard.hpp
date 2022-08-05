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

#ifdef INTEL_FPGA_EXT
  // Sometimes it's this path (2022.0.2)
  #include <sycl/ext/intel/fpga_extensions.hpp>
// Sometimes it's this path (2021.2.0)
//#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif
#include "standalone_algorithms.hpp"
#include "standalone_csr.hpp"
#include <iostream>

#ifndef SYCL_DEVICE_ONLY
  #define EMULATE_ATOMIC_ADD_FLOAT
  #define EMULATE_ATOMIC_ADD_DOUBLE
#endif

#ifdef ICX
  #define EMULATE_ATOMIC_ADD_DOUBLE
#endif

#ifndef __JACCARD_HPP__
  #define __JACCARD_HPP__

// Kernels are implemented as functors or lambdas in SYCL
template <typename T>
class FillKernel {
  cl::sycl::accessor<T, 1, cl::sycl::access::mode::discard_write> ptr;
  T value;
  size_t n;

public:
  FillKernel(cl::sycl::accessor<T, 1, cl::sycl::access::mode::discard_write> ptr, T value, size_t n)
      : ptr{ptr}, value{value}, n{n} {
  }
  const void operator()(cl::sycl::nd_item<1> tid_info) const;
};

namespace sygraph {
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
  // Must be marked external since main.cpp uses it
  SYCL_EXTERNAL const void operator()(cl::sycl::nd_item<2> tid_info) const;
};

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
class Jaccard_IsKernel {
  vertex_t n;
  cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd;
  // FIXME, what to do if this isn't present?
  #ifdef NEEDS_NULL_DEVICE_PTR
  std::conditional_t<weighted, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read>,
                     cl::sycl::device_ptr<std::nullptr_t>>
      v;
  #else
  std::conditional_t<weighted, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read>,
                     std::nullptr_t>
      v;
  #endif
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s;

public:
  Jaccard_IsKernel<true>(
      vertex_t n, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> v,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s)
      : n{n}, csrInd{csrInd}, csrPtr{csrPtr}, v{v}, work{work}, weight_i{weight_i}, weight_s{
                                                                                        weight_s} {
  }
  // When not using weights, we don't care about v
  Jaccard_IsKernel<false>(
      vertex_t n, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s)
      : n{n}, csrInd{csrInd}, csrPtr{csrPtr}, work{work}, weight_i{weight_i}, weight_s{weight_s} {
  }
  const void operator()(cl::sycl::nd_item<3> tid_info) const;
};

template <typename vertex_t, typename edge_t, typename weight_t>
class Jaccard_ec_scan {
  edge_t e;
  vertex_t n;
  cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read_write> dest_ind;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_j;

public:
  Jaccard_ec_scan(edge_t e, vertex_t n,
                  cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
                  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read_write> dest_ind,
                  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_j)
      : e{e}, n{n}, csrPtr{csrPtr}, dest_ind{dest_ind}, weight_j{weight_j} {
  }
  const void operator()(cl::sycl::nd_item<1> tid_info) const;
};

// Edge-centric-unweighted-kernel
template <typename vertex_t, typename edge_t, typename weight_t>
class Jaccard_ec_unweighted {
  edge_t e;
  vertex_t n;
  cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> dest_ind;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_j;

public:
  Jaccard_ec_unweighted(
      edge_t e, vertex_t n, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> dest_ind,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_j)
      : e{e}, n{n}, csrPtr{csrPtr}, csrInd{csrInd}, dest_ind{dest_ind}, weight_j{weight_j} {
  }
  const void operator()(cl::sycl::nd_item<1> tid_info) const;
};

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Using list of node pairs
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
class Jaccard_IsPairsKernel {
  edge_t num_pairs;
  cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> first_pair;
  cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> second_pair;
  // FIXME, what to do if this isn't present?
  #ifdef NEEDS_NULL_DEVICE_PTR
  std::conditional_t<weighted, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read>,
                     cl::sycl::device_ptr<std::nullptr_t>>
      v;
  #else
  std::conditional_t<weighted, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read>,
                     std::nullptr_t>
      v;
  #endif
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s;

public:
  Jaccard_IsPairsKernel<true>(
      edge_t num_pairs, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> first_pair,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> second_pair,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> v,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s)
      : num_pairs{num_pairs}, csrInd{csrInd}, csrPtr{csrPtr}, first_pair{first_pair},
        second_pair{second_pair}, v{v}, work{work}, weight_i{weight_i}, weight_s{weight_s} {
  }
  // When not using weights, we don't care about v
  Jaccard_IsPairsKernel<false>(
      edge_t num_pairs, cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> first_pair,
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> second_pair,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i,
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s)
      : num_pairs{num_pairs}, csrInd{csrInd}, csrPtr{csrPtr}, first_pair{first_pair},
        second_pair{second_pair}, work{work}, weight_i{weight_i}, weight_s{weight_s} {
  }
  const void operator()(cl::sycl::nd_item<3> tid_info) const;
};

// Jaccard  weights (*weight)
template <typename vertex_t, typename edge_t, typename weight_t>
class Jaccard_JwKernel {
  edge_t e;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_i;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_s;
  cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_j;

public:
  Jaccard_JwKernel(edge_t e, cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_i,
                   cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_s,
                   cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_j)
      : e{e}, weight_i{weight_i}, weight_s{weight_s}, weight_j{weight_j} {
  }
  const void operator()(cl::sycl::nd_item<1> tid_info) const;
};

} // namespace detail
} // namespace sygraph

#endif //__JACCARD_HPP__
