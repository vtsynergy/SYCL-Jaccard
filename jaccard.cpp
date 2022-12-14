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
/** ---------------------------------------------------------------------------*
 * @brief The sygraph Jaccard core functionality
 *
 * @file jaccard.cpp
 * ---------------------------------------------------------------------------**/
#ifndef STANDALONE
  #include "graph.hpp"
  #include "utilities/graph_utils.cuh"
  #include <rmm/thrust_rmm_allocator.h>
  #include <utilities/error.hpp>
#else
  #ifdef INTEL_FPGA_EXT
    // Sometimes it's this path (2022.0.2)
    #include <sycl/ext/intel/fpga_extensions.hpp>
  // Sometimes it's this path (2021.2.0)
  //#include <CL/sycl/INTEL/fpga_extensions.hpp>
  #endif
  #include "jaccard.hpp"
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

// From RAFT at commit 048063dc08
constexpr inline int warp_size() {
  return 32;
}

constexpr inline unsigned int warp_full_mask() {
  return 0xffffffff;
}

// Kernels are implemented as functors or lambdas in SYCL
// Custom Thrust simplifications
template <typename T>
const void FillKernel<T>::operator()(cl::sycl::nd_item<1> tid_info) const {
  // equivalent to: idx = threadIdx.x + blockIdx.x*blockIdx.x;
  size_t idx = tid_info.get_global_id(0);
  // equivalent to: incr = blockDim.x*gridDim.x;
  size_t incr = tid_info.get_global_range(0);
  for (; idx < n; idx += incr) {
    ptr[idx] = value;
  }
}

template <typename T>
const cl::sycl::event FillKernel<T>::invoke(size_t n, cl::sycl::buffer<T> &x, T value,
                                            cl::sycl::queue &q) {
  // FIXME: De-CUDA the MAX_KERNEL_THREADS and MAX_BLOCKS defines
  size_t block = std::min((size_t)n, (size_t)CUDA_MAX_KERNEL_THREADS);
  size_t grid = std::min((size_t)(n / block) + ((n % block) ? 1 : 0), (size_t)CUDA_MAX_BLOCKS);
  // TODO, do we need to emulate their stream behavior?
  cl::sycl::event ret_event;
  try {
    ret_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::discard_write> x_acc =
          x.template get_access<cl::sycl::access::mode::discard_write>(cgh, cl::sycl::range<1>(n));
      FillKernel fill_kern(x_acc, value, n);
      cgh.parallel_for(
          cl::sycl::nd_range<1>{cl::sycl::range<1>{grid * block}, cl::sycl::range<1>{block}},
          fill_kern);
    });
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during Fill enqueue\n\t" << e.what() << std::endl;
  }
  return ret_event;
}

  #ifdef EMULATE_ATOMIC_ADD_FLOAT
// Inspired by the older CUDA C Programming Guide
float myAtomicAdd(cl::sycl::atomic<uint32_t> &address, float val) {
  uint32_t old = address.load();
  // uint64_t atomic_load
  bool success = false;
  do {
    // old = atomicCAS(address_as_ull, assumed,
    //                __double_as_longlong(val +
    //                       __longlong_as_double(assumed)));
    // success = address.compare_exchange_strong(old,
    // reintpret_cast<uint64_t>(val+reinterpret_cast<double>(old)));
    float temp = val + *reinterpret_cast<float *>(&old);
    // success = dummy.compare_exchange_strong(const_cast<uint64_t&>(old),
    // *reinterpret_cast<uint64_t*>(&temp));
    success = address.compare_exchange_strong(old, *reinterpret_cast<uint32_t *>(&temp));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (!success);

  return *reinterpret_cast<float *>(&old);
}
  #endif // EMULATE_ATOMIC_ADD_FLOAT
  #ifdef EMULATE_ATOMIC_ADD_DOUBLE
// Inspired by the older CUDA C Programming Guide
double myAtomicAdd(cl::sycl::atomic<uint64_t> &address, double val) {
  uint64_t old = address.load();
  // uint64_t atomic_load
  bool success = false;
  do {
    // old = atomicCAS(address_as_ull, assumed,
    //                __double_as_longlong(val +
    //                       __longlong_as_double(assumed)));
    // success = address.compare_exchange_strong(old,
    // reintpret_cast<uint64_t>(val+reinterpret_cast<double>(old)));
    double temp = val + *reinterpret_cast<double *>(&old);
    // success = dummy.compare_exchange_strong(const_cast<uint64_t&>(old),
    // *reinterpret_cast<uint64_t*>(&temp));
    success = address.compare_exchange_strong(old, *reinterpret_cast<uint64_t *>(&temp));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (!success);

  return *reinterpret_cast<double *>(&old);
}
  #endif // EMULATE_ATOMIC_ADD_DOUBLE

#endif // STANDALONE

namespace sygraph {
namespace detail {
// Volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
// Must be marked external since main.cpp uses it
extern SYCL_EXTERNAL const void
Jaccard_RowSumKernel<weighted, vertex_t, edge_t, weight_t>::operator()(
    cl::sycl::nd_item<2> tid_info) const {
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

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
const cl::sycl::event Jaccard_RowSumKernel<weighted, vertex_t, edge_t, weight_t>::invoke(
    vertex_t n, edge_t e, cl::sycl::buffer<edge_t> &csrPtr, cl::sycl::buffer<vertex_t> &csrInd,
    cl::sycl::buffer<weight_t> *weight_in, cl::sycl::buffer<weight_t> &work, cl::sycl::queue &q) {
  // Needs to be 1 for barriers in warp intrinsic emulation
  size_t y = 1;

  // setup launch configuration
  // SYCL: INVERT THE ORDER OF MULTI-DIMENSIONAL THREAD INDICES
  cl::sycl::range<2> sum_local{y, 32};
  cl::sycl::range<2> sum_global{std::min((size_t)(n + sum_local.get(0) - 1) / sum_local.get(0),
                                         (size_t)vertex_t{CUDA_MAX_BLOCKS}) *
                                    sum_local.get(0),
                                sum_local.get(1)};

  cl::sycl::event sum_event;
  try {
    // launch kernel
    sum_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr_acc =
          csrPtr.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)n + 1});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd_acc =
          csrInd.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                   cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> work_acc =
          work.template get_access<cl::sycl::access::mode::discard_write>(
              cgh, cl::sycl::range<1>{(size_t)n});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          shfl_temp(sum_local.get(0) * sum_local.get(1), cgh);
      if constexpr (weighted) {
        cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_in_acc =
            weight_in->template get_access<cl::sycl::access::mode::read>(
                cgh, cl::sycl::range<1>{(size_t)n});
        Jaccard_RowSumKernel<true, vertex_t, edge_t, weight_t> sum_kernel(
            n, csrPtr_acc, csrInd_acc, weight_in_acc, work_acc, shfl_temp);
        cgh.parallel_for(cl::sycl::nd_range<2>{sum_global, sum_local}, sum_kernel);
      } else {
        Jaccard_RowSumKernel<false, vertex_t, edge_t, weight_t> sum_kernel(
            n, csrPtr_acc, csrInd_acc, work_acc, shfl_temp);
        cgh.parallel_for(cl::sycl::nd_range<2>{sum_global, sum_local}, sum_kernel);
      }
    });
    // CUDA actually had a sync here, force a queue flush
    q.wait();
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during VC RowSum\n\t" << e.what() << std::endl;
  }
  return sum_event;
}
// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
const void Jaccard_IsKernel<weighted, vertex_t, edge_t, weight_t>::operator()(
    cl::sycl::nd_item<3> tid_info) const {
  edge_t i, j, Ni, Nj;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  vertex_t row_start = tid_info.get_global_id(0);
  vertex_t row_incr = tid_info.get_global_range(0);
  edge_t j_off = tid_info.get_global_id(1);
  edge_t j_incr = tid_info.get_global_range(1);
  edge_t i_off = tid_info.get_global_id(2);
  edge_t i_incr = tid_info.get_global_range(2);
  for (row = row_start; row < n; row += row_incr) {
    for (j = csrPtr[row] + j_off; j < csrPtr[row + 1]; j += j_incr) {
      col = csrInd[j];
      // find which row has least elements (and call it reference row)
      Ni = csrPtr[row + 1] - csrPtr[row];
      Nj = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;

      // compute new sum weights
      weight_s[j] = work[row] + work[col];

      // compute new intersection weights
      // search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + i_off; i < csrPtr[ref + 1]; i += i_incr) {
        match = -1;
        ref_col = csrInd[i];
        // Must be if constexpr so it doesn't try to evaluate v when it's a nullptr_t
        if constexpr (weighted) {
          ref_val = v[ref_col];
        } else {
          ref_val = 1.0;
        }

        // binary search (column indices are sorted within each row)
        edge_t left = csrPtr[cur];
        edge_t right = csrPtr[cur + 1] - 1;
        while (left <= right) {
          edge_t middle = (left + right) >> 1;
          cur_col = csrInd[middle];
          if (cur_col > ref_col) {
            right = middle - 1;
          } else if (cur_col < ref_col) {
            left = middle + 1;
          } else {
            match = middle;
            break;
          }
        }

        // if the element with the same column index in the reference row has been found
        if (match != -1) {
          // FIXME: Update to SYCL 2020 atomic_refs
          if constexpr (std::is_same<weight_t, double>::value) {
            // if constexpr (typeid(weight_t) == typeid(double)) {
#ifdef EMULATE_ATOMIC_ADD_DOUBLE
            cl::sycl::atomic<uint64_t> atom_weight{
                cl::sycl::global_ptr<uint64_t>{(uint64_t *)&weight_i[j]}};
            myAtomicAdd(atom_weight, ref_val);
#else
            cl::sycl::atomic<weight_t> atom_weight{cl::sycl::global_ptr<weight_t>{&weight_i[j]}};
            atom_weight.fetch_add(ref_val);
#endif
          }
          // if constexpr (typeid(weight_t) == typeid(float)) {
          if constexpr (std::is_same<weight_t, float>::value) {
#ifdef EMULATE_ATOMIC_ADD_FLOAT
            cl::sycl::atomic<uint32_t> atom_weight{
                cl::sycl::global_ptr<uint32_t>{(uint32_t *)&weight_i[j]}};
            myAtomicAdd(atom_weight, ref_val);
#else
            cl::sycl::atomic<weight_t> atom_weight{cl::sycl::global_ptr<weight_t>{&weight_i[j]}};
            atom_weight.fetch_add(ref_val);
#endif
          }
          // FIXME: Use the below with a sycl::atomic once hipSYCL supports the 2020 Floating
          // atomics atomicAdd(&weight_i[j], ref_val);
        }
      }
    }
  }
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
const cl::sycl::event Jaccard_IsKernel<weighted, vertex_t, edge_t, weight_t>::invoke(
    vertex_t n, edge_t e, cl::sycl::buffer<edge_t> &csrPtr, cl::sycl::buffer<vertex_t> &csrInd,
    cl::sycl::buffer<weight_t> *weight_in, cl::sycl::buffer<weight_t> &work,
    cl::sycl::buffer<weight_t> &weight_i, cl::sycl::buffer<weight_t> &weight_s,
    cl::sycl::queue &q) {
  // Back to previous value since this doesn't require barriers
  size_t y = 4;

  // setup launch configuration
  // SYCL: INVERT THE ORDER OF MULTI-DIMENSIONAL THREAD INDICES
  // FIXME: De-CUDA the MAX_KERNEL_THREADS and MAX_BLOCKS defines
  cl::sycl::range<3> is_local{8, y, 32 / y};
  cl::sycl::range<3> is_global{std::min((size_t)(n + is_local.get(0) - 1) / is_local.get(0),
                                        (size_t)vertex_t{CUDA_MAX_BLOCKS}) *
                                   is_local.get(0),
                               1 * is_local.get(1), 1 * is_local.get(2)};

  cl::sycl::event is_event;
  try {
    // launch kernel
    // FIXME: Implement in SYCL lamda
    is_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr_acc =
          csrPtr.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)n + 1});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd_acc =
          csrInd.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                   cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work_acc =
          work.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                 cl::sycl::range<1>{(size_t)n});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i_acc =
          weight_i.template get_access<cl::sycl::access::mode::read_write>(
              cgh, cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s_acc =
          weight_s.template get_access<cl::sycl::access::mode::discard_write>(
              cgh, cl::sycl::range<1>{(size_t)e});
      if constexpr (weighted) {
        cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_in_acc =
            weight_in->template get_access<cl::sycl::access::mode::read>(
                cgh, cl::sycl::range<1>{(size_t)n});
        Jaccard_IsKernel<true, vertex_t, edge_t, weight_t> is_kernel(
            n, csrPtr_acc, csrInd_acc, weight_in_acc, work_acc, weight_i_acc, weight_s_acc);
        cgh.parallel_for(cl::sycl::nd_range<3>{is_global, is_local}, is_kernel);
      } else {
        Jaccard_IsKernel<false, vertex_t, edge_t, weight_t> is_kernel(
            n, csrPtr_acc, csrInd_acc, work_acc, weight_i_acc, weight_s_acc);
        cgh.parallel_for(cl::sycl::nd_range<3>{is_global, is_local}, is_kernel);
      }
    });
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during VC Intersection\n\t" << e.what() << std::endl;
  }

  return is_event;
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Using list of node pairs
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
const void Jaccard_IsPairsKernel<weighted, vertex_t, edge_t, weight_t>::operator()(
    cl::sycl::nd_item<3> tid_info) const {
  edge_t i, idx, Ni, Nj, match;
  vertex_t row, col, ref, cur, ref_col, cur_col;
  weight_t ref_val;

  for (idx = tid_info.get_global_id(0); idx < num_pairs; idx += tid_info.get_global_range(0)) {
    row = first_pair[idx];
    col = second_pair[idx];

    // find which row has least elements (and call it reference row)
    Ni = csrPtr[row + 1] - csrPtr[row];
    Nj = csrPtr[col + 1] - csrPtr[col];
    ref = (Ni < Nj) ? row : col;
    cur = (Ni < Nj) ? col : row;

    // compute new sum weights
    weight_s[idx] = work[row] + work[col];

    // compute new intersection weights
    // search for the element with the same column index in the reference row
    for (i = csrPtr[ref] + tid_info.get_global_id(2); i < csrPtr[ref + 1];
         i += tid_info.get_global_range(2)) {
      match = -1;
      ref_col = csrInd[i];
      if constexpr (weighted) {
        ref_val = v[ref_col];
      } else {
        ref_val = 1.0;
      }

      // binary search (column indices are sorted within each row)
      edge_t left = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;
      while (left <= right) {
        edge_t middle = (left + right) >> 1;
        cur_col = csrInd[middle];
        if (cur_col > ref_col) {
          right = middle - 1;
        } else if (cur_col < ref_col) {
          left = middle + 1;
        } else {
          match = middle;
          break;
        }
      }

      // if the element with the same column index in the reference row has been found
      if (match != -1) {
        // FIXME: Update to SYCL 2020 atomic_refs
        if constexpr (std::is_same<weight_t, double>::value) {
          // if constexpr (typeid(weight_t) == typeid(double)) {
#ifdef EMULATE_ATOMIC_ADD_DOUBLE
          cl::sycl::atomic<uint64_t> atom_weight{
              cl::sycl::global_ptr<uint64_t>{(uint64_t *)&weight_i[i]}};
          myAtomicAdd(atom_weight, ref_val);
#else
          cl::sycl::atomic<weight_t> atom_weight{cl::sycl::global_ptr<weight_t>{&weight_i[i]}};
          atom_weight.fetch_add(ref_val);
#endif
        }
        // if constexpr (typeid(weight_t) == typeid(float)) {
        if constexpr (std::is_same<weight_t, float>::value) {
#ifdef EMULATE_ATOMIC_ADD_FLOAT
          cl::sycl::atomic<uint32_t> atom_weight{
              cl::sycl::global_ptr<uint32_t>{(uint32_t *)&weight_i[i]}};
          myAtomicAdd(atom_weight, ref_val);
#else
          cl::sycl::atomic<weight_t> atom_weight{cl::sycl::global_ptr<weight_t>{&weight_i[i]}};
          atom_weight.fetch_add(ref_val);
#endif
        }
        // FIXME: Use the below with a sycl::atomic once hipSYCL supports the 2020 Floating
        // atomics atomicAdd(&weight_i[j], ref_val);
      }
    }
  }
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
const cl::sycl::event Jaccard_IsPairsKernel<weighted, vertex_t, edge_t, weight_t>::invoke(
    vertex_t n, edge_t num_pairs, cl::sycl::buffer<edge_t> &csrPtr,
    cl::sycl::buffer<vertex_t> &csrInd, cl::sycl::buffer<vertex_t> &first_pair,
    cl::sycl::buffer<vertex_t> &second_pair, cl::sycl::buffer<weight_t> *weight_in,
    cl::sycl::buffer<weight_t> &work, cl::sycl::buffer<weight_t> &weight_i,
    cl::sycl::buffer<weight_t> &weight_s,

    cl::sycl::buffer<weight_t> &weight_j, cl::sycl::queue &q) {
  // setup launch configuration
  // FIXME: De-CUDA the MAX_KERNEL_THREADS and MAX_BLOCKS defines
  cl::sycl::range<3> is_local{32, 1, 8};
  cl::sycl::range<3> is_global{1 * is_local.get(0), 1 * is_local.get(1),
                               std::min((size_t)(n + is_local.get(2) - 1) / is_local.get(2),
                                        (size_t)vertex_t{CUDA_MAX_BLOCKS}) *
                                   is_local.get(2)};
  cl::sycl::event ispairs_event;
  try {
    // launch kernel
    // FIXME: Implement in SYCL lamda
    ispairs_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr_acc =
          csrPtr.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)n + 1});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd_acc =
          csrInd.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)num_pairs});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> first_pair_acc =
          first_pair.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)num_pairs});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> second_pair_acc =
          second_pair.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)num_pairs});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> work_acc =
          work.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                 cl::sycl::range<1>{(size_t)n});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_i_acc =
          weight_i.template get_access<cl::sycl::access::mode::read_write>(
              cgh, cl::sycl::range<1>{(size_t)num_pairs});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_s_acc =
          weight_s.template get_access<cl::sycl::access::mode::discard_write>(
              cgh, cl::sycl::range<1>{(size_t)num_pairs});
      ;
      if constexpr (weighted) {
        cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_in_acc =
            weight_in->template get_access<cl::sycl::access::mode::read>(
                cgh, cl::sycl::range<1>{(size_t)num_pairs});
        ;
        Jaccard_IsPairsKernel<true, vertex_t, edge_t, weight_t> is_kernel(
            num_pairs, csrPtr_acc, csrInd_acc, first_pair_acc, second_pair_acc, weight_in_acc,
            work_acc, weight_i_acc, weight_s_acc);
        cgh.parallel_for(cl::sycl::nd_range<3>{is_global, is_local}, is_kernel);
      } else {
        Jaccard_IsPairsKernel<false, vertex_t, edge_t, weight_t> is_kernel(
            num_pairs, csrPtr_acc, csrInd_acc, first_pair_acc, second_pair_acc, work_acc,
            weight_i_acc, weight_s_acc);
        cgh.parallel_for(cl::sycl::nd_range<3>{is_global, is_local}, is_kernel);
      }
    });
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during VC IsPairs\n\t" << e.what() << std::endl;
  }
  return ispairs_event;
}

// Jaccard  weights (*weight)
template <typename vertex_t, typename edge_t, typename weight_t>
const void
Jaccard_JwKernel<vertex_t, edge_t, weight_t>::operator()(cl::sycl::nd_item<1> tid_info) const {
  edge_t j;
  weight_t Wi, Ws, Wu;

  for (j = tid_info.get_global_id(0); j < e; j += tid_info.get_global_range(0)) {
    Wi = weight_i[j];
    Ws = weight_s[j];
    Wu = Ws - Wi;
    weight_j[j] = (Wi / Wu);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
const cl::sycl::event Jaccard_JwKernel<vertex_t, edge_t, weight_t>::invoke(
    edge_t e, cl::sycl::buffer<weight_t> &weight_i, cl::sycl::buffer<weight_t> &weight_s,
    cl::sycl::buffer<weight_t> &weight_j, cl::sycl::queue &q) {
  // setup launch configuration
  cl::sycl::range<1> jw_local{std::min((size_t)e, (size_t)edge_t{CUDA_MAX_KERNEL_THREADS})};
  cl::sycl::range<1> jw_global{std::min((size_t)(e + jw_local.get(0) - 1) / jw_local.get(0),
                                        (size_t)edge_t{CUDA_MAX_BLOCKS}) *
                               jw_local.get(0)};

  cl::sycl::event jw_event;
  try {
    // launch kernel
    jw_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_i_acc =
          weight_i.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                     cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read> weight_s_acc =
          weight_s.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                     cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::discard_write> weight_j_acc =
          weight_j.template get_access<cl::sycl::access::mode::discard_write>(
              cgh, cl::sycl::range<1>{(size_t)e});
      Jaccard_JwKernel<vertex_t, edge_t, weight_t> jw_kernel(e, weight_i_acc, weight_s_acc,
                                                             weight_j_acc);
      cgh.parallel_for(cl::sycl::nd_range<1>{jw_global, jw_local}, jw_kernel);
    });
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during VC Weights\n\t" << e.what() << std::endl;
  }
  return jw_event;
}
template <typename vertex_t, typename edge_t, typename weight_t>
const void
Jaccard_ec_scan<vertex_t, edge_t, weight_t>::operator()(cl::sycl::nd_item<1> tid_info) const {
  edge_t j, i;
  for (j = tid_info.get_global_id(0); j < n; j += tid_info.get_global_range(0)) {
    for (i = csrPtr[j]; i < csrPtr[j + 1]; i++) {
      dest_ind[i] = j;
      weight_j[i] = 0;
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
const cl::sycl::event Jaccard_ec_scan<vertex_t, edge_t, weight_t>::invoke(
    edge_t e, vertex_t n, cl::sycl::buffer<edge_t> &csrPtr, cl::sycl::buffer<vertex_t> &dest_ind,
    cl::sycl::buffer<weight_t> &weight_j, cl::sycl::queue &q) {
  cl::sycl::range<1> local{std::min((size_t)n, (size_t)vertex_t{CUDA_MAX_KERNEL_THREADS})};
  cl::sycl::range<1> global{
      std::min((size_t)(n + local.get(0) - 1) / local.get(0), (size_t)vertex_t{CUDA_MAX_BLOCKS}) *
      local.get(0)};

  cl::sycl::event scan_event;
  // Scan kernel to set up adjacency list
  try {
    scan_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr_acc =
          csrPtr.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)n + 1});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read_write> dest_ind_acc =
          dest_ind.template get_access<cl::sycl::access::mode::read_write>(
              cgh, cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_j_acc =
          weight_j.template get_access<cl::sycl::access::mode::read_write>(
              cgh, cl::sycl::range<1>{(size_t)e});
      Jaccard_ec_scan<vertex_t, edge_t, weight_t> escan_kernel(e, n, csrPtr_acc, dest_ind_acc,
                                                               weight_j_acc);
      cgh.parallel_for(cl::sycl::nd_range<1>{global, local}, escan_kernel);
    });

#ifdef DEBUG_2
    q.wait();
#endif // DEBUG_2
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during EC-Scan enqueue\n\t" << e.what() << std::endl;
  }
  return scan_event;
}

// Edge-centric-unweighted-kernel
template <typename vertex_t, typename edge_t, typename weight_t>
const void
Jaccard_ec_unweighted<vertex_t, edge_t, weight_t>::operator()(cl::sycl::nd_item<1> tid_info) const {
  edge_t i, j, Ni, Nj, tid;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  for (tid = tid_info.get_global_id(0); tid < e; tid += tid_info.get_global_range(0)) {
    row = csrInd[tid];
    col = dest_ind[tid];

    // find which row has least elements (and call it reference row)
    Ni = csrPtr[row + 1] - csrPtr[row];
    Nj = csrPtr[col + 1] - csrPtr[col];
    ref = (Ni < Nj) ? row : col;
    cur = (Ni < Nj) ? col : row;

    // compute new sum weights
    for (i = csrPtr[ref]; i < csrPtr[ref + 1]; i++) {
      ref_col = csrInd[i];
      // binary search (column indices are sorted within each row)
      edge_t left = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;
      while (left <= right) {
        edge_t middle = (left + right) >> 1;
        cur_col = csrInd[middle];
        if (cur_col > ref_col) {
          right = middle - 1;
        } else if (cur_col < ref_col) {
          left = middle + 1;
        } else {
          weight_j[tid] = weight_j[tid] + 1;
          break;
        }
      }
    }
    // compute JS
    weight_j[tid] = weight_j[tid] / ((weight_t)(Ni + Nj) - weight_j[tid]);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
const cl::sycl::event Jaccard_ec_unweighted<vertex_t, edge_t, weight_t>::invoke(
    edge_t e, vertex_t n, cl::sycl::buffer<edge_t> &csrPtr, cl::sycl::buffer<vertex_t> &csrInd,
    cl::sycl::buffer<vertex_t> &dest_ind, cl::sycl::buffer<weight_t> &weight_j,
    cl::sycl::queue &q) {
  cl::sycl::range<1> local{std::min((size_t)e, (size_t)edge_t{CUDA_MAX_KERNEL_THREADS})};
  cl::sycl::range<1> global{
      std::min((size_t)(e + local.get(0) - 1) / local.get(0), (size_t)edge_t{CUDA_MAX_BLOCKS}) *
      local.get(0)};
  cl::sycl::event edgec_event;

  try {
    // Edge-centric kernel
    edgec_event = q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<edge_t, 1, cl::sycl::access::mode::read> csrPtr_acc =
          csrPtr.template get_access<cl::sycl::access::mode::read>(
              cgh, cl::sycl::range<1>{(size_t)n + 1});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> csrInd_acc =
          csrInd.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                   cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<vertex_t, 1, cl::sycl::access::mode::read> dest_ind_acc =
          dest_ind.template get_access<cl::sycl::access::mode::read>(cgh,
                                                                     cl::sycl::range<1>{(size_t)e});
      cl::sycl::accessor<weight_t, 1, cl::sycl::access::mode::read_write> weight_j_acc =
          weight_j.template get_access<cl::sycl::access::mode::read_write>(
              cgh, cl::sycl::range<1>{(size_t)e});
      Jaccard_ec_unweighted<vertex_t, edge_t, weight_t> ec_kernel(e, n, csrPtr_acc, csrInd_acc,
                                                                  dest_ind_acc, weight_j_acc);
      cgh.parallel_for(cl::sycl::nd_range<1>{global, local}, ec_kernel);
    });
#ifdef DEBUG_2
    q.wait();
#endif // DEBUG_2
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during EC-unweighted enqueue\n\t" << e.what() << std::endl;
  }
  return edgec_event;
}

template <bool edge_centric, bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard(vertex_t n, edge_t e, cl::sycl::buffer<edge_t> &csrPtr,
            cl::sycl::buffer<vertex_t> &csrInd, cl::sycl::buffer<weight_t> *weight_in,
            cl::sycl::buffer<weight_t> &work, cl::sycl::buffer<weight_t> &weight_i,
            cl::sycl::buffer<weight_t> &weight_s, cl::sycl::buffer<vertex_t> &dest_ind,
            cl::sycl::buffer<weight_t> &weight_j, cl::sycl::queue &q) {
  if constexpr (edge_centric) { // Edge-Centric

    cl::sycl::event scan_event =
        Jaccard_ec_scan<vertex_t, edge_t, weight_t>::invoke(e, n, csrPtr, dest_ind, weight_j, q);

    cl::sycl::event edgec_event = Jaccard_ec_unweighted<vertex_t, edge_t, weight_t>::invoke(
        e, n, csrPtr, csrInd, dest_ind, weight_j, q);

#ifdef EVENT_PROFILE
    try {
      wait_and_print(scan, "ECScan")
    } catch (sycl::exception e) {
      std::cerr << "SYCL Exception while waiting for EC-scan\n\t" << e.what() << std::endl;
    }
    try {
      wait_and_print(edgec, "ECUnweighted")
    } catch (sycl::exception e) {
      std::cerr << "SYCL Exception while waiting for EC-unweighted\n\t" << e.what() << std::endl;
    }
#endif // EVENT_PROFILE

    weight_t thresh = 0.00001;
    int count = 0;
    auto debug_res =
        weight_j.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(e));
    for (edge_t i = 0; i < e; i++) {
      // std::cout << debug_res[i] << std::endl;
      if (debug_res[i] > thresh) count++;
    }
    std::cout << "vertices " << n << "edges " << e << "non zero pairs " << count << std::endl;
  } else { // Vertex-Centric
    cl::sycl::event sum_event = Jaccard_RowSumKernel<weighted, vertex_t, edge_t, weight_t>::invoke(
        n, e, csrPtr, csrInd, weight_in, work, q);
#ifdef DEBUG_2
    //  cl::sycl::queue debug = cl::sycl::queue(cl::sycl::cpu_selector());
    std::cout << "DEBUG: Post-RowSum Work matrix of " << n << " elements" << std::endl;
    {
      //    debug.submit([&](cl::sycl::handler &cgh){
      auto debug_acc =
          work.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(n));
      for (int i = 0; i < n; i++) {
        std::cout << debug_acc[i] << std::endl;
      }
      //    });
    }
#endif // DEBUG_2
    cl::sycl::event fill_event = FillKernel<weight_t>::invoke(e, weight_i, weight_t{0.0}, q);
#ifdef DEBUG_2
    q.wait();
    std::cout << "DEBUG: Post-Fill Weight_i matrix of " << e << " elements" << std::endl;
    {
      //    debug.submit([&](cl::sycl::handler &cgh){
      auto debug_acc =
          weight_i.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(n));
      for (int i = 0; i < e; i++) {
        std::cout << debug_acc[i] << std::endl;
      }
      //    });
    }
#endif // DEBUG_2
    cl::sycl::event is_event = Jaccard_IsKernel<weighted, vertex_t, edge_t, weight_t>::invoke(
        n, e, csrPtr, csrInd, weight_in, work, weight_i, weight_s, q);
#ifdef DEBUG_2
    q.wait();
    std::cout << "DEBUG: Post-IS Weight_i and Weight_s matrices of " << e << " elements"
              << std::endl;
    {
      //    debug.submit([&](cl::sycl::handler &cgh){
      auto debug_acc =
          weight_i.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(n));
      auto debug2_acc =
          weight_s.template get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(n));
      for (int i = 0; i < e; i++) {
        std::cout << debug_acc[i] << " " << debug2_acc[i] << std::endl;
      }
      //    });
    }
#endif // DEBUG_2

    cl::sycl::event jw_event =
        Jaccard_JwKernel<vertex_t, edge_t, weight_t>::invoke(e, weight_i, weight_s, weight_j, q);
#ifdef DEBUG_2
    q.wait();
#endif // DEBUG_2

#ifdef EVENT_PROFILE
    try {
      wait_and_print(sum, "VCRowSum")
    } catch (sycl::exception e) {
      std::cerr << "SYCL Exception while waiting for VC RowSum\n\t" << e.what() << std::endl;
    }
    try {
      wait_and_print(fill, "VCFill")
    } catch (sycl::exception e) {
      std::cerr << "SYCL Exception while waiting for VC Fill\n\t" << e.what() << std::endl;
    }
    try {
      wait_and_print(is, "VCIntersection")
    } catch (sycl::exception e) {
      std::cerr << "SYCL Exception while waiting for VC Intersection\n\t" << e.what() << std::endl;
    }
    try {
      wait_and_print(jw, "VCJaccardWeight")
    } catch (sycl::exception e) {
      std::cerr << "SYCL Exception while waiting for VC Weights\n\t" << e.what() << std::endl;
    }
#endif // EVENT_PROFILE
  }
  return 0;
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard_pairs(vertex_t n, edge_t num_pairs, cl::sycl::buffer<edge_t> &csrPtr,
                  cl::sycl::buffer<vertex_t> &csrInd, cl::sycl::buffer<vertex_t> &first_pair,
                  cl::sycl::buffer<vertex_t> &second_pair, cl::sycl::buffer<weight_t> *weight_in,
                  cl::sycl::buffer<weight_t> &work, cl::sycl::buffer<weight_t> &weight_i,
                  cl::sycl::buffer<weight_t> &weight_s,

                  cl::sycl::buffer<weight_t> &weight_j, cl::sycl::queue &q) {
  Jaccard_RowSumKernel<weighted, vertex_t, edge_t, weight_t>::invoke(n, num_pairs, csrPtr, csrInd,
                                                                     weight_in, work, q);
  q.wait();

  // NOTE: initilized weight_i vector with 0.0
  // fill(num_pairs, weight_i, weight_t{0.0}, q);

  Jaccard_IsPairsKernel<weighted, vertex_t, edge_t, weight_t>::invoke(
      n, num_pairs, csrPtr, csrInd, first_pair, second_pair, weight_in, work, weight_i, weight_s,
      q);

  Jaccard_JwKernel<vertex_t, edge_t, weight_t>::invoke(num_pairs, weight_i, weight_s, weight_j, q);

  return 0;
}
} // namespace detail

#ifndef DISABLE_WEIGHTED
template <bool edge_centric, typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> &graph, cl::sycl::buffer<WT> &weights,
             cl::sycl::buffer<WT> &result, cl::sycl::queue &q) {

  cl::sycl::buffer<WT> weight_i(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> weight_s(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> work(cl::sycl::range<1>(graph.number_of_vertices));
  cl::sycl::buffer<VT> dest_ind(cl::sycl::range<1>(graph.number_of_edges));
  sygraph::detail::jaccard<edge_centric, true, VT, ET, WT>(
      graph.number_of_vertices, graph.number_of_edges, graph.offsets, graph.indices, &weights, work,
      weight_i, weight_s, dest_ind, result, q);
  // Buffers autodestruct at end of function scope
}
#endif // DISABLE_WEIGHTED

#ifndef DISABLE_UNWEIGHTED
template <bool edge_centric, typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> &graph, cl::sycl::buffer<WT> &result, cl::sycl::queue &q) {

  cl::sycl::buffer<WT> weight_i(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> weight_s(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> work(cl::sycl::range<1>(graph.number_of_vertices));
  cl::sycl::buffer<VT> dest_ind(cl::sycl::range<1>(graph.number_of_edges));
  sygraph::detail::jaccard<edge_centric, false, VT, ET, WT>(
      graph.number_of_vertices, graph.number_of_edges, graph.offsets, graph.indices, nullptr, work,
      weight_i, weight_s, dest_ind, result, q);
  // Buffers autodestruct at end of function scope
}
#endif // DISABLE_UNWEIGHTED

#ifndef DISABLE_LIST
  #ifndef DISABLE_WEIGHTED
template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> &graph, cl::sycl::buffer<WT> &weights, ET num_pairs,
                  cl::sycl::buffer<VT> &first, cl::sycl::buffer<VT> &second,
                  cl::sycl::buffer<WT> &result, cl::sycl::queue &q) {
  cl::sycl::buffer<WT> weight_i(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> weight_s(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> work(cl::sycl::range<1>(graph.number_of_vertices));

  sygraph::detail::jaccard_pairs<true, VT, ET, WT>(graph.number_of_vertices, num_pairs,
                                                   graph.offsets, graph.indices, first, second,
                                                   &weights, work, weight_i, weight_s, result, q);
  // Buffers autodestruct at end of function scope
}
  #endif // DISABLE_WEIGHTED

  #ifdef DISABLE_UNWEIGHTED
template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> &graph, ET num_pairs, cl::sycl::buffer<VT> &first,
                  cl::sycl::buffer<VT> &second, cl::sycl::buffer<WT> &result, cl::sycl::queue &q) {
  cl::sycl::buffer<WT> weight_i(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> weight_s(cl::sycl::range<1>(graph.number_of_edges));
  cl::sycl::buffer<WT> work(cl::sycl::range<1>(graph.number_of_vertices));

  sygraph::detail::jaccard_pairs<false, VT, ET, WT>(graph.number_of_vertices, num_pairs,
                                                    graph.offsets, graph.indices, first, second,
                                                    nullptr, work, weight_i, weight_s, result, q);
  // Buffers autodestruct at end of function scope
}
  #endif // DISABLE_UNWEIGHTED
#endif   // DISABLE_LIST

#ifndef DISABLE_WEIGHTED
template void jaccard<true, int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> &,
                                                     cl::sycl::buffer<float> &,
                                                     cl::sycl::buffer<float> &, cl::sycl::queue &q);
template void jaccard<false, int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> &,
                                                      cl::sycl::buffer<float> &,
                                                      cl::sycl::buffer<float> &,
                                                      cl::sycl::queue &q);
  #ifndef DISABLE_DP_INDEX
template void jaccard<true, int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> &,
                                                     cl::sycl::buffer<float> &,
                                                     cl::sycl::buffer<float> &, cl::sycl::queue &q);
template void jaccard<false, int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> &,
                                                      cl::sycl::buffer<float> &,
                                                      cl::sycl::buffer<float> &,
                                                      cl::sycl::queue &q);
  #endif // DISABLE_DP_INDEX
#endif   // DISABLE_WEIGHTED
#ifndef DISABLE_UNWEIGHTED
template void jaccard<true, int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> &,
                                                     cl::sycl::buffer<float> &, cl::sycl::queue &q);
template void jaccard<false, int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> &,
                                                      cl::sycl::buffer<float> &,
                                                      cl::sycl::queue &q);
  #ifndef DISABLE_DP_INDEX
template void jaccard<true, int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> &,
                                                     cl::sycl::buffer<float> &, cl::sycl::queue &q);
template void jaccard<false, int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> &,
                                                      cl::sycl::buffer<float> &,
                                                      cl::sycl::queue &q);
  #endif // DISABLE_DP_INDEX
#endif   // DISABLE_UNWEIGHTED
#ifndef DISABLE_LIST
  #ifndef DISABLE_WEIGHTED
template void jaccard_list<int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> &,
                                                    cl::sycl::buffer<float> &, int32_t,
                                                    cl::sycl::buffer<int32_t> &,
                                                    cl::sycl::buffer<int32_t> &,
                                                    cl::sycl::buffer<float> &, cl::sycl::queue &q);
    #ifndef DISABLE_DP_INDEX
template void jaccard_list<int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> &,
                                                    cl::sycl::buffer<float> &, int64_t,
                                                    cl::sycl::buffer<int64_t> &,
                                                    cl::sycl::buffer<int64_t> &,
                                                    cl::sycl::buffer<float> &, cl::sycl::queue &q);
    #endif // DISABLE_DP_INDEX
  #endif   // DISABLE_WEIGHTED
  #ifndef DISABLE_UNWEIGHTED
template void jaccard_list<int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> &,
                                                    int32_t, cl::sycl::buffer<int32_t> &,
                                                    cl::sycl::buffer<int32_t> &,
                                                    cl::sycl::buffer<float> &, cl::sycl::queue &q);
    #ifndef DISABLE_DP_INDEX
template void jaccard_list<int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> &,
                                                    int64_t, cl::sycl::buffer<int64_t> &,
                                                    cl::sycl::buffer<int64_t> &,
                                                    cl::sycl::buffer<float> &, cl::sycl::queue &q);
    #endif // DISABLE_DP_INDEX
  #endif   // DISABLE_UNWEIGHTED
#endif     // DISABLE_LIST
#ifndef DISABLE_DP_WEIGHT
  #ifndef DISABLE_WEIGHTED
template void jaccard<true, int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> &,
                                                      cl::sycl::buffer<double> &,
                                                      cl::sycl::buffer<double> &,
                                                      cl::sycl::queue &q);
template void jaccard<false, int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> &,
                                                       cl::sycl::buffer<double> &,
                                                       cl::sycl::buffer<double> &,
                                                       cl::sycl::queue &q);
    #ifndef DISABLE_DP_INDEX
template void jaccard<true, int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> &,
                                                      cl::sycl::buffer<double> &,
                                                      cl::sycl::buffer<double> &,
                                                      cl::sycl::queue &q);
template void jaccard<false, int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> &,
                                                       cl::sycl::buffer<double> &,
                                                       cl::sycl::buffer<double> &,
                                                       cl::sycl::queue &q);
    #endif // DISABLE_DP_INDEX
  #endif   // DISABLE_WEIGHTED
  #ifndef DISABLE_UNWEIGHTED
template void jaccard<true, int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> &,
                                                      cl::sycl::buffer<double> &,
                                                      cl::sycl::queue &q);
template void jaccard<false, int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> &,
                                                       cl::sycl::buffer<double> &,
                                                       cl::sycl::queue &q);
    #ifndef DISABLE_DP_INDEX
template void jaccard<true, int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> &,
                                                      cl::sycl::buffer<double> &,
                                                      cl::sycl::queue &q);
template void jaccard<false, int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> &,
                                                       cl::sycl::buffer<double> &,
                                                       cl::sycl::queue &q);
    #endif // DISABLE_DP_INDEX
  #endif   // DISABLE_UNWEIGHTED
  #ifndef DISABLE_LIST
    #ifndef DISABLE_WEIGHTED
template void jaccard_list<int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> &,
                                                     cl::sycl::buffer<double> &, int32_t,
                                                     cl::sycl::buffer<int32_t> &,
                                                     cl::sycl::buffer<int32_t> &,
                                                     cl::sycl::buffer<double> &,
                                                     cl::sycl::queue &q);
      #ifndef DISABLE_DP_INDEX
template void jaccard_list<int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> &,
                                                     cl::sycl::buffer<double> &, int64_t,
                                                     cl::sycl::buffer<int64_t> &,
                                                     cl::sycl::buffer<int64_t> &,
                                                     cl::sycl::buffer<double> &,
                                                     cl::sycl::queue &q);
      #endif // DISABLE_DP_INDEX
    #endif   // DISABLE_WEIGHTED
    #ifndef DISABLE_UNWEIGHTED
template void jaccard_list<int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> &,
                                                     int32_t, cl::sycl::buffer<int32_t> &,
                                                     cl::sycl::buffer<int32_t> &,
                                                     cl::sycl::buffer<double> &,
                                                     cl::sycl::queue &q);
      #ifndef DISABLE_DP_INDEX
template void jaccard_list<int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> &,
                                                     int64_t, cl::sycl::buffer<int64_t> &,
                                                     cl::sycl::buffer<int64_t> &,
                                                     cl::sycl::buffer<double> &,
                                                     cl::sycl::queue &q);
      #endif // DISABLE_DP_INDEX
    #endif   // DISABLE_UNWEIGHTED
  #endif     // DISABLE_LIST
#endif       // DISABLE_DP_WEIGHT
} // namespace sygraph
