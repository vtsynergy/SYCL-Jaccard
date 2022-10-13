/*
 * Original CUDA Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
 * @brief The cugraph Jaccard core functionality
 *
 * @file jaccard.cu
 * ---------------------------------------------------------------------------**/

#ifndef STANDALONE
#include <rmm/thrust_rmm_allocator.h>
#include <utilities/error.hpp>
#include "graph.hpp"
#include "utilities/graph_utils.cuh"
#else
#include <iostream>
#include "standalone_algorithms.hpp"
#include "standalone_csr.hpp"
//From RAFT at commit 048063dc08
__host__ __device__ constexpr inline int warp_size() { return 32; }

__host__ __device__ constexpr inline unsigned int warp_full_mask() {
  return 0xffffffff;
}
//From utilties/graph_utils.cuh
template <typename count_t, typename index_t, typename value_t>
__inline__ __device__ value_t parallel_prefix_sum(count_t n, index_t const *ind, value_t const *w)
{
  count_t i, j, mn;
  value_t v, last;
  value_t sum = 0.0;
  bool valid;

  // Parallel prefix sum (using __shfl)
  mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x);  // n in multiple of blockDim.x
  for (i = threadIdx.x; i < mn; i += blockDim.x) {
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
    last = __shfl_sync(warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

    // if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    // do prefix sum (of size warpSize=blockDim.x =< 32)
    for (j = 1; j < blockDim.x; j *= 2) {
      v = __shfl_up_sync(warp_full_mask(), sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    // shift by last
    sum += last;
    // notice that no __threadfence or __syncthreads are needed in this implementation
  }
  // get the value of the last thread (to all threads)
  last = __shfl_sync(warp_full_mask(), sum, blockDim.x - 1, blockDim.x);

  return last;
}

//Custom Thrust simplifications
template <typename T>
void __global__ fill_kernel(T* ptr, T value, size_t n) {
  int idx = threadIdx.x + blockIdx.x*blockIdx.x;
  int incr = blockDim.x*gridDim.x;
  for (; idx < n; idx += incr) {
    ptr[idx] = value;
  }
}

template <typename T>
void fill(size_t n, T *x, T value)
{
  size_t block = min(n, (size_t)CUDA_MAX_KERNEL_THREADS);
  size_t grid = min((n/block)+((n%block)?1:0), (size_t)CUDA_MAX_BLOCKS); 
  //TODO, do we need to emulate their stream behavior?
  fill_kernel<<<grid, block>>>(x, value, n);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cerr << "Error in fill_kernel " << error << std::endl;
}
//Directly from the older CUDA C Programming Guide
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#endif

namespace cugraph {
namespace detail {

// Volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_row_sum(
  vertex_t n, edge_t const *csrPtr, vertex_t const *csrInd, weight_t const *v, weight_t *work)
{
  vertex_t row;
  edge_t start, end, length;
  weight_t sum;

  for (row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
    start  = csrPtr[row];
    end    = csrPtr[row + 1];
    length = end - start;

    // compute row sums
    if (weighted) {
      sum = parallel_prefix_sum(length, csrInd + start, v);
      if (threadIdx.x == 0) work[row] = sum;
    } else {
      work[row] = static_cast<weight_t>(length);
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is(vertex_t n,
                           edge_t const *csrPtr,
                           vertex_t const *csrInd,
                           weight_t const *v,
                           weight_t *work,
                           weight_t *weight_i,
                           weight_t *weight_s)
{
  edge_t i, j, Ni, Nj;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
         j += gridDim.y * blockDim.y) {
      col = csrInd[j];
      // find which row has least elements (and call it reference row)
      Ni  = csrPtr[row + 1] - csrPtr[row];
      Nj  = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;

      // compute new sum weights
      weight_s[j] = work[row] + work[col];

      // compute new intersection weights
      // search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
           i += gridDim.x * blockDim.x) {
        match   = -1;
        ref_col = csrInd[i];
        if (weighted) {
          ref_val = v[ref_col];
        } else {
          ref_val = 1.0;
        }

        // binary search (column indices are sorted within each row)
        edge_t left  = csrPtr[cur];
        edge_t right = csrPtr[cur + 1] - 1;
        while (left <= right) {
          edge_t middle = (left + right) >> 1;
          cur_col       = csrInd[middle];
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
        if (match != -1) { atomicAdd(&weight_i[j], ref_val); }
      }
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Using list of node pairs
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is_pairs(edge_t num_pairs,
                                 edge_t const *csrPtr,
                                 vertex_t const *csrInd,
                                 vertex_t const *first_pair,
                                 vertex_t const *second_pair,
                                 weight_t const *v,
                                 weight_t *work,
                                 weight_t *weight_i,
                                 weight_t *weight_s)
{
  edge_t i, idx, Ni, Nj, match;
  vertex_t row, col, ref, cur, ref_col, cur_col;
  weight_t ref_val;

  for (idx = threadIdx.z + blockIdx.z * blockDim.z; idx < num_pairs;
       idx += gridDim.z * blockDim.z) {
    row = first_pair[idx];
    col = second_pair[idx];

    // find which row has least elements (and call it reference row)
    Ni  = csrPtr[row + 1] - csrPtr[row];
    Nj  = csrPtr[col + 1] - csrPtr[col];
    ref = (Ni < Nj) ? row : col;
    cur = (Ni < Nj) ? col : row;

    // compute new sum weights
    weight_s[idx] = work[row] + work[col];

    // compute new intersection weights
    // search for the element with the same column index in the reference row
    for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
         i += gridDim.x * blockDim.x) {
      match   = -1;
      ref_col = csrInd[i];
      if (weighted) {
        ref_val = v[ref_col];
      } else {
        ref_val = 1.0;
      }

      // binary search (column indices are sorted within each row)
      edge_t left  = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;
      while (left <= right) {
        edge_t middle = (left + right) >> 1;
        cur_col       = csrInd[middle];
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
      if (match != -1) { atomicAdd(&weight_i[idx], ref_val); }
    }
  }
}

// Jaccard  weights (*weight)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_jw(edge_t e,
                           weight_t const *weight_i,
                           weight_t const *weight_s,
                           weight_t *weight_j)
{
  edge_t j;
  weight_t Wi, Ws, Wu;

  for (j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    Wi          = weight_i[j];
    Ws          = weight_s[j];
    Wu          = Ws - Wi;
    weight_j[j] = (Wi / Wu);
  }
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard(vertex_t n,
            edge_t e,
            edge_t const *csrPtr,
            vertex_t const *csrInd,
            weight_t const *weight_in,
            weight_t *work,
            weight_t *weight_i,
            weight_t *weight_s,
            weight_t *weight_j)
{
  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = min((n + nthreads.y - 1) / nthreads.y, vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z  = 1;

  // launch kernel
  cudaError_t error = cudaSuccess;
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) std::cerr << "Error in jaccard_row_sum " << error << std::endl;
  fill(e, weight_i, weight_t{0.0});

  // setup launch configuration
  nthreads.x = 32 / y;
  nthreads.y = y;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, vertex_t{CUDA_MAX_BLOCKS});  // 1;

  // launch kernel
  jaccard_is<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work, weight_i, weight_s);
  error = cudaDeviceSynchronize(); //Added, not necessary
  if (error != cudaSuccess) std::cerr << "Error in jaccard_is " << error << std::endl;

  // setup launch configuration
  nthreads.x = min(e, edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((e + nthreads.x - 1) / nthreads.x, edge_t{CUDA_MAX_BLOCKS});
  nblocks.y  = 1;
  nblocks.z  = 1;

  // launch kernel
  jaccard_jw<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(e, weight_i, weight_s, weight_j);
  error = cudaDeviceSynchronize(); //Added, not necessary
  if (error != cudaSuccess) std::cerr << "Error in jaccard_jw " << error << std::endl;

  return 0;
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard_pairs(vertex_t n,
                  edge_t num_pairs,
                  edge_t const *csrPtr,
                  vertex_t const *csrInd,
                  vertex_t const *first_pair,
                  vertex_t const *second_pair,
                  weight_t const *weight_in,
                  weight_t *work,
                  weight_t *weight_i,
                  weight_t *weight_s,
                  weight_t *weight_j)
{
  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = min((n + nthreads.y - 1) / nthreads.y, vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z  = 1;

  // launch kernel
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
  cudaDeviceSynchronize();

  // NOTE: initilized weight_i vector with 0.0
  // fill(num_pairs, weight_i, weight_t{0.0});

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = 1;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, vertex_t{CUDA_MAX_BLOCKS});  // 1;

  // launch kernel
  jaccard_is_pairs<weighted, vertex_t, edge_t, weight_t><<<nblocks, nthreads>>>(
    num_pairs, csrPtr, csrInd, first_pair, second_pair, weight_in, work, weight_i, weight_s);

  // setup launch configuration
  nthreads.x = min(num_pairs, edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((num_pairs + nthreads.x - 1) / nthreads.x, (edge_t)CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  // launch kernel
  jaccard_jw<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(num_pairs, weight_i, weight_s, weight_j);

  return 0;
}
}  // namespace detail

template <typename VT, typename ET, typename WT>
void jaccard(GraphCSRView<VT, ET, WT> const &graph, WT const *weights, WT *result)
{
  WT * weight_i;
  cudaMalloc(&weight_i, graph.number_of_edges * sizeof(WT));
  WT * weight_s;
  cudaMalloc(&weight_s, graph.number_of_edges * sizeof(WT));
  WT * work;
  cudaMalloc(&work, graph.number_of_vertices * sizeof(WT));

  if (weights == nullptr) {
    cugraph::detail::jaccard<false, VT, ET, WT>(graph.number_of_vertices,
                                                graph.number_of_edges,
                                                graph.offsets,
                                                graph.indices,
                                                weights,
                                                work,
                                                weight_i,
                                                weight_s,
                                                result);
  } else {
    cugraph::detail::jaccard<true, VT, ET, WT>(graph.number_of_vertices,
                                               graph.number_of_edges,
                                               graph.offsets,
                                               graph.indices,
                                               weights,
                                               work,
                                               weight_i,
                                               weight_s,
                                               result);
  }
  cudaFree(weight_i);
  cudaFree(weight_s);
  cudaFree(work);
}

template <typename VT, typename ET, typename WT>
void jaccard_list(GraphCSRView<VT, ET, WT> const &graph,
                  WT const *weights,
                  ET num_pairs,
                  VT const *first,
                  VT const *second,
                  WT *result)
{
  WT * weight_i;
  cudaMalloc(&weight_i, graph.number_of_edges * sizeof(WT));
  fill(graph.number_of_edges, weight_i, (WT)0.0);
  WT * weight_s;
  cudaMalloc(&weight_s, graph.number_of_edges * sizeof(WT));
  WT * work;
  cudaMalloc(&work, graph.number_of_vertices * sizeof(WT));

  if (weights == nullptr) {
    cugraph::detail::jaccard_pairs<false, VT, ET, WT>(graph.number_of_vertices,
                                                      num_pairs,
                                                      graph.offsets,
                                                      graph.indices,
                                                      first,
                                                      second,
                                                      weights,
                                                      work,
                                                      weight_i,
                                                      weight_s,
                                                      result);
  } else {
    cugraph::detail::jaccard_pairs<true, VT, ET, WT>(graph.number_of_vertices,
                                                     num_pairs,
                                                     graph.offsets,
                                                     graph.indices,
                                                     first,
                                                     second,
                                                     weights,
                                                     work,
                                                     weight_i,
                                                     weight_s,
                                                     result);
  }
  cudaFree(weight_i);
  cudaFree(weight_s);
  cudaFree(work);
}

template void jaccard<int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> const &,
                                               float const *,
                                               float *);
template void jaccard<int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> const &,
                                                double const *,
                                                double *);
template void jaccard<int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> const &,
                                               float const *,
                                               float *);
template void jaccard<int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> const &,
                                                double const *,
                                                double *);
template void jaccard_list<int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> const &,
                                                    float const *,
                                                    int32_t,
                                                    int32_t const *,
                                                    int32_t const *,
                                                    float *);
template void jaccard_list<int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> const &,
                                                     double const *,
                                                     int32_t,
                                                     int32_t const *,
                                                     int32_t const *,
                                                     double *);
template void jaccard_list<int64_t, int64_t, float>(GraphCSRView<int64_t, int64_t, float> const &,
                                                    float const *,
                                                    int64_t,
                                                    int64_t const *,
                                                    int64_t const *,
                                                    float *);
template void jaccard_list<int64_t, int64_t, double>(GraphCSRView<int64_t, int64_t, double> const &,
                                                     double const *,
                                                     int64_t,
                                                     int64_t const *,
                                                     int64_t const *,
                                                     double *);

}  // namespace cugraph
