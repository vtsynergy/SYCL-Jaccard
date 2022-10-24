This work contains portions of code that are (C) Nvidia and licensed under Apache 2.0.
They have been modified via hand-translation to SYCL for functionality on alternative platforms and thus function as a "derivative work" for license purposes.
Many portions have been adapted to function outside their origin or have been repurposed and remixed.
This file shall serve as documentation of where those portions originated and where they ended up *when first introduced to this repository*.
Any subsequent modification is self-documented by way of the repository history to show how those derivative portions evolved during translation and may or may not still exist in the up-to-date SYCL version, albeit in a new language.


To the best of our knowledge the segments detailed below are a complete list of all derivative elements, which should retain (C) Nvidia.
If we've missed anything, please file a bug report and we will happily correct the omission, in the spirit of giving open-source credit where it is appropriately due.
All other new elements, including edge-centric components and utility files, should be considered (C) Virginia Tech licensable under the same Apache 2.0 terms.


From RAFT (https://github.com/rapidsai/raft) revision 048063dc0 (Feb. 17, 2021)
* From `cpp/include/raft/util/cudart_utils.hpp`
  * Function definitions for `warp_size` and `warp_full_mask` are used unchanged in `jaccard.cu`

From cuGraph (https://github.com/rapidsai/cugraph) revision 3f13ffcdf (Feb. 17, 2021)
* From `cpp/include/graph.hpp`
  * The `GraphViewBase : GraphCompressedSparseBaseView : GraphCSRView` lineage has been flattened and simplified into the single `GraphCSRView` class of `standalone_csr.hpp`
* From `cpp/include/algorithms.hpp`
  * The `cugraph` namespace has been reused in multiple places
  * Template function declarations for `jaccard` and `jaccard_list` and associated Doxygen are used unchanged in `standalone_algorithms.hpp`
* From `cpp/src/utilities/graph_utils.cuh`
  * Preprocessor defines for `CUDA_MAX_BLOCKS` and `CUDA_MAX_KERNEL_THREADS` are used unchanged in `standalone_csr.hpp`
  * Template function definition for `parallel_prefix_sum` is used unchanged in jaccard.cu
  * The original `fill` wrapper function definition was reimplemented from scratch with the same prototype in `jaccard.cu`
    * Removed dependency on Thrust (kernel and pointer types)
    * Removed dependency on RMM (execution policy)
    * Removed CUDA Stream functionality
* From `cpp/src/link_predicition/jaccard.cu`
  * Largely unchanged other than the additions mentioned above
  * Double-precision atomicAdd for compute capability < 6.0 is implemented with atomicCAS according to the older versions of the CUDA C Programming Guide
  * In the `jaccard` and `jaccard_list` wrappers, the RMM templated device vectors and their allocations are replaced with raw pointers and standard `cudaMalloc` calls.
    * CUGRAPH_EXPECTS input checks are removed
