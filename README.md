A SYCL implementation of Edge-connected Jaccard Similarity

In addition to a bespoke edge-centric kernel, this work leverages a hand-translated version of the vertex-centric kernel pipeline from cuGraph (https://github.com/rapidsai/cugraph) licensed under Apache 2.0. Please see NOTICE for a description of which elements were reused and where.

Please Cite:
 **Edge-Connected Jaccard Similarity for Graph Link Prediction on FPGA**, Paul Sathre, Atharva Gondhalekar, Wu-chun Feng, In *Proceedings of the IEEE High Performance Extreme Computing Conference (HPEC)*, Waltham, MA, September 2022. (Insert DOI once available)
* For reproducibility, refer to the 7 revisions with the key `hpec22` in their tag. Input data can be found here: [HPEC'22 Input CSR Data](https://chrec.cs.vt.edu/SYCL-Jaccard/HPEC22-Data/index.html)
