#/!bin/bash
COMPILER=$1
DEV_NUM=$2
if [[ $COMPILER == "HIPSYCL" ]]; then
  export COMPILER=HIPSYCL
  export HIPSYCL_TARGETS="\"omp\""
    echo "============================"
    echo "Building for CPU OPENMP"
    echo "============================"

  export HIPSYCL_PATH=~/jaccardWorkspace/tools/hipSYCL-intel-build-"$HOSTNAME"/install
  export HIPSYCL_CLANG_PATH=~/jaccardWorkspace/tools/intel-llvm-build-"$HOSTNAME"/install
  export SYCL_C_FLAGS="-DDISABLE_DP_WEIGHT -DDISABLE_LIST -DDISABLE_DP_INDEX -DEVENT_PROFILE"

elif [[ $COMPILER == "ICX" || $COMPILER == "DPCPP" ]]; then
  export COMPILER=ICX
  if [[ $DEV_NUM == "0" ]]; then
    export FPGA=INTEL_EMU
    echo "============================"
    echo "Building for FPGA EMULATION"
    echo "============================" 
  elif [[ $DEV_NUM == "1" ]]; then
    export FPGA=INTEL
    echo "============================"
    echo "Building for FPGA HARDWARE"
    echo "============================"
  elif [[ $DEV_NUM == "2" ]]; then
    echo "============================"
    echo "Building for CPU ONEAPI"
    echo "============================"
    export SYCL_C_FLAGS="-DDISABLE_DP_WEIGHT -DDISABLE_LIST -DDISABLE_DP_INDEX"
  elif [[ $DEV_NUM == "2" ]]; then
    echo "============================"
    echo "Building for SYCL HOST"
    echo "============================"
    export SYCL_C_FLAGS="-DDISABLE_DP_WEIGHT -DDISABLE_LIST -DDISABLE_DP_INDEX"
  fi
fi
#make clean
make jaccardSYCL

