#ifndef WORDER_KERNELS_CUH_
#define WORDER_KERNELS_CUH_

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace kernels
{
    __global__ void countWords(int* c, const int* a, const int* b);
}

#endif // WORDER_KERNELS_CUH_