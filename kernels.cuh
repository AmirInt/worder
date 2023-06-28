#ifndef WORDER_KERNELS_CUH_
#define WORDER_KERNELS_CUH_

// C++
#include <string>
#include <regex>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace kernels
{
    __global__ void addKernel(int* c, const int* a, const int* b);
}

#endif // WORDER_KERNELS_CUH_