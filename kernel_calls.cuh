#ifndef WORDER_KERNEL_CALLS_CUH_
#define WORDER_KERNEL_CALLSS_CUH_

// C++
#include <iostream>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Package
#include "kernels.cuh"

namespace kernel_calls
{
	cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
}

#endif // WORDER_KERNEL_CALLSS_CUH_