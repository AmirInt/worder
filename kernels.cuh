#ifndef WORDER_KERNELS_CUH_
#define WORDER_KERNELS_CUH_

#ifndef __CUDACC__
#define __CUDACC__
#endif

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Package
#include "general.hpp"

namespace kernels
{
    __global__ void countWords(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram);
}

#endif // WORDER_KERNELS_CUH_