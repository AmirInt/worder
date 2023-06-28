#ifndef WORDER_KERNEL_CALLS_CUH_
#define WORDER_KERNEL_CALLS_CUH_

// C++
#include <iostream>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Package
#include "kernels.cuh"

namespace kernel_calls
{
	void processDataWithCuda(
		const char* data
		, const size_t data_length
		, const char* keywords
		, const size_t keywords_length
		, const size_t word_size
		, int* histogram);
}

#endif // WORDER_KERNEL_CALLS_CUH_