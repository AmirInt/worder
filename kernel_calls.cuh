#ifndef WORDER_KERNEL_CALLS_CUH_
#define WORDER_KERNEL_CALLS_CUH_

// C++
#include <iostream>
#include <string>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Package
#include "general.hpp"
#include "kernels.cuh"

namespace kernel_calls
{
	void processDataWithCuda(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram
		, float* compute_time
		, float* total_time);
}

#endif // WORDER_KERNEL_CALLS_CUH_