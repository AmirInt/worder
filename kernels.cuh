#ifndef WORDER_KERNELS_CUH_
#define WORDER_KERNELS_CUH_

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// To make the IDE recognise CUDA functions
#ifndef __CUDACC__
#define __CUDACC__
#endif

// Package
#include "general.hpp"


namespace kernels
{
	/// <summary>
	/// Kernel function to preprocess data for lowercasing
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	__global__ void lowerData(
		char* data
		, const size_t data_length);

	
	/// <summary>
	/// Kernel function to preprocess data for punctuation removal
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	__global__ void removeExcessives(
		char* data
		, const size_t data_length);

	
	/// <summary>
	/// Kernel function to process data to produce the keywords histogram
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	/// <param name="keywords">The keywords to search the data for</param>
	/// <param name="histogram">The histogram of keywords to update</param
	__global__ void countWords(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram);
}

#endif // WORDER_KERNELS_CUH_