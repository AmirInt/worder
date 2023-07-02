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
	// The basic global kernel configurations
	constexpr size_t block_size{ 1024 };
	constexpr size_t grid_size{ 1024 };
	
	// The number of streams to use
	constexpr int n_streams{ 16 };


	/// <summary>
	/// Executes data processing using CUDA; handles memory de/allocation, copying,
	/// launching the kernel and measuring times
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	/// <param name="keywords">The list of keywords</param>
	/// <param name="histogram">The histogram array to update</param>
	/// <param name="compute_time">The kernel execution time from kernel launch
	/// to device synchronisation.</param>
	/// <param name="total_time">The total time to execute allocation, copying
	/// and executing the kernel</param>
	/// <exception>Throws an instance of std::runtime_error if any of the
	/// CUDA operations fail for whatever reason</exception>
	void processDataWithCuda(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram
		, float* compute_time
		, float* total_time);


	/// <summary>
	/// Executes data preprocessing and processing using CUDA; handles memory
	/// de/allocation, copying, launching the kernel and measuring times
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	/// <param name="keywords">The list of keywords</param>
	/// <param name="histogram">The histogram array to update</param>
	/// <param name="compute_time">The kernel execution time from kernel launch
	/// to device synchronisation.</param>
	/// <param name="total_time">The total time to execute allocation, copying
	/// and executing the kernel</param>
	/// <exception>Throws an instance of std::runtime_error if any of the
	/// CUDA operations fail for whatever reason</exception>
	void processDataWithCudaPreprocess(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram
		, float* compute_time
		, float* total_time);


	/// <summary>
	/// Executes data preprocessing and processing using CUDA streams; handles
	/// memory de/allocation, copying, launching the kernel and measuring times
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	/// <param name="keywords">The list of keywords</param>
	/// <param name="histogram">The histogram array to update</param>
	/// <param name="compute_time">The kernel execution time from kernel launch
	/// to device synchronisation.</param>
	/// <param name="total_time">The total time to execute allocation, copying
	/// and executing the kernel</param>
	/// <exception>Throws an instance of std::runtime_error if any of the
	/// CUDA operations fail for whatever reason</exception>
	void processDataWithCudaStreamsPreprocess(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram
		, float* compute_time
		, float* total_time);
}

#endif // WORDER_KERNEL_CALLS_CUH_