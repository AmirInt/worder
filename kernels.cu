﻿#include "kernels.cuh"

namespace kernels
{
	__global__ void lowerData(
		char* data
		, const size_t data_length)
	{
		int tx{ threadIdx.x };
		int bx{ blockIdx.x };
		int bdim{ blockDim.x };
		int gdim{ gridDim.x * blockDim.x };

		const size_t data_size{ data_length * general::word_size };

		char c;
		for (size_t i{ bx * bdim + tx }; i < data_size; i += gdim) {
			c = data[i];
			if (c < 'Z' and c > 'A')
				data[i] += 'a' - 'A';
		}
	}

	__global__ void removeExcessives(
		char* data
		, const size_t data_length)
	{
		size_t tx{ threadIdx.x };
		size_t bx{ blockIdx.x };
		size_t bdim{ blockDim.x };
		size_t gdim{ gridDim.x * blockDim.x };

		size_t wdidx;
		size_t j;
		char c;
		for (size_t i{ bx * bdim + tx }; i < data_length; i += gdim) {
			// Word index
			wdidx = i * general::word_size;

			c = data[wdidx];
			if (c < 'z' and c > 'a')
				continue;

			for (j = 1; j < general::word_size; ++j) {
				c = data[wdidx + j];
				if (c < 'z' and c > 'a') {
					while (c < 'z' and c > 'a') {
						data[wdidx] = c;
						++wdidx;
						c = data[wdidx + j];
					}
					data[wdidx] = '\0';
					break;
				}
			}
		}
	}

    __global__ void countWords(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram)
    {
		__shared__ char kws[general::keywords_length * general::word_size];
		__shared__ int lcl_hist[general::keywords_length];
		
		size_t tx{ threadIdx.x };
		size_t bx{ blockIdx.x };
		size_t bdim{ blockDim.x };
		size_t gdim{ gridDim.x * blockDim.x };

		// Copy keywords to shared memory
		size_t i{ tx };
		for (; i < general::keywords_length * general::word_size; i += bdim)
			kws[i] = keywords[i];


		// Initialise local histogram
		for (i = tx; i < general::keywords_length; i += bdim)
			lcl_hist[i] = 0;

		__syncthreads();

		// Process data
		size_t wdidx;
		size_t j;
		size_t kwidx;
		bool equal;
		size_t k;
		for (i = bx * bdim + tx; i < data_length; i += gdim) {

			// Word index
			wdidx = i * general::word_size;

			// Search the keywords to find the index and update local histogram
			for (j = 0; j < general::keywords_length; ++j) {
				// Keyword index
				kwidx = j * general::word_size;
				equal = true;
				for (k = 0; k < general::word_size; ++k) {
					if (kws[kwidx + k] == '\0' and data[wdidx + k] == '\0')
						break;
					if (kws[kwidx + k] != data[wdidx + k]) {
						equal = false;
						break;
					}
				}
				if (equal) {
					atomicAdd(&lcl_hist[j], 1);
					break; // Stop searching the keywords
				}
			}
		}

		__syncthreads();

		// Add local histogram onto the global one
		i = tx;
		for (; i < general::keywords_length; i += bdim)
			atomicAdd(&histogram[i], lcl_hist[i]);
    }
}