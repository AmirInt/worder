#include "kernels.cuh"

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

		for (int i{ bx * bdim + tx }; i < data_length; i += gdim) {
			char c{ data[i] };
			if (c < 'Z' and c > 'A')
				data[i] += 'a' - 'A';
		}
	}

	__global__ void removeExcessives(
		char* data
		, const size_t data_length)
	{

	}

    __global__ void countWords(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram)
    {
		__shared__ char kws[general::keywords_length * general::word_size];
		__shared__ int lcl_hist[general::keywords_length];
		
		int tx{ threadIdx.x };
		int bx{ blockIdx.x };
		int bdim{ blockDim.x };
		int gdim{ gridDim.x * blockDim.x };

		// Copy keywords to shared memory
		size_t i{ tx };
		for (; i < general::keywords_length * general::word_size; i += bdim)
			kws[i] = keywords[i];


		// Initialise local histogram
		for (i = tx; i < general::keywords_length; i += bdim)
			lcl_hist[i] = 0;

		__syncthreads();

		// Process data
		for (i = bx * bdim + tx; i < data_length; i += gdim) {

			// Word index
			size_t wdidx;
			wdidx = i * general::word_size;

			// Search the keywords to find the index and update local histogram
			size_t j;
			for (j = 0; j < general::keywords_length; ++j) {
				// Keyword index
				size_t kwidx;
				bool equal;
				kwidx = j * general::word_size;
				equal = true;
				size_t k;
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