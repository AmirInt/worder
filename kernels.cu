#include "kernels.cuh"

namespace kernels
{
    __global__ void countWords(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram)
    {
		__shared__ char kws[general::keywords_length * general::word_size];
		__shared__ int lcl_hist[general::keywords_length]{};
		
		int tx{ threadIdx.x };
		int bx{ blockIdx.x };
		int bdim{ blockDim.x };
		int gdim{ gridDim.x };
		
		// Copy keywords to shared memory
		int i{ tx };
		for (; i < general::keywords_length * general::word_size; i += bdim)
			kws[i] = keywords[i];

		__syncthreads();

		// Process data
		for (i = bx * bdim + tx; i < data_length; i += gdim) {
			// Word index
			static int wdidx;
			wdidx = i * general::word_size;

			// Search the keywords to find the index and update local histogram
			static int j;
			for (j = 0; j < general::keywords_length; ++j) {
				// Keyword index
				static int kwidx;
				static bool equal;
				kwidx = j * general::word_size;
				equal = true;
				static int k;
				for (k = 0; k < general::word_size; ++k, ++kwidx, ++wdidx) {
					if (kws[kwidx] == '\0' && data[wdidx] == '\0')
						break;
					if (kws[kwidx] != data[wdidx]) {
						equal = false;
						break;
					}
				}
				if (equal) {
					atomicAdd(&lcl_hist[j], 1);
					break; // Finish searching the keywords
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