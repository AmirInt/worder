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
		int tx{ threadIdx.x };
		
		// Copy keywords to shared memory
		int i{ tx };
		for (; i < general::keywords_length * general::word_size; i += blockDim.x)
			kws[i] = keywords[i];

		__syncthreads();



    }
}