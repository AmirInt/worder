#include "kernels.cuh"

namespace kernels
{
    __global__ void countWords(int* c, const int* a, const int* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] + b[i];
    }
}