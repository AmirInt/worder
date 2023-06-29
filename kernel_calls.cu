#include "kernel_calls.cuh"

namespace kernel_calls
{
    // Helper function for using CUDA to add vectors in parallel.
    void processDataWithCuda(
        const char* data
        , const size_t data_length
        , const char* keywords
        , int* histogram)
    {
        int* dev_data{};
        int* dev_keywords{};
        int* dev_histogram{};
        cudaError_t cudaStatus;

        try {
            // Choose which GPU to run on, change this on a multi-GPU system.
            cudaStatus = cudaSetDevice(0);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

            // Allocate GPU buffers for three vectors (two input, one output)    .
            cudaStatus = cudaMalloc((void**)&dev_data, data_length * general::word_size * sizeof(char));
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed!");

            cudaStatus = cudaMalloc((void**)&dev_keywords, general::keywords_length * general::word_size * sizeof(char));
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed!");

            cudaStatus = cudaMalloc((void**)&dev_histogram, general::keywords_length * sizeof(int));
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed!");

            // Copy input vectors from host memory to GPU buffers.
            cudaStatus = cudaMemcpy(dev_data, data, data_length * general::word_size * sizeof(char), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed!");

            cudaStatus = cudaMemcpy(dev_keywords, keywords, general::keywords_length * general::word_size * sizeof(char), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed!");

            cudaStatus = cudaMemcpy(dev_histogram, histogram, general::keywords_length * sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed!");

            // Launch a kernel on the GPU with one thread for each element.
            kernels::countWords << <1, size >> > (dev_data, data_length, dev_keywords, dev_histogram);

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("addKernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize returned error code " + std::to_string(cudaStatus) + "after launching addKernel!");

            // Copy output vector from GPU buffer to host memory.
            cudaStatus = cudaMemcpy(histogram, dev_histogram, general::keywords_length * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed!");
        }
        catch (std::runtime_error& e) {
            cudaFree(dev_data);
            cudaFree(dev_keywords);
            cudaFree(dev_histogram);
            throw std::runtime_error(e);
        }
    }

}