#include "kernel_calls.cuh"

namespace kernel_calls
{
    constexpr size_t block_size{ 1024 };
    constexpr size_t grid_size{ 1024 };

    // Helper function for using CUDA to add vectors in parallel.
    void processDataWithCuda(
        const char* data
        , const size_t data_length
        , const char* keywords
        , int* histogram
        , float* compute_time
        , float* total_time)
    {
        char* dev_data{};
        char* dev_keywords{};
        int* dev_histogram{};
        cudaError_t cudaStatus;
        cudaEvent_t start;
        cudaEvent_t start_c;
        cudaEvent_t end;
        cudaEvent_t end_c;

        try {
            // Choose which GPU to run on, change this on a multi-GPU system.
            cudaStatus = cudaSetDevice(0);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

            cudaStatus = cudaEventCreate(&start);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to create start event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventCreate(&start_c);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to create start_c event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventCreate(&end);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to create end event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventCreate(&end_c);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to create end_c event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventRecord(start);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to record start event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            // Allocate GPU buffers for three vectors (two input, one output)
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

            cudaStatus = cudaEventRecord(start_c);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to record start_c event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            // Launch a kernel on the GPU with one thread for each element.
            kernels::countWords << <grid_size, block_size>> > (dev_data, data_length, dev_keywords, dev_histogram);

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize returned error code " + std::to_string(cudaStatus) + "after launching addKernel!" + cudaGetErrorString(cudaStatus));

            cudaStatus = cudaEventRecord(end_c);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to record end_c event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            // Copy output vector from GPU buffer to host memory.
            cudaStatus = cudaMemcpy(histogram, dev_histogram, general::keywords_length * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed!");
            
            cudaStatus = cudaEventRecord(end);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to record end event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventSynchronize(end_c);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to sync end_c event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventSynchronize(end);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to sync end event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventElapsedTime(total_time, start, end);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to take elapsed end event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaStatus = cudaEventElapsedTime(compute_time, start_c, end_c);
            if (cudaStatus != cudaSuccess)
                throw std::runtime_error("Failed to take elapsed end event (error code: " + std::string(cudaGetErrorString(cudaStatus)) + ")!");

            cudaFree(dev_data);
            cudaFree(dev_keywords);
            cudaFree(dev_histogram);
        }
        catch (std::runtime_error& e) {
            cudaFree(dev_data);
            cudaFree(dev_keywords);
            cudaFree(dev_histogram);
            throw std::runtime_error(e);
        }
    }

}