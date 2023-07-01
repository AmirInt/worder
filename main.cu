// Windows
#include <windows.h>

// C++
#include <iostream>
#include <chrono>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Package
#include "kernel_calls.cuh"
#include "general.hpp"


int main()
{
    std::string keyword_file{ "./data/google-10000-english-no-swears.txt" };
    char* keywords{ new char[general::keywords_length * general::word_size] };

    general::readWordFile(
        keyword_file
        , keywords
        , general::keywords_length
        , general::keyword_offset
        , general::word_size);

    // Reading data
    std::string data_file{ general::huge_data_file };
    constexpr size_t data_length{ general::huge_data_length };
    char* data{ new char[data_length * general::word_size] };

    general::readWordFile(
        data_file
        , data
        , data_length
        , general::no_offset
        , general::word_size);

    // Histogram
    int* histogram{ new int[general::keywords_length]() };

    // Run on CPU
    auto premillis{ general::preprocessData(data, data_length) };

    auto millis{ general::processData(data, data_length, keywords, histogram) };
    std::cout << "CPU Duration(ms):\nPreprocess: " << premillis.count() << "\nProcess: " << millis.count() << "\n\n";

    // Run on GPU
    cudaError_t cudaStatus;

    // Process data in parallel.
    try {
        float compute_time{};
        float total_time{};

        kernel_calls::processDataWithCudaStreamsPreprocess(
            data
            , data_length
            , keywords
            , histogram
            , &compute_time
            , &total_time);

        std::cout << "GPU Duration(ms):\nCompute Time: " << compute_time << "\nTotal Time: " << total_time << '\n';
    }
    catch (std::runtime_error& e) {
        std::cout << e.what() << '\n';
    }
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
        throw std::runtime_error("cudaDeviceReset failed!");

    //// Print the histogram
    //for (int i{}; i < general::keywords_length; ++i) {
    //    std::cout << &keywords[i * general::word_size] << ": " << histogram[i] << '\n';
    //}


    return 0;
}
