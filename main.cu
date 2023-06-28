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
    constexpr size_t word_size{ 32 }; // bytes
    // Reading keywords
    constexpr size_t keywords_length{ 1'024 }; // words
    std::string keyword_file{ "./data/google-10000-english-no-swears.txt" };
    char* keywords{ new char[keywords_length * word_size] };

    general::readWordFile(keyword_file, keywords, keywords_length, word_size);

    // Reading data
    constexpr size_t small_data_length{ 131'072 }; // words
    std::string small_data_file{ "./data/small.txt" };
    
    constexpr size_t medium_data_length{ 393'216 }; // words
    std::string medium_data_file{ "./data/medium.txt" };
    
    constexpr size_t large_data_length{ 786'432 }; // words
    std::string large_data_file{ "./data/large.txt" };

    constexpr size_t huge_data_length{ 1'572'864 }; // words
    std::string huge_data_file{ "./data/huge.txt" };

    std::string data_file{ huge_data_file };
    constexpr size_t data_length{ huge_data_length };
    char* data{ new char[data_length * word_size] };

    general::readWordFile(data_file, data, data_length, word_size);

    // Histogram
    int* histogram{ new int[keywords_length]() };

    auto millis{ general::processData(data, data_length, keywords, keywords_length, word_size, histogram) };

    //for (int i{}; i < keywords_length; ++i) {
    //    std::cout << &keywords[i * word_size] << ": " << histogram[i] << '\n';
    //}

    std::cout << "Duration: " << millis.count() << '\n';

    //cudaError_t cudaStatus;

    //// Process data in parallel.
    //kernel_calls::processDataWithCuda(
    //    data
    //    , small_data_length
    //    , keywords
    //    , keywords_length
    //    , word_size
    //    , histogram);
    //
    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess)
    //    throw std::runtime_error("cudaDeviceReset failed!");

    return 0;
}
