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
    std::string data_file{ general::small_data_file };
    constexpr size_t data_length{ general::small_data_length };
    char* data{ new char[data_length * general::word_size] };

    general::readWordFile(
        data_file
        , data
        , data_length
        , general::no_offset
        , general::word_size);

    // Histogram
    int* histogram{ new int[general::keywords_length]() };

    auto millis{ general::processData(data, data_length, keywords, histogram) };

    for (int i{}; i < general::keywords_length; ++i) {
        std::cout << &keywords[i * general::word_size] << ": " << histogram[i] << '\n';
    }

    std::cout << "\n\nDuration: " << millis.count() << '\n';

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
