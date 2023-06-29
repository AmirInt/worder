#ifndef WORDER_GENERAL_HPP_
#define WORDER_GENERAL_HPP_

// C++
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <chrono>

//There is 1 device supporting CUDA
//
//Device 0: "NVIDIA GeForce GTX 1650"
//Major revision number : 7
//Minor revision number : 5
//Total amount of global memory : 4294639616 bytes
//Number of multiprocessors : 14
//Number of cores : 112
//Total amount of constant memory : 65536 bytes
//Total amount of shared memory per block : 49152 bytes
//Total amount of shared memory per SM : 65536 bytes
//Total number of registers available per block : 65536
//Warp size : 32
//Maximum number of threads per block : 1024
//Maximum sizes of each dimension of a block : 1024 x 1024 x 64
//Maximum sizes of each dimension of a grid : 2147483647 x 65535 x 65535
//Maximum memory pitch : 2147483647 bytes
//Texture alignment : 512 bytes
//Clock rate : 1.51 GHz
//Concurrent copy and execution : Yes
//
//TEST PASSED

namespace general
{
	constexpr size_t word_size{ 32 };

	constexpr size_t no_offset{ 0 };
	constexpr size_t keyword_offset{ 512 };
	constexpr size_t keywords_length{ 512 };

	constexpr size_t small_data_length{ 131'072 }; // words
	const std::string small_data_file{ "./data/small.txt" };

	constexpr size_t medium_data_length{ 393'216 }; // words
	const std::string medium_data_file{ "./data/medium.txt" };

	constexpr size_t large_data_length{ 786'432 }; // words
	const std::string large_data_file{ "./data/large.txt" };

	constexpr size_t huge_data_length{ 1'572'864 }; // words
	const std::string huge_data_file{ "./data/huge.txt" };

	
	void readWordFile(
		const std::string& file_path
		, char* word_array
		, size_t num
		, size_t offset
		, size_t word_size);


	std::chrono::milliseconds processData(const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram);
}

#endif // WORDER_GENERAL_HPP_