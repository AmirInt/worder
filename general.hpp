#ifndef WORDER_GENERAL_HPP_
#define WORDER_GENERAL_HPP_

// C++
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <chrono>


namespace general
{
	// Basic application configurations
	// Word size
	constexpr size_t word_size{ 32 };

	// Keywords
	constexpr size_t no_offset{ 0 };
	constexpr size_t keyword_offset{ 512 };
	constexpr size_t keywords_length{ 512 };

	// Datasets
	constexpr size_t small_data_length{ 131'072 }; // words
	const std::string small_data_file{ "./data/small.txt" };

	constexpr size_t medium_data_length{ 393'216 }; // words
	const std::string medium_data_file{ "./data/medium.txt" };

	constexpr size_t large_data_length{ 786'432 }; // words
	const std::string large_data_file{ "./data/large.txt" };

	constexpr size_t huge_data_length{ 1'572'864 }; // words
	const std::string huge_data_file{ "./data/huge.txt" };


	/// <summary>
	/// Opens the file in the given, reads words and puts them into the given
	/// memory address as tokens
	/// </summary>
	/// <param name="file_path">The address of the file</param>
	/// <param name="word_array">The memory address to store word tokens in</param>
	/// <param name="num">The number of words to read</param>
	/// <param name="offset">The number of initial words to skip while readin
	/// the dataset</param>
	void readWordFile(
		const std::string& file_path
		, char* word_array
		, size_t num
		, size_t offset);


	/// <summary>
	/// Preprocesses the given data to lowercase words and remove punctuation marks
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	/// <returns>The time of preprocessing</returns>
	std::chrono::milliseconds preprocessData(char* data, const size_t data_length);


	/// <summary>
	/// Processes data to produce the histogram of the given keywords
	/// </summary>
	/// <param name="data">The target data</param>
	/// <param name="data_length">The length of the data in words</param>
	/// <param name="keywords">The list of keywords</param>
	/// <param name="histogram">The histogram array to update</param>
	/// <returns>The time of processing</returns>
	std::chrono::milliseconds processData(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram);
}

#endif // WORDER_GENERAL_HPP_