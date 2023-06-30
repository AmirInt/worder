#include "general.hpp"

namespace general
{
	void readWordFile(
		const std::string& file_path
		, char* word_array
		, size_t num
		, size_t offset
		, size_t word_size)
	{
		std::string input_word;
		std::ifstream word_file(file_path);
		if (word_file.is_open()) {
			for (size_t i{}; i < offset; ++i)
				word_file >> input_word;

			for (size_t i{}; i < num && word_file; ++i) {
				word_file >> input_word;
				if (input_word.length() < word_size)
					strcpy(&word_array[i * word_size], input_word.c_str());
			}
		}
	}

	std::chrono::milliseconds preprocessData(
		char* data, const size_t data_length)
	{
		auto start{ std::chrono::system_clock::now() };

		for (size_t i{ 0 }; i < data_length; ++i) {
			static size_t wdidx;
			wdidx = i * word_size;
			static size_t j;
			static char c;
			for (j = 0; j < word_size; ++j) {
				c = data[wdidx + j];
				c = tolower(c);
				if (c <= 'z' && c >= 'a') {
					while (c <= 'z' && c >= 'a') {
						data[wdidx] = c;
						++wdidx;
						c = data[wdidx + j];
						c = tolower(c);
					}
					data[wdidx] = '\0';
					break;
				}
			}
		}

		auto end{ std::chrono::system_clock::now() };

		auto millis{ std::chrono::duration_cast<std::chrono::milliseconds>(end - start) };

		return millis;
	}

	std::chrono::milliseconds processData(
		const char* data
		, const size_t data_length
		, const char* keywords
		, int* histogram)
	{
		auto start{ std::chrono::system_clock::now() };

		for (int i{ 0 }; i < data_length; ++i) {
			static int j;
			j = 0;
			for (; j < keywords_length; ++j) {
				if (strcmp(&data[i * word_size], &keywords[j * word_size]) == 0) {
					++histogram[j];
					break;
				}
			}
		}
		
		auto end{ std::chrono::system_clock::now() };
		
		auto millis{ std::chrono::duration_cast<std::chrono::milliseconds>(end - start) };
		
		return millis;
	}
}