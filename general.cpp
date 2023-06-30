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
			for (int i{}; i < offset; ++i)
				word_file >> input_word;

			for (int i{}; i < num && word_file; ++i) {
				word_file >> input_word;
				if (input_word.length() < word_size)
					strcpy(&word_array[i * word_size], input_word.c_str());
			}
		}
	}

	std::chrono::milliseconds preprocessData(
		char* data, const size_t data_length)
	{
		std::regex word_regex("(\\w+)");
		std::smatch match;

		auto start{ std::chrono::system_clock::now() };

		for (int i{ 0 }; i < data_length; ++i) {
			static std::string str;
			str = &data[i * word_size];
			std::regex_search(str, match, word_regex);

			str = match[0].str();

			std::transform(str.begin(), str.end(), str.begin()
				, [](unsigned char c) { return std::tolower(c); });

			strcpy(&data[i * word_size], str.c_str());
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