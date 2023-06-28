#include "general.hpp"

namespace general
{
	void readWordFile(const char* file_path, char* word_array, size_t num, size_t word_size)
	{
		std::string input_word;
		std::ifstream word_file(file_path);
		if (word_file.is_open()) {
			for (int i{ 0 }; i < num && word_file; ++i) {
				word_file >> input_word;
				if (input_word.length() < word_size)
					strcpy(&word_array[i * word_size], input_word.c_str());
			}
		}
	}
}