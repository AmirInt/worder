#ifndef WORDER_GENERAL_HPP_
#define WORDER_GENERAL_HPP_

// C++
#include <fstream>
#include <string>
#include <cstdlib>


namespace general
{
	void readWordFile(const char* file_path, char* word_array, size_t num, size_t word_size);
}

#endif // WORDER_GENERAL_HPP_