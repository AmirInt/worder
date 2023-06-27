#ifndef WORDER_GENERAL_HPP_
#define WORDER_GENERAL_HPP_

// C++

namespace general
{
	void readKeywordFile(const char* file_path, char** keywords, size_t size);

	void readDataFile(const char* file_path, char** words, size_t size);
}

#endif // WORDER_GENERAL_HPP_