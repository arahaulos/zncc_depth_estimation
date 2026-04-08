#include "Utils.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

namespace Utils
{


uint64_t timestampUs()
{
    using clock = std::chrono::steady_clock;

    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count());
}

std::string readTextFile(std::string filename)
{
    std::cout << "Reading file: " << filename << "... ";

    std::ifstream file(filename);
    if (file.is_open()) {
        std::string str;

        file.seekg(0, std::ios::end);
        size_t file_len = file.tellg();
        file.seekg(0, std::ios::beg);

        str.resize(file_len);

        file.read(str.data(), file_len);

        std::cout << "done." << std::endl;

        return str;
    }

    std::cout << "Error. Cannot read file" << std::endl;

    return "";
}



}
