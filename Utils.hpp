#pragma once

#include <stdint.h>
#include <string>

namespace Utils
{
    uint64_t timestamp_us();
    std::string read_text_file(std::string filename);
}
