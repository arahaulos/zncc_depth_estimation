#pragma once

#include <stdint.h>
#include <string>

namespace Utils
{
    uint64_t timestampUs();
    std::string readTextFile(std::string filename);
}
