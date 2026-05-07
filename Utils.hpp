#pragma once

#include <stdint.h>
#include <string>
#include <map>
#include <vector>


namespace Utils
{
    uint64_t timestampUs();
    std::string readTextFile(std::string filename);


    struct Profiler
    {
        Profiler();
        Profiler(const Profiler&) = delete;
        Profiler& operator=(const Profiler&) = delete;

        static Profiler& getInstance() {
            static Profiler instance;
            return instance;
        }

        void clear();

        void sectionStart(const std::string &section);
        void sectionEnd(const std::string &section);

        void sectionTimes(const std::string &section, std::pair<uint64_t, uint64_t> times);

        uint64_t getSectionAverageTime(const std::string &sectionName);
        std::vector<std::string> getSectionNames();

        void printAllAverageTimes();
    private:
        std::map<std::string, std::vector<uint64_t>> start_time_points;
        std::map<std::string, std::vector<uint64_t>> end_time_points;
    };

}
