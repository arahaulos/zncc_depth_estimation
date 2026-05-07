#pragma once

#include <stdint.h>
#include <string>
#include <map>
#include <vector>


namespace Utils
{
    uint64_t timestampUs();
    std::string readTextFile(std::string filename);


    struct ProfilerGuard
    {
        ProfilerGuard(std::vector<uint64_t> &end_times, size_t index) : vec(end_times), idx(index) {};
        ~ProfilerGuard() {
            vec[idx] = timestampUs();
        }

    private:
        std::vector<uint64_t> &vec;
        size_t idx;
    };

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
        ProfilerGuard section(const std::string &section_name);
        void sectionTimes(const std::string &section_name, std::pair<uint64_t, uint64_t> times);

        uint64_t getSectionAverageTime(const std::string &sectionName);
        std::vector<std::string> getSectionNames();

        void printAllAverageTimes();
    private:
        std::map<std::string, std::vector<uint64_t>> start_time_points;
        std::map<std::string, std::vector<uint64_t>> end_time_points;
    };

}
