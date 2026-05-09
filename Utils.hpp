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
        //Uses RAII to guard that code section end timestamp gets recorded
        //When Profiler section method stores start time and creates ProfilerGuard
        //When ProfilerGuard instance goes out of the scope, it stores code section end timestamp

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
        //Purpose of this singleton is to provide easy way to profile how much code sections exeuction takes
        //Idea is that when entering interesting code section, ProfilerGuard is created using section-method
        //When ProfilerGuard instance goes outside of the scope, it records end timestamp
        //Those timestamp are later used to calculate average execution times

        Profiler();
        Profiler(const Profiler&) = delete; //Singleton, so copy constructor and copy assignment should be deleted
        Profiler& operator=(const Profiler&) = delete;

        static Profiler& getInstance() {
            static Profiler instance;
            return instance;
        }

        void clear();
        ProfilerGuard section(const std::string &section_name); //Returns ProfileGuard which then records end timestamp when destroyed

        //Code section start and endtimes can be added manually.
        //Useful for profiling opencl kernel execution time
        void sectionTimes(const std::string &section_name, std::pair<uint64_t, uint64_t> times);

        uint64_t getSectionAverageTime(const std::string &sectionName);
        std::vector<std::string> getSectionNames();

        void printAllAverageTimes(); //Calculates average execution times for all recorded section names, and prints them
    private:
        std::map<std::string, std::vector<uint64_t>> start_time_points;
        std::map<std::string, std::vector<uint64_t>> end_time_points;
    };

}
