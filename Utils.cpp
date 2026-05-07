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




Profiler::Profiler() {

}

uint64_t Profiler::getSectionAverageTime(const std::string &section_name)
{
    std::vector<uint64_t> start_times = start_time_points[section_name];
    std::vector<uint64_t> end_times = end_time_points[section_name];

    uint64_t total_execution_time = 0;
    for (size_t i = 0; i < end_times.size(); i++) {
        total_execution_time += end_times[i] - start_times[i];
    }

    return total_execution_time / end_times.size();
}

std::vector<std::string> Profiler::getSectionNames()
{
    std::vector<std::string> names;
    for (auto it = end_time_points.begin(); it != end_time_points.end(); ++it) {
        names.push_back(it->first);
    }
    return names;
}

void Profiler::sectionTimes(const std::string &section_name, std::pair<uint64_t, uint64_t> times)
{
    if (start_time_points.find(section_name) != start_time_points.end()) {
        start_time_points[section_name].push_back(times.first);
        end_time_points[section_name].push_back(times.second);
    } else {
        start_time_points[section_name] = {times.first};
        end_time_points[section_name] = {times.second};
    }
}

void Profiler::printAllAverageTimes()
{
    for (const std::string &name :getSectionNames()) {
        uint64_t us = getSectionAverageTime(name);
        double ms = static_cast<double>(us) / 1000;
        std::cout << name << ": " << ms << "ms" << std::endl;
    }
}

void Profiler::clear()
{
    start_time_points.clear();
    end_time_points.clear();
}

ProfilerGuard Profiler::section(const std::string &section_name)
{
    if (start_time_points.find(section_name) != start_time_points.end()) {
        start_time_points[section_name].push_back(timestampUs());
        end_time_points[section_name].push_back(0);

        return ProfilerGuard(end_time_points[section_name], end_time_points[section_name].size()-1);

    } else {
        start_time_points[section_name] = {timestampUs()};
        end_time_points[section_name] = {0};

        return ProfilerGuard(end_time_points[section_name], end_time_points[section_name].size()-1);
    }
}


}
