#pragma once

#include <vector>
#include <CL/cl.h>

struct OpenCLContext
{
    OpenCLContext();

    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;

    static OpenCLContext& getInstance() {
        static OpenCLContext instance;
        return instance;
    }

    ~OpenCLContext();

    bool checkError(cl_int err);

    void printDeviceInfo();

    std::pair<uint64_t, uint64_t> getProfilingResults(cl_event event);
    std::pair<uint64_t, uint64_t> getProfilingResults(cl_event first_event, cl_event last_event);

    std::vector<cl_platform_id>  platforms;
    cl_device_id                 device;
    cl_context                   context;
    cl_command_queue             queue;
    cl_program                   program;
};
