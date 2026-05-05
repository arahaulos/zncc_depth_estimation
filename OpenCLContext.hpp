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

    std::vector<cl_platform_id>  platforms;
    cl_device_id                 device;
    cl_context                   context;
    cl_command_queue             queue;
    cl_program                   program;
};
