#pragma once

#include <vector>
#include <CL/cl.h>

struct OpenCLContext
{
    OpenCLContext();
    ~OpenCLContext();
    bool checkError(cl_int err);

    std::vector<cl_platform_id>  platforms;
    cl_device_id                 device;
    cl_context                   context;
    cl_command_queue             queue;
};
