#pragma once

#include <string>
#include <vector>
#include <CL/cl.h>

struct LocalBuffer {
    size_t size;
};

inline void setKernelArgs(cl_kernel kernel, int index) {};

template<typename... Rest>
void setKernelArgs(cl_kernel kernel, int index, LocalBuffer local, Rest... rest) {
    clSetKernelArg(kernel, index, local.size, NULL);
    setKernelArgs(kernel, index + 1, rest...);
}
template<typename T, typename... Rest>
void setKernelArgs(cl_kernel kernel, int index, T first, Rest... rest) {
    clSetKernelArg(kernel, index, sizeof(T), &first);
    setKernelArgs(kernel, index + 1, rest...);
}


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

    cl_kernel createKernel(const std::string &name);
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
