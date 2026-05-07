#include "OpenCLContext.hpp"
#include <iostream>
#include "Utils.hpp"


OpenCLContext::OpenCLContext() {
    cl_int err;

    cl_uint num_platforms;

    err = clGetPlatformIDs(1, NULL, &num_platforms);
    printf("Num platforms detected: %d\n", num_platforms);

    if (num_platforms < 1) {
        std::cout << "No platforms detected!" << std::endl;
        return;
    }

    platforms.resize(num_platforms);

    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);

    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);


    std::string kernels_text = Utils::readTextFile("kernels_opencl.hpp");

    const char *kernels_text_c_str = kernels_text.c_str();

    program = clCreateProgramWithSource(context, 1, &kernels_text_c_str, NULL, &err);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (checkError(err)) {
        std::cout << "Failed to build program!" << std::endl;

        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        printf("Build log:\n%s\n", log);

        delete [] log;

        return;
    }

    printDeviceInfo();
}

OpenCLContext::~OpenCLContext() {
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

bool OpenCLContext::checkError(cl_int err) {
    if (err != CL_SUCCESS) {
        std::cout << "OpenCL error: " << err << std::endl;
        return true;
    }
    return false;
}


void OpenCLContext::printDeviceInfo()
{
    cl_device_local_mem_type mem_type;
    cl_ulong mem_size;
    cl_uint compute_units;
    cl_uint clock_freq;
    cl_ulong const_buf_size;
    size_t max_group_size;
    size_t work_item_sizes[16];
    size_t work_item_ret_size;


    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,           sizeof(cl_device_local_mem_type), &mem_type,        NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,           sizeof(cl_ulong),                 &mem_size,        NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,        sizeof(cl_uint),                  &compute_units,   NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,      sizeof(cl_device_local_mem_type), &clock_freq,      NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(const_buf_size),           &const_buf_size,  NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,      sizeof(size_t),                   &max_group_size,  NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,      sizeof(work_item_sizes),           work_item_sizes, &work_item_ret_size);


    std::cout << "Local memory type: " << mem_type << std::endl;
    std::cout << "Local memory size: " << mem_size / 1024 << " kB" << std::endl;
    std::cout << "Clock frequency: " << clock_freq << " MHz" << std::endl;
    std::cout << "Constant buffer size: " << const_buf_size / 1024 << " kB" << std::endl;
    std::cout << "Max group size: " << max_group_size << std::endl;

    std::cout << "Work item sizes: ";
    for (int i = 0; i < work_item_ret_size / sizeof(size_t); i++) {
        std::cout << work_item_sizes[i] << " ";
    }
    std::cout << std::endl;
}

std::pair<uint64_t, uint64_t> OpenCLContext::getProfilingResults(cl_event event)
{
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,   NULL);

    clReleaseEvent(event);

    return {start / 1000, end / 1000};
}

std::pair<uint64_t, uint64_t> OpenCLContext::getProfilingResults(cl_event first_event, cl_event last_event)
{
    cl_ulong start, end;
    clGetEventProfilingInfo(first_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,   NULL);

    clReleaseEvent(first_event);
    clReleaseEvent(last_event);

    return {start / 1000, end / 1000};
}
