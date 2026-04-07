#include "OpenCLContext.hpp"
#include <iostream>


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

    queue = clCreateCommandQueue(context, device, 0, &err);
}

OpenCLContext::~OpenCLContext() {
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
