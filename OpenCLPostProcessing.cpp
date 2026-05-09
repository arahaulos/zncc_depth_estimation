#include <CL/cl.h>
#include <math.h>

#include "Utils.hpp"
#include "OpenCLPostProcessing.hpp"
#include "OpenCLContext.hpp"
#include "OpenCLImage.hpp"


namespace Disparity
{

OpenCLPostProcessor::OpenCLPostProcessor()
{
    auto &ctx = OpenCLContext::getInstance();

    crosscheck_kernel = ctx.createKernel("crosscheck");
    erosion_kernel = ctx.createKernel("erosion");
    fill_kernel = ctx.createKernel("fill");
}

OpenCLPostProcessor::~OpenCLPostProcessor()
{
    clReleaseKernel(crosscheck_kernel);
    clReleaseKernel(erosion_kernel);
    clReleaseKernel(fill_kernel);
}


std::unique_ptr<Image> OpenCLPostProcessor::crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff)
{
    auto &prof = Utils::Profiler::getInstance();
    auto &ctx = OpenCLContext::getInstance();

    auto pg = prof.section("postprocessing_crosscheck");

    cl_event event;

    auto result = std::make_unique<OpenCLImage>();

    int width = disparity.leftToRight->width;
    int height = disparity.rightToLeft->height;

    result->allocate(width, height);

    bool temp_input_buffers = false;

    cl_mem left_to_right;
    cl_mem right_to_left;

    try {
        //Try to upcast Images to OpenCLImages to get buffers
        left_to_right = dynamic_cast<OpenCLImage&>(*disparity.leftToRight).getOpenCLBuffer();
        right_to_left = dynamic_cast<OpenCLImage&>(*disparity.rightToLeft).getOpenCLBuffer();
    } catch (...) {
        temp_input_buffers = true;

        left_to_right  = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, width*height*sizeof(uint8_t), NULL, NULL);
        right_to_left  = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, width*height*sizeof(uint8_t), NULL, NULL);

        clEnqueueWriteBuffer(ctx.queue, left_to_right, CL_FALSE, 0, width*height*sizeof(uint8_t), disparity.leftToRight->pixels.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(ctx.queue, right_to_left, CL_FALSE, 0, width*height*sizeof(uint8_t), disparity.rightToLeft->pixels.data(), 0, NULL, NULL);
    }


    size_t global_work_size[] = {(size_t)width, (size_t)height};

    setKernelArgs(crosscheck_kernel, 0, left_to_right, right_to_left, result->getOpenCLBuffer(), width, height, min_disparity, max_disparity, max_disp_diff);
    clEnqueueNDRangeKernel(ctx.queue, crosscheck_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);

    if (temp_input_buffers) {
        clReleaseMemObject(left_to_right);
        clReleaseMemObject(right_to_left);
    }

    clWaitForEvents(1, &event);
    prof.sectionTimes("crosscheck_kernel", ctx.getProfilingResults(event));

    return result;
}

std::unique_ptr<Image> OpenCLPostProcessor::erosion(Image &in)
{
    auto &prof = Utils::Profiler::getInstance();
    auto &ctx = OpenCLContext::getInstance();

    auto pg = prof.section("postprocessing_erosion");

    cl_event event;

    auto result = std::make_unique<OpenCLImage>();
    result->allocate(in.width, in.height);

    bool temp_input_buffers = false;
    cl_mem input_buff;
    try {
        input_buff = dynamic_cast<OpenCLImage&>(in).getOpenCLBuffer();
    } catch (...) {
        temp_input_buffers = true;
        input_buff  = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, in.width*in.height*sizeof(uint8_t), NULL, NULL);
        clEnqueueWriteBuffer(ctx.queue, input_buff, CL_FALSE, 0, in.width*in.height*sizeof(uint8_t), in.pixels.data(),  0, NULL, NULL);
    }

    size_t global_work_size[] = {(size_t)in.width, (size_t)in.height};

    setKernelArgs(erosion_kernel, 0, input_buff, result->getOpenCLBuffer(), in.width, in.height);
    clEnqueueNDRangeKernel(ctx.queue, erosion_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);

    if (temp_input_buffers) {
        clReleaseMemObject(input_buff);
    }

    clWaitForEvents(1, &event);
    prof.sectionTimes("erosion_kernel", ctx.getProfilingResults(event));

    return result;
}

std::unique_ptr<Image> OpenCLPostProcessor::fill(Image &in)
{
    auto &prof = Utils::Profiler::getInstance();
    auto &ctx = OpenCLContext::getInstance();

    auto pg = prof.section("postprocessing_fill");

    cl_event event;

    auto result = std::make_unique<OpenCLImage>();
    result->allocate(in.width, in.height);

    bool temp_input_buffers = false;
    cl_mem input_buff;
    try {
        input_buff = dynamic_cast<OpenCLImage&>(in).getOpenCLBuffer();
    } catch (...) {
        temp_input_buffers = true;
        input_buff  = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, in.width*in.height*sizeof(uint8_t), NULL, NULL);
        clEnqueueWriteBuffer(ctx.queue, input_buff, CL_FALSE, 0, in.width*in.height*sizeof(uint8_t), in.pixels.data(),  0, NULL, NULL);
    }

    size_t global_work_size[] = {(size_t)in.width, (size_t)in.height};

    setKernelArgs(fill_kernel, 0, input_buff, result->getOpenCLBuffer(), in.width, in.height);
    clEnqueueNDRangeKernel(ctx.queue, fill_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);

    if (temp_input_buffers) {
        clReleaseMemObject(input_buff);
    }

    clWaitForEvents(1, &event);
    prof.sectionTimes("fill_kernel", ctx.getProfilingResults(event));

    {
        auto pg = prof.section("read_to_host_mem");
        result->copyDeviceToHost();
    }

    return result;
}

}


