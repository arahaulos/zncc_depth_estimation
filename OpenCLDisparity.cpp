#include <iostream>

#include "OpenCLDisparity.hpp"
#include "Utils.hpp"

namespace Disparity
{


OpenCLDisparityEstimator::OpenCLDisparityEstimator(std::shared_ptr<OpenCLContext> c) {
    context = c;

    image_w = -1;
    image_h = -1;

    cl_int err;

    std::string kernels_text = Utils::readTextFile("kernels_opencl.hpp");

    const char *kernels_text_c_str = kernels_text.c_str();

    program = clCreateProgramWithSource(context->context, 1, &kernels_text_c_str, NULL, &err);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (context->checkError(err)) {
        std::cout << "Failed to build program!" << std::endl;

        size_t log_size;
        clGetProgramBuildInfo(program, context->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char* log = new char[log_size];
        clGetProgramBuildInfo(program, context->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        printf("Build log:\n%s\n", log);

        delete [] log;

        return;
    }

    preprocessing_kernel = clCreateKernel(program, "preprocess", &err);

    if (context->checkError(err)) {
        std::cout << "Failed to create preprocess kernel!" << std::endl;
        return;
    }

    disparity_kernel = clCreateKernel(program, "disparity", &err);

    if (context->checkError(err)) {
        std::cout << "Failed to create disparity kernel!" << std::endl;
    }
}

OpenCLDisparityEstimator::~OpenCLDisparityEstimator() {
    clReleaseKernel(preprocessing_kernel);
    clReleaseKernel(disparity_kernel);
    clReleaseProgram(program);

    if (image_w != -1 || image_h != -1) {
        //There is buffers allocated
        //Lets deallocate those
        deallocateBuffers();
    }
}


DisparityResult OpenCLDisparityEstimator::estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity) {
    int width = left.width;
    int height = left.height;

    DisparityResult result;

    result.leftToRight.allocate(width, height, 1);
    result.rightToLeft.allocate(width, height, 1);

    size_t global_work_size[2] = {width, height};

    //Check that buffer are allocated and right size
    checkBufferSize(width, height);

    //Write input images to GPU mem
    clEnqueueWriteBuffer(context->queue, left_input_buffer,  CL_FALSE, 0, width*height*sizeof(uint8_t), left.pixels.data(),  0, NULL, NULL);
    clEnqueueWriteBuffer(context->queue, right_input_buffer, CL_FALSE, 0, width*height*sizeof(uint8_t), right.pixels.data(), 0, NULL, NULL);

    //Run preprocessing for left image
    setPreprocessingKernelArgs(left_input_buffer, tmp_buffers[0], tmp_buffers[1], tmp_buffers[2], width, height, win_size);
    clEnqueueNDRangeKernel(context->queue, preprocessing_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    //Run preprocessing for right image
    setPreprocessingKernelArgs(right_input_buffer, tmp_buffers[3], tmp_buffers[4], tmp_buffers[5], width, height, win_size);
    clEnqueueNDRangeKernel(context->queue, preprocessing_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    //Run disparity algorithm for left to right image
    setDisparityKernelArgs(left_output_buffer,
                           tmp_buffers[0], tmp_buffers[3],
                           tmp_buffers[1], tmp_buffers[4],
                           tmp_buffers[2], tmp_buffers[5],
                           width, height, win_size, min_disparity, max_disparity);

    clEnqueueNDRangeKernel(context->queue, disparity_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    //Run disparity algorithm for right to left image
    setDisparityKernelArgs(right_output_buffer,
                           tmp_buffers[3], tmp_buffers[0],
                           tmp_buffers[4], tmp_buffers[1],
                           tmp_buffers[5], tmp_buffers[2],
                           width, height, win_size, -max_disparity, -min_disparity);

    clEnqueueNDRangeKernel(context->queue, disparity_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);


    //Read output buffers from GPU mem
    //Use blocking flag, which means that call will blocked until everything is executed
    clEnqueueReadBuffer(context->queue, left_output_buffer, CL_TRUE, 0, width*height*sizeof(uint8_t), result.leftToRight.pixels.data(), 0, NULL, NULL);
    clEnqueueReadBuffer(context->queue, right_output_buffer, CL_TRUE, 0, width*height*sizeof(uint8_t), result.rightToLeft.pixels.data(), 0, NULL, NULL);


    return result;
}

void OpenCLDisparityEstimator::setPreprocessingKernelArgs(cl_mem input_img,
                                                          cl_mem output_img,
                                                          cl_mem output_stdmean,
                                                          cl_mem output_stddev,
                                                          int width,
                                                          int height,
                                                          int win_size)
{
    clSetKernelArg(preprocessing_kernel, 0, sizeof(cl_mem), &input_img);
    clSetKernelArg(preprocessing_kernel, 1, sizeof(cl_mem), &output_img);
    clSetKernelArg(preprocessing_kernel, 2, sizeof(cl_mem), &output_stdmean);
    clSetKernelArg(preprocessing_kernel, 3, sizeof(cl_mem), &output_stddev);

    clSetKernelArg(preprocessing_kernel, 4, sizeof(int), &width);
    clSetKernelArg(preprocessing_kernel, 5, sizeof(int), &height);
    clSetKernelArg(preprocessing_kernel, 6, sizeof(int), &win_size);
}



void OpenCLDisparityEstimator::setDisparityKernelArgs(cl_mem disp,
                                                      cl_mem left_img,     cl_mem right_img,
                                                      cl_mem left_stdmean, cl_mem right_stdmean,
                                                      cl_mem left_stddev,  cl_mem right_stddev,
                                                      int width, int height,
                                                      int win_size,
                                                      int min_disparity, int max_disparity)
{
    clSetKernelArg(disparity_kernel, 0, sizeof(cl_mem), &disp);

    clSetKernelArg(disparity_kernel, 1, sizeof(cl_mem), &left_img);
    clSetKernelArg(disparity_kernel, 2, sizeof(cl_mem), &right_img);
    clSetKernelArg(disparity_kernel, 3, sizeof(cl_mem), &left_stdmean);
    clSetKernelArg(disparity_kernel, 4, sizeof(cl_mem), &right_stdmean);
    clSetKernelArg(disparity_kernel, 5, sizeof(cl_mem), &left_stddev);
    clSetKernelArg(disparity_kernel, 6, sizeof(cl_mem), &right_stddev);

    clSetKernelArg(disparity_kernel, 7,  sizeof(int), &width);
    clSetKernelArg(disparity_kernel, 8,  sizeof(int), &height);
    clSetKernelArg(disparity_kernel, 9,  sizeof(int), &win_size);
    clSetKernelArg(disparity_kernel, 10, sizeof(int), &min_disparity);
    clSetKernelArg(disparity_kernel, 11, sizeof(int), &max_disparity);
}


void OpenCLDisparityEstimator::deallocateBuffers()
{
    clReleaseMemObject(left_input_buffer);
    clReleaseMemObject(right_input_buffer);
    clReleaseMemObject(left_output_buffer);
    clReleaseMemObject(right_output_buffer);

    for (int i = 0; i < 6; i++) {
        clReleaseMemObject(tmp_buffers[i]);
    }
}

void OpenCLDisparityEstimator::checkBufferSize(int w, int h) {
    if (w == image_w && h == image_h) {
        return;
    }

    if (image_w != -1 || image_h != -1) {
        //There is buffers allocated
        //Lets deallocate those
        deallocateBuffers();
    }

    //Allocate buffers
    cl_int err;

    left_output_buffer  = clCreateBuffer(context->context, CL_MEM_WRITE_ONLY, w*h*sizeof(uint8_t), NULL, &err);
    right_output_buffer = clCreateBuffer(context->context, CL_MEM_WRITE_ONLY, w*h*sizeof(uint8_t), NULL, &err);
    left_input_buffer   = clCreateBuffer(context->context, CL_MEM_READ_WRITE, w*h*sizeof(uint8_t), NULL, &err);
    right_input_buffer  = clCreateBuffer(context->context, CL_MEM_READ_WRITE, w*h*sizeof(uint8_t), NULL, &err);

    for (int i = 0; i < 6; i++) {
        tmp_buffers[i] = clCreateBuffer(context->context, CL_MEM_READ_WRITE, w*h*sizeof(float), NULL, &err);
    }

    image_w = w;
    image_h = h;
}


}
