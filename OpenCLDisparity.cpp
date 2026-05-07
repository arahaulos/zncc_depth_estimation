#include <CL/cl.h>
#include <iostream>

#include "OpenCLDisparity.hpp"
#include "Utils.hpp"
#include "OpenCLImage.hpp"


constexpr int TILE_SIZE = 16;
constexpr int BATCH_SIZE = 16;

namespace Disparity
{


OpenCLDisparityEstimator::OpenCLDisparityEstimator() {
    auto &ctx = OpenCLContext::getInstance();

    image_w = -1;
    image_h = -1;

    cl_int err;

    preprocessing_kernel = clCreateKernel(ctx.program, "preprocess", &err);

    if (ctx.checkError(err)) {
        std::cout << "Failed to create preprocess kernel!" << std::endl;
        return;
    }

    disparity_kernel = clCreateKernel(ctx.program, "disparity", &err);

    if (ctx.checkError(err)) {
        std::cout << "Failed to create disparity kernel!" << std::endl;
    }

    disparity_kernel2 = clCreateKernel(ctx.program, "disparity2", &err);

    if (ctx.checkError(err)) {
        std::cout << "Failed to create disparity2 kernel!" << std::endl;
    }


    use_tiling = true;
}

OpenCLDisparityEstimator::~OpenCLDisparityEstimator() {
    clReleaseKernel(preprocessing_kernel);
    clReleaseKernel(disparity_kernel);
    clReleaseKernel(disparity_kernel2);

    if (image_w != -1 || image_h != -1) {
        //There is buffers allocated
        //Lets deallocate those
        deallocateBuffers();
    }
}


DisparityResult OpenCLDisparityEstimator::estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity) {

    auto &ctx = OpenCLContext::getInstance();
    auto &prof = Utils::Profiler::getInstance();

    cl_event precompute_start;
    cl_event precompute_end;

    cl_event disparity_start;
    cl_event disparity_end;

    cl_event read_result_start;
    cl_event read_result_end;

    int width = left.width;
    int height = left.height;

    //Make sure that window size is odd
    win_size = win_size | 0x1;

    DisparityResult result;

    result.leftToRight.allocate(width, height, 1);
    result.rightToLeft.allocate(width, height, 1);

    size_t global_work_size[2] = {(size_t)width, (size_t)height};

    //Check that buffer are allocated and right size
    checkBufferSize(width, height);

    bool temp_input_buffers = false;

    cl_mem left_input_buffer;
    cl_mem right_input_buffer;

    try {
        //Try to upcast Images to OpenCLImages to get buffers
        left_input_buffer = dynamic_cast<OpenCLImage&>(left).getOpenCLBuffer();
        right_input_buffer = dynamic_cast<OpenCLImage&>(right).getOpenCLBuffer();
    } catch (...) {
        //Okay inputs are not OpenCLImages, lets allocate input buffers and send pixel data to GPU mem
        temp_input_buffers = true;

        left_input_buffer   = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, width*height*sizeof(uint8_t), NULL, NULL);
        right_input_buffer  = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, width*height*sizeof(uint8_t), NULL, NULL);

        //Write input images to GPU mem
        clEnqueueWriteBuffer(ctx.queue, left_input_buffer,  CL_FALSE, 0, width*height*sizeof(uint8_t), left.pixels.data(),  0, NULL, NULL);
        clEnqueueWriteBuffer(ctx.queue, right_input_buffer, CL_FALSE, 0, width*height*sizeof(uint8_t), right.pixels.data(), 0, NULL, NULL);
    }

    //Run preprocessing for left image
    setPreprocessingKernelArgs(left_input_buffer, tmp_buffers[0], tmp_buffers[1], tmp_buffers[2], width, height, win_size);
    clEnqueueNDRangeKernel(ctx.queue, preprocessing_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &precompute_start);

    //Run preprocessing for right image
    setPreprocessingKernelArgs(right_input_buffer, tmp_buffers[3], tmp_buffers[4], tmp_buffers[5], width, height, win_size);
    clEnqueueNDRangeKernel(ctx.queue, preprocessing_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &precompute_end);

    if (use_tiling) {
        //Round up image size for global work dimensions
        size_t global_work_size2[2] = {(size_t)((width  + TILE_SIZE - 1)/TILE_SIZE)*TILE_SIZE,
                                       (size_t)((height + TILE_SIZE - 1)/TILE_SIZE)*TILE_SIZE};

        //Tile is work group
        size_t local_work_size2[2] = {(size_t)TILE_SIZE, (size_t)TILE_SIZE};

        float min_zncc = -100000.0f;

        clEnqueueFillBuffer(ctx.queue, tmp_buffers[6], &min_zncc, sizeof(float), 0, width*height*sizeof(float), 0, NULL, NULL);

        //Tiled kernel uses batches, so that we don't need to launch for each disparity value
        for (int d = min_disparity; d <= max_disparity; d += BATCH_SIZE) {
            //Set argument for tiled disparity kernel
            setDisparityKernelArgs2(left_output_buffer,
                                    tmp_buffers[0], tmp_buffers[3],
                                    tmp_buffers[1], tmp_buffers[4],
                                    tmp_buffers[2], tmp_buffers[5],
                                    width, height, win_size, d, std::min(d + BATCH_SIZE, max_disparity), tmp_buffers[6]);

            clEnqueueNDRangeKernel(ctx.queue, disparity_kernel2, 2, NULL, global_work_size2, local_work_size2, 0, NULL, (d == min_disparity ? &disparity_start : NULL));
        }

        //Same for right to left
        clEnqueueFillBuffer(ctx.queue, tmp_buffers[6], &min_zncc, sizeof(float), 0, width*height*sizeof(float), 0, NULL, NULL);

        for (int d = -max_disparity; d <= -min_disparity; d += BATCH_SIZE) {
            setDisparityKernelArgs2(right_output_buffer,
                                    tmp_buffers[3], tmp_buffers[0],
                                    tmp_buffers[4], tmp_buffers[1],
                                    tmp_buffers[5], tmp_buffers[2],
                                    width, height, win_size, d, std::min(d + BATCH_SIZE, -min_disparity), tmp_buffers[6]);

            clEnqueueNDRangeKernel(ctx.queue, disparity_kernel2, 2, NULL, global_work_size2, local_work_size2, 0, NULL, (d + BATCH_SIZE >= -min_disparity ? &disparity_end : NULL));
        }
    } else {

        //Run disparity algorithm for left to right image
        setDisparityKernelArgs(left_output_buffer,
                            tmp_buffers[0], tmp_buffers[3],
                            tmp_buffers[1], tmp_buffers[4],
                            tmp_buffers[2], tmp_buffers[5],
                            width, height, win_size, min_disparity, max_disparity);

        clEnqueueNDRangeKernel(ctx.queue, disparity_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &disparity_start);

        //Run disparity algorithm for right to left image
        setDisparityKernelArgs(right_output_buffer,
                            tmp_buffers[3], tmp_buffers[0],
                            tmp_buffers[4], tmp_buffers[1],
                            tmp_buffers[5], tmp_buffers[2],
                            width, height, win_size, -max_disparity, -min_disparity);

        clEnqueueNDRangeKernel(ctx.queue, disparity_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &disparity_end);
    }


    //Read output buffers from GPU mem
    //Use blocking flag, which means that call will blocked until everything is executed
    clEnqueueReadBuffer(ctx.queue, left_output_buffer, CL_TRUE, 0, width*height*sizeof(uint8_t), result.leftToRight.pixels.data(), 0, NULL, &read_result_start);
    clEnqueueReadBuffer(ctx.queue, right_output_buffer, CL_TRUE, 0, width*height*sizeof(uint8_t), result.rightToLeft.pixels.data(), 0, NULL, &read_result_end);

    //Temporary input buffers are allocated when input images are not OpenCLImages. Lets release memory
    if (temp_input_buffers) {
        clReleaseMemObject(left_input_buffer);
        clReleaseMemObject(right_input_buffer);
    }

    prof.sectionTimes("disparity_precompute", ctx.getProfilingResults(precompute_start, precompute_end));
    prof.sectionTimes("disparity_kernels", ctx.getProfilingResults(disparity_start, disparity_end));
    prof.sectionTimes("disparity_read_result", ctx.getProfilingResults(read_result_start, read_result_end));

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



void OpenCLDisparityEstimator::setDisparityKernelArgs2(cl_mem disp,
                                                        cl_mem left_img,     cl_mem right_img,
                                                        cl_mem left_stdmean, cl_mem right_stdmean,
                                                        cl_mem left_stddev,  cl_mem right_stddev,
                                                        int width, int height,
                                                        int win_size,
                                                        int min_disp, int max_disp, cl_mem best_zncc)
{
    int halo_size = TILE_SIZE + (win_size >> 1)*2;
    int right_halo_width = TILE_SIZE + BATCH_SIZE + (win_size >> 1)*2;

    clSetKernelArg(disparity_kernel2, 0, sizeof(cl_mem), &disp);

    clSetKernelArg(disparity_kernel2, 1, sizeof(cl_mem), &left_img);
    clSetKernelArg(disparity_kernel2, 2, sizeof(cl_mem), &right_img);
    clSetKernelArg(disparity_kernel2, 3, sizeof(cl_mem), &left_stdmean);
    clSetKernelArg(disparity_kernel2, 4, sizeof(cl_mem), &right_stdmean);
    clSetKernelArg(disparity_kernel2, 5, sizeof(cl_mem), &left_stddev);
    clSetKernelArg(disparity_kernel2, 6, sizeof(cl_mem), &right_stddev);

    clSetKernelArg(disparity_kernel2, 7,  sizeof(int), &width);
    clSetKernelArg(disparity_kernel2, 8,  sizeof(int), &height);
    clSetKernelArg(disparity_kernel2, 9,  sizeof(int), &win_size);
    clSetKernelArg(disparity_kernel2, 10, halo_size*halo_size       *sizeof(float), NULL);
    clSetKernelArg(disparity_kernel2, 11, halo_size*right_halo_width*sizeof(float), NULL);

    clSetKernelArg(disparity_kernel2, 12, sizeof(int), &min_disp);
    clSetKernelArg(disparity_kernel2, 13, sizeof(int), &max_disp);
    clSetKernelArg(disparity_kernel2, 14, sizeof(cl_mem), &best_zncc);
}




void OpenCLDisparityEstimator::deallocateBuffers()
{
    clReleaseMemObject(left_output_buffer);
    clReleaseMemObject(right_output_buffer);

    for (int i = 0; i < 7; i++) {
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

    auto &ctx = OpenCLContext::getInstance();

    //Allocate buffers
    cl_int err;

    left_output_buffer  = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, w*h*sizeof(uint8_t), NULL, &err);
    right_output_buffer = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, w*h*sizeof(uint8_t), NULL, &err);

    for (int i = 0; i < 7; i++) {
        tmp_buffers[i] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, w*h*sizeof(float), NULL, &err);
    }

    image_w = w;
    image_h = h;
}


}
