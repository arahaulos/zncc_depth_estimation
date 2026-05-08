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

    preprocessing_kernel = ctx.createKernel("preprocess");
    disparity_kernel = ctx.createKernel("disparity");
    disparity_kernel2 = ctx.createKernel("disparity2");

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

    int width = left.width;
    int height = left.height;

    //Make sure that window size is odd
    win_size = win_size | 0x1;

    auto leftToRight = std::make_shared<OpenCLImage>();
    auto rightToLeft = std::make_shared<OpenCLImage>();

    leftToRight->allocate(width, height, 1);
    rightToLeft->allocate(width, height, 1);

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
    setKernelArgs(preprocessing_kernel, 0, left_input_buffer, tmp_buffers[0], tmp_buffers[1], tmp_buffers[2], width, height, win_size);
    clEnqueueNDRangeKernel(ctx.queue, preprocessing_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &precompute_start);

    //Run preprocessing for right image
    setKernelArgs(preprocessing_kernel, 0, right_input_buffer, tmp_buffers[3], tmp_buffers[4], tmp_buffers[5], width, height, win_size);
    clEnqueueNDRangeKernel(ctx.queue, preprocessing_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &precompute_end);

    if (use_tiling) {
        //Round up image size for global work dimensions
        size_t global_work_size2[2] = {(size_t)((width  + TILE_SIZE - 1)/TILE_SIZE)*TILE_SIZE,
                                       (size_t)((height + TILE_SIZE - 1)/TILE_SIZE)*TILE_SIZE};

        //Tile is work group
        size_t local_work_size2[2] = {(size_t)TILE_SIZE, (size_t)TILE_SIZE};


        //Calculate width of local buffers needed
        int left_halo_width  = TILE_SIZE + (win_size >> 1)*2; //Height for both halos is same
        int right_halo_width = TILE_SIZE + BATCH_SIZE + (win_size >> 1)*2;

        //Calculate sizes of local buffers
        size_t left_halo_size = left_halo_width*left_halo_width*sizeof(float);
        size_t right_halo_size = left_halo_width*right_halo_width*sizeof(float);

        float min_zncc = -100000.0f;

        clEnqueueFillBuffer(ctx.queue, tmp_buffers[6], &min_zncc, sizeof(float), 0, width*height*sizeof(float), 0, NULL, NULL);

        //Tiled kernel uses batches, so that we don't need to launch for each disparity value
        for (int d = min_disparity; d <= max_disparity; d += BATCH_SIZE) {
            //Set argument for tiled disparity kernel
            setKernelArgs(disparity_kernel2, 0,
                          leftToRight->getOpenCLBuffer(),
                          tmp_buffers[0], tmp_buffers[3],
                          tmp_buffers[1], tmp_buffers[4],
                          tmp_buffers[2], tmp_buffers[5],
                          width, height, win_size,
                          LocalBuffer{left_halo_size}, LocalBuffer{right_halo_size},
                          d, std::min(d + BATCH_SIZE, max_disparity + 1), tmp_buffers[6]);

            clEnqueueNDRangeKernel(ctx.queue, disparity_kernel2, 2, NULL, global_work_size2, local_work_size2, 0, NULL, (d == min_disparity ? &disparity_start : NULL));
        }

        //Same for right to left
        clEnqueueFillBuffer(ctx.queue, tmp_buffers[6], &min_zncc, sizeof(float), 0, width*height*sizeof(float), 0, NULL, NULL);

        for (int d = -max_disparity; d <= -min_disparity; d += BATCH_SIZE) {
            setKernelArgs(disparity_kernel2, 0,
                          rightToLeft->getOpenCLBuffer(),
                          tmp_buffers[3], tmp_buffers[0],
                          tmp_buffers[4], tmp_buffers[1],
                          tmp_buffers[5], tmp_buffers[2],
                          width, height, win_size,
                          LocalBuffer{left_halo_size}, LocalBuffer{right_halo_size},
                          d, std::min(d + BATCH_SIZE, -min_disparity + 1), tmp_buffers[6]);

            clEnqueueNDRangeKernel(ctx.queue, disparity_kernel2, 2, NULL, global_work_size2, local_work_size2, 0, NULL, (d + BATCH_SIZE >= -min_disparity ? &disparity_end : NULL));
        }
    } else {

        //Run disparity algorithm for left to right image
        setKernelArgs(disparity_kernel, 0,
                      leftToRight->getOpenCLBuffer(),
                      tmp_buffers[0], tmp_buffers[3],
                      tmp_buffers[1], tmp_buffers[4],
                      tmp_buffers[2], tmp_buffers[5],
                      width, height, win_size, min_disparity, max_disparity);

        clEnqueueNDRangeKernel(ctx.queue, disparity_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &disparity_start);

        //Run disparity algorithm for right to left image
        setKernelArgs(disparity_kernel, 0,
                      rightToLeft->getOpenCLBuffer(),
                      tmp_buffers[3], tmp_buffers[0],
                      tmp_buffers[4], tmp_buffers[1],
                      tmp_buffers[5], tmp_buffers[2],
                      width, height, win_size, -max_disparity, -min_disparity);

        clEnqueueNDRangeKernel(ctx.queue, disparity_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &disparity_end);
    }

    //Temporary input buffers are allocated when input images are not OpenCLImages. Lets release memory
    if (temp_input_buffers) {
        clReleaseMemObject(left_input_buffer);
        clReleaseMemObject(right_input_buffer);
    }

    clWaitForEvents(1, &disparity_end);

    prof.sectionTimes("disparity_precompute", ctx.getProfilingResults(precompute_start, precompute_end));
    prof.sectionTimes("disparity_kernels", ctx.getProfilingResults(disparity_start, disparity_end));

    return {leftToRight, rightToLeft};
}


void OpenCLDisparityEstimator::deallocateBuffers()
{
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

    for (int i = 0; i < 7; i++) {
        tmp_buffers[i] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, w*h*sizeof(float), NULL, &err);
    }

    image_w = w;
    image_h = h;
}


}
