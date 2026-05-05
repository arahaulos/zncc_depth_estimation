#include "OpenCLImage.hpp"
#include "OpenCLContext.hpp"
#include <CL/cl.h>
#include <iostream>


OpenCLImage::OpenCLImage()
{
    loadKernels();
    allocated_buffer_size = -1;
}

OpenCLImage::~OpenCLImage()
{
    if (allocated_buffer_size > 0) {
        clReleaseMemObject(buffer);
    }
    clReleaseKernel(filter2d_kernel);
    clReleaseKernel(resize_kernel);
    clReleaseKernel(grayscale_kernel);
}




OpenCLImage::OpenCLImage(int w, int h)
{
    loadKernels();
    allocated_buffer_size = -1;
    allocate(w, h);
}

OpenCLImage::OpenCLImage(std::string path)
{
    loadKernels();
    allocated_buffer_size = -1;
    load(path);

    copyHostToDevice();
}

OpenCLImage::OpenCLImage(const Image &img)
{
    loadKernels();
    allocated_buffer_size = -1;

    pixels = img.pixels;
    width = img.width;
    height = img.height;
    bytes_per_pixel = img.bytes_per_pixel;

    copyHostToDevice();
}

OpenCLImage::OpenCLImage(const OpenCLImage &climg)
{
    loadKernels();

    pixels = climg.pixels;
    width = climg.width;
    height = climg.height;
    bytes_per_pixel = climg.bytes_per_pixel;

    allocated_buffer_size = climg.allocated_buffer_size;

    auto &ctx = OpenCLContext::getInstance();

    cl_int err;
    buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, allocated_buffer_size, NULL, &err);

    clEnqueueCopyBuffer(ctx.queue, climg.buffer,buffer, 0, 0, allocated_buffer_size, 0, nullptr, nullptr);
}

void OpenCLImage::loadKernels()
{
    auto &ctx = OpenCLContext::getInstance();
    cl_int err;

    grayscale_kernel = clCreateKernel(ctx.program, "image_to_grayscale", &err);
    resize_kernel    = clCreateKernel(ctx.program, "image_resize", &err);
    filter2d_kernel = clCreateKernel(ctx.program, "image_filter2d", &err);
}

void swap(OpenCLImage &a, OpenCLImage &b)
{
    std::swap(a.width, b.width);
    std::swap(a.height, b.height);
    std::swap(a.bytes_per_pixel, b.bytes_per_pixel);
    std::swap(a.pixels, b.pixels);

    std::swap(a.allocated_buffer_size, b.allocated_buffer_size);
    std::swap(a.buffer, b.buffer);
    std::swap(a.filter2d_kernel, b.filter2d_kernel);
    std::swap(a.resize_kernel, b.resize_kernel);
    std::swap(a.grayscale_kernel, b.grayscale_kernel);
}

OpenCLImage& OpenCLImage::operator=(OpenCLImage other)
{
    swap(*this, other);

    return *this;
}

OpenCLImage& OpenCLImage::operator=(const Image &img)
{
    pixels = img.pixels;
    width = img.width;
    height = img.height;
    bytes_per_pixel = img.bytes_per_pixel;

    copyHostToDevice();

    return *this;
}

void OpenCLImage::filter2d(const uint8_t *kernel, int kernel_size)
{
    //__kernel void image_filtering(__global const uchar *input_img, __global uchar *output_img, const int width, const int height, const int bytes_per_pixel, __global const uchar *filter, int filter_size)

    auto &ctx = OpenCLContext::getInstance();
    size_t global_work_size[2] = {(size_t)width, (size_t)height};

    cl_int err;
    cl_mem kernel_buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, kernel_size*kernel_size, NULL, &err);
    cl_mem new_buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, width*height*bytes_per_pixel, NULL, &err);

    clEnqueueWriteBuffer(ctx.queue, kernel_buffer, CL_FALSE, 0, kernel_size*kernel_size, kernel, 0, NULL, NULL);

    clSetKernelArg(filter2d_kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(filter2d_kernel, 1, sizeof(cl_mem), &new_buffer);

    clSetKernelArg(filter2d_kernel, 2, sizeof(int), &width);
    clSetKernelArg(filter2d_kernel, 3, sizeof(int), &height);
    clSetKernelArg(filter2d_kernel, 4, sizeof(int), &bytes_per_pixel);

    clSetKernelArg(filter2d_kernel, 5, sizeof(cl_mem), &kernel_buffer);
    clSetKernelArg(filter2d_kernel, 6, sizeof(int),    &kernel_size);

    clEnqueueNDRangeKernel(ctx.queue, filter2d_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    clReleaseMemObject(kernel_buffer);
    clReleaseMemObject(buffer);

    buffer = new_buffer;
}

void OpenCLImage::downsample()
{
    static const std::array<uint8_t, 5*5> kernel = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1,
    };

    filter2d(kernel.data(), 5);
    resize(width/2, height/2);
}

void OpenCLImage::downsample(int times)
{
    for (int i = 0; i < times; i++) {
        downsample();
    }
}


void OpenCLImage::resize(int new_width, int new_height)
{
    //__kernel void image_resize(__global const uchar *input_img, __global uchar *output_img, const int width, const int height, const int out_width, const int out_height, const int bytes_per_pixel)

    auto &ctx = OpenCLContext::getInstance();
    size_t global_work_size[2] = {(size_t)new_width, (size_t)new_height};

    cl_int err;
    cl_mem new_buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, new_width*new_height*bytes_per_pixel, NULL, &err);

    clSetKernelArg(resize_kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(resize_kernel, 1, sizeof(cl_mem), &new_buffer);

    clSetKernelArg(resize_kernel, 2, sizeof(int), &width);
    clSetKernelArg(resize_kernel, 3, sizeof(int), &height);
    clSetKernelArg(resize_kernel, 4, sizeof(int), &new_width);
    clSetKernelArg(resize_kernel, 5, sizeof(int), &new_height);
    clSetKernelArg(resize_kernel, 6, sizeof(int), &bytes_per_pixel);

    clEnqueueNDRangeKernel(ctx.queue, resize_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    clReleaseMemObject(buffer);

    width = new_width;
    height = new_height;
    pixels.resize(width*height*bytes_per_pixel);

    allocated_buffer_size = width*height*bytes_per_pixel;
    buffer = new_buffer;
}

void OpenCLImage::convertToGrayscale(const std::array<float, 3> &coeff)
{
    if (bytes_per_pixel == 1) {
        return;
    }

    auto &ctx = OpenCLContext::getInstance();
    size_t global_work_size[2] = {(size_t)width, (size_t)height};

    cl_int err;
    cl_mem new_buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, width*height, NULL, &err);

    clSetKernelArg(grayscale_kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(grayscale_kernel, 1, sizeof(cl_mem), &new_buffer);

    clSetKernelArg(grayscale_kernel, 2, sizeof(int), &width);
    clSetKernelArg(grayscale_kernel, 3, sizeof(int), &height);
    clSetKernelArg(grayscale_kernel, 4, sizeof(int), &bytes_per_pixel);
    clSetKernelArg(grayscale_kernel, 5, sizeof(float), &coeff[0]);
    clSetKernelArg(grayscale_kernel, 6, sizeof(float), &coeff[1]);
    clSetKernelArg(grayscale_kernel, 7, sizeof(float), &coeff[2]);

    clEnqueueNDRangeKernel(ctx.queue, grayscale_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    clReleaseMemObject(buffer);

    bytes_per_pixel = 1;
    pixels.resize(width*height);

    allocated_buffer_size = width*height;
    buffer = new_buffer;
}

void OpenCLImage::copyDeviceToHost()
{
    if (allocated_buffer_size < 0) {
        return;
    }

    auto &ctx = OpenCLContext::getInstance();

    int read_size = std::min(getNeededBufferSize(), allocated_buffer_size);

    clEnqueueReadBuffer(ctx.queue, buffer, CL_TRUE, 0, read_size, pixels.data(), 0, NULL, NULL);
}

void OpenCLImage::copyHostToDevice()
{
    auto &ctx = OpenCLContext::getInstance();

    cl_int err;
    if (getNeededBufferSize() != allocated_buffer_size) {
        if (allocated_buffer_size > 0) {
            clReleaseMemObject(buffer);
        }

        allocated_buffer_size = getNeededBufferSize();
        buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, allocated_buffer_size, NULL, &err);
    }

    clEnqueueWriteBuffer(ctx.queue, buffer, CL_TRUE, 0, allocated_buffer_size, pixels.data(), 0, NULL, NULL);
}






