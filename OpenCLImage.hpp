#pragma once

#include <memory.h>
#include "CL/cl.h"
#include "Image.hpp"
#include "OpenCLContext.hpp"


struct OpenCLImage : public Image
{
    OpenCLImage();
    OpenCLImage(int w, int h);
    OpenCLImage(std::string filepath);
    ~OpenCLImage();

    OpenCLImage(const Image &img);
    OpenCLImage(const OpenCLImage &climg);


    friend void swap(OpenCLImage &a, OpenCLImage &b);

    OpenCLImage& operator=(OpenCLImage other); //Uses copy and swap idiom
    OpenCLImage& operator=(const Image &other);

    void allocate(int w, int h, int bpp = 1) override;

    void downsample() override;
    void downsample(int times) override;
    void filter2d(const uint8_t *kernel, int kernel_size);
    void resize(int new_width, int new_height) override;
    void convertToGrayscale(const std::array<float, 3> &coeff) override;

    cl_mem getOpenCLBuffer() {
        return buffer;
    }

    void copyDeviceToHost() override;
    void copyHostToDevice();

    int getNeededBufferSize() {
        return width*height*bytes_per_pixel;
    }

private:
    void loadKernels();

    int allocated_buffer_size;
    cl_mem buffer;

    cl_kernel filter2d_kernel;
    cl_kernel resize_kernel;
    cl_kernel grayscale_kernel;
};
