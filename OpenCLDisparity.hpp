#pragma once

#include <memory>
#include "Image.hpp"
#include "Disparity.hpp"
#include "OpenCLContext.hpp"

namespace Disparity
{


class OpenCLDisparityEstimator: public DisparityEstimator
{
public:
    OpenCLDisparityEstimator(std::shared_ptr<OpenCLContext> c);
    ~OpenCLDisparityEstimator();

    DisparityResult estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity);

private:
    void setPreprocessingKernelArgs(cl_mem input_img,
                                    cl_mem output_img,
                                    cl_mem output_stdmean,
                                    cl_mem output_stddev,
                                    int width,
                                    int height,
                                    int win_size);



    void setDisparityKernelArgs(cl_mem disp,
                                cl_mem left_img,     cl_mem right_img,
                                cl_mem left_stdmean, cl_mem right_stdmean,
                                cl_mem left_stddev,  cl_mem right_stddev,
                                int width, int height,
                                int win_size,
                                int min_disparity, int max_disparity);


    void deallocateBuffers();

    void checkBufferSize(int w, int h);

    cl_program program;

    cl_kernel  preprocessing_kernel;
    cl_kernel  disparity_kernel;

    cl_mem     left_input_buffer;
    cl_mem     right_input_buffer;

    cl_mem     left_output_buffer;
    cl_mem     right_output_buffer;

    cl_mem     tmp_buffers[6];


    int image_w;
    int image_h;

    std::shared_ptr<OpenCLContext> context;
};

};
