#pragma once

#include "Image.hpp"
#include "CL/cl.h"
#include "Disparity.hpp"
#include "PostProcessing.hpp"


namespace Disparity
{

class OpenCLPostProcessor: public PostProcessor
{
    //Inherits interface from CPU postprocessing implementation
    //Implements same functionality using opencl

public:
    OpenCLPostProcessor();
    ~OpenCLPostProcessor();

    std::unique_ptr<Image> crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff) override;

    std::unique_ptr<Image> erosion(Image &in) override;

    std::unique_ptr<Image> fill(Image &in) override;
private:
    cl_kernel crosscheck_kernel;
    cl_kernel erosion_kernel;
    cl_kernel fill_kernel;
};

}
