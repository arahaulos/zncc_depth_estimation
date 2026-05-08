#pragma once

#include "Image.hpp"
#include "CL/cl.h"
#include "Disparity.hpp"
#include "PostProcessing.hpp"


namespace Disparity
{

class OpenCLPostProcessor: public PostProcessor
{
public:
    OpenCLPostProcessor();
    ~OpenCLPostProcessor();

    std::shared_ptr<Image> crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff) override;

    std::shared_ptr<Image> erosion(Image &in) override;
private:
    cl_kernel crosscheck_kernel;
    cl_kernel erosion_kernel;
};

}
