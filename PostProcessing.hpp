#pragma once

#include "Image.hpp"
#include "Disparity.hpp"


namespace Disparity
{

class PostProcessor
{

    //This class defines interface for postprocessing steps
    //It also has (serial) CPU implementation for each step

public:
    virtual std::unique_ptr<Image> crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff);

    virtual std::unique_ptr<Image> erosion(Image &in);

    virtual std::unique_ptr<Image> fill(Image &in);
};

}
