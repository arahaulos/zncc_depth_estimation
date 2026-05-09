#pragma once

#include "Image.hpp"
#include "Disparity.hpp"


namespace Disparity
{

class PostProcessor
{

public:
    virtual std::unique_ptr<Image> crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff);

    virtual std::unique_ptr<Image> erosion(Image &in);

    virtual std::unique_ptr<Image> fill(Image &in);
};

}
