#pragma once

#include "Image.hpp"
#include "Disparity.hpp"


namespace Disparity
{

class PostProcessor
{

public:
    virtual std::shared_ptr<Image> crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff);

    virtual std::shared_ptr<Image> erosion(Image &in);

    virtual std::shared_ptr<Image> fill(Image &in);
};

}
