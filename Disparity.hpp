#pragma once

#include "Image.hpp"
#include <memory>

namespace Disparity
{

struct DisparityResult
{
    std::shared_ptr<Image> leftToRight;
    std::shared_ptr<Image> rightToLeft;
};

class DisparityEstimator
{
    //This is just interface
    //All implementations implement this
public:
    virtual DisparityResult estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity) = 0;
};

}
