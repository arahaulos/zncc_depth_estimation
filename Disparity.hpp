#pragma once

#include "Image.hpp"

namespace Disparity
{

struct DisparityResult
{
    Image leftToRight;
    Image rightToLeft;
};

class DisparityEstimator
{
public:
    virtual DisparityResult estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity) = 0;
};

}
