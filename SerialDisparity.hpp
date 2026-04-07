#pragma once

#include "Image.hpp"
#include "Disparity.hpp"

namespace Disparity
{
    class SerialDisparityEstimator: public DisparityEstimator
    {
    public:
        DisparityResult estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity);
    };
}
