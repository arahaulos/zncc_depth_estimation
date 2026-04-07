#pragma once

#include "Image.hpp"
#include "Disparity.hpp"
#include <vector>
#include <tuple>


namespace PostProcessing
{
    typedef std::vector<std::pair<int,int>> OccludedPixels;

    struct CrossCheckResult
    {
        OccludedPixels occluded;
        Image output;
    };

    CrossCheckResult crossCheck(Disparity::DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff);

    Image fill(CrossCheckResult &result);
}
