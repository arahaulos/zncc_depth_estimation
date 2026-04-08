#pragma once

#include "Image.hpp"
#include "ThreadPool.hpp"
#include "Disparity.hpp"


namespace Disparity
{

    class MultiThreadedDisparityEstimator: public DisparityEstimator
    {
    public:
        DisparityResult estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity);

        ThreadPool &getThreadPool() {
            return pool;
        }
    private:
        ThreadPool pool;
    };
}
