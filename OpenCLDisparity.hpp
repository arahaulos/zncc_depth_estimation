#pragma once

#include <memory>
#include "Image.hpp"
#include "Disparity.hpp"
#include "OpenCLContext.hpp"

namespace Disparity
{


class OpenCLDisparityEstimator: public DisparityEstimator
{
public:
    OpenCLDisparityEstimator();
    ~OpenCLDisparityEstimator();

    DisparityResult estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity);

    void enableTiling(bool e) {
        use_tiling = e;
    }

private:
    void deallocateBuffers();

    void checkBufferSize(int w, int h);

    cl_kernel  preprocessing_kernel;
    cl_kernel  preprocessing_kernel2;
    cl_kernel  disparity_kernel;
    cl_kernel  disparity_kernel2;

    cl_mem     tmp_buffers[7];

    int image_w;
    int image_h;

    bool use_tiling;
};

};
