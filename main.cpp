#include <iostream>
#include "Utils.hpp"
#include "Image.hpp"

#include "SerialDisparity.hpp"
#include "MultiThreadedDisparity.hpp"
#include "OpenCLDisparity.hpp"
#include "PostProcessing.hpp"


enum DisparityEstimatorImplementation {
    DISPARITY_SERIAL,
    DISPARITY_MULTITHREADED,
    DISPARITY_OPENCL
};


Image estimateDepthMap(std::shared_ptr<Disparity::DisparityEstimator> disp_estimator, Image &left, Image &right, int win_size, int min_disparity, int max_disparity)
{
    Disparity::DisparityResult disparity = disp_estimator->estimate(left, right, win_size, min_disparity, max_disparity);

    PostProcessing::CrossCheckResult cc_result = PostProcessing::cross_check(disparity, min_disparity, max_disparity, 5);

    return PostProcessing::fill(cc_result);
}


std::shared_ptr<Disparity::DisparityEstimator> createDisparityEstimator(DisparityEstimatorImplementation implementation)
{

    if (implementation == DISPARITY_SERIAL) {

        return std::make_shared<Disparity::SerialDisparityEstimator>();

    } else if (implementation == DISPARITY_MULTITHREADED) {

        return std::make_shared<Disparity::MultiThreadedDisparityEstimator>();

    } else if (implementation == DISPARITY_OPENCL) {

        static std::shared_ptr<OpenCLContext> open_cl_context;
        if (!open_cl_context) {
            open_cl_context = std::make_shared<OpenCLContext>();
        }

        return std::make_shared<Disparity::OpenCLDisparityEstimator>(open_cl_context);
    } else {

        std::cout << "Why?!" << std::endl;

        return std::shared_ptr<Disparity::DisparityEstimator>();
    }
}


std::pair<Image, Image> loadTestImages(std::string left_filepath, std::string right_filepath)
{
    std::array<float, 3> graycoeff {0.2126, 0.7152, 0.0722};

    Image left(left_filepath);
    Image right(right_filepath);

    return {left.downsample(2).toGrayscale(graycoeff), right.downsample(2).toGrayscale(graycoeff)};
}


int main()
{
    std::shared_ptr<Disparity::DisparityEstimator> disp_estimator = createDisparityEstimator(DISPARITY_MULTITHREADED);

    auto [left, right] = loadTestImages("im0.png", "im1.png");

    Image depth_map;

    uint64_t t0 = Utils::timestamp_us();

    for (int i = 0; i < 10; i++) {
        depth_map = estimateDepthMap(disp_estimator, left, right, 9, 0, 65);
    }


    uint64_t us = (Utils::timestamp_us() - t0) / 10;

    std::cout << "Time: " << static_cast<double>(us) / 1000000 << "s" << std::endl;

    depth_map.save("disp.png");

    return 0;
}
