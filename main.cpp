#include <iostream>
#include <iomanip>
#include "OpenCLImage.hpp"
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

enum ImageImplementation {
    IMAGE_NORMAL,
    IMAGE_OPENCL
};


Image estimateDepthMap(std::shared_ptr<Disparity::DisparityEstimator> disp_estimator, Image &left, Image &right, int win_size, int min_disparity, int max_disparity)
{
    Disparity::DisparityResult disparity = disp_estimator->estimate(left, right, win_size, min_disparity, max_disparity);

    //disparity.leftToRight.save("left_disp.png");
    //disparity.rightToLeft.save("right_disp.png");

    PostProcessing::CrossCheckResult cc_result = PostProcessing::crossCheck(disparity, min_disparity, max_disparity, 5);

    //cc_result.output.save("disp2.png");

    return PostProcessing::fill(cc_result);
}


std::shared_ptr<Disparity::DisparityEstimator> createDisparityEstimator(DisparityEstimatorImplementation implementation)
{

    if (implementation == DISPARITY_SERIAL) {

        return std::make_shared<Disparity::SerialDisparityEstimator>();

    } else if (implementation == DISPARITY_MULTITHREADED) {

        return std::make_shared<Disparity::MultiThreadedDisparityEstimator>();

    } else if (implementation == DISPARITY_OPENCL) {
        return std::make_shared<Disparity::OpenCLDisparityEstimator>();
    } else {
        return std::shared_ptr<Disparity::DisparityEstimator>();
    }
}


std::pair<std::unique_ptr<Image>, std::unique_ptr<Image>> loadTestImages(std::string left_filepath, std::string right_filepath, ImageImplementation image_type)
{
    std::array<float, 3> graycoeff {0.2126, 0.7152, 0.0722};

    std::unique_ptr<Image> left;
    std::unique_ptr<Image> right;

    if (image_type == IMAGE_NORMAL) {
        left = std::make_unique<Image>(left_filepath);
        right = std::make_unique<Image>(right_filepath);
    } else if (image_type == IMAGE_OPENCL) {
        left = std::make_unique<OpenCLImage>(left_filepath);
        right = std::make_unique<OpenCLImage>(right_filepath);
    }

    left->downsample(2);
    right->downsample(2);

    left->convertToGrayscale(graycoeff);
    right->convertToGrayscale(graycoeff);

    return {std::move(left), std::move(right)};
}


void benchMultiThreadedImpelementation(std::vector<int> cnts, int window_size = 9)
{
    //Load test images
    auto [left, right] = loadTestImages("images/im0.png", "images/im1.png", IMAGE_NORMAL);

    std::cout << "Test image size: " << left->width << "x" << right->height << std::endl;
    std::cout << "Window size: " << window_size << "x" << window_size << std::endl;

    std::shared_ptr<Disparity::MultiThreadedDisparityEstimator> disp_estimator = std::make_shared<Disparity::MultiThreadedDisparityEstimator>();

    for (int threads : cnts) {

        disp_estimator->getThreadPool().setThreads(threads);

        uint64_t t0 = Utils::timestampUs();

        for (int i = 0; i < 10; i++) {
            Image depth_map = estimateDepthMap(disp_estimator, *left, *right, 9, 0, 65);
        }

        uint64_t us = (Utils::timestampUs() - t0) / 10;

        std::cout << "Threads: " << threads << "  Avg frametime: " << std::setprecision(4) << (static_cast<double>(us) / 1000) << " ms" << std::endl;
    }
}




int main()
{
    //benchMultiThreadedImpelementation({1, 2, 4, 8, 16, 32, 64});

    //return 0;

    int window_size = 9;

    std::shared_ptr<Disparity::DisparityEstimator> disp_estimator = createDisparityEstimator(DISPARITY_OPENCL);

    auto [left, right] = loadTestImages("images/im0.png", "images/im1.png", IMAGE_OPENCL);

    std::cout << "Test image size: " << left->width << "x" << left->height << std::endl;
    std::cout << "Window size: " << window_size << "x" << window_size << std::endl;

    Image depth_map;

    uint64_t t0 = Utils::timestampUs();

    for (int i = 0; i < 100; i++) {
        depth_map = estimateDepthMap(disp_estimator, *left, *right, window_size, 0, 65);
    }

    uint64_t us = (Utils::timestampUs() - t0) / 100;

    std::cout << "Time: " << static_cast<double>(us) / 1000 << " ms" << std::endl;

    depth_map.save("images/disp.png");

    return 0;
}
