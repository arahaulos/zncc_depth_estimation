#include <iostream>
#include "OpenCLImage.hpp"
#include "OpenCLPostProcessing.hpp"
#include "ThreadPool.hpp"
#include "Utils.hpp"
#include "Image.hpp"

#include "SerialDisparity.hpp"
#include "MultiThreadedDisparity.hpp"
#include "OpenCLDisparity.hpp"
#include "PostProcessing.hpp"


std::unique_ptr<Image> estimateDepthMap(std::shared_ptr<Disparity::DisparityEstimator> disp_estimator,
                                        std::shared_ptr<Disparity::PostProcessor> post_processor,
                                        Image &left, Image &right,
                                        int win_size, int min_disparity, int max_disparity)
{
    auto &prof = Utils::Profiler::getInstance();

    auto frame_pg = prof.section("frame"); //End time is recorded when frame_pg goes outside of scope (destructor called)

    Disparity::DisparityResult disparity = disp_estimator->estimate(left, right, win_size, min_disparity, max_disparity);
    std::unique_ptr<Image> cc_result = post_processor->crossCheck(disparity, min_disparity, max_disparity, 5);
    std::unique_ptr<Image> eroded = post_processor->erosion(*cc_result);

    return post_processor->fill(*eroded);
}



enum ImageImplementation {IMAGE_NORMAL, IMAGE_OPENCL};
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


void benchSerialImplementation(int window_size = 9)
{
    auto &prof = Utils::Profiler::getInstance();

    //Load test images
    auto [left, right] = loadTestImages("images/im0.png", "images/im1.png", IMAGE_NORMAL);

    std::cout << "Test image size: " << left->width << "x" << right->height << std::endl;
    std::cout << "Window size: " << window_size << "x" << window_size << std::endl;

    auto disp_estimator = std::make_shared<Disparity::SerialDisparityEstimator>();
    auto post_processor = std::make_shared<Disparity::PostProcessor>();

    std::shared_ptr<Image> depth_map;

    for (int i = 0; i < 10; i++) {
        depth_map = estimateDepthMap(disp_estimator, post_processor, *left, *right, 9, 0, 65);
    }

    std::cout << "Avg frametime: " << prof.getSectionAverageTime("frame") << " ms" << std::endl;
    prof.printAllAverageTimes();
    prof.clear();

    depth_map->save("images/disp.png");
}



void benchMultiThreadedImpelementation(std::vector<int> cnts, int window_size = 9)
{
    auto &prof = Utils::Profiler::getInstance();

    //Load test images
    auto [left, right] = loadTestImages("images/im0.png", "images/im1.png", IMAGE_NORMAL);

    std::cout << "Test image size: " << left->width << "x" << right->height << std::endl;
    std::cout << "Window size: " << window_size << "x" << window_size << std::endl;

    auto disp_estimator = std::make_shared<Disparity::MultiThreadedDisparityEstimator>();
    auto post_processor = std::make_shared<Disparity::PostProcessor>();

    std::shared_ptr<Image> depth_map;
    for (int threads : cnts) {

        disp_estimator->getThreadPool().setThreads(threads);

        for (int i = 0; i < 100; i++) {
            depth_map = estimateDepthMap(disp_estimator, post_processor, *left, *right, 9, 0, 65);
        }

        std::cout << "\n\nThreads: " << threads << "  Avg frametime: " << prof.getSectionAverageTime("frame") << " ms" << std::endl;
        prof.printAllAverageTimes();
        prof.clear();
    }

    depth_map->save("images/disp.png");
}


void benchOpenCLImplementation(int window_size = 9) {
    auto &prof = Utils::Profiler::getInstance();

    auto [left, right] = loadTestImages("images/im0.png", "images/im1.png", IMAGE_OPENCL);

    std::cout << "Test image size: " << left->width << "x" << right->height << std::endl;
    std::cout << "Window size: " << window_size << "x" << window_size << std::endl;

    auto disp_estimator = std::make_shared<Disparity::OpenCLDisparityEstimator>();
    auto post_processor = std::make_shared<Disparity::OpenCLPostProcessor>();

    std::shared_ptr<Image> depth_map;

    //First test using non tiled implementation
    disp_estimator->enableTiling(false);

    for (int i = 0; i < 1000; i++) {
        depth_map = estimateDepthMap(disp_estimator, post_processor, *left, *right, 9, 0, 65);
    }

    std::cout << "\n\nNon tiled implementation: " << std::endl;

    //depth_map.save("images/disp.png");

    prof.printAllAverageTimes();
    prof.clear();


    //Test tiled implementation
    disp_estimator->enableTiling(true);
    for (int i = 0; i < 1000; i++) {
        depth_map = estimateDepthMap(disp_estimator, post_processor, *left, *right, 9, 0, 65);
    }

    std::cout << "\n\nTiled implementation: " << std::endl;
    prof.printAllAverageTimes();
    prof.clear();

    depth_map->save("images/disp.png");
}


int main()
{
    //benchSerialImplementation();
    //benchMultiThreadedImpelementation({32});
    benchOpenCLImplementation();

    return 0;
}


