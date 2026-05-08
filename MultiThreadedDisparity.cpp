#include <iostream>
#include <algorithm>

#include "MultiThreadedDisparity.hpp"
#include "Utils.hpp"
#include "kernels_cpu.hpp"

namespace Disparity
{

constexpr int lines_per_task = 2;


void disparityScanlines(Image &disp, int sy,
                         float *left_img, float *right_img,
                         float *left_stdmean, float *right_stdmean,
                         float *left_stddev, float *right_stddev,
                         int win_size, int min_disparity, int max_disparity)
{
    //This function is used to calculate disparity with precomputed stdmean and stddev buffers
    //sy is start scanline
    //global variable lines_per_task tells how many lines are processed with single call

    //This is supposed to be called from threadpool

    int width = disp.width;
    int height = disp.height;

    int ws = (win_size-1)/2;

    for (int y = sy; y < std::min(height, sy+lines_per_task); y++) {
        for (int x = 0; x < width; x++) {

            //Clip disparity range so that offset window cannot go outsize image
            int min_disp = std::max(min_disparity, x - width + ws + 1);
            int max_disp = std::min(max_disparity, x         - ws);

            //Check if disparity range is valid
            //And that window is inside image
            if (min_disp > max_disp ||
                x+ws >= width ||
                x-ws < 0) {

                disp.pixels[y * width + x] = 0;
                continue;
            }

            int best_disp = 0;
            float best_zncc = -std::numeric_limits<float>::infinity();


            for (int d = min_disp; d <= max_disp; d++) {

                #ifdef USE_AVX2

                float z = ZNCC_avx2(left_img, right_img,
                                    left_stdmean, right_stdmean,
                                    left_stddev, right_stddev,
                                    width, height,
                                    x, y, d, win_size);

                #else

                float z = ZNCC(left_img, right_img,
                               left_stdmean, right_stdmean,
                               left_stddev, right_stddev,
                               width, height,
                               x, y, d, win_size);

                #endif // USE_AVX2

                if (z > best_zncc) {
                    best_zncc = z;
                    best_disp = d;
                }
            }

            disp.pixels[y * width + x] = std::abs(best_disp);
        }
    }
}


void disparityPreprocessScanlines(int sy, int win_size, Image &input_img, std::vector<float> &img, std::vector<float> &stdmean, std::vector<float> &stddev)
{
    //This function is used to precompute window std mean and devs
    //sy is start scanline
    //global variable lines_per_task tells how many lines are processed with single call

    //This is supposed to be called from threadpool

    int width = input_img.width;
    int height = input_img.height;

    for (int y = sy; y < std::min(height, sy+lines_per_task); y++) {
        for (int x = 0; x < width; x++) {
            img[y * width + x] = static_cast<float>(input_img.pixels[y * width + x]);

            stdmean[y * width + x] = stdmean_kernel(input_img, x, y, win_size);
            stddev[y * width + x] = stddev_kernel(input_img, stdmean[y * width + x], x, y, win_size);
        }
    }
}



void disparity(ThreadPool &pool, Image &disp,
                float *left_img, float *right_img,
                float *left_stdmean, float *right_stdmean,
                float *left_stddev, float *right_stddev,
                int win_size, int min_disparity, int max_disparity)
{
    //Adds disparity jobs to threadpool

    for (int y = 0; y < disp.height; y += lines_per_task) {
        pool.addWork([=, &disp] {disparityScanlines(disp, y, left_img, right_img,
                                                    left_stdmean, right_stdmean,
                                                    left_stddev, right_stddev,
                                                    win_size,min_disparity,max_disparity);});
    }
}





DisparityResult MultiThreadedDisparityEstimator::estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity)
{
    left.copyDeviceToHost();
    right.copyDeviceToHost();

    auto &prof = Utils::Profiler::getInstance();

    //Make sure that window size is odd
    win_size = win_size | 0x1;

    DisparityResult result;

    result.leftToRight = std::make_shared<Image>();
    result.rightToLeft = std::make_shared<Image>();

    result.leftToRight->allocate(left.width, right.height, 1);
    result.rightToLeft->allocate(left.width, right.height, 1);

    int width = left.width;
    int height = left.height;

    std::vector<float> left_img(width*height + 8); //Add small padding to prevent AVX2 kernel reading out of the bounds
    std::vector<float> right_img(width*height + 8);

    std::vector<float> left_stdmean(width*height + 8);
    std::vector<float> right_stdmean(width*height + 8);

    std::vector<float> left_stddev(width*height + 8);
    std::vector<float> right_stddev(width*height + 8);

    {
        auto pg = prof.section("disparity_precompute");
        for (int y = 0; y < height; y += lines_per_task) {
            //Job is lambda wrapped with std::function
            //Lambda captures arguments and calls function which does actual processing job
            pool.addWork( [=, &left,  &left_img,  &left_stdmean,  &left_stddev]  {
                disparityPreprocessScanlines(y, win_size, left,  left_img,  left_stdmean,  left_stddev);
            });
            pool.addWork( [=, &right, &right_img, &right_stdmean, &right_stddev] {
                disparityPreprocessScanlines(y, win_size, right, right_img, right_stdmean, right_stddev);
            });
        }

        //Wait that preprocessing is finished before starting disparity processing
        pool.wait();
    }

    {
        auto pg = prof.section("disparity_processing");
        disparity(pool, *result.leftToRight,
                left_img.data(), right_img.data(),
                left_stdmean.data(), right_stdmean.data(),
                left_stddev.data(), right_stddev.data(),
                win_size, min_disparity, max_disparity);

        disparity(pool, *result.rightToLeft,
                right_img.data(), left_img.data(),
                right_stdmean.data(), left_stdmean.data(),
                right_stddev.data(), left_stddev.data(),
                win_size, -max_disparity, -min_disparity);

        //Wait that disparity jobs have been finished
        pool.wait();
    }

    return result;
}

}
