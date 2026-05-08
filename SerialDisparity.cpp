#include <iostream>
#include <algorithm>

#include "SerialDisparity.hpp"
#include "Utils.hpp"
#include "kernels_cpu.hpp"


namespace Disparity
{


void disparity(Image &disp,
               float *left_img, float *right_img,
               float *left_stdmean, float *right_stdmean,
               float *left_stddev, float *right_stddev,
               int width, int height, int win_size,
               int min_disparity, int max_disparity)
{
    disp.allocate(width, height, 1);

    int ws = (win_size-1)/2;

    for (int y = 0; y < height; y++) {
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



DisparityResult SerialDisparityEstimator::estimate(Image &left, Image &right, int win_size, int min_disparity, int max_disparity)
{
    left.copyDeviceToHost();
    right.copyDeviceToHost();

    //Make sure that window size is odd
    win_size = win_size | 0x1;

    int width = left.width;
    int height = left.height;

    std::vector<float> left_img(width*height + 8); //Add small padding to prevent AVX2 kernel reading out of the bounds
    std::vector<float> right_img(width*height + 8);

    std::vector<float> left_stdmean(width*height + 8);
    std::vector<float> right_stdmean(width*height + 8);

    std::vector<float> left_stddev(width*height + 8);
    std::vector<float> right_stddev(width*height + 8);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            left_img[y * width + x] = static_cast<float>(left.pixels[y * width + x]);
            right_img[y * width + x] = static_cast<float>(right.pixels[y * width + x]);

            left_stdmean[y * width + x] = stdmean_kernel(left, x, y, win_size);
            right_stdmean[y * width + x] = stdmean_kernel(right, x, y, win_size);

            left_stddev[y * width + x] = stddev_kernel(left, left_stdmean[y * width + x], x, y, win_size);
            right_stddev[y * width + x] = stddev_kernel(right, right_stdmean[y * width + x], x, y, win_size);
        }
    }


    DisparityResult result;

    result.leftToRight = std::make_shared<Image>();
    result.rightToLeft = std::make_shared<Image>();

    disparity(*result.leftToRight,
              left_img.data(), right_img.data(),
              left_stdmean.data(), right_stdmean.data(),
              left_stddev.data(), right_stddev.data(),
              width, height, win_size, min_disparity, max_disparity);

    disparity(*result.rightToLeft,
              right_img.data(), left_img.data(),
              right_stdmean.data(), left_stdmean.data(),
              right_stddev.data(), left_stddev.data(),
              width, height, win_size, -max_disparity, -min_disparity);

    return result;
}


};
