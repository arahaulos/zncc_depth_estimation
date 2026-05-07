#include <iostream>
#include <algorithm>
#include <math.h>
#include <queue>

#include "PostProcessing.hpp"
#include "Utils.hpp"


namespace PostProcessing
{

uint8_t findClosestNonZeroScanline(Image &img, int x, int y)
{
    uint8_t *scanline = &img.pixels[y * img.width];

    int rx = x;
    while (rx < img.width && scanline[rx] == 0) {
        rx++;
    }

    int lx = x;
    while (lx >= 0 && scanline[lx] == 0) {
        lx--;
    }

    int ldist = std::abs(lx - x);
    int rdist = std::abs(rx - x);

    if (rx < img.width && lx >= 0) {
        if (ldist == rdist) {
            return (scanline[rx] + scanline[lx]) / 2;
        } else if (ldist < rdist) {
            return scanline[lx];
        } else {
            return scanline[rx];
        }
    } else if (lx >= 0) {
        return scanline[lx];
    } else if (rx < img.width) {
        return scanline[rx];
    } else {
        return 0;
    }
}

uint8_t findClosestNonZero(Image &img, int x, int y)
{
    std::vector<bool> searched(img.width*img.height);
    std::fill(searched.begin(), searched.end(), false);
    searched[y * img.width + x] = true;

    std::queue<std::pair<int,int>> search_queue;

    constexpr int neighbors[8*2] =
    {
         1, -1,
         1,  1,
        -1, -1,
        -1,  1,
         0, -1,
         0,  1,
        -1,  0,
         1,  0
    };

    search_queue.emplace(x,y);
    while (!search_queue.empty()) {
        int x = search_queue.front().first;
        int y = search_queue.front().second;
        search_queue.pop();

        for (int i = 0; i < 8; i++) {
            int nx = x + neighbors[i*2 + 0];
            int ny = y + neighbors[i*2 + 1];

            if (nx >= 0 && nx < img.width &&
                ny >= 0 && ny < img.height) {

                int index = ny * img.width + nx;

                if (img.pixels[index]) {
                    return img.pixels[index];
                }

                if (!searched[index]) {
                    searched[index] = true;

                    search_queue.emplace(nx, ny);
                }
            }
        }
    }

    std::cout << "Didn't found!!!" << std::endl;

    return 0;
}




CrossCheckResult crossCheck(Disparity::DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff)
{
    int width = disparity.leftToRight.width;
    int height = disparity.rightToLeft.height;

    CrossCheckResult result;

    result.output.allocate(width, height, 1);
    result.occluded.reserve(width*height/4);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int leftdp = disparity.leftToRight.pixels[y*width + x];
            int rightdp = std::abs(disparity.rightToLeft.pixels[y*width + std::clamp(x - leftdp, 0, width-1)]);

            if (std::abs(leftdp - rightdp) > max_disp_diff || leftdp == 0 || rightdp == 0) {
                result.output.pixels[y*width + x] = 0;
                result.occluded.emplace_back(x, y);
            } else {
                result.output.pixels[y*width + x] = (leftdp - min_disparity) * 255 / (max_disparity - min_disparity);
            }
        }
    }

    return result;
}


Image fill(CrossCheckResult &cc_result)
{
    Image result;

    result = cc_result.output;

    for (std::pair<int,int> p : cc_result.occluded) {
        result.pixels[p.second*result.width + p.first] = findClosestNonZeroScanline(cc_result.output, p.first, p.second);
    }

    return result;
}


}


