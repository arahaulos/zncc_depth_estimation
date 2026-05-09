#include <iostream>
#include <algorithm>
#include <math.h>
#include <queue>

#include "PostProcessing.hpp"
#include "Utils.hpp"


namespace Disparity
{

uint8_t findClosestNonZeroScanline(Image &img, int x, int y)
{
    //Finds closest non zero point from same scanline

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
    //Uses simple breadth first search to find closest on zero point

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




std::unique_ptr<Image> PostProcessor::crossCheck(DisparityResult disparity, int min_disparity, int max_disparity, int max_disp_diff)
{
    auto pg = Utils::Profiler::getInstance().section("postprocessing_crosscheck");

    disparity.leftToRight->copyDeviceToHost(); //Make sure that pixel data is at host memory (if normal Image is used, doesn't do anything)
    disparity.rightToLeft->copyDeviceToHost();

    int width = disparity.leftToRight->width;
    int height = disparity.rightToLeft->height;

    auto result = std::make_unique<Image>();

    result->allocate(width, height); //Allocate result

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //Read disparity value from leftToRight image
            //And find corresponding value from rightToLeft image
            int leftdp = disparity.leftToRight->pixels[y*width + x];
            int rightdp = disparity.rightToLeft->pixels[y*width + std::clamp(x - leftdp, 0, width-1)];

            if (std::abs(leftdp - rightdp) > max_disp_diff || leftdp == 0 || rightdp == 0) {
                result->pixels[y*width + x] = 0; //There is too big difference
            } else {
                //Difference is ok.
                //Disparity value gets remapped from absolute disparity values to 0-255
                result->pixels[y*width + x] = (leftdp - min_disparity) * 255 / (max_disparity - min_disparity);
            }
        }
    }

    return result;
}

std::unique_ptr<Image> PostProcessor::erosion(Image &in)
{
    auto pg = Utils::Profiler::getInstance().section("postprocessing_erosion");

    in.copyDeviceToHost(); //Make sure that pixel data is at host memory (if normal Image is used, doesn't do anything)

    auto out = std::make_unique<Image>(in.width, in.height); //Create resulting image


    auto has_zero_neighbors = [&] (int x, int y) -> bool {
        //returns true when window has zero pixels
        //return false when window doesn't have zero pixels'
        static const int neightbor_coords[] =
        {
            -1, 0,
             1, 0,
            -1, 1,
             0, 1,
             1, 1,
            -1, -1,
             0, -1,
             1, -1
        };

        for (int i = 0; i < sizeof(neightbor_coords)/sizeof(int); i+= 2) {
            int nx = x + neightbor_coords[i+0];
            int ny = y + neightbor_coords[i+1];

            if (nx < 0 || nx >= in.width ||
                ny < 0 || ny >= in.height) {
                continue;
            }
            if (in.pixels[ny*in.width + nx] == 0) {
                return true;
            }
        }
        return false;
    };

    //Iterates all pixels and if they have zero neighbors, set them 0
    for (int y = 0; y < in.height; y++) {
        for (int x = 0; x < in.width; x++) {
            if (has_zero_neighbors(x, y)) {
                out->pixels[y*in.width+x] = 0;
            } else {
                out->pixels[y*in.width+x] = in.pixels[y*in.width+x];
            }
        }
    }
    return out;
}


std::unique_ptr<Image> PostProcessor::fill(Image &in)
{
    auto pg = Utils::Profiler::getInstance().section("postprocessing_fill");

    in.copyDeviceToHost(); //Make sure that pixel data is at host memory (if normal Image is used, doesn't do anything)

    auto result = std::make_unique<Image>(in); //Makes copy from input


    //This algorithms is more optimized version findClosestNonZeroScanline
    //It walks each scanlines, and when zero region is found,
    //it splits line half and fills first half with last non-zero value and last half with next non-zero color
    for (int y = 0; y < in.height; y++) {
        auto output_row = result->pixels.begin() + (y*result->width);

        uint8_t start_color = 0;
        int start_x = 0;

        for (int x = 0; x < in.width; x++) {
            uint8_t c = in.pixels[y*in.width + x];

            if (c == 0) {
                if (start_x < 0) {
                    start_x = x;
                }
                continue;
            }

            if (start_x >= 0) {
                uint8_t end_color = c;
                int end_x = x;
                int mid_x = (end_x + start_x) / 2;

                if (start_color == 0) {
                    std::fill(output_row + start_x, output_row + end_x, end_color);
                } else {
                    std::fill(output_row + start_x, output_row + mid_x, start_color);
                    std::fill(output_row + mid_x, output_row + end_x, end_color);
                }
            }
            start_x = -1;
            start_color = c;
        }

        if (start_x >= 0) {
            std::fill(output_row + start_x, output_row + result->width, start_color);
        }
    }

    return result;
}


}


