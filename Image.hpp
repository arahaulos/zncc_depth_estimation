#pragma once

#include <stdint.h>
#include <vector>
#include <array>
#include <string>

struct Image
{
    Image() {

    }
    Image(int w, int h) {
        allocate(w, h);
    }

    Image(std::string filename) {
        load(filename);
    }

    void load(std::string filename);
    void save(std::string filename);
    void allocate(int w, int h, int bpp = 1);

    Image downsample();
    Image downsample(int times);

    Image resized(int new_width, int new_height);
    Image toGrayscale(const std::array<float, 3> &coeff);

    int width;
    int height;
    std::vector<uint8_t> pixels;

    int bytes_per_pixel;
};
