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

    virtual ~Image() {};

    virtual void load(std::string filename);
    virtual void save(std::string filename);
    virtual void allocate(int w, int h, int bpp = 1);

    virtual void downsample();
    virtual void downsample(int times);

    virtual void resize(int new_width, int new_height);
    virtual void convertToGrayscale(const std::array<float, 3> &coeff);


    int width;
    int height;
    std::vector<uint8_t> pixels;

    int bytes_per_pixel;
};
