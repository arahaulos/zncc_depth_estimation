#include "Image.hpp"

#include <iostream>
#include <algorithm>

#include "lodepng/lodepng.h"


void Image::load(std::string filename)
{
    unsigned int w, h;

    lodepng::decode(pixels, w, h, filename);

    width = w;
    height = h;

    bytes_per_pixel = pixels.size() / (width*height);

    std::cout << "Loaded image " << filename << std::endl;
    std::cout << "Size: " << width << "x" << height << std::endl;
    std::cout << "Bytes per pixel: " << bytes_per_pixel << std::endl;
}

void Image::save(std::string filename)
{
    LodePNGColorType color_type;
    switch (bytes_per_pixel) {
        case 4:
            color_type = LCT_RGBA;
            break;
        case 3:
            color_type = LCT_RGB;
            break;
        case 1:
            color_type = LCT_GREY;
            break;
        default:
            std::cout << "Bad bytes per pixels: " << bytes_per_pixel << std::endl;
            return;
    }
    lodepng::encode(filename, pixels, width, height, color_type);
}

Image Image::resized(int new_width, int new_height)
{
    Image new_image;
    new_image.width = new_width;
    new_image.height = new_height;
    new_image.bytes_per_pixel = bytes_per_pixel;

    new_image.pixels.reserve(new_width*new_height*bytes_per_pixel);

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = x*width / new_width;
            int src_y = y*height / new_height;

            for (int c = 0; c < bytes_per_pixel; c++) {
                new_image.pixels.push_back(pixels[(src_y*width + src_x)*bytes_per_pixel + c]);
            }
        }
    }

    return new_image;
}

Image Image::downsample()
{
    Image new_img;
    new_img.allocate(width / 2, height / 2, bytes_per_pixel);

    static const std::array<int, 5*5> kernel = {
    1,  4,  6,  4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1,  4,  6,  4, 1,
    };

    for (int y = 0; y < height/2; y++) {
        for (int x = 0; x < width/2; x++) {

            std::array<int, 4> sample = {0, 0, 0, 0};

            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int k = kernel[ky*5+kx];

                    int ny = std::clamp((y*2 + ky - 2), 0, height-1);
                    int nx = std::clamp((x*2 + kx - 2), 0, width-1);

                    for (int c = 0; c < bytes_per_pixel; c++) {
                        int cc = pixels[(ny * width + nx)*bytes_per_pixel + c];

                        sample[c] += cc*k;
                    }
                }
            }

            for (int c = 0; c < bytes_per_pixel; c++) {
                new_img.pixels[(y * new_img.width + x)*bytes_per_pixel + c] = sample[c]/256;
            }
        }
    }
    return new_img;
}

Image Image::downsample(int times)
{
    Image new_img = *this;
    for (int i = 0; i < times; i++) {
        Image n = new_img.downsample();
        new_img = n;
    }
    return new_img;
}

Image Image::toGrayscale(const std::array<float, 3> &coeff)
{
    Image new_image;
    new_image.width = width;
    new_image.height = height;
    new_image.bytes_per_pixel = 1;

    if (bytes_per_pixel == 1) {
        new_image.pixels = pixels;
        return new_image;
    }

    new_image.pixels.reserve(width*height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float value = 0;
            for (int c = 0; c < 3; c++) {
                value += static_cast<float>(pixels[(y*width + x)*bytes_per_pixel + c]) * coeff[c];
            }
            value = std::clamp(value, 0.0f, 255.0f);

            new_image.pixels.push_back(static_cast<uint8_t>(value));
        }
    }
    return new_image;
}

void Image::allocate(int w, int h, int bpp)
{
    width = w;
    height = h;
    bytes_per_pixel = bpp;
    pixels.resize(w*h*bpp);
}

