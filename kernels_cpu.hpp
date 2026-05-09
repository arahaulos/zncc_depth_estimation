#pragma once

#include <algorithm>
#include <math.h>
#include "Image.hpp"

#define USE_AVX2

#ifdef USE_AVX2

#include <x86gprintrin.h>
#include <x86intrin.h>

#endif


inline float stdmean_kernel(Image &img, int x, int y, int win_size)
{
    //This function calculates std mean inside 2d window/kernel
    int ws = win_size >> 1;
    int mean_sum = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        for (int kx = -ws; kx <= ws; kx++) {
            int nx = std::clamp(x + kx, 0, img.width-1);
            int ny = std::clamp(y + ky, 0, img.height-1);

            mean_sum += static_cast<int>(img.pixels[ny * img.width + nx]);
        }
    }
    return (static_cast<float>(mean_sum))/(win_size*win_size);
}

inline float stddev_kernel(Image &img, float mean, int x, int y, int win_size)
{
    //This function calculates std deviation inside 2d window/kernel
    int ws = win_size >> 1;
    float sum = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        for (int kx = -ws; kx <= ws; kx++) {
            int nx = std::clamp(x + kx, 0, img.width-1);
            int ny = std::clamp(y + ky, 0, img.height-1);

            float p = static_cast<float>(img.pixels[ny * img.width + nx]);

            sum += (p - mean)*(p - mean);
        }
    }

    return sqrt(sum);
}



inline float ZNCC(const float *left_img, const float *right_img,
                  const float *left_stdmean, const float *right_stdmean,
                  const float *left_stddev, const float *right_stddev,
                  int width, int height,
                  int x, int y, int x_offset, int win_size)
{
    int ws = win_size >> 1;

    float lmean = left_stdmean[y * width + x];
    float rmean = right_stdmean[y * width + x - x_offset];

    float ldev = left_stddev[y * width + x];
    float rdev = right_stddev[y * width + x - x_offset];

    float cc = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        int ny = std::clamp(y + ky, 0, height-1);

        for (int kx = -ws; kx <= ws; kx++) {
            int nx0 = x + kx;
            int nx1 = x + kx - x_offset;

            cc += (left_img[ny * width + nx0] - lmean) * (right_img[ny * width + nx1] - rmean);
        }
    }
    float denom = ldev * rdev;
    if (denom < 0.0001f) {
        return 0.0f;
    } else {
        return cc / denom;
    }
}

#ifdef USE_AVX2

inline float ZNCC_avx2(const float* __restrict__ left_img, const float* __restrict__ right_img,
                       const float* __restrict__ left_stdmean, const float* __restrict__ right_stdmean,
                       const float* __restrict__ left_stddev, const float* __restrict__ right_stddev,
                       int width, int height,
                       int x, int y, int x_offset, int win_size)
{
    //Very low effort attempt to vectorize ZNCC kernel
    //Autovectrization probably makes better job with right compiler flags (-funsafe-math etc.)
    int ws = win_size >> 1;

    //Get std mean and deviations from precomputed images

    float lmean = left_stdmean[y * width + x];
    float rmean = right_stdmean[y * width + x - x_offset];

    float ldev = left_stddev[y * width + x];
    float rdev = right_stddev[y * width + x - x_offset];

    __m256 lm = _mm256_set1_ps(lmean);
    __m256 rm = _mm256_set1_ps(rmean);

    __m256 sums[2] {_mm256_set1_ps(0), _mm256_set1_ps(0)};

    int ky = -ws;
    for (; ky + 2 <= ws; ky += 3) {
        int idx_y0 = std::clamp(y + ky + 0, 0, height-1)*width;
        int idx_y1 = std::clamp(y + ky + 1, 0, height-1)*width;
        int idx_y2 = std::clamp(y + ky + 2, 0, height-1)*width;

        int kx = -ws;
        for (; kx <= ws; kx += 8) {
            int idx_x0 = x + kx;
            int idx_x1 = x + kx - x_offset;


            //If last element of the vector goes out of window
            //Set tail flag true. Tail is summed to another register
            //At end, that register is masked so that only valid elements gets added to final sum
            bool tail = (kx + 7 > ws);

            __m256 left0 = _mm256_loadu_ps(&left_img[idx_y0 + idx_x0]);
            __m256 left1 = _mm256_loadu_ps(&left_img[idx_y1 + idx_x0]);
            __m256 left2 = _mm256_loadu_ps(&left_img[idx_y2 + idx_x0]);

            __m256 right0 = _mm256_loadu_ps(&right_img[idx_y0 + idx_x1]);
            __m256 right1 = _mm256_loadu_ps(&right_img[idx_y1 + idx_x1]);
            __m256 right2 = _mm256_loadu_ps(&right_img[idx_y2 + idx_x1]);

            sums[tail] = _mm256_fmadd_ps(_mm256_sub_ps(left0, lm), _mm256_sub_ps(right0, rm), sums[tail]);
            sums[tail] = _mm256_fmadd_ps(_mm256_sub_ps(left1, lm), _mm256_sub_ps(right1, rm), sums[tail]);
            sums[tail] = _mm256_fmadd_ps(_mm256_sub_ps(left2, lm), _mm256_sub_ps(right2, rm), sums[tail]);
        }
    }
    for (; ky <= ws && ky <= ws; ky++) {
        int idx_y0 = std::clamp(y + ky + 0, 0, height-1)*width;
        int kx = -ws;
        for (; kx <= ws; kx += 8) {
            bool tail = (kx + 8 > ws);

            __m256 left0 = _mm256_loadu_ps(&left_img[idx_y0 + x + kx]);
            __m256 right0 = _mm256_loadu_ps(&right_img[idx_y0 + x + kx - x_offset]);

            sums[tail] = _mm256_fmadd_ps(_mm256_sub_ps(left0, lm), _mm256_sub_ps(right0, rm), sums[tail]);
        }
    }

    static const __m256i tail_masks[8] = {
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0,  0),
        _mm256_set_epi32(0,  0,  0,  0,  0,  0,  0, ~0),
        _mm256_set_epi32(0,  0,  0,  0,  0,  0, ~0, ~0),
        _mm256_set_epi32(0,  0,  0,  0,  0, ~0, ~0, ~0),
        _mm256_set_epi32(0,  0,  0,  0, ~0, ~0, ~0, ~0),
        _mm256_set_epi32(0,  0,  0, ~0, ~0, ~0, ~0, ~0),
        _mm256_set_epi32(0,  0, ~0, ~0, ~0, ~0, ~0, ~0),
        _mm256_set_epi32(0, ~0, ~0, ~0, ~0, ~0, ~0, ~0)
    };

    __m256i masked_tail = _mm256_and_si256(_mm256_castps_si256(sums[1]), tail_masks[win_size & 0x7]); //Masks trash from tail
    __m256 sum = _mm256_add_ps(sums[0], _mm256_castsi256_ps(masked_tail)); //Add tail to good vector


    //Lets sum values horizontally to get single sum
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 s  = _mm_add_ps(lo, hi);

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);

    float final_sum = _mm_cvtss_f32(s);

    float denom = ldev * rdev;
    if (denom < 0.0001f) {
        return 0.0f;
    } else {
        return final_sum / denom;
    }
}

#endif // USE_AVX2
