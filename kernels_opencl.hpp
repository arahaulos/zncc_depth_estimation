

inline float window_stdmean(__global const uchar *pixels, const int w, const int h, int x, int y, const int win_size)
{
    int ws = win_size >> 1;

    int mean_sum = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        for (int kx = -ws; kx <= ws; kx++) {

            int nx = max(min(x + kx, w-1), 0);
            int ny = max(min(y + ky, h-1), 0);

            mean_sum += pixels[ny * w + nx];
        }
    }
    return ((float)mean_sum)/(win_size*win_size);
}

inline float window_stddev(__global const uchar *pixels, const int w, const int h, float mean, int x, int y, const int win_size)
{
    int ws = win_size >> 1;

    float sum = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        for (int kx = -ws; kx <= ws; kx++) {

            int nx = max(min(x + kx, w-1), 0);
            int ny = max(min(y + ky, h-1), 0);

            float p = (float)pixels[ny * w + nx];

            sum += (p - mean)*(p - mean);
        }
    }

    return sqrt(sum);
}




inline float ZNCC(__global const float *left_img,     __global const float *right_img,
                  __global const float *left_stdmean, __global const float *right_stdmean,
                  __global const float *left_stddev,  __global const float *right_stddev,
                  const int width, const int height,
                  int x, int y, int x_offset, const int win_size)
{
    int ws = win_size >> 1;

    float lmean = left_stdmean[y * width + x];
    float rmean = right_stdmean[y * width + max(x - x_offset, 0)];

    float ldev = left_stddev[y * width + x];
    float rdev = right_stddev[y * width + max(x - x_offset, 0)];

    float cc = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        int ny = min(max(y + ky, 0), height-1);

        for (int kx = -ws; kx <= ws; kx++) {
            //int nx0 = min(max(x + kx,            0), width-1);
            //int nx1 = min(max(x + kx - x_offset, 0), width-1);

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



__kernel void disparity(__global uchar *disp,
                        __global const float *left_img,     __global const float *right_img,
                        __global const float *left_stdmean, __global const float *right_stdmean,
                        __global const float *left_stddev,  __global const float *right_stddev,
                        const int width, const int height, const int win_size,
                        const int min_disparity, const int max_disparity)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int ws = win_size >> 1;

    int min_disp = max(min_disparity, x - width + ws + 1);
    int max_disp = min(max_disparity, x         - ws);

    if (x + ws >= width ||
        x - ws < 0 ||
        min_disp > max_disp) {
        disp[y * width + x] = 0;
        return;
    }

    int best_disp = 0;
    float best_zncc = -10000.0f;
    for (int d = min_disparity; d <= max_disp; d++) {

        float z = ZNCC(left_img, right_img,
                       left_stdmean, right_stdmean,
                       left_stddev, right_stddev,
                       width, height,
                       x, y, d, win_size);

        if (z > best_zncc) {
            best_zncc = z;
            best_disp = d;
        }
    }

    disp[y * width + x] = abs(best_disp);    uchar pixel[4];
}



__kernel void preprocess(__global const uchar *input_img,
                         __global float *output_img,
                         __global float *output_stdmean,
                         __global float *output_stddev,
                         const int width, const int height, const int win_size)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    float pixel = (float)input_img[y*width + x];
    float mean = window_stdmean(input_img, width, height, x, y, win_size);
    float dev = window_stddev(input_img, width, height, mean, x, y, win_size);


    output_img[y*width + x] = pixel;
    output_stdmean[y*width + x] = mean;
    output_stddev[y*width + x] = dev;


}


__kernel void image_filter2d(__global const uchar *input_img, __global uchar *output_img, const int width, const int height, const int bytes_per_pixel, __global const uchar *filter, int filter_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int ws = filter_size >> 1;

    int div = 0;
    int sum[4] = {0, 0, 0, 0};

    for (int ky = 0; ky < filter_size; ky++) {
        int iy = y + ky - ws;
        for (int kx = 0; kx < filter_size; kx++) {
            int ix = x + kx - ws;

            if (iy < 0 || iy >= height ||
                ix < 0 || ix >= width) {
                continue;
            }

            int k = filter[ky * filter_size + kx];
            for (int i = 0; i < bytes_per_pixel; i++) {
                sum[i] += k*input_img[(iy * width + ix)*bytes_per_pixel + i];
            }

            div += k;
        }
    }


    for (int i = 0; i < bytes_per_pixel; i++) {
        output_img[(y * width + x)*bytes_per_pixel + i] = (uchar)max(min(sum[i] / div, 255), 0);
    }
}



__kernel void image_resize(__global const uchar *input_img, __global uchar *output_img, const int width, const int height, const int out_width, const int out_height, const int bytes_per_pixel)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= out_width || y >= out_height)
        return;


    float x_coord = ((float)x / (out_width-1)) * (width-1);
    float y_coord = ((float)y / (out_height-1)) * (height-1);

    float fx = x_coord - floor(x_coord);
    float fy = y_coord - floor(y_coord);

    int sx = (int)x_coord;
    int sy = (int)y_coord;

    float out[4];
    for (int i = 0; i < bytes_per_pixel; i++) {
        float c0 = (float)input_img[(sy                  * width + sx                )*bytes_per_pixel + i];
        float c1 = (float)input_img[(sy                  * width + min(sx+1, width-1))*bytes_per_pixel + i];
        float c2 = (float)input_img[(min(sy+1, height-1) * width + sx                )*bytes_per_pixel + i];
        float c3 = (float)input_img[(min(sy+1, height-1) * width + min(sx+1, width-1))*bytes_per_pixel + i];


        float vc0 = fx * c1 + (1.0f - fx)*c0;
        float vc1 = fx * c3 + (1.0f - fx)*c2;

        out[i] = vc1 * fy + (1.0f - fy)*vc0;
    }

    for (int i = 0; i < bytes_per_pixel; i++) {
        output_img[(y * out_width + x)*bytes_per_pixel + i] = (uchar)max(min(out[i], 255.0f), 0.0f);
    }

}


__kernel void image_to_grayscale(__global const uchar *input_img, __global uchar *output_img, const int width, const int height, const int bytes_per_pixel, const float r_coef, const float g_coef, const float b_coef)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;


    float out = 0.0f;

    out += (float)input_img[(y*width + x)*bytes_per_pixel + 0] * r_coef;
    out += (float)input_img[(y*width + x)*bytes_per_pixel + 1] * g_coef;
    out += (float)input_img[(y*width + x)*bytes_per_pixel + 2] * b_coef;

    output_img[y*width + x] = (uchar)max(min(out, 255.0f), 0.0f);
}


