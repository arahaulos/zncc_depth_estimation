



inline float window_stdmean(__global const uchar *pixels, const int w, const int h, int x, int y, const int win_size)
{
    int ws = (win_size - 1) / 2;

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
    int ws = (win_size - 1) / 2;
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
    int ws = (win_size - 1) / 2;

    float lmean = left_stdmean[y * width + x];
    float rmean = right_stdmean[y * width + max(x - x_offset, 0)];

    float ldev = left_stddev[y * width + x];
    float rdev = right_stddev[y * width + max(x - x_offset, 0)];

    float cc = 0;
    for (int ky = -ws; ky <= ws; ky++) {
        int ny = min(max(y + ky, 0), height-1);

        for (int kx = -ws; kx <= ws; kx++) {
            int nx0 = min(max(x + kx,            0), width-1);
            int nx1 = min(max(x + kx - x_offset, 0), width-1);

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

    int best_disp = 0;
    float best_zncc = -10000.0f;
    for (int d = min_disparity; d <= max_disparity; d++) {

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

    disp[y * width + x] = abs(best_disp);
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




