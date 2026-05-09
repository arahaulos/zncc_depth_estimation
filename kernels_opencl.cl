#define TILE_SIZE 16
#define BATCH_SIZE 16


__kernel void disparity2(__global uchar *best_disp,
                         __global const float *left_img,     __global const float *right_img,
                         __global const float *left_stdmean, __global const float *right_stdmean,
                         __global const float *left_stddev,  __global const float *right_stddev,
                         const int width, const int height, const int win_size,
                         __local float *left_halo, __local float *right_halo,  const int disp_batch_start, const int disp_batch_end, __global float *best_zncc)
{
    //This is kernel for doing ZNCC disparity
    //Uses tiling, local memory and disparity batching

    int global_x = get_group_id(0) * TILE_SIZE;
    int global_y = get_group_id(1) * TILE_SIZE;

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int x = global_x + local_x;
    int y = global_y + local_y;

    int ws = win_size >> 1;

    //Halo is surrounding area of tile
    //Calculate halo sizes

    //Left halo width is TILE_SIZE + 2*(window_size / 2)
    int halo_size = TILE_SIZE + 2*ws;

    //Right halo is bigger, because there is multiple x_offsets to calculate ZNCC
    int right_halo_width = TILE_SIZE + BATCH_SIZE + 2*ws;

    //Load cooperatively left halo to local memory
    int lid = local_y * TILE_SIZE + local_x;

    for (int i = lid; i < halo_size*halo_size; i += TILE_SIZE*TILE_SIZE) {
        int hx = i % halo_size;
        int hy = i / halo_size;

        int sx = max(min(global_x + hx - ws, width -1), 0);
        int sy = max(min(global_y + hy - ws, height-1), 0);

        left_halo[hy * halo_size + hx] = left_img[sy * width + sx];
    }

    //Load cooperatively right halo to local memory
    for (int i = lid; i < halo_size*right_halo_width; i += TILE_SIZE*TILE_SIZE) {

        int hx = i % right_halo_width;
        int hy = i / right_halo_width;

        int sx = max(min(global_x + hx - disp_batch_end - ws + 1, width -1), 0);
        int sy = max(min(global_y + hy                  - ws    , height-1), 0);

        right_halo[hy * right_halo_width + hx] = right_img[sy * width + sx];
    }

    //Syncronize to make sure each worker/thread has finished loading
    barrier(CLK_LOCAL_MEM_FENCE);

    //Because image size is not necessarily divisible by tile size, so some workers are outside of image boundaries
    if (x >= width || y >= height)
        return;

    //Load mean and deviations for left window
    float lmean = left_stdmean[y * width + x];
    float ldev  = left_stddev[y * width + x];

    float batch_best_zncc = -10000;
    int batch_best_disp = 0;
    for (int d = disp_batch_start; d < disp_batch_end; d++) {
        //Skip when either window is outside of image boundariers
        //best_zncc buffer is filled with smaller value than kernel batch_best_zncc
        //This ensures that in vertical edges where is no valid windows, gets filled with 0
        if (x - d - ws < 0 ||
            x     - ws < 0 ||
            x - d + ws >= width ||
            x     + ws >= width) {

            continue;
        }

        //Load precomputer mean and devitations values for offset (right) window
        float rmean = right_stdmean[y * width + x - d];
        float rdev  = right_stddev [y * width + x - d];

        //Calculate x offset relative to halo
        int tile_x_offset = disp_batch_end - d - 1;


        //Calculate zncc
        float cc = 0.0f;
        for (int ky = 0; ky <= ws*2; ky++) {
            int hy = local_y + ky;
            for (int kx = 0; kx <= ws*2; kx++) {
                int hx  = local_x + kx;
                int hx2 = local_x + kx + tile_x_offset;

                cc += (left_halo[hy * halo_size + hx] - lmean) * (right_halo[hy * right_halo_width + hx2] - rmean);
            }
        }

        float denom = ldev * rdev;
        if (denom > 0.0001f) {
            float zncc = cc / denom;

            if (batch_best_zncc < zncc) {
                batch_best_zncc = zncc;
                batch_best_disp = d;
            }
        }
    }

    //Check if best zncc value in this batch is greater than previous value
    if (best_zncc[y * width + x] < batch_best_zncc) {
        best_zncc[y * width + x] = batch_best_zncc;
        best_disp[y * width + x] = abs(batch_best_disp);
    }
}



__kernel void preprocess2(__global const uchar *input_img,
                          __global float *output_img,
                          __global float *output_stdmean,
                          __global float *output_stddev,
                          const int width, const int height, const int win_size, __local uchar *halo)
{
    //This kernel precalculates window means and deviations for ZNCC kernel
    //Uses tiling and local memory

    int global_x = get_group_id(0) * TILE_SIZE;
    int global_y = get_group_id(1) * TILE_SIZE;

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int x = global_x + local_x;
    int y = global_y + local_y;

    int ws = win_size >> 1;

    int halo_size = TILE_SIZE + 2*ws;
    int lid = local_y * TILE_SIZE + local_x;
    for (int i = lid; i < halo_size*halo_size; i += TILE_SIZE*TILE_SIZE) {
        int hx = i % halo_size;
        int hy = i / halo_size;

        int sx = max(min(global_x + hx - ws, width -1), 0);
        int sy = max(min(global_y + hy - ws, height-1), 0);

        halo[hy * halo_size + hx] = input_img[sy * width + sx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= width || y >= height)
        return;


    float pixel = (float)input_img[y * width + x];
    float mean = 0.0f;
    float deviation = 0.0f;

    int sum = 0;
    for (int ky = 0; ky <= ws*2; ky++) {
        int hy = local_y + ky;
        for (int kx = 0; kx <= ws*2; kx++) {
            int hx  = local_x + kx;

            sum += halo[hy * halo_size + hx];
        }
    }

    mean = (float)sum / (win_size*win_size);

    for (int ky = 0; ky <= ws*2; ky++) {
        int hy = local_y + ky;
        for (int kx = 0; kx <= ws*2; kx++) {
            int hx  = local_x + kx;

            float value = halo[hy * halo_size + hx];

            deviation += (value - mean) * (value - mean);
        }
    }

    output_img    [y * width + x] = pixel;
    output_stdmean[y * width + x] = mean;
    output_stddev [y * width + x] = sqrt(deviation);
}





inline float window_stdmean(__global const uchar *pixels, const int w, const int h, int x, int y, const int win_size)
{
    //This is unoptimized mean kernel for initial OpenCL version

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
    //This is unoptimized deviation kernel for initial OpenCL version

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

__kernel void preprocess(__global const uchar *input_img,
                         __global float *output_img,
                         __global float *output_stdmean,
                         __global float *output_stddev,
                         const int width, const int height, const int win_size)
{
    //This is unoptimized OpenCL implementation
    //This calculates window mean and deviations for disparity kernel

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



inline float ZNCC(__global const float *left_img,     __global const float *right_img,
                  __global const float *left_stdmean, __global const float *right_stdmean,
                  __global const float *left_stddev,  __global const float *right_stddev,
                  const int width, const int height,
                  int x, int y, int x_offset, const int win_size)
{
    //This is unoptimized OpenCL implementation
    //This calculates ZNCC value

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
    //This is unoptimized OpenCL implementation
    //Calculates disparity value using ZNCC

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

    disp[y * width + x] = abs(best_disp);
}


__kernel void image_filter2d(__global const uchar *input_img, __global uchar *output_img, const int width, const int height, const int bytes_per_pixel, __global const uchar *filter, int filter_size)
{
    //Unoptimized 2d filtering kernel
    //OpenCLImage class uses this

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
    //Unoptimized scaling kernel
    //Uses bilinear interpolation
    //OpenCLImage class uses this

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
    //Unoptimized kernel to convert image to grayscale
    //OpenCLImage class uses this

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





__kernel void crosscheck(__global const uchar *left_img, __global const uchar *right_img, __global uchar *output, const int width, const int height, const int min_disparity, const int max_disparity, const int max_disp_diff)
{
    //Unoptimzed crosscheck kernel
    //OpenCLPostProcessing class uses this

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    int leftdp = left_img[y*width + x];
    int rightdp = right_img[y*width + min(max(x - leftdp, 0), width-1)];

    if (abs(leftdp - rightdp) > max_disp_diff || leftdp == 0 || rightdp == 0) {
        output[y*width + x] = 0;
    } else {
        output[y*width + x] = (leftdp - min_disparity) * 255 / (max_disparity - min_disparity);
    }
}



__kernel void erosion(__global const uchar *input, __global uchar *output, const int width, const int height)
{
    //Unoptimzed morphological erosion kernel
    //OpenCLPostProcessing class uses this

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    __constant int neightbor_coords[] =
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

    for (int i = 0; i < sizeof(neightbor_coords)/sizeof(int); i += 2) {
        int nx = x + neightbor_coords[i+0];
        int ny = y + neightbor_coords[i+1];

        if (nx < 0 || nx >= width ||
            ny < 0 || ny >= height) {
            continue;
        }

        if (input[ny*width + nx] == 0) {
            output[y*width + x] = 0;
            return;
        }
    }
    output[y*width + x] = input[y*width + x];
}


__kernel void fill(__global const uchar *input, __global uchar *output, const int width, const int height)
{
    //Unoptimzed fill kernel
    //Finds closes non-zero point on same scanline
    //This is far from efficient
    //OpenCLPostProcessing class uses this

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;


    if (input[y*width + x] != 0) {
        output[y*width + x] = input[y*width + x];
        return;
    }

    uchar *scanline = &input[y * width];

    int rx = x;
    while (rx < width && scanline[rx] == 0) {
        rx++;
    }

    int lx = x;
    while (lx >= 0 && scanline[lx] == 0) {
        lx--;
    }

    int ldist = abs(lx - x);
    int rdist = abs(rx - x);


    uchar out_color = 0;

    if (rx < width && lx >= 0) {
        if (ldist == rdist) {
            out_color = (scanline[rx] + scanline[lx]) / 2;
        } else if (ldist < rdist) {
            out_color = scanline[lx];
        } else {
            out_color = scanline[rx];
        }
    } else if (lx >= 0) {
        out_color = scanline[lx];
    } else if (rx < width) {
        out_color = scanline[rx];
    }

    output[y*width + x] = out_color;
}
