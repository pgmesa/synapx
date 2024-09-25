#include <stdio.h>

// Compile library with: gcc -fPIC -shared -o conv_tools.so conv_tools.c

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int padH, int padW)
{
    row -= padH;
    col -= padW;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}


void im2col(
    float* data_im,
    int batches, int channels, int height, int width,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    float* data_col
) {
    int c,h,w;
    int height_col = (height + 2*pH - kH) / sH + 1;
    int width_col = (width + 2*pW - kW) / sW + 1;
    int L = height_col * width_col;

    int channels_col = channels * kH * kW;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kW;
        int h_offset = (c / kW) % kH;
        int c_im = c / kH / kW;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * sH;
                int im_col = w_offset + w * sW;
                int col_index = (c * height_col + h) * width_col + w;
                int px = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pH, pW);
                data_col[col_index] = px;
            }
        }
    }
}