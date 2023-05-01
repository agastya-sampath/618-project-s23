#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <stdio.h>

#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

////////////////////////////////
/* - Enhance (Equalization) - */
////////////////////////////////

__global__ void histogramKernel(const unsigned char *d_img, int *d_hist_r, int *d_hist_g, int *d_hist_b, int width, int height)
{
    // Calculate histogram for each color channel
    // [d_hist_r == hist_r]
    // [d_hist_g == hist_g]
    // [d_hist_b == hist_b]

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int offset = y * width + x;
        atomicAdd(&d_hist_r[d_img[offset]], 1);
        atomicAdd(&d_hist_g[d_img[offset + width * height]], 1);
        atomicAdd(&d_hist_b[d_img[offset + 2 * width * height]], 1);
    }
}

__global__ void equalizationKernel(const unsigned char *d_img, unsigned char *d_res, float *d_cum_hist, int width, int height)
{
    // Apply histogram equalization to each color channel
    // [d_res == res]
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int offset = y * width + x;
        d_res[offset] = (unsigned char)(d_cum_hist[d_img[offset]] * 255.0f);
        d_res[offset + width * height] = (unsigned char)(d_cum_hist[d_img[offset + width * height]] * 255.0f);
        d_res[offset + 2 * width * height] = (unsigned char)(d_cum_hist[d_img[offset + 2 * width * height]] * 255.0f);
    }
}

__global__ void compute_hist_avg(const int *d_hist_r, const int *d_hist_g, const int *d_hist_b,
                                 const int total_pixels, const int bins, float *d_hist_avg)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < bins)
    {
        d_hist_avg[i] = (d_hist_r[i] + d_hist_g[i] + d_hist_b[i]) / (3.0f * total_pixels);
    }
}

void CUDAHistogramEqualization(const unsigned char *img, unsigned char *res, const int width, const int height)
{
    const int bins = 256;

    // Allocate device memory
    unsigned char *d_img, *d_res;
    cudaMalloc((void **)&d_img, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_res, width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(d_img, img, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int *d_hist_r, *d_hist_g, *d_hist_b;
    cudaMalloc((void **)&d_hist_r, bins * sizeof(int));
    cudaMalloc((void **)&d_hist_g, bins * sizeof(int));
    cudaMalloc((void **)&d_hist_b, bins * sizeof(int));
    cudaMemset(d_hist_r, 0, bins * sizeof(int));
    cudaMemset(d_hist_g, 0, bins * sizeof(int));
    cudaMemset(d_hist_b, 0, bins * sizeof(int));

    float *d_hist_avg;
    cudaMalloc(&d_hist_avg, bins * sizeof(float));
    cudaMemset(d_hist_avg, 0, bins * sizeof(float));

    float *d_cum_hist;
    cudaMalloc(&d_cum_hist, bins * sizeof(float));
    cudaMemset(d_cum_hist, 0, bins * sizeof(float));

    // Compute histograms
    dim3 blockDims(16, 16);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y);

    histogramKernel<<<gridDims, blockDims>>>(d_img, d_hist_r, d_hist_g, d_hist_b, width, height);
    cudaDeviceSynchronize();

    // Compute average histogram
    int total_pixels = width * height;
    compute_hist_avg<<<gridDims, blockDims>>>(d_hist_r, d_hist_g, d_hist_b, total_pixels, bins, d_hist_avg);

    // Compute cumulative histogram
    float hist_avg[bins] = {0};
    cudaMemcpy(hist_avg, d_hist_avg, bins * sizeof(float), cudaMemcpyDeviceToHost);

    float cum_hist[bins] = {0};
    cum_hist[0] = hist_avg[0];

    for (int i = 1; i < bins; i++)
    {
        cum_hist[i] = cum_hist[i - 1] + hist_avg[i];
    }

    // Apply equalization
    cudaMemcpy(d_cum_hist, cum_hist, bins * sizeof(float), cudaMemcpyHostToDevice);
    equalizationKernel<<<gridDims, blockDims>>>(d_img, d_res, d_cum_hist, width, height);

    cudaMemcpy(res, d_res, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_img);
    cudaFree(d_hist_r);
    cudaFree(d_hist_g);
    cudaFree(d_hist_b);
    cudaFree(d_hist_avg);
    cudaFree(d_cum_hist);
}

////////////////////
/* - Conversion - */
////////////////////

__global__ void RGBtoYCbCrKernel(const unsigned char *img, unsigned char *res, const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Ref: https://cimg.eu/reference/storage.html
    const unsigned char r = img[y * width + x];
    const unsigned char g = img[(height + y) * width + x];
    const unsigned char b = img[(height * 2 + y) * width + x];

    // Perform RGB to YCbCr conversion
    const double Y = 0.299 * r + 0.587 * g + 0.114 * b;
    const double Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    const double Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;

    // Store the converted values back into the image
    res[y * width + x] = (unsigned char)Y;
    res[(height + y) * width + x] = (unsigned char)Cb;
    res[(height * 2 + y) * width + x] = (unsigned char)Cr;
}

void CUDARGBtoYCbCr(const unsigned char *img, unsigned char *res, int width, int height)
{
    const int w = width;
    const int h = height;
    const int size = w * h * 3 * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMemcpy(d_input, img, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    RGBtoYCbCrKernel<<<grid, block>>>(d_input, d_output, w, h);

    cudaMemcpy(res, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void YCbCrtoRGBKernel(const unsigned char *img, unsigned char *res, const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Ref: https://cimg.eu/reference/storage.html
    const unsigned char Y = img[y * width + x];
    const unsigned char Cb = img[(height + y) * width + x];
    const unsigned char Cr = img[(height * 2 + y) * width + x];

    // Perform YCbCr to RGB conversion
    const double R = Y + 1.402 * (Cr - 128);
    const double G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128);
    const double B = Y + 1.772 * (Cb - 128);

    // Store the converted values back into the image
    res[y * width + x] = (unsigned char)R;
    res[(height + y) * width + x] = (unsigned char)G;
    res[(height * 2 + y) * width + x] = (unsigned char)B;
}

void CUDAYCbCrtoRGB(const unsigned char *img, unsigned char *res, int width, int height)
{
    const int w = width;
    const int h = height;
    const int size = w * h * 3 * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMemcpy(d_input, img, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    YCbCrtoRGBKernel<<<grid, block>>>(d_input, d_output, w, h);

    cudaMemcpy(res, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}