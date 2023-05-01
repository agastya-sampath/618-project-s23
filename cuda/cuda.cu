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