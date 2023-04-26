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

__global__ void RGBtoYCbCrKernel(const unsigned char *img, unsigned char *res, const int w, const int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    const int idx = (y * w + x) * 3;
    const unsigned char r = img[idx];
    const unsigned char g = img[idx + 1];
    const unsigned char b = img[idx + 2];

    // Perform RGB to YCbCr conversion
    const double Y = 0.299 * r + 0.587 * g + 0.114 * b;
    const double Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    const double Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;

    // Store the converted values back into the image
    const int res_idx = (y * w + x) * 3;
    res[res_idx] = (unsigned char)Y;
    res[res_idx + 1] = (unsigned char)Cb;
    res[res_idx + 2] = (unsigned char)Cr;
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