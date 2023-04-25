#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "denoise.h"

__global__ void MedianFilterDenoiseKernel(unsigned char *orig, unsigned char *res, int width, int height, int k, int perc)
{
    int mid = (k - 1) / 2;

    // Calculate the output indices for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= mid && x < width - mid && y >= mid && y < height - mid)
    {
        int idx = (y * width + x) * 3; // index of the current pixel in the original image

        thrust::device_vector<rgb_t> rgbNeighbors;
        rgb_t rgbAccumulate(0.0f, 0.0f, 0.0f);

        // Find neighbors of current pixel within filter range
        for (int i = -mid; i <= mid; i++)
        {
            for (int j = -mid; j <= mid; j++)
            {
                int neighborIdx = ((y + i) * width + (x + j)) * 3; // index of the current neighbor in the original image
                rgbNeighbors.push_back(rgb_t(orig[neighborIdx], orig[neighborIdx + 1], orig[neighborIdx + 2]));
            }
        }

        // Find median pixel
        int median = (k % 2) ? (k * k - 1) / 2 : k * k / 2;

        // Number of neighbors (filter size * the percentage scaling)
        int n_neighbors = (k * k) / 2 * (int(perc) / 100);

        for (int colors = 0; colors < 3; colors++)
        {
            // Sort values
            if (colors == 0)
            {
                thrust::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareR);
                for (int i = -n_neighbors; i <= n_neighbors; i++)
                {
                    rgb_t element = rgbNeighbors[median + i];
                    rgbAccumulate.r += element.r;
                }
            }
            else if (colors == 1)
            {
                thrust::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareG);
                for (int i = -n_neighbors; i <= n_neighbors; i++)
                {
                    rgb_t element = rgbNeighbors[median + i];
                    rgbAccumulate.g += element.g;
                }
            }
            else
            {
                thrust::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareB);
                for (int i = -n_neighbors; i <= n_neighbors; i++)
                {
                    rgb_t element = rgbNeighbors[median + i];
                    rgbAccumulate.b += element.b;
                }
            }
        }

        // Average
        rgbAccumulate *= rgb_t(1. / float(2 * n_neighbors + 1), 1. / float(2 * n_neighbors + 1), 1. / float(2 * n_neighbors + 1));

        // Assign to result
        int resIdx = ((y - mid) * (width - 2 * mid) + (x - mid)) * 3; // index of the current pixel in the result image
        res[resIdx] = rgbAccumulate.r;
        res[resIdx + 1] = rgbAccumulate.g;
        res[resIdx + 2] = rgbAccumulate.b;
    }
}

void CUDAMedianFilterDenoise(const unsigned char *orig_data, unsigned char *res, int orig_width, int orig_height, int orig_size, int k, int perc)
{
    // Calculate grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((orig_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (orig_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Allocate device memory for input and output data
    unsigned char *d_orig, *d_res;
    cudaMalloc((void **)&d_orig, orig_size);
    cudaMalloc((void **)&d_res, orig_size);

    // Copy input data from host to device
    cudaMemcpy(d_orig, orig_data, orig_size, cudaMemcpyHostToDevice);

    // Call kernel function
    MedianFilterDenoiseKernel<<<numBlocks, threadsPerBlock>>>(d_orig, d_res, orig_width, orig_height, k, perc);

    // Copy output data from device to host
    cudaMemcpy(res, d_res, orig_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_orig);
    cudaFree(d_res);
}