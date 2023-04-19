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

__global__ void cudaMedianFilterDenoiseKernel(
    unsigned char *orig,
    int width, int height,
    int mid,
    int k,
    int n_neighbors,
    int median,
    int *indices,
    int *x_offsets,
    int *y_offsets,
    unsigned char *rgbNeighbors,
    int *rValues,
    int *gValues,
    int *bValues,
    int *rAccumulate,
    int *gAccumulate,
    int *bAccumulate)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * width + col;

    if (row >= height || col >= width)
        return;

    int x, y, nIndex, nIndexRGB;
    int r, g, b;
    int rVal, gVal, bVal;

    // Loop through all the neighbors
    for (int i = 0; i < n_neighbors; ++i)
    {
        x = col + x_offsets[i];
        y = row + y_offsets[i];

        if (x < 0 || y < 0 || x >= width || y >= height)
            continue;

        nIndex = y * width + x;
        nIndexRGB = nIndex * 3;

        r = orig[nIndexRGB];
        g = orig[nIndexRGB + 1];
        b = orig[nIndexRGB + 2];

        rValues[i] = r;
        gValues[i] = g;
        bValues[i] = b;
        indices[i] = i;
    }

    // Sort the RGB values
    thrust::sort_by_key(thrust::device, indices, indices + n_neighbors, rgbNeighbors);

    // Compute the median RGB value
    rVal = rgbNeighbors[median];
    gVal = rgbNeighbors[median + k];
    bVal = rgbNeighbors[median + 2 * k];

    // Update the result
    rAccumulate[index] = rVal;
    gAccumulate[index] = gVal;
    bAccumulate[index] = bVal;
}

void CUDAMedianFilterDenoise(const unsigned char *orig_data, unsigned char *res, int orig_width, int orig_height, int orig_size,
                             int k, int perc)
{
    int mid = (k - 1) / 2;
    int size = (orig_width - 2 * mid) * (orig_height - 2 * mid);

    // Create device vectors
    thrust::device_vector<rgb_t> d_rgbNeighbors(k * k * size);
    thrust::device_vector<float> d_rValues(k * k * size);
    thrust::device_vector<float> d_gValues(k * k * size);
    thrust::device_vector<float> d_bValues(k * k * size);
    thrust::device_vector<float> d_rAccumulate(size);
    thrust::device_vector<float> d_gAccumulate(size);
    thrust::device_vector<float> d_bAccumulate(size);

    // Copy original image to device memory
    thrust::device_vector<unsigned char> d_orig(orig_data, orig_data + orig_size);

    // Initialize indices vector
    thrust::device_vector<int> indices(k * k);
    thrust::sequence(indices.begin(), indices.end());

    // Compute offsets for neighbor pixels
    thrust::device_vector<int> x_offsets(k * k);
    thrust::device_vector<int> y_offsets(k * k);
    for (int x = -mid, idx = 0; x <= mid; x++)
    {
        for (int y = -mid; y <= mid; y++, idx++)
        {
            x_offsets[idx] = x;
            y_offsets[idx] = y;
        }
    }

    // Compute the median index
    int median = (k % 2) ? (k * k - 1) / 2 : k * k / 2;

    // Compute the number of neighbors to consider
    int n_neighbors = (k * k) / 2 * (int(perc) / 100);

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaMedianFilterDenoiseKernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_orig.data()),
        orig_width, orig_height,
        mid,
        k,
        n_neighbors,
        median,
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(x_offsets.data()),
        thrust::raw_pointer_cast(y_offsets.data()),
        thrust::raw_pointer_cast(d_rgbNeighbors.data()),
        thrust::raw_pointer_cast(d_rValues.data()),
        thrust::raw_pointer_cast(d_gValues.data()),
        thrust::raw_pointer_cast(d_bValues.data()),
        thrust::raw_pointer_cast(d_rAccumulate.data()),
        thrust::raw_pointer_cast(d_gAccumulate.data()),
        thrust::raw_pointer_cast(d_bAccumulate.data()));

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Compute final result
    thrust::device_vector<unsigned char> d_res(size * 3);
    thrust::transform(d_rAccumulate.begin(), d_rAccumulate.end(), d_gAccumulate.begin(), d_bAccumulate.begin(), d_res.begin() + 0, rgbToUnsignedChar());
    thrust::transform(d_gAccumulate.begin(), d_gAccumulate.end(), d_gAccumulate.begin(), d_res.begin() + 1, rgbToUnsignedChar());
    thrust::transform(d_bAccumulate.begin(), d_bAccumulate.end(), d_bAccumulate.begin(), d_res.begin() + 2, rgbToUnsignedChar());

    // Copy final result back to host memory
    thrust::host_vector<unsigned char> h_res = d_res;
    res.assign(orig_width - 2 * mid, orig_height - 2 * mid, 1, 3, 0);

    for (int i = mid; i < orig_height - mid; i++) // No padding
    {
        for (int j = mid; j < orig_width - mid; j++) // No padding
        {
            int offset = (i - mid) * (orig_width - 2 * mid) * 3 + (j - mid) * 3;
            res(j - mid, i - mid, 0, 0) = h_res[offset];
            res(j - mid, i - mid, 0, 1) = h_res[offset + 1];
            res(j - mid, i - mid, 0, 2) = h_res[offset + 2];
        }
    }
}
