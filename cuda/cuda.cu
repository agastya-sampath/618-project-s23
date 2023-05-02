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

/////////////////////////
/* - Enhance (CLAHE) - */
/////////////////////////

__global__ void claheKernel(const unsigned char *img, unsigned char *res, const float max_slope, const int half_win, const int width, const int height, const int gridSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    // Calculate the range of pixels within the current window
    const int x_min = fmaxf(0, x - half_win);
    const int x_max = fminf(width - 1, x + half_win);
    const int y_min = fmaxf(0, y - half_win);
    const int y_max = fminf(height - 1, y + half_win);

    // Compute the histogram for the current window
    int local_hist_r[256], local_hist_g[256], local_hist_b[256];
    memset(local_hist_r, 0, 256 * sizeof(int));
    memset(local_hist_g, 0, 256 * sizeof(int));
    memset(local_hist_b, 0, 256 * sizeof(int));

    for (int i = x_min; i <= x_max; i++)
    {
        for (int j = y_min; j <= y_max; j++)
        {
            int offset = j * width + i;

            local_hist_r[img[offset]]++;
            local_hist_g[img[offset + width * height]]++;
            local_hist_b[img[offset + 2 * width * height]]++;

            // Note: somehow "atomicAdd" is not working
            // atomicAdd(&local_hist_r[img[offset]], 1);
            // atomicAdd(&local_hist_g[img[offset + width * height]], 1);
            // atomicAdd(&local_hist_b[img[offset + 2 * width * height]], 1);
        }
    }

    // Calculate the cumulative histogram for the current window for each channel
    float cum_hist_r_local[256], cum_hist_g_local[256], cum_hist_b_local[256];
    cum_hist_r_local[0] = local_hist_r[0] / (float)(gridSize * gridSize);
    cum_hist_g_local[0] = local_hist_g[0] / (float)(gridSize * gridSize);
    cum_hist_b_local[0] = local_hist_b[0] / (float)(gridSize * gridSize);

    for (int i = 1; i < 256; i++)
    {
        cum_hist_r_local[i] = cum_hist_r_local[i - 1] + local_hist_r[i] / (float)(gridSize * gridSize);
        cum_hist_g_local[i] = cum_hist_g_local[i - 1] + local_hist_g[i] / (float)(gridSize * gridSize);
        cum_hist_b_local[i] = cum_hist_b_local[i - 1] + local_hist_b[i] / (float)(gridSize * gridSize);
    }

    float cum_hist_avg_local[256];
    for (int i = 0; i < 256; i++)
    {
        cum_hist_avg_local[i] = (cum_hist_r_local[i] + cum_hist_g_local[i] + cum_hist_b_local[i]) / 3.0;
    }

    // 3 channels - hardcoding
    for (int c = 0; c < 3; c++)
    {
        // Calculate the equalized value for each pixel in the current channel
        for (int x_res = x_min; x_res <= x_max; x_res++)
        {
            for (int y_res = y_min; y_res <= y_max; y_res++)
            {
                // Get the original pixel
                const int x_orig = fmaxf(0, fminf(width - 1, x_res));
                const int y_orig = fmaxf(0, fminf(height - 1, y_res));

                int offset = y_orig * width + x_orig;
                const int val_orig = img[offset + c * width * height];

                // Calculate the equalized value
                const float cum_prob = cum_hist_avg_local[val_orig];
                const float eq_val = fmaxf(0.0f, fminf(255.0f, 256 * (cum_prob - max_slope) / (1 - max_slope)));
                offset = y_res * width + x_res;
                res[offset + c * width * height] = (unsigned char)eq_val;
            }
        }
    }
}

void CUDACLAHE(const unsigned char *img, unsigned char *res, const int width, const int height, const int clipLimit, const int gridSize)
{
    // Calculate the maximum allowed slope
    const float max_slope = clipLimit / (float)(gridSize * gridSize);

    // Calculate the half window size for the local histograms
    const int half_win = gridSize / 2;

    // Compute histograms
    dim3 blockDims(4, 4);
    dim3 gridDims((width + blockDims.x - 1) / blockDims.x, (height + blockDims.y - 1) / blockDims.y);

    // Allocate device memory
    unsigned char *d_img, *d_res;
    cudaMalloc((void **)&d_img, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_res, width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(d_img, img, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Call the kernel
    claheKernel<<<gridDims, blockDims>>>(d_img, d_res, max_slope, half_win, width, height, gridSize);
    cudaMemcpy(res, d_res, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_res);
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

////////////////////
/* - Denoising - */
////////////////////

#define k 5

__global__ void MedianFilterKernel(const unsigned char* orig, unsigned char* res, const int width, const int height, const int perc)
{
    int mid = (k - 1) / 2;
    int i = blockIdx.y * blockDim.y + threadIdx.y + mid;
    int j = blockIdx.x * blockDim.x + threadIdx.x + mid;

    if (i < height - mid && j < width - mid)
    {
        int n_pixels = k * k;
        int n_neighbors = n_pixels / 2 * (int(perc) / 100);

        // Find median pixel index
        int median = (k % 2) ? (n_pixels - 1) / 2 : n_pixels / 2;

        // Store R, G, B values of neighbors in local array
        float neighbors[3][k*k];
        for (int y = -mid; y <= mid; y++)
        {
            for (int x = -mid; x <= mid; x++)
            {
                int idx = (y + mid) * k + x + mid;
                neighbors[0][idx] = orig[(i + y) * width + (j + x)]; // R
                neighbors[1][idx] = orig[(i + y) * width + (j + x) + height * width]; // G
                neighbors[2][idx] = orig[(i + y) * width + (j + x) + 2 * height * width]; // B
            }
        }

        // Sort R, G, B values independently
        for (int c = 0; c < 3; c++)
        {
            for (int p = 0; p < n_pixels; p++)
            {
                for (int q = p + 1; q < n_pixels; q++)
                {
                    if (neighbors[c][p] > neighbors[c][q])
                    {
                        float temp = neighbors[c][p];
                        neighbors[c][p] = neighbors[c][q];
                        neighbors[c][q] = temp;
                    }
                }
            }
        }

        // Compute average RGB values of neighbors within range
        float accumRGB[3] = { 0.0f, 0.0f, 0.0f };
        for (int p = median - n_neighbors; p <= median + n_neighbors; p++)
        {
            accumRGB[0] += neighbors[0][p];
            accumRGB[1] += neighbors[1][p];
            accumRGB[2] += neighbors[2][p];
        }
        accumRGB[0] /= (2 * n_neighbors + 1);
        accumRGB[1] /= (2 * n_neighbors + 1);
        accumRGB[2] /= (2 * n_neighbors + 1);


        // Assign to result image
        int idx = (i - mid) * (width - 2 * mid) + (j - mid);
        res[idx] = (unsigned char)accumRGB[0];
        res[idx + (width - 2 * mid) * (height - 2 * mid)] = (unsigned char)accumRGB[1];
        res[idx + 2 * (width - 2 * mid) * (height - 2 * mid)] = (unsigned char)accumRGB[2];
    }
}


void CUDAMedianFilterDenoise(const unsigned char *img, unsigned char *res, const int perc, const int width, const int height)
{
    // const int k = 3; // Kernel size
    const int mid = (k - 1) / 2; // Half of kernel size
    // const int n_pixels = k * k;
    // const int n_neighbors = n_pixels / 2 * (perc / 100);

    unsigned char *dev_img, *dev_res;
    cudaMalloc((void**)&dev_img, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&dev_res, (width - 2 * mid) * (height - 2 * mid) * 3 * sizeof(unsigned char));

    // Copy input image to device memory
    cudaMemcpy(dev_img, img, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(4, 4);
    dim3 gridSize((width - 2 * mid + blockSize.x - 1) / blockSize.x, (height - 2 * mid + blockSize.y - 1) / blockSize.y);

    // Call kernel
    MedianFilterKernel<<<gridSize, blockSize>>>(dev_img, dev_res, width, height, perc);

    // Copy result image from device memory
    cudaMemcpy(res, dev_res, (width - 2 * mid) * (height - 2 * mid) * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_img);
    cudaFree(dev_res);
}
