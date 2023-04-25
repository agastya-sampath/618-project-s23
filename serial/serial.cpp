// #include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <../CImg/CImg.h>
#include <vector>

#include "serial.h"
#include "../common.h"

using namespace cimg_library;

void SerialMedianFilterDenoise(CImg<unsigned char> &orig, CImg<unsigned char> &res, int k, int perc)
{
    int mid = (k - 1) / 2;

    // 1 channel + 3 colors + 0 prefilled
    res.assign(orig.width() - 2 * mid, orig.height() - 2 * mid, 1, 3, 0);

    for (int i = mid; i < orig.height() - mid; i++) // No padding
    {
        for (int j = mid; j < orig.width() - mid; j++) // No padding
        {
            std::vector<rgb_t> rgbNeighbors;

            rgb_t rgbAccumulate(0.0f, 0.0f, 0.0f);

            // Find neighbors of current pixel within filter range
            for (int x = -mid; x <= mid; x++)
            {
                for (int y = -mid; y <= mid; y++)
                {
                    rgbNeighbors.push_back(rgb_t(orig(j + y, i + x, 0, 0), orig(j + y, i + x, 0, 1), orig(j + y, i + x, 0, 2)));
                }
            }

            // Find median pixel
            int median = (k % 2) ? (k * k - 1) / 2 : k * k / 2;

            // Number of neighbors (filter size * the percentage scaling)
            int n_neighbors = (k * k) / 2 * (int(perc) / 100);

            for (int colors = 0; colors < 3; colors++)
            {
                // Sort r values
                if (colors == 0)
                {
                    std::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareR);
                    for (int idx = -n_neighbors; idx <= n_neighbors; idx++)
                    {
                        rgbAccumulate.r += rgbNeighbors[median + idx].r;
                    }
                }
                // Sort g values
                else if (colors == 1)
                {
                    std::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareG);
                    for (int idx = -n_neighbors; idx <= n_neighbors; idx++)
                    {
                        rgbAccumulate.g += rgbNeighbors[median + idx].g;
                    }
                }
                // Sort b values
                else
                {
                    std::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareB);
                    for (int idx = -n_neighbors; idx <= n_neighbors; idx++)
                    {
                        rgbAccumulate.b += rgbNeighbors[median + idx].b;
                    }
                }
            }

            // Average
            rgbAccumulate *= rgb_t(1. / float(2 * n_neighbors + 1), 1. / float(2 * n_neighbors + 1), 1. / float(2 * n_neighbors + 1));

            // Assign to result
            res(j - mid, i - mid, 0, 0) = rgbAccumulate.r;
            res(j - mid, i - mid, 0, 1) = rgbAccumulate.g;
            res(j - mid, i - mid, 0, 2) = rgbAccumulate.b;
        }
    }
}

// Define the non-local means denoising function
void SerialNonLocalMeansDenoising(CImg<unsigned char> &image, CImg<unsigned char> &res, int h, int patchSize, int searchWindowSize)
{

    // Calculate the half patch and search window sizes
    int halfPatchSize = patchSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

    res = image;

    // Iterate over each pixel in the image
    cimg_forXY(image, x, y)
    {

        // Initialize the pixel values for the red, green, and blue channels
        rgb_t rgbAccumulate(0.0f, 0.0f, 0.0f);
        double weightSum = 0.0;

        // Iterate over each pixel in the search window
        for (int i = x - halfSearchWindowSize; i <= x + halfSearchWindowSize; i++)
        {
            for (int j = y - halfSearchWindowSize; j <= y + halfSearchWindowSize; j++)
            {

                // Make sure the search window pixel is within the image bounds
                if (i >= 0 && j >= 0 && i < image.width() && j < image.height())
                {

                    // Initialize the patch pixel values for the red, green, and blue channels
                    rgb_t rgbPatch(0.0f, 0.0f, 0.0f);

                    // Iterate over each pixel in the patch window
                    for (int k = i - halfPatchSize; k <= i + halfPatchSize; k++)
                    {
                        for (int l = j - halfPatchSize; l <= j + halfPatchSize; l++)
                        {

                            // Make sure the patch window pixel is within the image bounds
                            if (k >= 0 && l >= 0 && k < image.width() && l < image.height())
                            {

                                // Calculate the distance between the patch window and the search window pixels
                                double rDistance = pow(image(k, l, 0) - image(x, y, 0), 2);
                                double gDistance = pow(image(k, l, 1) - image(x, y, 1), 2);
                                double bDistance = pow(image(k, l, 2) - image(x, y, 2), 2);
                                double patchDistance = rDistance + gDistance + bDistance;

                                // Calculate the weight for the patch window pixel
                                double weight = exp(-patchDistance / (h * h));

                                // Update the patch pixel values
                                rgbPatch += rgb_t(image(k, l, 0), image(k, l, 1), image(k, l, 2)) * weight;

                                // Update the weight sum
                                weightSum += weight;
                            }
                        }
                    }

                    // Update the search window pixel values
                    rgbAccumulate += rgbPatch;
                }
            }
        }

        // Calculate the final pixel values for the red, green, and blue channels
        res(x, y, 0) = (unsigned char)std::round(rgbAccumulate.r / weightSum);
        res(x, y, 1) = (unsigned char)std::round(rgbAccumulate.g / weightSum);
        res(x, y, 2) = (unsigned char)std::round(rgbAccumulate.b / weightSum);
    }
}

void SerialHistogramEqualization(CImg<unsigned char> &img, CImg<unsigned char> &res)
{
    res = img;
    const int width = img.width();
    const int height = img.height();
    const int channels = img.spectrum();
    const int bins = 256;

    // Calculate histogram for each color channel
    CImg<unsigned int> hist_r(bins), hist_g(bins), hist_b(bins);
    hist_r.fill(0);
    hist_g.fill(0);
    hist_b.fill(0);
    cimg_forXY(img, x, y)
    {
        hist_r(img(x, y, 0), 0)++;
        hist_g(img(x, y, 1), 0)++;
        hist_b(img(x, y, 2), 0)++;
    }

    // Compute the average histogram
    CImg<float> hist_avg(bins);
    hist_avg.fill(0);
    for (int i = 0; i < bins; i++)
    {
        hist_avg(i) = (hist_r(i, 0) + hist_g(i, 0) + hist_b(i, 0)) / (float)(3 * width * height);
    }

    // Calculate cumulative histogram using the average
    CImg<float> cum_hist(bins);
    cum_hist(0) = hist_avg(0);
    for (int i = 1; i < bins; i++)
    {
        cum_hist(i) = cum_hist(i - 1) + hist_avg(i);
    }

    // Apply histogram equalization to each color channel
    cimg_forXY(img, x, y)
    {
        const int r = img(x, y, 0);
        const int g = img(x, y, 1);
        const int b = img(x, y, 2);
        res(x, y, 0) = (unsigned char)(cum_hist(r) * 255.0f);
        res(x, y, 1) = (unsigned char)(cum_hist(g) * 255.0f);
        res(x, y, 2) = (unsigned char)(cum_hist(b) * 255.0f);
    }
}

int main()
{
    // Load input image
    CImg<unsigned char> inputDenoising("images-input/inputDenoising.png");
    CImg<unsigned char> inputHequalization("images-input/inputHequalization.bmp");
    CImg<unsigned char> resmedian, resnlm, resheq, resclahe; // Define output image

    // Set denoising parameters
    int h = 30;
    int patchSize = 5;
    int searchWindowSize = 15;
    // Also tried : 20, 10, 25

    Timer NLMSimulationTimer;
    // Apply non-local means denoising
    SerialNonLocalMeansDenoising(inputDenoising, resmedian, h, patchSize, searchWindowSize);
    float nlmSimulationTime = NLMSimulationTimer.elapsed();
    std::cout << "NLM simulation time: " << nlmSimulationTime << std::endl;

    // Save denoised image
    resmedian.save("images-output/denoise-nlm.png");

    Timer MedianSimulationTimer;
    SerialMedianFilterDenoise(inputDenoising, resnlm, 5, 50); // Apply denoising with 5x5 kernel and 50% percile
    float medianSimulationTime = MedianSimulationTimer.elapsed();
    std::cout << "Median simulation time: " << medianSimulationTime << std::endl;

    resnlm.save("images-output/denoise-median.png"); // Save output image

    Timer HEqualizationSimulationTimer;
    SerialHistogramEqualization(inputHequalization, resheq);
    float hequalizationSimulationTime = HEqualizationSimulationTimer.elapsed();
    std::cout << "HEqualization simulation time: " << hequalizationSimulationTime << std::endl;

    resheq.save("images-output/enhance-equalization.png");

    return 0;
}