// #include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <../CImg/CImg.h>
#include <vector>

#include "denoise.h"
#include "../common.h"

using namespace cimg_library;

void OMPMedianFilterDenoise(CImg<unsigned char> &orig, CImg<unsigned char> &res, int k, int perc)
{
    int mid = (k - 1) / 2;

    // 1 channel + 3 colors + 0 prefilled
    res.assign(orig.width() - 2 * mid, orig.height() - 2 * mid, 1, 3, 0);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = mid; i < orig.height() - mid; i++)
    {
        #pragma omp parallel for schedule(dynamic, 8)
        for (int j = mid; j < orig.width() - mid; j++)
        {
            std::vector<rgb_t> rgbNeighbors;

            rgb_t rgbAccumulate(0.0f, 0.0f, 0.0f); 

            for (int x = i - mid; x <= i + mid; x++)
            {
                for (int y = j - mid; y <= j + mid; y++)
                {
                    rgbNeighbors.push_back(rgb_t(orig(y, x, 0, 0), orig(y, x, 0, 1), orig(y, x, 0, 2)));
                }
            }

            int median = (k % 2) ? (k * k - 1) / 2 : k * k / 2;

            int n_elements = (k * k) / 2 * (int(perc) / 100);

            for (int colors = 0; colors < 3; colors++) {
                if (colors == 0) {
                    std::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareR);
                    for (int idx = (median - n_elements); idx <= (median + n_elements); idx++)
                    {
                        rgbAccumulate.r += rgbNeighbors[idx].r;
                    }
                }
                else if (colors == 1) {
                    std::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareG);
                    for (int idx = (median - n_elements); idx <= (median + n_elements); idx++)
                    {
                        rgbAccumulate.g += rgbNeighbors[idx].g;
                    }
                }
                else {
                    std::sort(rgbNeighbors.begin(), rgbNeighbors.end(), compareB);
                    for (int idx = (median - n_elements); idx <= (median + n_elements); idx++)
                    {
                        rgbAccumulate.b += rgbNeighbors[idx].b;
                    }
                }
            }

            if (n_elements >= 1)
            {
                rgbAccumulate *= rgb_t(1./float(2 * n_elements + 1), 1./float(2 * n_elements + 1), 1./float(2 * n_elements + 1));
            }

            res(j - mid, i - mid, 0, 0) = rgbAccumulate.r;
            res(j - mid, i - mid, 0, 1) = rgbAccumulate.g;
            res(j - mid, i - mid, 0, 2) = rgbAccumulate.b;
        }
    }
}

int main()
{
    CImg<unsigned char> orig("input.png"); // Load input image
    CImg<unsigned char> res;               // Define output image

    Timer TotalSimulationTimer;
    OMPMedianFilterDenoise(orig, res, 5, 50); // Apply denoising with 5x5 kernel and 50% percile
    float totalSimulationTime = TotalSimulationTimer.elapsed();
    std::cout << "Total simulation time: "<< totalSimulationTime << std::endl;

    res.save("output.png"); // Save output image
    return 0;
}