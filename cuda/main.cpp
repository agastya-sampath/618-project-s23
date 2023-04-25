#include <../CImg/CImg.h>
#include <iostream>
#include "../common.h"
#include "denoise.h"

void CUDAMedianFilterDenoise(const unsigned char *orig_data, unsigned char *res, int orig_width, int orig_height,
                             int orig_size, int k, int perc);

using namespace cimg_library;

void cudaDenoise(const char *filename, int k, int perc)
{
    // Load image using CImg
    CImg<unsigned char> orig(filename);

    // Create output image
    CImg<unsigned char> res(orig.width(), orig.height(), 1, 3);

    // Call CUDA kernel to denoise image
    CUDAMedianFilterDenoise(orig, res, orig.width(), orig.height(), orig.size(), k, perc);

    // Save denoised image
    res.save("denoised.png");
}

int main()
{
    cudaDenoise("input.png", 5, 50);
    return 0;
}
