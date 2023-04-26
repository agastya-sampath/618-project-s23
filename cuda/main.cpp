#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <../CImg/CImg.h>
#include <vector>

#include "main.h"
#include "../common.h"

using namespace cimg_library;

// Conversion - RGBtoYCbCr
void CUDARGBtoYCbCr(const unsigned char *img, unsigned char *res, int width, int height);

int main()
{
    CImg<unsigned char> inputImage("images-input/inputConversion.bmp");
    // create an empty output image with 3 channels
    CImg<unsigned char> outputImage(inputImage.width(), inputImage.height(), 1, 3);

    CUDARGBtoYCbCr(inputImage.data(), outputImage.data(), inputImage.width(), inputImage.height());

    outputImage.save("conversion-ycbcr.png");
}
