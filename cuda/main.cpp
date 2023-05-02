#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <../CImg/CImg.h>
#include <vector>

#include "main.h"
#include "../common.h"

using namespace cimg_library;

// Conversion
void CUDARGBtoYCbCr(const unsigned char *img, unsigned char *res, int width, int height);
void CUDAYCbCrtoRGB(const unsigned char *img, unsigned char *res, int width, int height);

// Enhancement
void CUDAHistogramEqualization(const unsigned char *img, unsigned char *res, const int width, const int height);
void CUDACLAHE(const unsigned char *img, unsigned char *res, const int width, const int height, const int clipLimit, const int gridSize);

// Denoising
void CUDAMedianFilterDenoise(const unsigned char *img, unsigned char *res, const int perc, const int width, const int height);

int main()
{
    CImg<unsigned char> inputImage("images-input/inputDenoising.png");
    const int k = 5;
    CImg<unsigned char> outputImage(inputImage.width()-2*(k-1)/2, inputImage.height()-2*(k-1)/2, 1, 3);
    CUDAMedianFilterDenoise(inputImage, outputImage, 50, inputImage.width(), inputImage.height());
    cimg_forXY(outputImage, x, y) {
        printf("\n[%d][%d] r %d g %d b %d\n", x, y, outputImage(x,y,0,0), outputImage(x,y,0,1), outputImage(x,y,0,2));
    }
    outputImage.save("images-output/denoise-median.png");
//     /////////////////////
//     /* - Parse Input - */
//     /////////////////////

//     std::string functionality;
//     std::string algorithm;
//     std::string inputPath;
//     std::string outputPath;

//     std::cout << "Enter functionality (TODO/enhance/conversion): ";
//     std::cin >> functionality;

//     if (functionality == "enhance")
//     {
//         std::cout << "Enter algorithm (equalization/clahe): ";
//         std::cin >> algorithm;
//     }
//     else if (functionality == "conversion")
//     {
//         ;
//     }
//     else
//     {
//         std::cerr << "Invalid Functionality Selection" << std::endl;
//         return 1;
//     }

//     std::cout << "Enter input image path: ";
//     std::cin >> inputPath;

//     ///////////////////////
//     /* - Run Algorithm - */
//     ///////////////////////

//     CImg<unsigned char> inputImage(inputPath.c_str());
//     Timer MyTimer;

//     if (functionality == "enhance")
//     {
//         ///////////////////
//         /* - Enhancing - */
//         ///////////////////

//         if (algorithm == "equalization")
//         {
//             CImg<unsigned char> outputImage(inputImage.width(), inputImage.height(), 1, 3);
//             CUDAHistogramEqualization(inputImage, outputImage, inputImage.width(), inputImage.height());

//             float duration = MyTimer.elapsed();
//             outputImage.save("images-output/enhance-equalization.png");
//             std::cout << "Simulation time: " << duration << std::endl;
//         }
//         else if (algorithm == "clahe")
//         {
//             int clipLimit = 2;
//             int gridSize = 96;
//             CImg<unsigned char> outputImage(inputImage.width(), inputImage.height(), 1, 3);
//             CUDACLAHE(inputImage, outputImage, inputImage.width(), inputImage.height(), clipLimit, gridSize);

//             float duration = MyTimer.elapsed();
//             outputImage.save("images-output/enhance-clahe.png");
//             std::cout << "Simulation time: " << duration << std::endl;
//         }
//         else
//         {
//             std::cerr << "Error: Unknown enhancement algorithm (Choices: equalization / clahe)\n";
//             return 1;
//         }
//     }
//     else if (functionality == "conversion")
//     {
//         ////////////////////
//         /* - Conversion - */
//         ////////////////////

//         // RGB to YCBCR + Grayscale
//         CImg<unsigned char> resgray(inputImage.width(), inputImage.height(), 1, 3);
//         CImg<unsigned char> resrgb(inputImage.width(), inputImage.height(), 1, 3);
//         CUDARGBtoYCbCr(inputImage, resgray, inputImage.width(), inputImage.height());

//         float duration = MyTimer.elapsed();
//         std::cout << "YCBCR/Grayscale simulation time: " << duration << std::endl;
//         resgray.save("images-output/conversion-ycbcr.png");
//         resgray.get_channel(0).save("images-output/conversion-grayscale.png");

//         // YCBCR to RGB
//         Timer RGBSimulationTimer;
//         CUDAYCbCrtoRGB(resgray, resrgb, resgray.width(), resgray.height());
//         float rgbSimulationTime = RGBSimulationTimer.elapsed();
//         std::cout << "RGB simulation time: " << rgbSimulationTime << std::endl;
//         resrgb.save("images-output/conversion-rgb.png");
//     }
}
