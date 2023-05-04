# Serial
## Denoising - Median Filtering
```scala
Input: orig (original image), res (resulting image), k (kernel size), perc (percentage scaling)
Output: res (resulting image) -> by reference

1. Set mid = (k - 1) / 2
2. Set res size to: (orig.width() - 2 * mid, orig.height() - 2 * mid), since we loose resolution
3. For each pixel in image (excluding padding):
4.      Create empty vector of rgb values called rgbNeighbors
5.      Create rgbAccumulate initialized to 0
6.      For each pixel within the filter range of the current pixel:
7.          Append the RGB value to rgbNeighbors
8.      Find the median pixel value within rgbNeighbors (main sorting happens here)
9.      Set n_neighbors to k^2 / 2 times the percentage scaling (filter size * the percentage scaling)
10.     For each color (R, G, B):
11.         Sort rgbNeighbors based on current color
12.         Sum the n_neighbors pixel values around the median value for current color
13.     Divide each accumulated color value by 2*n_neighbors+1 to get the average
14.     Assign the averaged RGB value to the corresponding pixel in res
```

## Denoising - NLM Filtering
```scala
Input: image (original image), res (resulting image) -> by reference, h (parameter for Non-Local Means denoising), patchSize (size of the patch window), searchWindowSize (size of the search window)
Output: res (resulting image) -> by reference

1. Calculate halfPatchSize = patchSize / 2 and halfSearchWindowSize = searchWindowSize / 2
2. Set res = image
3. For each pixel (x,y) in the image:
4.      Initialize rgbAccumulate to (0,0,0) and weightSum to 0
5.      For each pixel (i,j) in the search window around (x,y):
6.          If (i,j) is within the bounds of the image:
7.              Initialize rgbPatch to (0,0,0)
8.              For each pixel (k,l) in the patch window around (i,j):
9.                  If (k,l) is within the bounds of the image:
10.                     Calculate rDistance, gDistance, and bDistance between the patch window pixel and the search window pixel
11.                     Calculate patchDistance = rDistance + gDistance + bDistance
12.                     Calculate weight = exp(-patchDistance / (h * h))
13.                     Update rgbPatch += (image(k,l,0), image(k,l,1), image(k,l,2)) * weight
14.                     Update weightSum += weight
15.             Update rgbAccumulate += rgbPatch
16.      Calculate the final pixel values for (x,y) by averaging the accumulated RGB values
```

## Enhancement - Histogram Equalization
```scala
Input: img (original image), res (resulting image) -> by reference
Output: res (resulting image) -> by reference

1. Set res equal to img
2. Set width equal to img.width()
3. Set height equal to img.height()
4. Set bins equal to 256 (for color channels)
5. Create empty histogram for each color channel: hist_r, hist_g, hist_b
6. Set each element of hist_r, hist_g, and hist_b to 0
7. For each pixel (x, y) in img:
8.      Increment the corresponding bin in hist_r, hist_g, and hist_b by 1
9.      Compute the average histogram, hist_avg, by averaging the corresponding values in hist_r, hist_g, and hist_b
10.     Create an empty cumulative histogram, cum_hist
11.     Set the first element of cum_hist to the first element of hist_avg
12. For each element in hist_avg (starting at index 1):
13.     Set the corresponding element in cum_hist to the sum of the current element in hist_avg and the previous element in cum_hist
14. For each pixel (x, y) in img:
15.     Get the R, G, and B values of the pixel
16.     Set the R, G, and B values of the corresponding pixel in res to the corresponding value in cum_hist scaled to the range [0, 255]    
```

## Enhancement - CLAHE
```scala
Input: img (original image), res (resulting image) -> by reference, clipLimit (integer), gridSize (integer)
Output: res (resulting image) -> by reference

1. Set res equal to img
2. Set width equal to img.width()
3. Set height equal to img.height()
4. Set bins equal to 256 (for color channels)
5. Calculate max_slope as clipLimit divided by gridSize squared
6. Calculate half_win as gridSize divided by 2
7. For each pixel (x, y) in img:
8.      Set x_min, x_max, y_min, y_max
9.      Create empty histograms for each color channel: local_hist_r, local_hist_g, local_hist_b
10.     For each pixel (i, j) within the range of (x_min, y_min) and (x_max, y_max):
11.          Increment the corresponding bin in local_hist_r, local_hist_g, and local_hist_b by 1
12.     For each element in local_hist_r, local_hist_g, and local_hist_b (starting at index 1):
13.         Set the corresponding element in cum_hist_r/g/b_local
14.     For each color channel c:
15.         Set cum_hist_local to cum_hist_avg_local
16.         For each pixel (x_res, y_res) within the range of (x_min, y_min) and (x_max, y_max):
17.             Set x_orig to the maximum between 0 and the minimum between width - 1 and x_res
18.             Obtain original pixel from the coordinates calculated above
19.             Calculate equalized value, and set the corresponding coordinate pixel of res
```

# Report Notes
## CImg RGB access
```cpp
// CImg Alignment Reference: https://cimg.eu/reference/storage.html
__global__ void kernel(){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // What we thought the access is:
    int offset = (y * width + x) * 3
    int r = img[offset];
    int g = img[offset + 1];
    int b = img[offset + 2];

    // What the access should be:
    int offset = y * width + x;
    int r = img[offset];
    int g = img[offset + width * height];
    int b = img[offset + 2 * width * height];
}
```
