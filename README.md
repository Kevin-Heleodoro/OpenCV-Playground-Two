# Project Two

## Tasks

Inputs:
T = target image
B = image database
F = method for computing features
D(Ft, Fi) = distance metric for two images
N = desired number of output images

-   [ ] Baseline Matching
-   [ ] Histogram Matching
-   [ ] Multi-histogram Matching
-   [ ] Texture and Color
-   [ ] Deep Network Embeddings
-   [ ] Compare DNN Embeddings and Classic Features
-   [ ] Custom Design

## How to compile

> Make sure you have opencv installed on your system.

1. To compile a new executable enter the following in the terminal from within the `/src` dir:
   `make <insert name from makefile>` i.e. `make photo` for the photo viewer.

2. Then navigate to the `/bin` directory and run the name of the executable i.e. `./photo.exe`

## Run these programs

1. Navigate to the directory containing the `.exe` files and run them as such:

-   `./photo.exe starry_night.jpg`
    > Relative path to any image file.
-   `./vid.exe`
    > Ensure the `haarcascade_frontalface_alt2.xml` file is in the same directory.

### Baseline Matching

v1.0

This version iterates through the sample_images directory every time it is called to match the
target image vectors to the top N matches within the directory. This is not sustainable long-term
if we will be comparing many images.

![feature extracting and matching](./data/screenshots/baseline_match_v1.0.png)

v2.0

This version consists of two parts. The first extracts all of the features into a .csv file and the second references the feature vectors within the csv file instead of having to create them
every iteration

![feature extraction](./data/screenshots/feature_extraction.png)

![matching](./data/screenshots/baseline_match_v2.0.png)

### Histogram Matching

Use a single normalized color histogram as the feature vector. Use histogram intersection as the distance metric.

**The RGB histogram is probably printing out 0.000 for all distances**

```shell
Top 3 matches for ./sample_images/pic.0219.jpg:
pic.0066.jpg9_kmeans.jpg: 0.000000
pic.0236.jpg: 0.000000
pic.0168.jpg: 0.000000
pic.0183.jpg: 0.000000
pic.0186.jpg: 0.000000
pic.0135.jpg: 0.000000
pic.0121.jpg: 0.000000
pic.0281.jpg: 0.000000
pic.0097.jpg: 0.000000
pic.0066.jpg6_kmeans.jpg: 0.000000
pic.0078.jpg: 0.000000
pic.0066.jpg3_kmeans.jpg: 0.000000
pic.0276.jpg: 0.000000
pic.0066.jpg: 0.000000
pic.0175.jpg: 0.000000
Terminating
```

> The solution for this was in the way I was initializing the rgb histogram.

```c++
if (histogramType == 0 || histogramType == 2)
{
    // convert image to RGB
    printf("Creating RG chromaticity histogram with %d bins ...\n", histSize);
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB); // I was assigning the color values to the wrong cv::Mat so rgbImage was being passed in without any actual information.
    targetRgbHist = calcRgbHist(rgbImage, histSize);
    printf("Target RGB histogram size: %d x %d\n", targetRgbHist.rows, targetRgbHist.cols);
}
```
