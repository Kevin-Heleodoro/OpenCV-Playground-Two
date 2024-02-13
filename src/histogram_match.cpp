// Author: Kevin Heleodoro
// Date: February 5, 2024
// Purpose: Given a directory of images and feature set it uses a single normalized color histogram
//         to match the target image to the images in the directory.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "csv_util.h"
#include "feature_utils.h"

/**
 * @brief Calculate the intersection of two histograms
 *
 * @param histA The first histogram
 * @param histB The second histogram
 * @return float The intersection of the two histograms
 */
float histIntersect(const cv::Mat &histA, const cv::Mat &histB)
{
    printf("Calculating histogram intersection ...\n");
    printf("histA size: %d x %d\n", histA.rows, histA.cols);
    printf("histB size: %d x %d\n", histB.rows, histB.cols);
    CV_Assert(histA.size == histB.size);
    float intersection;
    for (int h = 0; h < histA.rows; h++)
    {
        for (int s = 0; s < histA.cols; s++)
        {
            // Intersection is the minimum value of the two histograms at each bin
            intersection += std::min(histA.at<float>(h, s), histB.at<float>(h, s));
        }
    }
    return intersection;
}

cv::Mat calcRgbHist(const cv::Mat &image, int histSize)
{
    printf("Calculating RGB chromaticity histogram ...\n");
    // Initialize histogram
    cv::Mat hist = cv::Mat::zeros(cv::Size(histSize, histSize), CV_32FC1);
    cv::Mat src;
    image.copyTo(src);
    float max = 0;

    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++)
        {
            // Get RGB values
            float blue = ptr[j][0];
            float green = ptr[j][1];
            float red = ptr[j][2];

            // Compute the r,g chromaticity
            float divisor = red + green + blue;
            divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros
            float r = red / divisor;
            float g = green / divisor;

            // Calculate bin index
            int gIndex = static_cast<int>(g * (histSize - 1) + 0.5);
            int rIndex = static_cast<int>(r * (histSize - 1) + 0.5);

            // Check if the indices are within the range of the histogram
            if (gIndex >= 0 && gIndex < histSize && rIndex >= 0 && rIndex < histSize)
            {
                // Increment histogram
                hist.at<float>(rIndex, gIndex)++;
            }
        }
    }

    printf("Normalizing histogram ...\n");
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    return hist;
}

/**
 * @brief Calculate the histogram of an image
 *
 * @param hsvImage The HSV image
 * @param hBins The number of hue bins
 * @param sBins The number of saturation bins
 * @return cv::Mat The histogram
 */
cv::Mat calcHsvHist(const cv::Mat &hsvImage, int hBins, int sBins)
{
    printf("Calculating HSV histogram ...\n");
    // Initialize histogram
    cv::Mat hist = cv::Mat::zeros(hBins, sBins, CV_32F);

    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            // Get HSV values
            uchar hue = hsvImage.at<cv::Vec3b>(i, j)[0];
            uchar sat = hsvImage.at<cv::Vec3b>(i, j)[1];

            // Calculate bin index
            int hIndex = static_cast<int>((hue * hBins) / 180.0);
            int sIndex = static_cast<int>((sat * sBins) / 256.0);

            // Increment histogram
            hist.at<float>(hIndex, sIndex) += 1.0f;
        }
    }

    printf("Normalizing histogram ...\n");
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    return hist;
}

std::vector<std::pair<std::string, float>> compareHistograms(struct dirent *dp, char *dirPath, DIR *dirp,
                                                             char *targetImagePath, cv::Mat targetHist, char *buffer,
                                                             int histType)
{
    printf("Processing images in directory ...\n");
    std::vector<std::pair<std::string, float>> imageMatches;

    while ((dp = readdir(dirp)) != NULL)
    {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            // printf("processing image file: %s\n", dp->d_name);
            strcpy(buffer, dirPath);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            cv::Mat src = cv::imread(buffer);
            if (!src.data)
            {
                printf("No image data\n");
                continue;
            }

            if (strcmp(buffer, targetImagePath) == 0)
            {
                continue;
            }

            cv::Mat srcHist;
            // float distance;
            if (histType == 1)
            {
                int hBins = 30;
                int sBins = 30;
                cv::cvtColor(src, srcHist, cv::COLOR_BGR2HSV);
                srcHist = calcHsvHist(srcHist, hBins, sBins);
                // distance = histIntersect(targetHist, srcHist);
            }
            else if (histType == 0)
            {
                int histSize = 256;
                // printf("histSize: %d\n", histSize);
                srcHist = calcRgbHist(src, histSize);
                // printf("srcHist size: %d x %d\n", srcHist.rows, srcHist.cols);
                // distance = histIntersect(targetHist, srcHist);
            }

            float distance = histIntersect(targetHist, srcHist);
            imageMatches.push_back(std::make_pair(dp->d_name, distance));
        }
    }

    return imageMatches;
}

/**
 * @brief Main function to find the top N matches for a target image in a directory of images
 *
 * This function takes a target image and creates an HSV histogram for it. It then compares the target histogram to the
 * histograms of all the images in the sample_images directory and prints the top 3 matches.
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @return int The exit status
 */
int main(int argc, char *argv[])
{
    if (argc < 1)
    {
        printf("Usage: %s <targetImage> [histogramType] [vectorCsvFile] \n", argv[0]);
        printf("Histogram type: 0 for RGB, 1 for HSV, 2 for both\n");
        exit(-1);
    }

    printf("\n\n========== Histogram Match ==========\n\n");

    cv::Mat image;
    cv::Mat hsvImage;
    cv::Mat rgbImage;
    cv::Mat targetHsvHist;
    cv::Mat targetRgbHist;
    char targetImagePath[256];
    char vectorCsv[256];
    int histogramType = 1;
    const int hBins = 30;
    const int sBins = 30;
    const int histSize = 256;
    DIR *dirp;

    strcpy(targetImagePath, argv[1]);
    printf("Target image set to %s\n", targetImagePath);
    image = cv::imread(targetImagePath);
    if (!image.data)
    {
        printf("No image data\n");
        return -1;
    }

    if (argc < 3)
    {
        printf("Using default histogram type: 1 (HSV)\n");
        histogramType = 1;
    }
    else
    {
        histogramType = atoi(argv[2]);
        printf("Using histogram type: %d\n", histogramType);
    }

    if (histogramType == 1 || histogramType == 2)
    {
        // convert image to HSV
        printf("Converting image to HSV ...\n");
        cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

        printf("Creating histogram with %d hue bins and %d saturation bins ...\n", hBins, sBins);
        targetHsvHist = calcHsvHist(hsvImage, hBins, sBins);
        printf("Target HSV histogram size: %d x %d\n", targetHsvHist.rows, targetHsvHist.cols);
    }

    if (histogramType == 0 || histogramType == 2)
    {
        // convert image to RGB
        printf("Creating histogram with %d bins ...\n", histSize);
        targetRgbHist = calcRgbHist(rgbImage, histSize);
        printf("Target RGB histogram size: %d x %d\n", targetRgbHist.rows, targetRgbHist.cols);
    }

    printf("Comparing target histogram to images in directory ...\n");
    dirp = opendir("./sample_images");
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", "./sample_images");
        exit(-1);
    }

    std::vector<std::pair<std::string, float>> hsvImageMatches;
    std::vector<std::pair<std::string, float>> rgbImageMatches;
    char buffer[256];
    char dirPath[256] = "./sample_images";
    struct dirent *dp;

    if (histogramType == 2)
    {
        printf("\nCalculating both HSV and RGB histograms...\n");
        hsvImageMatches = compareHistograms(dp, dirPath, dirp, targetImagePath, targetHsvHist, buffer, 1);
        rgbImageMatches = compareHistograms(dp, dirPath, dirp, targetImagePath, targetRgbHist, buffer, 0);
    }
    else if (histogramType == 1)
    {
        printf("\nCalculating HSV histograms...\n");
        hsvImageMatches = compareHistograms(dp, dirPath, dirp, targetImagePath, targetHsvHist, buffer, 1);
    }
    else if (histogramType == 0)
    {
        printf("\nCalculating RGB histograms...\n");
        rgbImageMatches = compareHistograms(dp, dirPath, dirp, targetImagePath, targetRgbHist, buffer, 0);
    }

    // printf("Processing images in directory ...\n");
    // while ((dp = readdir(dirp)) != NULL)
    // {
    //     if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") ||
    //         strstr(dp->d_name, ".tif"))
    //     {
    //         // printf("processing image file: %s\n", dp->d_name);
    //         strcpy(buffer, dirPath);
    //         strcat(buffer, "/");
    //         strcat(buffer, dp->d_name);

    //         cv::Mat src = cv::imread(buffer);
    //         if (!src.data)
    //         {
    //             printf("No image data\n");
    //             return -1;
    //         }

    //         if (strcmp(buffer, targetImagePath) == 0)
    //         {
    //             continue;
    //         }

    //         // convert image to HSV
    //         cv::Mat srcHsv;
    //         cv::cvtColor(src, srcHsv, cv::COLOR_BGR2HSV);

    //         cv::Mat srcHist = calcHsvHist(srcHsv, hBins, sBins);
    //         float distance = histIntersect(targetHsvHist, srcHist);
    //         imageMatches.push_back(std::make_pair(dp->d_name, distance));
    //     }
    // }

    std::vector<std::pair<std::string, float>> imageMatches;

    if (histogramType == 2)
    {
        printf("Combining matches...\n");
        for (int i = 0; i < hsvImageMatches.size(); i++)
        {
            std::string filename = hsvImageMatches[i].first;
            float distance = hsvImageMatches[i].second;
            distance += rgbImageMatches[i].second;
            imageMatches.push_back(std::make_pair(filename, distance));
        }
    }
    else if (histogramType == 1)
    {
        imageMatches = hsvImageMatches;
    }
    else if (histogramType == 0)
    {
        imageMatches = rgbImageMatches;
    }

    printf("Sorting matches...\n");
    std::sort(imageMatches.begin(), imageMatches.end(),
              [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
                  return a.second > b.second;
              });

    printf("\n=====================================\n\n");
    printf("Top 3 matches for %s:\n", targetImagePath);
    // for (int i = 0; i < 3; i++)
    for (int i = 0; i < imageMatches.size(); i++)
    {
        printf("%s: %f\n", imageMatches[i].first.c_str(), imageMatches[i].second);
    }

    printf("Terminating\n\n");

    return 0;
}