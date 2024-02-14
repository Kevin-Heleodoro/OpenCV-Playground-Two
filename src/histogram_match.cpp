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
#include "histogram_utils.h"

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
    const int histSize = 30;
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
        printf("Creating HSV histogram with %d hue bins and %d saturation bins ...\n", hBins, sBins);
        cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
        targetHsvHist = calcHsvHist(hsvImage, hBins, sBins);
        printf("Target HSV histogram size: %d x %d\n", targetHsvHist.rows, targetHsvHist.cols);
    }

    if (histogramType == 0 || histogramType == 2)
    {
        // convert image to RGB
        printf("Creating RG chromaticity histogram with %d bins ...\n", histSize);
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
        targetRgbHist = calcRgbHist(rgbImage, histSize);
        printf("Target RGB histogram size: %d x %d\n", targetRgbHist.rows, targetRgbHist.cols);
    }

    std::vector<std::pair<std::string, float>> hsvImageMatches;
    std::vector<std::pair<std::string, float>> rgbImageMatches;
    char buffer[256];
    char dirPath[256] = "./sample_images";
    struct dirent *dp;

    if (histogramType == 2)
    {
        printf("\nCalculating both HSV and RGB histograms...\n");
        hsvImageMatches = compareHistograms(dp, dirPath, targetImagePath, targetHsvHist, buffer, 1);
        rgbImageMatches = compareHistograms(dp, dirPath, targetImagePath, targetRgbHist, buffer, 0);
    }
    else if (histogramType == 1)
    {
        printf("\nCalculating HSV histograms...\n");
        hsvImageMatches = compareHistograms(dp, dirPath, targetImagePath, targetHsvHist, buffer, 1);
    }
    else if (histogramType == 0)
    {
        printf("\nCalculating RGB histograms...\n");
        rgbImageMatches = compareHistograms(dp, dirPath, targetImagePath, targetRgbHist, buffer, 0);
    }

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
    for (int i = 0; i < imageMatches.size(); i++)
    {
        printf("%s: %f\n", imageMatches[i].first.c_str(), imageMatches[i].second);
    }

    printf("Terminating\n\n");

    return 0;
}