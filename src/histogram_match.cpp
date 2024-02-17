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
#include "filter.h"
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
        printf("Usage: %s <targetImage> [histogramType] \n", argv[0]);
        printf("Histogram type: \n0 for RG Chromaticity \n1 for HSV \n2 for RG Chromaticity & HSV \n3 for color & "
               "texture \n4 for Deep Network Embedding \n5 for CBIR\n");
        exit(-1);
    }

    printf("\n\n========== Histogram Match ==========\n\n");

    DIR *dirp;
    cv::Mat image;
    cv::Mat histImageOne;
    cv::Mat histImageTwo;
    cv::Mat targetHistOne;
    cv::Mat targetHistTwo;
    char targetImagePath[256];
    char vectorCsv[256];
    int histogramType = 1;
    const int hBins = 30;
    const int sBins = 30;
    const int histSize = 30;
    const int fullHistSize = 256;

    std::string resNetCsv = "./feature_vectors/ResNet18_olym.csv";
    std::vector<std::pair<std::string, std::vector<float>>> resNetVectors;

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

    if (histogramType < 0 || histogramType > 5)
    {
        printf("Invalid histogram type: %d\n", histogramType);
        return -1;
    }

    if (histogramType == 5)
    {
        // Extract DNN feature vector for target image
        printf("Using CBIR\n");
        printf("Extracting DNN feature vector for target image ...\n");
        resNetVectors = readFeatureVectorsFromCSV(resNetCsv);
        printf("=> Read %lu feature vectors\n", resNetVectors.size());
        // Extract Colors
        printf("Extracting color features for target image ...\n");
        cv::cvtColor(image, histImageOne, cv::COLOR_BGR2RGB);
        targetHistOne = calcColorHist(histImageOne, fullHistSize);
        printf("=> Target RGB histogram size: %d x %d\n", targetHistOne.rows, targetHistOne.cols);
        // Extract Texture
        printf("Extracting texture features for target image ...\n");
        magnitude(image, histImageTwo);
        histImageTwo.convertTo(histImageTwo, CV_32F, 1.0 / 255.0);
        targetHistTwo = calcTextureHist(histImageTwo, fullHistSize);
        printf("=> Target texture histogram size: %d x %d\n", targetHistTwo.rows, targetHistTwo.cols);
    }

    if (histogramType == 4)
    {
        // Use deep network embedding
        printf("Using deep network embedding\n");
        resNetVectors = readFeatureVectorsFromCSV(resNetCsv);
        printf("Read %lu feature vectors\n", resNetVectors.size());
    }

    if (histogramType == 3)
    {
        // convert image to RGB histogram
        printf("\nCreating RGB histogram with %d bins ...\n", fullHistSize);
        cv::cvtColor(image, histImageOne, cv::COLOR_BGR2RGB);
        targetHistOne = calcColorHist(histImageOne, fullHistSize);
        printf("Target RGB histogram size: %d x %d\n", targetHistOne.rows, targetHistOne.cols);

        // convert image to texture histogram
        printf("\nCreating texture histogram with %d bins ...\n", fullHistSize);
        magnitude(image, histImageTwo);
        histImageTwo.convertTo(histImageTwo, CV_32F, 1.0 / 255.0);
        targetHistTwo = calcTextureHist(histImageTwo, fullHistSize);
        printf("Target texture histogram size: %d x %d\n", targetHistTwo.rows, targetHistTwo.cols);
    }

    if (histogramType == 1 || histogramType == 2)
    {
        // convert image to HSV histogram
        printf("\nCreating HSV histogram with %d hue bins and %d saturation bins ...\n", hBins, sBins);
        cv::cvtColor(image, histImageOne, cv::COLOR_BGR2HSV);
        targetHistOne = calcHsvHist(histImageOne, hBins, sBins);
        printf("Target HSV histogram size: %d x %d\n", targetHistOne.rows, targetHistOne.cols);
    }

    if (histogramType == 0 || histogramType == 2)
    {
        // convert image to RG Chromaticity histogram
        printf("\nCreating RG chromaticity histogram with %d bins ...\n", histSize);
        cv::cvtColor(image, histImageTwo, cv::COLOR_BGR2RGB);
        targetHistTwo = calcRgbHist(histImageTwo, histSize);
        printf("Target RG Chromaticity histogram size: %d x %d\n", targetHistTwo.rows, targetHistTwo.cols);
    }

    std::vector<std::pair<std::string, float>> histImageOneMatches;
    std::vector<std::pair<std::string, float>> histImageTwoMatches;
    std::vector<std::pair<std::string, float>> dnnMatches;
    char buffer[256];
    char dirPath[256] = "./sample_images";
    struct dirent *dp;

    printf("\n");

    if (histogramType == 5)
    {
        printf("====================================\n");
        printf("\nCalculating Deep Network Embedding matches ...\n");
        dnnMatches = compareDeepNetworkEmbedding(resNetVectors, targetImagePath, buffer);
        printf("\nCalculating color matches ...\n");
        histImageOneMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistOne, buffer, 3);
        printf("\nCalculating texture matches ...\n");
        histImageTwoMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistTwo, buffer, 3);
    }
    if (histogramType == 4)
    {
        printf("====================================\n");
        printf("Calculating deep network embedding matches ...\n");
        histImageOneMatches = compareDeepNetworkEmbedding(resNetVectors, targetImagePath, buffer);
    }
    if (histogramType == 3)
    {
        printf("====================================\n");
        printf("Calculating color histograms ...\n");
        histImageOneMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistOne, buffer, 3);
        printf("====================================\n");
        printf("Calculating texture histograms ...\n");
        histImageTwoMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistTwo, buffer, 3);
    }
    if (histogramType == 2)
    {
        printf("====================================\n");
        printf("Calculating both HSV ...\n");
        histImageOneMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistOne, buffer, 1);
        printf("====================================\n");
        printf("Calculating RG Chromaticity ...\n");
        histImageTwoMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistTwo, buffer, 0);
    }
    if (histogramType == 1)
    {
        printf("====================================\n");
        printf("Calculating HSV histograms...\n");
        histImageOneMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistOne, buffer, 1);
    }
    if (histogramType == 0)
    {
        printf("====================================\n");
        printf("Calculating RG Chromaticity histograms...\n");
        histImageTwoMatches = compareHistograms(dp, dirPath, targetImagePath, targetHistTwo, buffer, 0);
    }

    std::vector<std::pair<std::string, float>> imageMatches;

    if (histogramType == 5)
    {
        printf("Combining Deep Network Embedding, Color & Texture matches...\n");
        for (int i = 0; i < histImageOneMatches.size(); i++)
        {
            std::string filename = histImageOneMatches[i].first;
            float distance = histImageOneMatches[i].second;
            distance += histImageTwoMatches[i].second;
            distance += dnnMatches[i].second;
            imageMatches.push_back(std::make_pair(filename, distance));
        }
    }
    if (histogramType == 4)
    {
        imageMatches = histImageOneMatches;
    }
    if (histogramType == 3)
    {
        printf("Combining Color & Texture matches...\n");
        for (int i = 0; i < histImageOneMatches.size(); i++)
        {
            std::string filename = histImageOneMatches[i].first;
            float distance = histImageOneMatches[i].second;
            distance += histImageTwoMatches[i].second;
            imageMatches.push_back(std::make_pair(filename, distance));
        }
    }
    if (histogramType == 2)
    {
        printf("Combining RG Chromaticity & HSV matches...\n");
        for (int i = 0; i < histImageOneMatches.size(); i++)
        {
            std::string filename = histImageOneMatches[i].first;
            float distance = histImageOneMatches[i].second;
            distance += histImageTwoMatches[i].second;
            imageMatches.push_back(std::make_pair(filename, distance));
        }
    }
    if (histogramType == 1)
    {
        imageMatches = histImageOneMatches;
    }
    if (histogramType == 0)
    {
        imageMatches = histImageTwoMatches;
    }

    printf("Sorting matches...\n");
    std::sort(imageMatches.begin(), imageMatches.end(),
              [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
                  return a.second > b.second;
              });

    printf("\n=====================================\n\n");
    printf("Top 3 matches for %s:\n", targetImagePath);
    cv::Mat original, one, two, three;
    std::vector<cv::Mat> mats = {one, two, three};

    for (int i = 0; i < 5; i++)
    {
        printf("%s: %f\n", imageMatches[i].first.c_str(), imageMatches[i].second);
        createDisplayHist(targetHistOne, histImageOne, hBins);
    }

    printf("Terminating\n\n");

    return 0;
}