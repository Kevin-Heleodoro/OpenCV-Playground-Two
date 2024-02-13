// Author: Kevin Heleodoro
// Date: January 31, 2024
// Purpose: Given a directory of images and feature set it writes the feature vector for each image to a file.

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
 * @brief Main function to find the top N matches for a target image in a directory of images
 *
 * This function takes a target image and a directory of images and finds the top N matches for the target image in the
 * directory of images. The target image and image directory are specified as command line arguments. The top N matches
 * can also be specified as a command line argument. If no top N is specified, the default value of 3 is used.
 *
 * @param argc The number of command line arguments
 * @param argv The command line arguments
 * @return int The exit status
 */
int main(int argc, char *argv[])
{
    if (argc < 1)
    {
        printf("Usage: %s <targetImage> [topN] [vectorCsvFile] \n", argv[0]);
        exit(-1);
    }

    printf("\n\n========== Baseline Match v2.0 ==========\n\n");

    char targetImagePath[256];
    char vectorCsv[256];
    int topN = 3;
    DIR *dirp;

    strcpy(targetImagePath, argv[1]);
    printf("Target image set to %s\n", targetImagePath);
    cv::Mat image = cv::imread(targetImagePath);
    if (!image.data)
    {
        printf("No image data\n");
        return -1;
    }

    printf("Extracting feature vector for target image ...\n");
    std::vector<float> targetVector = extractFeatureVector(targetImagePath);

    if (argc < 3)
    {
        printf("Using default feature vector file: feature_vectors/feature_vectors.csv\n");
        strcpy(vectorCsv, "feature_vectors/feature_vectors.csv");
    }
    else
    {
        printf("Using feature vector file: %s\n", argv[3]);
        strcpy(vectorCsv, argv[3]);
    }

    FILE *fp = fopen(vectorCsv, "r");
    if (fp == NULL)
    {
        printf("Cannot open feature vector file %s\n", vectorCsv);
        exit(-1);
    }

    printf("Reading feature vectors from file...\n");
    std::vector<std::pair<std::string, std::vector<float>>> featureVectors;
    try
    {
        featureVectors = readFeatureVectorsFromCSV(vectorCsv);
        printf("Read %lu feature vectors\n", featureVectors.size());
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    if (argv[2] != NULL && argc > 2)
    {
        try
        {
            printf("Using topN: %s\n", argv[2]);
            topN = std::stoi(argv[2]);
        }
        catch (std::invalid_argument &e)
        {
            printf("Invalid argument for topN: %s\n", argv[3]);
            exit(-1);
        }
    }
    else
    {
        printf("Using default topN: %d\n", topN);
    }

    printf("Finding Top %d Matches\n", topN);
    auto topMatches = findTopNMatches(targetVector, featureVectors, topN);
    printf("\n================\n\n");

    printf("Top matches: \n");
    for (const auto &match : topMatches)
    {
        std::cout << "Image: " << match.filename << ", Distance: " << match.distance << std::endl;
    }
    printf("\n================\n\n");

    return 0;
}