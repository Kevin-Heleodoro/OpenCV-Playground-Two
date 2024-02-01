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
 * @brief A struct to hold the filename and distance of a matching image
 *
 * @param filename The filename of the matching image
 * @param distance The distance of the matching image
 */
struct ImageMatch
{
    std::string filename;
    float distance;
};

/**
 * @brief Find the top N matches for a target image in a directory of images
 *
 * @param targetImage The target image to match
 * @param imageDir The directory of images to search
 * @param topN The number of top matches to return
 * @return std::vector<ImageMatch> A vector of ImageMatch structs containing the filename and distance of the top N
 * matches
 */
std::vector<ImageMatch> findTopNMatches(const std::string &targetImage, const std::string &imageDir, int topN = 3)
{
    printf("Extracting feature vector for target image ...\n");
    std::vector<float> targetVector = extractFeatureVector(targetImage);
    printf("Target vector size: %lu\n", targetVector.size());
    printf("Target distance to self: %f\n", computeDistance(targetVector, targetVector));
    std::vector<ImageMatch> matches;

    printf("Extracting feature vectors for directory images ...\n");
    for (const auto &entry : std::__fs::filesystem::directory_iterator(imageDir))
    {
        printf("Processing image: %s\n", entry.path().string().c_str());
        std::vector<float> featureVector = extractFeatureVector(entry.path().string());
        float distance = computeDistance(targetVector, featureVector);
        if (distance < 0.0)
        {
            printf("Error: distance is negative\n");
            continue;
        }
        else if (distance == 0.0)
        {
            continue;
        }
        else
        {
            matches.push_back({entry.path().string(), distance});
        }
    }

    printf("Found %lu matches\n", matches.size());
    printf("Sorting matches...\n");
    std::sort(matches.begin(), matches.end(),
              [](const ImageMatch &a, const ImageMatch &b) { return a.distance < b.distance; });

    if (matches.size() > topN)
    {
        matches.resize(topN);
    }

    return matches;
}

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
    if (argc < 2)
    {
        printf("Usage: %s <target_image> <image_directory> [topN]\n", argv[0]);
        exit(-1);
    }

    printf("\n\n========== Baseline Match ==========\n\n");

    char targetImagePath[256];
    char dirPath[256];
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

    strcpy(dirPath, argv[2]);
    printf("Image directory set to %s\n", dirPath);
    dirp = opendir(dirPath);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirPath);
        exit(-1);
    }

    if (argv[2] != NULL && argc > 3)
    {
        try
        {
            topN = std::stoi(argv[3]);
        }
        catch (std::invalid_argument &e)
        {
            printf("Invalid argument for topN: %s\n", argv[3]);
            exit(-1);
        }
    }

    printf("Finding Top %d Matches\n", topN);
    auto topMatches = findTopNMatches(targetImagePath, dirPath, topN);
    printf("\n================\n\n");

    printf("Top matches: \n");
    for (const auto &match : topMatches)
    {
        std::cout << "Image: " << match.filename << ", Distance: " << match.distance << std::endl;
    }
    printf("\n================\n\n");

    return 0;
}