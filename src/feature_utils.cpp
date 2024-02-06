// Author: Kevin Heleodoro
// Date: February 1, 2024
// Purpose: Contains utility functions for extracting feature vectors from images and computing distances between
// feature vectors.

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "feature_utils.h"

/**
 * @brief Extract a feature vector from an image
 *
 * @param imagePath The path to the image
 * @return std::vector<float> The feature vector
 */
std::vector<float> extractFeatureVector(const std::string &imagePath)
{
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        throw std::runtime_error("Could not read image: " + imagePath);
    }

    int centerX = image.cols / 2;
    int centerY = image.rows / 2;

    cv::Rect roi(centerX - 3, centerY - 3, 7, 7); // 7x7 feature vector
    cv::Mat croppedImage = image(roi).clone();    // By cloning we ensure that the cropped image is continuous in memory

    // Another way to convert the cropped image to a feature vector,
    // but it does not allow us to work with the floating point values directly.
    // std::vector<float> featureVector(croppedImage.begin<float>(), croppedImage.end<float>());

    std::vector<float> featureVector;
    croppedImage.reshape(1, 1).copyTo(featureVector);

    return featureVector;
}

/**
 * @brief Compute the Euclidean distance between two feature vectors
 *
 * @param vector1 The first feature vector
 * @param vector2 The second feature vector
 * @return float The Euclidean distance between the two feature vectors
 */
float computeDistance(const std::vector<float> &vector1, const std::vector<float> &vector2)
{
    float distance = 0.0;
    for (size_t i = 0; i < vector1.size(); i++)
    {
        float difference = vector1[i] - vector2[i];
        distance += difference * difference;
    }
    return distance;
}

// /**
//  * @brief A struct to hold the filename and distance of a matching image
//  *
//  * @param filename The filename of the matching image
//  * @param distance The distance of the matching image
//  */
// struct ImageMatch
// {
//     std::string filename;
//     float distance;
// };

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

std::vector<ImageMatch> findTopNMatches(const std::vector<float> &targetVector,
                                        const std::vector<std::pair<std::string, std::vector<float>>> &featureVectors,
                                        int topN = 3)
{
    std::vector<ImageMatch> matches;

    // printf("Finding top %d matches for target vector ...\n", topN);
    // for (const auto &[filename, vector] : featureVectors)
    for (const auto &pair : featureVectors)
    {
        const std::string &filename = pair.first;
        const std::vector<float> &vector = pair.second;
        printf("Processing image: %s\n", filename.c_str());
        float distance = computeDistance(targetVector, vector);
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
            matches.push_back({filename, distance});
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