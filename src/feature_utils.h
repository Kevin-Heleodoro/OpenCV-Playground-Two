// Author: Kevin Heleodoro
// Date: February 1, 2024
// Purpose: Contains utility functions for extracting feature vectors from images and computing distances between
// feature vectors.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#ifndef FEATURE_UTILS_H
#define FEATURE_UTILS_H

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
 * @brief Extract a feature vector from an image
 *
 * @param imagePath The path to the image
 * @return std::vector<float> The feature vector
 */
std::vector<float> extractFeatureVector(const std::string &imagePath);

/**
 * @brief Compute the Euclidean distance between two feature vectors
 *
 * @param vector1 The first feature vector
 * @param vector2 The second feature vector
 * @return float The Euclidean distance between the two feature vectors
 */
float computeDistance(const std::vector<float> &vector1, const std::vector<float> &vector2);

std::vector<ImageMatch> findTopNMatches(const std::string &targetImage, const std::string &imageDir, int topN);

std::vector<ImageMatch> findTopNMatches(const std::vector<float> &targetVector,
                                        const std::vector<std::pair<std::string, std::vector<float>>> &featureVectors,
                                        int topN);

#endif