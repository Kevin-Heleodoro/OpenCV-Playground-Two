// Author: Kevin Heleodoro
// Date: February 1, 2024
// Purpose: Contains utility functions for extracting feature vectors from images and computing distances between
// feature vectors.

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
