// Author: Kevin Heleodoro
// Date: February 13, 2024
// Purpose: Contains utility functions for creating a comparing histograms

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#ifndef HISTOGRAM_UTILS_H
#define HISTOGRAM_UTILS_H

/**
 * @brief Calculate the intersection of two histograms
 *
 * @param histA The first histogram
 * @param histB The second histogram
 * @return float The intersection of the two histograms
 */
float histIntersect(const cv::Mat &histA, const cv::Mat &histB);

cv::Mat calcColorHist(const cv::Mat &image, int bins);

cv::Mat calcTextureHist(const cv::Mat &image, int bins);

/**
 * @brief Calculate the histogram of an image
 *
 * @param hsvImage The HSV image
 * @param hBins The number of hue bins
 * @param sBins The number of saturation bins
 * @return cv::Mat The histogram
 */
cv::Mat calcHsvHist(const cv::Mat &hsvImage, int hBins, int sBins);

/**
 * @brief Calculate the RG Chromaticity histogram of an image
 *
 * @param image The RGB image
 * @param histSize The number of bins
 * @return cv::Mat The histogram
 */
cv::Mat calcRgbHist(const cv::Mat &image, int histSize);

/**
 * @brief Compare the histograms of images in a directory
 *
 * @param dp The directory pointer
 * @param dirPath The directory path
 * @param targetImagePath The path of the target image
 * @param targetHist The target histogram
 * @param buffer The buffer for the image path
 * @param histType The type of histogram to calculate
 * @return std::vector<std::pair<std::string, float>> The list of image matches
 */
std::vector<std::pair<std::string, float>> compareHistograms(struct dirent *dp, char *dirPath, char *targetImagePath,
                                                             cv::Mat targetHist, char *buffer, int histType);

/**
 * @brief Creates the display histogram
 *
 * @param hist The histogram
 * @param dst The destination image
 * @param histsize The size of the histogram
 * @return cv::Mat The destination image
 */
cv::Mat createDisplayHist(cv::Mat &hist, cv::Mat &dst, int histsize);

std::vector<std::pair<std::string, float>> compareDeepNetworkEmbedding(
    std::vector<std::pair<std::string, std::vector<float>>> resNetCsv, std::string targetImagePath, std::string buffer);

std::pair<std::string, std::vector<float>> extractTargetFeatureVectorFromFile(
    std::vector<std::pair<std::string, std::vector<float>>> csvFeatures, std::string targetImagePath);
#endif