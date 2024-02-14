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

/**
 * @brief Calculate the histogram of an image
 *
 * @param hsvImage The HSV image
 * @param hBins The number of hue bins
 * @param sBins The number of saturation bins
 * @return cv::Mat The histogram
 */
cv::Mat calcHsvHist(const cv::Mat &hsvImage, int hBins, int sBins);

cv::Mat calcRgbHist(const cv::Mat &image, int histSize);

std::vector<std::pair<std::string, float>> compareHistograms(struct dirent *dp, char *dirPath, char *targetImagePath,
                                                             cv::Mat targetHist, char *buffer, int histType);

#endif