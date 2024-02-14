// Author: Kevin Heleodoro
// Date: February 13, 2024
// Purpose: Contains utility functions for creating a comparing histograms

#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "histogram_utils.h"

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
    float intersection = 0;
    for (int h = 0; h < histA.rows; h++)
    {
        for (int s = 0; s < histA.cols; s++)
        {
            // Intersection is the minimum value of the two histograms at each bin
            // printf("histA: %f, histB: %f\n", histA.at<float>(h, s), histB.at<float>(h, s));
            intersection += std::min(histA.at<float>(h, s), histB.at<float>(h, s));
            // printf("Intersection: %f\n", intersection);
        }
    }
    return intersection;
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

cv::Mat calcRgbHist(const cv::Mat &image, int histSize)
{
    printf("Calculating RGB chromaticity histogram ...\n");
    // Initialize histogram
    cv::Mat hist = cv::Mat::zeros(histSize, histSize, CV_32FC1);
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
            int gIndex = (int)(g * (histSize - 1) + 0.5);
            int rIndex = (int)(r * (histSize - 1) + 0.5);

            // Check if the indices are within the range of the histogram
            // if (gIndex >= 0 && gIndex < histSize && rIndex >= 0 && rIndex < histSize)
            // {
            // Increment histogram
            hist.at<float>(rIndex, gIndex)++;
            max = hist.at<float>(rIndex, gIndex) > max ? hist.at<float>(rIndex, gIndex) : max;
            // }
        }
    }

    printf("Normalizing histogram ...\n");
    printf("The largest bucket has %d pixels in it\n", (int)max);
    // cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    printf("hist size: %d x %d\n", hist.rows, hist.cols);
    hist /= (src.rows * src.cols);

    return hist;
}

std::vector<std::pair<std::string, float>> compareHistograms(struct dirent *dp, char *dirPath, char *targetImagePath,
                                                             cv::Mat targetHist, char *buffer, int histType)
{
    printf("\nProcessing images in directory ...\n");
    printf("Parameters: dirPath: %s, targetImagePath: %s, histType: %d\n", dirPath, targetImagePath, histType);
    std::vector<std::pair<std::string, float>> imageMatches;

    DIR *dirp = opendir(dirPath);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", "./sample_images");
        exit(-1);
    }

    printf("dirp: %p\n", dirp);
    while ((dp = readdir(dirp)) != NULL)
    {
        printf("Processing image file: %s\n", dp->d_name);
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
            if (histType == 1)
            {
                int hBins = 30;
                int sBins = 30;
                cv::cvtColor(src, srcHist, cv::COLOR_BGR2HSV);
                srcHist = calcHsvHist(srcHist, hBins, sBins);
            }
            else if (histType == 0)
            {
                int histSize = 30;
                cv::cvtColor(src, srcHist, cv::COLOR_BGR2RGB);
                srcHist = calcRgbHist(srcHist, histSize);
            }

            float distance = histIntersect(targetHist, srcHist);
            imageMatches.push_back(std::make_pair(dp->d_name, distance));
        }
    }

    printf("Closing directory ...\n");
    closedir(dirp);
    return imageMatches;
}
