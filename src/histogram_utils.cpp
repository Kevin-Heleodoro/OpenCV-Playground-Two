// Author: Kevin Heleodoro
// Date: February 13, 2024
// Purpose: Contains utility functions for creating a comparing histograms

#include <dirent.h>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "filter.h"
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
    CV_Assert(histA.size == histB.size);
    float intersection = 0;
    for (int h = 0; h < histA.rows; h++)
    {
        for (int s = 0; s < histA.cols; s++)
        {
            // Intersection is the minimum value of the two histograms at each bin
            intersection += std::min(histA.at<float>(h, s), histB.at<float>(h, s));
        }
    }
    return intersection;
}

/**
 * @brief Calculate the color histogram of an image
 *
 * @param image The RGB image
 * @param bins The number of bins
 * @return cv::Mat The histogram
 */
cv::Mat calcColorHist(const cv::Mat &image, int bins)
{
    cv::Mat hist = cv::Mat::zeros(bins, bins, CV_32F);
    cv::Mat src;
    image.copyTo(src);

    for (int i = 0; i < src.rows; i++)
    {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++)
        {
            float blue = ptr[j][0];
            float green = ptr[j][1];
            float red = ptr[j][2];

            int bIndex = (int)(blue * (bins - 1) / 255.0);
            int gIndex = (int)(green * (bins - 1) / 255.0);
            int rIndex = (int)(red * (bins - 1) / 255.0);

            hist.at<float>(bIndex, gIndex)++;
            hist.at<float>(gIndex, rIndex)++;
            hist.at<float>(rIndex, bIndex)++;
        }
    }

    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    return hist;
}

/**
 * @brief Calculate the texture histogram of an image
 *
 * @param image The image
 * @param bins The number of bins
 * @return cv::Mat The histogram
 */
cv::Mat calcTextureHist(const cv::Mat &image, int bins)
{
    cv::Mat hist = cv::Mat::zeros(bins, bins, CV_32F);
    cv::Mat src;
    image.copyTo(src);

    double min, max;
    cv::minMaxLoc(src, &min, &max);

    float binWidth = (float)(max - min) / bins;

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            float val = src.at<float>(i, j);
            int index = (int)((val - min) / binWidth);
            hist.at<float>(index) += 1.0f;
        }
    }

    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    return hist;
}

/**
 * @brief Calculate the HSV histogram of an image
 *
 * @param hsvImage The HSV image
 * @param hBins The number of hue bins
 * @param sBins The number of saturation bins
 * @return cv::Mat The histogram
 */
cv::Mat calcHsvHist(const cv::Mat &hsvImage, int hBins, int sBins)
{
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

/**
 * @brief Calculate the RG Chromaticity histogram of an image
 *
 * @param image The RGB image
 * @param histSize The number of bins
 * @return cv::Mat The histogram
 */
cv::Mat calcRgbHist(const cv::Mat &image, int histSize)
{
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

            hist.at<float>(rIndex, gIndex)++;
            max = hist.at<float>(rIndex, gIndex) > max ? hist.at<float>(rIndex, gIndex) : max;
            // }
        }
    }

    // printf("The largest bucket has %d pixels in it\n", (int)max);
    hist /= (src.rows * src.cols);

    return hist;
}

/**
 * @brief Calculate the cosine distance between two feature vectors
 *
 * @param v1 The first feature vector
 * @param v2 The second feature vector
 */
float cosineDistance(const std::vector<float> &v1, const std::vector<float> &v2)
{
    float dotProduct = 0.0;
    float mag1 = 0.0;
    float mag2 = 0.0;
    for (int i = 0; i < v1.size(); ++i)
    {
        dotProduct += v1[i] * v2[i];
        mag1 += v1[i] * v1[i];
        mag2 += v2[i] * v2[i];
    }
    return 1 - dotProduct / (sqrt(mag1) * sqrt(mag2));
}

/**
 * @brief Extract the target feature vector from the CSV file
 *
 * @param csvFeatures The CSV feature vectors
 * @param targetImagePath The path of the target image
 * @return std::pair<std::string, std::vector<float>> The target feature vector
 *
 */
std::pair<std::string, std::vector<float>> extractTargetFeatureVectorFromFile(
    std::vector<std::pair<std::string, std::vector<float>>> csvFeatures, std::string targetImagePath)
{
    std::vector<float> targetVector;
    std::__fs::filesystem::path pathObj(targetImagePath);
    std::string targetImage = pathObj.filename().string();

    for (int i = 0; i < csvFeatures.size(); i++)
    {
        if (csvFeatures[i].first == targetImage)
        {
            targetVector = csvFeatures[i].second;
            break;
        }
    }

    return std::make_pair(targetImage, targetVector);
}

/**
 * @brief Compare the deep network embeddings of images in a directory
 *
 * @param resNetCsv The ResNet CSV file
 * @param targetImagePath The path of the target image
 * @param buffer The buffer for the image path
 * @return std::vector<std::pair<std::string, float>> The list of image matches
 */
std::vector<std::pair<std::string, float>> compareDeepNetworkEmbedding(
    std::vector<std::pair<std::string, std::vector<float>>> resNetCsv, std::string targetImagePath, std::string buffer)
{
    std::vector<std::pair<std::string, float>> imageMatches;

    std::pair<std::string, std::vector<float>> targetFeatureVector =
        extractTargetFeatureVectorFromFile(resNetCsv, targetImagePath);

    for (const auto features : resNetCsv)
    {
        if (features.first == targetFeatureVector.first)
        {
            continue;
        }

        float distance = cosineDistance(targetFeatureVector.second, features.second);

        imageMatches.push_back(std::make_pair(features.first, distance));
    }

    return imageMatches;
}

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
                                                             cv::Mat targetHist, char *buffer, int histType)
{
    printf("\nProcessing images in directory ...");
    std::vector<std::pair<std::string, float>> imageMatches;

    DIR *dirp = opendir(dirPath);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", "./sample_images");
        exit(-1);
    }
    int count = 0;

    // printf("Reading directory ...\n");
    while ((dp = readdir(dirp)) != NULL)
    {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            // printf(".", dp->d_name);
            if (count % 10 == 0)
            {
                printf(".");
            }
            count++;
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

            if (histType == 3)
            {
                int histSize = 256;
                magnitude(src, srcHist);
                srcHist.convertTo(srcHist, CV_32F, 1.0 / 255.0);
                srcHist = calcTextureHist(srcHist, histSize);
                cv::cvtColor(src, srcHist, cv::COLOR_BGR2RGB);
                srcHist = calcColorHist(srcHist, histSize);
            }
            else if (histType == 2)
            {
                int histSize = 256;
                cv::cvtColor(src, srcHist, cv::COLOR_BGR2RGB);
                srcHist = calcColorHist(srcHist, histSize);
            }
            else if (histType == 1)
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

    printf("Processed %d images\n", count);
    printf("Closing directory ...\n");
    closedir(dirp);
    return imageMatches;
}

/**
 * @brief Creates the display histogram
 *
 * @param hist The histogram
 * @param dst The destination image
 * @param histsize The size of the histogram
 * @return cv::Mat The destination image
 */
cv::Mat createDisplayHist(cv::Mat &hist, cv::Mat &dst, int histsize)
{
    dst.create(hist.size(), CV_8UC3);
    for (int i = 0; i < hist.rows; i++)
    {
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
        float *hptr = hist.ptr<float>(i);
        for (int j = 0; j < hist.cols; j++)
        {
            if (i + j > hist.rows)
            {
                ptr[j] = cv::Vec3b(200, 120, 60); // default color
                continue;
            }

            float rcolor = (float)i / histsize;
            float gcolor = (float)j / histsize;
            float bcolor = 1 - (rcolor + gcolor);

            ptr[j][0] = hptr[j] > 0 ? hptr[j] * 128 + 128 * bcolor : 0;
            ptr[j][1] = hptr[j] > 0 ? hptr[j] * 128 + 128 * gcolor : 0;
            ptr[j][2] = hptr[j] > 0 ? hptr[j] * 128 + 128 * rcolor : 0;
        }
    }
    return dst;
}