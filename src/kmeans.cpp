/*
  EDITED BY:
  Kevin Heleodoro
  February 2,2024

  Edits include the addition of the main function to run kmeans.

  ====================================================================================================


  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Implementation of a K-means algorithm
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "kmeans.h"

/*
  data: a std::vector of pixels
  means: a std:vector of means, will contain the cluster means when the function returns
  labels: an allocated array of type int, the same size as the data, contains the labels when the function returns
  K: the number of clusters
  maxIterations: maximum number of E-M interactions, default is 10
  stopThresh: if the means change less than the threshold, the E-M loop terminates, default is 0

  Executes K-means clustering on the data
 */
int kmeans(std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations,
           int stopThresh)
{
    // error checking
    if (K > data.size())
    {
        printf("error: K must be less than the number of data points\n");
        return (-1);
    }

    // clear the means vector
    printf("Clearing means vector ...\n");
    means.clear();

    // initialize the K mean values
    // use comb sampling to select K values
    printf("Initializing K mean values ...\n");
    int delta = data.size() / K;
    printf("delta: %d\n", delta);

    // We need to account for the case where the result of data.size() % K is 0.
    int d_size = (data.size() % K) > 0 ? data.size() % K : 1;
    int istep = rand() % d_size;
    printf("istep: %d\n", istep);
    for (int i = 0; i < K; i++)
    {
        int index = (istep + i * delta) % data.size();
        means.push_back(data[index]);
    }
    // have K initial means

    // loop the E-M steps
    for (int i = 0; i < maxIterations; i++)
    {

        // classify each data point using SSD
        printf("\nClassifying each data point using SSD ...\n");
        for (int j = 0; j < data.size(); j++)
        {
            int minssd = SSD(means[0], data[j]);
            int minidx = 0;
            for (int k = 1; k < K; k++)
            {
                int tssd = SSD(means[k], data[j]);
                if (tssd < minssd)
                {
                    minssd = tssd;
                    minidx = k;
                }
            }
            labels[j] = minidx;
        }

        // calculate the new means
        printf("Calculating new means ...\n");
        std::vector<cv::Vec4i> tmeans(means.size(), cv::Vec4i(0, 0, 0, 0)); // initialize with zeros
        for (int j = 0; j < data.size(); j++)
        {
            tmeans[labels[j]][0] += data[j][0];
            tmeans[labels[j]][1] += data[j][1];
            tmeans[labels[j]][2] += data[j][2];
            tmeans[labels[j]][3]++; // counter
        }

        int sum = 0;
        printf("Updating means ...\n");
        for (int k = 0; k < tmeans.size(); k++)
        {
            int divisor = tmeans[k][3] > 0 ? tmeans[k][3] : 1;
            // tmeans[k][0] /= tmeans[k][3];
            // tmeans[k][1] /= tmeans[k][3];
            // tmeans[k][2] /= tmeans[k][3];
            tmeans[k][0] /= divisor;
            tmeans[k][1] /= divisor;
            tmeans[k][2] /= divisor;

            // compute the SSD between the new and old means
            sum += SSD(tmeans[k], means[k]);

            means[k][0] = tmeans[k][0]; // update the mean
            means[k][1] = tmeans[k][1]; // update the mean
            means[k][2] = tmeans[k][2]; // update the mean
        }

        // check if we can stop early
        printf("Iteration %d, sum: %d\n\n", i, sum);
        if (sum <= stopThresh)
        {
            break;
        }
    }

    // the labels and updated means are the final values

    return (0);
}

int main(int argc, char *argv[])
{
    if (argc < 1)
    {
        printf("Usage: %s <image filename> <# of colors>\n", argv[0]);
        exit(-1);
    }

    srand(time(NULL));

    printf("\n\n========== K-means Clustering ==========\n\n");
    char filename[256];

    strcpy(filename, argv[1]);
    printf("Image set to %s\n", filename);
    cv::Mat image = cv::imread(filename);
    cv::Mat original = image.clone();
    if (image.empty())
    {
        printf("No image data\n");
        return -1;
    }

    // kmeans clustering
    int K = atoi(argv[2]);
    printf("Number of colors set to %d\n", K);
    std::vector<cv::Vec3b> data;
    std::vector<cv::Vec3b> means;

    printf("Creating labels ...\n");
    int *labels = new int[image.rows * image.cols];

    printf("Extracting pixels from image ...\n");
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            data.push_back(image.at<cv::Vec3b>(i, j));
        }
    }

    try
    {
        int data_size = data.size();
        printf("Data size: %d\n", data_size);
        printf("Valid K: %d\n", (data_size % K));
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    int maxIterations = 10;
    int stopThresh = 0;
    printf("\n=============================\n\n");
    printf("Running kmeans ...\n");
    kmeans(data, means, labels, K, maxIterations, stopThresh);

    printf("Updating image with kmeans ...\n");
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int idx = labels[i * image.cols + j];
            image.at<cv::Vec3b>(i, j) = means[idx];
        }
    }

    printf("Presenting images ...\n");

    cv::imshow("Original", original);
    cv::imshow("K-means", image);
    cv::waitKey(0);
    cv::imwrite(filename + std::to_string(K) + "_kmeans.jpg", image);

    delete[] labels;

    return 0;
}