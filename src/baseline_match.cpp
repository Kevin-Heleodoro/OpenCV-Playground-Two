// Author: Kevin Heleodoro
// Date: January 31, 2024
// Purpose: Given a directory of images and feature set it writes the feature vector for each image to a file.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "csv_util.h"

int extract_features_from_image(char *filename)
{
    // Read in the image
    cv::Mat image = cv::imread(filename, 1);
    if (!image.data)
    {
        printf("No image data\n");
        return -1;
    }

    printf("Image size: %d x %d\n", image.rows, image.cols);
    return 0;
    // Extract the features
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }

    // Take in a directory path and feature set --- What is the feature set?
    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);
    // Open the directory
    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // For each image in the directory
    while ((dp = readdir(dirp)) != NULL)
    {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            printf("full path name: %s\n", buffer);
            extract_features_from_image(buffer);
        }
    }

    // Extract the features

    // Write the features to a file
    return 0;
}