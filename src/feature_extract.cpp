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

int main(int argc, char *argv[])
{
    if (argc < 1)
    {
        printf("Usage: ./feature_extract %s <image_directory>\n", argv[0]);
        exit(-1);
    }

    printf("\n\n========== Feature Extract ==========\n\n");

    char dirPath[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;

    strcpy(dirPath, argv[1]);
    printf("Image directory set to %s\n", dirPath);
    dirp = opendir(dirPath);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirPath);
        exit(-1);
    }

    printf("Creating feature_vectors directory...\n\n\n");
    std::string feature_vectors_dir = "feature_vectors";
    // std::__fs::filesystem::create_directory(feature_vectors_dir);
    // Create a .csv file
    std::string feature_vectors_csv = feature_vectors_dir + "/feature_vectors.csv";
    FILE *fp = fopen(feature_vectors_csv.c_str(), "w");

    // loop over all the files in the image file listing

    while ((dp = readdir(dirp)) != NULL)
    {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif"))
        {
            printf("processing image file: %s\n", dp->d_name);
            strcpy(buffer, dirPath);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            // printf("full path name: %s\n", buffer);

            std::vector<float> featureVector = extractFeatureVector(buffer);
            append_image_data_csv(feature_vectors_csv.c_str(), dp->d_name, featureVector, 0);
        }
    }

    printf("\n=====================================\n\n");
    printf("Completed feature extraction\n");
    printf("Terminating\n\n");

    return 0;
}