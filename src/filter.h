// Author: Kevin Heleodoro
// Date: January 9, 2024
// Purpose: Display live video using OpenCV.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#ifndef FILTER_H
#define FILTER_H

/**
 * @brief Convert a color image to greyscale.
 *
 * This function converts a color image to greyscale. It does so by subtracting
 * the red value of each pixel from 255 and setting the red, green, and blue
 * values of each pixel to the resulting value.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Convert a color image to sepia tone.
 *
 * This function converts a color image to sepia tone. It does so by applying
 * sepia coefficients to each pixel.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int sepiaTone(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 5x5 Gaussian kernel.
 *
 * This function blurs a color image using a 5x5 Gaussian kernel. It does so by
 * applying the kernel to each pixel within a nested loop.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 1x5 Gaussian kernel.
 *
 * This function blurs a color image using a 1x5 Gaussian kernel. It does so by
 * applying separable 1x5 filters to each pixel in two passes (horizontal and veritcal).
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 5x5 Gaussian kernel.
 *
 * This function blurs a color image using a 5x5 Gaussian kernel. It does so by
 * applying the kernel to each pixel within a nested loop. This version of the function
 * uses the .at method to access the pixels. It adds the values of each pixel in the kernel
 * without looping through it.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int blur5x5_3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 5x5 Gaussian kernel.
 *
 * This function blurs a color image using a 5x5 Gaussian kernel. It does so by
 * applying the kernel to each pixel within a nested loop. This version of the function uses the .ptr method to access
 * pixels. It also does not loop through the kernel, but instead calculates the sum of each row and column of the kernel
 * separately.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 **/
int blur5x5_4(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 1x5 Gaussian kernel.
 *
 * This function blurs a color image using a 1x5 Gaussian kernel. It does so by
 * applying separable 1x5 filters to each pixel in two passes (horizontal and veritcal). This version of the function
 * uses the .ptr method to access pixels. It also does not loop through the kernel, but instead calculates the sum of
 * each row and column of the kernel separately.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int blur5x5_5(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 3x3 Gaussian kernel.
 *
 * This function blurs a color image using a 3x3 Gaussian kernel.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int gauss3x3at(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Enhance vertical lines in an image using a 3x3 Sobel kernel.
 *
 * This function enhances vertical lines in an image using a 3x3 Sobel kernel. It does so by
 * applying the kernel to each pixel within a nested loop.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Enhance horizontal lines in an image using a 3x3 Sobel kernel.
 *
 * This function enhances horizontal lines in an image using a 3x3 Sobel kernel. It does so by
 * applying the kernel to each pixel within a nested loop.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Calculate the gradient magnitude of an image.
 *
 * This function calculates the gradient magnitude of an image. It does so by
 * applying horizontal and vertical sobel filters to each pixel within a nested loop.
 *
 * @param sx The source image with a sobel x filter applied.
 * @param sy The source image with a sobel y filter applied.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * @brief Calculate the gradient magnitude of an image.
 *
 * This function calculates the gradient magnitude of an image. It does so by
 * applying horizontal and vertical sobel filters to each pixel within a nested loop. This version of the function uses
 * the .ptr method to access pixels. It also does not loop through the kernel, but instead calculates the sum of each
 * row and column of the kernel separately.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int magnitude(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Blur a color image using a 5x5 Gaussian kernel. Quantize the image to a specified number of levels.
 *
 * This function blurs a color image using a 5x5 Gaussian kernel. It then quantizes the image to a specified number of
 * levels. It does so by dividing the image into buckets and setting each pixel to the average value of the bucket.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @param levels The number of levels to quantize the image to.
 * @return 0 if successful, -1 if error.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/**
 * @brief Apply an emboss effect to an image.
 *
 * This function applies an emboss effect to an image. It does so by applying a 3x3 Sobel filter to the image and
 * adding 128 to the result. It then clamps the result to the range [0, 255].
 *
 * @param sx The source image with a sobel x filter applied.
 * @param sy The source image with a sobel y filter applied.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int embossEffect(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * @brief Adjust the brightness of an image.
 *
 * This function adjusts the brightness of an image. It does so by multiplying each pixel by a specified value.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @param brightness The brightness value to multiply each pixel by.
 * @return 0 if successful, -1 if error.
 */
int adjustBrightness(cv::Mat &src, cv::Mat &dst, double brightness);

/**
 * @brief Create a negative of an image
 *
 * This function creates a negative of an image. It does so by subtracting each pixel from 255.
 *
 * @param src The source image.
 * @param dst The destination image.
 * @return 0 if successful, -1 if error.
 */
int negativeFilter(cv::Mat &src, cv::Mat &dst);

#endif