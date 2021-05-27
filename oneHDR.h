#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#pragma warning(disable:4996)

using namespace std;

cv::Mat oneHDR(cv::Mat &inputImg);
cv::Mat get_enhance_params(cv::Mat &inputImg, float &gamma, float &beta);
int tsmooth(float *I, float *t_W, int rows, float lambd = 0.5f, int sigma = 5, float sharpness = 0.001f);
void computeTextureWeights(float *I, int rows, int sigma, float sharpness, float *wx, float *wy);
int solveLinearEquation(float *I, int rows, float *wx, float *wy, float lambd, float *t_W);
float* convertCol(float *wx, int rows, float lambd);
float* convertCol_delay_row(float *wx, int rows, float lambd);
float* convertCol_delay_col(float *wx, int rows, float lambd);
int maxEntropyEnhance(cv::Mat &I, cv::Mat &t, float &gamma, float &beta, float bad_threshold = 0.5f, float mink = 1.0f, float maxk = 10.0f, float a = -0.3293f, float b = 1.1258f);
float entropy(float *Y, int num);
float negative_entropy(float *Y, int num, float k, float a, float b);
int calc_timing(int64 srart_time, int64 end_time, const char *string);
