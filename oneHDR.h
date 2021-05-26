#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

cv::Mat oneHDR(cv::Mat &inputImg, float mu = 0.5f, float a = -0.3293f, float b = 1.1258f);
float* tsmooth(float *I, int rows, float lambd = 0.5f, int sigma = 5, float sharpness = 0.001f);
void computeTextureWeights(float *I, int rows, int sigma, float sharpness, float *wx, float *wy);
float* solveLinearEquation(float *I, int rows, float *wx, float *wy, float lambd);
float* convertCol(float *wx, int rows, float lambd);
float* convertCol_delay_row(float *wx, int rows, float lambd);
float* convertCol_delay_col(float *wx, int rows, float lambd);
