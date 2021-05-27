
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

#include "oneHDR.h"
#include "SpSolve.h"

int main( int argc, char** argv )
{
	cv::Mat inputImg, outputImg;
	inputImg = cv::imread("..\\testing images\\input.jpg");
	int64 t = cv::getTickCount();
    // processing
	cv::Mat yuv(inputImg.rows*3/2, inputImg.cols, CV_8UC1);
	cv::cvtColor(inputImg, yuv, cv::COLOR_BGR2YUV_I420);

	cv::Rect rt(0,0, inputImg.cols, inputImg.rows);
	cv::Mat Y; Y = yuv(rt);
	/*
	cv::Mat outY = oneHDR(Y);
	yuv(rt) = outY;
	cv::cvtColor(yuv, outputImg, cv::COLOR_YUV2BGR_I420);
	*/

	outputImg = oneHDR(inputImg);
	double time = (cv::getTickCount() - t) / cv::getTickFrequency() * 1000;
	printf("time: %f ms\n", time / 10);
	// output
	cv::imshow("inputImg", inputImg);
	cv::imshow("outputImg", outputImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
