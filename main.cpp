
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
