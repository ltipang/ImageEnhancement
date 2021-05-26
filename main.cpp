
#include "opencv2/opencv.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;

#include "oneHDR.h"

int main( int argc, char** argv )
{
	cv::Mat inputImg, outputImg;
	inputImg = cv::imread("images\\input.jpg");
    printf("Stitching ... \n");
	int64 t = getTickCount();
    // processing
	outputImg = oneHDR(inputImg);
	double time = (getTickCount() - t) / getTickFrequency() * 1000;
	//printf("time: %f\n", time / 10);
	// output
	/*cv::imshow("inputImg", inputImg);
	cv::imshow("outputImg", outputImg);
	cv::waitKey(0);
	cv::destroyAllWindows();*/
	return 0;
}
