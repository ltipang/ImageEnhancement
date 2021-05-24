
#include "opencv2/opencv.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
	cv::Mat image1, image2, image3;
	cv::Mat image0_1, image0_2, image1_1, image1_3;

	

	cv::Mat out;
    printf("Stitching ... \n");
	int64 t = getTickCount();
   // processing

	double time = (getTickCount() - t) / getTickFrequency() * 1000;
	printf("time: %f\n", time / 10);

	cv::imwrite("result.jpg", out);

    return 0;
}
