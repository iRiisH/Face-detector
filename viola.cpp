#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "mpi.h"

using namespace cv;
using namespace std;

void imageIntegrale(const Mat& input, Mat& output)
{
	int m = input.rows, n = input.cols;
	output = Mat::zeros(m, n, CV_32F);
	Mat s = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == 0)
				s.at<float>(i, j) = input.at<float>(i, j);
			else
				s.at<float>(i, j) = s.at<float>(i - 1, j) + input.at<float>(i, j);
			if (j == 0)
				output.at<float>(i, j) = s.at<float>(i, j);
			else
				output.at<float>(i, j) = output.at<float>(i, j - 1) + s.at<float>(i, j);
		}
	}
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Finalize();
	return 0;
}