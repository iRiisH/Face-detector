#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mpi.h>

#include "feature.h"
#include "classifier.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	srand(time(NULL));

	image = imread("../im1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image.convertTo(image, CV_32F);
	Mat image;
	Mat ii;
	imageIntegrale(image, ii);
	vector<float> localFeature;
	vector<float> features;
	calcFeatures(ii, localFeature);
	cout << "Computed vector : " << features.size() << endl;

	MPI_Finalize();
	return 0;
}

