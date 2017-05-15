#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mpi.h>

#include "feature.h"

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

void intToDimensions(int n, int &x, int &y, int &w, int &h)
{
	// 112 / 4 = 28, 92 / 4 = 23
	// n = h + 23 * w + 23 * 28 * y + 23 * 23 * 28 * x
	h = 4 * (n % 23);
	int reste = (n - (n % 23)) / 23;
	w = 4 * (reste % 28);
	reste = (reste - (reste % 28)) / 28;
	y = 4 * (reste % 23);
	reste = (reste - (reste % 23)) / 23;
	x = 4 * reste;
}


void filltab(vector<float>& localresult, vector<float>& tab, vector<float>& result){
	for (int i=0; i<localresult.size();i++){
		if (rank>1){
		result[tab[rank-2]+i]=localresult[i]; }//see if rank -1 ou -2
		else {
			result[i]=localresult[i]; }
		}
	
}

void calcFeatures(Mat& ii, vector<float>& result)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	for (int i = rank; i < 23 + 23 * 28 + 23 * 23 * 28 + 23 * 23 * 28 * 28; i+=size)
	{
		int x, y, w, h;
		intToDimensions(i, x, y, w, h);
		for (int type = 0; type <= 7; type++)
		{
			FeatureType ftype = (FeatureType)type;
			Feature f(ftype, w, h, x, y);

			if (!f.fits() || w < 8 || h < 8)
				continue;
			result.push_back(f.val(ii));
		}
	}
	tab[rank]=result.size();
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	srand(time(NULL));


	
	int rank, size;
	
	srand(time(NULL));
	MPI_Init(&argc, &argv);


	int tab[size];
	
	Mat image;
	image = imread("../im1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image.convertTo(image, CV_32F);
	//pas besoin de broadcast ou scatter en fait je crois

	Mat ii;
	imageIntegrale(image, ii);
	vector<float> localFeature;
	vector<float> features;
	calcFeatures(ii, localFeature);
	cout << "Computed vector : " << features.size() << endl;

	if (rank==0) {
		for (int i=1; i<size;i++) {
			tab[i]=tab[i]+tab[i-1];}
			}


	MPI_Bcast(tab, size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(result, tab[size-1], MPI_INT, 0, MPI_COMM_WORLD);

	filltab(localFeature, tab, features);

	MPI_Finalize();
	return 0;
}

