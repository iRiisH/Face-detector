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

