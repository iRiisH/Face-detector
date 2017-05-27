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

void test1();
void test2();

int main(int argc, char **argv)
{
	// initialisation
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	srand(time(NULL));

	//test1();
	test2();


	MPI_Finalize();
	return 0;
}

void test1()
// this tests the shareComputation function, on a simple example
{
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	float *tab = new float[10];
	for (int i = 0; i < 10; i++)
		tab[i] = (float)i;
	vector<float> v;
	for (int i = rank; i < 10; i += size)
	{
		v.push_back(2 * i);
	}
	int localSize = v.size();
	float* localRes = vectorToArray<float>(v);
	int totalSize;
	MPI_Barrier(MPI_COMM_WORLD);
	float *new_tab = new float[10];
	shareComputation(localRes, localSize, new_tab, totalSize);
	if (rank == 0)
	{
		cout << "printing array :" << endl;
		for (int j = 0; j < totalSize; j++)
			cout << new_tab[j] << endl;
	}
	
}

void test2()
// this function tests the distributed features computation
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	Mat img;
	img = imread("../../im1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32F);
	Mat ii;
	imageIntegrale(img, ii);

	int nFeatures = calcNFeatures();
	float* result = new float[nFeatures];
	calcFeatures(ii, result, nFeatures);
	if (rank == PROC_MASTER)
	{
		cout << "Computed vector:" << endl;
		for (int i = 0; i < 20; i++)
		{
			if (i < nFeatures)
				cout << result[i] << endl;
		}
		cout << "..." << endl;
		cout << "[the vector has actually " << nFeatures << " coordinates]" << endl;
	}
	delete result;
}
