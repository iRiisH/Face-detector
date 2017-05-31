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
void test3();
bool classify(Mat& img, WeakClassifier* wc, float* alpha, int nFeatures);
void detector(Mat& img, WeakClassifier* wc, float* alpha);

int main(int argc, char **argv)
{
	// initialisation
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	srand(time(NULL));

	//test1();
	//test2();
	test3();

	// closing
	MPI_Finalize();
	return 0;
}

void test1()
// this tests the shareComputation function, on a simple example
{
	int test_size = 15;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	float *tab = new float[test_size];
	for (int i = 0; i < test_size; i++)
		tab[i] = (float)i;
	vector<float> v;
	for (int i = rank; i < test_size; i += size)
	{
		v.push_back(2 * i);
	}
	int localSize = v.size();
	float* localRes = vectorToArray<float>(v);
	int totalSize;
	MPI_Barrier(MPI_COMM_WORLD);
	float *new_tab = new float[test_size];
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
	cout << "computing " << nFeatures << " features..." << endl;
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
	delete[] result;
}

void test3()
// trains and tests the weak classifiers on the validation set
// see classifier.h to set the hyperparameters
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	WeakClassifierSet wcs;
	if (rank == PROC_MASTER)
		cout << "Training..." << endl;
	wcs.train();
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == PROC_MASTER)
	{
		cout << "Training complete" << endl;
		cout << "Testing..." << endl;
	}
	// notice that the obtained score corresponds to the ratio of faces in the databases,
	// which indicates that most weak classifiers are meaningless.
	float score = wcs.testWholeValidationSet();
	if (rank == PROC_MASTER)
	{
		score *= 100.;
		cout << endl << "Testing complete" << endl;
		cout << "Score: " << score << "%" << endl;
	}
}

void test4()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	WeakClassifierSet wcs;
	if (rank == PROC_MASTER)
		cout << "Training..." << endl;
	wcs.train();
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == PROC_MASTER)
	{
		cout << "Training complete" << endl;
		cout << "Boosting..." << endl;
	}
	adaboost();
}

bool classify(Mat& img, WeakClassifier* wc, int* indexes, float* alpha, int nFeatures)
{
	int cascade_size = 5;
	int cascade[5] = { 1, 10, 25, 50, N };
	assert(img.cols == 92 && img.rows == 112);
	assert(wc.size() == N);
	Mat ii;
	imageIntegrale(img, ii);
	float* features = new float[nFeatures];
	calcFeatures(ii, features, nFeatures);

	for (int i = 0; i < cascade_size; i++)
	{
		float sum1 = 0., sum2 = 0.;
		for (int j = 0; j < cascade[i]; j++)
		{
			sum1 += alpha[j] * wc[j].h(features[indexes[j]]);
			sum2 += alpha[j];
		}
		if (sum1 < THETA*sum2)
			return false;
	}
	return true;
}

void detector(Mat& img, WeakClassifier* wc, float* alpha)
{
	const float ratio = 112. / 92.;
	for (int w = 20; w < img.cols; w += 6)
	{
		int h = (int)(ratio*w);

	}
}
