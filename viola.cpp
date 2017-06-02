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

// this tests the shareComputation function, on a simple example
void test1();
// this function tests the distributed features computation
void test2();
// trains and tests the weak classifiers on the validation set
// see classifier.h to set the hyperparameters
void test3();
// tests the file manipulation functions
void test4();
// trains the adaboost classifier and tests it
void test5();

bool classify(Mat& img, WeakClassifier* wc, int* indexes, float* alpha, int nFeatures);
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
	//test3();
	//test4();
	test5();

	// closing
	MPI_Finalize();
	return 0;
}

void test1()
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
		cout << "saving..." << endl;
		//cout << "Testing..." << endl;
	}
	// notice that the obtained score corresponds to the ratio of faces in the databases,
	// which indicates that most weak classifiers are meaningless.
	//float score = wcs.testWholeValidationSet();
	if (rank == PROC_MASTER)
	{
		wcs.save("save.txt");
		/*score *= 100.;
		cout << endl << "Testing complete" << endl;
		cout << "Score: " << score << "%" << endl;*/
	}
}

void test4()
{
	int rank, i = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == PROC_MASTER)
	{
		float val[3];
		int ind[3];
		read_line("test_file.txt", i, val, ind, 3);
		cout << "Line " << i << " of test_file.txt contains:" << endl;
		for (int i = 0; i < 3; i++)
			cout << val[i] << " at pos " << ind[i] << endl;

		float column[3] = { 1.1, 1.2, 1.3 };
		int index[3] = { 1,4,9 };
		cout << "Adding a new column..." << endl;
		appendColumn("test_file.txt", column, index);
		
		float val2[4] = { 1.1, 1.1, 1.1, 1.1 }; 
		int ind2[4] = { 1, 2, 3, 4 };
		rewrite_line("test_file.txt", 1, val2, ind2, 4);
	}
}

void test5()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int nFeatures = calcNFeatures();

	WeakClassifierSet wcs;
	/*if (rank == PROC_MASTER)
		cout << "Training..." << endl;
	wcs.train();
	MPI_Barrier(MPI_COMM_WORLD);*/

	if (rank == PROC_MASTER)
	{
		wcs.load("save.txt");
		cout << "Training complete" << endl;
		cout << "Boosting..." << endl;
	}
	MPI_Bcast(wcs.get_w1(), nFeatures, MPI_FLOAT, PROC_MASTER, MPI_COMM_WORLD);
	MPI_Bcast(wcs.get_w2(), nFeatures, MPI_FLOAT, PROC_MASTER, MPI_COMM_WORLD);
	vector<WeakClassifier> result;
	vector<float> alpha;
	vector<int> indexes;
	adaboost(wcs.get_w1 (), wcs.get_w2 (), nFeatures, result, alpha, indexes);
	cout << "Boosting finished, " << result.size() << " features." << endl;
}

bool classify(Mat& img, WeakClassifier* wc, int* indexes, float* alpha, int nFeatures)
{
	int cascade_size = 5;
	int cascade[5] = { 1, 10, 25, 50, N };
	assert(img.cols == 92 && img.rows == 112);
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
