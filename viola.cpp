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
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	srand(time(NULL));
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
	//MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		cout << "printing array :" << endl;
		for (int j = 0; j < totalSize; j++)
			cout << new_tab[j] << endl;
	}
	MPI_Finalize();
	return 0;
}

