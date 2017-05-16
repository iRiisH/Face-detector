#include "classifier.h"


WeakClassifier::WeakClassifier()
{
	w1 = 1;
	w2 = 0;
}

WeakClassifier::~WeakClassifier()
{
}
int WeakClassifier::h(float x) const
{
	return (w1 * x + w2 >= 0) ? 1 : -1;
}

void WeakClassifier::train_step(float x_ki, int c_k)
{
	w1 = w1 - EPSILON * (h(x_ki) - c_k) * x_ki;
	w2 = w2 - EPSILON * (h(x_ki) - c_k);
}

void pickRandomImage(Mat &img, int c_k)
{
	int n = rand() % TOTAL_IMGS;
	if (n < POS_IMGS)
		img = imread("../app/pos/im"+to_string (n)+".jpg", CV_LOAD_IMAGE_GRAYSCALE);
	else
		img = imread("../app/neg/im" + to_string(n-POS_IMGS) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32F);
}

void initClassifier(WeakClassifier *wc, int &nFeatures)
{
}

void train (WeakClassifier* wc, int nFeatures)
{
	for (int i = 0; i < K; i++)
	{
		Mat img, ii;
		int c_k;
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		float* featuresVec;
		if (rank == PROC_MASTER)
		{
			pickRandomImage(img, c_k);
			imageIntegrale(img, ii);
			int nF;
			calcFeatures(ii, featuresVec, nF); // this function is already distributed
			// there is no need to make several processors compute these features
		}
		MPI_Bcast(featuresVec, nFeatures, MPI_FLOAT, PROC_MASTER, MPI_COMM_WORLD);
		
		for (int j = rank; j < nFeatures; j+= size)
		{
			wc[i].train_step(featuresVec[j], c_k);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

bool testImg(WeakClassifier* wc, int nFeatures, const Mat& img)
{
	Mat ii;
	imageIntegrale(img, ii);
	float *featuresVec = new float[nFeatures];
	int nF;
	calcFeatures(ii, featuresVec, nF);
	int score = 0;
	for (int i = 0; i < nFeatures; i++)
	{
		score += wc[i].h(featuresVec[i]);
	}
	delete featuresVec;
	return (score >= 0);
}

bool testValid()
{
	return true;
}
