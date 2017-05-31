#ifndef CLASSIFIER
#define CLASSIFIER

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>
#include <math.h>

#include "feature.h"

using namespace cv;
using namespace std;


#define K 3000
#define EPSILON 0.0001

#define TOTAL_IMGS 5233
#define POS_IMGS 818
#define NEG_IMGS 4415

#define VALIDATION_SIZE 500

#define N 200 // 6000 are used in the paper
#define THETA 0.5

class WeakClassifier
{
public:
	WeakClassifier(float w_1, float w_2);
	WeakClassifier();
	~WeakClassifier();
	int h(float x) const;

	// performs a train step, which consists in updating w1 & w2 given a coordinate
	// of the original image, and the class associated with the image (1 or -1)
	void train_step(float x_ki, int c_k);
	float get_w1() const;
	float get_w2() const;
private:
	float w1, w2;
};

// returns the id of a randomly picked image
int pickRandomImage(int& c_k);

class WeakClassifierSet
{
public:
	WeakClassifierSet();
	~WeakClassifierSet();

	// trains the weak classifiers
	void train();
	// returns true if the image is classified as a face, false else
	bool testImg(const Mat& img) const;
	// returns the rate of good classification on a dataset of size VALIDATION_SIZE
	float testValid() const;
	// basically the same function as above, but on the whole validation set (slower)
	float testWholeValidationSet() const;

private:
	// it would be convenient to encode the list as a WeakClassifier array, but for
	// distribution purposes it is easier to manipulate float arrays.
	float* w1_list;
	float* w2_list;
	int nFeatures;
};

int E(int h, int c);
void weightedError(WeakClassifier* wc, float* lambda, float* errors, int nFeatures);
float alpha(float epsilon);
int minInd(float* error, int nFeatures);
void updateWeights(float alpha, WeakClassifier h_k, int ind, float* lambda, int nFeatures);
void adaboost(float* w1_list, float* w2_list, int nFeatures, vector<WeakClassifier>& result,
	vector<float>& alpha_list, vector<int>& indexes);

#endif
