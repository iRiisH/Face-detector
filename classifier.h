#ifndef CLASSIFIER
#define CLASSIFIER

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>

#include "feature.h"

using namespace cv;
using namespace std;


#define K 300
#define EPSILON 0.5

#define TOTAL_IMGS 5233
#define POS_IMGS 818
#define NEG_IMGS 4415

#define VALIDATION_SIZE 300

class WeakClassifier
{
public:
	WeakClassifier(float w_1, float w_2);
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

	void train();
	bool testImg(const Mat& img) const;
	float testValid() const;

private:
	// it would be convenient to encode the list as a WeakClassifier array, but for
	// distribution purposes it is easier to manipulate float arrays.
	float* w1_list;
	float* w2_list;
	int nFeatures;
};

#endif
