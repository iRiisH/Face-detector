#ifndef CLASSIFIER
#define CLASSIFIER

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>

#include "feature.h"

using namespace cv;
using namespace std;


#define K 10
#define EPSILON 0.1

#define TOTAL_IMGS 5233
#define POS_IMGS 818
#define NEG_IMGS 5233

#define VALIDATION_SIZE 300

class WeakClassifier
{
public:
	WeakClassifier();
	~WeakClassifier();
	int h(float x) const;
	void train_step(float x_ki, int c_k);
private:
	float w1, w2;
};

void pickRandomImage(Mat &img, int c_k);
void train();

#endif
