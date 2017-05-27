#ifndef FEATURE
#define FEATURE

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>

using namespace cv;
using namespace std; 

#define PROC_MASTER 0

#define MAX_WIDTH 112
#define MAX_HEIGHT 92

// computes the integral image
void imageIntegrale(const Mat& input, Mat& output);

// the distribution of the features' computation is done along 4 coordinates. This function
// is a bijection between all these coordinates and a segment to simplify the distribution
void intToDimensions(int n, int &x, int &y, int &w, int &h);

// usually the computations are distributed according to that pattern:
// for (int i = rank ; i < totalSize ; i++)
//		localResult.push_back (computation (i)) ;
// this function is used to merge the local results, putting them in the original order,
// so that the array computed does not depend on the number of processors used
void shareComputation(float *localArray, int localSize, float *result, int& totalSize);

// computes the number of features associated with a 112x92 image
int calcNFeatures();

// used to distribute the computation of features
void calcLocalFeatures(Mat& ii, vector<float>& localResult);

// merges the local features previously computed
void calcFeatures(Mat& ii, float *finalResult, int& nFeatures);

enum FeatureType {
	doubleH1 = 0,
	doubleH2 = 1,
	doubleV1 = 2,
	doubleV2 = 3,
	tripleH = 4,
	tripleV = 5,
	quadrupleH = 6,
	quadrupleV = 7
};

// the value of the sum of the pixels contained in a rectangle in a constant time,
//  using Viola's method
float rectangleVal(const Mat& integralImage, int xmin, int ymin, int xmax, int ymax);

class Feature
{
private:
	FeatureType type;
	int rectangleWidth, rectangleHeight;
	int origX, origY;
public :
	Feature(FeatureType type, int rectangleW, int rectangleH, int oX, int oY);
	~Feature();
	bool fits() const;
	
	float val(const Mat& integralImage) const;
};

template<typename T>
T* vectorToArray (vector<T> v)
// don't forget to delete the array after use
{
	T* result = new T[v.size()];
	for (int i = 0; i < v.size(); i++)
		result[i] = v[i];
	return result;
}


#endif