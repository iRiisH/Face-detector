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

void imageIntegrale(const Mat& input, Mat& output);
void intToDimensions(int n, int &x, int &y, int &w, int &h);
void filltab(vector<float>& localresult, vector<float>& tab, vector<float>& result);
void calcLocalFeatures(Mat& ii, vector<float>& localResult);
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