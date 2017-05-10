#ifndef FEATURE
#define FEATURE

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

#define MAX_WIDTH 92
#define MAX_HEIGHT 112

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

#endif