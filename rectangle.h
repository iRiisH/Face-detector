#ifndef FEATURE
#define FEATURE

#define MAX_WIDTH 92
#define MAX_HEIGHT 112

enum FeatureType {
	doubleH1,
	doubleH2,
	doubleV1,
	doubleV2,
	tripleH,
	tripleV,
	quadrupleH,
	quadrupleV
};

float rectangleVal(const Mat& integralImage, int xmin, int ymin, int xmax, int ymax) const;

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