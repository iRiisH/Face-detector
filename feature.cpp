#include "feature.h"


FeatureType type;
int rectangleWidth, rectangleHeight;
int origX, origY;

Feature::Feature (FeatureType t, int rectangleW, int rectangleH, int oX, int oY) :
	type (t), rectangleWidth (rectangleW), rectangleHeight (rectangleH),
	origX (oX), origY (oY) 
{
}

Feature::~Feature()
{ }

bool Feature::fits() const
{
	int horizontalFact (0), verticalFact (0);
	switch (type)
	{
	case doubleH1:
	case doubleH2:
		horizontalFact = 1;
		verticalFact = 2;
		break;
	case doubleV1:
	case doubleV2:
		horizontalFact = 2;
		verticalFact = 1;
		break;
	case tripleH:
		horizontalFact = 3;
		verticalFact = 1;
		break;
	case tripleV:
		horizontalFact = 1;
		verticalFact = 3;
		break;
	case quadrupleH:
	case quadrupleV:
		horizontalFact = 2;
		verticalFact = 2;
		break;
	}
	int width = horizontalFact * rectangleWidth;
	int height = verticalFact * rectangleHeight;
	return (origX >= 0 && origY >= 0)
		&& (origX + width < MAX_WIDTH && origY + height < MAX_HEIGHT);
}

float rectangleVal(const Mat& integralImage, int xmin, int ymin, int xmax, int ymax)
{
	float a = integralImage.at<float>(ymin, xmin);
	float b = integralImage.at<float>(ymin, xmax);
	float c = integralImage.at<float>(ymax, xmin);
	float d = integralImage.at<float>(ymax, xmax);
	return d + a - b - c;
}

float Feature::val(const Mat& integralImage) const
// the user must check the feature fits prior to this computation
{
	float res = 0., vPos, vNeg;
	switch (type)
	{
	case doubleH1:
		vNeg = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vPos = rectangleVal(integralImage, origX, origY + rectangleHeight,
			origX + rectangleWidth, origY + 2 * rectangleHeight);
		res = vPos - vNeg;
		break;
	case doubleH2:
		vPos = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight); 
		vNeg = rectangleVal(integralImage, origX, origY + rectangleHeight,
			origX + rectangleWidth, origY + 2 * rectangleHeight);
		res = vPos - vNeg;
		break;
	case doubleV1:
		vNeg = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vPos = rectangleVal(integralImage, origX+rectangleWidth, origY,
			origX + 2*rectangleWidth, origY + rectangleHeight);
		res = vPos - vNeg;
		break;
	case doubleV2 :
		vPos = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vNeg = rectangleVal(integralImage, origX + rectangleWidth, origY,
			origX + 2 * rectangleWidth, origY + rectangleHeight);
		res = vPos - vNeg;
		break;
	case tripleH:
		vPos = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vPos += rectangleVal(integralImage, origX + 2 * rectangleWidth, origY,
			origX + 3 * rectangleWidth, origY + rectangleHeight);
		vNeg = rectangleVal(integralImage, origX + rectangleWidth, origY,
			origX + 2 * rectangleWidth, origY + rectangleHeight);
		res = vPos - vNeg;
		break;
	case tripleV:
		vPos = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vPos += rectangleVal(integralImage, origX, origY+2*rectangleHeight,
			origX+rectangleWidth, origY + 3*rectangleHeight);
		vNeg = rectangleVal(integralImage, origX, origY+rectangleHeight,
			origX + rectangleWidth, origY + 2*rectangleHeight);
		res = vPos - vNeg;
		break;
	case quadrupleH:
		vPos = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vPos += rectangleVal(integralImage, origX + rectangleWidth,
			origY + rectangleHeight, origX + 2 * rectangleWidth, origY + 2 * rectangleHeight);
		vNeg = rectangleVal(integralImage, origX + rectangleWidth,
			origY, origX + 2 * rectangleWidth, origY + rectangleHeight);
		vNeg += rectangleVal(integralImage, origX, origY + rectangleHeight,
			origX + rectangleWidth, origY + 2 * rectangleHeight);
		res = vPos - vNeg;
			break;
	case quadrupleV:
		vNeg = rectangleVal(integralImage, origX, origY, origX + rectangleWidth,
			origY + rectangleHeight);
		vNeg += rectangleVal(integralImage, origX + rectangleWidth,
			origY + rectangleHeight, origX + 2 * rectangleWidth, origY + 2 * rectangleHeight);
		vPos = rectangleVal(integralImage, origX + rectangleWidth,
			origY, origX + 2 * rectangleWidth, origY + rectangleHeight);
		vPos += rectangleVal(integralImage, origX, origY + rectangleHeight,
			origX + rectangleWidth, origY + 2 * rectangleHeight);
		res = vPos - vNeg;
		break;
	}
	return res;
}
