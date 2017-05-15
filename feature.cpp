#include "feature.h"



void imageIntegrale(const Mat& input, Mat& output)
{
	int m = input.rows, n = input.cols;
	output = Mat::zeros(m, n, CV_32F);
	Mat s = Mat::zeros(m, n, CV_32F);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == 0)
				s.at<float>(i, j) = input.at<float>(i, j);
			else
				s.at<float>(i, j) = s.at<float>(i - 1, j) + input.at<float>(i, j);
			if (j == 0)
				output.at<float>(i, j) = s.at<float>(i, j);
			else
				output.at<float>(i, j) = output.at<float>(i, j - 1) + s.at<float>(i, j);
		}
	}
}

void intToDimensions(int n, int &x, int &y, int &w, int &h)
{
	// 112 / 4 = 28, 92 / 4 = 23
	// n = h + 23 * w + 23 * 28 * y + 23 * 23 * 28 * x
	h = 4 * (n % 23);
	int reste = (n - (n % 23)) / 23;
	w = 4 * (reste % 28);
	reste = (reste - (reste % 28)) / 28;
	y = 4 * (reste % 23);
	reste = (reste - (reste % 23)) / 23;
	x = 4 * reste;
}


void filltab(vector<float>& localresult, vector<float>& tab, vector<float>& result) {
	for (int i = 0; i<localresult.size(); i++) {
		if (rank>1) {
			result[tab[rank - 2] + i] = localresult[i];
		}//see if rank -1 ou -2
		else {
			result[i] = localresult[i];
		}
	}

}

void calcFeatures(Mat& ii, vector<float>& result)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	for (int i = rank; i < 23 + 23 * 28 + 23 * 23 * 28 + 23 * 23 * 28 * 28; i += size)
	{
		int x, y, w, h;
		intToDimensions(i, x, y, w, h);
		for (int type = 0; type <= 7; type++)
		{
			FeatureType ftype = (FeatureType)type;
			Feature f(ftype, w, h, x, y);

			if (!f.fits() || w < 8 || h < 8)
				continue;
			result.push_back(f.val(ii));
		}
	}
	tab[rank] = result.size();
}

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
