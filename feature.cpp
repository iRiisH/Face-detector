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


void intToDimensions(int n, int &x, int &y, int &w, int &h)
// this function is used to distribute efficiently the computation, given that there would
// be four imbricated loops in a sequential algorithm
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




void shareComputation(float *localArray, int localSize, float *result, int& totalSize)
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int* sizes = new int[size];
	MPI_Gather(&localSize, 1, MPI_INT, sizes, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
	int* cumulativeSizes;
	if (rank == PROC_MASTER)
	{
		totalSize = 0;
		cumulativeSizes = new int[size];
		cumulativeSizes[0] = 0;
		for (int i = 0; i < size; i++)
		{
			totalSize += sizes[i];
			if (i == 0)
				continue;
			cumulativeSizes[i] = cumulativeSizes[i - 1] + sizes[i - 1];
		}
	}
	MPI_Bcast(&totalSize, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
	float *gathered = new float[totalSize];
	//for (int i = 0; i < localSize; i++)
	//{
		//cout << rank << " - " << i << " - " << localArray[i] << endl;
	//}

	//cout << localSize << " - " << rank << endl;
	//cout << "total size: " << totalSize << '[' << rank << ']' << endl;
	//cout << "cumulative size " << cumulativeSizes[2] << "-" <<rank << endl;
	cout << "yay" << endl;
	MPI_Gatherv(localArray, localSize, MPI_FLOAT, gathered, sizes, cumulativeSizes, MPI_FLOAT, PROC_MASTER, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//cout << "inter" << rank << endl;
	if (rank == PROC_MASTER)
	{
		int cur = 0;
		for (int i = 0; i < size; i++)
		{
			int cursor = i;
			for (int j = 0; j < sizes[i]; j++)
			{
				result[cursor] = gathered[cur];
				cursor += size;
				cur++;
			}
		}
	}
	//cout << "barrier" << rank << endl;

	MPI_Bcast(result, totalSize, MPI_FLOAT, PROC_MASTER, MPI_COMM_WORLD);
	delete[] gathered;
	delete[] sizes;
	if (rank == PROC_MASTER)
		delete[] cumulativeSizes;
	///cout << "finex" << endl;

}

void calcLocalFeatures(Mat& ii, vector<float>& localResult)
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
			localResult.push_back(f.val(ii));
		}
	}
}

int calcNFeatures()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// opens a random image
	Mat img;
	img = imread("../../im1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img, CV_32F);
	Mat ii;
	imageIntegrale(img, ii);
	
	vector<float> v;
	calcLocalFeatures(ii, v);
	float* localResult = vectorToArray<float>(v);
	int localSize = v.size();
	int* sizes = new int[size];
	MPI_Gather(&localSize, 1, MPI_INT, sizes, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
	int totalSize = 0;
	if (rank == PROC_MASTER)
	{
		for (int i = 0; i < size; i++)
			totalSize += sizes[i];
	}
	delete[] localResult;
	MPI_Bcast(&totalSize, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
	return totalSize;
}


void calcFeatures(Mat& ii, float *result, int nFeatures)
// we have to be careful when regrouping the features, because doing it sequentially 
// would suppress the acceleration we got from distributed computation of the features
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	vector<float> v;
	
	calcLocalFeatures(ii, v);
	
	float* localResult = vectorToArray<float>(v);
	int localSize = v.size();
	int* sizes = new int[size];
	cout << "barrier " << rank << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "local size = " << localSize << endl;
	cout << "checking... " << localResult[localSize - 1] << endl;
	shareComputation(localResult, localSize, result, nFeatures);
	delete[] localResult;
	delete[] sizes;
}
