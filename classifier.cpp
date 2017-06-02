#include "classifier.h"

/******************************/
/**** class WeakClassifier ****/
/******************************/

WeakClassifier::WeakClassifier(float w_1, float w_2) :w1(w_1), w2(w_2) {}
WeakClassifier::WeakClassifier() : w1(1.), w2(0.) {}

WeakClassifier::~WeakClassifier() {}

int WeakClassifier::h(float x) const
{
	return (w1 * x + w2 >= 0) ? 1 : -1;
}

void WeakClassifier::train_step(float x_ki, int c_k)
{
	w1 = w1 - EPSILON * (h(x_ki) - c_k) * x_ki;
	w2 = w2 - EPSILON * (h(x_ki) - c_k);
}

float WeakClassifier::get_w1() const { return w1; }

float WeakClassifier::get_w2() const { return w2; }

/// STATIC FUNCTIONS

int pickRandomImage(int& c_k)
{
	int n = rand() % TOTAL_IMGS;
	if (n < POS_IMGS)
	{
		c_k = 1;
		return n;
	}
	c_k = -1;
	return n - POS_IMGS;
}

/*********************************/
/**** class WeakClassifierSet ****/
/*********************************/

WeakClassifierSet::WeakClassifierSet()
{
	nFeatures = calcNFeatures();
	w1_list = new float[nFeatures];
	w2_list = new float[nFeatures];
	for (int i = 0; i < nFeatures; i++)
	{
		w1_list[i] = 1.;
		w2_list[i] = 0.;
	}
}

WeakClassifierSet::~WeakClassifierSet()
{
	delete w1_list;
	delete w2_list;
}

float* WeakClassifierSet::get_w1() const
{
	return w1_list;
}

float* WeakClassifierSet::get_w2() const
{
	return w2_list;
}


void WeakClassifierSet::train ()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//ofstream fichier1("../../w1.txt", ios::out | ios::trunc);
	//ofstream fichier2("../../w2.txt", ios::out | ios::trunc);
	const int barWidth = 50;
	int nFeatures = calcNFeatures();
	vector<float> w1_trained, w2_trained;
	for (int j = rank; j < nFeatures; j += size)
	{
		w1_trained.push_back(w1_list[j]);
		w2_trained.push_back(w2_list[j]);
	}

	// actual training loop
	for (int i = 0; i < K; i++)
	{
		Mat img, ii;
		int c_k;
		// randomly choosing an image to use, then computing its features
		float* featuresVec;
		int n_img;
		if (rank == PROC_MASTER)
		{
			//fichier1 << w1_trained[9000] << endl;
			//fichier2 << w2_trained[9000] << endl;
			n_img = pickRandomImage(c_k);
			string s = (c_k == 1) ? "pos/" : "neg/";
			float progress = (float)i / (float)K;
			cout << "[";
			for (int j = 0; j < barWidth; j++)
			{
				if (j < progress*barWidth)
					cout << "=";
				else
					cout << " ";
			}
			cout << "] " << (int)(progress*100) << "% [" << s << "im" + to_string(n_img) + ".jpg]\r";
			cout.flush();
		}
		MPI_Bcast(&n_img, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&c_k, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);

		string path;
		if (c_k == 1)
			path = "../../train/pos/im";
		else
			path = "../../train/neg/im";
		path += to_string(n_img) + ".jpg";

		img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

		img.convertTo(img, CV_32F);
		imageIntegrale(img, ii);

		featuresVec = new float[nFeatures];
		calcFeatures(ii, featuresVec, nFeatures);

		// distributed updating of w1 & w2
		int cur = rank;
		for (int j = 0; j < w1_trained.size(); j++)
		{
			WeakClassifier wc(w1_trained[j], w2_trained[j]);
			wc.train_step(featuresVec[cur], c_k);
			w1_trained[j] = wc.get_w1();
			w2_trained[j] = wc.get_w2();
			cur += size;
		}
		delete featuresVec;
		MPI_Barrier(MPI_COMM_WORLD);
	}
	//fichier1.close();
	//fichier2.close();
	if (rank == PROC_MASTER)
		cout << endl;
	float* localW1 = vectorToArray<float>(w1_trained);
	float* localW2 = vectorToArray<float>(w2_trained);
	shareComputationReorder(localW1, w1_trained.size(), w1_list, nFeatures);
	shareComputationReorder(localW2, w2_trained.size(), w2_list, nFeatures);
}



bool WeakClassifierSet::testImg(const Mat& img) const
{
	Mat ii;
	imageIntegrale(img, ii);
	float *featuresVec = new float[nFeatures];
	calcFeatures(ii, featuresVec, nFeatures);

	int score = 0;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == PROC_MASTER)
	{
		for (int i = 0; i < nFeatures; i++)
		{
			WeakClassifier wc(w1_list[i], w2_list[i]);
			score += wc.h(featuresVec[i]);
		}
	}
	MPI_Bcast(&score, 1, MPI_INT, PROC_MASTER, MPI_COMM_WORLD);
	delete featuresVec;
	return (score >= 0);
}

float WeakClassifierSet::testValid() const
{
	float score = 0.;
	int rank, barWidth = 60;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for (int i = 0; i < VALIDATION_SIZE; i++)
	{
		Mat img;
		int n = rand() % TOTAL_IMGS;
		if (rank == PROC_MASTER && (i + 1) % 10 == 0)
		{
			float progress = (float)i / (float)VALIDATION_SIZE;
			cout << "[";
			for (int j = 0; j < barWidth; j++)
			{
				if ((float)j < progress*(float)barWidth)
					cout << "=";
				else
					cout << " ";
			}
			cout << "] " << (int)(progress*100) << "% [im" << n << ".jpg]\r";
			cout.flush();
		}
		//cout << "testing on img No. " << n << endl;
		int c_k;
		if (n < POS_IMGS)
		{
			img = imread("../../valid/pos/im" + to_string(n) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
			c_k = 1;
		}
		else
		{
			img = imread("../../valid/neg/im" + to_string(n - POS_IMGS) + ".jpg",
				CV_LOAD_IMAGE_GRAYSCALE);
			c_k = -1;
		}
		img.convertTo(img, CV_32F);
		bool val = testImg(img);
		if ((val && c_k == 1) || (!val && c_k == -1))
			score+=1.;
	}
	return score / (float)VALIDATION_SIZE;
}

float WeakClassifierSet::testWholeValidationSet() const
{
	int size, rank;
	const int barWidth = 50;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	float score = 0.;
	for (int i = 0; i < TOTAL_IMGS; i++)
	{
		int c_k;
		string path;
		if (i < POS_IMGS)
		{
			path = "../../valid/pos/im" + to_string(i) + ".jpg";
			c_k = 1;
		}
		else
		{
			path = "../../valid/neg/im" + to_string(i - POS_IMGS) + ".jpg";
			c_k = -1;
		}
		if (rank == PROC_MASTER && (i + 1) % 10 == 0)
		{
			float progress = (float)i / (float)TOTAL_IMGS;
			cout << "[";
			for (int j = 0; j < barWidth; j++)
			{
				if ((float)j < progress*(float)barWidth)
					cout << "=";
				else
					cout << " ";
			}
			cout << "] " << (int)(progress * 100) << "% [im" << i << ".jpg]\r";
			cout.flush();
		}
		Mat img;
		img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
		img.convertTo(img, CV_32F);
		bool val = testImg(img);
		if ((val && c_k == 1) || (!val && c_k == -1))
			score += 1.;
	}
	return score / (float)TOTAL_IMGS;
}

void WeakClassifierSet::save(string filename) const
{
	ofstream file("../../" + filename, ios::out | ios::trunc);
	if (file)
	{
		for (int i = 0; i < nFeatures; i++)
			file << w1_list[i] << " " << w2_list[i] << endl;
		file.close();
	}
	else
		cout << "Erreur d'ouverture du fichier " << filename << endl;
	return;
}

void WeakClassifierSet::load(string filename) const
{
	cout << "loading..." << endl;
	ifstream file("../../" + filename, ios::in);
	if (file)
	{
		for (int i = 0; i < nFeatures; i++)
		{
			file >> w1_list[i] >> w2_list[i];
		}
		cout << "finished" << endl;
		file.close();
	}
	else
		cout << "Erreur d'ouverture du fichier " << filename << endl;
}

/************************************/
/************* ADABOOST *************/
/************************************/

void quickSort(float* arr, int* indexes, int left, int right)
{
	int i = left, j = right;
	float tmp;
	int tmp2;
	float pivot = arr[(left + right) / 2];

	/* partition */
	while (i <= j) {
		while (arr[i] < pivot)
			i++;
		while (arr[j] > pivot)
			j--;
		if (i <= j) {
			tmp = arr[i];
			tmp2 = indexes[i];
			arr[i] = arr[j];
			indexes[i] = indexes[j];
			arr[j] = tmp;
			indexes[j] = tmp2;
			i++;
			j--;
		}
	};

	/* recursion */
	if (left < j)
		quickSort(arr, indexes, left, j);
	if (i < right)
		quickSort(arr, indexes, i, right);
}

int E(int h, int c)
{
	return (h == c) ? 0 : 1;
}

void initFeatures(int nFeatures, float** features, int** indexes, int* pivotPoint, WeakClassifier* wc)
{
	int rank;
	const int barWidth = 50;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == PROC_MASTER)
	{
		pivotPoint = new int[nFeatures];
		features = new float*[nFeatures];
		indexes = new int*[nFeatures];
		for (int i = 0; i < nFeatures; i++)
		{
			features[i] = new float[TOTAL_IMGS];
			indexes[i] = new int[TOTAL_IMGS];
		}
		cout << "Initializing features array..." << endl;
	}
	// initializing features array
	for (int j = 0; j < TOTAL_IMGS; j++)
	{
		string path;
		int c_k;
		if (j < POS_IMGS)
		{
			path = "../../valid/pos/im" + to_string(j) + ".jpg";
			c_k = 1;
		}
		else
		{
			path = "../../valid/neg/im" + to_string(j - POS_IMGS) + ".jpg";
			c_k = -1;
		}
		Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE), ii;
		img.convertTo(img, CV_32F);
		imageIntegrale(img, ii);
		float* temp = new float[nFeatures];
		calcFeatures(ii, temp, nFeatures);
		if (rank == PROC_MASTER)
		{
			for (int i = 0; i < nFeatures; i++)
			{
				features[i][j] = temp[i];
				indexes[i][j] = j;
			}
			float progress = (float)j / (float)TOTAL_IMGS;
			cout << "[";
			for (int j = 0; j < barWidth; j++)
			{
				if (j < progress*barWidth)
					cout << "=";
				else
					cout << " ";
			}
			cout << "] " << (int)(progress * 100) << "%\r";
			cout.flush();
		}
		delete[] temp;		
	}

	for (int i = 0; i < nFeatures; i++)
		quickSort(features[i], indexes[i], 0, TOTAL_IMGS - 1);

	for (int i = 0; i < nFeatures; i++)
	{
		int prec, pivot = 0;
		bool found = false;
		for (int j = 0; j < TOTAL_IMGS; j++)
		{
			if (found)
				break;
			int val = wc[i].h(features[i][j]);
			if (j > 0)
			{
				if (val != prec)
				{
					pivotPoint[i] = j;
					found = true;
				}
			}
			prec = val;
		}
	}

	if (rank == PROC_MASTER)
		cout << endl << "done!" << endl;

}

void computeSums(float& tP, float& tM, float* sP, float* sM, float* lambda,
	const int* pivotPoint, int nFeatures, int** indexes)
{
	for (int i = 0; i < nFeatures; i++)
	{
		sP[i] = 0.;
		sM[i] = 0.;
		for (int j = 0; j < pivotPoint[i]; j++)
		{
			int ind = indexes[i][j];
			if (ind < POS_IMGS)
				sP[i] += lambda[ind];
			else
				sM[i] += lambda[ind];
		}
	}
	tP = 0.;
	tM = 0.;
	for (int j = 0; j < TOTAL_IMGS; j++)
	{
		if (j < POS_IMGS)
			tP += lambda[j];
		else
			tM += lambda[j];
	}
}

void weightedErrors(float tP, float tM, float* sP, float* sM, float* errors, int nFeatures)
{
	for (int i = 0; i < nFeatures; i++)
		errors[i] = min(sP[i] + (tM - sM[i]), sM[i] + (tP - sP[i]));
}

float alpha(float epsilon)
{
	return 0.5*log((1. - epsilon) / epsilon);
}


int minInd (float* error, int nFeatures)
{
	float min = error[0];
	int ind = 0;
	for (int i = 0; i < nFeatures; i++)
	{
		if (error[i] < min)
		{
			ind = i;
			min = error[i];
		}
	}
	return ind;
}

void updateWeights(float alpha, WeakClassifier h_k, int ind, float* lambda, float** features, int nFeatures)
{
	float sum = 0.;
	for (int j = 0; j < TOTAL_IMGS; j++)
	{
		int c_j = (j < POS_IMGS) ? 1 : -1;
		lambda[j] = lambda[j] * exp(-c_j*alpha*h_k.h(features[ind][j]));
	}

	// renormalization
	for (int j = 0; j < TOTAL_IMGS; j++)
		lambda[j] /= sum;
}

void adaboost(float* w1_list, float* w2_list, int nFeatures, vector<WeakClassifier>& result,
	vector<float>& alpha_list, vector<int>& indexes)
{
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	float tP, tM;
	float* lambda = new float[TOTAL_IMGS], *errors = new float[nFeatures],
		*sP = new float[nFeatures], *sM = new float[nFeatures];
	int* pivotPoint;
	WeakClassifier* wc = new WeakClassifier[nFeatures];
	float** features;
	int** order;

	// initializing...
	for (int i = 0; i < TOTAL_IMGS; i++)
		lambda[i] = 1. / (float)TOTAL_IMGS;
	for (int i = 0; i < nFeatures; i++)
		wc[i] = WeakClassifier(w1_list[i], w2_list[i]);
	initFeatures(nFeatures, features, order, pivotPoint, wc);
	// actual loop
	for (int i = 0; i < N; i++)
	{
		if (rank == PROC_MASTER)
		{
			cout << "Loop No. " << i + 1 << endl;
			cout << "computing weighted error" << endl;
			cout << features[0][0] << endl;
			computeSums(tP, tM, sP, sM, lambda, pivotPoint, nFeatures, order);
			weightedErrors(tP, tM, sP, sM, errors, nFeatures);
			cout << "seeking for min classifier" << endl;
			int ind = minInd(errors, nFeatures);
			result.push_back(wc[ind]);
			float epsilon = errors[ind];
			float a = alpha(epsilon);
			alpha_list.push_back(a);
			indexes.push_back(ind);
			cout << "updating weights" << endl;
			updateWeights(a, wc[ind], ind, lambda, features, nFeatures);
		}
	}
	delete[] wc;
	delete[] errors;
	delete[] lambda;
	if (rank == PROC_MASTER)
	{
		delete[] pivotPoint;
		for (int i = 0; i < nFeatures; i++)
		{
			delete[] features[i];
			delete[] order[i];
		}
		delete[] order;
		delete[] features;
	}
}
