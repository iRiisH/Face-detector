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
	shareComputation(localW1, w1_trained.size(), w1_list, nFeatures);
	shareComputation(localW2, w2_trained.size(), w2_list, nFeatures);
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

/************************************/
/************* ADABOOST *************/
/************************************/



int E(int h, int c)
{
	return (h == c) ? 0 : 1;
}

void weightedError(WeakClassifier* wc, float* lambda, float* errors, int nFeatures)
{
	float* features = new float[nFeatures];
	for (int i = 0; i < nFeatures; i++)
		errors[i] = 0.;
	
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
		calcFeatures(ii, features, nFeatures);
		// this is the biggest computation, as there are 5233x214000 features to compute
		for (int i = 0; i < nFeatures; i++)
		{
			errors[i] += lambda[j] * E(wc[i].h(features[i]), c_k);
		}
	}
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

void updateWeights(float alpha, WeakClassifier h_k, int ind, float* lambda, int nFeatures)
{
	float* features = new float[nFeatures];
	float sum = 0.;
	for (int j = 0; j < TOTAL_IMGS; j++)
	{
		string path;
		int c_j;
		if (j < POS_IMGS)
		{
			path = "../../valid/pos/im" + to_string(j) + ".jpg";
			c_j = 1;
		}
		else
		{
			path = "../../valid/neg/im" + to_string(j - POS_IMGS) + ".jpg";
			c_j = -1;
		}
		Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE), ii;
		img.convertTo(img, CV_32F);
		imageIntegrale(img, ii);
		calcFeatures(ii, features, nFeatures);
		lambda[j] = lambda[j] * exp(-c_j*alpha*h_k.h(features[ind]));
	}

	// renormalization
	for (int j = 0; j < TOTAL_IMGS; j++)
		lambda[j] /= sum;
	delete[] features;
}

void adaboost(float* w1_list, float* w2_list, int nFeatures, vector<WeakClassifier>& result,
	vector<float>& alpha_list, vector<int>& indexes)
{
	float* lambda = new float[TOTAL_IMGS];
	WeakClassifier* wc = new WeakClassifier[nFeatures];

	float* errors = new float[nFeatures];

	// initializing...
	for (int i = 0; i < TOTAL_IMGS; i++)
		lambda[i] = 1. / (float)TOTAL_IMGS;
	for (int i = 0; i < nFeatures; i++)
		wc[i] = WeakClassifier(w1_list[i], w2_list[i]);
	
	// actual loop
	for (int i = 0; i < N; i++)
	{
		weightedError(wc, lambda, errors, nFeatures);
		int ind = minInd(errors, nFeatures);
		result.push_back(wc[ind]);
		float epsilon = errors[ind];
		float a = alpha(epsilon);
		alpha_list.push_back(a);
		indexes.push_back(ind);
		updateWeights(a, wc[ind], ind, lambda, nFeatures);
	}
	delete[] wc;
	delete[] errors;
	delete[] lambda;
}
