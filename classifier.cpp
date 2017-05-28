#include "classifier.h"

/******************************/
/**** class WeakClassifier ****/
/******************************/

WeakClassifier::WeakClassifier(float w_1, float w_2) :w1(w_1), w2(w_2) {}

WeakClassifier::~WeakClassifier() {}

int WeakClassifier::h(float x) const
{
	return (w1 * x + w2 >= 0) ? 1 : -1;
}

int WeakClassifier::h(float * x) const
{
	return h(x[h.nFeatures]);
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
			n_img = pickRandomImage(c_k);
			char c = (c_k == 1) ? '+' : '-';
			cout << (i + 1) << ": training on " << c << "im" + to_string(n_img) + ".jpg" << endl;
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
	int score = 0.;
	for (int i = 0; i < VALIDATION_SIZE; i++)
	{
		Mat img;
		int n = rand() % TOTAL_IMGS;
		//cout << "testing on img No. " << n << endl;
		if (n < POS_IMGS)
			img = imread("../../valid/pos/im" + to_string(n) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		else
			img = imread("../../valid/neg/im" + to_string(n - POS_IMGS) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		img.convertTo(img, CV_32F);
		if (testImg(img))
			score+=1.;
	}
	return score / (float)VALIDATION_SIZE;
}
