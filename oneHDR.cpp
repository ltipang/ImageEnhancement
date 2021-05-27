#include "oneHDR.h"
#include "SpSolve.h"

bool debug = false;

cv::Mat oneHDR(cv::Mat &inputImg)
{
	cv::Mat I;	inputImg.convertTo(I, CV_32FC1); I /= 255.0;

	// finalize
	float gamma = 0.0f, beta = 0.0f;
	float temp;
	cv::Mat W = get_enhance_params(I, gamma, beta);
	int size = inputImg.cols * inputImg.rows;
	float *fused = new float[size], *I_ = (float*)I.data, *W_ = (float*)W.data;
	for (int i = 0; i < size; i++) {
		temp = pow(I_[i], gamma) * beta;
		fused[i] = ((I_[i] * W_[i]) + temp * (1.0f - W_[i])) * 255.0f;
	}
	cv::Mat dst(I.rows, I.cols, CV_32FC1, fused), final_img;
	dst.convertTo(final_img, CV_8UC1);
	
	delete[] fused;
	return final_img;
}

cv::Mat get_enhance_params(cv::Mat &I, float &gamma, float &beta)
{
	int size;
	int64 t1 = cv::getTickCount();
	cv::Mat I1;
	cv::resize(I, I1, cv::Size(256, 256), 0, 0, CV_INTER_CUBIC);
	size = I1.rows * I1.rows;
	float *t_our = new float[size * 2];
	tsmooth((float*)I1.data, t_our, I1.rows);
	float *W_f = t_our + size;
	//
	cv::Mat t(I1.rows, I1.cols, CV_32FC1, t_our);
	cv::Mat W(I1.rows, I1.cols, CV_32FC1, W_f);
	cv::resize(W, W, cv::Size(I.rows, I.cols), 0, 0, CV_INTER_CUBIC);
#if debug
	cv::resize(t, t, cv::Size(I.rows, I.cols), 0, 0, CV_INTER_CUBIC);
	cv::imwrite("W.jpg", W * 255);
	cv::imwrite("1-W.jpg", (1 - W) * 255);
	cv::imwrite("t.jpg", t * 255);
	cv::imwrite("1-t.jpg", (1 - t) * 255);
#endif
	//
	int64 t2 = cv::getTickCount();
	maxEntropyEnhance(I, t, gamma, beta);
	int64 t3 = cv::getTickCount();
	calc_timing(t2, t3, "maxEntropyEnhance time");
	calc_timing(t1, t3, "total paramerization time");
	delete[] t_our;
	return W;
}

int tsmooth(float *I, float *t_W, int rows, float lambd, int sigma, float sharpness)
{
	int64 t0 = cv::getTickCount();
	int size = rows * rows;
	float *wx = new float[size], *wy = new float[size];
	computeTextureWeights(I, rows, sigma, sharpness, wx, wy);
	int64 t1 = cv::getTickCount();
	solveLinearEquation(I, rows, wx, wy, lambd, t_W);
	int64 t2 = cv::getTickCount();
	calc_timing(t0, t1, "computeTextureWeights time");
	calc_timing(t1, t2, "solveLinearEquation time");

	delete[] wx;
	delete[] wy;

	return 1;
}

void computeTextureWeights(float *fin, int rows, int sigma, float sharpness, float *W_h, float *W_v)
{
	int size = rows * rows;
	float *dt0_v = new float[size];
	float *dt0_h = new float[size];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rows; j++) {
			int next_i = i < rows - 1 ? i + 1 : 0;
			int next_j = j < rows - 1 ? j + 1 : 0;
			dt0_v[i * rows + j] = fin[next_i * rows + j] - fin[i * rows + j];
			dt0_h[i * rows + j] = fin[i * rows + next_j] - fin[i * rows + j];
		}
	}
	cv::Mat dt0_v_(rows, rows, CV_32FC1, dt0_v);
	cv::Mat dt0_h_(rows, rows, CV_32FC1, dt0_h);
	cv::Mat gauker_h, gauker_v, kernel_h(1, sigma, CV_32FC1, 1.0f), kernel_v(sigma, 1, CV_32FC1, 1.0f);
	cv::filter2D(dt0_v_, gauker_v, -1, kernel_v, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(dt0_h_, gauker_h, -1, kernel_h, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

	float *gauker_h_t, *gauker_v_t;
	gauker_h_t = (float*)gauker_h.data; gauker_v_t = (float*)gauker_v.data;
	for (int ind = 0; ind < size; ind++) {
		W_h[ind] = 1.0f / (abs(gauker_h_t[ind]) * abs(dt0_h[ind]) + sharpness);
		W_v[ind] = 1.0f / (abs(gauker_v_t[ind]) * abs(dt0_v[ind]) + sharpness);
	}
	delete[]dt0_v;
	delete[]dt0_h;
}

int solveLinearEquation(float *IN, int rows, float *wx, float *wy, float lambd, float *t_W)
{
	int size = rows * rows;
	float *dx, *dy, *dxa, *dya, *matCol;
	float *dxd1 = new float[size], *dyd1 = new float[size], *dxd2 = new float[size], *dyd2 = new float[size];
	dx = convertCol(wx, rows, -lambd);
	dy = convertCol(wy, rows, -lambd);
	dxa = convertCol_delay_col(wx, rows, -lambd);
	dya = convertCol_delay_row(wy, rows, -lambd);
	memset(dxd1, 0, sizeof(float) * size); memset(dyd1, 0, sizeof(float) * size);
	memcpy(dxd2, dx, sizeof(float) * size); memcpy(dyd2, dy, sizeof(float) * size);
	for (int i = 0; i < rows; i++) {
		dxd1[i] = dxa[i];
		dyd1[i * rows] = dya[i * rows];
		dxd2[(rows - 1) * rows + i] = 0;
		dyd2[i * rows + (rows - 1)] = 0;
	}
	// spSolve
	std::vector<T> tripletlist;
	for (int i = 0; i < size; i++){
		tripletlist.push_back(T(i, i, (1.0f - (dx[i] + dy[i] + dxa[i] + dya[i])) / 1.0f));
		int i1 = size - rows + i; 
		if (i1 < size) {
			if (dxd1[i] != 0.0f) {
				tripletlist.push_back(T(i1, i, dxd1[i] / 1.0f));
				tripletlist.push_back(T(i, i1, dxd1[i] / 1.0f));
			}
		}
		int i2 = rows + i;
		if (i2 < size){
			if (dxd2[i] != 0.0f) {
				tripletlist.push_back(T(i2, i, dxd2[i] / 1.0f));
				tripletlist.push_back(T(i, i2, dxd2[i] / 1.0f));
			}
		}
		int i3 = i2 - 1;
		if (i3 < size) {
			if (dyd1[i] != 0.0f) {
				tripletlist.push_back(T(i3, i, dyd1[i] / 1.0f));
				tripletlist.push_back(T(i, i3, dyd1[i] / 1.0f));
			}
		}
		int i4 = i + 1;
		if (i4 < size) {
			if (dyd2[i] != 0.0f) {
				tripletlist.push_back(T(i4, i, dyd2[i] / 1.0f));
				tripletlist.push_back(T(i, i4, dyd2[i] / 1.0f));
			}
		}
	}
	matCol = convertCol(IN, rows, 1.0f);
	SparseMatrixType A(size, size);
	Eigen::VectorXf x;
	Eigen::VectorXf b;
	A.setFromTriplets(tripletlist.begin(), tripletlist.end());
	A.makeCompressed();
	b.resize(size);
	for (int i = 0; i < size; i++)
	{
		b(i) = matCol[i];
	}
	Solve *p_A = new Solve(A);
	x = p_A->solve(b);
	float *t = t_W, *W = t_W + size;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rows; j++) {
			t[i * rows + j] = x[j * rows + i];
			W[i * rows + j] = sqrt(x[j * rows + i]);
		}
	}

	delete[]dx; delete[]dy; delete[]dxa; delete[]dya; delete[]matCol;
	delete[]dxd1; delete[]dyd1; delete[]dxd2; delete[]dyd2;
	delete p_A;
	tripletlist.clear();
	return 1;
}

float* convertCol(float *wx, int rows, float lambd)
{
	float *dx = new float[rows * rows];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rows; j++) {
			dx[i * rows + j] = lambd * wx[j * rows + i];
		}
	}
	return dx;
}

float* convertCol_delay_row(float *wx, int rows, float lambd)
{
	float *dx = new float[rows * rows];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < rows; j++) {
			int next_j = j > 0 ? j - 1 : rows - 1;
			dx[i * rows + j] = lambd * wx[next_j * rows + i];
		}
	}
	return dx;
}

float* convertCol_delay_col(float *wx, int rows, float lambd)
{
	float *dx = new float[rows * rows];
	for (int i = 0; i < rows; i++) {
		int next_i = i > 0 ? i - 1 : rows - 1;
		for (int j = 0; j < rows; j++) {
			dx[i * rows + j] = lambd * wx[j * rows + next_i];
		}
	}
	return dx;
}

int maxEntropyEnhance(cv::Mat &I, cv::Mat &t, float &gamma, float &beta, float bad_threshold, float mink, float maxk, float a, float b)
{
	cv::Mat Y1, t1;
	cv::resize(I, Y1, cv::Size(50, 50));
	cv::resize(t, t1, cv::Size(50, 50));
	float *Y = new float[50 * 50]; int Y_len = 0;
	float *t1_ = (float*)t1.data, *Y1_ = (float*)Y1.data, factor = 1.0f / 3.0f;
	for (int i = 0; i < 50 * 50; i++) {
		if (t1_[i] < bad_threshold) {
			float temp = Y1_[i];
			if (temp > 0.0f) {
				Y[Y_len++] = temp;
			}
		}
	}
	// fminbound
	//float lb = -3.0f, ub = 6.0f, g = (sqrt(5.0f) - 1.0f) / 2.0f, E=1e-3f;
	float E = 1e-3f, lb = mink, ub = maxk, g = (sqrt(5.0f) - 1.0f) / 2.0f;
	float f_lb, f_ub1, lb1, ub1; int repeated_num = 0;
	lb1 = lb + (1 - g)*(ub - lb); ub1 = lb + g*(ub - lb);
	f_lb = negative_entropy(Y, Y_len, lb, a, b); f_ub1 = negative_entropy(Y, Y_len, ub1, a, b);
	while ((ub - lb) > E) {
		if (f_lb > f_ub1) {
			lb = lb1; f_lb = negative_entropy(Y, Y_len, lb, a, b);
			lb1 = ub1;
			ub1 = lb + g * (ub - lb); f_ub1 = negative_entropy(Y, Y_len, ub1, a, b);
		}
		else {
			ub = ub1; 
			ub1 = lb1; f_ub1 = negative_entropy(Y, Y_len, ub1, a, b);
			lb1 = lb + (1 - g)*(ub - lb);
		}
		repeated_num++;
	}
	float opt_k = (ub + lb) / 2;
#if debug
	cout << "-entropy(Y): " << -entropy(Y, Y_len) << endl;
	cout << opt_val << ", " << negative_entropy(Y, Y_len, ub1, a, b) << endl;
	cout << "repeated_num: " << repeated_num << endl;
#endif
	//applyK(I, opt_k)
	delete[]Y;
	gamma = pow(opt_k, a); beta = exp((1 - gamma) * b);
	return 1;
}

float entropy(float *Y, int num)
{
	float logsum = 0.0f, sum = 0.0f;
	for (int i = 0; i < num; i++) {
		logsum += Y[i] * log(Y[i]);
		sum += Y[i];
	}
	return -(logsum / sum - log(sum));
}

float negative_entropy(float *Y, int num, float k, float a, float b)
{
	float gamma = pow(k, a), beta = exp((1 - gamma) * b);
	float logsum = 0.0f, sum = 0.0f;
	for (int i = 0; i < num; i++) {
		float temp = pow(Y[i], gamma) * beta;
		logsum += temp * log(temp);
		sum += temp;
	}
	return (logsum / sum - log(sum));
}

int calc_timing(int64 srart_time, int64 end_time, const char *string)
{
	double time = (end_time - srart_time) / cv::getTickFrequency() * 1000;
	printf("%s: %f ms\n", string, time / 10);
	return 1;
}