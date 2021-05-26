#include "oneHDR.h"
#include "SpSolve.h"

void display(float *array, int cols, int row)
{
	cout << row << "th row:\n";
	for (int i = 0; i < cols; i++) {
		cout << i << "," << array[row * cols + i] << endl;
	}
	cout << endl;
}

void display(float *array, int cols, int row1, int row2)
{
	cout << row1 << "," << row2 << "th row:\n";
	for (int i = 0; i < cols; i++) {
		cout << i << "," << array[row1 * cols + i] << "," << array[row2 * cols + i] << endl;
	}
	cout << endl;
}

cv::Mat oneHDR(cv::Mat &inputImg, float mu, float a, float b)
{
	uchar *I_d_ind;
	int size;
	float *t_b, *t_our;
	cv::Mat I;
	cv::resize(inputImg, I, cv::Size(256, 256), 0, 0, CV_INTER_CUBIC);

	size = I.rows * I.cols;
	// initialize
	t_b = new float[size]; //cv::Mat aa(size, 1, CV_32FC1, t_b);
	//
	I_d_ind = I.data;
	for (int ind = 0; ind < size; ind++) {
		t_b[ind] = max(max(I_d_ind[0], I_d_ind[1]), I_d_ind[2]) / 255.0f;
		//cout << inputImg.at<Vec3b>(ind / inputImg.cols, ind % inputImg.cols) << ", " << t_b[ind] << endl;
		I_d_ind += 3;
	}
	/*cv::Mat t_b_(inputImg.rows, inputImg.cols, CV_32FC1, t_b), I;
	cv::resize(t_b_, I, cv::Size(256, 256), 0, 0, CV_INTER_CUBIC);
	t_our = tsmooth((float*)I.data, I.rows);*/
	t_our = tsmooth((float*)t_b, I.rows);
	/*cout << I.at<Vec3b>(0, 0) << endl;
	cout << I_d[0] / 255.0 << "," << I_d[1] / 255.0 << "," << I_d[2] / 255.0  << endl;*/
	/*img2.data;
	img2 /= 255.0;
	t_b = cv::Mat(inputImg.rows, inputImg.cols, CV_32FC1);
	for (int i = 0; i < inputImg.rows; i++) {
		for (int j = 0; j < inputImg.cols; j++) {
			Vec3f pixel = img2.at<Vec3f>(i, j);
			t_b.at<float>(i, j) = max(max(pixel[0], pixel[1]), pixel[2]);
		}
	}
	cout << "t_b (" << t_b.rows << ", " << t_b.cols << ")" << endl;
	cv::resize(t_b, t_b2, cv::Size(256, 256), interpolation=CV_INTER_CUBIC);
	t_our = tsmooth(t_b2);*/

	//dst *= 255.0;
	//dst.convertTo(img3, CV_8UC3);
	//sample_solve();
	return inputImg;
}

float* tsmooth(float *I, int rows, float lambd, int sigma, float sharpness)
{
	int size = rows * rows;
	float *wx = new float[size], *wy = new float[size], *S;
	computeTextureWeights(I, rows, sigma, sharpness, wx, wy);
	S = solveLinearEquation(I, rows, wx, wy, lambd);
	return S;
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
	cv::Mat gauker_h, gauker_v, kernel_v(1, sigma, CV_32FC1, 1.0f), kernel_h(sigma, 1, CV_32FC1, 1.0f);
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

float* solveLinearEquation(float *IN, int rows, float *wx, float *wy, float lambd)
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
	float max_t = tripletlist[0].value(), min_t = tripletlist[0].value();
	for (size_t i = 1; i < tripletlist.size(); i++) {
		max_t = max(max_t, tripletlist[i].value());
		min_t = min(min_t, tripletlist[i].value());
	}
	cout << max_t << " " << min_t << endl;
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
	float max_x = x(0), min_x = x(0);
	for (int i = 1; i < size; i++)
	{
		max_x = max(max_x, x(i));
		min_x = min(min_x, x(i));
	}
	cout << max_x / 1.0f << " " << min_x / 1.0f << endl;
	return IN;
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
		int next_i = i > 0 ? i - 1 : rows - 1;
		for (int j = 0; j < rows; j++) {
			dx[i * rows + j] = lambd * wx[j * rows + next_i];
		}
	}
	return dx;
}

float* convertCol_delay_col(float *wx, int rows, float lambd)
{
	float *dx = new float[rows * rows];
	for (int j = 0; j < rows; j++) {
		int next_j = j > 0 ? j - 1 : rows - 1;
		for (int i = 0; i < rows; i++) {
			dx[i * rows + j] = lambd * wx[next_j * rows + i];
		}
	}
	return dx;
}
