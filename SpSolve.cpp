#include "SpSolve.h"

typedef Eigen::SparseMatrix<double> SparseMatrixType;
typedef Eigen::Triplet<double> T;
typedef Eigen::SimplicialCholesky<SparseMatrixType> Solve;


int sample_solve()
{
	int row_A, col_A, row_b;
	col_A = 100; row_A = 100;
	row_b = row_A;

	SparseMatrixType A(row_A, col_A);
	Eigen::VectorXd x;
	Eigen::VectorXd b;
	std::vector<T> tripletlist;

	b.resize(row_b);
	for (int i = 0; i < row_b; i++)
	{
		b(i) = i + 1;
	}

	for (int i = 0; i < row_A; i++)
	{
		for (int j = 0; j < col_A; j++)
		{
			tripletlist.push_back(T((i + 3) % row_A, j, i + j));
			tripletlist.push_back(T(i, j, i + 1));
		}
	}
	A.setFromTriplets(tripletlist.begin(), tripletlist.end());
	A.makeCompressed();

	Solve *p_A = new Solve(A);
	x = p_A->solve(b);

	for (int i = 0; i < x.size(); i++)
	{
		std::cout << x(i) << "\n";
	}
	return 0;
}
