#pragma once
#include<Eigen\Sparse>
#include<Eigen\Dense>
#include<vector>
#include<iostream>

typedef Eigen::SparseMatrix<float> SparseMatrixType;
typedef Eigen::Triplet<float> T;
typedef Eigen::SimplicialCholesky<SparseMatrixType> Solve;
int sample_solve();
