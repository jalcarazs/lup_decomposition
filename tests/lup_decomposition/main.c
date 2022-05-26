/*
* Copyright (c) 2016-2019 Irlan Robson
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/

/*
* Javier Alcaraz detected a bug in the code when calculating the determinant of a matrix.
* Sometimes, the code returned the correct determinant but with an incorrect sign.
* For example in a 3x3 matrix like this:
* [1,      1,     0.067]
* [-1.23, -1.23, -0.173]
* [-1.34, -1.95, -0.856]
* it returned 0.054 instead of -0.054.
* The code has been corrected, and now the sign of the determinant is calculated based on the variable
* num_swaps, which gives the number of row interchanges in the matrix
*/

#define lup_decomposition_implementation
#include "lup_decomposition.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
//#include <math.h>

float random_float(float a, float b)
{
	float r = (float)(rand()) / (float)(RAND_MAX);
	float d = b - a;
	return a + r * d;
}

void clone_floats(float* dst, float* src, int n)
{
	for (int i = 0; i != n; ++i)
	{
		dst[i] = src[i];
	}
}

void print_matrix(const float* A, int AM, int AN)
{
	for (int i = 0; i < AM; ++i)
	{
		printf("[");
		for (int j = 0; j < AN; ++j)
		{
			if (j == 0)
			{
				printf("%f", A[i + AM * j]);
			}
			else
			{
				printf("\t%f", A[i + AM * j]);
			}
		}
		printf("]\n");
	}
}

void mul_matrix(float* C, const float* A, int AM, int AN, const float* B, int BM, int BN)
{
	for (int i = 0; i < AM; ++i)
	{
		for (int j = 0; j < BN; ++j)
		{
			C[i + AM * j] = 0.0f;

			for (int k = 0; k < AN; ++k)
			{
				C[i + AM * j] += A[i + AM * k] * B[k + BM * j];
			}
		}
	}
}

void test_decompose()
{
#define n 6

	float A[n * n];
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			A[i + n * j] = random_float(-1000.0f, 1000.0f);
		}
	}

	printf("A = \n\n");
	print_matrix(A, n, n);
	printf("\n");

	// Copy A
	float G[n * n];
	clone_floats(G, A, n * n);

	// P
	int p[n];

	// Run LU factorization.
	lup_decomposition_decompose(G, p, n);

	printf("G(A) = \n\n");
	print_matrix(G, n, n);
	printf("\n");

	printf("p = \n\n");
	for (int i = 0; i < n; ++i)
	{
		printf("[%d]\n", p[i]);
	}
	printf("\n");

	// L(PA) = I + L(G)
	float L[n * n];
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (i < j)
			{
				L[i + n * j] = 0;
				continue;
			}

			if (i > j)
			{
				L[i + n * j] = G[i + n * j];
				continue;
			}

			if (i == j)
			{
				L[i + n * j] = 1;
				continue;
			}
		}
	}

	printf("L(PA) = I + L(G) = \n\n");
	print_matrix(L, n, n);
	printf("\n");

	// U(PA) = U(G)
	float U[n * n];
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			if (i <= j)
			{
				U[i + n * j] = G[i + n * j];
				continue;
			}

			if (i > j)
			{
				U[i + n * j] = 0;
				continue;
			}
		}
	}

	printf("U(PA) = U(G) = \n\n");
	print_matrix(U, n, n);
	printf("\n");

	// PA = LU
	float L_U[n * n];
	mul_matrix(L_U, L, n, n, U, n, n);

	printf("LU = \n\n");
	print_matrix(L_U, n, n);
	printf("\n");

	// PA
	float P_A[n * n];
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < n; ++i)
		{
			int k = p[i];

			P_A[i + n * j] = A[k + n * j];
		}
	}

	printf("PA = \n\n");
	print_matrix(P_A, n, n);
	printf("\n");

#undef n
}

void test_solve()
{
#define n 6

	float A[n * n] =
	{
		1, 0, 1, 6, -4, 0,
		-1, 5, 0, -1, 2, 7,
		4, -2, 5, 2, 0, -1,
		0, 7, 7, 3, 5, 5,
		2, 8, 3, 0, -5, 4,
		9, 4, -2, 8, 3, -2
	};

	float b[n] =
	{
		19, 2, 13, -7, -9, 2
	};

	printf("A = \n\n");
	print_matrix(A, n, n);
	printf("\n");

	printf("b = \n\n");
	print_matrix(b, n, 1);
	printf("\n");

	float G[n * n];
	clone_floats(G, A, n * n);

	int p[n];
	lup_decomposition_decompose(G, p, n);

	float x[n];
	int solved = lup_decomposition_solve(x, G, p, b, n);
	assert(solved == 1);

	printf("x = \n\n");
	print_matrix(x, n, 1);
	printf("\n");

	float xm[n] =
	{
		-1.761817043997860f,
		0.896228033874012f,
		4.051931404116157f,
		-1.617130802539541f,
		2.041913538501913f,
		0.151832487155935f,
	};

	printf("x (MATLAB) = \n\n");
	print_matrix(xm, n, 1);
	printf("\n");

#undef n
}

void test_determinant()
{
#define n 4
    
    int num_swaps;

	float A[n * n] =
	{
		8, 0, -3, 1,
		11, -7, -7, 1,
		2, 2, 2, 2,
		8, -1, 1, 4
	};

	printf("A = \n\n");
	print_matrix(A, n, n);
	printf("\n");

	float GA[n * n];
	clone_floats(GA, A, n * n);

	int pA[n];
    num_swaps=lup_decomposition_decompose(GA, pA, n);

	double detA = lup_decomposition_determinant(GA, pA, n, num_swaps);
	printf("det(A) = %f\n", detA);
	printf("det(A) (MATLAB) = %f\n\n", 76.0f);

	float B[n * n] =
	{
		1,  0,  0, 0,
		-2, 7, 4, 0,
		0, 1, 4, 0,
		5, 5, 0, 2
	};

	printf("B = \n\n");
	print_matrix(B, n, n);
	printf("\n");

	float GB[n * n];
	clone_floats(GB, B, n * n);

	int pB[n];
    num_swaps=lup_decomposition_decompose(GB, pB, n);

	double detB = lup_decomposition_determinant(GB, pB, n, num_swaps);
	printf("det(B) = %f\n", detB);
	printf("det(B) (MATLAB) = %f\n\n", 48.0f);

#undef n
}

void test_inverse()
{
#define n 6

	float A[n * n] =
	{
		2, 2, 2, 3, -8, 1,
		5, 1, 2, 1, 2, 0,
		56, -100, 2, 56, -2, 0,
		1, 12, 46, 3, 2, 0,
		90, 80, 2, 3, 2, 892,
		0, 2, 2, 1, 23, 0
	};

	printf("A = \n\n");
	print_matrix(A, n, n);
	printf("\n");

	float G[n * n];
	clone_floats(G, A, n * n);

	int p[n];
	lup_decomposition_decompose(G, p, n);

	float B[n * n];
	int inverted = lup_decomposition_inverse(B, G, p, n);
	assert(inverted == 1);

	printf("A^-1 = \n\n");
	print_matrix(B, n, n);
	printf("\n");

	float matlab_inverse_A[n * n] =
	{
			-0.0524332761f, 0.2234121316f, -0.0002084597f, -0.0058107137f, 0.0000587817f, -0.0371827621f,
			0.1223027961f, 0.0391669519f, -0.0074947566f, -0.0083940122f, -0.0001371108f, 0.0392243999f,
			-0.0477005814f, -0.0052473957f, 0.0016440853f, 0.0247565241f, 0.0000534760f, -0.0181496387f,
			0.2718808594f, -0.1531512668f, 0.0046343648f, -0.0100978907f, -0.0003047992f, 0.1091923269f,
			-0.0183080560f, 0.0037092241f, 0.0003072599f, -0.0009837884f, 0.0000205247f, 0.0368981805f,
			-0.0064449037f, -0.0255357823f, 0.0006732468f, 0.0013197684f, 0.0011283015f, -0.0001755353f
	};

	printf("A^-1 (MATLAB) = \n\n");
	print_matrix(matlab_inverse_A, n, n);
	printf("\n");

#undef n
}

int main(int argument_count, char** arguments)
{
	printf("Decompose:\n\n");
	test_decompose();
	printf("\n");

	printf("Solve:\n\n");
	test_solve();
	printf("\n");

	printf("Determinant:\n\n");
	test_determinant();
	printf("\n");

	printf("Inverse:\n\n");
	test_inverse();
	printf("\n");

	return 0;
}
