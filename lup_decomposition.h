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

#if !defined(lup_decomposition_h)

// Decompose a given square matrix A stored in columj-major order of order n into PA = LU.
// The LU factorization will be encoded in A. Therefore, A will be overwritten.
// The permutation matrix P will be encoded in p. Therefore, an array of integers p of size n must be provided.
// Let A' be the matrix A after this algorithm is called.
// The lower triangular part L is defined as L = I + L(A'), where I is the identity matrix and L(A') is 
// the lower unit triangular part of A'.
// The upper triangular part U is defined as U = U(A'), the upper triangular part of A' including the diagonal.
// Note: Multiplying U by L yields in PA = LU, not the original matrix A.
// Suggestion: The original matrix can be recovered by computing A = P^-1 L U.
// Warning: If the matrix is ill-conditioned (close to singularity) then the decomposition can be a rough estimate of the exact decomposition leading to inexact determinants and solutions.
// In that case, consider using SVD (single value decomposition) instead.
// This algorithm has a memory cost of O(n^2) + O(n) and basically a time cost of O(n^3).
// You can call this function once to compute the decomposition and then solve a sequence of linear systems.
void lup_decomposition_decompose(float* A, int* p, int n);

// Solve Ax = b given the LUP decomposition of A encoded in A.
// This function will return 1 if a unique solution has been found, or 0 if the matrix is singular.
int lup_decomposition_solve(float* x, const float* A, const int* p, const float* b, int n);

// Compute the determinant of a matrix A given the LUP decomposition of A encoded in A.
// This function will return a zero determinant if the matrix is singular.
double lup_decomposition_determinant(const float* A, const int* p, int n);

// Invert a matrix A given the encoded LUP decomposition of A.
// The final inverse matrix will be stored in the given matrix X.
// This function will return 1 if the matrix was sucessfully inverted, and 0 if the matrix A is singular.
int lup_decomposition_inverse(float* X, const float* A, const int* p, int n);

#define lup_decomposition_h // #if !defined(lup_decomposition_h)

#endif // #if !defined(lup_decomposition_h)

#if defined(lup_decomposition_implementation)

#if !defined (lup_decomposition_implementation_once)
#define lup_decomposition_implementation_once

float lup_decomposition_abs_float(float x) 
{ 
	return x < 0.0f ? -x : x; 
}

void lup_decomposition_swap_integers(int* a, int* b) 
{ 
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

void lup_decomposition_swap_floats(float* a, float* b) 
{ 
	float tmp = *a;
	*a = *b;
	*b = tmp;
}

// LUP factorization via Gaussian elimination.
void lup_decomposition_decompose(float* A, int* p, int n)
{
	// Initialize permutation.
	for (int i = 0; i < n; ++i)
	{
		p[i] = i;
	}

	// Loop over the diagonal elements trying to zero 
	// the elements in the i-th column below the i-th row.
	for (int i = 0; i < n; ++i)
	{
		// Perform pivoting.
		// This ensures the pivot element is a non-zero element.
		// This also ensures the current pivot element is the largest number in the current column for 
		// improving numerical stability.
		int max_row_index = i;
		float max_col_value = lup_decomposition_abs_float(A[i + n * i]);
		for (int i1 = i + 1; i1 < n; ++i1)
		{
			float value = lup_decomposition_abs_float(A[i1 + n * i]);
			if (value > max_col_value)
			{
				max_row_index = i1;
				max_col_value = value;
			}
		}

		if (max_col_value == 0.0f)
		{
			// The matrix is singular.
			continue;
		}

		if (max_row_index != i)
		{
			// Swap the rows.
			for (int j = 0; j < n; ++j)
			{
				lup_decomposition_swap_floats(A + i + n * j, A + max_row_index + n * j);
			}

			// Swap the permutations.
			lup_decomposition_swap_integers(p + i, p + max_row_index);
		}

		float a = A[i + n * i];
		float inv_a = 1.0f / a;

		// Eliminate each value in the pivot column below the pivot row.
		for (int i1 = i + 1; i1 < n; ++i1)
		{
			float k = inv_a * A[i1 + n * i];

			// Store the multiplier k in the subpart of A.
			A[i1 + n * i] = k;

			// Subtract the row skipping the columns which have the 
			// current multiplier and earlier multipliers stored in.
			for (int j = i + 1; j < n; ++j)
			{
				A[i1 + n * j] -= k * A[i + n * j];
			}
		}
	}
}

// Solve Ax = b given LUP factorization.
int lup_decomposition_solve(float* x, const float* A, const int* p, const float* b, int n)
{
	// Check singularity.
	for (int i = 0; i < n; ++i)
	{
		if (A[i + n * i] == 0.0f)
		{
			return 0;
		}
	}

	// Solve PAx = Pb
	
	// LUx = Pb
	// L(Ux) = Pb
	// Ly = Pb
	// y = L^-1 Pb

	// Ux = L^-1 Pb
	// Ux = y

	// Solve Ly = Pb using forward substitution storing 
	// the solution y in x as it goes.
	for (int i = 0; i < n; ++i)
	{
		// xi = Pbi
		int pi = p[i];
		x[i] = b[pi];

		float sum = 0.0f;
		for (int j = 0; j < i; ++j)
		{
			// This loop accesses only the elements above the current row.
			// Therefore, the diagonal is not accessed, so no special treatment is required.
			// It's okay to overwrite the solution xi in the current xi as it moves forward.
			sum += A[i + n * j] * x[j];
		}

		// This is actually
		// xi = (bi - s) / aii
		// Thus,
		// L(A) = I + L(PA)
		// aii = 1
		// xi = bi - s
		x[i] = x[i] - sum;
	}

	// Solve Ux = y via backward substitution storing 
	// the solution in x as it advances.
	// This loop was designed for unsigned integers so if using unsigned integers no 
	// loop modification is necessary.
	for (int ip = n; ip > 0; --ip)
	{
		int i = ip - 1;

		float sum = 0.0f;
		for (int j = i + 1; j < n; ++j)
		{
			// This loop accesses only the elements below the current row.
			// It's okay to overwrite the solution xi in the current xi as it moves backwards.
			// s = s + aij * xj
			sum += A[i + n * j] * x[j];
		}

		// xi = [bi - s] / aii
		x[i] = (x[i] - sum) / A[i + n * i];
	}

	// Success.
	return 1;
}

// Return the determinant given LUP factorization.
// It can be very large. This is why we double the floating-point precision.
double lup_decomposition_determinant(const float* A, const int* p, int n)
{
	// Sign according to permutation.
	double sign = 1.0;
	
	// det(A) = det(L) * det(U) 
	// det(L) = 1
	double det = 1.0;
	
	for (int i = 0; i < n; ++i)
	{
		if (p[i] != i)
		{
			sign = -sign;
		}

		det *= (double)(A[i + n * i]);
	}

	return sign * det;
}

// Solve AX = I.
void lup_decomposition_inverse_solve(float* x, const float* A, const int* p, int bi, int n)
{
	for (int i = 0; i < n; ++i)
	{
		int pi = p[i];
		
		// xi = bpi
		// bpi = 0, pi != bi
		// bpi = 1, pi == bi
		x[i] = (pi == bi) ? 1.0f : 0.0f;

		float sum = 0.0f;
		for (int j = 0; j < i; ++j)
		{
			sum += A[i + n * j] * x[j];
		}

		x[i] = x[i] - sum;
	}

	for (int ip = n; ip > 0; --ip)
	{
		int i = ip - 1;

		float sum = 0.0f;
		for (int j = i + 1; j < n; ++j)
		{
			sum += A[i + n * j] * x[j];
		}

		x[i] = (x[i] - sum) / A[i + n * i];
	}
}

// Matrix inverse via precomputed LUP decomposition.
int lup_decomposition_inverse(float* X, const float* A, const int* p, int n)
{
	// Check singularity.
	for (int i = 0; i < n; ++i)
	{
		if (A[i + n * i] == 0.0f)
		{
			// A is singular.
			return 0;
		}
	}

	// Solve AX = I. 
	// This loop is easely paralellizable.
	for (int j = 0; j < n; ++j)
	{
		// Thanks to column major storage we can do this.
		float* x = X + (0 + n * j);
		
		// Solve Ax = b with bi = j
		lup_decomposition_inverse_solve(x, A, p, j, n);
	}

	// Success.
	return 1;
}

#endif // #if !defined (lup_decomposition_implementation_once)

#endif // #if defined(lup_decomposition_implementation)
