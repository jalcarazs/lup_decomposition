# LUP Decomposition

Decompose a matrix into LUP form.
The original code by Irlan Robson presented a bug when calculating the determinant of a matrix.
Sometimes, the code returned the correct determinant but with an incorrect sign.
For example in a 3x3 matrix like this:
[1,      1,     0.067]
[-1.23, -1.23, -0.173]
[-1.34, -1.95, -0.856]
it returned 0.054 instead of -0.054.
The code has been corrected, and now the sign of the determinant is calculated based on the variable
num_swaps, which gives the number of row interchanges performed in the matrix by the procedure "lup_decomposition_decompose".

The changes have been made in the files "lup_decomposition.h" and "main.c"

