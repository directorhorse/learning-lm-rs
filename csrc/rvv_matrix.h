#ifndef __RVV_MATRIX
#define __RVV_MATRIX

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <riscv_vector.h>
void matrix_transpose_multiply_basic(const float *A, const float *B, float *C, float beta, float alpha,
                                    int M, int N, int K);
                                    
#endif