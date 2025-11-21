#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>
#include "rvv_matrix.h"

void print_matrix(const float *matrix, size_t rows, size_t cols, const char *name) {
    printf("%s (%zux%zu):\n", name, rows, cols);
    for (size_t i = 0; i < rows && i < 4; i++) {
        for (size_t j = 0; j < cols && j < 4; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_transpose_multiply_scalar(const float *A, const float *B, float *C,float beta,float alpha,
                                     int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];  // B[j][k] 而不是 B[k][j]
            }
            C[i * N + j] = beta*C[i * N + j] + alpha*sum;
        }
    }
}

void matrix_transpose_multiply_basic(const float *A, const float *B, float *C, float beta, float alpha,
                                    int M, int N, int K) {
    // print_matrix(A,M,K,"A");
    // print_matrix(B,N,K,"B");
    // print_matrix(C,M,N,"C");
    // printf("beta is %f,alpha is %f\n",beta,alpha);
    int vl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            // 内积: C[i][j] = sum(A[i][k] * B[j][k]) for k=0..K-1
            vfloat32m1_t vsum = __riscv_vfmv_s_f_f32m1(0.0f, vl);
            for (int k = 0; k < K; k += vl) {
                vl = __riscv_vsetvl_e32m8(K - k);
                
                // 加载 A 的第 i 行片段和 B 的第 j 行片段
                vfloat32m8_t va = __riscv_vle32_v_f32m8(&A[i * K + k], vl);
                vfloat32m8_t vb = __riscv_vle32_v_f32m8(&B[j * K + k], vl);
                
                // 乘积累加
                vfloat32m8_t vprod = __riscv_vfmul_vv_f32m8(va, vb, vl);
                vsum = __riscv_vfredusum_vs_f32m8_f32m1(vprod, vsum,vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(vsum);
            C[i * N + j] = beta*C[i * N + j] +alpha*sum;
        }
    }
    // print_matrix(A,M,K,"A");
    // print_matrix(B,N,K,"B");
    // print_matrix(C,M,N,"C");
}