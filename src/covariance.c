#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "covariance.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

#define RUN_ON_CPU

/*
Source code:
__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_M)
    {
        mean[j] = 0.0;

        int i;
        for(i = 0; i < _PB_N; i++)
        {
            mean[j] += data[i * M + j];
        }
        mean[j] /= (DATA_TYPE)FLOAT_N;
    }
}


__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_N) && (j < _PB_M))
    {
        data[i * M + j] -= mean[j];
    }
}


__global__ void covar_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data)
{
    int j1 = blockIdx.x * blockDim.x + threadIdx.x;
    int i, j2;

    if (j1 < _PB_M)
    {
        for (j2 = j1; j2 < _PB_M; j2++)
        {
            symmat[j1*M + j2] = 0.0;
            for(i = 0; i < _PB_N; i++)
            {
                symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
            }
            symmat[j2 * M + j1] = symmat[j1 * M + j2];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z11mean_kerneliiPfS_\n"
                                ""
                                ".visible .entry _Z11mean_kerneliiPfS_(\n"
                                "	.param .u32 _Z11mean_kerneliiPfS__param_0,\n"
                                "	.param .u32 _Z11mean_kerneliiPfS__param_1,\n"
                                "	.param .u64 _Z11mean_kerneliiPfS__param_2,\n"
                                "	.param .u64 _Z11mean_kerneliiPfS__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<30>;\n"
                                "	.reg .b32 	%r<33>;\n"
                                "	.reg .b64 	%rd<15>;\n"
                                ""
                                "	ld.param.u32 	%r16, [_Z11mean_kerneliiPfS__param_0];\n"
                                "	ld.param.u32 	%r15, [_Z11mean_kerneliiPfS__param_1];\n"
                                "	ld.param.u64 	%rd3, [_Z11mean_kerneliiPfS__param_2];\n"
                                "	ld.param.u64 	%rd4, [_Z11mean_kerneliiPfS__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd4;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r16;\n"
                                "	@%p1 bra 	BB0_12;\n"
                                ""
                                "	cvta.to.global.u64 	%rd5, %rd3;\n"
                                "	mul.wide.s32 	%rd6, %r4, 4;\n"
                                "	add.s64 	%rd2, %rd5, %rd6;\n"
                                "	mov.u32 	%r30, 0;\n"
                                "	st.global.u32 	[%rd2], %r30;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.lt.s32	%p2, %r15, 1;\n"
                                "	@%p2 bra 	BB0_11;\n"
                                ""
                                "	and.b32  	%r21, %r15, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.eq.s32	%p3, %r21, 0;\n"
                                "	@%p3 bra 	BB0_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r21, 1;\n"
                                "	@%p4 bra 	BB0_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r21, 2;\n"
                                "	@%p5 bra 	BB0_6;\n"
                                ""
                                "	add.s64 	%rd8, %rd1, %rd6;\n"
                                "	ld.global.f32 	%f14, [%rd8];\n"
                                "	add.f32 	%f29, %f14, 0f00000000;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	mov.u32 	%r30, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	shl.b32 	%r23, %r30, 11;\n"
                                "	add.s32 	%r24, %r23, %r4;\n"
                                "	mul.wide.s32 	%rd9, %r24, 4;\n"
                                "	add.s64 	%rd10, %rd1, %rd9;\n"
                                "	ld.global.f32 	%f15, [%rd10];\n"
                                "	add.f32 	%f29, %f15, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r30, %r30, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	shl.b32 	%r25, %r30, 11;\n"
                                "	add.s32 	%r26, %r25, %r4;\n"
                                "	mul.wide.s32 	%rd11, %r26, 4;\n"
                                "	add.s64 	%rd12, %rd1, %rd11;\n"
                                "	ld.global.f32 	%f16, [%rd12];\n"
                                "	add.f32 	%f29, %f16, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r30, %r30, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	setp.lt.u32	%p6, %r15, 4;\n"
                                "	@%p6 bra 	BB0_11;\n"
                                ""
                                "	mad.lo.s32 	%r31, %r30, 2048, %r4;\n"
                                ""
                                "BB0_10:\n"
                                "	mul.wide.s32 	%rd13, %r31, 4;\n"
                                "	add.s64 	%rd14, %rd1, %rd13;\n"
                                "	ld.global.f32 	%f17, [%rd14];\n"
                                "	add.f32 	%f18, %f17, %f29;\n"
                                "	st.global.f32 	[%rd2], %f18;\n"
                                "	ld.global.f32 	%f19, [%rd14+8192];\n"
                                "	add.f32 	%f20, %f19, %f18;\n"
                                "	st.global.f32 	[%rd2], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd14+16384];\n"
                                "	add.f32 	%f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd2], %f22;\n"
                                "	ld.global.f32 	%f23, [%rd14+24576];\n"
                                "	add.f32 	%f29, %f23, %f22;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r31, %r31, 8192;\n"
                                "	add.s32 	%r30, %r30, 4;\n"
                                "	setp.lt.s32	%p7, %r30, %r15;\n"
                                "	@%p7 bra 	BB0_10;\n"
                                ""
                                "BB0_11:\n"
                                "	div.rn.f32 	%f24, %f29, 0f4A442E10;\n"
                                "	st.global.f32 	[%rd2], %f24;\n"
                                ""
                                "BB0_12:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z13reduce_kerneliiPfS_\n"
                                ".visible .entry _Z13reduce_kerneliiPfS_(\n"
                                "	.param .u32 _Z13reduce_kerneliiPfS__param_0,\n"
                                "	.param .u32 _Z13reduce_kerneliiPfS__param_1,\n"
                                "	.param .u64 _Z13reduce_kerneliiPfS__param_2,\n"
                                "	.param .u64 _Z13reduce_kerneliiPfS__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<4>;\n"
                                "	.reg .f32 	%f<4>;\n"
                                "	.reg .b32 	%r<13>;\n"
                                "	.reg .b64 	%rd<9>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z13reduce_kerneliiPfS__param_0];\n"
                                "	ld.param.u32 	%r4, [_Z13reduce_kerneliiPfS__param_1];\n"
                                "	ld.param.u64 	%rd1, [_Z13reduce_kerneliiPfS__param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z13reduce_kerneliiPfS__param_3];\n"
                                "	mov.u32 	%r5, %ctaid.x;\n"
                                "	mov.u32 	%r6, %ntid.x;\n"
                                "	mov.u32 	%r7, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r6, %r5, %r7;\n"
                                "	mov.u32 	%r8, %ntid.y;\n"
                                "	mov.u32 	%r9, %ctaid.y;\n"
                                "	mov.u32 	%r10, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r8, %r9, %r10;\n"
                                "	setp.ge.s32	%p1, %r2, %r4;\n"
                                "	setp.ge.s32	%p2, %r1, %r3;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd1;\n"
                                "	mul.wide.s32 	%rd4, %r1, 4;\n"
                                "	add.s64 	%rd5, %rd3, %rd4;\n"
                                "	shl.b32 	%r11, %r2, 11;\n"
                                "	add.s32 	%r12, %r11, %r1;\n"
                                "	cvta.to.global.u64 	%rd6, %rd2;\n"
                                "	mul.wide.s32 	%rd7, %r12, 4;\n"
                                "	add.s64 	%rd8, %rd6, %rd7;\n"
                                "	ld.global.f32 	%f1, [%rd8];\n"
                                "	ld.global.f32 	%f2, [%rd5];\n"
                                "	sub.f32 	%f3, %f1, %f2;\n"
                                "	st.global.f32 	[%rd8], %f3;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z12covar_kerneliiPfS_\n"
                                ".visible .entry _Z12covar_kerneliiPfS_(\n"
                                "	.param .u32 _Z12covar_kerneliiPfS__param_0,\n"
                                "	.param .u32 _Z12covar_kerneliiPfS__param_1,\n"
                                "	.param .u64 _Z12covar_kerneliiPfS__param_2,\n"
                                "	.param .u64 _Z12covar_kerneliiPfS__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<9>;\n"
                                "	.reg .f32 	%f<36>;\n"
                                "	.reg .b32 	%r<47>;\n"
                                "	.reg .b64 	%rd<29>;\n"
                                ""
                                "	ld.param.u32 	%r23, [_Z12covar_kerneliiPfS__param_0];\n"
                                "	ld.param.u32 	%r24, [_Z12covar_kerneliiPfS__param_1];\n"
                                "	ld.param.u64 	%rd6, [_Z12covar_kerneliiPfS__param_2];\n"
                                "	ld.param.u64 	%rd5, [_Z12covar_kerneliiPfS__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd6;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r23;\n"
                                "	@%p1 bra 	BB2_13;\n"
                                ""
                                "	cvta.to.global.u64 	%rd7, %rd5;\n"
                                "	shl.b32 	%r5, %r4, 11;\n"
                                "	and.b32  	%r6, %r24, 3;\n"
                                "	mul.wide.s32 	%rd8, %r4, 4;\n"
                                "	add.s64 	%rd2, %rd7, %rd8;\n"
                                "	mov.u32 	%r40, %r4;\n"
                                ""
                                "BB2_2:\n"
                                "	add.s32 	%r25, %r40, %r5;\n"
                                "	mul.wide.s32 	%rd9, %r25, 4;\n"
                                "	add.s64 	%rd3, %rd1, %rd9;\n"
                                "	mov.u32 	%r43, 0;\n"
                                "	st.global.u32 	[%rd3], %r43;\n"
                                "	mov.f32 	%f35, 0f00000000;\n"
                                "	setp.lt.s32	%p2, %r24, 1;\n"
                                "	@%p2 bra 	BB2_12;\n"
                                ""
                                "	mov.f32 	%f35, 0f00000000;\n"
                                "	setp.eq.s32	%p3, %r6, 0;\n"
                                "	@%p3 bra 	BB2_9;\n"
                                ""
                                "	setp.eq.s32	%p4, %r6, 1;\n"
                                "	@%p4 bra 	BB2_8;\n"
                                ""
                                "	setp.eq.s32	%p5, %r6, 2;\n"
                                "	@%p5 bra 	BB2_7;\n"
                                ""
                                "	ld.global.f32 	%f14, [%rd2];\n"
                                "	mul.wide.s32 	%rd11, %r40, 4;\n"
                                "	add.s64 	%rd12, %rd7, %rd11;\n"
                                "	ld.global.f32 	%f15, [%rd12];\n"
                                "	fma.rn.f32 	%f35, %f14, %f15, 0f00000000;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	mov.u32 	%r43, 1;\n"
                                ""
                                "BB2_7:\n"
                                "	shl.b32 	%r31, %r43, 11;\n"
                                "	add.s32 	%r32, %r31, %r4;\n"
                                "	mul.wide.s32 	%rd14, %r32, 4;\n"
                                "	add.s64 	%rd15, %rd7, %rd14;\n"
                                "	add.s32 	%r33, %r31, %r40;\n"
                                "	mul.wide.s32 	%rd16, %r33, 4;\n"
                                "	add.s64 	%rd17, %rd7, %rd16;\n"
                                "	ld.global.f32 	%f16, [%rd17];\n"
                                "	ld.global.f32 	%f17, [%rd15];\n"
                                "	fma.rn.f32 	%f35, %f17, %f16, %f35;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	add.s32 	%r43, %r43, 1;\n"
                                ""
                                "BB2_8:\n"
                                "	shl.b32 	%r34, %r43, 11;\n"
                                "	add.s32 	%r35, %r34, %r4;\n"
                                "	mul.wide.s32 	%rd19, %r35, 4;\n"
                                "	add.s64 	%rd20, %rd7, %rd19;\n"
                                "	add.s32 	%r36, %r34, %r40;\n"
                                "	mul.wide.s32 	%rd21, %r36, 4;\n"
                                "	add.s64 	%rd22, %rd7, %rd21;\n"
                                "	ld.global.f32 	%f18, [%rd22];\n"
                                "	ld.global.f32 	%f19, [%rd20];\n"
                                "	fma.rn.f32 	%f35, %f19, %f18, %f35;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	add.s32 	%r43, %r43, 1;\n"
                                ""
                                "BB2_9:\n"
                                "	setp.lt.u32	%p6, %r24, 4;\n"
                                "	@%p6 bra 	BB2_12;\n"
                                ""
                                "	shl.b32 	%r37, %r43, 11;\n"
                                "	add.s32 	%r45, %r4, %r37;\n"
                                "	add.s32 	%r44, %r40, %r37;\n"
                                ""
                                "BB2_11:\n"
                                "	mul.wide.s32 	%rd23, %r45, 4;\n"
                                "	add.s64 	%rd24, %rd7, %rd23;\n"
                                "	mul.wide.s32 	%rd25, %r44, 4;\n"
                                "	add.s64 	%rd26, %rd7, %rd25;\n"
                                "	ld.global.f32 	%f20, [%rd26];\n"
                                "	ld.global.f32 	%f21, [%rd24];\n"
                                "	fma.rn.f32 	%f22, %f21, %f20, %f35;\n"
                                "	st.global.f32 	[%rd3], %f22;\n"
                                "	ld.global.f32 	%f23, [%rd26+8192];\n"
                                "	ld.global.f32 	%f24, [%rd24+8192];\n"
                                "	fma.rn.f32 	%f25, %f24, %f23, %f22;\n"
                                "	st.global.f32 	[%rd3], %f25;\n"
                                "	ld.global.f32 	%f26, [%rd26+16384];\n"
                                "	ld.global.f32 	%f27, [%rd24+16384];\n"
                                "	fma.rn.f32 	%f28, %f27, %f26, %f25;\n"
                                "	st.global.f32 	[%rd3], %f28;\n"
                                "	ld.global.f32 	%f29, [%rd26+24576];\n"
                                "	ld.global.f32 	%f30, [%rd24+24576];\n"
                                "	fma.rn.f32 	%f35, %f30, %f29, %f28;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	add.s32 	%r45, %r45, 8192;\n"
                                "	add.s32 	%r44, %r44, 8192;\n"
                                "	add.s32 	%r43, %r43, 4;\n"
                                "	setp.lt.s32	%p7, %r43, %r24;\n"
                                "	@%p7 bra 	BB2_11;\n"
                                ""
                                "BB2_12:\n"
                                "	shl.b32 	%r38, %r40, 11;\n"
                                "	add.s32 	%r39, %r38, %r4;\n"
                                "	mul.wide.s32 	%rd27, %r39, 4;\n"
                                "	add.s64 	%rd28, %rd1, %rd27;\n"
                                "	st.global.f32 	[%rd28], %f35;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                "	setp.lt.s32	%p8, %r40, %r23;\n"
                                "	@%p8 bra 	BB2_2;\n"
                                ""
                                "BB2_13:\n"
                                "	ret;\n"
                                "}\n";
void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            data[i][j] = ((DATA_TYPE)i * j) / M;
        }
    }
}

void covariance(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m), DATA_TYPE POLYBENCH_1D(mean, M, m))
{
    int i, j, j1, j2;

    /* Determine mean of column vectors of input data matrix */
    for (j = 0; j < _PB_M; j++)
    {
        mean[j] = 0.0;
        for (i = 0; i < _PB_N; i++)
        {
            mean[j] += data[i][j];
        }
        mean[j] /= FLOAT_N;
    }

    /* Center the column vectors. */
    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < _PB_M; j++)
        {
            data[i][j] -= mean[j];
        }
    }

    /* Calculate the m * m covariance matrix. */
    for (j1 = 0; j1 < _PB_M; j1++)
    {
        for (j2 = j1; j2 < _PB_M; j2++)
        {
            symmat[j1][j2] = 0.0;
            for (i = 0; i < _PB_N; i++)
            {
                symmat[j1][j2] += data[i][j1] * data[i][j2];
            }
            symmat[j2][j1] = symmat[j1][j2];
        }
    }
}

void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, M, m, m))
{
    int i, j, fail;
    fail = 0;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void covarianceCuda(CUdevice device, int m, int n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m), DATA_TYPE POLYBENCH_1D(mean,M,m), 
		DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu,M,M,m,m))
{
    CUdeviceptr data_gpu, mean_gpu, symmat_gpu;
    
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3=NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&data_gpu, sizeof(DATA_TYPE) * M * N));
    cuError(cuMemAlloc(&symmat_gpu, sizeof(DATA_TYPE) * M * M));
    cuError(cuMemAlloc(&mean_gpu, sizeof(DATA_TYPE) * M));
    cuError(cuMemcpyHtoD(data_gpu, data, sizeof(DATA_TYPE) * M * N));
    cuError(cuMemcpyHtoD(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * M));
    cuError(cuMemcpyHtoD(mean_gpu, mean, sizeof(DATA_TYPE) * M));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z11mean_kerneliiPfS_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z13reduce_kerneliiPfS_"));
    cuError(cuModuleGetFunction(&func3, module, "_Z12covar_kerneliiPfS_"));

    unsigned grid1_x = (size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X));
    unsigned grid2_x = (size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X));
    unsigned grid2_y = (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X));
    unsigned grid3_x = (size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X));

    void *args1[] = {&m,&n,&mean_gpu,&data_gpu, NULL};
    void *args2[] = {&m,&n,&symmat_gpu,&data_gpu, NULL};
    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid1_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func2, grid2_x, grid2_y, 1, DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func3, grid3_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y, 1, 0, NULL, args2, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N));
	
    cuError(cuMemFree(data_gpu));
    cuError(cuMemFree(symmat_gpu));
    cuError(cuMemFree(mean_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,M,m,m);	

	init_arrays(m, n, POLYBENCH_ARRAY(data));
    
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting covariance on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        covarianceCuda(device, m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(symmat_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test covariance on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		covariance(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);	

  	return 0;
}
#include "../include/polybench.c"
