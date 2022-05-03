#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "2mm.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
Source code:
__global__ void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NJ))
    {
        tmp[i * NJ + j] = 0;
        int k;
        for (k = 0; k < _PB_NK; k++)
        {
            tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
        }
    }
}


__global__ void mm2_kernel2(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *C, DATA_TYPE *D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NL))
    {
        D[i * NL + j] *= beta;
        int k;
        for (k = 0; k < _PB_NJ; k++)
        {
            D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z11mm2_kernel1iiiiffPfS_S_\n"
                                ""
                                ".visible .entry _Z11mm2_kernel1iiiiffPfS_S_(\n"
                                "	.param .u32 _Z11mm2_kernel1iiiiffPfS_S__param_0,\n"
                                "	.param .u32 _Z11mm2_kernel1iiiiffPfS_S__param_1,\n"
                                "	.param .u32 _Z11mm2_kernel1iiiiffPfS_S__param_2,\n"
                                "	.param .u32 _Z11mm2_kernel1iiiiffPfS_S__param_3,\n"
                                "	.param .f32 _Z11mm2_kernel1iiiiffPfS_S__param_4,\n"
                                "	.param .f32 _Z11mm2_kernel1iiiiffPfS_S__param_5,\n"
                                "	.param .u64 _Z11mm2_kernel1iiiiffPfS_S__param_6,\n"
                                "	.param .u64 _Z11mm2_kernel1iiiiffPfS_S__param_7,\n"
                                "	.param .u64 _Z11mm2_kernel1iiiiffPfS_S__param_8\n"
                                "){\n"
                                "	.reg .pred 	%p<10>;\n"
                                "	.reg .f32 	%f<41>;\n"
                                "	.reg .b32 	%r<45>;\n"
                                "	.reg .b64 	%rd<31>;\n"
                                ""
                                "	ld.param.u32 	%r21, [_Z11mm2_kernel1iiiiffPfS_S__param_0];\n"
                                "	ld.param.u32 	%r22, [_Z11mm2_kernel1iiiiffPfS_S__param_1];\n"
                                "	ld.param.u32 	%r20, [_Z11mm2_kernel1iiiiffPfS_S__param_2];\n"
                                "	ld.param.f32 	%f9, [_Z11mm2_kernel1iiiiffPfS_S__param_4];\n"
                                "	ld.param.u64 	%rd7, [_Z11mm2_kernel1iiiiffPfS_S__param_6];\n"
                                "	ld.param.u64 	%rd9, [_Z11mm2_kernel1iiiiffPfS_S__param_7];\n"
                                "	ld.param.u64 	%rd8, [_Z11mm2_kernel1iiiiffPfS_S__param_8];\n"
                                "	cvta.to.global.u64 	%rd1, %rd9;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	mov.u32 	%r5, %ntid.y;\n"
                                "	mov.u32 	%r6, %ctaid.y;\n"
                                "	mov.u32 	%r7, %tid.y;\n"
                                "	mad.lo.s32 	%r8, %r5, %r6, %r7;\n"
                                "	setp.ge.s32	%p1, %r8, %r21;\n"
                                "	setp.ge.s32	%p2, %r4, %r22;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd10, %rd7;\n"
                                "	shl.b32 	%r9, %r8, 10;\n"
                                "	add.s32 	%r23, %r9, %r4;\n"
                                "	mul.wide.s32 	%rd11, %r23, 4;\n"
                                "	add.s64 	%rd2, %rd10, %rd11;\n"
                                "	mov.u32 	%r40, 0;\n"
                                "	st.global.u32 	[%rd2], %r40;\n"
                                "	setp.lt.s32	%p4, %r20, 1;\n"
                                "	@%p4 bra 	BB0_11;\n"
                                ""
                                "	and.b32  	%r28, %r20, 3;\n"
                                "	mov.f32 	%f37, 0f00000000;\n"
                                "	setp.eq.s32	%p5, %r28, 0;\n"
                                "	@%p5 bra 	BB0_8;\n"
                                ""
                                "	setp.eq.s32	%p6, %r28, 1;\n"
                                "	@%p6 bra 	BB0_7;\n"
                                ""
                                "	setp.eq.s32	%p7, %r28, 2;\n"
                                "	@%p7 bra 	BB0_6;\n"
                                ""
                                "	mul.wide.s32 	%rd12, %r9, 4;\n"
                                "	add.s64 	%rd13, %rd1, %rd12;\n"
                                "	ld.global.f32 	%f13, [%rd13];\n"
                                "	mul.f32 	%f14, %f13, %f9;\n"
                                "	cvta.to.global.u64 	%rd14, %rd8;\n"
                                "	mul.wide.s32 	%rd15, %r4, 4;\n"
                                "	add.s64 	%rd16, %rd14, %rd15;\n"
                                "	ld.global.f32 	%f15, [%rd16];\n"
                                "	fma.rn.f32 	%f37, %f14, %f15, 0f00000000;\n"
                                "	st.global.f32 	[%rd2], %f37;\n"
                                "	mov.u32 	%r40, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	add.s32 	%r30, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd17, %r30, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	ld.global.f32 	%f16, [%rd18];\n"
                                "	mul.f32 	%f17, %f16, %f9;\n"
                                "	shl.b32 	%r31, %r40, 10;\n"
                                "	add.s32 	%r32, %r31, %r4;\n"
                                "	cvta.to.global.u64 	%rd19, %rd8;\n"
                                "	mul.wide.s32 	%rd20, %r32, 4;\n"
                                "	add.s64 	%rd21, %rd19, %rd20;\n"
                                "	ld.global.f32 	%f18, [%rd21];\n"
                                "	fma.rn.f32 	%f37, %f17, %f18, %f37;\n"
                                "	st.global.f32 	[%rd2], %f37;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	add.s32 	%r33, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd22, %r33, 4;\n"
                                "	add.s64 	%rd23, %rd1, %rd22;\n"
                                "	ld.global.f32 	%f19, [%rd23];\n"
                                "	mul.f32 	%f20, %f19, %f9;\n"
                                "	shl.b32 	%r34, %r40, 10;\n"
                                "	add.s32 	%r35, %r34, %r4;\n"
                                "	cvta.to.global.u64 	%rd24, %rd8;\n"
                                "	mul.wide.s32 	%rd25, %r35, 4;\n"
                                "	add.s64 	%rd26, %rd24, %rd25;\n"
                                "	ld.global.f32 	%f21, [%rd26];\n"
                                "	fma.rn.f32 	%f37, %f20, %f21, %f37;\n"
                                "	st.global.f32 	[%rd2], %f37;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	setp.lt.u32	%p8, %r20, 4;\n"
                                "	@%p8 bra 	BB0_11;\n"
                                ""
                                "	mad.lo.s32 	%r43, %r40, 1024, %r4;\n"
                                "	mul.lo.s32 	%r37, %r5, %r6;\n"
                                "	mad.lo.s32 	%r38, %r37, 1024, %r40;\n"
                                "	mad.lo.s32 	%r39, %r7, 1024, %r38;\n"
                                "	mul.wide.s32 	%rd27, %r39, 4;\n"
                                "	add.s64 	%rd30, %rd1, %rd27;\n"
                                "	cvta.to.global.u64 	%rd4, %rd8;\n"
                                ""
                                "BB0_10:\n"
                                "	ld.global.f32 	%f22, [%rd30];\n"
                                "	mul.f32 	%f23, %f22, %f9;\n"
                                "	mul.wide.s32 	%rd28, %r43, 4;\n"
                                "	add.s64 	%rd29, %rd4, %rd28;\n"
                                "	ld.global.f32 	%f24, [%rd29];\n"
                                "	fma.rn.f32 	%f25, %f23, %f24, %f37;\n"
                                "	st.global.f32 	[%rd2], %f25;\n"
                                "	ld.global.f32 	%f26, [%rd30+4];\n"
                                "	mul.f32 	%f27, %f26, %f9;\n"
                                "	ld.global.f32 	%f28, [%rd29+4096];\n"
                                "	fma.rn.f32 	%f29, %f27, %f28, %f25;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	ld.global.f32 	%f30, [%rd30+8];\n"
                                "	mul.f32 	%f31, %f30, %f9;\n"
                                "	ld.global.f32 	%f32, [%rd29+8192];\n"
                                "	fma.rn.f32 	%f33, %f31, %f32, %f29;\n"
                                "	st.global.f32 	[%rd2], %f33;\n"
                                "	ld.global.f32 	%f34, [%rd30+12];\n"
                                "	mul.f32 	%f35, %f34, %f9;\n"
                                "	ld.global.f32 	%f36, [%rd29+12288];\n"
                                "	fma.rn.f32 	%f37, %f35, %f36, %f33;\n"
                                "	st.global.f32 	[%rd2], %f37;\n"
                                "	add.s32 	%r43, %r43, 4096;\n"
                                "	add.s64 	%rd30, %rd30, 16;\n"
                                "	add.s32 	%r40, %r40, 4;\n"
                                "	setp.lt.s32	%p9, %r40, %r20;\n"
                                "	@%p9 bra 	BB0_10;\n"
                                ""
                                "BB0_11:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11mm2_kernel2iiiiffPfS_S_\n"
                                ".visible .entry _Z11mm2_kernel2iiiiffPfS_S_(\n"
                                "	.param .u32 _Z11mm2_kernel2iiiiffPfS_S__param_0,\n"
                                "	.param .u32 _Z11mm2_kernel2iiiiffPfS_S__param_1,\n"
                                "	.param .u32 _Z11mm2_kernel2iiiiffPfS_S__param_2,\n"
                                "	.param .u32 _Z11mm2_kernel2iiiiffPfS_S__param_3,\n"
                                "	.param .f32 _Z11mm2_kernel2iiiiffPfS_S__param_4,\n"
                                "	.param .f32 _Z11mm2_kernel2iiiiffPfS_S__param_5,\n"
                                "	.param .u64 _Z11mm2_kernel2iiiiffPfS_S__param_6,\n"
                                "	.param .u64 _Z11mm2_kernel2iiiiffPfS_S__param_7,\n"
                                "	.param .u64 _Z11mm2_kernel2iiiiffPfS_S__param_8\n"
                                "){\n"
                                "	.reg .pred 	%p<10>;\n"
                                "	.reg .f32 	%f<33>;\n"
                                "	.reg .b32 	%r<45>;\n"
                                "	.reg .b64 	%rd<31>;\n"
                                ""
                                "	ld.param.u32 	%r20, [_Z11mm2_kernel2iiiiffPfS_S__param_0];\n"
                                "	ld.param.u32 	%r19, [_Z11mm2_kernel2iiiiffPfS_S__param_1];\n"
                                "	ld.param.u32 	%r21, [_Z11mm2_kernel2iiiiffPfS_S__param_3];\n"
                                "	ld.param.f32 	%f10, [_Z11mm2_kernel2iiiiffPfS_S__param_5];\n"
                                "	ld.param.u64 	%rd9, [_Z11mm2_kernel2iiiiffPfS_S__param_6];\n"
                                "	ld.param.u64 	%rd7, [_Z11mm2_kernel2iiiiffPfS_S__param_7];\n"
                                "	ld.param.u64 	%rd8, [_Z11mm2_kernel2iiiiffPfS_S__param_8];\n"
                                "	cvta.to.global.u64 	%rd1, %rd9;\n"
                                "	mov.u32 	%r22, %ntid.x;\n"
                                "	mov.u32 	%r1, %ctaid.x;\n"
                                "	mov.u32 	%r2, %tid.x;\n"
                                "	mad.lo.s32 	%r3, %r22, %r1, %r2;\n"
                                "	mov.u32 	%r4, %ntid.y;\n"
                                "	mov.u32 	%r5, %ctaid.y;\n"
                                "	mov.u32 	%r6, %tid.y;\n"
                                "	mad.lo.s32 	%r7, %r4, %r5, %r6;\n"
                                "	setp.ge.s32	%p1, %r7, %r20;\n"
                                "	setp.ge.s32	%p2, %r3, %r21;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB1_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd10, %rd8;\n"
                                "	shl.b32 	%r8, %r7, 10;\n"
                                "	add.s32 	%r23, %r8, %r3;\n"
                                "	mul.wide.s32 	%rd11, %r23, 4;\n"
                                "	add.s64 	%rd2, %rd10, %rd11;\n"
                                "	ld.global.f32 	%f11, [%rd2];\n"
                                "	mul.f32 	%f1, %f11, %f10;\n"
                                "	st.global.f32 	[%rd2], %f1;\n"
                                "	setp.lt.s32	%p4, %r19, 1;\n"
                                "	@%p4 bra 	BB1_11;\n"
                                ""
                                "	and.b32  	%r27, %r19, 3;\n"
                                "	mov.u32 	%r40, 0;\n"
                                "	setp.eq.s32	%p5, %r27, 0;\n"
                                "	@%p5 bra 	BB1_8;\n"
                                ""
                                "	setp.eq.s32	%p6, %r27, 1;\n"
                                "	@%p6 bra 	BB1_7;\n"
                                ""
                                "	setp.eq.s32	%p7, %r27, 2;\n"
                                "	@%p7 bra 	BB1_6;\n"
                                ""
                                "	mul.wide.s32 	%rd12, %r8, 4;\n"
                                "	add.s64 	%rd13, %rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd14, %rd7;\n"
                                "	mul.wide.s32 	%rd15, %r3, 4;\n"
                                "	add.s64 	%rd16, %rd14, %rd15;\n"
                                "	ld.global.f32 	%f12, [%rd16];\n"
                                "	ld.global.f32 	%f13, [%rd13];\n"
                                "	fma.rn.f32 	%f1, %f13, %f12, %f1;\n"
                                "	st.global.f32 	[%rd2], %f1;\n"
                                "	mov.u32 	%r40, 1;\n"
                                ""
                                "BB1_6:\n"
                                "	add.s32 	%r29, %r40, %r8;\n"
                                "	mul.wide.s32 	%rd17, %r29, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	shl.b32 	%r30, %r40, 10;\n"
                                "	add.s32 	%r31, %r30, %r3;\n"
                                "	cvta.to.global.u64 	%rd19, %rd7;\n"
                                "	mul.wide.s32 	%rd20, %r31, 4;\n"
                                "	add.s64 	%rd21, %rd19, %rd20;\n"
                                "	ld.global.f32 	%f14, [%rd21];\n"
                                "	ld.global.f32 	%f15, [%rd18];\n"
                                "	fma.rn.f32 	%f1, %f15, %f14, %f1;\n"
                                "	st.global.f32 	[%rd2], %f1;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB1_7:\n"
                                "	add.s32 	%r32, %r40, %r8;\n"
                                "	mul.wide.s32 	%rd22, %r32, 4;\n"
                                "	add.s64 	%rd23, %rd1, %rd22;\n"
                                "	shl.b32 	%r33, %r40, 10;\n"
                                "	add.s32 	%r34, %r33, %r3;\n"
                                "	cvta.to.global.u64 	%rd24, %rd7;\n"
                                "	mul.wide.s32 	%rd25, %r34, 4;\n"
                                "	add.s64 	%rd26, %rd24, %rd25;\n"
                                "	ld.global.f32 	%f16, [%rd26];\n"
                                "	ld.global.f32 	%f17, [%rd23];\n"
                                "	fma.rn.f32 	%f1, %f17, %f16, %f1;\n"
                                "	st.global.f32 	[%rd2], %f1;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB1_8:\n"
                                "	setp.lt.u32	%p8, %r19, 4;\n"
                                "	@%p8 bra 	BB1_11;\n"
                                ""
                                "	mad.lo.s32 	%r43, %r40, 1024, %r3;\n"
                                "	mul.lo.s32 	%r37, %r4, %r5;\n"
                                "	mad.lo.s32 	%r38, %r37, 1024, %r40;\n"
                                "	mad.lo.s32 	%r39, %r6, 1024, %r38;\n"
                                "	mul.wide.s32 	%rd27, %r39, 4;\n"
                                "	add.s64 	%rd30, %rd1, %rd27;\n"
                                "	cvta.to.global.u64 	%rd4, %rd7;\n"
                                ""
                                "BB1_10:\n"
                                "	mul.wide.s32 	%rd28, %r43, 4;\n"
                                "	add.s64 	%rd29, %rd4, %rd28;\n"
                                "	ld.global.f32 	%f18, [%rd29];\n"
                                "	ld.global.f32 	%f19, [%rd30];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f1;\n"
                                "	st.global.f32 	[%rd2], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd29+4096];\n"
                                "	ld.global.f32 	%f22, [%rd30+4];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd2], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd29+8192];\n"
                                "	ld.global.f32 	%f25, [%rd30+8];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd2], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd29+12288];\n"
                                "	ld.global.f32 	%f28, [%rd30+12];\n"
                                "	fma.rn.f32 	%f1, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd2], %f1;\n"
                                "	add.s32 	%r43, %r43, 4096;\n"
                                "	add.s64 	%rd30, %rd30, 16;\n"
                                "	add.s32 	%r40, %r40, 4;\n"
                                "	setp.lt.s32	%p9, %r40, %r19;\n"
                                "	@%p9 bra 	BB1_10;\n"
                                ""
                                "BB1_11:\n"
                                "	ret;\n"
                                "}"

void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), 
		DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj), 
		DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
	int i, j;
	*alpha = 32412;
	*beta = 2123;
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}
	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
	for (i = 0; i < nl; i++)
	{
		for (j = 0; j < nj; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;
	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			if (percentDiff(D[i][j], D_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void mm2_cpu(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
	int i, j, k;
	
	/* D := alpha*A*B*C + beta*D */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			tmp[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				tmp[i][j] += alpha * A[i][k] * B[k][j];
			}
		}
	}

	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			D[i][j] *= beta;
			for (k = 0; k < _PB_NJ; ++k)
			{
				D[i][j] += tmp[i][k] * C[k][j];
			}
		}
	}
}

void mm2Cuda(CUdevice device, int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), 
	DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), 
	DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu,NI,NL,ni,nl))
{
    CUdeviceptr tmp_gpu, A_gpu, B_gpu, C_gpu, D_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&tmp_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK));
    cuError(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
    cuError(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NL * NJ));
    cuError(cuMemAlloc(&D_gpu, sizeof(DATA_TYPE) * NI * NL));

    cuError(cuMemcpyHtoD(tmp_gpu, tmp, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NK));
    cuError(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ));
    cuError(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NL * NJ));
    cuError(cuMemcpyHtoD(D_gpu, D, sizeof(DATA_TYPE) * NI * NL));

	cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z11mm2_kernel1iiiiffPfS_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z11mm2_kernel2iiiiffPfS_S_"));

    unsigned grid1_x = (size_t)ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_X);
    unsigned grid1_y = (size_t)ceil( ((float)NI) / ((float)DIM_THREAD_BLOCK_Y));
    unsigned grid2_x = (size_t)ceil( ((float)NL) / ((float)DIM_THREAD_BLOCK_X) );
    unsigned grid2_y = (size_t)ceil( ((float)NI) / ((float)DIM_THREAD_BLOCK_Y));

    void *args1[] = {&ni, &nj, &nk, &nl, &alpha, &beta, &tmp_gpu, &A_gpu, &B_gpu};
    void *args2[] = {&ni, &nj, &nk, &nl, &alpha, &beta, &tmp_gpu, &C_gpu, &D_gpu};
    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid1_x, grid1_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
	cuError(cuLaunchKernel(func2, grid2_x, grid2_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * NI * NL));

    cuError(cuMemFree(tmp_gpu));
    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
    cuError(cuMemFree(C_gpu));
    cuError(cuMemFree(D_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
	POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
	POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu,DATA_TYPE,NI,NL,ni,nl);
	
	/* Initialize array(s). */
  	init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting 2mm on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
	    mm2Cuda(device, ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test 2mm on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(D_outputFromGpu);

  	return 0;
}

#include "../include/polybench.c"