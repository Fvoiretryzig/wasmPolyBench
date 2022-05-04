#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>

#include "3mm.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

/*
Source Code:
__global__ void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NJ))
    {
        E[i * NJ + j] = 0;
        int k;
        for(k=0; k < _PB_NK; k++)
        {
            E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
        }
    }
}


__global__ void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NJ) && (j < _PB_NL))
    {
        F[i * NL + j] = 0;
        int k;
        for(k=0; k < _PB_NM; k++)
        {
            F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
        }
    }
}


__global__ void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NL))
    {
        G[i * NL + j] = 0;
        int k;
        for(k=0; k < _PB_NJ; k++)
        {
            G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z11mm3_kernel1iiiiiPfS_S_\n"
                                ""
                                ".visible .entry _Z11mm3_kernel1iiiiiPfS_S_(\n"
                                "	.param .u32 _Z11mm3_kernel1iiiiiPfS_S__param_0,\n"
                                "	.param .u32 _Z11mm3_kernel1iiiiiPfS_S__param_1,\n"
                                "	.param .u32 _Z11mm3_kernel1iiiiiPfS_S__param_2,\n"
                                "	.param .u32 _Z11mm3_kernel1iiiiiPfS_S__param_3,\n"
                                "	.param .u32 _Z11mm3_kernel1iiiiiPfS_S__param_4,\n"
                                "	.param .u64 _Z11mm3_kernel1iiiiiPfS_S__param_5,\n"
                                "	.param .u64 _Z11mm3_kernel1iiiiiPfS_S__param_6,\n"
                                "	.param .u64 _Z11mm3_kernel1iiiiiPfS_S__param_7\n"
                                "){\n"
                                "	.reg .pred 	%p<10>;\n"
                                "	.reg .f32 	%f<33>;\n"
                                "	.reg .b32 	%r<45>;\n"
                                "	.reg .b64 	%rd<31>;\n"
                                ""
                                "	ld.param.u32 	%r21, [_Z11mm3_kernel1iiiiiPfS_S__param_0];\n"
                                "	ld.param.u32 	%r22, [_Z11mm3_kernel1iiiiiPfS_S__param_1];\n"
                                "	ld.param.u32 	%r20, [_Z11mm3_kernel1iiiiiPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd9, [_Z11mm3_kernel1iiiiiPfS_S__param_5];\n"
                                "	ld.param.u64 	%rd7, [_Z11mm3_kernel1iiiiiPfS_S__param_6];\n"
                                "	ld.param.u64 	%rd8, [_Z11mm3_kernel1iiiiiPfS_S__param_7];\n"
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
                                "	cvta.to.global.u64 	%rd10, %rd8;\n"
                                "	shl.b32 	%r9, %r8, 9;\n"
                                "	add.s32 	%r23, %r9, %r4;\n"
                                "	mul.wide.s32 	%rd11, %r23, 4;\n"
                                "	add.s64 	%rd2, %rd10, %rd11;\n"
                                "	mov.u32 	%r40, 0;\n"
                                "	st.global.u32 	[%rd2], %r40;\n"
                                "	setp.lt.s32	%p4, %r20, 1;\n"
                                "	@%p4 bra 	BB0_11;\n"
                                ""
                                "	and.b32  	%r28, %r20, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
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
                                "	cvta.to.global.u64 	%rd14, %rd7;\n"
                                "	mul.wide.s32 	%rd15, %r4, 4;\n"
                                "	add.s64 	%rd16, %rd14, %rd15;\n"
                                "	ld.global.f32 	%f12, [%rd16];\n"
                                "	ld.global.f32 	%f13, [%rd13];\n"
                                "	fma.rn.f32 	%f29, %f13, %f12, 0f00000000;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	mov.u32 	%r40, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	add.s32 	%r30, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd17, %r30, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	shl.b32 	%r31, %r40, 9;\n"
                                "	add.s32 	%r32, %r31, %r4;\n"
                                "	cvta.to.global.u64 	%rd19, %rd7;\n"
                                "	mul.wide.s32 	%rd20, %r32, 4;\n"
                                "	add.s64 	%rd21, %rd19, %rd20;\n"
                                "	ld.global.f32 	%f14, [%rd21];\n"
                                "	ld.global.f32 	%f15, [%rd18];\n"
                                "	fma.rn.f32 	%f29, %f15, %f14, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	add.s32 	%r33, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd22, %r33, 4;\n"
                                "	add.s64 	%rd23, %rd1, %rd22;\n"
                                "	shl.b32 	%r34, %r40, 9;\n"
                                "	add.s32 	%r35, %r34, %r4;\n"
                                "	cvta.to.global.u64 	%rd24, %rd7;\n"
                                "	mul.wide.s32 	%rd25, %r35, 4;\n"
                                "	add.s64 	%rd26, %rd24, %rd25;\n"
                                "	ld.global.f32 	%f16, [%rd26];\n"
                                "	ld.global.f32 	%f17, [%rd23];\n"
                                "	fma.rn.f32 	%f29, %f17, %f16, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	setp.lt.u32	%p8, %r20, 4;\n"
                                "	@%p8 bra 	BB0_11;\n"
                                ""
                                "	mad.lo.s32 	%r43, %r40, 512, %r4;\n"
                                "	mul.lo.s32 	%r37, %r5, %r6;\n"
                                "	mad.lo.s32 	%r38, %r37, 512, %r40;\n"
                                "	mad.lo.s32 	%r39, %r7, 512, %r38;\n"
                                "	mul.wide.s32 	%rd27, %r39, 4;\n"
                                "	add.s64 	%rd30, %rd1, %rd27;\n"
                                "	cvta.to.global.u64 	%rd4, %rd7;\n"
                                ""
                                "BB0_10:\n"
                                "	mul.wide.s32 	%rd28, %r43, 4;\n"
                                "	add.s64 	%rd29, %rd4, %rd28;\n"
                                "	ld.global.f32 	%f18, [%rd29];\n"
                                "	ld.global.f32 	%f19, [%rd30];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f29;\n"
                                "	st.global.f32 	[%rd2], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd29+2048];\n"
                                "	ld.global.f32 	%f22, [%rd30+4];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd2], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd29+4096];\n"
                                "	ld.global.f32 	%f25, [%rd30+8];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd2], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd29+6144];\n"
                                "	ld.global.f32 	%f28, [%rd30+12];\n"
                                "	fma.rn.f32 	%f29, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r43, %r43, 2048;\n"
                                "	add.s64 	%rd30, %rd30, 16;\n"
                                "	add.s32 	%r40, %r40, 4;\n"
                                "	setp.lt.s32	%p9, %r40, %r20;\n"
                                "	@%p9 bra 	BB0_10;\n"
                                ""
                                "BB0_11:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11mm3_kernel2iiiiiPfS_S_\n"
                                ".visible .entry _Z11mm3_kernel2iiiiiPfS_S_(\n"
                                "	.param .u32 _Z11mm3_kernel2iiiiiPfS_S__param_0,\n"
                                "	.param .u32 _Z11mm3_kernel2iiiiiPfS_S__param_1,\n"
                                "	.param .u32 _Z11mm3_kernel2iiiiiPfS_S__param_2,\n"
                                "	.param .u32 _Z11mm3_kernel2iiiiiPfS_S__param_3,\n"
                                "	.param .u32 _Z11mm3_kernel2iiiiiPfS_S__param_4,\n"
                                "	.param .u64 _Z11mm3_kernel2iiiiiPfS_S__param_5,\n"
                                "	.param .u64 _Z11mm3_kernel2iiiiiPfS_S__param_6,\n"
                                "	.param .u64 _Z11mm3_kernel2iiiiiPfS_S__param_7\n"
                                "){\n"
                                "	.reg .pred 	%p<10>;\n"
                                "	.reg .f32 	%f<33>;\n"
                                "	.reg .b32 	%r<45>;\n"
                                "	.reg .b64 	%rd<31>;\n"
                                ""
                                "	ld.param.u32 	%r21, [_Z11mm3_kernel2iiiiiPfS_S__param_1];\n"
                                "	ld.param.u32 	%r22, [_Z11mm3_kernel2iiiiiPfS_S__param_3];\n"
                                "	ld.param.u32 	%r20, [_Z11mm3_kernel2iiiiiPfS_S__param_4];\n"
                                "	ld.param.u64 	%rd9, [_Z11mm3_kernel2iiiiiPfS_S__param_5];\n"
                                "	ld.param.u64 	%rd7, [_Z11mm3_kernel2iiiiiPfS_S__param_6];\n"
                                "	ld.param.u64 	%rd8, [_Z11mm3_kernel2iiiiiPfS_S__param_7];\n"
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
                                "	@%p3 bra 	BB1_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd10, %rd8;\n"
                                "	shl.b32 	%r9, %r8, 9;\n"
                                "	add.s32 	%r23, %r9, %r4;\n"
                                "	mul.wide.s32 	%rd11, %r23, 4;\n"
                                "	add.s64 	%rd2, %rd10, %rd11;\n"
                                "	mov.u32 	%r40, 0;\n"
                                "	st.global.u32 	[%rd2], %r40;\n"
                                "	setp.lt.s32	%p4, %r20, 1;\n"
                                "	@%p4 bra 	BB1_11;\n"
                                ""
                                "	and.b32  	%r28, %r20, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.eq.s32	%p5, %r28, 0;\n"
                                "	@%p5 bra 	BB1_8;\n"
                                ""
                                "	setp.eq.s32	%p6, %r28, 1;\n"
                                "	@%p6 bra 	BB1_7;\n"
                                ""
                                "	setp.eq.s32	%p7, %r28, 2;\n"
                                "	@%p7 bra 	BB1_6;\n"
                                ""
                                "	mul.wide.s32 	%rd12, %r9, 4;\n"
                                "	add.s64 	%rd13, %rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd14, %rd7;\n"
                                "	mul.wide.s32 	%rd15, %r4, 4;\n"
                                "	add.s64 	%rd16, %rd14, %rd15;\n"
                                "	ld.global.f32 	%f12, [%rd16];\n"
                                "	ld.global.f32 	%f13, [%rd13];\n"
                                "	fma.rn.f32 	%f29, %f13, %f12, 0f00000000;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	mov.u32 	%r40, 1;\n"
                                ""
                                "BB1_6:\n"
                                "	add.s32 	%r30, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd17, %r30, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	shl.b32 	%r31, %r40, 9;\n"
                                "	add.s32 	%r32, %r31, %r4;\n"
                                "	cvta.to.global.u64 	%rd19, %rd7;\n"
                                "	mul.wide.s32 	%rd20, %r32, 4;\n"
                                "	add.s64 	%rd21, %rd19, %rd20;\n"
                                "	ld.global.f32 	%f14, [%rd21];\n"
                                "	ld.global.f32 	%f15, [%rd18];\n"
                                "	fma.rn.f32 	%f29, %f15, %f14, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB1_7:\n"
                                "	add.s32 	%r33, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd22, %r33, 4;\n"
                                "	add.s64 	%rd23, %rd1, %rd22;\n"
                                "	shl.b32 	%r34, %r40, 9;\n"
                                "	add.s32 	%r35, %r34, %r4;\n"
                                "	cvta.to.global.u64 	%rd24, %rd7;\n"
                                "	mul.wide.s32 	%rd25, %r35, 4;\n"
                                "	add.s64 	%rd26, %rd24, %rd25;\n"
                                "	ld.global.f32 	%f16, [%rd26];\n"
                                "	ld.global.f32 	%f17, [%rd23];\n"
                                "	fma.rn.f32 	%f29, %f17, %f16, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB1_8:\n"
                                "	setp.lt.u32	%p8, %r20, 4;\n"
                                "	@%p8 bra 	BB1_11;\n"
                                ""
                                "	mad.lo.s32 	%r43, %r40, 512, %r4;\n"
                                "	mul.lo.s32 	%r37, %r5, %r6;\n"
                                "	mad.lo.s32 	%r38, %r37, 512, %r40;\n"
                                "	mad.lo.s32 	%r39, %r7, 512, %r38;\n"
                                "	mul.wide.s32 	%rd27, %r39, 4;\n"
                                "	add.s64 	%rd30, %rd1, %rd27;\n"
                                "	cvta.to.global.u64 	%rd4, %rd7;\n"
                                ""
                                "BB1_10:\n"
                                "	mul.wide.s32 	%rd28, %r43, 4;\n"
                                "	add.s64 	%rd29, %rd4, %rd28;\n"
                                "	ld.global.f32 	%f18, [%rd29];\n"
                                "	ld.global.f32 	%f19, [%rd30];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f29;\n"
                                "	st.global.f32 	[%rd2], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd29+2048];\n"
                                "	ld.global.f32 	%f22, [%rd30+4];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd2], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd29+4096];\n"
                                "	ld.global.f32 	%f25, [%rd30+8];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd2], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd29+6144];\n"
                                "	ld.global.f32 	%f28, [%rd30+12];\n"
                                "	fma.rn.f32 	%f29, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r43, %r43, 2048;\n"
                                "	add.s64 	%rd30, %rd30, 16;\n"
                                "	add.s32 	%r40, %r40, 4;\n"
                                "	setp.lt.s32	%p9, %r40, %r20;\n"
                                "	@%p9 bra 	BB1_10;\n"
                                ""
                                "BB1_11:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11mm3_kernel3iiiiiPfS_S_\n"
                                ".visible .entry _Z11mm3_kernel3iiiiiPfS_S_(\n"
                                "	.param .u32 _Z11mm3_kernel3iiiiiPfS_S__param_0,\n"
                                "	.param .u32 _Z11mm3_kernel3iiiiiPfS_S__param_1,\n"
                                "	.param .u32 _Z11mm3_kernel3iiiiiPfS_S__param_2,\n"
                                "	.param .u32 _Z11mm3_kernel3iiiiiPfS_S__param_3,\n"
                                "	.param .u32 _Z11mm3_kernel3iiiiiPfS_S__param_4,\n"
                                "	.param .u64 _Z11mm3_kernel3iiiiiPfS_S__param_5,\n"
                                "	.param .u64 _Z11mm3_kernel3iiiiiPfS_S__param_6,\n"
                                "	.param .u64 _Z11mm3_kernel3iiiiiPfS_S__param_7\n"
                                "){\n"
                                "	.reg .pred 	%p<10>;\n"
                                "	.reg .f32 	%f<33>;\n"
                                "	.reg .b32 	%r<45>;\n"
                                "	.reg .b64 	%rd<31>;\n"
                                ""
                                "	ld.param.u32 	%r21, [_Z11mm3_kernel3iiiiiPfS_S__param_0];\n"
                                "	ld.param.u32 	%r20, [_Z11mm3_kernel3iiiiiPfS_S__param_1];\n"
                                "	ld.param.u32 	%r22, [_Z11mm3_kernel3iiiiiPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd9, [_Z11mm3_kernel3iiiiiPfS_S__param_5];\n"
                                "	ld.param.u64 	%rd7, [_Z11mm3_kernel3iiiiiPfS_S__param_6];\n"
                                "	ld.param.u64 	%rd8, [_Z11mm3_kernel3iiiiiPfS_S__param_7];\n"
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
                                "	@%p3 bra 	BB2_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd10, %rd8;\n"
                                "	shl.b32 	%r9, %r8, 9;\n"
                                "	add.s32 	%r23, %r9, %r4;\n"
                                "	mul.wide.s32 	%rd11, %r23, 4;\n"
                                "	add.s64 	%rd2, %rd10, %rd11;\n"
                                "	mov.u32 	%r40, 0;\n"
                                "	st.global.u32 	[%rd2], %r40;\n"
                                "	setp.lt.s32	%p4, %r20, 1;\n"
                                "	@%p4 bra 	BB2_11;\n"
                                ""
                                "	and.b32  	%r28, %r20, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.eq.s32	%p5, %r28, 0;\n"
                                "	@%p5 bra 	BB2_8;\n"
                                ""
                                "	setp.eq.s32	%p6, %r28, 1;\n"
                                "	@%p6 bra 	BB2_7;\n"
                                ""
                                "	setp.eq.s32	%p7, %r28, 2;\n"
                                "	@%p7 bra 	BB2_6;\n"
                                ""
                                "	mul.wide.s32 	%rd12, %r9, 4;\n"
                                "	add.s64 	%rd13, %rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd14, %rd7;\n"
                                "	mul.wide.s32 	%rd15, %r4, 4;\n"
                                "	add.s64 	%rd16, %rd14, %rd15;\n"
                                "	ld.global.f32 	%f12, [%rd16];\n"
                                "	ld.global.f32 	%f13, [%rd13];\n"
                                "	fma.rn.f32 	%f29, %f13, %f12, 0f00000000;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	mov.u32 	%r40, 1;\n"
                                ""
                                "BB2_6:\n"
                                "	add.s32 	%r30, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd17, %r30, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	shl.b32 	%r31, %r40, 9;\n"
                                "	add.s32 	%r32, %r31, %r4;\n"
                                "	cvta.to.global.u64 	%rd19, %rd7;\n"
                                "	mul.wide.s32 	%rd20, %r32, 4;\n"
                                "	add.s64 	%rd21, %rd19, %rd20;\n"
                                "	ld.global.f32 	%f14, [%rd21];\n"
                                "	ld.global.f32 	%f15, [%rd18];\n"
                                "	fma.rn.f32 	%f29, %f15, %f14, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB2_7:\n"
                                "	add.s32 	%r33, %r40, %r9;\n"
                                "	mul.wide.s32 	%rd22, %r33, 4;\n"
                                "	add.s64 	%rd23, %rd1, %rd22;\n"
                                "	shl.b32 	%r34, %r40, 9;\n"
                                "	add.s32 	%r35, %r34, %r4;\n"
                                "	cvta.to.global.u64 	%rd24, %rd7;\n"
                                "	mul.wide.s32 	%rd25, %r35, 4;\n"
                                "	add.s64 	%rd26, %rd24, %rd25;\n"
                                "	ld.global.f32 	%f16, [%rd26];\n"
                                "	ld.global.f32 	%f17, [%rd23];\n"
                                "	fma.rn.f32 	%f29, %f17, %f16, %f29;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB2_8:\n"
                                "	setp.lt.u32	%p8, %r20, 4;\n"
                                "	@%p8 bra 	BB2_11;\n"
                                ""
                                "	mad.lo.s32 	%r43, %r40, 512, %r4;\n"
                                "	mul.lo.s32 	%r37, %r5, %r6;\n"
                                "	mad.lo.s32 	%r38, %r37, 512, %r40;\n"
                                "	mad.lo.s32 	%r39, %r7, 512, %r38;\n"
                                "	mul.wide.s32 	%rd27, %r39, 4;\n"
                                "	add.s64 	%rd30, %rd1, %rd27;\n"
                                "	cvta.to.global.u64 	%rd4, %rd7;\n"
                                ""
                                "BB2_10:\n"
                                "	mul.wide.s32 	%rd28, %r43, 4;\n"
                                "	add.s64 	%rd29, %rd4, %rd28;\n"
                                "	ld.global.f32 	%f18, [%rd29];\n"
                                "	ld.global.f32 	%f19, [%rd30];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f29;\n"
                                "	st.global.f32 	[%rd2], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd29+2048];\n"
                                "	ld.global.f32 	%f22, [%rd30+4];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd2], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd29+4096];\n"
                                "	ld.global.f32 	%f25, [%rd30+8];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd2], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd29+6144];\n"
                                "	ld.global.f32 	%f28, [%rd30+12];\n"
                                "	fma.rn.f32 	%f29, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd2], %f29;\n"
                                "	add.s32 	%r43, %r43, 2048;\n"
                                "	add.s64 	%rd30, %rd30, 16;\n"
                                "	add.s32 	%r40, %r40, 4;\n"
                                "	setp.lt.s32	%p9, %r40, %r20;\n"
                                "	@%p9 bra 	BB2_10;\n"
                                ""
                                "BB2_11:\n"
                                "	ret;\n"
                                "}\n";
void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm), DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
    int i, j;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / ni;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nm; j++)
        {
            C[i][j] = ((DATA_TYPE)i * (j + 3)) / nl;
        }
    }

    for (i = 0; i < nm; i++)
    {
        for (j = 0; j < nl; j++)
        {
            D[i][j] = ((DATA_TYPE)i * (j + 2)) / nk;
        }
    }
}

void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
    int i, j, fail;
    fail = 0;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nl; j++)
        {
            if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void mm3_cpu(int ni, int nj, int nk, int nl, int nm,
             DATA_TYPE POLYBENCH_2D(E, NI, NJ, ni, nj),
             DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
             DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
             DATA_TYPE POLYBENCH_2D(F, NJ, NL, nj, nl),
             DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
             DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl),
             DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl))
{
    int i, j, k;
    /* E := A*B */
    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NJ; j++)
        {
            E[i][j] = 0;
            for (k = 0; k < _PB_NK; ++k)
            {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    /* F := C*D */
    for (i = 0; i < _PB_NJ; i++)
    {
        for (j = 0; j < _PB_NL; j++)
        {
            F[i][j] = 0;
            for (k = 0; k < _PB_NM; ++k)
            {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }
    /* G := E*F */
    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NL; j++)
        {
            G[i][j] = 0;
            for (k = 0; k < _PB_NJ; ++k)
            {
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }
}
void mm3Cuda(CUdevice device, int ni, int nj, int nk, int nl, int nm,
             DATA_TYPE POLYBENCH_2D(E, NI, NJ, ni, nj),
             DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
             DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
             DATA_TYPE POLYBENCH_2D(F, NJ, NL, nj, nl),
             DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
             DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl),
             DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl),
             DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
    CUdeviceptr A_gpu, B_gpu, C_gpu, D_gpu, E_gpu, F_gpu, G_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK));
    cuError(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
    cuError(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NJ * NM));
    cuError(cuMemAlloc(&D_gpu, sizeof(DATA_TYPE) * NM * NL));
    cuError(cuMemAlloc(&E_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&F_gpu, sizeof(DATA_TYPE) * NJ * NL));
    cuError(cuMemAlloc(&G_gpu, sizeof(DATA_TYPE) * NI * NL));

    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NK));
    cuError(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ));
    cuError(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM));
    cuError(cuMemcpyHtoD(D_gpu, D, sizeof(DATA_TYPE) * NM * NL));
    cuError(cuMemcpyHtoD(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL));
    cuError(cuMemcpyHtoD(G_gpu, G, sizeof(DATA_TYPE) * NI * NL));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z11mm3_kernel1iiiiiPfS_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z11mm3_kernel2iiiiiPfS_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z11mm3_kernel3iiiiiPfS_S_"));

    unsigned grid1_x = (size_t)(ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X)));
    unsigned grid1_y = (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y)));
    unsigned grid2_x = (size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X)));
    unsigned grid2_y = (size_t)(ceil((float)NJ / ((float)DIM_THREAD_BLOCK_Y)));
    unsigned grid3_x = (size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X)));
    unsigned grid3_y = (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y)));

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid1((size_t)(ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X))), (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))));
    dim3 grid2((size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))), (size_t)(ceil((float)NJ / ((float)DIM_THREAD_BLOCK_Y))));
    dim3 grid3((size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))), (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))));

    /* Start timer. */
    polybench_start_instruments;
    void *args1[] = {&ni, &nj, &nk, &nl, &nm, &A_gpu, &B_gpu, &E_gpu, NULL};
    void *args2[] = {&ni, &nj, &nk, &nl, &nm, &C_gpu, &D_gpu, &F_gpu, NULL};
    void *args3[] = {&ni, &nj, &nk, &nl, &nm, &E_gpu, &F_gpu, &G_gpu, NULL};
    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid1_x, grid1_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func2, grid2_x, grid2_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
    cuError(cuLaunchKernel(func3, grid3_x, grid3_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args3, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL));

    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
    cuError(cuMemFree(C_gpu));
    cuError(cuMemFree(D_gpu));
    cuError(cuMemFree(E_gpu));
    cuError(cuMemFree(F_gpu));
    cuError(cuMemFree(G_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main(int argc, char **argv)
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    int nm = NM;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
    POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
    POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

    init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

    int deviceCount = 0;
    CUdevice device = 0;
    char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        fprintf(stdout, "\nTesting mm3 on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        mm3Cuda(device, ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E),
                POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test mm3 on GPU device %d Success\n", i);
    }
#ifdef RUN_ON_CPU
    polybench_start_instruments;
    SET_TIME(CPU_START)
    mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E),
            POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G));
    SET_TIME(CPU_END)
    fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
    compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));
#else
    print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(E);
    POLYBENCH_FREE_ARRAY(F);
    POLYBENCH_FREE_ARRAY(G);
    POLYBENCH_FREE_ARRAY(G_outputFromGpu);

    return 0;
}

#include "../include/polybench.c"
