#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

#include "gemver.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

/*
Source Code:
__global__ void gemver_kernel1(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *v1, DATA_TYPE *v2, DATA_TYPE *u1, DATA_TYPE *u2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_N) && (j < _PB_N))
    {
        a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
    }
}


__global__ void gemver_kernel2(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for(j = 0; j < _PB_N; j++)
        {
            x[i] += beta * a[j * N + i] * y[j];
        }
        x[i] += z[i];
    }
}


__global__ void gemver_kernel3(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 0) && (i < _PB_N))
    {
        int j;
        for(j = 0; j < _PB_N; j++)
        {
            w[i] += alpha * a[i*N + j] * x[j];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z14gemver_kernel1iffPfS_S_S_S_\n"
                                ""
                                ".visible .entry _Z14gemver_kernel1iffPfS_S_S_S_(\n"
                                "	.param .u32 _Z14gemver_kernel1iffPfS_S_S_S__param_0,\n"
                                "	.param .f32 _Z14gemver_kernel1iffPfS_S_S_S__param_1,\n"
                                "	.param .f32 _Z14gemver_kernel1iffPfS_S_S_S__param_2,\n"
                                "	.param .u64 _Z14gemver_kernel1iffPfS_S_S_S__param_3,\n"
                                "	.param .u64 _Z14gemver_kernel1iffPfS_S_S_S__param_4,\n"
                                "	.param .u64 _Z14gemver_kernel1iffPfS_S_S_S__param_5,\n"
                                "	.param .u64 _Z14gemver_kernel1iffPfS_S_S_S__param_6,\n"
                                "	.param .u64 _Z14gemver_kernel1iffPfS_S_S_S__param_7\n"
                                "){\n"
                                "	.reg .pred 	%p<4>;\n"
                                "	.reg .f32 	%f<9>;\n"
                                "	.reg .b32 	%r<12>;\n"
                                "	.reg .b64 	%rd<19>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z14gemver_kernel1iffPfS_S_S_S__param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z14gemver_kernel1iffPfS_S_S_S__param_3];\n"
                                "	ld.param.u64 	%rd2, [_Z14gemver_kernel1iffPfS_S_S_S__param_4];\n"
                                "	ld.param.u64 	%rd3, [_Z14gemver_kernel1iffPfS_S_S_S__param_5];\n"
                                "	ld.param.u64 	%rd4, [_Z14gemver_kernel1iffPfS_S_S_S__param_6];\n"
                                "	ld.param.u64 	%rd5, [_Z14gemver_kernel1iffPfS_S_S_S__param_7];\n"
                                "	mov.u32 	%r4, %ctaid.x;\n"
                                "	mov.u32 	%r5, %ntid.x;\n"
                                "	mov.u32 	%r6, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r5, %r4, %r6;\n"
                                "	mov.u32 	%r7, %ntid.y;\n"
                                "	mov.u32 	%r8, %ctaid.y;\n"
                                "	mov.u32 	%r9, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r7, %r8, %r9;\n"
                                "	setp.ge.s32	%p1, %r2, %r3;\n"
                                "	setp.ge.s32	%p2, %r1, %r3;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd6, %rd4;\n"
                                "	mul.wide.s32 	%rd7, %r2, 4;\n"
                                "	add.s64 	%rd8, %rd6, %rd7;\n"
                                "	cvta.to.global.u64 	%rd9, %rd2;\n"
                                "	mul.wide.s32 	%rd10, %r1, 4;\n"
                                "	add.s64 	%rd11, %rd9, %rd10;\n"
                                "	ld.global.f32 	%f1, [%rd11];\n"
                                "	ld.global.f32 	%f2, [%rd8];\n"
                                "	cvta.to.global.u64 	%rd12, %rd5;\n"
                                "	add.s64 	%rd13, %rd12, %rd7;\n"
                                "	cvta.to.global.u64 	%rd14, %rd3;\n"
                                "	add.s64 	%rd15, %rd14, %rd10;\n"
                                "	ld.global.f32 	%f3, [%rd15];\n"
                                "	ld.global.f32 	%f4, [%rd13];\n"
                                "	mul.f32 	%f5, %f4, %f3;\n"
                                "	fma.rn.f32 	%f6, %f2, %f1, %f5;\n"
                                "	shl.b32 	%r10, %r2, 12;\n"
                                "	add.s32 	%r11, %r10, %r1;\n"
                                "	cvta.to.global.u64 	%rd16, %rd1;\n"
                                "	mul.wide.s32 	%rd17, %r11, 4;\n"
                                "	add.s64 	%rd18, %rd16, %rd17;\n"
                                "	ld.global.f32 	%f7, [%rd18];\n"
                                "	add.f32 	%f8, %f7, %f6;\n"
                                "	st.global.f32 	[%rd18], %f8;\n"
                                ""
                                "BB0_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z14gemver_kernel2iffPfS_S_S_\n"
                                ".visible .entry _Z14gemver_kernel2iffPfS_S_S_(\n"
                                "	.param .u32 _Z14gemver_kernel2iffPfS_S_S__param_0,\n"
                                "	.param .f32 _Z14gemver_kernel2iffPfS_S_S__param_1,\n"
                                "	.param .f32 _Z14gemver_kernel2iffPfS_S_S__param_2,\n"
                                "	.param .u64 _Z14gemver_kernel2iffPfS_S_S__param_3,\n"
                                "	.param .u64 _Z14gemver_kernel2iffPfS_S_S__param_4,\n"
                                "	.param .u64 _Z14gemver_kernel2iffPfS_S_S__param_5,\n"
                                "	.param .u64 _Z14gemver_kernel2iffPfS_S_S__param_6\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<43>;\n"
                                "	.reg .b32 	%r<31>;\n"
                                "	.reg .b64 	%rd<30>;\n"
                                ""
                                "	ld.param.u32 	%r15, [_Z14gemver_kernel2iffPfS_S_S__param_0];\n"
                                "	ld.param.f32 	%f10, [_Z14gemver_kernel2iffPfS_S_S__param_2];\n"
                                "	ld.param.u64 	%rd9, [_Z14gemver_kernel2iffPfS_S_S__param_3];\n"
                                "	ld.param.u64 	%rd7, [_Z14gemver_kernel2iffPfS_S_S__param_4];\n"
                                "	ld.param.u64 	%rd10, [_Z14gemver_kernel2iffPfS_S_S__param_5];\n"
                                "	ld.param.u64 	%rd8, [_Z14gemver_kernel2iffPfS_S_S__param_6];\n"
                                "	cvta.to.global.u64 	%rd1, %rd10;\n"
                                "	cvta.to.global.u64 	%rd2, %rd9;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r15;\n"
                                "	@%p1 bra 	BB1_14;\n"
                                ""
                                "	cvta.to.global.u64 	%rd11, %rd7;\n"
                                "	mul.wide.s32 	%rd12, %r4, 4;\n"
                                "	add.s64 	%rd3, %rd11, %rd12;\n"
                                "	setp.lt.s32	%p2, %r15, 1;\n"
                                "	@%p2 bra 	BB1_13;\n"
                                ""
                                "	and.b32  	%r17, %r15, 3;\n"
                                "	mov.u32 	%r30, 0;\n"
                                "	setp.eq.s32	%p3, %r17, 0;\n"
                                "	@%p3 bra 	BB1_10;\n"
                                ""
                                "	setp.eq.s32	%p4, %r17, 1;\n"
                                "	@%p4 bra 	BB1_6;\n"
                                "	bra.uni 	BB1_4;\n"
                                ""
                                "BB1_6:\n"
                                "	ld.global.f32 	%f41, [%rd3];\n"
                                "	mov.u32 	%r27, 0;\n"
                                "	bra.uni 	BB1_9;\n"
                                ""
                                "BB1_4:\n"
                                "	setp.ne.s32	%p5, %r17, 2;\n"
                                "	@%p5 bra 	BB1_7;\n"
                                ""
                                "	ld.global.f32 	%f40, [%rd3];\n"
                                "	mov.u32 	%r26, 0;\n"
                                "	bra.uni 	BB1_8;\n"
                                ""
                                "BB1_7:\n"
                                "	add.s64 	%rd14, %rd2, %rd12;\n"
                                "	ld.global.f32 	%f11, [%rd14];\n"
                                "	mul.f32 	%f12, %f11, %f10;\n"
                                "	ld.global.f32 	%f13, [%rd1];\n"
                                "	ld.global.f32 	%f14, [%rd3];\n"
                                "	fma.rn.f32 	%f40, %f12, %f13, %f14;\n"
                                "	st.global.f32 	[%rd3], %f40;\n"
                                "	mov.u32 	%r26, 1;\n"
                                ""
                                "BB1_8:\n"
                                "	shl.b32 	%r21, %r26, 12;\n"
                                "	add.s32 	%r22, %r21, %r4;\n"
                                "	mul.wide.s32 	%rd15, %r22, 4;\n"
                                "	add.s64 	%rd16, %rd2, %rd15;\n"
                                "	ld.global.f32 	%f15, [%rd16];\n"
                                "	mul.f32 	%f16, %f15, %f10;\n"
                                "	mul.wide.u32 	%rd17, %r26, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	ld.global.f32 	%f17, [%rd18];\n"
                                "	fma.rn.f32 	%f41, %f16, %f17, %f40;\n"
                                "	st.global.f32 	[%rd3], %f41;\n"
                                "	add.s32 	%r27, %r26, 1;\n"
                                ""
                                "BB1_9:\n"
                                "	shl.b32 	%r23, %r27, 12;\n"
                                "	add.s32 	%r24, %r23, %r4;\n"
                                "	mul.wide.s32 	%rd19, %r24, 4;\n"
                                "	add.s64 	%rd20, %rd2, %rd19;\n"
                                "	ld.global.f32 	%f18, [%rd20];\n"
                                "	mul.f32 	%f19, %f18, %f10;\n"
                                "	mul.wide.s32 	%rd21, %r27, 4;\n"
                                "	add.s64 	%rd22, %rd1, %rd21;\n"
                                "	ld.global.f32 	%f20, [%rd22];\n"
                                "	fma.rn.f32 	%f21, %f19, %f20, %f41;\n"
                                "	st.global.f32 	[%rd3], %f21;\n"
                                "	add.s32 	%r30, %r27, 1;\n"
                                ""
                                "BB1_10:\n"
                                "	setp.lt.u32	%p6, %r15, 4;\n"
                                "	@%p6 bra 	BB1_13;\n"
                                ""
                                "	ld.global.f32 	%f42, [%rd3];\n"
                                "	mul.wide.s32 	%rd23, %r30, 4;\n"
                                "	add.s64 	%rd29, %rd1, %rd23;\n"
                                "	mad.lo.s32 	%r29, %r30, 4096, %r4;\n"
                                ""
                                "BB1_12:\n"
                                "	mul.wide.s32 	%rd24, %r29, 4;\n"
                                "	add.s64 	%rd25, %rd2, %rd24;\n"
                                "	ld.global.f32 	%f22, [%rd25];\n"
                                "	mul.f32 	%f23, %f22, %f10;\n"
                                "	ld.global.f32 	%f24, [%rd29];\n"
                                "	fma.rn.f32 	%f25, %f23, %f24, %f42;\n"
                                "	st.global.f32 	[%rd3], %f25;\n"
                                "	ld.global.f32 	%f26, [%rd25+16384];\n"
                                "	mul.f32 	%f27, %f26, %f10;\n"
                                "	ld.global.f32 	%f28, [%rd29+4];\n"
                                "	fma.rn.f32 	%f29, %f27, %f28, %f25;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	ld.global.f32 	%f30, [%rd25+32768];\n"
                                "	mul.f32 	%f31, %f30, %f10;\n"
                                "	ld.global.f32 	%f32, [%rd29+8];\n"
                                "	fma.rn.f32 	%f33, %f31, %f32, %f29;\n"
                                "	st.global.f32 	[%rd3], %f33;\n"
                                "	ld.global.f32 	%f34, [%rd25+49152];\n"
                                "	mul.f32 	%f35, %f34, %f10;\n"
                                "	ld.global.f32 	%f36, [%rd29+12];\n"
                                "	fma.rn.f32 	%f42, %f35, %f36, %f33;\n"
                                "	st.global.f32 	[%rd3], %f42;\n"
                                "	add.s64 	%rd29, %rd29, 16;\n"
                                "	add.s32 	%r29, %r29, 16384;\n"
                                "	add.s32 	%r30, %r30, 4;\n"
                                "	setp.lt.s32	%p7, %r30, %r15;\n"
                                "	@%p7 bra 	BB1_12;\n"
                                ""
                                "BB1_13:\n"
                                "	cvta.to.global.u64 	%rd26, %rd8;\n"
                                "	add.s64 	%rd28, %rd26, %rd12;\n"
                                "	ld.global.f32 	%f37, [%rd3];\n"
                                "	ld.global.f32 	%f38, [%rd28];\n"
                                "	add.f32 	%f39, %f38, %f37;\n"
                                "	st.global.f32 	[%rd3], %f39;\n"
                                ""
                                "BB1_14:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z14gemver_kernel3iffPfS_S_\n"
                                ".visible .entry _Z14gemver_kernel3iffPfS_S_(\n"
                                "	.param .u32 _Z14gemver_kernel3iffPfS_S__param_0,\n"
                                "	.param .f32 _Z14gemver_kernel3iffPfS_S__param_1,\n"
                                "	.param .f32 _Z14gemver_kernel3iffPfS_S__param_2,\n"
                                "	.param .u64 _Z14gemver_kernel3iffPfS_S__param_3,\n"
                                "	.param .u64 _Z14gemver_kernel3iffPfS_S__param_4,\n"
                                "	.param .u64 _Z14gemver_kernel3iffPfS_S__param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<11>;\n"
                                "	.reg .f32 	%f<40>;\n"
                                "	.reg .b32 	%r<28>;\n"
                                "	.reg .b64 	%rd<29>;\n"
                                ""
                                "	ld.param.u32 	%r13, [_Z14gemver_kernel3iffPfS_S__param_0];\n"
                                "	ld.param.f32 	%f10, [_Z14gemver_kernel3iffPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd11, [_Z14gemver_kernel3iffPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd12, [_Z14gemver_kernel3iffPfS_S__param_4];\n"
                                "	ld.param.u64 	%rd10, [_Z14gemver_kernel3iffPfS_S__param_5];\n"
                                "	cvta.to.global.u64 	%rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd2, %rd11;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.lt.s32	%p1, %r4, 0;\n"
                                "	setp.ge.s32	%p2, %r4, %r13;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	setp.lt.s32	%p4, %r13, 1;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	@%p5 bra 	BB2_12;\n"
                                ""
                                "	cvta.to.global.u64 	%rd13, %rd10;\n"
                                "	shl.b32 	%r5, %r4, 12;\n"
                                "	mul.wide.s32 	%rd14, %r4, 4;\n"
                                "	add.s64 	%rd3, %rd13, %rd14;\n"
                                "	and.b32  	%r15, %r13, 3;\n"
                                "	mov.u32 	%r27, 0;\n"
                                "	setp.eq.s32	%p6, %r15, 0;\n"
                                "	@%p6 bra 	BB2_9;\n"
                                ""
                                "	setp.eq.s32	%p7, %r15, 1;\n"
                                "	@%p7 bra 	BB2_5;\n"
                                "	bra.uni 	BB2_3;\n"
                                ""
                                "BB2_5:\n"
                                "	ld.global.f32 	%f38, [%rd3];\n"
                                "	mov.u32 	%r25, 0;\n"
                                "	bra.uni 	BB2_8;\n"
                                ""
                                "BB2_3:\n"
                                "	setp.ne.s32	%p8, %r15, 2;\n"
                                "	@%p8 bra 	BB2_6;\n"
                                ""
                                "	ld.global.f32 	%f37, [%rd3];\n"
                                "	mov.u32 	%r24, 0;\n"
                                "	bra.uni 	BB2_7;\n"
                                ""
                                "BB2_6:\n"
                                "	mul.wide.s32 	%rd15, %r5, 4;\n"
                                "	add.s64 	%rd16, %rd2, %rd15;\n"
                                "	ld.global.f32 	%f11, [%rd16];\n"
                                "	mul.f32 	%f12, %f11, %f10;\n"
                                "	ld.global.f32 	%f13, [%rd1];\n"
                                "	ld.global.f32 	%f14, [%rd3];\n"
                                "	fma.rn.f32 	%f37, %f12, %f13, %f14;\n"
                                "	st.global.f32 	[%rd3], %f37;\n"
                                "	mov.u32 	%r24, 1;\n"
                                ""
                                "BB2_7:\n"
                                "	add.s32 	%r19, %r24, %r5;\n"
                                "	mul.wide.s32 	%rd17, %r19, 4;\n"
                                "	add.s64 	%rd18, %rd2, %rd17;\n"
                                "	ld.global.f32 	%f15, [%rd18];\n"
                                "	mul.f32 	%f16, %f15, %f10;\n"
                                "	mul.wide.u32 	%rd19, %r24, 4;\n"
                                "	add.s64 	%rd20, %rd1, %rd19;\n"
                                "	ld.global.f32 	%f17, [%rd20];\n"
                                "	fma.rn.f32 	%f38, %f16, %f17, %f37;\n"
                                "	st.global.f32 	[%rd3], %f38;\n"
                                "	add.s32 	%r25, %r24, 1;\n"
                                ""
                                "BB2_8:\n"
                                "	add.s32 	%r20, %r25, %r5;\n"
                                "	mul.wide.s32 	%rd21, %r20, 4;\n"
                                "	add.s64 	%rd22, %rd2, %rd21;\n"
                                "	ld.global.f32 	%f18, [%rd22];\n"
                                "	mul.f32 	%f19, %f18, %f10;\n"
                                "	mul.wide.s32 	%rd23, %r25, 4;\n"
                                "	add.s64 	%rd24, %rd1, %rd23;\n"
                                "	ld.global.f32 	%f20, [%rd24];\n"
                                "	fma.rn.f32 	%f21, %f19, %f20, %f38;\n"
                                "	st.global.f32 	[%rd3], %f21;\n"
                                "	add.s32 	%r27, %r25, 1;\n"
                                ""
                                "BB2_9:\n"
                                "	setp.lt.u32	%p9, %r13, 4;\n"
                                "	@%p9 bra 	BB2_12;\n"
                                ""
                                "	ld.global.f32 	%f39, [%rd3];\n"
                                "	mul.wide.s32 	%rd25, %r27, 4;\n"
                                "	add.s64 	%rd28, %rd1, %rd25;\n"
                                "	mul.lo.s32 	%r21, %r1, %r2;\n"
                                "	mad.lo.s32 	%r22, %r21, 4096, %r27;\n"
                                "	mad.lo.s32 	%r23, %r3, 4096, %r22;\n"
                                "	mul.wide.s32 	%rd26, %r23, 4;\n"
                                "	add.s64 	%rd27, %rd2, %rd26;\n"
                                ""
                                "BB2_11:\n"
                                "	ld.global.f32 	%f22, [%rd27];\n"
                                "	mul.f32 	%f23, %f22, %f10;\n"
                                "	ld.global.f32 	%f24, [%rd28];\n"
                                "	fma.rn.f32 	%f25, %f23, %f24, %f39;\n"
                                "	st.global.f32 	[%rd3], %f25;\n"
                                "	ld.global.f32 	%f26, [%rd27+4];\n"
                                "	mul.f32 	%f27, %f26, %f10;\n"
                                "	ld.global.f32 	%f28, [%rd28+4];\n"
                                "	fma.rn.f32 	%f29, %f27, %f28, %f25;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	ld.global.f32 	%f30, [%rd27+8];\n"
                                "	mul.f32 	%f31, %f30, %f10;\n"
                                "	ld.global.f32 	%f32, [%rd28+8];\n"
                                "	fma.rn.f32 	%f33, %f31, %f32, %f29;\n"
                                "	st.global.f32 	[%rd3], %f33;\n"
                                "	ld.global.f32 	%f34, [%rd27+12];\n"
                                "	mul.f32 	%f35, %f34, %f10;\n"
                                "	ld.global.f32 	%f36, [%rd28+12];\n"
                                "	fma.rn.f32 	%f39, %f35, %f36, %f33;\n"
                                "	st.global.f32 	[%rd3], %f39;\n"
                                "	add.s64 	%rd28, %rd28, 16;\n"
                                "	add.s64 	%rd27, %rd27, 16;\n"
                                "	add.s32 	%r27, %r27, 4;\n"
                                "	setp.lt.s32	%p10, %r27, %r13;\n"
                                "	@%p10 bra 	BB2_11;\n"
                                ""
                                "BB2_12:\n"
                                "	ret;\n"
                                "}\n";
void gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(u1, N, n), DATA_TYPE POLYBENCH_1D(v1, N, n),
            DATA_TYPE POLYBENCH_1D(u2, N, n), DATA_TYPE POLYBENCH_1D(v2, N, n), DATA_TYPE POLYBENCH_1D(w, N, n), DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n),
            DATA_TYPE POLYBENCH_1D(z, N, n))
{
    int i, j;

    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < _PB_N; j++)
        {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < _PB_N; j++)
        {
            x[i] = x[i] + beta * A[j][i] * y[j];
        }
    }

    for (i = 0; i < _PB_N; i++)
    {
        x[i] = x[i] + z[i];
    }

    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < _PB_N; j++)
        {
            w[i] = w[i] + alpha * A[i][j] * x[j];
        }
    }
}

void init(int n, DATA_TYPE *alpha,
          DATA_TYPE *beta,
          DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
          DATA_TYPE POLYBENCH_1D(u1, N, n),
          DATA_TYPE POLYBENCH_1D(v1, N, n),
          DATA_TYPE POLYBENCH_1D(u2, N, n),
          DATA_TYPE POLYBENCH_1D(v2, N, n),
          DATA_TYPE POLYBENCH_1D(w, N, n),
          DATA_TYPE POLYBENCH_1D(x, N, n),
          DATA_TYPE POLYBENCH_1D(y, N, n),
          DATA_TYPE POLYBENCH_1D(z, N, n))
{
    int i, j;

    *alpha = 43532;
    *beta = 12313;

    for (i = 0; i < N; i++)
    {
        u1[i] = i;
        u2[i] = (i + 1) / N / 2.0;
        v1[i] = (i + 1) / N / 4.0;
        v2[i] = (i + 1) / N / 6.0;
        y[i] = (i + 1) / N / 8.0;
        z[i] = (i + 1) / N / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;

        for (j = 0; j < N; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_1D(w1, N, n), DATA_TYPE POLYBENCH_1D(w2, N, n))
{
    int i, fail;
    fail = 0;

    for (i = 0; i < N; i++)
    {
        if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
    }

    // Print results
    printf("Number of misses: %d\n", fail);
}
void gemverCuda(CUdevice device, int n, DATA_TYPE alpha, DATA_TYPE beta,
                DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                DATA_TYPE POLYBENCH_1D(u1, N, n),
                DATA_TYPE POLYBENCH_1D(v1, N, n),
                DATA_TYPE POLYBENCH_1D(u2, N, n),
                DATA_TYPE POLYBENCH_1D(v2, N, n),
                DATA_TYPE POLYBENCH_1D(w, N, n),
                DATA_TYPE POLYBENCH_1D(w_outputFromGpu, N, n),
                DATA_TYPE POLYBENCH_1D(x, N, n),
                DATA_TYPE POLYBENCH_1D(y, N, n),
                DATA_TYPE POLYBENCH_1D(z, N, n))
{
    CUdeviceptr A_gpu, x_gpu, y_gpu, z_gpu, v1_gpu, v2_gpu, u1_gpu, u2_gpu, w_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL;
    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemAlloc(&x_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&y_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&z_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&w_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&v1_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&v2_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&u1_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&u2_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemcpyHtoD(x_gpu, x, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(y_gpu, y, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(z_gpu, z, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(w_gpu, w, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(v1_gpu, v1, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(v2_gpu, v2, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(u1_gpu, u1, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(u2_gpu, u2, sizeof(DATA_TYPE) * N));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z14gemver_kernel1iffPfS_S_S_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z14gemver_kernel2iffPfS_S_S_"));
    cuError(cuModuleGetFunction(&func3, module, "_Z14gemver_kernel3iffPfS_S_"));

    unsigned grid1_x = (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X));
    unsigned grid1_y = (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_Y));
    unsigned grid2_x = (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X));
    unsigned grid3_x = (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X));

    void *args1[] = {&n, &alpha, &beta, &A_gpu, &v1_gpu, &v2_gpu, &u1_gpu, &u2_gpu, NULL};
    void *args2[] = {&n, &alpha, &beta, &A_gpu, &x_gpu, &y_gpu, &z_gpu, NULL};
    void *args3[] = {&n, &alpha, &beta, &A_gpu, &x_gpu, &w_gpu, NULL};
    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid1_x, grid1_y, 1, DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func2, grid2_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y, 1, 0, NULL, args2, NULL));
    cuError(cuLaunchKernel(func3, grid3_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y, 1, 0, NULL, args3, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(w_outputFromGpu, w_gpu, sizeof(DATA_TYPE) * N));

    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(x_gpu));
    cuError(cuMemFree(y_gpu));
    cuError(cuMemFree(z_gpu));
    cuError(cuMemFree(w_gpu));
    cuError(cuMemFree(v1_gpu));
    cuError(cuMemFree(v2_gpu));
    cuError(cuMemFree(u1_gpu));
    cuError(cuMemFree(u2_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w_outputFromGpu, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);

    init(n, &alpha, &beta,
         POLYBENCH_ARRAY(A),
         POLYBENCH_ARRAY(u1),
         POLYBENCH_ARRAY(v1),
         POLYBENCH_ARRAY(u2),
         POLYBENCH_ARRAY(v2),
         POLYBENCH_ARRAY(w),
         POLYBENCH_ARRAY(x),
         POLYBENCH_ARRAY(y),
         POLYBENCH_ARRAY(z));

    int deviceCount = 0;
    CUdevice device = 0;
    char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        fprintf(stdout, "\nTesting gemver on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        gemverCuda(device, n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2),
                   POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test gemver on GPU device %d Success\n", i);
    }
#ifdef RUN_ON_CPU
    polybench_start_instruments;
    SET_TIME(CPU_START)
    gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2),
           POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));
    SET_TIME(CPU_END)
    fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
    compareResults(n, POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu));
#else
    print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
#endif // RUN_ON_CPU


    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(w_outputFromGpu);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);
    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(v2);

    return 0;
}

#include "../include/polybench.c"