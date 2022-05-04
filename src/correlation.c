#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#include "correlation.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x, j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

#define RUN_ON_CPU

/*
Source Code:
__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_M)
    {
        mean[j] = 0.0;

        int i;
        for(i=0; i < _PB_N; i++)
        {
            mean[j] += data[i*M + j];
        }

        mean[j] /= (DATA_TYPE)FLOAT_N;
    }
}


__global__ void std_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_M)
    {
        std[j] = 0.0;

        int i;
        for(i = 0; i < _PB_N; i++)
        {
            std[j] += (data[i*M + j] - mean[j]) * (data[i*M + j] - mean[j]);
        }
        std[j] /= (FLOAT_N);
        std[j] = sqrt(std[j]);
        if(std[j] <= EPS)
        {
            std[j] = 1.0;
        }
    }
}


__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_N) && (j < _PB_M))
    {
        data[i*M + j] -= mean[j];
        data[i*M + j] /= (sqrt(FLOAT_N) * std[j]);
    }
}


__global__ void corr_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data)
{
    int j1 = blockIdx.x * blockDim.x + threadIdx.x;

    int i, j2;
    if (j1 < (_PB_M-1))
    {
        symmat[j1*M + j1] = 1.0;

        for (j2 = (j1 + 1); j2 < _PB_M; j2++)
        {
            symmat[j1*M + j2] = 0.0;

            for(i = 0; i < _PB_N; i++)
            {
                symmat[j1*M + j2] += data[i*M + j1] * data[i*M + j2];
            }
            symmat[j2*M + j1] = symmat[j1*M + j2];
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
                                "	// .globl	_Z10std_kerneliiPfS_S_\n"
                                ".visible .entry _Z10std_kerneliiPfS_S_(\n"
                                "	.param .u32 _Z10std_kerneliiPfS_S__param_0,\n"
                                "	.param .u32 _Z10std_kerneliiPfS_S__param_1,\n"
                                "	.param .u64 _Z10std_kerneliiPfS_S__param_2,\n"
                                "	.param .u64 _Z10std_kerneliiPfS_S__param_3,\n"
                                "	.param .u64 _Z10std_kerneliiPfS_S__param_4\n"
                                "){\n"
                                "	.reg .pred 	%p<9>;\n"
                                "	.reg .f32 	%f<45>;\n"
                                "	.reg .b32 	%r<34>;\n"
                                "	.reg .b64 	%rd<19>;\n"
                                ""
                                "	ld.param.u32 	%r16, [_Z10std_kerneliiPfS_S__param_0];\n"
                                "	ld.param.u32 	%r15, [_Z10std_kerneliiPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd4, [_Z10std_kerneliiPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd5, [_Z10std_kerneliiPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd6, [_Z10std_kerneliiPfS_S__param_4];\n"
                                "	cvta.to.global.u64 	%rd1, %rd6;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r16;\n"
                                "	@%p1 bra 	BB1_13;\n"
                                ""
                                "	cvta.to.global.u64 	%rd7, %rd5;\n"
                                "	mul.wide.s32 	%rd8, %r4, 4;\n"
                                "	add.s64 	%rd2, %rd7, %rd8;\n"
                                "	mov.u32 	%r31, 0;\n"
                                "	st.global.u32 	[%rd2], %r31;\n"
                                "	mov.f32 	%f44, 0f00000000;\n"
                                "	setp.lt.s32	%p2, %r15, 1;\n"
                                "	@%p2 bra 	BB1_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd9, %rd4;\n"
                                "	add.s64 	%rd3, %rd9, %rd8;\n"
                                "	and.b32  	%r21, %r15, 3;\n"
                                "	mov.f32 	%f44, 0f00000000;\n"
                                "	setp.eq.s32	%p3, %r21, 0;\n"
                                "	@%p3 bra 	BB1_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r21, 1;\n"
                                "	@%p4 bra 	BB1_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r21, 2;\n"
                                "	@%p5 bra 	BB1_6;\n"
                                ""
                                "	add.s64 	%rd12, %rd1, %rd8;\n"
                                "	ld.global.f32 	%f14, [%rd3];\n"
                                "	ld.global.f32 	%f15, [%rd12];\n"
                                "	sub.f32 	%f16, %f15, %f14;\n"
                                "	fma.rn.f32 	%f44, %f16, %f16, 0f00000000;\n"
                                "	st.global.f32 	[%rd2], %f44;\n"
                                "	mov.u32 	%r31, 1;\n"
                                ""
                                "BB1_6:\n"
                                "	shl.b32 	%r23, %r31, 11;\n"
                                "	add.s32 	%r24, %r23, %r4;\n"
                                "	mul.wide.s32 	%rd13, %r24, 4;\n"
                                "	add.s64 	%rd14, %rd1, %rd13;\n"
                                "	ld.global.f32 	%f17, [%rd3];\n"
                                "	ld.global.f32 	%f18, [%rd14];\n"
                                "	sub.f32 	%f19, %f18, %f17;\n"
                                "	fma.rn.f32 	%f44, %f19, %f19, %f44;\n"
                                "	st.global.f32 	[%rd2], %f44;\n"
                                "	add.s32 	%r31, %r31, 1;\n"
                                ""
                                "BB1_7:\n"
                                "	shl.b32 	%r25, %r31, 11;\n"
                                "	add.s32 	%r26, %r25, %r4;\n"
                                "	mul.wide.s32 	%rd15, %r26, 4;\n"
                                "	add.s64 	%rd16, %rd1, %rd15;\n"
                                "	ld.global.f32 	%f20, [%rd3];\n"
                                "	ld.global.f32 	%f21, [%rd16];\n"
                                "	sub.f32 	%f22, %f21, %f20;\n"
                                "	fma.rn.f32 	%f44, %f22, %f22, %f44;\n"
                                "	st.global.f32 	[%rd2], %f44;\n"
                                "	add.s32 	%r31, %r31, 1;\n"
                                ""
                                "BB1_8:\n"
                                "	setp.lt.u32	%p6, %r15, 4;\n"
                                "	@%p6 bra 	BB1_11;\n"
                                ""
                                "	mad.lo.s32 	%r32, %r31, 2048, %r4;\n"
                                ""
                                "BB1_10:\n"
                                "	mul.wide.s32 	%rd17, %r32, 4;\n"
                                "	add.s64 	%rd18, %rd1, %rd17;\n"
                                "	ld.global.f32 	%f23, [%rd3];\n"
                                "	ld.global.f32 	%f24, [%rd18];\n"
                                "	sub.f32 	%f25, %f24, %f23;\n"
                                "	fma.rn.f32 	%f26, %f25, %f25, %f44;\n"
                                "	st.global.f32 	[%rd2], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd3];\n"
                                "	ld.global.f32 	%f28, [%rd18+8192];\n"
                                "	sub.f32 	%f29, %f28, %f27;\n"
                                "	fma.rn.f32 	%f30, %f29, %f29, %f26;\n"
                                "	st.global.f32 	[%rd2], %f30;\n"
                                "	ld.global.f32 	%f31, [%rd3];\n"
                                "	ld.global.f32 	%f32, [%rd18+16384];\n"
                                "	sub.f32 	%f33, %f32, %f31;\n"
                                "	fma.rn.f32 	%f34, %f33, %f33, %f30;\n"
                                "	st.global.f32 	[%rd2], %f34;\n"
                                "	ld.global.f32 	%f35, [%rd3];\n"
                                "	ld.global.f32 	%f36, [%rd18+24576];\n"
                                "	sub.f32 	%f37, %f36, %f35;\n"
                                "	fma.rn.f32 	%f44, %f37, %f37, %f34;\n"
                                "	st.global.f32 	[%rd2], %f44;\n"
                                "	add.s32 	%r32, %r32, 8192;\n"
                                "	add.s32 	%r31, %r31, 4;\n"
                                "	setp.lt.s32	%p7, %r31, %r15;\n"
                                "	@%p7 bra 	BB1_10;\n"
                                ""
                                "BB1_11:\n"
                                "	div.rn.f32 	%f38, %f44, 0f4A442E10;\n"
                                "	sqrt.rn.f32 	%f39, %f38;\n"
                                "	st.global.f32 	[%rd2], %f39;\n"
                                "	setp.gtu.f32	%p8, %f39, 0f3BA3D70A;\n"
                                "	@%p8 bra 	BB1_13;\n"
                                ""
                                "	mov.u32 	%r28, 1065353216;\n"
                                "	st.global.u32 	[%rd2], %r28;\n"
                                ""
                                "BB1_13:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z13reduce_kerneliiPfS_S_\n"
                                ".visible .entry _Z13reduce_kerneliiPfS_S_(\n"
                                "	.param .u32 _Z13reduce_kerneliiPfS_S__param_0,\n"
                                "	.param .u32 _Z13reduce_kerneliiPfS_S__param_1,\n"
                                "	.param .u64 _Z13reduce_kerneliiPfS_S__param_2,\n"
                                "	.param .u64 _Z13reduce_kerneliiPfS_S__param_3,\n"
                                "	.param .u64 _Z13reduce_kerneliiPfS_S__param_4\n"
                                "){\n"
                                "	.reg .pred 	%p<4>;\n"
                                "	.reg .f32 	%f<7>;\n"
                                "	.reg .b32 	%r<13>;\n"
                                "	.reg .b64 	%rd<12>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z13reduce_kerneliiPfS_S__param_0];\n"
                                "	ld.param.u32 	%r4, [_Z13reduce_kerneliiPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd1, [_Z13reduce_kerneliiPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z13reduce_kerneliiPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd3, [_Z13reduce_kerneliiPfS_S__param_4];\n"
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
                                "	@%p3 bra 	BB2_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd4, %rd3;\n"
                                "	cvta.to.global.u64 	%rd5, %rd1;\n"
                                "	mul.wide.s32 	%rd6, %r1, 4;\n"
                                "	add.s64 	%rd7, %rd5, %rd6;\n"
                                "	shl.b32 	%r11, %r2, 11;\n"
                                "	add.s32 	%r12, %r11, %r1;\n"
                                "	mul.wide.s32 	%rd8, %r12, 4;\n"
                                "	add.s64 	%rd9, %rd4, %rd8;\n"
                                "	ld.global.f32 	%f1, [%rd9];\n"
                                "	ld.global.f32 	%f2, [%rd7];\n"
                                "	sub.f32 	%f3, %f1, %f2;\n"
                                "	st.global.f32 	[%rd9], %f3;\n"
                                "	cvta.to.global.u64 	%rd10, %rd2;\n"
                                "	add.s64 	%rd11, %rd10, %rd6;\n"
                                "	ld.global.f32 	%f4, [%rd11];\n"
                                "	mul.f32 	%f5, %f4, 0f44E01A51;\n"
                                "	div.rn.f32 	%f6, %f3, %f5;\n"
                                "	st.global.f32 	[%rd9], %f6;\n"
                                ""
                                "BB2_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11corr_kerneliiPfS_\n"
                                ".visible .entry _Z11corr_kerneliiPfS_(\n"
                                "	.param .u32 _Z11corr_kerneliiPfS__param_0,\n"
                                "	.param .u32 _Z11corr_kerneliiPfS__param_1,\n"
                                "	.param .u64 _Z11corr_kerneliiPfS__param_2,\n"
                                "	.param .u64 _Z11corr_kerneliiPfS__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<10>;\n"
                                "	.reg .f32 	%f<36>;\n"
                                "	.reg .b32 	%r<51>;\n"
                                "	.reg .b64 	%rd<31>;\n"
                                ""
                                "	ld.param.u32 	%r24, [_Z11corr_kerneliiPfS__param_0];\n"
                                "	ld.param.u32 	%r25, [_Z11corr_kerneliiPfS__param_1];\n"
                                "	ld.param.u64 	%rd6, [_Z11corr_kerneliiPfS__param_2];\n"
                                "	ld.param.u64 	%rd5, [_Z11corr_kerneliiPfS__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd6;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	add.s32 	%r26, %r24, -1;\n"
                                "	setp.ge.s32	%p1, %r4, %r26;\n"
                                "	@%p1 bra 	BB3_14;\n"
                                ""
                                "	mul.lo.s32 	%r27, %r4, 2049;\n"
                                "	mul.wide.s32 	%rd7, %r27, 4;\n"
                                "	add.s64 	%rd8, %rd1, %rd7;\n"
                                "	mov.u32 	%r28, 1065353216;\n"
                                "	st.global.u32 	[%rd8], %r28;\n"
                                "	add.s32 	%r44, %r4, 1;\n"
                                "	setp.ge.s32	%p2, %r44, %r24;\n"
                                "	@%p2 bra 	BB3_14;\n"
                                ""
                                "	cvta.to.global.u64 	%rd9, %rd5;\n"
                                "	shl.b32 	%r6, %r4, 11;\n"
                                "	and.b32  	%r7, %r25, 3;\n"
                                "	mul.wide.s32 	%rd10, %r4, 4;\n"
                                "	add.s64 	%rd2, %rd9, %rd10;\n"
                                ""
                                "BB3_3:\n"
                                "	add.s32 	%r29, %r44, %r6;\n"
                                "	mul.wide.s32 	%rd11, %r29, 4;\n"
                                "	add.s64 	%rd3, %rd1, %rd11;\n"
                                "	mov.u32 	%r47, 0;\n"
                                "	st.global.u32 	[%rd3], %r47;\n"
                                "	mov.f32 	%f35, 0f00000000;\n"
                                "	setp.lt.s32	%p3, %r25, 1;\n"
                                "	@%p3 bra 	BB3_13;\n"
                                ""
                                "	mov.f32 	%f35, 0f00000000;\n"
                                "	setp.eq.s32	%p4, %r7, 0;\n"
                                "	@%p4 bra 	BB3_10;\n"
                                ""
                                "	setp.eq.s32	%p5, %r7, 1;\n"
                                "	@%p5 bra 	BB3_9;\n"
                                ""
                                "	setp.eq.s32	%p6, %r7, 2;\n"
                                "	@%p6 bra 	BB3_8;\n"
                                ""
                                "	ld.global.f32 	%f14, [%rd2];\n"
                                "	mul.wide.s32 	%rd13, %r44, 4;\n"
                                "	add.s64 	%rd14, %rd9, %rd13;\n"
                                "	ld.global.f32 	%f15, [%rd14];\n"
                                "	fma.rn.f32 	%f35, %f14, %f15, 0f00000000;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	mov.u32 	%r47, 1;\n"
                                ""
                                "BB3_8:\n"
                                "	shl.b32 	%r35, %r47, 11;\n"
                                "	add.s32 	%r36, %r35, %r4;\n"
                                "	mul.wide.s32 	%rd16, %r36, 4;\n"
                                "	add.s64 	%rd17, %rd9, %rd16;\n"
                                "	add.s32 	%r37, %r35, %r44;\n"
                                "	mul.wide.s32 	%rd18, %r37, 4;\n"
                                "	add.s64 	%rd19, %rd9, %rd18;\n"
                                "	ld.global.f32 	%f16, [%rd19];\n"
                                "	ld.global.f32 	%f17, [%rd17];\n"
                                "	fma.rn.f32 	%f35, %f17, %f16, %f35;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	add.s32 	%r47, %r47, 1;\n"
                                ""
                                "BB3_9:\n"
                                "	shl.b32 	%r38, %r47, 11;\n"
                                "	add.s32 	%r39, %r38, %r4;\n"
                                "	mul.wide.s32 	%rd21, %r39, 4;\n"
                                "	add.s64 	%rd22, %rd9, %rd21;\n"
                                "	add.s32 	%r40, %r38, %r44;\n"
                                "	mul.wide.s32 	%rd23, %r40, 4;\n"
                                "	add.s64 	%rd24, %rd9, %rd23;\n"
                                "	ld.global.f32 	%f18, [%rd24];\n"
                                "	ld.global.f32 	%f19, [%rd22];\n"
                                "	fma.rn.f32 	%f35, %f19, %f18, %f35;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	add.s32 	%r47, %r47, 1;\n"
                                ""
                                "BB3_10:\n"
                                "	setp.lt.u32	%p7, %r25, 4;\n"
                                "	@%p7 bra 	BB3_13;\n"
                                ""
                                "	shl.b32 	%r41, %r47, 11;\n"
                                "	add.s32 	%r49, %r4, %r41;\n"
                                "	add.s32 	%r48, %r44, %r41;\n"
                                ""
                                "BB3_12:\n"
                                "	mul.wide.s32 	%rd25, %r49, 4;\n"
                                "	add.s64 	%rd26, %rd9, %rd25;\n"
                                "	mul.wide.s32 	%rd27, %r48, 4;\n"
                                "	add.s64 	%rd28, %rd9, %rd27;\n"
                                "	ld.global.f32 	%f20, [%rd28];\n"
                                "	ld.global.f32 	%f21, [%rd26];\n"
                                "	fma.rn.f32 	%f22, %f21, %f20, %f35;\n"
                                "	st.global.f32 	[%rd3], %f22;\n"
                                "	ld.global.f32 	%f23, [%rd28+8192];\n"
                                "	ld.global.f32 	%f24, [%rd26+8192];\n"
                                "	fma.rn.f32 	%f25, %f24, %f23, %f22;\n"
                                "	st.global.f32 	[%rd3], %f25;\n"
                                "	ld.global.f32 	%f26, [%rd28+16384];\n"
                                "	ld.global.f32 	%f27, [%rd26+16384];\n"
                                "	fma.rn.f32 	%f28, %f27, %f26, %f25;\n"
                                "	st.global.f32 	[%rd3], %f28;\n"
                                "	ld.global.f32 	%f29, [%rd28+24576];\n"
                                "	ld.global.f32 	%f30, [%rd26+24576];\n"
                                "	fma.rn.f32 	%f35, %f30, %f29, %f28;\n"
                                "	st.global.f32 	[%rd3], %f35;\n"
                                "	add.s32 	%r49, %r49, 8192;\n"
                                "	add.s32 	%r48, %r48, 8192;\n"
                                "	add.s32 	%r47, %r47, 4;\n"
                                "	setp.lt.s32	%p8, %r47, %r25;\n"
                                "	@%p8 bra 	BB3_12;\n"
                                ""
                                "BB3_13:\n"
                                "	shl.b32 	%r42, %r44, 11;\n"
                                "	add.s32 	%r43, %r42, %r4;\n"
                                "	mul.wide.s32 	%rd29, %r43, 4;\n"
                                "	add.s64 	%rd30, %rd1, %rd29;\n"
                                "	st.global.f32 	[%rd30], %f35;\n"
                                "	add.s32 	%r44, %r44, 1;\n"
                                "	setp.lt.s32	%p9, %r44, %r24;\n"
                                "	@%p9 bra 	BB3_3;\n"
                                ""
                                "BB3_14:\n"
                                "	ret;\n"
                                "}\n";

    void
    init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
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

void correlation(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), DATA_TYPE POLYBENCH_1D(stddev, M, m),
                 DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n))
{
    int i, j, j1, j2;

    // Determine mean of column vectors of input data matrix
    for (j = 0; j < _PB_M; j++)
    {
        mean[j] = 0.0;

        for (i = 0; i < _PB_N; i++)
        {
            mean[j] += data[i][j];
        }

        mean[j] /= (DATA_TYPE)FLOAT_N;
    }

    // Determine standard deviations of column vectors of data matrix.
    for (j = 0; j < _PB_M; j++)
    {
        stddev[j] = 0.0;

        for (i = 0; i < _PB_N; i++)
        {
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
        }

        stddev[j] /= FLOAT_N;
        stddev[j] = sqrt_of_array_cell(stddev, j);
        stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
    }

    // Center and reduce the column vectors.
    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < _PB_M; j++)
        {
            data[i][j] -= mean[j];
            data[i][j] /= (sqrt(FLOAT_N) * stddev[j]);
        }
    }

    // Calculate the m * m correlation matrix.
    for (j1 = 0; j1 < _PB_M - 1; j1++)
    {
        symmat[j1][j1] = 1.0;

        for (j2 = j1 + 1; j2 < _PB_M; j2++)
        {
            symmat[j1][j2] = 0.0;

            for (i = 0; i < _PB_N; i++)
            {
                symmat[j1][j2] += (data[i][j1] * data[i][j2]);
            }

            symmat[j2][j1] = symmat[j1][j2];
        }
    }

    symmat[M - 1][M - 1] = 1.0;
}

void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
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

    // print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}
void correlationCuda(CUdevice device, int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), 
			DATA_TYPE POLYBENCH_1D(stddev, M, m), DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), 
			DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
{
    CUdeviceptr data_gpu, stddev_gpu, mean_gpu, symmat_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL, func4 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&data_gpu, sizeof(DATA_TYPE) * M * N));
    cuError(cuMemAlloc(&symmat_gpu, sizeof(DATA_TYPE) * M * N));
    cuError(cuMemAlloc(&stddev_gpu, sizeof(DATA_TYPE) * M));
    cuError(cuMemAlloc(&mean_gpu, sizeof(DATA_TYPE) * M));
    cuError(cuMemcpyHtoD(data_gpu, data, sizeof(DATA_TYPE) * M * N));
    cuError(cuMemcpyHtoD(symmat_gpu, symmat, sizeof(DATA_TYPE) * M * N));
    cuError(cuMemcpyHtoD(stddev_gpu, stddev, sizeof(DATA_TYPE) * M));
    cuError(cuMemcpyHtoD(mean_gpu, mean, sizeof(DATA_TYPE) * M));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z11mean_kerneliiPfS_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z10std_kerneliiPfS_S_"));
    cuError(cuModuleGetFunction(&func3, module, "_Z13reduce_kerneliiPfS_S_"));
    cuError(cuModuleGetFunction(&func4, module, "_Z11corr_kerneliiPfS_"));

    unsigned grid1_x = (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X));
    unsigned grid2_x = (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X));
    unsigned grid3_x = (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X));
    unsigned grid3_y = (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y));
    unsigned grid4_x = (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X));

    void *args1[] = {&m, &n, &mean_gpu, &data_gpu, NULL};
    void *args2[] = {&m, &n, &mean_gpu,&stddev_gpu,&data_gpu, NULL};
    void *args3[] = {&m, &n, &mean_gpu,&stddev_gpu,&data_gpu, NULL};
    void *args4[] = {&m, &n, &symmat_gpu,&data_gpu, NULL};
    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid1_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func2, grid2_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y, 1, 0, NULL, args2, NULL));
    cuError(cuLaunchKernel(func3, grid3_x, grid3_y, 1, DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y, 1, 0, NULL, args3, NULL));
    cuError(cuLaunchKernel(func4, grid4_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y, 1, 0, NULL, args4, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

	DATA_TYPE valueAtSymmatIndexMTimesMPlus1PlusMPoint = 1.0;
	
    cuError(cuMemcpyDtoH(&symmat_gpu+((M-1)*M+(M-1))*sizeof(DATA_TYPE), &valueAtSymmatIndexMTimesMPlus1PlusMPoint, sizeof(DATA_TYPE)));
    cuError(cuMemcpyDtoH(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N));

    cuError(cuMemFree(data_gpu));
    cuError(cuMemFree(symmat_gpu));
    cuError(cuMemFree(stddev_gpu));
    cuError(cuMemFree(mean_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
int main()
{
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  	POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,N,m,n);
  	
	init_arrays(m, n, POLYBENCH_ARRAY(data));
    
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting correlation on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        correlationCuda(device, m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test correlation on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		correlation(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(stddev);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);

  	return 0;
}

#include "../include/polybench.c"
