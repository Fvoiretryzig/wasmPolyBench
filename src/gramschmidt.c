#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

#include "gramschmidt.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

/*
Source code:
__global__ void gramschmidt_kernel1(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid==0)
    {
        DATA_TYPE nrm = 0.0;
        int i;
        for (i = 0; i < _PB_NI; i++)
        {
            nrm += a[i * NJ + k] * a[i * NJ + k];
        }
            r[k * NJ + k] = sqrt(nrm);
    }
}


__global__ void gramschmidt_kernel2(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_NI)
    {
        q[i * NJ + k] = a[i * NJ + k] / r[k * NJ + k];
    }
}


__global__ void gramschmidt_kernel3(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((j > k) && (j < _PB_NJ))
    {
        r[k*NJ + j] = 0.0;

        int i;
        for (i = 0; i < _PB_NI; i++)
        {
            r[k*NJ + j] += q[i*NJ + k] * a[i*NJ + j];
        }

        for (i = 0; i < _PB_NI; i++)
        {
            a[i*NJ + j] -= q[i*NJ + k] * r[k*NJ + j];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z19gramschmidt_kernel1iiPfS_S_i\n"
                                ""
                                ".visible .entry _Z19gramschmidt_kernel1iiPfS_S_i(\n"
                                "	.param .u32 _Z19gramschmidt_kernel1iiPfS_S_i_param_0,\n"
                                "	.param .u32 _Z19gramschmidt_kernel1iiPfS_S_i_param_1,\n"
                                "	.param .u64 _Z19gramschmidt_kernel1iiPfS_S_i_param_2,\n"
                                "	.param .u64 _Z19gramschmidt_kernel1iiPfS_S_i_param_3,\n"
                                "	.param .u64 _Z19gramschmidt_kernel1iiPfS_S_i_param_4,\n"
                                "	.param .u32 _Z19gramschmidt_kernel1iiPfS_S_i_param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<30>;\n"
                                "	.reg .b32 	%r<33>;\n"
                                "	.reg .b64 	%rd<15>;\n"
                                ""
                                "	ld.param.u32 	%r11, [_Z19gramschmidt_kernel1iiPfS_S_i_param_0];\n"
                                "	ld.param.u64 	%rd3, [_Z19gramschmidt_kernel1iiPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z19gramschmidt_kernel1iiPfS_S_i_param_3];\n"
                                "	ld.param.u32 	%r12, [_Z19gramschmidt_kernel1iiPfS_S_i_param_5];\n"
                                "	cvta.to.global.u64 	%rd1, %rd3;\n"
                                "	mov.u32 	%r13, %ntid.x;\n"
                                "	mov.u32 	%r14, %ctaid.x;\n"
                                "	mul.lo.s32 	%r15, %r13, %r14;\n"
                                "	mov.u32 	%r16, %tid.x;\n"
                                "	neg.s32 	%r17, %r16;\n"
                                "	setp.ne.s32	%p1, %r15, %r17;\n"
                                "	@%p1 bra 	BB0_12;\n"
                                ""
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.lt.s32	%p2, %r11, 1;\n"
                                "	@%p2 bra 	BB0_11;\n"
                                ""
                                "	and.b32  	%r21, %r11, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	mov.u32 	%r30, 0;\n"
                                "	setp.eq.s32	%p3, %r21, 0;\n"
                                "	@%p3 bra 	BB0_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r21, 1;\n"
                                "	@%p4 bra 	BB0_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r21, 2;\n"
                                "	@%p5 bra 	BB0_6;\n"
                                ""
                                "	mul.wide.s32 	%rd4, %r12, 4;\n"
                                "	add.s64 	%rd5, %rd1, %rd4;\n"
                                "	ld.global.f32 	%f14, [%rd5];\n"
                                "	fma.rn.f32 	%f29, %f14, %f14, 0f00000000;\n"
                                "	mov.u32 	%r30, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	shl.b32 	%r23, %r30, 11;\n"
                                "	add.s32 	%r24, %r23, %r12;\n"
                                "	mul.wide.s32 	%rd6, %r24, 4;\n"
                                "	add.s64 	%rd7, %rd1, %rd6;\n"
                                "	ld.global.f32 	%f15, [%rd7];\n"
                                "	fma.rn.f32 	%f29, %f15, %f15, %f29;\n"
                                "	add.s32 	%r30, %r30, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	shl.b32 	%r25, %r30, 11;\n"
                                "	add.s32 	%r26, %r25, %r12;\n"
                                "	mul.wide.s32 	%rd8, %r26, 4;\n"
                                "	add.s64 	%rd9, %rd1, %rd8;\n"
                                "	ld.global.f32 	%f16, [%rd9];\n"
                                "	fma.rn.f32 	%f29, %f16, %f16, %f29;\n"
                                "	add.s32 	%r30, %r30, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	setp.lt.u32	%p6, %r11, 4;\n"
                                "	@%p6 bra 	BB0_11;\n"
                                ""
                                "	mad.lo.s32 	%r31, %r30, 2048, %r12;\n"
                                ""
                                "BB0_10:\n"
                                "	mul.wide.s32 	%rd10, %r31, 4;\n"
                                "	add.s64 	%rd11, %rd1, %rd10;\n"
                                "	ld.global.f32 	%f17, [%rd11];\n"
                                "	fma.rn.f32 	%f18, %f17, %f17, %f29;\n"
                                "	ld.global.f32 	%f19, [%rd11+8192];\n"
                                "	fma.rn.f32 	%f20, %f19, %f19, %f18;\n"
                                "	ld.global.f32 	%f21, [%rd11+16384];\n"
                                "	fma.rn.f32 	%f22, %f21, %f21, %f20;\n"
                                "	ld.global.f32 	%f23, [%rd11+24576];\n"
                                "	fma.rn.f32 	%f29, %f23, %f23, %f22;\n"
                                "	add.s32 	%r31, %r31, 8192;\n"
                                "	add.s32 	%r30, %r30, 4;\n"
                                "	setp.lt.s32	%p7, %r30, %r11;\n"
                                "	@%p7 bra 	BB0_10;\n"
                                ""
                                "BB0_11:\n"
                                "	cvta.to.global.u64 	%rd12, %rd2;\n"
                                "	mul.lo.s32 	%r27, %r12, 2049;\n"
                                "	mul.wide.s32 	%rd13, %r27, 4;\n"
                                "	add.s64 	%rd14, %rd12, %rd13;\n"
                                "	sqrt.rn.f32 	%f24, %f29;\n"
                                "	st.global.f32 	[%rd14], %f24;\n"
                                ""
                                "BB0_12:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z19gramschmidt_kernel2iiPfS_S_i\n"
                                ".visible .entry _Z19gramschmidt_kernel2iiPfS_S_i(\n"
                                "	.param .u32 _Z19gramschmidt_kernel2iiPfS_S_i_param_0,\n"
                                "	.param .u32 _Z19gramschmidt_kernel2iiPfS_S_i_param_1,\n"
                                "	.param .u64 _Z19gramschmidt_kernel2iiPfS_S_i_param_2,\n"
                                "	.param .u64 _Z19gramschmidt_kernel2iiPfS_S_i_param_3,\n"
                                "	.param .u64 _Z19gramschmidt_kernel2iiPfS_S_i_param_4,\n"
                                "	.param .u32 _Z19gramschmidt_kernel2iiPfS_S_i_param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<2>;\n"
                                "	.reg .f32 	%f<4>;\n"
                                "	.reg .b32 	%r<10>;\n"
                                "	.reg .b64 	%rd<12>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z19gramschmidt_kernel2iiPfS_S_i_param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z19gramschmidt_kernel2iiPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z19gramschmidt_kernel2iiPfS_S_i_param_3];\n"
                                "	ld.param.u64 	%rd3, [_Z19gramschmidt_kernel2iiPfS_S_i_param_4];\n"
                                "	ld.param.u32 	%r2, [_Z19gramschmidt_kernel2iiPfS_S_i_param_5];\n"
                                "	mov.u32 	%r4, %ctaid.x;\n"
                                "	mov.u32 	%r5, %ntid.x;\n"
                                "	mov.u32 	%r6, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r5, %r4, %r6;\n"
                                "	setp.ge.s32	%p1, %r1, %r3;\n"
                                "	@%p1 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd4, %rd1;\n"
                                "	shl.b32 	%r7, %r1, 11;\n"
                                "	add.s32 	%r8, %r7, %r2;\n"
                                "	mul.wide.s32 	%rd5, %r8, 4;\n"
                                "	add.s64 	%rd6, %rd4, %rd5;\n"
                                "	mul.lo.s32 	%r9, %r2, 2049;\n"
                                "	cvta.to.global.u64 	%rd7, %rd2;\n"
                                "	mul.wide.s32 	%rd8, %r9, 4;\n"
                                "	add.s64 	%rd9, %rd7, %rd8;\n"
                                "	ld.global.f32 	%f1, [%rd9];\n"
                                "	ld.global.f32 	%f2, [%rd6];\n"
                                "	div.rn.f32 	%f3, %f2, %f1;\n"
                                "	cvta.to.global.u64 	%rd10, %rd3;\n"
                                "	add.s64 	%rd11, %rd10, %rd5;\n"
                                "	st.global.f32 	[%rd11], %f3;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z19gramschmidt_kernel3iiPfS_S_i\n"
                                ".visible .entry _Z19gramschmidt_kernel3iiPfS_S_i(\n"
                                "	.param .u32 _Z19gramschmidt_kernel3iiPfS_S_i_param_0,\n"
                                "	.param .u32 _Z19gramschmidt_kernel3iiPfS_S_i_param_1,\n"
                                "	.param .u64 _Z19gramschmidt_kernel3iiPfS_S_i_param_2,\n"
                                "	.param .u64 _Z19gramschmidt_kernel3iiPfS_S_i_param_3,\n"
                                "	.param .u64 _Z19gramschmidt_kernel3iiPfS_S_i_param_4,\n"
                                "	.param .u32 _Z19gramschmidt_kernel3iiPfS_S_i_param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<16>;\n"
                                "	.reg .f32 	%f<74>;\n"
                                "	.reg .b32 	%r<79>;\n"
                                "	.reg .b64 	%rd<40>;\n"
                                ""
                                "	ld.param.u32 	%r31, [_Z19gramschmidt_kernel3iiPfS_S_i_param_0];\n"
                                "	ld.param.u32 	%r33, [_Z19gramschmidt_kernel3iiPfS_S_i_param_1];\n"
                                "	ld.param.u64 	%rd5, [_Z19gramschmidt_kernel3iiPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd6, [_Z19gramschmidt_kernel3iiPfS_S_i_param_3];\n"
                                "	ld.param.u64 	%rd7, [_Z19gramschmidt_kernel3iiPfS_S_i_param_4];\n"
                                "	ld.param.u32 	%r32, [_Z19gramschmidt_kernel3iiPfS_S_i_param_5];\n"
                                "	cvta.to.global.u64 	%rd1, %rd5;\n"
                                "	cvta.to.global.u64 	%rd2, %rd7;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.le.s32	%p1, %r4, %r32;\n"
                                "	setp.ge.s32	%p2, %r4, %r33;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB2_21;\n"
                                ""
                                "	cvta.to.global.u64 	%rd8, %rd6;\n"
                                "	shl.b32 	%r34, %r32, 11;\n"
                                "	add.s32 	%r35, %r4, %r34;\n"
                                "	mul.wide.s32 	%rd9, %r35, 4;\n"
                                "	add.s64 	%rd3, %rd8, %rd9;\n"
                                "	mov.u32 	%r73, 0;\n"
                                "	st.global.u32 	[%rd3], %r73;\n"
                                "	mul.wide.s32 	%rd10, %r32, 4;\n"
                                "	add.s64 	%rd4, %rd2, %rd10;\n"
                                "	mov.f32 	%f71, 0f00000000;\n"
                                "	setp.lt.s32	%p4, %r31, 1;\n"
                                "	@%p4 bra 	BB2_11;\n"
                                ""
                                "	and.b32  	%r40, %r31, 3;\n"
                                "	mov.u32 	%r69, 0;\n"
                                "	mov.f32 	%f71, 0f00000000;\n"
                                "	setp.eq.s32	%p5, %r40, 0;\n"
                                "	@%p5 bra 	BB2_8;\n"
                                ""
                                "	setp.eq.s32	%p6, %r40, 1;\n"
                                "	@%p6 bra 	BB2_7;\n"
                                ""
                                "	setp.eq.s32	%p7, %r40, 2;\n"
                                "	@%p7 bra 	BB2_6;\n"
                                ""
                                "	ld.global.f32 	%f18, [%rd4];\n"
                                "	mul.wide.s32 	%rd11, %r4, 4;\n"
                                "	add.s64 	%rd12, %rd1, %rd11;\n"
                                "	ld.global.f32 	%f19, [%rd12];\n"
                                "	fma.rn.f32 	%f71, %f18, %f19, 0f00000000;\n"
                                "	st.global.f32 	[%rd3], %f71;\n"
                                "	mov.u32 	%r69, 1;\n"
                                ""
                                "BB2_6:\n"
                                "	shl.b32 	%r42, %r69, 11;\n"
                                "	add.s32 	%r43, %r42, %r32;\n"
                                "	mul.wide.s32 	%rd13, %r43, 4;\n"
                                "	add.s64 	%rd14, %rd2, %rd13;\n"
                                "	add.s32 	%r44, %r42, %r4;\n"
                                "	mul.wide.s32 	%rd15, %r44, 4;\n"
                                "	add.s64 	%rd16, %rd1, %rd15;\n"
                                "	ld.global.f32 	%f20, [%rd16];\n"
                                "	ld.global.f32 	%f21, [%rd14];\n"
                                "	fma.rn.f32 	%f71, %f21, %f20, %f71;\n"
                                "	st.global.f32 	[%rd3], %f71;\n"
                                "	add.s32 	%r69, %r69, 1;\n"
                                ""
                                "BB2_7:\n"
                                "	shl.b32 	%r45, %r69, 11;\n"
                                "	add.s32 	%r46, %r45, %r32;\n"
                                "	mul.wide.s32 	%rd17, %r46, 4;\n"
                                "	add.s64 	%rd18, %rd2, %rd17;\n"
                                "	add.s32 	%r47, %r45, %r4;\n"
                                "	mul.wide.s32 	%rd19, %r47, 4;\n"
                                "	add.s64 	%rd20, %rd1, %rd19;\n"
                                "	ld.global.f32 	%f22, [%rd20];\n"
                                "	ld.global.f32 	%f23, [%rd18];\n"
                                "	fma.rn.f32 	%f71, %f23, %f22, %f71;\n"
                                "	st.global.f32 	[%rd3], %f71;\n"
                                "	add.s32 	%r69, %r69, 1;\n"
                                ""
                                "BB2_8:\n"
                                "	setp.lt.u32	%p8, %r31, 4;\n"
                                "	@%p8 bra 	BB2_11;\n"
                                ""
                                "	shl.b32 	%r48, %r69, 11;\n"
                                "	add.s32 	%r71, %r32, %r48;\n"
                                "	add.s32 	%r70, %r4, %r48;\n"
                                ""
                                "BB2_10:\n"
                                "	mul.wide.s32 	%rd21, %r71, 4;\n"
                                "	add.s64 	%rd22, %rd2, %rd21;\n"
                                "	mul.wide.s32 	%rd23, %r70, 4;\n"
                                "	add.s64 	%rd24, %rd1, %rd23;\n"
                                "	ld.global.f32 	%f24, [%rd24];\n"
                                "	ld.global.f32 	%f25, [%rd22];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f71;\n"
                                "	st.global.f32 	[%rd3], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd24+8192];\n"
                                "	ld.global.f32 	%f28, [%rd22+8192];\n"
                                "	fma.rn.f32 	%f29, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	ld.global.f32 	%f30, [%rd24+16384];\n"
                                "	ld.global.f32 	%f31, [%rd22+16384];\n"
                                "	fma.rn.f32 	%f32, %f31, %f30, %f29;\n"
                                "	st.global.f32 	[%rd3], %f32;\n"
                                "	ld.global.f32 	%f33, [%rd24+24576];\n"
                                "	ld.global.f32 	%f34, [%rd22+24576];\n"
                                "	fma.rn.f32 	%f71, %f34, %f33, %f32;\n"
                                "	st.global.f32 	[%rd3], %f71;\n"
                                "	add.s32 	%r71, %r71, 8192;\n"
                                "	add.s32 	%r70, %r70, 8192;\n"
                                "	add.s32 	%r69, %r69, 4;\n"
                                "	setp.lt.s32	%p9, %r69, %r31;\n"
                                "	@%p9 bra 	BB2_10;\n"
                                ""
                                "BB2_11:\n"
                                "	@%p4 bra 	BB2_21;\n"
                                ""
                                "	and.b32  	%r53, %r31, 3;\n"
                                "	setp.eq.s32	%p11, %r53, 0;\n"
                                "	@%p11 bra 	BB2_18;\n"
                                ""
                                "	setp.eq.s32	%p12, %r53, 1;\n"
                                "	@%p12 bra 	BB2_17;\n"
                                ""
                                "	setp.eq.s32	%p13, %r53, 2;\n"
                                "	@%p13 bra 	BB2_16;\n"
                                ""
                                "	ld.global.f32 	%f35, [%rd4];\n"
                                "	mul.f32 	%f36, %f35, %f71;\n"
                                "	mul.wide.s32 	%rd26, %r4, 4;\n"
                                "	add.s64 	%rd27, %rd1, %rd26;\n"
                                "	ld.global.f32 	%f37, [%rd27];\n"
                                "	sub.f32 	%f38, %f37, %f36;\n"
                                "	st.global.f32 	[%rd27], %f38;\n"
                                "	ld.global.f32 	%f71, [%rd3];\n"
                                "	mov.u32 	%r73, 1;\n"
                                ""
                                "BB2_16:\n"
                                "	shl.b32 	%r59, %r73, 11;\n"
                                "	add.s32 	%r60, %r59, %r32;\n"
                                "	mul.wide.s32 	%rd28, %r60, 4;\n"
                                "	add.s64 	%rd29, %rd2, %rd28;\n"
                                "	ld.global.f32 	%f39, [%rd29];\n"
                                "	mul.f32 	%f40, %f39, %f71;\n"
                                "	add.s32 	%r61, %r59, %r4;\n"
                                "	mul.wide.s32 	%rd30, %r61, 4;\n"
                                "	add.s64 	%rd31, %rd1, %rd30;\n"
                                "	ld.global.f32 	%f41, [%rd31];\n"
                                "	sub.f32 	%f42, %f41, %f40;\n"
                                "	st.global.f32 	[%rd31], %f42;\n"
                                "	add.s32 	%r73, %r73, 1;\n"
                                "	ld.global.f32 	%f71, [%rd3];\n"
                                ""
                                "BB2_17:\n"
                                "	shl.b32 	%r62, %r73, 11;\n"
                                "	add.s32 	%r63, %r62, %r32;\n"
                                "	mul.wide.s32 	%rd32, %r63, 4;\n"
                                "	add.s64 	%rd33, %rd2, %rd32;\n"
                                "	ld.global.f32 	%f43, [%rd33];\n"
                                "	mul.f32 	%f44, %f43, %f71;\n"
                                "	add.s32 	%r64, %r62, %r4;\n"
                                "	mul.wide.s32 	%rd34, %r64, 4;\n"
                                "	add.s64 	%rd35, %rd1, %rd34;\n"
                                "	ld.global.f32 	%f45, [%rd35];\n"
                                "	sub.f32 	%f46, %f45, %f44;\n"
                                "	st.global.f32 	[%rd35], %f46;\n"
                                "	add.s32 	%r73, %r73, 1;\n"
                                ""
                                "BB2_18:\n"
                                "	setp.lt.u32	%p14, %r31, 4;\n"
                                "	@%p14 bra 	BB2_21;\n"
                                ""
                                "	shl.b32 	%r65, %r73, 11;\n"
                                "	add.s32 	%r77, %r32, %r65;\n"
                                "	add.s32 	%r76, %r4, %r65;\n"
                                ""
                                "BB2_20:\n"
                                "	mul.wide.s32 	%rd36, %r77, 4;\n"
                                "	add.s64 	%rd37, %rd2, %rd36;\n"
                                "	ld.global.f32 	%f47, [%rd3];\n"
                                "	ld.global.f32 	%f48, [%rd37];\n"
                                "	mul.f32 	%f49, %f48, %f47;\n"
                                "	mul.wide.s32 	%rd38, %r76, 4;\n"
                                "	add.s64 	%rd39, %rd1, %rd38;\n"
                                "	ld.global.f32 	%f50, [%rd39];\n"
                                "	sub.f32 	%f51, %f50, %f49;\n"
                                "	st.global.f32 	[%rd39], %f51;\n"
                                "	ld.global.f32 	%f52, [%rd3];\n"
                                "	ld.global.f32 	%f53, [%rd37+8192];\n"
                                "	mul.f32 	%f54, %f53, %f52;\n"
                                "	ld.global.f32 	%f55, [%rd39+8192];\n"
                                "	sub.f32 	%f56, %f55, %f54;\n"
                                "	st.global.f32 	[%rd39+8192], %f56;\n"
                                "	ld.global.f32 	%f57, [%rd3];\n"
                                "	ld.global.f32 	%f58, [%rd37+16384];\n"
                                "	mul.f32 	%f59, %f58, %f57;\n"
                                "	ld.global.f32 	%f60, [%rd39+16384];\n"
                                "	sub.f32 	%f61, %f60, %f59;\n"
                                "	st.global.f32 	[%rd39+16384], %f61;\n"
                                "	ld.global.f32 	%f62, [%rd3];\n"
                                "	ld.global.f32 	%f63, [%rd37+24576];\n"
                                "	mul.f32 	%f64, %f63, %f62;\n"
                                "	ld.global.f32 	%f65, [%rd39+24576];\n"
                                "	sub.f32 	%f66, %f65, %f64;\n"
                                "	st.global.f32 	[%rd39+24576], %f66;\n"
                                "	add.s32 	%r77, %r77, 8192;\n"
                                "	add.s32 	%r76, %r76, 8192;\n"
                                "	add.s32 	%r73, %r73, 4;\n"
                                "	setp.lt.s32	%p15, %r73, %r31;\n"
                                "	@%p15 bra 	BB2_20;\n"
                                ""
                                "BB2_21:\n"
                                "	ret;\n"
                                "}\n";
void gramschmidt(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj), DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
    int i, j, k;
    DATA_TYPE nrm;
    for (k = 0; k < _PB_NJ; k++)
    {
        nrm = 0;
        for (i = 0; i < _PB_NI; i++)
        {
            nrm += A[i][k] * A[i][k];
        }

        R[k][k] = sqrt(nrm);
        for (i = 0; i < _PB_NI; i++)
        {
            Q[i][k] = A[i][k] / R[k][k];
        }

        for (j = k + 1; j < _PB_NJ; j++)
        {
            R[k][j] = 0;
            for (i = 0; i < _PB_NI; i++)
            {
                R[k][j] += Q[i][k] * A[i][j];
            }
            for (i = 0; i < _PB_NI; i++)
            {
                A[i][j] = A[i][j] - Q[i][k] * R[k][j];
            }
        }
    }
}

/* Array initialization. */
void init_array(int ni, int nj,
                DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                DATA_TYPE POLYBENCH_2D(R, NJ, NJ, nj, nj),
                DATA_TYPE POLYBENCH_2D(Q, NI, NJ, ni, nj))
{
    int i, j;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / ni;
            Q[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nj; j++)
        {
            R[i][j] = ((DATA_TYPE)i * (j + 2)) / nj;
        }
    }
}

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(A_outputFromGpu, NI, NJ, ni, nj))
{
    int i, j, fail;
    fail = 0;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void gramschmidtCuda(CUdevice device, int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj), DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,NI,NJ,ni,nj))
{	
    CUdeviceptr A_gpu, R_gpu, Q_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&R_gpu, sizeof(DATA_TYPE) * NJ * NJ));
    cuError(cuMemAlloc(&Q_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));
	cuError(cuMemcpyHtoD(R_gpu, R, sizeof(DATA_TYPE) * NJ * NJ));
	cuError(cuMemcpyHtoD(Q_gpu, Q, sizeof(DATA_TYPE) * NI * NJ));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z19gramschmidt_kernel1iiPfS_S_i"));
    cuError(cuModuleGetFunction(&func2, module, "_Z19gramschmidt_kernel2iiPfS_S_i"));
    cuError(cuModuleGetFunction(&func3, module, "_Z19gramschmidt_kernel3iiPfS_S_i"));

    unsigned grid2_x = (size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X));
    unsigned grid3_x = (size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X));
	int k;
	SET_TIME(START)
    for (k = 0; k < _PB_NJ; k++)
	{
		void *args1[] = {&ni, &nj, &A_gpu, &R_gpu, &Q_gpu, &k, NULL};
        cuError(cuLaunchKernel(func1, 1, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
        cuError(cuLaunchKernel(func2, grid2_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
        cuError(cuLaunchKernel(func3, grid3_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
	}
	SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(R_gpu));
    cuError(cuMemFree(Q_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
int main()
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,NJ,NJ,nj,nj);
	POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,NI,NJ,ni,nj);
	
	init_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));
	
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting gramschmidt on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        gramschmidtCuda(device, ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q), POLYBENCH_ARRAY(A_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test gramschmidt on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		gramschmidt(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);
	POLYBENCH_FREE_ARRAY(R);
	POLYBENCH_FREE_ARRAY(Q);  

    	return 0;
}

#include "../include/polybench.c"
