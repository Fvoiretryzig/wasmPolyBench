#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define POLYBENCH_TIME 1

#include "polybenchUtilFuncts.h"
#include "polybench.h"
#include "cuda-helper.h"
#include "adi.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 2.5

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
 * Source code:
 * __global__ void adi_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X) {
 * 	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
 * 	if ((i1 < _PB_N)) {
 * 		for (int i2 = 1; i2 < _PB_N; i2++) {
 * 			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
 * 			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
 * 		}
 * 	}
 * }
 * __global__ void adi_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X) {
 * 	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
 * 	if ((i1 < _PB_N)) {
 * 		X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
 * 	}
 * }
 * __global__ void adi_kernel3(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X) {
 * 	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
 * 	if (i1 < _PB_N) {
 * 		for (int i2 = 0; i2 < _PB_N-2; i2++)
 * 		{
 * 			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
 * 		}
 * 	}
 * }
 * __global__ void adi_kernel4(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1) {
 * 	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
 * 	if (i2 < _PB_N) {
 * 		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
 * 		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
 * 	}
 * }
 * __global__ void adi_kernel5(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X) {
 * 	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
 * 	if (i2 < _PB_N) {
 * 		X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
 * 	}
 * }
 * __global__ void adi_kernel6(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1) {
 * 	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
 * 	if (i2 < _PB_N) {
 * 		X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
 * 	}
 * }
 */

static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z11adi_kernel1iPfS_S_\n"
                                ""
                                ".visible .entry _Z11adi_kernel1iPfS_S_(\n"
                                "	.param .u32 _Z11adi_kernel1iPfS_S__param_0,\n"
                                "	.param .u64 _Z11adi_kernel1iPfS_S__param_1,\n"
                                "	.param .u64 _Z11adi_kernel1iPfS_S__param_2,\n"
                                "	.param .u64 _Z11adi_kernel1iPfS_S__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<9>;\n"
                                "	.reg .f32 	%f<89>;\n"
                                "	.reg .b32 	%r<32>;\n"
                                "	.reg .b64 	%rd<37>;\n"
                                ""
                                "	ld.param.u32 	%r15, [_Z11adi_kernel1iPfS_S__param_0];\n"
                                "	ld.param.u64 	%rd11, [_Z11adi_kernel1iPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd12, [_Z11adi_kernel1iPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd13, [_Z11adi_kernel1iPfS_S__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd2, %rd11;\n"
                                "	cvta.to.global.u64 	%rd3, %rd13;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r15;\n"
                                "	setp.lt.s32	%p2, %r15, 2;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_10;\n"
                                ""
                                "	shl.b32 	%r5, %r4, 10;\n"
                                "	add.s32 	%r6, %r5, -1;\n"
                                "	add.s32 	%r7, %r15, -1;\n"
                                "	and.b32  	%r19, %r7, 3;\n"
                                "	mov.u32 	%r28, 1;\n"
                                "	setp.eq.s32	%p4, %r19, 0;\n"
                                "	@%p4 bra 	BB0_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r19, 1;\n"
                                "	@%p5 bra 	BB0_6;\n"
                                ""
                                "	setp.eq.s32	%p6, %r19, 2;\n"
                                "	@%p6 bra 	BB0_5;\n"
                                ""
                                "	mul.wide.s32 	%rd14, %r5, 4;\n"
                                "	add.s64 	%rd15, %rd3, %rd14;\n"
                                "	add.s64 	%rd16, %rd2, %rd14;\n"
                                "	ld.global.f32 	%f1, [%rd16+4];\n"
                                "	ld.global.f32 	%f2, [%rd15];\n"
                                "	mul.f32 	%f3, %f2, %f1;\n"
                                "	add.s64 	%rd17, %rd1, %rd14;\n"
                                "	ld.global.f32 	%f4, [%rd17];\n"
                                "	div.rn.f32 	%f5, %f3, %f4;\n"
                                "	ld.global.f32 	%f6, [%rd15+4];\n"
                                "	sub.f32 	%f7, %f6, %f5;\n"
                                "	st.global.f32 	[%rd15+4], %f7;\n"
                                "	ld.global.f32 	%f8, [%rd16+4];\n"
                                "	mul.f32 	%f9, %f8, %f8;\n"
                                "	ld.global.f32 	%f10, [%rd17];\n"
                                "	div.rn.f32 	%f11, %f9, %f10;\n"
                                "	ld.global.f32 	%f12, [%rd17+4];\n"
                                "	sub.f32 	%f13, %f12, %f11;\n"
                                "	st.global.f32 	[%rd17+4], %f13;\n"
                                "	mov.u32 	%r28, 2;\n"
                                ""
                                "BB0_5:\n"
                                "	add.s32 	%r21, %r28, %r5;\n"
                                "	mul.wide.s32 	%rd18, %r21, 4;\n"
                                "	add.s64 	%rd19, %rd3, %rd18;\n"
                                "	add.s32 	%r22, %r6, %r28;\n"
                                "	mul.wide.s32 	%rd20, %r22, 4;\n"
                                "	add.s64 	%rd21, %rd3, %rd20;\n"
                                "	add.s64 	%rd22, %rd2, %rd18;\n"
                                "	ld.global.f32 	%f14, [%rd22];\n"
                                "	ld.global.f32 	%f15, [%rd21];\n"
                                "	mul.f32 	%f16, %f15, %f14;\n"
                                "	add.s64 	%rd23, %rd1, %rd20;\n"
                                "	ld.global.f32 	%f17, [%rd23];\n"
                                "	div.rn.f32 	%f18, %f16, %f17;\n"
                                "	ld.global.f32 	%f19, [%rd19];\n"
                                "	sub.f32 	%f20, %f19, %f18;\n"
                                "	st.global.f32 	[%rd19], %f20;\n"
                                "	add.s64 	%rd24, %rd1, %rd18;\n"
                                "	ld.global.f32 	%f21, [%rd22];\n"
                                "	mul.f32 	%f22, %f21, %f21;\n"
                                "	ld.global.f32 	%f23, [%rd23];\n"
                                "	div.rn.f32 	%f24, %f22, %f23;\n"
                                "	ld.global.f32 	%f25, [%rd24];\n"
                                "	sub.f32 	%f26, %f25, %f24;\n"
                                "	st.global.f32 	[%rd24], %f26;\n"
                                "	add.s32 	%r28, %r28, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	add.s32 	%r23, %r28, %r5;\n"
                                "	mul.wide.s32 	%rd25, %r23, 4;\n"
                                "	add.s64 	%rd26, %rd3, %rd25;\n"
                                "	add.s32 	%r24, %r6, %r28;\n"
                                "	mul.wide.s32 	%rd27, %r24, 4;\n"
                                "	add.s64 	%rd28, %rd3, %rd27;\n"
                                "	add.s64 	%rd29, %rd2, %rd25;\n"
                                "	ld.global.f32 	%f27, [%rd29];\n"
                                "	ld.global.f32 	%f28, [%rd28];\n"
                                "	mul.f32 	%f29, %f28, %f27;\n"
                                "	add.s64 	%rd30, %rd1, %rd27;\n"
                                "	ld.global.f32 	%f30, [%rd30];\n"
                                "	div.rn.f32 	%f31, %f29, %f30;\n"
                                "	ld.global.f32 	%f32, [%rd26];\n"
                                "	sub.f32 	%f33, %f32, %f31;\n"
                                "	st.global.f32 	[%rd26], %f33;\n"
                                "	add.s64 	%rd31, %rd1, %rd25;\n"
                                "	ld.global.f32 	%f34, [%rd29];\n"
                                "	mul.f32 	%f35, %f34, %f34;\n"
                                "	ld.global.f32 	%f36, [%rd30];\n"
                                "	div.rn.f32 	%f37, %f35, %f36;\n"
                                "	ld.global.f32 	%f38, [%rd31];\n"
                                "	sub.f32 	%f39, %f38, %f37;\n"
                                "	st.global.f32 	[%rd31], %f39;\n"
                                "	add.s32 	%r28, %r28, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	setp.lt.u32	%p7, %r7, 4;\n"
                                "	@%p7 bra 	BB0_10;\n"
                                ""
                                "	mul.lo.s32 	%r25, %r1, %r2;\n"
                                "	mad.lo.s32 	%r26, %r25, 1024, %r28;\n"
                                "	mad.lo.s32 	%r27, %r3, 1024, %r26;\n"
                                "	mul.wide.s32 	%rd35, %r27, 4;\n"
                                "	mov.u64 	%rd36, %rd35;\n"
                                ""
                                "BB0_9:\n"
                                "	add.s64 	%rd32, %rd3, %rd35;\n"
                                "	add.s64 	%rd33, %rd2, %rd35;\n"
                                "	ld.global.f32 	%f40, [%rd33];\n"
                                "	ld.global.f32 	%f41, [%rd32+-4];\n"
                                "	mul.f32 	%f42, %f41, %f40;\n"
                                "	add.s64 	%rd34, %rd1, %rd36;\n"
                                "	ld.global.f32 	%f43, [%rd34+-4];\n"
                                "	div.rn.f32 	%f44, %f42, %f43;\n"
                                "	ld.global.f32 	%f45, [%rd32];\n"
                                "	sub.f32 	%f46, %f45, %f44;\n"
                                "	st.global.f32 	[%rd32], %f46;\n"
                                "	ld.global.f32 	%f47, [%rd33];\n"
                                "	mul.f32 	%f48, %f47, %f47;\n"
                                "	ld.global.f32 	%f49, [%rd34+-4];\n"
                                "	div.rn.f32 	%f50, %f48, %f49;\n"
                                "	ld.global.f32 	%f51, [%rd34];\n"
                                "	sub.f32 	%f52, %f51, %f50;\n"
                                "	st.global.f32 	[%rd34], %f52;\n"
                                "	ld.global.f32 	%f53, [%rd33+4];\n"
                                "	ld.global.f32 	%f54, [%rd32];\n"
                                "	mul.f32 	%f55, %f54, %f53;\n"
                                "	div.rn.f32 	%f56, %f55, %f52;\n"
                                "	ld.global.f32 	%f57, [%rd32+4];\n"
                                "	sub.f32 	%f58, %f57, %f56;\n"
                                "	st.global.f32 	[%rd32+4], %f58;\n"
                                "	ld.global.f32 	%f59, [%rd33+4];\n"
                                "	mul.f32 	%f60, %f59, %f59;\n"
                                "	ld.global.f32 	%f61, [%rd34];\n"
                                "	div.rn.f32 	%f62, %f60, %f61;\n"
                                "	ld.global.f32 	%f63, [%rd34+4];\n"
                                "	sub.f32 	%f64, %f63, %f62;\n"
                                "	st.global.f32 	[%rd34+4], %f64;\n"
                                "	ld.global.f32 	%f65, [%rd33+8];\n"
                                "	ld.global.f32 	%f66, [%rd32+4];\n"
                                "	mul.f32 	%f67, %f66, %f65;\n"
                                "	div.rn.f32 	%f68, %f67, %f64;\n"
                                "	ld.global.f32 	%f69, [%rd32+8];\n"
                                "	sub.f32 	%f70, %f69, %f68;\n"
                                "	st.global.f32 	[%rd32+8], %f70;\n"
                                "	ld.global.f32 	%f71, [%rd33+8];\n"
                                "	mul.f32 	%f72, %f71, %f71;\n"
                                "	ld.global.f32 	%f73, [%rd34+4];\n"
                                "	div.rn.f32 	%f74, %f72, %f73;\n"
                                "	ld.global.f32 	%f75, [%rd34+8];\n"
                                "	sub.f32 	%f76, %f75, %f74;\n"
                                "	st.global.f32 	[%rd34+8], %f76;\n"
                                "	ld.global.f32 	%f77, [%rd33+12];\n"
                                "	ld.global.f32 	%f78, [%rd32+8];\n"
                                "	mul.f32 	%f79, %f78, %f77;\n"
                                "	div.rn.f32 	%f80, %f79, %f76;\n"
                                "	ld.global.f32 	%f81, [%rd32+12];\n"
                                "	sub.f32 	%f82, %f81, %f80;\n"
                                "	st.global.f32 	[%rd32+12], %f82;\n"
                                "	ld.global.f32 	%f83, [%rd33+12];\n"
                                "	mul.f32 	%f84, %f83, %f83;\n"
                                "	ld.global.f32 	%f85, [%rd34+8];\n"
                                "	div.rn.f32 	%f86, %f84, %f85;\n"
                                "	ld.global.f32 	%f87, [%rd34+12];\n"
                                "	sub.f32 	%f88, %f87, %f86;\n"
                                "	st.global.f32 	[%rd34+12], %f88;\n"
                                "	add.s64 	%rd36, %rd36, 16;\n"
                                "	add.s64 	%rd35, %rd35, 16;\n"
                                "	add.s32 	%r28, %r28, 4;\n"
                                "	setp.lt.s32	%p8, %r28, %r15;\n"
                                "	@%p8 bra 	BB0_9;\n"
                                ""
                                "BB0_10:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11adi_kernel2iPfS_S_\n"
                                ".visible .entry _Z11adi_kernel2iPfS_S_(\n"
                                "	.param .u32 _Z11adi_kernel2iPfS_S__param_0,\n"
                                "	.param .u64 _Z11adi_kernel2iPfS_S__param_1,\n"
                                "	.param .u64 _Z11adi_kernel2iPfS_S__param_2,\n"
                                "	.param .u64 _Z11adi_kernel2iPfS_S__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<2>;\n"
                                "	.reg .f32 	%f<4>;\n"
                                "	.reg .b32 	%r<7>;\n"
                                "	.reg .b64 	%rd<8>;\n"
                                ""
                                "	ld.param.u32 	%r2, [_Z11adi_kernel2iPfS_S__param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z11adi_kernel2iPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z11adi_kernel2iPfS_S__param_3];\n"
                                "	mov.u32 	%r3, %ctaid.x;\n"
                                "	mov.u32 	%r4, %ntid.x;\n"
                                "	mov.u32 	%r5, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r4, %r3, %r5;\n"
                                "	setp.ge.s32	%p1, %r1, %r2;\n"
                                "	@%p1 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd2;\n"
                                "	shl.b32 	%r6, %r1, 10;\n"
                                "	mul.wide.s32 	%rd4, %r6, 4;\n"
                                "	add.s64 	%rd5, %rd3, %rd4;\n"
                                "	cvta.to.global.u64 	%rd6, %rd1;\n"
                                "	add.s64 	%rd7, %rd6, %rd4;\n"
                                "	ld.global.f32 	%f1, [%rd7+4092];\n"
                                "	ld.global.f32 	%f2, [%rd5+4092];\n"
                                "	div.rn.f32 	%f3, %f2, %f1;\n"
                                "	st.global.f32 	[%rd5+4092], %f3;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11adi_kernel3iPfS_S_\n"
                                ".visible .entry _Z11adi_kernel3iPfS_S_(\n"
                                "	.param .u32 _Z11adi_kernel3iPfS_S__param_0,\n"
                                "	.param .u64 _Z11adi_kernel3iPfS_S__param_1,\n"
                                "	.param .u64 _Z11adi_kernel3iPfS_S__param_2,\n"
                                "	.param .u64 _Z11adi_kernel3iPfS_S__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<50>;\n"
                                "	.reg .b32 	%r<44>;\n"
                                "	.reg .b64 	%rd<34>;\n"
                                ""
                                "	ld.param.u32 	%r16, [_Z11adi_kernel3iPfS_S__param_0];\n"
                                "	ld.param.u64 	%rd10, [_Z11adi_kernel3iPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd11, [_Z11adi_kernel3iPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd12, [_Z11adi_kernel3iPfS_S__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd11;\n"
                                "	cvta.to.global.u64 	%rd2, %rd10;\n"
                                "	cvta.to.global.u64 	%rd3, %rd12;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r16;\n"
                                "	@%p1 bra 	BB2_11;\n"
                                ""
                                "	add.s32 	%r5, %r16, -2;\n"
                                "	setp.lt.s32	%p2, %r5, 1;\n"
                                "	@%p2 bra 	BB2_11;\n"
                                ""
                                "	shl.b32 	%r6, %r4, 10;\n"
                                "	add.s32 	%r7, %r6, -3;\n"
                                "	add.s32 	%r8, %r6, 1021;\n"
                                "	and.b32  	%r20, %r5, 3;\n"
                                "	mov.u32 	%r40, 0;\n"
                                "	setp.eq.s32	%p3, %r20, 0;\n"
                                "	@%p3 bra 	BB2_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r20, 1;\n"
                                "	@%p4 bra 	BB2_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r20, 2;\n"
                                "	@%p5 bra 	BB2_6;\n"
                                ""
                                "	mul.wide.s32 	%rd13, %r6, 4;\n"
                                "	add.s64 	%rd14, %rd3, %rd13;\n"
                                "	mul.wide.s32 	%rd15, %r8, 4;\n"
                                "	add.s64 	%rd16, %rd2, %rd15;\n"
                                "	ld.global.f32 	%f1, [%rd16];\n"
                                "	ld.global.f32 	%f2, [%rd14+4084];\n"
                                "	mul.f32 	%f3, %f2, %f1;\n"
                                "	ld.global.f32 	%f4, [%rd14+4088];\n"
                                "	sub.f32 	%f5, %f4, %f3;\n"
                                "	add.s64 	%rd17, %rd1, %rd15;\n"
                                "	ld.global.f32 	%f6, [%rd17];\n"
                                "	div.rn.f32 	%f7, %f5, %f6;\n"
                                "	st.global.f32 	[%rd14+4088], %f7;\n"
                                "	mov.u32 	%r40, 1;\n"
                                ""
                                "BB2_6:\n"
                                "	mov.u32 	%r22, 1022;\n"
                                "	sub.s32 	%r23, %r22, %r40;\n"
                                "	add.s32 	%r24, %r23, %r6;\n"
                                "	mul.wide.s32 	%rd18, %r24, 4;\n"
                                "	add.s64 	%rd19, %rd3, %rd18;\n"
                                "	mov.u32 	%r25, 1024;\n"
                                "	sub.s32 	%r26, %r25, %r40;\n"
                                "	add.s32 	%r27, %r7, %r26;\n"
                                "	mul.wide.s32 	%rd20, %r27, 4;\n"
                                "	add.s64 	%rd21, %rd2, %rd20;\n"
                                "	ld.global.f32 	%f8, [%rd21];\n"
                                "	ld.global.f32 	%f9, [%rd19+-4];\n"
                                "	mul.f32 	%f10, %f9, %f8;\n"
                                "	ld.global.f32 	%f11, [%rd19];\n"
                                "	sub.f32 	%f12, %f11, %f10;\n"
                                "	sub.s32 	%r28, %r8, %r40;\n"
                                "	mul.wide.s32 	%rd22, %r28, 4;\n"
                                "	add.s64 	%rd23, %rd1, %rd22;\n"
                                "	ld.global.f32 	%f13, [%rd23];\n"
                                "	div.rn.f32 	%f14, %f12, %f13;\n"
                                "	st.global.f32 	[%rd19], %f14;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB2_7:\n"
                                "	mov.u32 	%r29, 1022;\n"
                                "	sub.s32 	%r30, %r29, %r40;\n"
                                "	add.s32 	%r31, %r30, %r6;\n"
                                "	mul.wide.s32 	%rd24, %r31, 4;\n"
                                "	add.s64 	%rd25, %rd3, %rd24;\n"
                                "	mov.u32 	%r32, 1024;\n"
                                "	sub.s32 	%r33, %r32, %r40;\n"
                                "	add.s32 	%r34, %r7, %r33;\n"
                                "	mul.wide.s32 	%rd26, %r34, 4;\n"
                                "	add.s64 	%rd27, %rd2, %rd26;\n"
                                "	ld.global.f32 	%f15, [%rd27];\n"
                                "	ld.global.f32 	%f16, [%rd25+-4];\n"
                                "	mul.f32 	%f17, %f16, %f15;\n"
                                "	ld.global.f32 	%f18, [%rd25];\n"
                                "	sub.f32 	%f19, %f18, %f17;\n"
                                "	sub.s32 	%r35, %r8, %r40;\n"
                                "	mul.wide.s32 	%rd28, %r35, 4;\n"
                                "	add.s64 	%rd29, %rd1, %rd28;\n"
                                "	ld.global.f32 	%f20, [%rd29];\n"
                                "	div.rn.f32 	%f21, %f19, %f20;\n"
                                "	st.global.f32 	[%rd25], %f21;\n"
                                "	add.s32 	%r40, %r40, 1;\n"
                                ""
                                "BB2_8:\n"
                                "	setp.lt.u32	%p6, %r5, 4;\n"
                                "	@%p6 bra 	BB2_11;\n"
                                ""
                                "	mul.lo.s32 	%r36, %r1, %r2;\n"
                                "	shl.b32 	%r37, %r3, 10;\n"
                                "	mad.lo.s32 	%r38, %r36, 1024, %r37;\n"
                                "	sub.s32 	%r39, %r38, %r40;\n"
                                "	mul.wide.s32 	%rd33, %r39, 4;\n"
                                ""
                                "BB2_10:\n"
                                "	add.s64 	%rd30, %rd3, %rd33;\n"
                                "	add.s64 	%rd31, %rd2, %rd33;\n"
                                "	ld.global.f32 	%f22, [%rd31+4084];\n"
                                "	ld.global.f32 	%f23, [%rd30+4084];\n"
                                "	mul.f32 	%f24, %f23, %f22;\n"
                                "	ld.global.f32 	%f25, [%rd30+4088];\n"
                                "	sub.f32 	%f26, %f25, %f24;\n"
                                "	add.s64 	%rd32, %rd1, %rd33;\n"
                                "	ld.global.f32 	%f27, [%rd32+4084];\n"
                                "	div.rn.f32 	%f28, %f26, %f27;\n"
                                "	st.global.f32 	[%rd30+4088], %f28;\n"
                                "	ld.global.f32 	%f29, [%rd31+4080];\n"
                                "	ld.global.f32 	%f30, [%rd30+4080];\n"
                                "	mul.f32 	%f31, %f30, %f29;\n"
                                "	ld.global.f32 	%f32, [%rd30+4084];\n"
                                "	sub.f32 	%f33, %f32, %f31;\n"
                                "	ld.global.f32 	%f34, [%rd32+4080];\n"
                                "	div.rn.f32 	%f35, %f33, %f34;\n"
                                "	st.global.f32 	[%rd30+4084], %f35;\n"
                                "	ld.global.f32 	%f36, [%rd31+4076];\n"
                                "	ld.global.f32 	%f37, [%rd30+4076];\n"
                                "	mul.f32 	%f38, %f37, %f36;\n"
                                "	ld.global.f32 	%f39, [%rd30+4080];\n"
                                "	sub.f32 	%f40, %f39, %f38;\n"
                                "	ld.global.f32 	%f41, [%rd32+4076];\n"
                                "	div.rn.f32 	%f42, %f40, %f41;\n"
                                "	st.global.f32 	[%rd30+4080], %f42;\n"
                                "	ld.global.f32 	%f43, [%rd31+4072];\n"
                                "	ld.global.f32 	%f44, [%rd30+4072];\n"
                                "	mul.f32 	%f45, %f44, %f43;\n"
                                "	ld.global.f32 	%f46, [%rd30+4076];\n"
                                "	sub.f32 	%f47, %f46, %f45;\n"
                                "	ld.global.f32 	%f48, [%rd32+4072];\n"
                                "	div.rn.f32 	%f49, %f47, %f48;\n"
                                "	st.global.f32 	[%rd30+4076], %f49;\n"
                                "	add.s64 	%rd33, %rd33, -16;\n"
                                "	add.s32 	%r40, %r40, 4;\n"
                                "	setp.lt.s32	%p7, %r40, %r5;\n"
                                "	@%p7 bra 	BB2_10;\n"
                                ""
                                "BB2_11:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11adi_kernel4iPfS_S_i\n"
                                ".visible .entry _Z11adi_kernel4iPfS_S_i(\n"
                                "	.param .u32 _Z11adi_kernel4iPfS_S_i_param_0,\n"
                                "	.param .u64 _Z11adi_kernel4iPfS_S_i_param_1,\n"
                                "	.param .u64 _Z11adi_kernel4iPfS_S_i_param_2,\n"
                                "	.param .u64 _Z11adi_kernel4iPfS_S_i_param_3,\n"
                                "	.param .u32 _Z11adi_kernel4iPfS_S_i_param_4\n"
                                "){\n"
                                "	.reg .pred 	%p<2>;\n"
                                "	.reg .f32 	%f<14>;\n"
                                "	.reg .b32 	%r<9>;\n"
                                "	.reg .b64 	%rd<11>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z11adi_kernel4iPfS_S_i_param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z11adi_kernel4iPfS_S_i_param_1];\n"
                                "	ld.param.u64 	%rd2, [_Z11adi_kernel4iPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd3, [_Z11adi_kernel4iPfS_S_i_param_3];\n"
                                "	ld.param.u32 	%r2, [_Z11adi_kernel4iPfS_S_i_param_4];\n"
                                "	mov.u32 	%r4, %ntid.x;\n"
                                "	mov.u32 	%r5, %ctaid.x;\n"
                                "	mov.u32 	%r6, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r4, %r5, %r6;\n"
                                "	setp.ge.s32	%p1, %r1, %r3;\n"
                                "	@%p1 bra 	BB3_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd4, %rd2;\n"
                                "	cvta.to.global.u64 	%rd5, %rd3;\n"
                                "	cvta.to.global.u64 	%rd6, %rd1;\n"
                                "	shl.b32 	%r7, %r2, 10;\n"
                                "	add.s32 	%r8, %r1, %r7;\n"
                                "	mul.wide.s32 	%rd7, %r8, 4;\n"
                                "	add.s64 	%rd8, %rd5, %rd7;\n"
                                "	add.s64 	%rd9, %rd6, %rd7;\n"
                                "	ld.global.f32 	%f1, [%rd9];\n"
                                "	ld.global.f32 	%f2, [%rd8+-4096];\n"
                                "	mul.f32 	%f3, %f2, %f1;\n"
                                "	add.s64 	%rd10, %rd4, %rd7;\n"
                                "	ld.global.f32 	%f4, [%rd10+-4096];\n"
                                "	div.rn.f32 	%f5, %f3, %f4;\n"
                                "	ld.global.f32 	%f6, [%rd8];\n"
                                "	sub.f32 	%f7, %f6, %f5;\n"
                                "	st.global.f32 	[%rd8], %f7;\n"
                                "	ld.global.f32 	%f8, [%rd9];\n"
                                "	mul.f32 	%f9, %f8, %f8;\n"
                                "	ld.global.f32 	%f10, [%rd10+-4096];\n"
                                "	div.rn.f32 	%f11, %f9, %f10;\n"
                                "	ld.global.f32 	%f12, [%rd10];\n"
                                "	sub.f32 	%f13, %f12, %f11;\n"
                                "	st.global.f32 	[%rd10], %f13;\n"
                                ""
                                "BB3_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11adi_kernel5iPfS_S_\n"
                                ".visible .entry _Z11adi_kernel5iPfS_S_(\n"
                                "	.param .u32 _Z11adi_kernel5iPfS_S__param_0,\n"
                                "	.param .u64 _Z11adi_kernel5iPfS_S__param_1,\n"
                                "	.param .u64 _Z11adi_kernel5iPfS_S__param_2,\n"
                                "	.param .u64 _Z11adi_kernel5iPfS_S__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<2>;\n"
                                "	.reg .f32 	%f<4>;\n"
                                "	.reg .b32 	%r<6>;\n"
                                "	.reg .b64 	%rd<8>;\n"
                                ""
                                "	ld.param.u32 	%r2, [_Z11adi_kernel5iPfS_S__param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z11adi_kernel5iPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z11adi_kernel5iPfS_S__param_3];\n"
                                "	mov.u32 	%r3, %ctaid.x;\n"
                                "	mov.u32 	%r4, %ntid.x;\n"
                                "	mov.u32 	%r5, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r4, %r3, %r5;\n"
                                "	setp.ge.s32	%p1, %r1, %r2;\n"
                                "	@%p1 bra 	BB4_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd2;\n"
                                "	mul.wide.s32 	%rd4, %r1, 4;\n"
                                "	add.s64 	%rd5, %rd3, %rd4;\n"
                                "	cvta.to.global.u64 	%rd6, %rd1;\n"
                                "	add.s64 	%rd7, %rd6, %rd4;\n"
                                "	ld.global.f32 	%f1, [%rd7+4190208];\n"
                                "	ld.global.f32 	%f2, [%rd5+4190208];\n"
                                "	div.rn.f32 	%f3, %f2, %f1;\n"
                                "	st.global.f32 	[%rd5+4190208], %f3;\n"
                                ""
                                "BB4_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11adi_kernel6iPfS_S_i\n"
                                ".visible .entry _Z11adi_kernel6iPfS_S_i(\n"
                                "	.param .u32 _Z11adi_kernel6iPfS_S_i_param_0,\n"
                                "	.param .u64 _Z11adi_kernel6iPfS_S_i_param_1,\n"
                                "	.param .u64 _Z11adi_kernel6iPfS_S_i_param_2,\n"
                                "	.param .u64 _Z11adi_kernel6iPfS_S_i_param_3,\n"
                                "	.param .u32 _Z11adi_kernel6iPfS_S_i_param_4\n"
                                "){\n"
                                "	.reg .pred 	%p<2>;\n"
                                "	.reg .f32 	%f<8>;\n"
                                "	.reg .b32 	%r<15>;\n"
                                "	.reg .b64 	%rd<12>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z11adi_kernel6iPfS_S_i_param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z11adi_kernel6iPfS_S_i_param_1];\n"
                                "	ld.param.u64 	%rd2, [_Z11adi_kernel6iPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd3, [_Z11adi_kernel6iPfS_S_i_param_3];\n"
                                "	ld.param.u32 	%r2, [_Z11adi_kernel6iPfS_S_i_param_4];\n"
                                "	mov.u32 	%r4, %ntid.x;\n"
                                "	mov.u32 	%r5, %ctaid.x;\n"
                                "	mov.u32 	%r6, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r4, %r5, %r6;\n"
                                "	setp.ge.s32	%p1, %r1, %r3;\n"
                                "	@%p1 bra 	BB5_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd4, %rd3;\n"
                                "	cvta.to.global.u64 	%rd5, %rd1;\n"
                                "	mov.u32 	%r7, 1022;\n"
                                "	sub.s32 	%r8, %r7, %r2;\n"
                                "	shl.b32 	%r9, %r8, 10;\n"
                                "	add.s32 	%r10, %r1, %r9;\n"
                                "	mul.wide.s32 	%rd6, %r10, 4;\n"
                                "	add.s64 	%rd7, %rd4, %rd6;\n"
                                "	mov.u32 	%r11, 1021;\n"
                                "	sub.s32 	%r12, %r11, %r2;\n"
                                "	shl.b32 	%r13, %r12, 10;\n"
                                "	add.s32 	%r14, %r1, %r13;\n"
                                "	mul.wide.s32 	%rd8, %r14, 4;\n"
                                "	add.s64 	%rd9, %rd5, %rd8;\n"
                                "	ld.global.f32 	%f1, [%rd9];\n"
                                "	ld.global.f32 	%f2, [%rd7+-4096];\n"
                                "	mul.f32 	%f3, %f2, %f1;\n"
                                "	ld.global.f32 	%f4, [%rd7];\n"
                                "	sub.f32 	%f5, %f4, %f3;\n"
                                "	cvta.to.global.u64 	%rd10, %rd2;\n"
                                "	add.s64 	%rd11, %rd10, %rd6;\n"
                                "	ld.global.f32 	%f6, [%rd11];\n"
                                "	div.rn.f32 	%f7, %f5, %f6;\n"
                                "	st.global.f32 	[%rd7], %f7;\n"
                                ""
                                "BB5_2:\n"
                                "	ret;\n"
                                "}\n";

void adiCuda(CUdevice device, int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), 
	DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_outputFromGpu,N,N,n,n)) {
	//DATA_TYPE* A_gpu;
	//DATA_TYPE* B_gpu;
	//DATA_TYPE* X_gpu;
    CUdeviceptr A_gpu, B_gpu, X_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL, func4 = NULL, func5 = NULL, func6 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemAlloc(&B_gpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemAlloc(&X_gpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(A_gpu, A, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(B_gpu, B, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(X_gpu, X, N * N * sizeof(DATA_TYPE)));

    //printf("  Loading module from ptx ...\n");
    cuError(cuModuleLoadData(&module, KERNEL_PTX));
    //printf("  Load module from ptx Success\n");

    //printf("  Loading function from module ...\n");
    cuError(cuModuleGetFunction(&func1, module, "_Z11adi_kernel1iPfS_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z11adi_kernel2iPfS_S_"));
    cuError(cuModuleGetFunction(&func3, module, "_Z11adi_kernel3iPfS_S_"));
    cuError(cuModuleGetFunction(&func4, module, "_Z11adi_kernel4iPfS_S_i"));
    cuError(cuModuleGetFunction(&func5, module, "_Z11adi_kernel5iPfS_S_"));
    cuError(cuModuleGetFunction(&func6, module, "_Z11adi_kernel6iPfS_S_i"));
    //printf("  Load function from module Success\n");

    unsigned grid_x = (size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_X) );
    
    void *args1[] = {&n, &A_gpu, &B_gpu, &X_gpu, NULL};
    SET_TIME(START)
    for (int t = 0; t < _PB_TSTEPS; t++) {
        cuError(cuLaunchKernel(func1, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
		cuError(cuLaunchKernel(func2, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
		cuError(cuLaunchKernel(func3, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
		for (int i1 = 1; i1 < _PB_N; i1++) {
            void *args2[] = {&n, &A_gpu, &B_gpu, &X_gpu, &i1, NULL}
            cuError(cuLaunchKernel(func4, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
		}
        cuError(cuLaunchKernel(func5, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
		for (int i1 = 0; i1 < _PB_N-2; i1++) {
            void *args2[] = {&n, &A_gpu, &B_gpu, &X_gpu, &i1, NULL}
            cuError(cuLaunchKernel(func6, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
		}
	}
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));

    cuError(cuMemcpyDtoH(B_outputFromGpu, B_gpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyDtoH(X_outputFromGpu, X_gpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
    cuError(cuMemFree(X_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

void adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n), DATA_TYPE POLYBENCH_2D(X, N, N, n, n)) {
    for (int t = 0; t < _PB_TSTEPS; t++) {
        for (int i1 = 0; i1 < _PB_N; i1++) {
            for (int i2 = 1; i2 < _PB_N; i2++) {
                X[i1][i2] = X[i1][i2] - X[i1][(i2 - 1)] * A[i1][i2] / B[i1][(i2 - 1)];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][(i2 - 1)];
            }
        }

        for (int i1 = 0; i1 < _PB_N; i1++) {
            X[i1][(N - 1)] = X[i1][(N - 1)] / B[i1][(N - 1)];
        }

        for (int i1 = 0; i1 < _PB_N; i1++) {
            for (int i2 = 0; i2 < _PB_N - 2; i2++) {
                X[i1][(N - i2 - 2)] = (X[i1][(N - 2 - i2)] - X[i1][(N - 2 - i2 - 1)] * A[i1][(N - i2 - 3)]) / B[i1][(N - 3 - i2)];
            }
        }

        for (int i1 = 1; i1 < _PB_N; i1++) {
            for (int i2 = 0; i2 < _PB_N; i2++) {
                X[i1][i2] = X[i1][i2] - X[(i1 - 1)][i2] * A[i1][i2] / B[(i1 - 1)][i2];
                B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[(i1 - 1)][i2];
            }
        }

        for (int i2 = 0; i2 < _PB_N; i2++) {
            X[(N - 1)][i2] = X[(N - 1)][i2] / B[(N - 1)][i2];
        }

        for (int i1 = 0; i1 < _PB_N - 2; i1++) {
            for (int i2 = 0; i2 < _PB_N; i2++) {
                X[(N - 2 - i1)][i2] = (X[(N - 2 - i1)][i2] - X[(N - i1 - 3)][i2] * A[(N - 3 - i1)][i2]) / B[(N - 2 - i1)][i2];
            }
        }
    }
}
void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n), DATA_TYPE POLYBENCH_2D(X, N, N, n, n)) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            X[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / N;
            A[i][j] = ((DATA_TYPE)(i - 1) * (j + 4) + 2) / N;
            B[i][j] = ((DATA_TYPE)(i + 3) * (j + 7) + 3) / N;
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_2D(B_cpu, N, N, n, n), DATA_TYPE POLYBENCH_2D(B_fromGpu, N, N, n, n), DATA_TYPE POLYBENCH_2D(X_cpu, N, N, n, n),
                    DATA_TYPE POLYBENCH_2D(X_fromGpu, N, N, n, n)) {
    int i, j, fail;
    fail = 0;
    // Compare b and x output on cpu and gpu
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (percentDiff(B_cpu[i][j], B_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (percentDiff(X_cpu[i][j], X_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }
    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
	int tsteps = TSTEPS;
	int n = N;

	GPU_argv_init();

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));

    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting adi on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        //fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START);
	    adiCuda(device, tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X_outputFromGpu));
        SET_TIME(GPU_END);
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test adi on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		adi(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	POLYBENCH_FREE_ARRAY(X);
	POLYBENCH_FREE_ARRAY(X_outputFromGpu);

	return 0;
}

#include "../include/polybench.c"