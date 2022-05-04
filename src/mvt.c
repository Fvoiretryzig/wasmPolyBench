#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "mvt.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
Source code:
__global__ void mvt_kernel1(int n, DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for(j=0; j < _PB_N; j++)
        {
            x1[i] += a[i * N + j] * y_1[j];
        }
    }
}


__global__ void mvt_kernel2(int n, DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for(j=0; j < _PB_N; j++)
        {
            x2[i] += a[j * N + i] * y_2[j];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z11mvt_kernel1iPfS_S_\n"
                                ""
                                ".visible .entry _Z11mvt_kernel1iPfS_S_(\n"
                                "	.param .u32 _Z11mvt_kernel1iPfS_S__param_0,\n"
                                "	.param .u64 _Z11mvt_kernel1iPfS_S__param_1,\n"
                                "	.param .u64 _Z11mvt_kernel1iPfS_S__param_2,\n"
                                "	.param .u64 _Z11mvt_kernel1iPfS_S__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<9>;\n"
                                "	.reg .f32 	%f<32>;\n"
                                "	.reg .b32 	%r<28>;\n"
                                "	.reg .b64 	%rd<29>;\n"
                                ""
                                "	ld.param.u32 	%r13, [_Z11mvt_kernel1iPfS_S__param_0];\n"
                                "	ld.param.u64 	%rd11, [_Z11mvt_kernel1iPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd10, [_Z11mvt_kernel1iPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd12, [_Z11mvt_kernel1iPfS_S__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd2, %rd11;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r13;\n"
                                "	setp.lt.s32	%p2, %r13, 1;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_12;\n"
                                ""
                                "	cvta.to.global.u64 	%rd13, %rd10;\n"
                                "	shl.b32 	%r5, %r4, 12;\n"
                                "	mul.wide.s32 	%rd14, %r4, 4;\n"
                                "	add.s64 	%rd3, %rd13, %rd14;\n"
                                "	and.b32  	%r15, %r13, 3;\n"
                                "	mov.u32 	%r27, 0;\n"
                                "	setp.eq.s32	%p4, %r15, 0;\n"
                                "	@%p4 bra 	BB0_9;\n"
                                ""
                                "	setp.eq.s32	%p5, %r15, 1;\n"
                                "	@%p5 bra 	BB0_5;\n"
                                "	bra.uni 	BB0_3;\n"
                                ""
                                "BB0_5:\n"
                                "	ld.global.f32 	%f30, [%rd3];\n"
                                "	mov.u32 	%r25, 0;\n"
                                "	bra.uni 	BB0_8;\n"
                                ""
                                "BB0_3:\n"
                                "	setp.ne.s32	%p6, %r15, 2;\n"
                                "	@%p6 bra 	BB0_6;\n"
                                ""
                                "	ld.global.f32 	%f29, [%rd3];\n"
                                "	mov.u32 	%r24, 0;\n"
                                "	bra.uni 	BB0_7;\n"
                                ""
                                "BB0_6:\n"
                                "	mul.wide.s32 	%rd15, %r5, 4;\n"
                                "	add.s64 	%rd16, %rd2, %rd15;\n"
                                "	ld.global.f32 	%f10, [%rd1];\n"
                                "	ld.global.f32 	%f11, [%rd16];\n"
                                "	ld.global.f32 	%f12, [%rd3];\n"
                                "	fma.rn.f32 	%f29, %f11, %f10, %f12;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	mov.u32 	%r24, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	add.s32 	%r19, %r24, %r5;\n"
                                "	mul.wide.s32 	%rd17, %r19, 4;\n"
                                "	add.s64 	%rd18, %rd2, %rd17;\n"
                                "	mul.wide.u32 	%rd19, %r24, 4;\n"
                                "	add.s64 	%rd20, %rd1, %rd19;\n"
                                "	ld.global.f32 	%f13, [%rd20];\n"
                                "	ld.global.f32 	%f14, [%rd18];\n"
                                "	fma.rn.f32 	%f30, %f14, %f13, %f29;\n"
                                "	st.global.f32 	[%rd3], %f30;\n"
                                "	add.s32 	%r25, %r24, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	add.s32 	%r20, %r25, %r5;\n"
                                "	mul.wide.s32 	%rd21, %r20, 4;\n"
                                "	add.s64 	%rd22, %rd2, %rd21;\n"
                                "	mul.wide.s32 	%rd23, %r25, 4;\n"
                                "	add.s64 	%rd24, %rd1, %rd23;\n"
                                "	ld.global.f32 	%f15, [%rd24];\n"
                                "	ld.global.f32 	%f16, [%rd22];\n"
                                "	fma.rn.f32 	%f17, %f16, %f15, %f30;\n"
                                "	st.global.f32 	[%rd3], %f17;\n"
                                "	add.s32 	%r27, %r25, 1;\n"
                                ""
                                "BB0_9:\n"
                                "	setp.lt.u32	%p7, %r13, 4;\n"
                                "	@%p7 bra 	BB0_12;\n"
                                ""
                                "	ld.global.f32 	%f31, [%rd3];\n"
                                "	mul.wide.s32 	%rd25, %r27, 4;\n"
                                "	add.s64 	%rd28, %rd1, %rd25;\n"
                                "	mul.lo.s32 	%r21, %r1, %r2;\n"
                                "	mad.lo.s32 	%r22, %r21, 4096, %r27;\n"
                                "	mad.lo.s32 	%r23, %r3, 4096, %r22;\n"
                                "	mul.wide.s32 	%rd26, %r23, 4;\n"
                                "	add.s64 	%rd27, %rd2, %rd26;\n"
                                ""
                                "BB0_11:\n"
                                "	ld.global.f32 	%f18, [%rd28];\n"
                                "	ld.global.f32 	%f19, [%rd27];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f31;\n"
                                "	st.global.f32 	[%rd3], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd28+4];\n"
                                "	ld.global.f32 	%f22, [%rd27+4];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd3], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd28+8];\n"
                                "	ld.global.f32 	%f25, [%rd27+8];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd3], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd28+12];\n"
                                "	ld.global.f32 	%f28, [%rd27+12];\n"
                                "	fma.rn.f32 	%f31, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd3], %f31;\n"
                                "	add.s64 	%rd28, %rd28, 16;\n"
                                "	add.s64 	%rd27, %rd27, 16;\n"
                                "	add.s32 	%r27, %r27, 4;\n"
                                "	setp.lt.s32	%p8, %r27, %r13;\n"
                                "	@%p8 bra 	BB0_11;\n"
                                ""
                                "BB0_12:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z11mvt_kernel2iPfS_S_\n"
                                ".visible .entry _Z11mvt_kernel2iPfS_S_(\n"
                                "	.param .u32 _Z11mvt_kernel2iPfS_S__param_0,\n"
                                "	.param .u64 _Z11mvt_kernel2iPfS_S__param_1,\n"
                                "	.param .u64 _Z11mvt_kernel2iPfS_S__param_2,\n"
                                "	.param .u64 _Z11mvt_kernel2iPfS_S__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<9>;\n"
                                "	.reg .f32 	%f<32>;\n"
                                "	.reg .b32 	%r<31>;\n"
                                "	.reg .b64 	%rd<26>;\n"
                                ""
                                "	ld.param.u32 	%r15, [_Z11mvt_kernel2iPfS_S__param_0];\n"
                                "	ld.param.u64 	%rd8, [_Z11mvt_kernel2iPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd7, [_Z11mvt_kernel2iPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd9, [_Z11mvt_kernel2iPfS_S__param_3];\n"
                                "	cvta.to.global.u64 	%rd1, %rd9;\n"
                                "	cvta.to.global.u64 	%rd2, %rd8;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r15;\n"
                                "	setp.lt.s32	%p2, %r15, 1;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB1_12;\n"
                                ""
                                "	cvta.to.global.u64 	%rd10, %rd7;\n"
                                "	mul.wide.s32 	%rd11, %r4, 4;\n"
                                "	add.s64 	%rd3, %rd10, %rd11;\n"
                                "	and.b32  	%r17, %r15, 3;\n"
                                "	mov.u32 	%r30, 0;\n"
                                "	setp.eq.s32	%p4, %r17, 0;\n"
                                "	@%p4 bra 	BB1_9;\n"
                                ""
                                "	setp.eq.s32	%p5, %r17, 1;\n"
                                "	@%p5 bra 	BB1_5;\n"
                                "	bra.uni 	BB1_3;\n"
                                ""
                                "BB1_5:\n"
                                "	ld.global.f32 	%f30, [%rd3];\n"
                                "	mov.u32 	%r27, 0;\n"
                                "	bra.uni 	BB1_8;\n"
                                ""
                                "BB1_3:\n"
                                "	setp.ne.s32	%p6, %r17, 2;\n"
                                "	@%p6 bra 	BB1_6;\n"
                                ""
                                "	ld.global.f32 	%f29, [%rd3];\n"
                                "	mov.u32 	%r26, 0;\n"
                                "	bra.uni 	BB1_7;\n"
                                ""
                                "BB1_6:\n"
                                "	add.s64 	%rd13, %rd2, %rd11;\n"
                                "	ld.global.f32 	%f10, [%rd1];\n"
                                "	ld.global.f32 	%f11, [%rd13];\n"
                                "	ld.global.f32 	%f12, [%rd3];\n"
                                "	fma.rn.f32 	%f29, %f11, %f10, %f12;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	mov.u32 	%r26, 1;\n"
                                ""
                                "BB1_7:\n"
                                "	shl.b32 	%r21, %r26, 12;\n"
                                "	add.s32 	%r22, %r21, %r4;\n"
                                "	mul.wide.s32 	%rd14, %r22, 4;\n"
                                "	add.s64 	%rd15, %rd2, %rd14;\n"
                                "	mul.wide.u32 	%rd16, %r26, 4;\n"
                                "	add.s64 	%rd17, %rd1, %rd16;\n"
                                "	ld.global.f32 	%f13, [%rd17];\n"
                                "	ld.global.f32 	%f14, [%rd15];\n"
                                "	fma.rn.f32 	%f30, %f14, %f13, %f29;\n"
                                "	st.global.f32 	[%rd3], %f30;\n"
                                "	add.s32 	%r27, %r26, 1;\n"
                                ""
                                "BB1_8:\n"
                                "	shl.b32 	%r23, %r27, 12;\n"
                                "	add.s32 	%r24, %r23, %r4;\n"
                                "	mul.wide.s32 	%rd18, %r24, 4;\n"
                                "	add.s64 	%rd19, %rd2, %rd18;\n"
                                "	mul.wide.s32 	%rd20, %r27, 4;\n"
                                "	add.s64 	%rd21, %rd1, %rd20;\n"
                                "	ld.global.f32 	%f15, [%rd21];\n"
                                "	ld.global.f32 	%f16, [%rd19];\n"
                                "	fma.rn.f32 	%f17, %f16, %f15, %f30;\n"
                                "	st.global.f32 	[%rd3], %f17;\n"
                                "	add.s32 	%r30, %r27, 1;\n"
                                ""
                                "BB1_9:\n"
                                "	setp.lt.u32	%p7, %r15, 4;\n"
                                "	@%p7 bra 	BB1_12;\n"
                                ""
                                "	ld.global.f32 	%f31, [%rd3];\n"
                                "	mul.wide.s32 	%rd22, %r30, 4;\n"
                                "	add.s64 	%rd25, %rd1, %rd22;\n"
                                "	mad.lo.s32 	%r29, %r30, 4096, %r4;\n"
                                ""
                                "BB1_11:\n"
                                "	mul.wide.s32 	%rd23, %r29, 4;\n"
                                "	add.s64 	%rd24, %rd2, %rd23;\n"
                                "	ld.global.f32 	%f18, [%rd25];\n"
                                "	ld.global.f32 	%f19, [%rd24];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f31;\n"
                                "	st.global.f32 	[%rd3], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd25+4];\n"
                                "	ld.global.f32 	%f22, [%rd24+16384];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd3], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd25+8];\n"
                                "	ld.global.f32 	%f25, [%rd24+32768];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd3], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd25+12];\n"
                                "	ld.global.f32 	%f28, [%rd24+49152];\n"
                                "	fma.rn.f32 	%f31, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd3], %f31;\n"
                                "	add.s64 	%rd25, %rd25, 16;\n"
                                "	add.s32 	%r29, %r29, 16384;\n"
                                "	add.s32 	%r30, %r30, 4;\n"
                                "	setp.lt.s32	%p8, %r30, %r15;\n"
                                "	@%p8 bra 	BB1_11;\n"
                                ""
                                "BB1_12:\n"
                                "	ret;\n"
                                "}\n";
void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        x1[i] = ((DATA_TYPE)i) / N;
        x2[i] = ((DATA_TYPE)i + 1) / N;
        y1[i] = ((DATA_TYPE)i + 3) / N;
        y2[i] = ((DATA_TYPE)i + 4) / N;
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
        }
    }
}

void runMvt(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
    int i, j;

    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < N; j++)
        {
            x1[i] = x1[i] + a[i][j] * y1[j];
        }
    }

    for (i = 0; i < _PB_N; i++)
    {
        for (j = 0; j < _PB_N; j++)
        {
            x2[i] = x2[i] + a[j][i] * y2[j];
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
    int i, fail;
    fail = 0;

    for (i = 0; i < n; i++)
    {
        if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }

        if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void mvtCuda(CUdevice device, int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y_1, N, n), DATA_TYPE POLYBENCH_1D(y_2, N, n),
             DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
    CUdeviceptr a_gpu, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&a_gpu, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemAlloc(&x1_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&x2_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&y_1_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&y_2_gpu, sizeof(DATA_TYPE) * N));

    cuError(cuMemcpyHtoD(a_gpu, a, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemcpyHtoD(x1_gpu, x1, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(x2_gpu, x2, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(y_1_gpu, y_1, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(y_2_gpu, y_2, sizeof(DATA_TYPE) * N));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z11mvt_kernel1iPfS_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z11mvt_kernel2iPfS_S_"));

    unsigned grid_x = (size_t)ceil((float)N / ((float)DIM_THREAD_BLOCK_X));
    void *args1[] = {&n, &a_gpu, &x1_gpu, &y_1_gpu, NULL};
    void *args2[] = {&n, &a_gpu, &x2_gpu, &y_2_gpu, NULL};

    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func2, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(x1_outputFromGpu, x1_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyDtoH(x2_outputFromGpu, x2_gpu, sizeof(DATA_TYPE) * N));

    cuError(cuMemFree(a_gpu));
    cuError(cuMemFree(x1_gpu));
    cuError(cuMemFree(x2_gpu));
    cuError(cuMemFree(y_1_gpu));
    cuError(cuMemFree(y_2_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x1_outputFromGpu, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x2_outputFromGpu, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

    int deviceCount = 0;
    CUdevice device = 0;
    char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        fprintf(stdout, "\nTesting mvt on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        mvtCuda(device, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test mvt on GPU device %d Success\n", i);
    }
#ifdef RUN_ON_CPU
    polybench_start_instruments;
    SET_TIME(CPU_START)
    runMvt(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));
    SET_TIME(CPU_END)
    fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
	compareResults(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(x2_outputFromGpu));
#else
    print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(x1);
    POLYBENCH_FREE_ARRAY(x2);
    POLYBENCH_FREE_ARRAY(x1_outputFromGpu);
    POLYBENCH_FREE_ARRAY(x2_outputFromGpu);
    POLYBENCH_FREE_ARRAY(y_1);
    POLYBENCH_FREE_ARRAY(y_2);

    return 0;
}

#include "../include/polybench.c"
