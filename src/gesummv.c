#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

#include "gesummv.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

#define RUN_ON_CPU

/*
Source Code:
__global__ void gesummv_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* tmp, DATA_TYPE* x, DATA_TYPE* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for(j = 0; j < _PB_N; j++)
        {
            tmp[i] += A[i * N + j] * x[j];
            y[i] += B[i * N + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta  * y[i];
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z14gesummv_kerneliffPfS_S_S_S_\n"
                                ""
                                ".visible .entry _Z14gesummv_kerneliffPfS_S_S_S_(\n"
                                "	.param .u32 _Z14gesummv_kerneliffPfS_S_S_S__param_0,\n"
                                "	.param .f32 _Z14gesummv_kerneliffPfS_S_S_S__param_1,\n"
                                "	.param .f32 _Z14gesummv_kerneliffPfS_S_S_S__param_2,\n"
                                "	.param .u64 _Z14gesummv_kerneliffPfS_S_S_S__param_3,\n"
                                "	.param .u64 _Z14gesummv_kerneliffPfS_S_S_S__param_4,\n"
                                "	.param .u64 _Z14gesummv_kerneliffPfS_S_S_S__param_5,\n"
                                "	.param .u64 _Z14gesummv_kerneliffPfS_S_S_S__param_6,\n"
                                "	.param .u64 _Z14gesummv_kerneliffPfS_S_S_S__param_7\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<63>;\n"
                                "	.reg .b32 	%r<28>;\n"
                                "	.reg .b64 	%rd<38>;\n"
                                ""
                                "	ld.param.u32 	%r13, [_Z14gesummv_kerneliffPfS_S_S_S__param_0];\n"
                                "	ld.param.f32 	%f1, [_Z14gesummv_kerneliffPfS_S_S_S__param_1];\n"
                                "	ld.param.f32 	%f2, [_Z14gesummv_kerneliffPfS_S_S_S__param_2];\n"
                                "	ld.param.u64 	%rd14, [_Z14gesummv_kerneliffPfS_S_S_S__param_3];\n"
                                "	ld.param.u64 	%rd15, [_Z14gesummv_kerneliffPfS_S_S_S__param_4];\n"
                                "	ld.param.u64 	%rd12, [_Z14gesummv_kerneliffPfS_S_S_S__param_5];\n"
                                "	ld.param.u64 	%rd16, [_Z14gesummv_kerneliffPfS_S_S_S__param_6];\n"
                                "	ld.param.u64 	%rd13, [_Z14gesummv_kerneliffPfS_S_S_S__param_7];\n"
                                "	cvta.to.global.u64 	%rd1, %rd15;\n"
                                "	cvta.to.global.u64 	%rd2, %rd16;\n"
                                "	cvta.to.global.u64 	%rd3, %rd14;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r13;\n"
                                "	@%p1 bra 	BB0_12;\n"
                                ""
                                "	cvta.to.global.u64 	%rd17, %rd12;\n"
                                "	mul.wide.s32 	%rd18, %r4, 4;\n"
                                "	add.s64 	%rd4, %rd17, %rd18;\n"
                                "	cvta.to.global.u64 	%rd19, %rd13;\n"
                                "	add.s64 	%rd5, %rd19, %rd18;\n"
                                "	setp.lt.s32	%p2, %r13, 1;\n"
                                "	@%p2 bra 	BB0_11;\n"
                                ""
                                "	shl.b32 	%r5, %r4, 12;\n"
                                "	and.b32  	%r17, %r13, 3;\n"
                                "	mov.u32 	%r24, 0;\n"
                                "	setp.eq.s32	%p3, %r17, 0;\n"
                                "	@%p3 bra 	BB0_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r17, 1;\n"
                                "	@%p4 bra 	BB0_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r17, 2;\n"
                                "	@%p5 bra 	BB0_6;\n"
                                ""
                                "	mul.wide.s32 	%rd20, %r5, 4;\n"
                                "	add.s64 	%rd21, %rd3, %rd20;\n"
                                "	ld.global.f32 	%f3, [%rd2];\n"
                                "	ld.global.f32 	%f4, [%rd21];\n"
                                "	ld.global.f32 	%f5, [%rd4];\n"
                                "	fma.rn.f32 	%f6, %f4, %f3, %f5;\n"
                                "	st.global.f32 	[%rd4], %f6;\n"
                                "	add.s64 	%rd22, %rd1, %rd20;\n"
                                "	ld.global.f32 	%f7, [%rd2];\n"
                                "	ld.global.f32 	%f8, [%rd22];\n"
                                "	ld.global.f32 	%f9, [%rd5];\n"
                                "	fma.rn.f32 	%f10, %f8, %f7, %f9;\n"
                                "	st.global.f32 	[%rd5], %f10;\n"
                                "	mov.u32 	%r24, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	add.s32 	%r19, %r24, %r5;\n"
                                "	mul.wide.s32 	%rd23, %r19, 4;\n"
                                "	add.s64 	%rd24, %rd3, %rd23;\n"
                                "	mul.wide.u32 	%rd25, %r24, 4;\n"
                                "	add.s64 	%rd26, %rd2, %rd25;\n"
                                "	ld.global.f32 	%f11, [%rd26];\n"
                                "	ld.global.f32 	%f12, [%rd24];\n"
                                "	ld.global.f32 	%f13, [%rd4];\n"
                                "	fma.rn.f32 	%f14, %f12, %f11, %f13;\n"
                                "	st.global.f32 	[%rd4], %f14;\n"
                                "	add.s64 	%rd27, %rd1, %rd23;\n"
                                "	ld.global.f32 	%f15, [%rd26];\n"
                                "	ld.global.f32 	%f16, [%rd27];\n"
                                "	ld.global.f32 	%f17, [%rd5];\n"
                                "	fma.rn.f32 	%f18, %f16, %f15, %f17;\n"
                                "	st.global.f32 	[%rd5], %f18;\n"
                                "	add.s32 	%r24, %r24, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	add.s32 	%r20, %r24, %r5;\n"
                                "	mul.wide.s32 	%rd28, %r20, 4;\n"
                                "	add.s64 	%rd29, %rd3, %rd28;\n"
                                "	mul.wide.s32 	%rd30, %r24, 4;\n"
                                "	add.s64 	%rd31, %rd2, %rd30;\n"
                                "	ld.global.f32 	%f19, [%rd31];\n"
                                "	ld.global.f32 	%f20, [%rd29];\n"
                                "	ld.global.f32 	%f21, [%rd4];\n"
                                "	fma.rn.f32 	%f22, %f20, %f19, %f21;\n"
                                "	st.global.f32 	[%rd4], %f22;\n"
                                "	add.s64 	%rd32, %rd1, %rd28;\n"
                                "	ld.global.f32 	%f23, [%rd31];\n"
                                "	ld.global.f32 	%f24, [%rd32];\n"
                                "	ld.global.f32 	%f25, [%rd5];\n"
                                "	fma.rn.f32 	%f26, %f24, %f23, %f25;\n"
                                "	st.global.f32 	[%rd5], %f26;\n"
                                "	add.s32 	%r24, %r24, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	setp.lt.u32	%p6, %r13, 4;\n"
                                "	@%p6 bra 	BB0_11;\n"
                                ""
                                "	mul.wide.s32 	%rd33, %r24, 4;\n"
                                "	add.s64 	%rd37, %rd2, %rd33;\n"
                                "	mul.lo.s32 	%r21, %r1, %r2;\n"
                                "	mad.lo.s32 	%r22, %r21, 4096, %r24;\n"
                                "	mad.lo.s32 	%r23, %r3, 4096, %r22;\n"
                                "	mul.wide.s32 	%rd36, %r23, 4;\n"
                                ""
                                "BB0_10:\n"
                                "	add.s64 	%rd34, %rd3, %rd36;\n"
                                "	ld.global.f32 	%f27, [%rd37];\n"
                                "	ld.global.f32 	%f28, [%rd34];\n"
                                "	ld.global.f32 	%f29, [%rd4];\n"
                                "	fma.rn.f32 	%f30, %f28, %f27, %f29;\n"
                                "	st.global.f32 	[%rd4], %f30;\n"
                                "	add.s64 	%rd35, %rd1, %rd36;\n"
                                "	ld.global.f32 	%f31, [%rd37];\n"
                                "	ld.global.f32 	%f32, [%rd35];\n"
                                "	ld.global.f32 	%f33, [%rd5];\n"
                                "	fma.rn.f32 	%f34, %f32, %f31, %f33;\n"
                                "	st.global.f32 	[%rd5], %f34;\n"
                                "	ld.global.f32 	%f35, [%rd37+4];\n"
                                "	ld.global.f32 	%f36, [%rd34+4];\n"
                                "	ld.global.f32 	%f37, [%rd4];\n"
                                "	fma.rn.f32 	%f38, %f36, %f35, %f37;\n"
                                "	st.global.f32 	[%rd4], %f38;\n"
                                "	ld.global.f32 	%f39, [%rd37+4];\n"
                                "	ld.global.f32 	%f40, [%rd35+4];\n"
                                "	ld.global.f32 	%f41, [%rd5];\n"
                                "	fma.rn.f32 	%f42, %f40, %f39, %f41;\n"
                                "	st.global.f32 	[%rd5], %f42;\n"
                                "	ld.global.f32 	%f43, [%rd37+8];\n"
                                "	ld.global.f32 	%f44, [%rd34+8];\n"
                                "	ld.global.f32 	%f45, [%rd4];\n"
                                "	fma.rn.f32 	%f46, %f44, %f43, %f45;\n"
                                "	st.global.f32 	[%rd4], %f46;\n"
                                "	ld.global.f32 	%f47, [%rd37+8];\n"
                                "	ld.global.f32 	%f48, [%rd35+8];\n"
                                "	ld.global.f32 	%f49, [%rd5];\n"
                                "	fma.rn.f32 	%f50, %f48, %f47, %f49;\n"
                                "	st.global.f32 	[%rd5], %f50;\n"
                                "	ld.global.f32 	%f51, [%rd37+12];\n"
                                "	ld.global.f32 	%f52, [%rd34+12];\n"
                                "	ld.global.f32 	%f53, [%rd4];\n"
                                "	fma.rn.f32 	%f54, %f52, %f51, %f53;\n"
                                "	st.global.f32 	[%rd4], %f54;\n"
                                "	ld.global.f32 	%f55, [%rd37+12];\n"
                                "	ld.global.f32 	%f56, [%rd35+12];\n"
                                "	ld.global.f32 	%f57, [%rd5];\n"
                                "	fma.rn.f32 	%f58, %f56, %f55, %f57;\n"
                                "	st.global.f32 	[%rd5], %f58;\n"
                                "	add.s64 	%rd37, %rd37, 16;\n"
                                "	add.s64 	%rd36, %rd36, 16;\n"
                                "	add.s32 	%r24, %r24, 4;\n"
                                "	setp.lt.s32	%p7, %r24, %r13;\n"
                                "	@%p7 bra 	BB0_10;\n"
                                ""
                                "BB0_11:\n"
                                "	ld.global.f32 	%f59, [%rd4];\n"
                                "	ld.global.f32 	%f60, [%rd5];\n"
                                "	mul.f32 	%f61, %f60, %f2;\n"
                                "	fma.rn.f32 	%f62, %f59, %f1, %f61;\n"
                                "	st.global.f32 	[%rd5], %f62;\n"
                                ""
                                "BB0_12:\n"
                                "	ret;\n"
                                "}\n";
void gesummv(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n), DATA_TYPE POLYBENCH_1D(tmp, N, n),
             DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n))
{
    int i, j;

    for (i = 0; i < _PB_N; i++)
    {
        tmp[i] = 0;
        y[i] = 0;
        for (j = 0; j < _PB_N; j++)
        {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
        }

        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

void init(int n, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
          DATA_TYPE POLYBENCH_1D(x, N, n))
{
    int i, j;

    *alpha = 43532;
    *beta = 12313;

    for (i = 0; i < n; i++)
    {
        x[i] = ((DATA_TYPE)i) / N;

        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
            B[i][j] = ((DATA_TYPE)i * j) / n;
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_1D(y, N, n), DATA_TYPE POLYBENCH_1D(y_outputFromGpu, N, n))
{
    int i, fail;
    fail = 0;

    for (i = 0; i < n; i++)
    {
        if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void gesummvCuda(CUdevice device, int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                 DATA_TYPE POLYBENCH_1D(tmp, N, n), DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n),
                 DATA_TYPE POLYBENCH_1D(y_outputFromGpu, N, n))
{
    CUdeviceptr A_gpu, B_gpu, x_gpu, y_gpu, tmp_gpu;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL;

    cuError(cuCtxCreate(&context, 0, device));

    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemAlloc(&x_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&y_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemAlloc(&tmp_gpu, sizeof(DATA_TYPE) * N));

    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemcpyHtoD(x_gpu, x, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(y_gpu, y, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyHtoD(tmp_gpu, tmp, sizeof(DATA_TYPE) * N));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z14gesummv_kerneliffPfS_S_S_S_"));

    unsigned grid_x = (unsigned int)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X));
    void *args1[] = {&n, &alpha, &beta, &A_gpu, &B_gpu, &tmp_gpu, &x_gpu, &y_gpu, NULL};

    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
    cuError(cuMemFree(x_gpu));
    cuError(cuMemFree(y_gpu));
    cuError(cuMemFree(tmp_gpu));
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
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu, DATA_TYPE, N, n);

    init(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x));

    int deviceCount = 0;
    CUdevice device = 0;
    char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        fprintf(stdout, "\nTesting gesummv on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        gesummvCuda(device, n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test gesummv on GPU device %d Success\n", i);
    }
#ifdef RUN_ON_CPU
    polybench_start_instruments;
    SET_TIME(CPU_START)
    gesummv(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));
    SET_TIME(CPU_END)
    fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
	compareResults(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

#else
    print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(y_outputFromGpu);

    return 0;
}

#include "../include/polybench.c"
