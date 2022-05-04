#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "jacobi2D.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define TSTEPS 20
#define N 1000

#define RUN_ON_CPU

/*
Source Code:
__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
    {
        B[i*N + j] = 0.2f * (A[i*N + j] + A[i*N + (j-1)] + A[i*N + (1 + j)] + A[(1 + i)*N + j] + A[(i-1)*N + j]);
    }
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (_PB_N-1)) && (j >= 1) && (j < (_PB_N-1)))
    {
        A[i*N + j] = B[i*N + j];
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z21runJacobiCUDA_kernel1iPfS_\n"
                                ""
                                ".visible .entry _Z21runJacobiCUDA_kernel1iPfS_(\n"
                                "	.param .u32 _Z21runJacobiCUDA_kernel1iPfS__param_0,\n"
                                "	.param .u64 _Z21runJacobiCUDA_kernel1iPfS__param_1,\n"
                                "	.param .u64 _Z21runJacobiCUDA_kernel1iPfS__param_2\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<11>;\n"
                                "	.reg .b32 	%r<12>;\n"
                                "	.reg .b64 	%rd<8>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z21runJacobiCUDA_kernel1iPfS__param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z21runJacobiCUDA_kernel1iPfS__param_1];\n"
                                "	ld.param.u64 	%rd2, [_Z21runJacobiCUDA_kernel1iPfS__param_2];\n"
                                "	mov.u32 	%r4, %ntid.y;\n"
                                "	mov.u32 	%r5, %ctaid.y;\n"
                                "	mov.u32 	%r6, %tid.y;\n"
                                "	mad.lo.s32 	%r1, %r4, %r5, %r6;\n"
                                "	mov.u32 	%r7, %ntid.x;\n"
                                "	mov.u32 	%r8, %ctaid.x;\n"
                                "	mov.u32 	%r9, %tid.x;\n"
                                "	mad.lo.s32 	%r2, %r7, %r8, %r9;\n"
                                "	setp.lt.s32	%p1, %r1, 1;\n"
                                "	add.s32 	%r10, %r3, -1;\n"
                                "	setp.ge.s32	%p2, %r1, %r10;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	setp.lt.s32	%p4, %r2, 1;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	setp.ge.s32	%p6, %r2, %r10;\n"
                                "	or.pred  	%p7, %p5, %p6;\n"
                                "	@%p7 bra 	BB0_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd1;\n"
                                "	cvta.to.global.u64 	%rd4, %rd2;\n"
                                "	mad.lo.s32 	%r11, %r1, 1000, %r2;\n"
                                "	mul.wide.s32 	%rd5, %r11, 4;\n"
                                "	add.s64 	%rd6, %rd3, %rd5;\n"
                                "	ld.global.f32 	%f1, [%rd6+-4];\n"
                                "	ld.global.f32 	%f2, [%rd6];\n"
                                "	add.f32 	%f3, %f2, %f1;\n"
                                "	ld.global.f32 	%f4, [%rd6+4];\n"
                                "	add.f32 	%f5, %f3, %f4;\n"
                                "	ld.global.f32 	%f6, [%rd6+4000];\n"
                                "	add.f32 	%f7, %f5, %f6;\n"
                                "	ld.global.f32 	%f8, [%rd6+-4000];\n"
                                "	add.f32 	%f9, %f7, %f8;\n"
                                "	mul.f32 	%f10, %f9, 0f3E4CCCCD;\n"
                                "	add.s64 	%rd7, %rd4, %rd5;\n"
                                "	st.global.f32 	[%rd7], %f10;\n"
                                ""
                                "BB0_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z21runJacobiCUDA_kernel2iPfS_\n"
                                ".visible .entry _Z21runJacobiCUDA_kernel2iPfS_(\n"
                                "	.param .u32 _Z21runJacobiCUDA_kernel2iPfS__param_0,\n"
                                "	.param .u64 _Z21runJacobiCUDA_kernel2iPfS__param_1,\n"
                                "	.param .u64 _Z21runJacobiCUDA_kernel2iPfS__param_2\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<2>;\n"
                                "	.reg .b32 	%r<12>;\n"
                                "	.reg .b64 	%rd<8>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z21runJacobiCUDA_kernel2iPfS__param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z21runJacobiCUDA_kernel2iPfS__param_1];\n"
                                "	ld.param.u64 	%rd2, [_Z21runJacobiCUDA_kernel2iPfS__param_2];\n"
                                "	mov.u32 	%r4, %ctaid.y;\n"
                                "	mov.u32 	%r5, %ntid.y;\n"
                                "	mov.u32 	%r6, %tid.y;\n"
                                "	mad.lo.s32 	%r1, %r5, %r4, %r6;\n"
                                "	mov.u32 	%r7, %ntid.x;\n"
                                "	mov.u32 	%r8, %ctaid.x;\n"
                                "	mov.u32 	%r9, %tid.x;\n"
                                "	mad.lo.s32 	%r2, %r7, %r8, %r9;\n"
                                "	setp.lt.s32	%p1, %r1, 1;\n"
                                "	add.s32 	%r10, %r3, -1;\n"
                                "	setp.ge.s32	%p2, %r1, %r10;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	setp.lt.s32	%p4, %r2, 1;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	setp.ge.s32	%p6, %r2, %r10;\n"
                                "	or.pred  	%p7, %p5, %p6;\n"
                                "	@%p7 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd2;\n"
                                "	mad.lo.s32 	%r11, %r1, 1000, %r2;\n"
                                "	mul.wide.s32 	%rd4, %r11, 4;\n"
                                "	add.s64 	%rd5, %rd3, %rd4;\n"
                                "	ld.global.f32 	%f1, [%rd5];\n"
                                "	cvta.to.global.u64 	%rd6, %rd1;\n"
                                "	add.s64 	%rd7, %rd6, %rd4;\n"
                                "	st.global.f32 	[%rd7], %f1;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}\n";

void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 10) / N;
            B[i][j] = ((DATA_TYPE)(i - 4) * (j - 1) + 11) / N;
        }
    }
}

void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
    for (int t = 0; t < _PB_TSTEPS; t++)
    {
        for (int i = 1; i < _PB_N - 1; i++)
        {
            for (int j = 1; j < _PB_N - 1; j++)
            {
                B[i][j] = 0.2f * (A[i][j] + A[i][(j - 1)] + A[i][(1 + j)] + A[(1 + i)][j] + A[(i - 1)][j]);
            }
        }

        for (int i = 1; i < _PB_N - 1; i++)
        {
            for (int j = 1; j < _PB_N - 1; j++)
            {
                A[i][j] = B[i][j];
            }
        }
    }
}
void compareResults(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu, N, N, n, n), DATA_TYPE POLYBENCH_2D(b, N, N, n, n), DATA_TYPE POLYBENCH_2D(b_outputFromGpu, N, N, n, n))
{
    int i, j, fail;
    fail = 0;

    // Compare output from CPU and GPU
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (percentDiff(b[i][j], b_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void runJacobi2DCUDA(CUdevice device, int tsteps, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n), DATA_TYPE POLYBENCH_2D(B_outputFromGpu, N, N, n, n))
{
    CUdeviceptr Agpu, Bgpu;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&Agpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemAlloc(&Bgpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(Agpu, A, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(Bgpu, B, N * N * sizeof(DATA_TYPE)));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z21runJacobiCUDA_kernel1iPfS_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z21runJacobiCUDA_kernel2iPfS_"));

    unsigned grid_x = (unsigned int)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X));
    unsigned grid_y = (unsigned int)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_Y));

    SET_TIME(START)
    void *args1[] = {&n, &Agpu, &Bgpu, NULL};
    for (int t = 0; t < _PB_TSTEPS; t++)
    {
        cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
        cuError(cuLaunchKernel(func2, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    }
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N * N));
    cuError(cuMemcpyDtoH(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N * N));

    cuError(cuMemFree(Agpu));
    cuError(cuMemFree(Bgpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
int main()
{
    /* Retrieve problem size. */
    int n = N;
    int tsteps = TSTEPS;

    POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

    int deviceCount = 0;
    CUdevice device = 0;
    char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        fprintf(stdout, "\nTesting atax on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        runJacobi2DCUDA(device, tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test atax on GPU device %d Success\n", i);
    }
#ifdef RUN_ON_CPU
    polybench_start_instruments;
    SET_TIME(CPU_START)
    runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
    SET_TIME(CPU_END)
    fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
	compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));
#else
    print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu);
    POLYBENCH_FREE_ARRAY(b);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu);

    return 0;
}

#include "../include/polybench.c"
