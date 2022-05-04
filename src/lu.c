#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "lu.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
Source code:
__global__ void lu_kernel1(int n, DATA_TYPE *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((j > k) && (j < _PB_N))
    {
        A[k*N + j] = A[k*N + j] / A[k*N + k];
    }
}


__global__ void lu_kernel2(int n, DATA_TYPE *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i > k) && (j > k) && (i < _PB_N) && (j < _PB_N))
    {
        A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z10lu_kernel1iPfi\n"
                                ""
                                ".visible .entry _Z10lu_kernel1iPfi(\n"
                                "	.param .u32 _Z10lu_kernel1iPfi_param_0,\n"
                                "	.param .u64 _Z10lu_kernel1iPfi_param_1,\n"
                                "	.param .u32 _Z10lu_kernel1iPfi_param_2\n"
                                "){\n"
                                "	.reg .pred 	%p<4>;\n"
                                "	.reg .f32 	%f<4>;\n"
                                "	.reg .b32 	%r<10>;\n"
                                "	.reg .b64 	%rd<7>;\n"
                                ""
                                ""
                                "	ld.param.u32 	%r3, [_Z10lu_kernel1iPfi_param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z10lu_kernel1iPfi_param_1];\n"
                                "	ld.param.u32 	%r2, [_Z10lu_kernel1iPfi_param_2];\n"
                                "	mov.u32 	%r4, %ntid.x;\n"
                                "	mov.u32 	%r5, %ctaid.x;\n"
                                "	mov.u32 	%r6, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r4, %r5, %r6;\n"
                                "	setp.le.s32	%p1, %r1, %r2;\n"
                                "	setp.ge.s32	%p2, %r1, %r3;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd2, %rd1;\n"
                                "	shl.b32 	%r7, %r2, 11;\n"
                                "	add.s32 	%r8, %r1, %r7;\n"
                                "	mul.wide.s32 	%rd3, %r8, 4;\n"
                                "	add.s64 	%rd4, %rd2, %rd3;\n"
                                "	mul.lo.s32 	%r9, %r2, 2049;\n"
                                "	mul.wide.s32 	%rd5, %r9, 4;\n"
                                "	add.s64 	%rd6, %rd2, %rd5;\n"
                                "	ld.global.f32 	%f1, [%rd6];\n"
                                "	ld.global.f32 	%f2, [%rd4];\n"
                                "	div.rn.f32 	%f3, %f2, %f1;\n"
                                "	st.global.f32 	[%rd4], %f3;\n"
                                ""
                                "BB0_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z10lu_kernel2iPfi\n"
                                ".visible .entry _Z10lu_kernel2iPfi(\n"
                                "	.param .u32 _Z10lu_kernel2iPfi_param_0,\n"
                                "	.param .u64 _Z10lu_kernel2iPfi_param_1,\n"
                                "	.param .u32 _Z10lu_kernel2iPfi_param_2\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<6>;\n"
                                "	.reg .b32 	%r<16>;\n"
                                "	.reg .b64 	%rd<9>;\n"
                                ""
                                ""
                                "	ld.param.u32 	%r4, [_Z10lu_kernel2iPfi_param_0];\n"
                                "	ld.param.u64 	%rd1, [_Z10lu_kernel2iPfi_param_1];\n"
                                "	ld.param.u32 	%r3, [_Z10lu_kernel2iPfi_param_2];\n"
                                "	mov.u32 	%r5, %ntid.x;\n"
                                "	mov.u32 	%r6, %ctaid.x;\n"
                                "	mov.u32 	%r7, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r5, %r6, %r7;\n"
                                "	mov.u32 	%r8, %ntid.y;\n"
                                "	mov.u32 	%r9, %ctaid.y;\n"
                                "	mov.u32 	%r10, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r8, %r9, %r10;\n"
                                "	setp.le.s32	%p1, %r2, %r3;\n"
                                "	setp.le.s32	%p2, %r1, %r3;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	setp.ge.s32	%p4, %r2, %r4;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	setp.ge.s32	%p6, %r1, %r4;\n"
                                "	or.pred  	%p7, %p5, %p6;\n"
                                "	@%p7 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd2, %rd1;\n"
                                "	shl.b32 	%r11, %r2, 11;\n"
                                "	add.s32 	%r12, %r11, %r1;\n"
                                "	mul.wide.s32 	%rd3, %r12, 4;\n"
                                "	add.s64 	%rd4, %rd2, %rd3;\n"
                                "	add.s32 	%r13, %r11, %r3;\n"
                                "	mul.wide.s32 	%rd5, %r13, 4;\n"
                                "	add.s64 	%rd6, %rd2, %rd5;\n"
                                "	shl.b32 	%r14, %r3, 11;\n"
                                "	add.s32 	%r15, %r1, %r14;\n"
                                "	mul.wide.s32 	%rd7, %r15, 4;\n"
                                "	add.s64 	%rd8, %rd2, %rd7;\n"
                                "	ld.global.f32 	%f1, [%rd8];\n"
                                "	ld.global.f32 	%f2, [%rd6];\n"
                                "	mul.f32 	%f3, %f2, %f1;\n"
                                "	ld.global.f32 	%f4, [%rd4];\n"
                                "	sub.f32 	%f5, %f4, %f3;\n"
                                "	st.global.f32 	[%rd4], %f5;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}\n";
void lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    for (int k = 0; k < _PB_N; k++)
    {
        for (int j = k + 1; j < _PB_N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }

        for (int i = k + 1; i < _PB_N; i++)
        {
            for (int j = k + 1; j < _PB_N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
    }
}

void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j + 1) / N;
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_2D(A_cpu, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n))
{
    int i, j, fail;
    fail = 0;

    // Compare a and b
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (percentDiff(A_cpu[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void luCuda(CUdevice device, int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n))
{
    CUdeviceptr AGpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL

                                 cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&AGpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(AGpu, A, N * N * sizeof(DATA_TYPE)));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z10lu_kernel1iPfi"));
    cuError(cuModuleGetFunction(&func2, module, "_Z10lu_kernel2iPfi"));

    dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
    dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
    dim3 grid1(1, 1, 1);
    dim3 grid2(1, 1, 1);

    SET_TIME(START)
    for (int k = 0; k < N; k++)
    {
        void *args1[] = {&n, &AGpu, &k, NULL};
        unsigned grid1_x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)));
        cuError(cuLaunchKernel(func1, grid1_x, 1, 1, DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y, 1, 0, NULL, args1, NULL));

        unsigned grid2_x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));
        unsigned grid2_y = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_Y)));

        cuError(cuLaunchKernel(func2, grid2_x, grid2_y, 1, DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y, 1, 0, NULL, args1, NULL));
    }
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(A_outputFromGpu, AGpu, N * N * sizeof(DATA_TYPE)));
    cuError(cuMemFree(AGpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(A));

    int deviceCount = 0;
    CUdevice device = 0;
    char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        fprintf(stdout, "\nTesting lu on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        luCuda(device, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test lu on GPU device %d Success\n", i);
    }
#ifdef RUN_ON_CPU
    polybench_start_instruments;
    SET_TIME(CPU_START)
    lu(n, POLYBENCH_ARRAY(A));
    SET_TIME(CPU_END)
    fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
    compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));
#else
    print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(A_outputFromGpu);

    return 0;
}

#include "../include/polybench.c"