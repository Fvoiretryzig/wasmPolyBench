#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

#include "2DConvolution.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
Source Code:
__global__ void convolution2D_kernel(int ni, int nj, DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
    c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
    c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

    if ((i < _PB_NI-1) && (j < _PB_NJ-1) && (i > 0) && (j > 0))
    {
        B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)]
            + c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
            + c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
    }
}
*/

static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z20convolution2D_kerneliiPfS_\n"
                                ""
                                ".visible .entry _Z20convolution2D_kerneliiPfS_(\n"
                                "	.param .u32 _Z20convolution2D_kerneliiPfS__param_0,\n"
                                "	.param .u32 _Z20convolution2D_kerneliiPfS__param_1,\n"
                                "	.param .u64 _Z20convolution2D_kerneliiPfS__param_2,\n"
                                "	.param .u64 _Z20convolution2D_kerneliiPfS__param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<19>;\n"
                                "	.reg .b32 	%r<16>;\n"
                                "	.reg .b64 	%rd<10>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z20convolution2D_kerneliiPfS__param_0];\n"
                                "	ld.param.u32 	%r4, [_Z20convolution2D_kerneliiPfS__param_1];\n"
                                "	ld.param.u64 	%rd1, [_Z20convolution2D_kerneliiPfS__param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z20convolution2D_kerneliiPfS__param_3];\n"
                                "	mov.u32 	%r5, %ntid.x;\n"
                                "	mov.u32 	%r6, %ctaid.x;\n"
                                "	mov.u32 	%r7, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r5, %r6, %r7;\n"
                                "	mov.u32 	%r8, %ntid.y;\n"
                                "	mov.u32 	%r9, %ctaid.y;\n"
                                "	mov.u32 	%r10, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r8, %r9, %r10;\n"
                                "	add.s32 	%r11, %r3, -1;\n"
                                "	setp.ge.s32	%p1, %r2, %r11;\n"
                                "	add.s32 	%r12, %r4, -1;\n"
                                "	setp.ge.s32	%p2, %r1, %r12;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	setp.lt.s32	%p4, %r2, 1;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	setp.lt.s32	%p6, %r1, 1;\n"
                                "	or.pred  	%p7, %p5, %p6;\n"
                                "	@%p7 bra 	BB0_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd1;\n"
                                "	cvta.to.global.u64 	%rd4, %rd2;\n"
                                "	shl.b32 	%r13, %r2, 12;\n"
                                "	add.s32 	%r14, %r13, %r1;\n"
                                "	add.s32 	%r15, %r14, -4097;\n"
                                "	mul.wide.s32 	%rd5, %r15, 4;\n"
                                "	add.s64 	%rd6, %rd3, %rd5;\n"
                                "	ld.global.f32 	%f1, [%rd6];\n"
                                "	ld.global.f32 	%f2, [%rd6+4];\n"
                                "	mul.f32 	%f3, %f2, 0f3F000000;\n"
                                "	fma.rn.f32 	%f4, %f1, 0f3E4CCCCD, %f3;\n"
                                "	ld.global.f32 	%f5, [%rd6+8];\n"
                                "	fma.rn.f32 	%f6, %f5, 0fBF4CCCCD, %f4;\n"
                                "	ld.global.f32 	%f7, [%rd6+16384];\n"
                                "	fma.rn.f32 	%f8, %f7, 0fBE99999A, %f6;\n"
                                "	mul.wide.s32 	%rd7, %r14, 4;\n"
                                "	add.s64 	%rd8, %rd3, %rd7;\n"
                                "	ld.global.f32 	%f9, [%rd8];\n"
                                "	fma.rn.f32 	%f10, %f9, 0f3F19999A, %f8;\n"
                                "	ld.global.f32 	%f11, [%rd6+16392];\n"
                                "	fma.rn.f32 	%f12, %f11, 0fBF666666, %f10;\n"
                                "	ld.global.f32 	%f13, [%rd6+32768];\n"
                                "	fma.rn.f32 	%f14, %f13, 0f3ECCCCCD, %f12;\n"
                                "	ld.global.f32 	%f15, [%rd6+32772];\n"
                                "	fma.rn.f32 	%f16, %f15, 0f3F333333, %f14;\n"
                                "	ld.global.f32 	%f17, [%rd6+32776];\n"
                                "	fma.rn.f32 	%f18, %f17, 0f3DCCCCCD, %f16;\n"
                                "	add.s64 	%rd9, %rd4, %rd7;\n"
                                "	st.global.f32 	[%rd9], %f18;\n"
                                ""
                                "BB0_2:\n"
                                "	ret;\n"
                                "}\n";

void conv2D(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
    int i, j;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2;
    c21 = +0.5;
    c31 = -0.8;
    c12 = -0.3;
    c22 = +0.6;
    c32 = -0.9;
    c13 = +0.4;
    c23 = +0.7;
    c33 = +0.10;

    for (i = 1; i < _PB_NI - 1; ++i) // 0
    {
        for (j = 1; j < _PB_NJ - 1; ++j) // 1
        {
            B[i][j] = c11 * A[(i - 1)][(j - 1)] + c12 * A[(i + 0)][(j - 1)] + c13 * A[(i + 1)][(j - 1)] + c21 * A[(i - 1)][(j + 0)] + c22 * A[(i + 0)][(j + 0)] + c23 * A[(i + 1)][(j + 0)] + c31 * A[(i - 1)][(j + 1)] + c32 * A[(i + 0)][(j + 1)] + c33 * A[(i + 1)][(j + 1)];
        }
    }
}

void init(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
    int i, j;
    for (i = 0; i < ni; ++i)
    {
        for (j = 0; j < nj; ++j)
        {
            A[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj))
{
    int i, j, fail;
    fail = 0;
    // Compare outputs from CPU and GPU
    for (i = 1; i < (ni - 1); i++)
    {
        for (j = 1; j < (nj - 1); j++)
        {
            if (percentDiff(B[i][j], B_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }
    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void convolution2DCuda(CUdevice device, int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),  DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj))
{
    CUdeviceptr A_gpu, B_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));
    cuError(cuModuleGetFunction(&func1, module, "_Z20convolution2D_kerneliiPfS_"));

    unsigned grid_x = (size_t)ceil( ((float)NI) / ((float)DIM_THREAD_BLOCK_X) );
    unsigned grid_y = (size_t)ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_Y) );
    void *args1[] = {&ni, &nj, &A_gpu, &B_gpu, NULL};
    SET_TIME(START);
    cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    SET_TIME(END);

    cuError(cuMemcpyDtoH(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ));
	
    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
	/* Retrieve problem size */
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

	//initialize the arrays
	init(ni, nj, POLYBENCH_ARRAY(A));
	
	int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];
    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting 2DConvolution on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START);
        convolution2DCuda(device, ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));
        SET_TIME(GPU_END);
        fprintf(stdout, "GPU total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test 2DConvolution on GPU device %d Success\n", i);
    }

    #ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		conv2D(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(ni, nj, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
  	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	
	return 0;
}

#include "../include/polybench.c"