#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

#include "atax.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define RUN_ON_CPU

/*
Source Code:
__global__ void convolution3D_kernel(int ni, int nj, int nk, DATA_TYPE* A, DATA_TYPE* B, int i)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +2;  c21 = +5;  c31 = -8;
    c12 = -3;  c22 = +6;  c32 = -9;
    c13 = +4;  c23 = +7;  c33 = +10;


    if ((i < (_PB_NI-1)) && (j < (_PB_NJ-1)) &&  (k < (_PB_NK-1)) && (i > 0) && (j > 0) && (k > 0))
    {
        B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
                         +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
                         +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
                         +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]
                         +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]
                         +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]
                         +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]
                         +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z20convolution3D_kerneliiiPfS_i\n"
                                ""
                                ".visible .entry _Z20convolution3D_kerneliiiPfS_i(\n"
                                "	.param .u32 _Z20convolution3D_kerneliiiPfS_i_param_0,\n"
                                "	.param .u32 _Z20convolution3D_kerneliiiPfS_i_param_1,\n"
                                "	.param .u32 _Z20convolution3D_kerneliiiPfS_i_param_2,\n"
                                "	.param .u64 _Z20convolution3D_kerneliiiPfS_i_param_3,\n"
                                "	.param .u64 _Z20convolution3D_kerneliiiPfS_i_param_4,\n"
                                "	.param .u32 _Z20convolution3D_kerneliiiPfS_i_param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<12>;\n"
                                "	.reg .f32 	%f<27>;\n"
                                "	.reg .b32 	%r<21>;\n"
                                "	.reg .b64 	%rd<10>;\n"
                                ""
                                "	ld.param.u32 	%r4, [_Z20convolution3D_kerneliiiPfS_i_param_0];\n"
                                "	ld.param.u32 	%r5, [_Z20convolution3D_kerneliiiPfS_i_param_1];\n"
                                "	ld.param.u32 	%r6, [_Z20convolution3D_kerneliiiPfS_i_param_2];\n"
                                "	ld.param.u64 	%rd1, [_Z20convolution3D_kerneliiiPfS_i_param_3];\n"
                                "	ld.param.u64 	%rd2, [_Z20convolution3D_kerneliiiPfS_i_param_4];\n"
                                "	ld.param.u32 	%r3, [_Z20convolution3D_kerneliiiPfS_i_param_5];\n"
                                "	mov.u32 	%r7, %ntid.x;\n"
                                "	mov.u32 	%r8, %ctaid.x;\n"
                                "	mov.u32 	%r9, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r7, %r8, %r9;\n"
                                "	mov.u32 	%r10, %ntid.y;\n"
                                "	mov.u32 	%r11, %ctaid.y;\n"
                                "	mov.u32 	%r12, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r10, %r11, %r12;\n"
                                "	add.s32 	%r13, %r4, -1;\n"
                                "	setp.le.s32	%p1, %r13, %r3;\n"
                                "	add.s32 	%r14, %r5, -1;\n"
                                "	setp.ge.s32	%p2, %r2, %r14;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	add.s32 	%r15, %r6, -1;\n"
                                "	setp.ge.s32	%p4, %r1, %r15;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	setp.lt.s32	%p6, %r3, 1;\n"
                                "	or.pred  	%p7, %p5, %p6;\n"
                                "	setp.lt.s32	%p8, %r2, 1;\n"
                                "	or.pred  	%p9, %p7, %p8;\n"
                                "	setp.lt.s32	%p10, %r1, 1;\n"
                                "	or.pred  	%p11, %p9, %p10;\n"
                                "	@%p11 bra 	BB0_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd1;\n"
                                "	cvta.to.global.u64 	%rd4, %rd2;\n"
                                "	shl.b32 	%r16, %r3, 16;\n"
                                "	add.s32 	%r17, %r1, %r16;\n"
                                "	shl.b32 	%r18, %r2, 8;\n"
                                "	add.s32 	%r19, %r17, %r18;\n"
                                "	add.s32 	%r20, %r19, -65793;\n"
                                "	mul.wide.s32 	%rd5, %r20, 4;\n"
                                "	add.s64 	%rd6, %rd3, %rd5;\n"
                                "	ld.global.f32 	%f1, [%rd6];\n"
                                "	ld.global.f32 	%f2, [%rd6+524288];\n"
                                "	mul.f32 	%f3, %f2, 0f40800000;\n"
                                "	fma.rn.f32 	%f4, %f1, 0f40000000, %f3;\n"
                                "	fma.rn.f32 	%f5, %f1, 0f40A00000, %f4;\n"
                                "	fma.rn.f32 	%f6, %f2, 0f40E00000, %f5;\n"
                                "	fma.rn.f32 	%f7, %f1, 0fC1000000, %f6;\n"
                                "	fma.rn.f32 	%f8, %f2, 0f41200000, %f7;\n"
                                "	ld.global.f32 	%f9, [%rd6+262148];\n"
                                "	fma.rn.f32 	%f10, %f9, 0fC0400000, %f8;\n"
                                "	mul.wide.s32 	%rd7, %r19, 4;\n"
                                "	add.s64 	%rd8, %rd3, %rd7;\n"
                                "	ld.global.f32 	%f11, [%rd8];\n"
                                "	fma.rn.f32 	%f12, %f11, 0f40C00000, %f10;\n"
                                "	ld.global.f32 	%f13, [%rd6+264196];\n"
                                "	fma.rn.f32 	%f14, %f13, 0fC1100000, %f12;\n"
                                "	ld.global.f32 	%f15, [%rd6+8];\n"
                                "	fma.rn.f32 	%f16, %f15, 0f40000000, %f14;\n"
                                "	ld.global.f32 	%f17, [%rd6+524296];\n"
                                "	fma.rn.f32 	%f18, %f17, 0f40800000, %f16;\n"
                                "	ld.global.f32 	%f19, [%rd6+1032];\n"
                                "	fma.rn.f32 	%f20, %f19, 0f40A00000, %f18;\n"
                                "	ld.global.f32 	%f21, [%rd6+525320];\n"
                                "	fma.rn.f32 	%f22, %f21, 0f40E00000, %f20;\n"
                                "	ld.global.f32 	%f23, [%rd6+2056];\n"
                                "	fma.rn.f32 	%f24, %f23, 0fC1000000, %f22;\n"
                                "	ld.global.f32 	%f25, [%rd6+526344];\n"
                                "	fma.rn.f32 	%f26, %f25, 0f41200000, %f24;\n"
                                "	add.s64 	%rd9, %rd4, %rd7;\n"
                                "	st.global.f32 	[%rd9], %f26;\n"
                                ""
                                "BB0_2:\n"
                                "	ret;\n"
                                "}\n";
void conv3D(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +2;
    c21 = +5;
    c31 = -8;
    c12 = -3;
    c22 = +6;
    c32 = -9;
    c13 = +4;
    c23 = +7;
    c33 = +10;

    for (i = 1; i < _PB_NI - 1; ++i) // 0
    {
        for (j = 1; j < _PB_NJ - 1; ++j) // 1
        {
            for (k = 1; k < _PB_NK - 1; ++k) // 2
            {
                B[i][j][k] = c11 * A[(i - 1)][(j - 1)][(k - 1)] + c13 * A[(i + 1)][(j - 1)][(k - 1)] + c21 * A[(i - 1)][(j - 1)][(k - 1)] + c23 * A[(i + 1)][(j - 1)][(k - 1)] + c31 * A[(i - 1)][(j - 1)][(k - 1)] + c33 * A[(i + 1)][(j - 1)][(k - 1)] + c12 * A[(i + 0)][(j - 1)][(k + 0)] + c22 * A[(i + 0)][(j + 0)][(k + 0)] + c32 * A[(i + 0)][(j + 1)][(k + 0)] + c11 * A[(i - 1)][(j - 1)][(k + 1)] + c13 * A[(i + 1)][(j - 1)][(k + 1)] + c21 * A[(i - 1)][(j + 0)][(k + 1)] + c23 * A[(i + 1)][(j + 0)][(k + 1)] + c31 * A[(i - 1)][(j + 1)][(k + 1)] + c33 * A[(i + 1)][(j + 1)][(k + 1)];
            }
        }
    }
}

void init(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k;

    for (i = 0; i < ni; ++i)
    {
        for (j = 0; j < nj; ++j)
        {
            for (k = 0; k < nk; ++k)
            {
                A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
            }
        }
    }
}

void compareResults(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k, fail;
    fail = 0;

    // Compare result from cpu and gpu
    for (i = 1; i < ni - 1; ++i) // 0
    {
        for (j = 1; j < nj - 1; ++j) // 1
        {
            for (k = 1; k < nk - 1; ++k) // 2
            {
                if (percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
                {
                    fail++;
                }
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void convolution3DCuda(CUdevice device, int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
    CUdeviceptr A_gpu, B_gpu;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK));
    cuError(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK));
    cuError(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ * NK));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z20convolution3D_kerneliiiPfS_i"));
	
    unsigned grid_x = (size_t)(ceil( ((float)NK) / ((float)DIM_THREAD_BLOCK_X) );
    unsigned grid_y = (size_t)(ceil( ((float)NJ) / ((float)DIM_THREAD_BLOCK_Y) ));
	
    
    SET_TIME(START)
	int i;
	for (i = 1; i < _PB_NI - 1; ++i) // 0
	{
        void *args1[] = {&ni, &nj, &nk, &A_gpu, &B_gpu, &i, NULL};
        cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
		convolution3D_kernel<<< grid, block >>>(ni, nj, nk, A_gpu, B_gpu, i);
	}
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK));
	cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
int main()
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;

	POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,NK,ni,nj,nk);
	POLYBENCH_3D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,NK,ni,nj,nk);
	POLYBENCH_3D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,NK,ni,nj,nk);

	init(ni, nj, nk, POLYBENCH_ARRAY(A));
	
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting convolution3D on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        convolution3DCuda(device, ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test convolution3D on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		conv3D(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));        
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
        compareResults(ni, nj, nk, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    	return 0;
}

#include "../include/polybench.c"
