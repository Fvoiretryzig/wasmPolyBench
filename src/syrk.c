#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>

#include "syrk.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

/*
Source code:
__global__ void syrk_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_NI) && (j < _PB_NI))
	{
		c[i * NI + j] *= beta;
		int k;		
		for(k=0; k < _PB_NJ; k++)
		{
			c[i * NI + j] += alpha * a[i * NJ + k] * a[j * NJ + k];
		}
	}
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
".target sm_30\n"
".address_size 64\n"
""
"	// .globl	_Z11syrk_kerneliiffPfS_\n"
""
".visible .entry _Z11syrk_kerneliiffPfS_(\n"
"	.param .u32 _Z11syrk_kerneliiffPfS__param_0,\n"
"	.param .u32 _Z11syrk_kerneliiffPfS__param_1,\n"
"	.param .f32 _Z11syrk_kerneliiffPfS__param_2,\n"
"	.param .f32 _Z11syrk_kerneliiffPfS__param_3,\n"
"	.param .u64 _Z11syrk_kerneliiffPfS__param_4,\n"
"	.param .u64 _Z11syrk_kerneliiffPfS__param_5\n"
"){\n"
"	.reg .pred 	%p<10>;\n"
"	.reg .f32 	%f<41>;\n"
"	.reg .b32 	%r<41>;\n"
"	.reg .b64 	%rd<26>;\n"
""
"	ld.param.u32 	%r18, [_Z11syrk_kerneliiffPfS__param_0];\n"
"	ld.param.u32 	%r17, [_Z11syrk_kerneliiffPfS__param_1];\n"
"	ld.param.f32 	%f10, [_Z11syrk_kerneliiffPfS__param_2];\n"
"	ld.param.f32 	%f11, [_Z11syrk_kerneliiffPfS__param_3];\n"
"	ld.param.u64 	%rd8, [_Z11syrk_kerneliiffPfS__param_4];\n"
"	ld.param.u64 	%rd7, [_Z11syrk_kerneliiffPfS__param_5];\n"
"	cvta.to.global.u64 	%rd25, %rd8;\n"
"	mov.u32 	%r19, %ntid.x;\n"
"	mov.u32 	%r1, %ctaid.x;\n"
"	mov.u32 	%r2, %tid.x;\n"
"	mad.lo.s32 	%r3, %r19, %r1, %r2;\n"
"	mov.u32 	%r4, %ntid.y;\n"
"	mov.u32 	%r5, %ctaid.y;\n"
"	mov.u32 	%r6, %tid.y;\n"
"	mad.lo.s32 	%r7, %r4, %r5, %r6;\n"
"	setp.ge.s32	%p1, %r7, %r18;\n"
"	setp.ge.s32	%p2, %r3, %r18;\n"
"	or.pred  	%p3, %p1, %p2;\n"
"	@%p3 bra 	BB0_11;\n"
""
"	cvta.to.global.u64 	%rd9, %rd7;\n"
"	shl.b32 	%r8, %r7, 10;\n"
"	add.s32 	%r20, %r8, %r3;\n"
"	mul.wide.s32 	%rd10, %r20, 4;\n"
"	add.s64 	%rd2, %rd9, %rd10;\n"
"	ld.global.f32 	%f12, [%rd2];\n"
"	mul.f32 	%f1, %f12, %f11;\n"
"	st.global.f32 	[%rd2], %f1;\n"
"	setp.lt.s32	%p4, %r17, 1;\n"
"	@%p4 bra 	BB0_11;\n"
""
"	shl.b32 	%r9, %r3, 10;\n"
"	and.b32  	%r24, %r17, 3;\n"
"	mov.u32 	%r37, 0;\n"
"	setp.eq.s32	%p5, %r24, 0;\n"
"	@%p5 bra 	BB0_8;\n"
""
"	setp.eq.s32	%p6, %r24, 1;\n"
"	@%p6 bra 	BB0_7;\n"
""
"	setp.eq.s32	%p7, %r24, 2;\n"
"	@%p7 bra 	BB0_6;\n"
""
"	mul.wide.s32 	%rd11, %r8, 4;\n"
"	add.s64 	%rd12, %rd25, %rd11;\n"
"	ld.global.f32 	%f13, [%rd12];\n"
"	mul.f32 	%f14, %f13, %f10;\n"
"	mul.wide.s32 	%rd13, %r9, 4;\n"
"	add.s64 	%rd14, %rd25, %rd13;\n"
"	ld.global.f32 	%f15, [%rd14];\n"
"	fma.rn.f32 	%f1, %f14, %f15, %f1;\n"
"	st.global.f32 	[%rd2], %f1;\n"
"	mov.u32 	%r37, 1;\n"
""
"BB0_6:\n"
"	add.s32 	%r26, %r37, %r8;\n"
"	mul.wide.s32 	%rd15, %r26, 4;\n"
"	add.s64 	%rd16, %rd25, %rd15;\n"
"	ld.global.f32 	%f16, [%rd16];\n"
"	mul.f32 	%f17, %f16, %f10;\n"
"	add.s32 	%r27, %r37, %r9;\n"
"	mul.wide.s32 	%rd17, %r27, 4;\n"
"	add.s64 	%rd18, %rd25, %rd17;\n"
"	ld.global.f32 	%f18, [%rd18];\n"
"	fma.rn.f32 	%f1, %f17, %f18, %f1;\n"
"	st.global.f32 	[%rd2], %f1;\n"
"	add.s32 	%r37, %r37, 1;\n"
""
"BB0_7:\n"
"	add.s32 	%r28, %r37, %r8;\n"
"	mul.wide.s32 	%rd19, %r28, 4;\n"
"	add.s64 	%rd20, %rd25, %rd19;\n"
"	ld.global.f32 	%f19, [%rd20];\n"
"	mul.f32 	%f20, %f19, %f10;\n"
"	add.s32 	%r29, %r37, %r9;\n"
"	mul.wide.s32 	%rd21, %r29, 4;\n"
"	add.s64 	%rd22, %rd25, %rd21;\n"
"	ld.global.f32 	%f21, [%rd22];\n"
"	fma.rn.f32 	%f1, %f20, %f21, %f1;\n"
"	st.global.f32 	[%rd2], %f1;\n"
"	add.s32 	%r37, %r37, 1;\n"
""
"BB0_8:\n"
"	setp.lt.u32	%p8, %r17, 4;\n"
"	@%p8 bra 	BB0_11;\n"
""
"	mul.lo.s32 	%r31, %r19, %r1;\n"
"	mad.lo.s32 	%r32, %r31, 1024, %r37;\n"
"	mul.lo.s32 	%r33, %r4, %r5;\n"
"	mad.lo.s32 	%r34, %r33, 1024, %r37;\n"
"	mad.lo.s32 	%r35, %r6, 1024, %r34;\n"
"	mul.wide.s32 	%rd3, %r35, 4;\n"
"	mad.lo.s32 	%r36, %r2, 1024, %r32;\n"
"	mul.wide.s32 	%rd4, %r36, 4;\n"
""
"BB0_10:\n"
"	add.s64 	%rd23, %rd25, %rd3;\n"
"	ld.global.f32 	%f22, [%rd23];\n"
"	mul.f32 	%f23, %f22, %f10;\n"
"	add.s64 	%rd24, %rd25, %rd4;\n"
"	ld.global.f32 	%f24, [%rd24];\n"
"	fma.rn.f32 	%f25, %f23, %f24, %f1;\n"
"	st.global.f32 	[%rd2], %f25;\n"
"	ld.global.f32 	%f26, [%rd23+4];\n"
"	mul.f32 	%f27, %f26, %f10;\n"
"	ld.global.f32 	%f28, [%rd24+4];\n"
"	fma.rn.f32 	%f29, %f27, %f28, %f25;\n"
"	st.global.f32 	[%rd2], %f29;\n"
"	ld.global.f32 	%f30, [%rd23+8];\n"
"	mul.f32 	%f31, %f30, %f10;\n"
"	ld.global.f32 	%f32, [%rd24+8];\n"
"	fma.rn.f32 	%f33, %f31, %f32, %f29;\n"
"	st.global.f32 	[%rd2], %f33;\n"
"	ld.global.f32 	%f34, [%rd23+12];\n"
"	mul.f32 	%f35, %f34, %f10;\n"
"	ld.global.f32 	%f36, [%rd24+12];\n"
"	fma.rn.f32 	%f1, %f35, %f36, %f33;\n"
"	st.global.f32 	[%rd2], %f1;\n"
"	add.s64 	%rd25, %rd25, 16;\n"
"	add.s32 	%r37, %r37, 4;\n"
"	setp.lt.s32	%p9, %r37, %r17;\n"
"	@%p9 bra 	BB0_10;\n"
""
"BB0_11:\n"
"	ret;\n"
"}\n";
void init_arrays(int ni, int nj,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni),
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;
	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < ni; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}
}


void syrk(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni))
{
	int i, j, k;
	
	/*  C := alpha*A*A' + beta*C */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			C[i][j] *= beta;
		}
	}
	
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			for (k = 0; k < _PB_NJ; k++)
			{
				C[i][j] += alpha * A[i][k] * A[j][k];
			}
		}
	}
}


void compareResults(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<ni; i++)
	{
		for (j=0; j<ni; j++)
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}
void syrkCuda(CUdevice device, int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), 
		DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
    CUdeviceptr A_gpu, C_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NI * NI));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NI * NI));

    unsigned grid_x = (size_t)(ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_X)));
    unsigned grid_y = (size_t)ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_Y));
	
    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z11syrk_kerneliiffPfS_"));
    void *args1[] = {&ni, &nj, &alpha, &beta, &A_gpu, &C_gpu, NULL};

    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NI));

    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(C_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NI,ni,ni);
  	POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NI,ni,ni);

	init_arrays(ni, nj, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
	
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting syrk on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        syrkCuda(device, ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test syrk on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		syrk(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(ni, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
  	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(C_outputFromGpu);

	return 0;
}

#include "../include/polybench.c"
