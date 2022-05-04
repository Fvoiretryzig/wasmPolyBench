#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "jacobi1D.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define RUN_ON_CPU

/*
Source code:
__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i > 0) && (i < (_PB_N-1)))
	{
		B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
	}
}


__global__ void runJacobiCUDA_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((j > 0) && (j < (_PB_N-1)))
	{
		A[j] = B[j];
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
"	.reg .pred 	%p<4>;\n"
"	.reg .f32 	%f<7>;\n"
"	.reg .b32 	%r<7>;\n"
"	.reg .f64 	%fd<3>;\n"
"	.reg .b64 	%rd<8>;\n"
""
"	ld.param.u32 	%r2, [_Z21runJacobiCUDA_kernel1iPfS__param_0];\n"
"	ld.param.u64 	%rd1, [_Z21runJacobiCUDA_kernel1iPfS__param_1];\n"
"	ld.param.u64 	%rd2, [_Z21runJacobiCUDA_kernel1iPfS__param_2];\n"
"	mov.u32 	%r3, %ntid.x;\n"
"	mov.u32 	%r4, %ctaid.x;\n"
"	mov.u32 	%r5, %tid.x;\n"
"	mad.lo.s32 	%r1, %r3, %r4, %r5;\n"
"	setp.lt.s32	%p1, %r1, 1;\n"
"	add.s32 	%r6, %r2, -1;\n"
"	setp.ge.s32	%p2, %r1, %r6;\n"
"	or.pred  	%p3, %p1, %p2;\n"
"	@%p3 bra 	BB0_2;\n"
""
"	cvta.to.global.u64 	%rd3, %rd1;\n"
"	cvta.to.global.u64 	%rd4, %rd2;\n"
"	mul.wide.s32 	%rd5, %r1, 4;\n"
"	add.s64 	%rd6, %rd3, %rd5;\n"
"	ld.global.f32 	%f1, [%rd6];\n"
"	ld.global.f32 	%f2, [%rd6+-4];\n"
"	add.f32 	%f3, %f2, %f1;\n"
"	ld.global.f32 	%f4, [%rd6+4];\n"
"	add.f32 	%f5, %f3, %f4;\n"
"	cvt.f64.f32	%fd1, %f5;\n"
"	mul.f64 	%fd2, %fd1, 0d3FD555475A31A4BE;\n"
"	cvt.rn.f32.f64	%f6, %fd2;\n"
"	add.s64 	%rd7, %rd4, %rd5;\n"
"	st.global.f32 	[%rd7], %f6;\n"
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
"	.reg .pred 	%p<4>;\n"
"	.reg .f32 	%f<2>;\n"
"	.reg .b32 	%r<7>;\n"
"	.reg .b64 	%rd<8>;\n"
""
"	ld.param.u32 	%r2, [_Z21runJacobiCUDA_kernel2iPfS__param_0];\n"
"	ld.param.u64 	%rd1, [_Z21runJacobiCUDA_kernel2iPfS__param_1];\n"
"	ld.param.u64 	%rd2, [_Z21runJacobiCUDA_kernel2iPfS__param_2];\n"
"	mov.u32 	%r3, %ctaid.x;\n"
"	mov.u32 	%r4, %ntid.x;\n"
"	mov.u32 	%r5, %tid.x;\n"
"	mad.lo.s32 	%r1, %r4, %r3, %r5;\n"
"	setp.lt.s32	%p1, %r1, 1;\n"
"	add.s32 	%r6, %r2, -1;\n"
"	setp.ge.s32	%p2, %r1, %r6;\n"
"	or.pred  	%p3, %p1, %p2;\n"
"	@%p3 bra 	BB1_2;\n"
""
"	cvta.to.global.u64 	%rd3, %rd2;\n"
"	mul.wide.s32 	%rd4, %r1, 4;\n"
"	add.s64 	%rd5, %rd3, %rd4;\n"
"	ld.global.f32 	%f1, [%rd5];\n"
"	cvta.to.global.u64 	%rd6, %rd1;\n"
"	add.s64 	%rd7, %rd6, %rd4;\n"
"	st.global.f32 	[%rd7], %f1;\n"
""
"BB1_2:\n"
"	ret;\n"
"}\n"
void init_array(int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	int i;

	for (i = 0; i < n; i++)
    	{
		A[i] = ((DATA_TYPE) 4 * i + 10) / N;
		B[i] = ((DATA_TYPE) 7 * i + 11) / N;
    	}
}


void runJacobi1DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
	{
		for (int i = 1; i < _PB_N - 1; i++)
		{
			B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
		}
		
		for (int j = 1; j < _PB_N - 1; j++)
		{
			A[j] = B[j];
		}
	}
}

void compareResults(int n, DATA_TYPE POLYBENCH_1D(a,N,n), DATA_TYPE POLYBENCH_1D(a_outputFromGpu,N,n), DATA_TYPE POLYBENCH_1D(b,N,n), DATA_TYPE POLYBENCH_1D(b_outputFromGpu,N,n))
{
	int i, fail;
	fail = 0;   

	// Compare a and c
	for (i=0; i < n; i++) 
	{
		if (percentDiff(a[i], a_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}

	for (i=0; i < n; i++) 
	{
		if (percentDiff(b[i], b_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{	
			fail++;
		}
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void runJacobi1DCUDA(CUdevice device, int tsteps, int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n), DATA_TYPE POLYBENCH_1D(A_outputFromGpu,N,n), 
			DATA_TYPE POLYBENCH_1D(B_outputFromGpu,N,n))
{
    CUdeviceptr Agpu, Bgpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&Agpu, N * sizeof(DATA_TYPE)));
    cuError(cuMemAlloc(&Bgpu, N * sizeof(DATA_TYPE)));

    cuError(cuMemcpyHtoD(Agpu, A, N * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(Bgpu, B, N * sizeof(DATA_TYPE)));
	
    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z21runJacobiCUDA_kernel1iPfS_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z21runJacobiCUDA_kernel2iPfS_"));

    unsigned grid_x = (unsigned int)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_X) );

    void *args1[] = {&n, &Agpu, &Bgpu, NULL};
    SET_TIME(START)
	for (int t = 0; t < _PB_TSTEPS ; t++)
	{
        cuError(cuLaunchKernel(func1, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
        cuError(cuLaunchKernel(func2, grid_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
	}
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(A_outputFromGpu, Agpu, sizeof(DATA_TYPE) * N));
    cuError(cuMemcpyDtoH(B_outputFromGpu, Bgpu, sizeof(DATA_TYPE) * N));

    cuError(cuMemFree(Agpu));
    cuError(cuMemFree(Bgpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

	POLYBENCH_1D_ARRAY_DECL(a,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(b,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting runJacobi1D on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        runJacobi1DCUDA(device, tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test runJacobi1D on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		runJacobi1DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(a_outputFromGpu);
	POLYBENCH_FREE_ARRAY(b);
	POLYBENCH_FREE_ARRAY(b_outputFromGpu);

	return 0;
}

#include "../include/polybench.c"
