#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "syr2k.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
Source code:
__global__ void syr2k_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NI) && (j < NI))
    {
        c[i * NI + j] *= beta;

        int k;
        for(k = 0; k < NJ; k++)
        {
            c[i * NI + j] += alpha * a[i * NJ + k] * b[j * NJ + k] + alpha * b[i * NJ + k] * a[j * NJ + k];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z12syr2k_kerneliiffPfS_S_\n"
                                ""
                                ".visible .entry _Z12syr2k_kerneliiffPfS_S_(\n"
                                "	.param .u32 _Z12syr2k_kerneliiffPfS_S__param_0,\n"
                                "	.param .u32 _Z12syr2k_kerneliiffPfS_S__param_1,\n"
                                "	.param .f32 _Z12syr2k_kerneliiffPfS_S__param_2,\n"
                                "	.param .f32 _Z12syr2k_kerneliiffPfS_S__param_3,\n"
                                "	.param .u64 _Z12syr2k_kerneliiffPfS_S__param_4,\n"
                                "	.param .u64 _Z12syr2k_kerneliiffPfS_S__param_5,\n"
                                "	.param .u64 _Z12syr2k_kerneliiffPfS_S__param_6\n"
                                "){\n"
                                "	.reg .pred 	%p<5>;\n"
                                "	.reg .f32 	%f<79>;\n"
                                "	.reg .b32 	%r<21>;\n"
                                "	.reg .b64 	%rd<21>;\n"
                                ""
                                "	ld.param.f32 	%f4, [_Z12syr2k_kerneliiffPfS_S__param_2];\n"
                                "	ld.param.f32 	%f5, [_Z12syr2k_kerneliiffPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd10, [_Z12syr2k_kerneliiffPfS_S__param_4];\n"
                                "	ld.param.u64 	%rd11, [_Z12syr2k_kerneliiffPfS_S__param_5];\n"
                                "	ld.param.u64 	%rd12, [_Z12syr2k_kerneliiffPfS_S__param_6];\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	mov.u32 	%r5, %ntid.y;\n"
                                "	mov.u32 	%r6, %ctaid.y;\n"
                                "	mov.u32 	%r7, %tid.y;\n"
                                "	mad.lo.s32 	%r8, %r5, %r6, %r7;\n"
                                "	setp.gt.s32	%p1, %r8, 1023;\n"
                                "	setp.gt.s32	%p2, %r4, 1023;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_3;\n"
                                ""
                                "	cvta.to.global.u64 	%rd13, %rd12;\n"
                                "	shl.b32 	%r12, %r8, 10;\n"
                                "	add.s32 	%r13, %r12, %r4;\n"
                                "	mul.wide.s32 	%rd14, %r13, 4;\n"
                                "	add.s64 	%rd1, %rd13, %rd14;\n"
                                "	ld.global.f32 	%f6, [%rd1];\n"
                                "	mul.f32 	%f78, %f6, %f5;\n"
                                "	st.global.f32 	[%rd1], %f78;\n"
                                "	mul.lo.s32 	%r14, %r1, %r2;\n"
                                "	shl.b32 	%r15, %r14, 10;\n"
                                "	mul.lo.s32 	%r16, %r5, %r6;\n"
                                "	shl.b32 	%r17, %r16, 10;\n"
                                "	mad.lo.s32 	%r18, %r7, 1024, %r17;\n"
                                "	mul.wide.s32 	%rd2, %r18, 4;\n"
                                "	mad.lo.s32 	%r19, %r3, 1024, %r15;\n"
                                "	mul.wide.s32 	%rd3, %r19, 4;\n"
                                "	cvta.to.global.u64 	%rd20, %rd10;\n"
                                "	cvta.to.global.u64 	%rd19, %rd11;\n"
                                "	mov.u32 	%r20, -1024;\n"
                                ""
                                "BB0_2:\n"
                                "	add.s64 	%rd15, %rd20, %rd2;\n"
                                "	ld.global.f32 	%f7, [%rd15];\n"
                                "	mul.f32 	%f8, %f7, %f4;\n"
                                "	add.s64 	%rd16, %rd19, %rd3;\n"
                                "	ld.global.f32 	%f9, [%rd16];\n"
                                "	add.s64 	%rd17, %rd19, %rd2;\n"
                                "	ld.global.f32 	%f10, [%rd17];\n"
                                "	mul.f32 	%f11, %f10, %f4;\n"
                                "	add.s64 	%rd18, %rd20, %rd3;\n"
                                "	ld.global.f32 	%f12, [%rd18];\n"
                                "	mul.f32 	%f13, %f11, %f12;\n"
                                "	fma.rn.f32 	%f14, %f8, %f9, %f13;\n"
                                "	add.f32 	%f15, %f78, %f14;\n"
                                "	st.global.f32 	[%rd1], %f15;\n"
                                "	ld.global.f32 	%f16, [%rd15+4];\n"
                                "	mul.f32 	%f17, %f16, %f4;\n"
                                "	ld.global.f32 	%f18, [%rd16+4];\n"
                                "	ld.global.f32 	%f19, [%rd17+4];\n"
                                "	mul.f32 	%f20, %f19, %f4;\n"
                                "	ld.global.f32 	%f21, [%rd18+4];\n"
                                "	mul.f32 	%f22, %f20, %f21;\n"
                                "	fma.rn.f32 	%f23, %f17, %f18, %f22;\n"
                                "	add.f32 	%f24, %f15, %f23;\n"
                                "	st.global.f32 	[%rd1], %f24;\n"
                                "	ld.global.f32 	%f25, [%rd15+8];\n"
                                "	mul.f32 	%f26, %f25, %f4;\n"
                                "	ld.global.f32 	%f27, [%rd16+8];\n"
                                "	ld.global.f32 	%f28, [%rd17+8];\n"
                                "	mul.f32 	%f29, %f28, %f4;\n"
                                "	ld.global.f32 	%f30, [%rd18+8];\n"
                                "	mul.f32 	%f31, %f29, %f30;\n"
                                "	fma.rn.f32 	%f32, %f26, %f27, %f31;\n"
                                "	add.f32 	%f33, %f24, %f32;\n"
                                "	st.global.f32 	[%rd1], %f33;\n"
                                "	ld.global.f32 	%f34, [%rd15+12];\n"
                                "	mul.f32 	%f35, %f34, %f4;\n"
                                "	ld.global.f32 	%f36, [%rd16+12];\n"
                                "	ld.global.f32 	%f37, [%rd17+12];\n"
                                "	mul.f32 	%f38, %f37, %f4;\n"
                                "	ld.global.f32 	%f39, [%rd18+12];\n"
                                "	mul.f32 	%f40, %f38, %f39;\n"
                                "	fma.rn.f32 	%f41, %f35, %f36, %f40;\n"
                                "	add.f32 	%f42, %f33, %f41;\n"
                                "	st.global.f32 	[%rd1], %f42;\n"
                                "	ld.global.f32 	%f43, [%rd15+16];\n"
                                "	mul.f32 	%f44, %f43, %f4;\n"
                                "	ld.global.f32 	%f45, [%rd16+16];\n"
                                "	ld.global.f32 	%f46, [%rd17+16];\n"
                                "	mul.f32 	%f47, %f46, %f4;\n"
                                "	ld.global.f32 	%f48, [%rd18+16];\n"
                                "	mul.f32 	%f49, %f47, %f48;\n"
                                "	fma.rn.f32 	%f50, %f44, %f45, %f49;\n"
                                "	add.f32 	%f51, %f42, %f50;\n"
                                "	st.global.f32 	[%rd1], %f51;\n"
                                "	ld.global.f32 	%f52, [%rd15+20];\n"
                                "	mul.f32 	%f53, %f52, %f4;\n"
                                "	ld.global.f32 	%f54, [%rd16+20];\n"
                                "	ld.global.f32 	%f55, [%rd17+20];\n"
                                "	mul.f32 	%f56, %f55, %f4;\n"
                                "	ld.global.f32 	%f57, [%rd18+20];\n"
                                "	mul.f32 	%f58, %f56, %f57;\n"
                                "	fma.rn.f32 	%f59, %f53, %f54, %f58;\n"
                                "	add.f32 	%f60, %f51, %f59;\n"
                                "	st.global.f32 	[%rd1], %f60;\n"
                                "	ld.global.f32 	%f61, [%rd15+24];\n"
                                "	mul.f32 	%f62, %f61, %f4;\n"
                                "	ld.global.f32 	%f63, [%rd16+24];\n"
                                "	ld.global.f32 	%f64, [%rd17+24];\n"
                                "	mul.f32 	%f65, %f64, %f4;\n"
                                "	ld.global.f32 	%f66, [%rd18+24];\n"
                                "	mul.f32 	%f67, %f65, %f66;\n"
                                "	fma.rn.f32 	%f68, %f62, %f63, %f67;\n"
                                "	add.f32 	%f69, %f60, %f68;\n"
                                "	st.global.f32 	[%rd1], %f69;\n"
                                "	ld.global.f32 	%f70, [%rd15+28];\n"
                                "	mul.f32 	%f71, %f70, %f4;\n"
                                "	ld.global.f32 	%f72, [%rd16+28];\n"
                                "	ld.global.f32 	%f73, [%rd17+28];\n"
                                "	mul.f32 	%f74, %f73, %f4;\n"
                                "	ld.global.f32 	%f75, [%rd18+28];\n"
                                "	mul.f32 	%f76, %f74, %f75;\n"
                                "	fma.rn.f32 	%f77, %f71, %f72, %f76;\n"
                                "	add.f32 	%f78, %f69, %f77;\n"
                                "	st.global.f32 	[%rd1], %f78;\n"
                                "	add.s64 	%rd20, %rd20, 32;\n"
                                "	add.s64 	%rd19, %rd19, 32;\n"
                                "	add.s32 	%r20, %r20, 8;\n"
                                "	setp.ne.s32	%p4, %r20, 0;\n"
                                "	@%p4 bra 	BB0_2;\n"
                                ""
                                "BB0_3:\n"
                                "	ret;\n"
                                "}\n" void
                                init_arrays(int ni, int nj,
                                            DATA_TYPE *alpha,
                                            DATA_TYPE *beta,
                                            DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
                                            DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
                                            DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni))
{
    int i, j;

    *alpha = 32412;
    *beta = 2123;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / ni;
            B[i][j] = ((DATA_TYPE)i * j) / ni;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < ni; j++)
        {
            C[i][j] = ((DATA_TYPE)i * j) / ni;
        }
    }
}

void syr2kCpu(int ni, int nj,
              DATA_TYPE alpha,
              DATA_TYPE beta,
              DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
              DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
              DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni))
{
    int i, j, k;

    /*    C := alpha*A*B' + alpha*B*A' + beta*C */
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
                C[i][j] += alpha * A[i][k] * B[j][k];
                C[i][j] += alpha * B[i][k] * A[j][k];
            }
        }
    }
}

void compareResults(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
    int i, j, fail;
    fail = 0;

    // Compare C with D
    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < ni; j++)
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

void syr2kCuda(CUdevice device, int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj), 
		DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni)) 
{
    CUdeviceptr A_gpu, B_gpu, C_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&B_gpu, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemAlloc(&C_gpu, sizeof(DATA_TYPE) * NI * NI)));
    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ));
    cuError(cuMemcpyHtoD(C_gpu, C, sizeof(DATA_TYPE) * NI * NI));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z12syr2k_kerneliiffPfS_S_"));

    unsigned grid_x = (size_t)ceil( ((float)NI) / ((float)DIM_THREAD_BLOCK_X) );
    unsigned grid_y = (size_t)ceil( ((float)NI) / ((float)DIM_THREAD_BLOCK_Y) );
    void *args1[] = {&ni, &nj, &alpha, &beta, &A_gpu, &B_gpu, &C_gpu, NULL};

    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));
	
    cuError(cuMemcpyDtoH(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NI));

    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(B_gpu));
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
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NI,ni,ni);
	POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NI,ni,ni);

	init_arrays(ni, nj, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
    
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting syr2k on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        syr2kCuda(device, ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test syr2k on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		syr2kCpu(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));        
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(C_outputFromGpu);

  	return 0;
}

#include "../include/polybench.c"
