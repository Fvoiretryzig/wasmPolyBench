#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#include "bicg.cuh"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#ifndef M_PI
#define M_PI 3.14159
#endif

#define RUN_ON_CPU

/*
source code:
__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_NY)
    {
        s[j] = 0.0f;

        int i;
        for(i = 0; i < _PB_NX; i++)
        {
            s[j] += r[i] * A[i * NY + j];
        }
    }
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_NX)
    {
        q[i] = 0.0f;

        int j;
        for(j=0; j < _PB_NY; j++)
        {
            q[i] += A[i * NY + j] * p[j];
        }
    }
}
*/
static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z12bicg_kernel1iiPfS_S_\n"
                                ""
                                ".visible .entry _Z12bicg_kernel1iiPfS_S_(\n"
                                "	.param .u32 _Z12bicg_kernel1iiPfS_S__param_0,\n"
                                "	.param .u32 _Z12bicg_kernel1iiPfS_S__param_1,\n"
                                "	.param .u64 _Z12bicg_kernel1iiPfS_S__param_2,\n"
                                "	.param .u64 _Z12bicg_kernel1iiPfS_S__param_3,\n"
                                "	.param .u64 _Z12bicg_kernel1iiPfS_S__param_4\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<33>;\n"
                                "	.reg .b32 	%r<33>;\n"
                                "	.reg .b64 	%rd<26>;\n"
                                "\n"
                                "	ld.param.u32 	%r15, [_Z12bicg_kernel1iiPfS_S__param_0];\n"
                                "	ld.param.u32 	%r16, [_Z12bicg_kernel1iiPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd8, [_Z12bicg_kernel1iiPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd9, [_Z12bicg_kernel1iiPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd7, [_Z12bicg_kernel1iiPfS_S__param_4];\n"
                                "	cvta.to.global.u64 	%rd1, %rd8;\n"
                                "	cvta.to.global.u64 	%rd2, %rd9;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r16;\n"
                                "	@%p1 bra 	BB0_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd10, %rd7;\n"
                                "	mul.wide.s32 	%rd11, %r4, 4;\n"
                                "	add.s64 	%rd3, %rd10, %rd11;\n"
                                "	mov.u32 	%r28, 0;\n"
                                "	st.global.u32 	[%rd3], %r28;\n"
                                "	setp.lt.s32	%p2, %r15, 1;\n"
                                "	@%p2 bra 	BB0_11;\n"
                                ""
                                "	and.b32  	%r21, %r15, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.eq.s32	%p3, %r21, 0;\n"
                                "	@%p3 bra 	BB0_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r21, 1;\n"
                                "	@%p4 bra 	BB0_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r21, 2;\n"
                                "	@%p5 bra 	BB0_6;\n"
                                ""
                                "	ld.global.f32 	%f12, [%rd2];\n"
                                "	add.s64 	%rd13, %rd1, %rd11;\n"
                                "	ld.global.f32 	%f13, [%rd13];\n"
                                "	fma.rn.f32 	%f29, %f12, %f13, 0f00000000;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	mov.u32 	%r28, 1;\n"
                                ""
                                "BB0_6:\n"
                                "	mul.wide.u32 	%rd14, %r28, 4;\n"
                                "	add.s64 	%rd15, %rd2, %rd14;\n"
                                "	shl.b32 	%r23, %r28, 12;\n"
                                "	add.s32 	%r24, %r23, %r4;\n"
                                "	mul.wide.s32 	%rd16, %r24, 4;\n"
                                "	add.s64 	%rd17, %rd1, %rd16;\n"
                                "	ld.global.f32 	%f14, [%rd17];\n"
                                "	ld.global.f32 	%f15, [%rd15];\n"
                                "	fma.rn.f32 	%f29, %f15, %f14, %f29;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	add.s32 	%r28, %r28, 1;\n"
                                ""
                                "BB0_7:\n"
                                "	mul.wide.s32 	%rd18, %r28, 4;\n"
                                "	add.s64 	%rd19, %rd2, %rd18;\n"
                                "	shl.b32 	%r25, %r28, 12;\n"
                                "	add.s32 	%r26, %r25, %r4;\n"
                                "	mul.wide.s32 	%rd20, %r26, 4;\n"
                                "	add.s64 	%rd21, %rd1, %rd20;\n"
                                "	ld.global.f32 	%f16, [%rd21];\n"
                                "	ld.global.f32 	%f17, [%rd19];\n"
                                "	fma.rn.f32 	%f29, %f17, %f16, %f29;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	add.s32 	%r28, %r28, 1;\n"
                                ""
                                "BB0_8:\n"
                                "	setp.lt.u32	%p6, %r15, 4;\n"
                                "	@%p6 bra 	BB0_11;\n"
                                ""
                                "	mad.lo.s32 	%r31, %r28, 4096, %r4;\n"
                                "	mul.wide.s32 	%rd22, %r28, 4;\n"
                                "	add.s64 	%rd25, %rd2, %rd22;\n"
                                ""
                                "BB0_10:\n"
                                "	mul.wide.s32 	%rd23, %r31, 4;\n"
                                "	add.s64 	%rd24, %rd1, %rd23;\n"
                                "	ld.global.f32 	%f18, [%rd24];\n"
                                "	ld.global.f32 	%f19, [%rd25];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f29;\n"
                                "	st.global.f32 	[%rd3], %f20;\n"
                                "	ld.global.f32 	%f21, [%rd24+16384];\n"
                                "	ld.global.f32 	%f22, [%rd25+4];\n"
                                "	fma.rn.f32 	%f23, %f22, %f21, %f20;\n"
                                "	st.global.f32 	[%rd3], %f23;\n"
                                "	ld.global.f32 	%f24, [%rd24+32768];\n"
                                "	ld.global.f32 	%f25, [%rd25+8];\n"
                                "	fma.rn.f32 	%f26, %f25, %f24, %f23;\n"
                                "	st.global.f32 	[%rd3], %f26;\n"
                                "	ld.global.f32 	%f27, [%rd24+49152];\n"
                                "	ld.global.f32 	%f28, [%rd25+12];\n"
                                "	fma.rn.f32 	%f29, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	add.s32 	%r31, %r31, 16384;\n"
                                "	add.s64 	%rd25, %rd25, 16;\n"
                                "	add.s32 	%r28, %r28, 4;\n"
                                "	setp.lt.s32	%p7, %r28, %r15;\n"
                                "	@%p7 bra 	BB0_10;\n"
                                ""
                                "BB0_11:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z12bicg_kernel2iiPfS_S_\n"
                                ".visible .entry _Z12bicg_kernel2iiPfS_S_(\n"
                                "	.param .u32 _Z12bicg_kernel2iiPfS_S__param_0,\n"
                                "	.param .u32 _Z12bicg_kernel2iiPfS_S__param_1,\n"
                                "	.param .u64 _Z12bicg_kernel2iiPfS_S__param_2,\n"
                                "	.param .u64 _Z12bicg_kernel2iiPfS_S__param_3,\n"
                                "	.param .u64 _Z12bicg_kernel2iiPfS_S__param_4\n"
                                "){\n"
                                "	.reg .pred 	%p<8>;\n"
                                "	.reg .f32 	%f<33>;\n"
                                "	.reg .b32 	%r<30>;\n"
                                "	.reg .b64 	%rd<29>;\n"
                                "\n"
                                "	ld.param.u32 	%r14, [_Z12bicg_kernel2iiPfS_S__param_0];\n"
                                "	ld.param.u32 	%r13, [_Z12bicg_kernel2iiPfS_S__param_1];\n"
                                "	ld.param.u64 	%rd11, [_Z12bicg_kernel2iiPfS_S__param_2];\n"
                                "	ld.param.u64 	%rd12, [_Z12bicg_kernel2iiPfS_S__param_3];\n"
                                "	ld.param.u64 	%rd10, [_Z12bicg_kernel2iiPfS_S__param_4];\n"
                                "	cvta.to.global.u64 	%rd1, %rd12;\n"
                                "	cvta.to.global.u64 	%rd2, %rd11;\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r4, %r1, %r2, %r3;\n"
                                "	setp.ge.s32	%p1, %r4, %r14;\n"
                                "	@%p1 bra 	BB1_11;\n"
                                ""
                                "	cvta.to.global.u64 	%rd13, %rd10;\n"
                                "	mul.wide.s32 	%rd14, %r4, 4;\n"
                                "	add.s64 	%rd3, %rd13, %rd14;\n"
                                "	mov.u32 	%r26, 0;\n"
                                "	st.global.u32 	[%rd3], %r26;\n"
                                "	setp.lt.s32	%p2, %r13, 1;\n"
                                "	@%p2 bra 	BB1_11;\n"
                                ""
                                "	shl.b32 	%r5, %r4, 12;\n"
                                "	and.b32  	%r19, %r13, 3;\n"
                                "	mov.f32 	%f29, 0f00000000;\n"
                                "	setp.eq.s32	%p3, %r19, 0;\n"
                                "	@%p3 bra 	BB1_8;\n"
                                ""
                                "	setp.eq.s32	%p4, %r19, 1;\n"
                                "	@%p4 bra 	BB1_7;\n"
                                ""
                                "	setp.eq.s32	%p5, %r19, 2;\n"
                                "	@%p5 bra 	BB1_6;\n"
                                ""
                                "	mul.wide.s32 	%rd15, %r5, 4;\n"
                                "	add.s64 	%rd16, %rd2, %rd15;\n"
                                "	ld.global.f32 	%f12, [%rd1];\n"
                                "	ld.global.f32 	%f13, [%rd16];\n"
                                "	fma.rn.f32 	%f29, %f13, %f12, 0f00000000;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	mov.u32 	%r26, 1;\n"
                                ""
                                "BB1_6:\n"
                                "	add.s32 	%r21, %r26, %r5;\n"
                                "	mul.wide.s32 	%rd17, %r21, 4;\n"
                                "	add.s64 	%rd18, %rd2, %rd17;\n"
                                "	mul.wide.u32 	%rd19, %r26, 4;\n"
                                "	add.s64 	%rd20, %rd1, %rd19;\n"
                                "	ld.global.f32 	%f14, [%rd20];\n"
                                "	ld.global.f32 	%f15, [%rd18];\n"
                                "	fma.rn.f32 	%f29, %f15, %f14, %f29;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	add.s32 	%r26, %r26, 1;\n"
                                ""
                                "BB1_7:\n"
                                "	add.s32 	%r22, %r26, %r5;\n"
                                "	mul.wide.s32 	%rd21, %r22, 4;\n"
                                "	add.s64 	%rd22, %rd2, %rd21;\n"
                                "	mul.wide.s32 	%rd23, %r26, 4;\n"
                                "	add.s64 	%rd24, %rd1, %rd23;\n"
                                "	ld.global.f32 	%f16, [%rd24];\n"
                                "	ld.global.f32 	%f17, [%rd22];\n"
                                "	fma.rn.f32 	%f29, %f17, %f16, %f29;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	add.s32 	%r26, %r26, 1;\n"
                                ""
                                "BB1_8:\n"
                                "	setp.lt.u32	%p6, %r13, 4;\n"
                                "	@%p6 bra 	BB1_11;\n"
                                ""
                                "	mul.wide.s32 	%rd25, %r26, 4;\n"
                                "	add.s64 	%rd28, %rd1, %rd25;\n"
                                "	mul.lo.s32 	%r23, %r1, %r2;\n"
                                "	mad.lo.s32 	%r24, %r23, 4096, %r26;\n"
                                "	mad.lo.s32 	%r25, %r3, 4096, %r24;\n"
                                "	mul.wide.s32 	%rd26, %r25, 4;\n"
                                "	add.s64 	%rd27, %rd2, %rd26;\n"
                                ""
                                "BB1_10:\n"
                                "	ld.global.f32 	%f18, [%rd28];\n"
                                "	ld.global.f32 	%f19, [%rd27];\n"
                                "	fma.rn.f32 	%f20, %f19, %f18, %f29;\n"
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
                                "	fma.rn.f32 	%f29, %f28, %f27, %f26;\n"
                                "	st.global.f32 	[%rd3], %f29;\n"
                                "	add.s64 	%rd28, %rd28, 16;\n"
                                "	add.s64 	%rd27, %rd27, 16;\n"
                                "	add.s32 	%r26, %r26, 4;\n"
                                "	setp.lt.s32	%p7, %r26, %r13;\n"
                                "	@%p7 bra 	BB1_10;\n"
                                ""
                                "BB1_11:\n"
                                "	ret;\n"
                                "}\n";

void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
	int i, j;
	
	for (i = 0; i < ny; i++)
	{
    		p[i] = i * M_PI;
	}

	for (i = 0; i < nx; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < ny; j++)
		{
      			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
}


void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_1D(s,NY,ny), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
		DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<nx; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<ny; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}
void bicg_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
		DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	int i,j;
	
  	for (i = 0; i < _PB_NY; i++)
	{
		s[i] = 0.0;
	}

	for (i = 0; i < _PB_NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < _PB_NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i][j];
	    		q[i] = q[i] + A[i][j] * p[j];
	  	}
	}
}
void bicgCuda(CUdevice device, int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
	DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
	DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
    CUdeviceptr A_gpu, q_gpu, p_gpu, r_gpu, s_gpu;

    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL, func4 = NULL, func5 = NULL, func6 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&A_gpu, sizeof(DATA_TYPE) * NX * NY));
    cuError(cuMemAlloc(&r_gpu, sizeof(DATA_TYPE) * NX));
    cuError(cuMemAlloc(&s_gpu, sizeof(DATA_TYPE) * NY));
    cuError(cuMemAlloc(&p_gpu, sizeof(DATA_TYPE) * NY));
    cuError(cuMemAlloc(&q_gpu, sizeof(DATA_TYPE) * NX));

    cuError(cuMemcpyHtoD(A_gpu, A, sizeof(DATA_TYPE) * NX * NY));
    cuError(cuMemcpyHtoD(r_gpu, r, sizeof(DATA_TYPE) * NX));
    cuError(cuMemcpyHtoD(s_gpu, s, sizeof(DATA_TYPE) * NY));
    cuError(cuMemcpyHtoD(p_gpu, p, sizeof(DATA_TYPE) * NY));
    cuError(cuMemcpyHtoD(q_gpu, q, sizeof(DATA_TYPE) * NX));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z12bicg_kernel1iiPfS_S_"));
    cuError(cuModuleGetFunction(&func2, module, "_Z12bicg_kernel2iiPfS_S_"));

    unsigned grid1_x = (size_t)(ceil( ((float)NY) / ((float)DIM_THREAD_BLOCK_X) ))
    unsigned grid2_x = (size_t)(ceil( ((float)NX) / ((float)DIM_THREAD_BLOCK_X)

    void *args1[] = {&nx, &ny, &A_gpu, &r_gpu, &s_gpu};
    void *args1[] = {&nx, &ny, &A_gpu, &p_gpu, &q_gpu};
    SET_TIME(START)
    cuError(cuLaunchKernel(func1, grid1_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
    cuError(cuLaunchKernel(func2, grid2_x, 1, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
    SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));
	
    cuError(cuMemcpyDtoH(s_outputFromGpu, s_gpu, sizeof(DATA_TYPE) * NY));
    cuError(cuMemcpyDtoH(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * NX));

    cuError(cuMemFree(A_gpu));
    cuError(cuMemFree(r_gpu));
    cuError(cuMemFree(s_gpu));
    cuError(cuMemFree(p_gpu));
    cuError(cuMemFree(q_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}

int main()
{
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(s,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(p,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(r,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

	int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting bicg on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        bicgCuda(device, nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test bicg on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		bicg_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(r);
	POLYBENCH_FREE_ARRAY(s);
	POLYBENCH_FREE_ARRAY(p);
	POLYBENCH_FREE_ARRAY(q);
	POLYBENCH_FREE_ARRAY(s_outputFromGpu);
	POLYBENCH_FREE_ARRAY(q_outputFromGpu);

  	return 0;
}

#include "../include/polybench.c"
