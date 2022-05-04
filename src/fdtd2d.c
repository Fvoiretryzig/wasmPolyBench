#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "fdtd2d.h"
#include "polybench.h"
#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

/*
Source code:
__global__ void fdtd_step1_kernel(int nx, int ny, DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NX) && (j < _PB_NY))
    {
        if (i == 0)
        {
            ey[i * NY + j] = _fict_[t];
        }
        else
        {
            ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
        }
    }
}



__global__ void fdtd_step2_kernel(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NX) && (j < _PB_NY) && (j > 0))
    {
        ex[i * NY + j] = ex[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
    }
}


__global__ void fdtd_step3_kernel(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < (_PB_NX-1)) && (j < (_PB_NY-1)))
    {
        hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * NY + (j+1)] - ex[i * NY + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}
*/

static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z17fdtd_step1_kerneliiPfS_S_S_i\n"
                                ""
                                ".visible .entry _Z17fdtd_step1_kerneliiPfS_S_S_i(\n"
                                "	.param .u32 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_0,\n"
                                "	.param .u32 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_1,\n"
                                "	.param .u64 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_2,\n"
                                "	.param .u64 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_3,\n"
                                "	.param .u64 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_4,\n"
                                "	.param .u64 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_5,\n"
                                "	.param .u32 _Z17fdtd_step1_kerneliiPfS_S_S_i_param_6\n"
                                "){\n"
                                "	.reg .pred 	%p<5>;\n"
                                "	.reg .f32 	%f<7>;\n"
                                "	.reg .b32 	%r<15>;\n"
                                "	.reg .b64 	%rd<16>;\n"
                                ""
                                "	ld.param.u32 	%r4, [_Z17fdtd_step1_kerneliiPfS_S_S_i_param_0];\n"
                                "	ld.param.u32 	%r5, [_Z17fdtd_step1_kerneliiPfS_S_S_i_param_1];\n"
                                "	ld.param.u64 	%rd2, [_Z17fdtd_step1_kerneliiPfS_S_S_i_param_2];\n"
                                "	ld.param.u64 	%rd4, [_Z17fdtd_step1_kerneliiPfS_S_S_i_param_4];\n"
                                "	ld.param.u64 	%rd3, [_Z17fdtd_step1_kerneliiPfS_S_S_i_param_5];\n"
                                "	ld.param.u32 	%r3, [_Z17fdtd_step1_kerneliiPfS_S_S_i_param_6];\n"
                                "	cvta.to.global.u64 	%rd1, %rd4;\n"
                                "	mov.u32 	%r6, %ntid.x;\n"
                                "	mov.u32 	%r7, %ctaid.x;\n"
                                "	mov.u32 	%r8, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r6, %r7, %r8;\n"
                                "	mov.u32 	%r9, %ntid.y;\n"
                                "	mov.u32 	%r10, %ctaid.y;\n"
                                "	mov.u32 	%r11, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r9, %r10, %r11;\n"
                                "	setp.ge.s32	%p1, %r2, %r4;\n"
                                "	setp.ge.s32	%p2, %r1, %r5;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_4;\n"
                                ""
                                "	setp.eq.s32	%p4, %r2, 0;\n"
                                "	@%p4 bra 	BB0_3;\n"
                                ""
                                "	cvta.to.global.u64 	%rd5, %rd3;\n"
                                "	shl.b32 	%r12, %r2, 11;\n"
                                "	add.s32 	%r13, %r12, %r1;\n"
                                "	mul.wide.s32 	%rd6, %r13, 4;\n"
                                "	add.s64 	%rd7, %rd1, %rd6;\n"
                                "	add.s64 	%rd8, %rd5, %rd6;\n"
                                "	add.s32 	%r14, %r13, -2048;\n"
                                "	mul.wide.s32 	%rd9, %r14, 4;\n"
                                "	add.s64 	%rd10, %rd5, %rd9;\n"
                                "	ld.global.f32 	%f1, [%rd10];\n"
                                "	ld.global.f32 	%f2, [%rd8];\n"
                                "	sub.f32 	%f3, %f2, %f1;\n"
                                "	ld.global.f32 	%f4, [%rd7];\n"
                                "	fma.rn.f32 	%f5, %f3, 0fBF000000, %f4;\n"
                                "	st.global.f32 	[%rd7], %f5;\n"
                                "	bra.uni 	BB0_4;\n"
                                ""
                                "BB0_3:\n"
                                "	cvta.to.global.u64 	%rd11, %rd2;\n"
                                "	mul.wide.s32 	%rd12, %r3, 4;\n"
                                "	add.s64 	%rd13, %rd11, %rd12;\n"
                                "	ld.global.f32 	%f6, [%rd13];\n"
                                "	mul.wide.s32 	%rd14, %r1, 4;\n"
                                "	add.s64 	%rd15, %rd1, %rd14;\n"
                                "	st.global.f32 	[%rd15], %f6;\n"
                                ""
                                "BB0_4:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z17fdtd_step2_kerneliiPfS_S_i\n"
                                ".visible .entry _Z17fdtd_step2_kerneliiPfS_S_i(\n"
                                "	.param .u32 _Z17fdtd_step2_kerneliiPfS_S_i_param_0,\n"
                                "	.param .u32 _Z17fdtd_step2_kerneliiPfS_S_i_param_1,\n"
                                "	.param .u64 _Z17fdtd_step2_kerneliiPfS_S_i_param_2,\n"
                                "	.param .u64 _Z17fdtd_step2_kerneliiPfS_S_i_param_3,\n"
                                "	.param .u64 _Z17fdtd_step2_kerneliiPfS_S_i_param_4,\n"
                                "	.param .u32 _Z17fdtd_step2_kerneliiPfS_S_i_param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<6>;\n"
                                "	.reg .f32 	%f<6>;\n"
                                "	.reg .b32 	%r<14>;\n"
                                "	.reg .b64 	%rd<10>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z17fdtd_step2_kerneliiPfS_S_i_param_0];\n"
                                "	ld.param.u32 	%r4, [_Z17fdtd_step2_kerneliiPfS_S_i_param_1];\n"
                                "	ld.param.u64 	%rd1, [_Z17fdtd_step2_kerneliiPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z17fdtd_step2_kerneliiPfS_S_i_param_4];\n"
                                "	mov.u32 	%r5, %ntid.x;\n"
                                "	mov.u32 	%r6, %ctaid.x;\n"
                                "	mov.u32 	%r7, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r5, %r6, %r7;\n"
                                "	mov.u32 	%r8, %ntid.y;\n"
                                "	mov.u32 	%r9, %ctaid.y;\n"
                                "	mov.u32 	%r10, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r8, %r9, %r10;\n"
                                "	setp.ge.s32	%p1, %r2, %r3;\n"
                                "	setp.ge.s32	%p2, %r1, %r4;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	setp.lt.s32	%p4, %r1, 1;\n"
                                "	or.pred  	%p5, %p3, %p4;\n"
                                "	@%p5 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd2;\n"
                                "	cvta.to.global.u64 	%rd4, %rd1;\n"
                                "	shl.b32 	%r11, %r2, 11;\n"
                                "	add.s32 	%r12, %r11, %r1;\n"
                                "	mul.wide.s32 	%rd5, %r12, 4;\n"
                                "	add.s64 	%rd6, %rd4, %rd5;\n"
                                "	add.s64 	%rd7, %rd3, %rd5;\n"
                                "	add.s32 	%r13, %r12, -1;\n"
                                "	mul.wide.s32 	%rd8, %r13, 4;\n"
                                "	add.s64 	%rd9, %rd3, %rd8;\n"
                                "	ld.global.f32 	%f1, [%rd9];\n"
                                "	ld.global.f32 	%f2, [%rd7];\n"
                                "	sub.f32 	%f3, %f2, %f1;\n"
                                "	ld.global.f32 	%f4, [%rd6];\n"
                                "	fma.rn.f32 	%f5, %f3, 0fBF000000, %f4;\n"
                                "	st.global.f32 	[%rd6], %f5;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z17fdtd_step3_kerneliiPfS_S_i\n"
                                ".visible .entry _Z17fdtd_step3_kerneliiPfS_S_i(\n"
                                "	.param .u32 _Z17fdtd_step3_kerneliiPfS_S_i_param_0,\n"
                                "	.param .u32 _Z17fdtd_step3_kerneliiPfS_S_i_param_1,\n"
                                "	.param .u64 _Z17fdtd_step3_kerneliiPfS_S_i_param_2,\n"
                                "	.param .u64 _Z17fdtd_step3_kerneliiPfS_S_i_param_3,\n"
                                "	.param .u64 _Z17fdtd_step3_kerneliiPfS_S_i_param_4,\n"
                                "	.param .u32 _Z17fdtd_step3_kerneliiPfS_S_i_param_5\n"
                                "){\n"
                                "	.reg .pred 	%p<4>;\n"
                                "	.reg .f32 	%f<10>;\n"
                                "	.reg .b32 	%r<15>;\n"
                                "	.reg .b64 	%rd<11>;\n"
                                ""
                                "	ld.param.u32 	%r3, [_Z17fdtd_step3_kerneliiPfS_S_i_param_0];\n"
                                "	ld.param.u32 	%r4, [_Z17fdtd_step3_kerneliiPfS_S_i_param_1];\n"
                                "	ld.param.u64 	%rd1, [_Z17fdtd_step3_kerneliiPfS_S_i_param_2];\n"
                                "	ld.param.u64 	%rd2, [_Z17fdtd_step3_kerneliiPfS_S_i_param_3];\n"
                                "	ld.param.u64 	%rd3, [_Z17fdtd_step3_kerneliiPfS_S_i_param_4];\n"
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
                                "	@%p3 bra 	BB2_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd4, %rd2;\n"
                                "	cvta.to.global.u64 	%rd5, %rd1;\n"
                                "	cvta.to.global.u64 	%rd6, %rd3;\n"
                                "	shl.b32 	%r13, %r2, 11;\n"
                                "	add.s32 	%r14, %r13, %r1;\n"
                                "	mul.wide.s32 	%rd7, %r14, 4;\n"
                                "	add.s64 	%rd8, %rd6, %rd7;\n"
                                "	add.s64 	%rd9, %rd5, %rd7;\n"
                                "	ld.global.f32 	%f1, [%rd9];\n"
                                "	ld.global.f32 	%f2, [%rd9+4];\n"
                                "	sub.f32 	%f3, %f2, %f1;\n"
                                "	add.s64 	%rd10, %rd4, %rd7;\n"
                                "	ld.global.f32 	%f4, [%rd10+8192];\n"
                                "	add.f32 	%f5, %f3, %f4;\n"
                                "	ld.global.f32 	%f6, [%rd10];\n"
                                "	sub.f32 	%f7, %f5, %f6;\n"
                                "	ld.global.f32 	%f8, [%rd8];\n"
                                "	fma.rn.f32 	%f9, %f7, 0fBF333333, %f8;\n"
                                "	st.global.f32 	[%rd8], %f9;\n"
                                ""
                                "BB2_2:\n"
                                "	ret;\n"
                                "}\n";

void init_arrays(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
                 DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny), DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int i, j;

    for (i = 0; i < tmax; i++)
    {
        _fict_[i] = (DATA_TYPE)i;
    }

    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            ex[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i][j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i][j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}

void runFdtd(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
             DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny), DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int t, i, j;

    for (t = 0; t < _PB_TMAX; t++)
    {
        for (j = 0; j < _PB_NY; j++)
        {
            ey[0][j] = _fict_[t];
        }

        for (i = 1; i < _PB_NX; i++)
        {
            for (j = 0; j < _PB_NY; j++)
            {
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[(i - 1)][j]);
            }
        }

        for (i = 0; i < _PB_NX; i++)
        {
            for (j = 1; j < _PB_NY; j++)
            {
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][(j - 1)]);
            }
        }

        for (i = 0; i < _PB_NX - 1; i++)
        {
            for (j = 0; j < _PB_NY - 1; j++)
            {
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][(j + 1)] - ex[i][j] + ey[(i + 1)][j] - ey[i][j]);
            }
        }
    }
}

void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_2D(hz1, NX, NY, nx, ny), DATA_TYPE POLYBENCH_2D(hz2, NX, NY, nx, ny))
{
    int i, j, fail;
    fail = 0;

    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            if (percentDiff(hz1[i][j], hz2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void fdtdCuda(CUdevice device, int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
	DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz_outputFromGpu,NX,NY,nx,ny))
{
    CUdeviceptr _fict_gpu, ex_gpu, ey_gpu, hz_gpu;
    
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2 = NULL, func3 = NULL;

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&_fict_gpu, sizeof(DATA_TYPE) * TMAX));
    cuError(cuMemAlloc(&ex_gpu, sizeof(DATA_TYPE) * NX * NY));
    cuError(cuMemAlloc(&ey_gpu, sizeof(DATA_TYPE) * NX * NY));
    cuError(cuMemAlloc(&hz_gpu, sizeof(DATA_TYPE) * NX * NY));

    cuError(cuMemcpyHtoD(_fict_gpu, _fict_,sizeof(DATA_TYPE) * TMAX));
    cuError(cuMemcpyHtoD(ex_gpu, ex,sizeof(DATA_TYPE) * NX * NY));
    cuError(cuMemcpyHtoD(ey_gpu, ey,sizeof(DATA_TYPE) * NX * NY));
    cuError(cuMemcpyHtoD(hz_gpu, hz,sizeof(DATA_TYPE) * NX * NY));

    cuError(cuModuleLoadData(&module, KERNEL_PTX));

    cuError(cuModuleGetFunction(&func1, module, "_Z17fdtd_step1_kerneliiPfS_S_S_i"));
    cuError(cuModuleGetFunction(&func2, module, "_Z17fdtd_step2_kerneliiPfS_S_i"));
    cuError(cuModuleGetFunction(&func3, module, "_Z17fdtd_step3_kerneliiPfS_S_i"));

    unsigned grid_x = (size_t)ceil(((float)NY) / ((float)DIM_THREAD_BLOCK_X));
    unsigned grid_y = (size_t)ceil(((float)NX) / ((float)DIM_THREAD_BLOCK_Y));

    SET_TIME(START)
	for(int t = 0; t < _PB_TMAX; t++)
	{
        void *args1[] = {&nx, &ny, &_fict_gpu, &ex_gpu, &ey_gpu, &hz_gpu, &t, NULL};
        cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args1, NULL));
        void *args2[] = {&nx, &ny, &ex_gpu, &ey_gpu, &hz_gpu, &t, NULL};
        cuError(cuLaunchKernel(func2, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
        cuError(cuLaunchKernel(func3, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args2, NULL));
	}
	SET_TIME(END)
    fprintf(stdout, "GPU  actual Runtime: %0.6lfms\n", GET_DURING(END, START));

    cuError(cuMemcpyDtoH(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY));
	
    cuError(cuMemFree(_fict_gpu));
    cuError(cuMemFree(ex_gpu));
    cuError(cuMemFree(ey_gpu));
    cuError(cuMemFree(hz_gpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);

}

int main()
{
	int tmax = TMAX;
	int nx = NX;
	int ny = NY;

	POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,TMAX);
	POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu,DATA_TYPE,NX,NY,nx,ny);

	init_arrays(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        fprintf(stdout, "\nTesting fdtd on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START)
        fdtdCuda(device, tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));
        SET_TIME(GPU_END)
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));
        fprintf(stdout, "Test fdtd on GPU device %d Success\n", i);
    }
	#ifdef RUN_ON_CPU
	  	polybench_start_instruments;
        SET_TIME(CPU_START)
		runFdtd(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));
        SET_TIME(CPU_END)
        fprintf(stdout, "CPU  total Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
		compareResults(nx, ny, POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));
	#else
		print_array(n, POLYBENCH_ARRAY(X_outputFromGpu));
	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(_fict_);
	POLYBENCH_FREE_ARRAY(ex);
	POLYBENCH_FREE_ARRAY(ey);
	POLYBENCH_FREE_ARRAY(hz);
	POLYBENCH_FREE_ARRAY(hz_outputFromGpu);

	return 0;
}

#include "../include/polybench.c"
