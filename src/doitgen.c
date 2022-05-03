#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include "polybenchUtilFuncts.h"
#include "cuda-helper.h"

/* Problem size. */
#define NR 128
#define NQ 128
#define NP 128

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

typedef float DATA_TYPE;

/*
* Source Code:
* __global__ void doitgen_kernel1(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r) {
* 	int p = blockIdx.x * blockDim.x + threadIdx.x;
* 	int q = blockIdx.y * blockDim.y + threadIdx.y;
* 	if ((p < NP) && (q < NQ)) {
* 		sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
* 		for (int s = 0; s < NP; s++) {
* 			sum[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p] + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
* 		}
* }
}
* __global__ void doitgen_kernel2(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r) {
* 	int p = blockIdx.x * blockDim.x + threadIdx.x;
* 	int q = blockIdx.y * blockDim.y + threadIdx.y;
* 	if ((p < NP) && (q < NQ)) {
* 		A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
* 	}
* }
*/

static const char *KERNEL_PTX = ".version 6.5\n"
                                ".target sm_30\n"
                                ".address_size 64\n"
                                ""
                                "	// .globl	_Z15doitgen_kernel1PfS_S_i\n"
                                ""
                                ".visible .entry _Z15doitgen_kernel1PfS_S_i(\n"
                                "	.param .u64 _Z15doitgen_kernel1PfS_S_i_param_0,\n"
                                "	.param .u64 _Z15doitgen_kernel1PfS_S_i_param_1,\n"
                                "	.param .u64 _Z15doitgen_kernel1PfS_S_i_param_2,\n"
                                "	.param .u32 _Z15doitgen_kernel1PfS_S_i_param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<5>;\n"
                                "	.reg .f32 	%f<28>;\n"
                                "	.reg .b32 	%r<26>;\n"
                                "	.reg .b64 	%rd<16>;\n"
                                ""
                                "	ld.param.u64 	%rd7, [_Z15doitgen_kernel1PfS_S_i_param_0];\n"
                                "	ld.param.u64 	%rd8, [_Z15doitgen_kernel1PfS_S_i_param_1];\n"
                                "	ld.param.u64 	%rd9, [_Z15doitgen_kernel1PfS_S_i_param_2];\n"
                                "	ld.param.u32 	%r14, [_Z15doitgen_kernel1PfS_S_i_param_3];\n"
                                "	mov.u32 	%r1, %ntid.x;\n"
                                "	mov.u32 	%r2, %ctaid.x;\n"
                                "	mov.u32 	%r3, %tid.x;\n"
                                "	mad.lo.s32 	%r24, %r1, %r2, %r3;\n"
                                "	mov.u32 	%r5, %ntid.y;\n"
                                "	mov.u32 	%r6, %ctaid.y;\n"
                                "	mov.u32 	%r7, %tid.y;\n"
                                "	mad.lo.s32 	%r8, %r5, %r6, %r7;\n"
                                "	setp.gt.s32	%p1, %r24, 127;\n"
                                "	setp.gt.s32	%p2, %r8, 127;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB0_3;\n"
                                ""
                                "	cvta.to.global.u64 	%rd1, %rd9;\n"
                                "	cvta.to.global.u64 	%rd10, %rd7;\n"
                                "	shl.b32 	%r16, %r8, 7;\n"
                                "	shl.b32 	%r17, %r14, 14;\n"
                                "	add.s32 	%r18, %r16, %r17;\n"
                                "	add.s32 	%r19, %r18, %r24;\n"
                                "	mul.wide.s32 	%rd11, %r19, 4;\n"
                                "	add.s64 	%rd2, %rd10, %rd11;\n"
                                "	mov.u32 	%r20, 0;\n"
                                "	st.global.u32 	[%rd2], %r20;\n"
                                "	mul.lo.s32 	%r21, %r5, %r6;\n"
                                "	mad.lo.s32 	%r22, %r21, 128, %r17;\n"
                                "	mad.lo.s32 	%r23, %r7, 128, %r22;\n"
                                "	mul.wide.s32 	%rd3, %r23, 4;\n"
                                "	cvta.to.global.u64 	%rd15, %rd8;\n"
                                "	mov.f32 	%f27, 0f00000000;\n"
                                "	mov.u32 	%r25, -128;\n"
                                ""
                                "BB0_2:\n"
                                "	add.s64 	%rd12, %rd15, %rd3;\n"
                                "	mul.wide.s32 	%rd13, %r24, 4;\n"
                                "	add.s64 	%rd14, %rd1, %rd13;\n"
                                "	ld.global.f32 	%f4, [%rd14];\n"
                                "	ld.global.f32 	%f5, [%rd12];\n"
                                "	fma.rn.f32 	%f6, %f5, %f4, %f27;\n"
                                "	st.global.f32 	[%rd2], %f6;\n"
                                "	ld.global.f32 	%f7, [%rd14+512];\n"
                                "	ld.global.f32 	%f8, [%rd12+4];\n"
                                "	fma.rn.f32 	%f9, %f8, %f7, %f6;\n"
                                "	st.global.f32 	[%rd2], %f9;\n"
                                "	ld.global.f32 	%f10, [%rd14+1024];\n"
                                "	ld.global.f32 	%f11, [%rd12+8];\n"
                                "	fma.rn.f32 	%f12, %f11, %f10, %f9;\n"
                                "	st.global.f32 	[%rd2], %f12;\n"
                                "	ld.global.f32 	%f13, [%rd14+1536];\n"
                                "	ld.global.f32 	%f14, [%rd12+12];\n"
                                "	fma.rn.f32 	%f15, %f14, %f13, %f12;\n"
                                "	st.global.f32 	[%rd2], %f15;\n"
                                "	ld.global.f32 	%f16, [%rd14+2048];\n"
                                "	ld.global.f32 	%f17, [%rd12+16];\n"
                                "	fma.rn.f32 	%f18, %f17, %f16, %f15;\n"
                                "	st.global.f32 	[%rd2], %f18;\n"
                                "	ld.global.f32 	%f19, [%rd14+2560];\n"
                                "	ld.global.f32 	%f20, [%rd12+20];\n"
                                "	fma.rn.f32 	%f21, %f20, %f19, %f18;\n"
                                "	st.global.f32 	[%rd2], %f21;\n"
                                "	ld.global.f32 	%f22, [%rd14+3072];\n"
                                "	ld.global.f32 	%f23, [%rd12+24];\n"
                                "	fma.rn.f32 	%f24, %f23, %f22, %f21;\n"
                                "	st.global.f32 	[%rd2], %f24;\n"
                                "	ld.global.f32 	%f25, [%rd14+3584];\n"
                                "	ld.global.f32 	%f26, [%rd12+28];\n"
                                "	fma.rn.f32 	%f27, %f26, %f25, %f24;\n"
                                "	st.global.f32 	[%rd2], %f27;\n"
                                "	add.s32 	%r24, %r24, 1024;\n"
                                "	add.s64 	%rd15, %rd15, 32;\n"
                                "	add.s32 	%r25, %r25, 8;\n"
                                "	setp.ne.s32	%p4, %r25, 0;\n"
                                "	@%p4 bra 	BB0_2;\n"
                                ""
                                "BB0_3:\n"
                                "	ret;\n"
                                "}\n"
                                ""
                                "	// .globl	_Z15doitgen_kernel2PfS_S_i\n"
                                ".visible .entry _Z15doitgen_kernel2PfS_S_i(\n"
                                "	.param .u64 _Z15doitgen_kernel2PfS_S_i_param_0,\n"
                                "	.param .u64 _Z15doitgen_kernel2PfS_S_i_param_1,\n"
                                "	.param .u64 _Z15doitgen_kernel2PfS_S_i_param_2,\n"
                                "	.param .u32 _Z15doitgen_kernel2PfS_S_i_param_3\n"
                                "){\n"
                                "	.reg .pred 	%p<4>;\n"
                                "	.reg .f32 	%f<2>;\n"
                                "	.reg .b32 	%r<14>;\n"
                                "	.reg .b64 	%rd<8>;\n"
                                ""
                                "	ld.param.u64 	%rd1, [_Z15doitgen_kernel2PfS_S_i_param_0];\n"
                                "	ld.param.u64 	%rd2, [_Z15doitgen_kernel2PfS_S_i_param_1];\n"
                                "	ld.param.u32 	%r3, [_Z15doitgen_kernel2PfS_S_i_param_3];\n"
                                "	mov.u32 	%r4, %ctaid.x;\n"
                                "	mov.u32 	%r5, %ntid.x;\n"
                                "	mov.u32 	%r6, %tid.x;\n"
                                "	mad.lo.s32 	%r1, %r5, %r4, %r6;\n"
                                "	mov.u32 	%r7, %ntid.y;\n"
                                "	mov.u32 	%r8, %ctaid.y;\n"
                                "	mov.u32 	%r9, %tid.y;\n"
                                "	mad.lo.s32 	%r2, %r7, %r8, %r9;\n"
                                "	setp.gt.s32	%p1, %r1, 127;\n"
                                "	setp.gt.s32	%p2, %r2, 127;\n"
                                "	or.pred  	%p3, %p1, %p2;\n"
                                "	@%p3 bra 	BB1_2;\n"
                                ""
                                "	cvta.to.global.u64 	%rd3, %rd1;\n"
                                "	shl.b32 	%r10, %r3, 14;\n"
                                "	add.s32 	%r11, %r1, %r10;\n"
                                "	shl.b32 	%r12, %r2, 7;\n"
                                "	add.s32 	%r13, %r11, %r12;\n"
                                "	mul.wide.s32 	%rd4, %r13, 4;\n"
                                "	add.s64 	%rd5, %rd3, %rd4;\n"
                                "	ld.global.f32 	%f1, [%rd5];\n"
                                "	cvta.to.global.u64 	%rd6, %rd2;\n"
                                "	add.s64 	%rd7, %rd6, %rd4;\n"
                                "	st.global.f32 	[%rd7], %f1;\n"
                                ""
                                "BB1_2:\n"
                                "	ret;\n"
                                "}";
void init_array(DATA_TYPE *A, DATA_TYPE *C4) {
  	for (int i = 0; i < NR; i++) {
    	for (int j = 0; j < NQ; j++) {
      		for (int k = 0; k < NP; k++) {
	 			A[i * (NQ * NP) + j * NP + k] = ((DATA_TYPE) i*j + k) / NP;
      		}
    	}
  	}

  	for (int i = 0; i < NP; i++) {
    	for (int j = 0; j < NP; j++) {
      		C4[i * NP + j] = ((DATA_TYPE) i*j) / NP;
    	}
  	}
}
void doitgenCuda(CUdevice device, DATA_TYPE* A, DATA_TYPE* C4, DATA_TYPE* sum, DATA_TYPE* sum_outputFromGpu) {
	double t_start, t_end;

	//DATA_TYPE* AGpu;
	//DATA_TYPE* C4Gpu;
	//DATA_TYPE* sumGpu;
	CUdeviceptr AGpu, C4Gpu, sumGpu;
	
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction func1 = NULL, func2=NULL;

    unsigned grid_x = (unsigned int)ceil( ((float)NP) / ((float)DIM_THREAD_BLOCK_X) );
    unsigned grid_y = (unsigned int)ceil( ((float)NR) / ((float)DIM_THREAD_BLOCK_Y) );

    cuError(cuCtxCreate(&context, 0, device));
    cuError(cuMemAlloc(&AGpu, NR * NQ * NP * sizeof(DATA_TYPE)));
    cuError(cuMemAlloc(&C4Gpu, NP * NP * sizeof(DATA_TYPE)));
    cuError(cuMemAlloc(&sumGpu, NR * NQ * NP * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(AGpu, A, NR * NQ * NP * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(C4Gpu, C4, NP * NP * sizeof(DATA_TYPE)));
    cuError(cuMemcpyHtoD(sumGpu, sum, NR * NQ * NP * sizeof(DATA_TYPE)));

    //printf("  Loading module from ptx ...\n");
    cuError(cuModuleLoadData(&module, KERNEL_PTX));
    //printf("  Load module from ptx Success\n");

    //doitgen_kernel1
    //printf("  Loading function from module ...\n");
    cuError(cuModuleGetFunction(&func1, module, "_Z15doitgen_kernel1PfS_S_i"));
    cuError(cuModuleGetFunction(&func2, module, "_Z15doitgen_kernel2PfS_S_i"));
    //printf("  Load function from module Success\n");
    SET_TIME(START)
    for (int r=0; r<NR; r++) {
        void *args[] = {&sumGpu, &AGpu, &C4Gpu, &r, NULL};
		cuError(cuLaunchKernel(func1, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args, NULL));
        cuError(cuLaunchKernel(func2, grid_x, grid_y, 1, DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, 1, 0, NULL, args, NULL));
    }
	SET_TIME(END)
    fprintf(stdout, "GPU actual Runtime: %0.6lfms\n", GET_DURING(END, START));
	//cudaMemcpy(sum_outputFromGpu, sumGpu, NR * NQ * NP * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cuError(cuMemcpyDtoH(sum_outputFromGpu, sumGpu, NR * NQ * NP * sizeof(DATA_TYPE)));
    cuError(cuMemFree(AGpu));
    cuError(cuMemFree(C4Gpu));
    cuError(cuMemFree(sumGpu));
    cuModuleUnload(module);
    cuCtxDestroy(context);
}
void doitgenCPU(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4) {
	for (int r = 0; r < NR; r++) {
		for (int q = 0; q < NQ; q++) {
			for (int p = 0; p < NP; p++) {
				sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
				for (int s = 0; s < NP; s++) {
					sum[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p] + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
				}
      		}
			for (int p = 0; p < NP; p++) {
				A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
			}
		}
	}
}
void compareResults(DATA_TYPE* sum, DATA_TYPE* sum_outputFromGpu) {
	int fail = 0;
	for (int r = 0; r < NR; r++) {
    	for (int q = 0; q < NQ; q++) {
            for (int p = 0; p < NP; p++) {
                if (percentDiff(sum[r * (NQ * NP) + q * NP + p], sum_outputFromGpu[r * (NQ * NP) + q * NP + p]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                    fail++;
				}
			}
		}
	}
	// Print results
	printf("Number of misses: %d\n", fail);
}
int main() {
	DATA_TYPE* A;
	DATA_TYPE* C4;
	DATA_TYPE* sum, *sum_outputFromGpu;

	A = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
	C4 = (DATA_TYPE*)malloc(NP * NP * sizeof(DATA_TYPE));
	sum = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
	sum_outputFromGpu = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
    
    init_array(A, C4);
    
    int deviceCount = 0;
    CUdevice device = 0;
	char name[GPU_DEVICE_NAME_SIZE];

    cuError(cuInit(0));
    cuError(cuDeviceGetCount(&deviceCount));
    fprintf(stdout, "GPU device count = %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        printf("\nTesting doitgen on GPU device %d ...\n", i);

        cuError(cuDeviceGet(&device, i));

        cuError(cuDeviceGetName(name, GPU_DEVICE_NAME_SIZE, device));
        fprintf(stdout, "  GPU device name is: '%s'\n", name);

        SET_TIME(GPU_START);
	    doitgenCuda(device, A, C4, sum, sum_outputFromGpu);
        SET_TIME(GPU_END);
        fprintf(stdout, "GPU  total Runtime: %0.6lfms\n", GET_DURING(GPU_END, GPU_START));

        fprintf(stdout, "Test doitgen on GPU device %d Success\n", i);
    }

	SET_TIME(CPU_START);
	doitgenCPU(sum, A, C4);
    SET_TIME(CPU_END);
	fprintf(stdout, "CPU Runtime: %0.6lfms\n", GET_DURING(CPU_END, CPU_START));
	
	compareResults(sum, sum_outputFromGpu);

	free(A);
	free(C4);
	free(sum);
	free(sum_outputFromGpu);
	
    return 0;
}
