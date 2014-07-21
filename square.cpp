// System includes
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>

// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples


#include "square.h"


void ptxJIT(CUmodule *phModule, CUfunction *phKernel, CUlinkState *lState)
{
    CUjit_option options[6];
    void *optionVals[6];
    float walltime;
    char error_log[8192],
         info_log[8192];
    unsigned int logSize = 8192;
    void *cuOut;
    size_t outSize;
    int myErr = 0;

    // Setup linker options
    // Return walltime from JIT compilation
    options[0] = CU_JIT_WALL_TIME;
    optionVals[0] = (void *) &walltime;
    // Pass a buffer for info messages
    options[1] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[1] = (void *) info_log;
    // Pass the size of the info buffer
    options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[2] = (void *) logSize;
    // Pass a buffer for error message
    options[3] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[3] = (void *) error_log;
    // Pass the size of the error buffer
    options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[4] = (void *) logSize;
    // Make the linker verbose
    options[5] = CU_JIT_LOG_VERBOSE;
    optionVals[5] = (void *) 1;

    // Create a pending linker invocation
    checkCudaErrors(cuLinkCreate(6, options, optionVals, lState));

    if (sizeof(void *)==8)
    {
        // Load the PTX from the string squarePtx64 (64-bit)
         printf("Loading squarePtx64[] program\n");
         myErr = cuLinkAddData(*lState, CU_JIT_INPUT_PTX, (void *)squarePtx64, strlen(squarePtx64)+1, 0, 0, 0, 0);
        // PTX May also be loaded from file, as per below.
		//printf("Loading from file square.ptx\n");
        //myErr = cuLinkAddFile(*lState, CU_JIT_INPUT_PTX, "square.ptx",0,0,0);
    }

    if (myErr != CUDA_SUCCESS)
    {
        // Errors will be put in error_log, per CU_JIT_ERROR_LOG_BUFFER option above.
        fprintf(stderr,"PTX Linker Error:\n%s\n",error_log);
    }

    // Complete the linker step
    checkCudaErrors(cuLinkComplete(*lState, &cuOut, &outSize));

    // Linker walltime and info_log were requested in options above.
    printf("CUDA Link Completed in %fms. Linker Output:\n%s\n",walltime,info_log);

    // Load resulting cuBin into module
    checkCudaErrors(cuModuleLoadData(phModule, cuOut));

    // Locate the kernel entry poin
    checkCudaErrors(cuModuleGetFunction(phKernel, *phModule, "_Z12squareKernelPii"));

    // Destroy the linker invocation
    checkCudaErrors(cuLinkDestroy(*lState));
}



int main(int argc, char **argv)
{
	int cuda_device = 0;
	int n = 1000;
	int sum = 0;

	CUmodule     hModule  = 0;
    CUfunction   hKernel  = 0;
    CUlinkState  lState;

	int *h_data;
	int *d_data;

	cudaDeviceProp deviceProp;

	printf("Square with PTX Just-In-Time(JIT) compilation\n");

	//pick the device with highest Gflops/s
	cuda_device = gpuGetMaxGflopsDeviceId();

	checkCudaErrors(cudaSetDevice(cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
    printf("> Using CUDA device [%d]: %s\n", cuda_device, deviceProp.name);

	if (deviceProp.major < 2)
    {
        fprintf(stderr, "Compute Capability 2.0 or greater required for this sample.\n");
        fprintf(stderr, "Maximum Compute Capability of device[%d] is %d.%d.\n", cuda_device,deviceProp.major,deviceProp.minor);
        exit(EXIT_WAIVED);
    }


	h_data = (int *)malloc(n * sizeof(int));
	checkCudaErrors(cudaMalloc(&d_data, n * sizeof(int)));


	// JIT Compile the Kernel from PTX and get the Handles (Driver API)
    ptxJIT(&hModule, &hKernel, &lState);

    // Set the kernel parameters (Driver API)
    checkCudaErrors(cuFuncSetBlockShape(hKernel, 512, 1, 1));


	int paramOffset = 0;
    checkCudaErrors(cuParamSetv(hKernel, paramOffset, &d_data, sizeof(d_data)));
	paramOffset += sizeof(d_data);
    checkCudaErrors(cuParamSetv(hKernel, paramOffset, &n, sizeof(n)));
	paramOffset += sizeof(n);
    checkCudaErrors(cuParamSetSize(hKernel, paramOffset));

	// Launch the kernel (Driver API_)
    checkCudaErrors(cuLaunchGrid(hKernel, n + (512 - 1) / 512, 1));
    std::cout << "CUDA kernel launched" << std::endl;

    // Copy the result to the host
    checkCudaErrors(cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));

	for(int j=0;j<n;j++)
	{
		sum = sum + h_data[j];
	}
	

	printf("square of %d = %d\n", n-2, h_data[n-2]);
	printf("square of %d = %d\n", n-1, h_data[n-1]);
	printf("sum = %d\n", sum);
 
    // Cleanup
    if (d_data)
    {
        checkCudaErrors(cudaFree(d_data));
        d_data = 0;
    }

    if (h_data)
    {
        free(h_data);
        h_data = 0;
    }

    if (hModule)
    {
        checkCudaErrors(cuModuleUnload(hModule));
        hModule = 0;
    }	 

    cudaDeviceReset();

	return 0;
}