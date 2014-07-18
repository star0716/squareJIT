#include <stdio.h>

#include <helper_functions.h>
#include <helper_cuda.h>

__global__
void squareKernel(int *data, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < N)
	{
		data[i] = i * i;
	}
}

int main(int argc, char **argv)
{
	int *h_data;
	int *d_data;
	int sum = 0;

	cudaHostAlloc(&h_data, 1000 * sizeof(int), cudaHostAllocPortable);
	cudaMalloc(&d_data, 1000 * sizeof(int));

	for(int i=0;i<1000;i++)
	{
		h_data[i] = i*i;
	}

	dim3 block(512);
	dim3 grid((1000 + block.x - 1) / block.x);

	cudaMemcpy(d_data, h_data, 1000 * sizeof(int), cudaMemcpyHostToDevice);
	squareKernel<<<grid, block>>>(d_data, 1000);
	cudaMemcpy(h_data, d_data, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

	for(int j=0;j<1000;j++)
	{
		sum = sum + h_data[j];
	}
	
	printf("h_data[998] = %d\n", h_data[998]);
	printf("h_data[999] = %d\n", h_data[999]);
	printf("sum = %d\n", sum);
	return 0;
}