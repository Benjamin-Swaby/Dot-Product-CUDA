//main.cu by benjamin-swaby in Dot-product-CUDA



#define multi 20
#define N 20

#include <stdio.h>
#include <iostream>

//DEVICE FUNCTION product

//args:
//	float *a -> array in 1
// 	float *b -> array in 2
// 	float *c -> c[i] = a[i] * b[i]

__global__ void product(float *a, float *b, float *c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
   	int stride = blockDim.x * gridDim.x;

   	for(int i = index; i < N; i += stride)
   	{
    	c[i] = a[i] * b[i];
   	}

}



Template<unsigned int block_size>
__device__ void warpReduce(volatile float *sdata, unsigned int tid)
{
	if(block_size >= 64) sdata[tid] += sdata[tid+32];
	if(block_size >= 32) sdata[tid] += sdata[tid+16];
	if(block_size >= 16) sdata[tid] += sdata[tid+8];
	if(block_size >= 8) sdata[tid] += sdata[tid+4];
	if(block_size >= 4) sdata[tid] += sdata[tid+2];
	if(block_size >= 2) sdata[tid] += sdata[tid+1];
}


Template<unsigned int block_size>
__global__ void reduce(float *in_data, float *out_data, unsigned int n) 
{
	extern __shared__ float sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * block_size * 2 + tid;
	unsigned int grid_size = block_size*2*gridDim.x;
	
	sdata[tid] = 0; //init the shared memory with a value

	while(i < n){
		sdata[tid] += in_data[i] + in_data[i+block_size];
		i += grid_size;
	}

	__syncthreads();

	if(block_size >= 512)
	{
		if(tid < 256){
			sdata[tid] += sdata[tid + 256];
		}

		__syncthreads();
	}
	
	if(block_size >= 256)
	{
		if(tid < 128){
			sdata[tid] += sdata[tid + 128];
		}

		__syncthreads();
	}
	
	if(block_size >= 128)
	{
		if(tid < 64){
			sdata[tid] += sdata[tid + 64];
		}

		__syncthreads();
	}


	if(tid < 32) warpReduce(sdata, tid, block_size);
	if(tid == 0) out_data[blockIdx.x] = sdata[0];
}

//DEVICE FUNCTION fill:

//args :
//	float *a -> array to be filled
// 	float x	 -> initialise value

__global__ void fill(float *a , float x)
{
   int index =  blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   
   for(int i = index; i < N; i += stride)
   {
       a[i] = x;
   }
}


//HOST FUNCTION getDetails:

//args:
//	int deviceId -> the cuda device's id

//returns:
//	cudaDeviceProp -> the cuda device properties data structure

cudaDeviceProp getDetails(int deviceId)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    return props;
}

int main(void)
{
	// define the 2 arrays and a tempory array for the products
	float *a, *b, *c;	
	
	// get the cuda device properties	
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props = getDetails(deviceId);	

	int size = N * sizeof(float); //calculate the size of the arrays
		
	// create device copies of the arrays
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size);
    
	// prefetch the data for faster access between devices
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
	cudaMemPrefetchAsync(c, size, deviceId);

	// get the number of sms and then calculate the optimum number of blocks
	int threads_per_block = 512;
    printf("number of sms :%d \n", props.multiProcessorCount);
    int number_of_blocks = props.multiProcessorCount * multi;
	
	// create 2 streans
	cudaStream_t stream_a; cudaStreamCreate(&stream_a);
	cudaStream_t stream_b; cudaStreamCreate(&stream_b);
	
 	cudaError_t asyncErr;

	// execute 2 kernels on 2 different streams to fill the arrays
	fill<<<threads_per_block,number_of_blocks,0,stream_a>>>(a,2.0);
	fill<<<threads_per_block,number_of_blocks,0,stream_b>>>(b,4.0);
	
	// execute a kernel to fill C with the product of a[i] * b[i]
	// the sum of c is the answer to the dot product
	product<<<threads_per_block,number_of_blocks>>>(a,b,c);

	cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);	
	
	cudaFree(a); cudaFree(b); //clean up

	float *result;
   	cudaMallocManaged(&result,size);	
	
	cudaMemPrefetchAsync(result, size, deviceId);

	//C is now an array containing the product of [A] * [B] it must now be split to be added. 
	reduce<<<5, number_of_blocks>>>(c,result,N);	
	
	cudaMemPrefetchAsync(result, size, cudaCpuDeviceId);

	std::cout << "result = " << result[0] << std::endl;

	// sync and check for errors	
	asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

}

