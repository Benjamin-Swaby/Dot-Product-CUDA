//main.cu by benjamin-swaby in Dot-product-CUDA



#define multi 20
#define elements 2048*2048

#include <stdio.h>
#include <iostream>

//DEVICE FUNCTION product

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE); 
    }
}


//args:
//	float *a -> array in 1
// 	float *b -> array in 2
// 	float *c -> c[i] = a[i] * b[i]

__global__ void product(float *a, float *b, float *c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
   	int stride = blockDim.x * gridDim.x;

   	for(int i = index; i < elements; i += stride)
   	{
    	c[i] = a[i] * b[i];
   	}

}



template<unsigned int block_size, typename T>
__device__ void warpReduce(volatile T *sdata, unsigned int tid)
{
	if(block_size >= 64) sdata[tid] += sdata[tid+32];
	if(block_size >= 32) sdata[tid] += sdata[tid+16];
	if(block_size >= 16) sdata[tid] += sdata[tid+8];
	if(block_size >= 8) sdata[tid] += sdata[tid+4];
	if(block_size >= 4) sdata[tid] += sdata[tid+2];
	if(block_size >= 2) sdata[tid] += sdata[tid+1];
}


template<unsigned int blockSize, typename T>
__global__ void reduceCUDA(T *in_data, T *out_data, size_t N) 
{
	__shared__ T sdata[blockSize];

		size_t tid = threadIdx.x;
		size_t i = blockIdx.x*(blockSize) + tid;
		size_t gridSize = blockSize*gridDim.x;
		sdata[tid] = 0;

		while (i < N) { sdata[tid] += in_data[i]; i += gridSize; }
		__syncthreads();

		if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
		if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

		if (tid < 32) warpReduce<blockSize>(sdata, tid);
		if (tid == 0) out_data[blockIdx.x] = sdata[0];
		
		
}


template<size_t blockSize, typename T>
T GPUReduction(T* dA, size_t N)
{
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    T* tmp;
    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    T* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);

    cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}


//DEVICE FUNCTION fill:

//args :
//	float *a -> array to be filled
// 	float x	 -> initialise value

__global__ void fill(float *a , float x)
{
   int index =  blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;
   
   for(int i = index; i < elements; i += stride)
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


void run(float ax , float bx)
{
	// define the 2 arrays and a tempory array for the products
	float *a, *b, *c;	
	
	// get the cuda device properties	
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props = getDetails(deviceId);	

	int size = elements * sizeof(float); //calculate the size of the arrays
		
	// create device copies of the arrays
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size);
    
	// prefetch the data for faster access between devices
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
	cudaMemPrefetchAsync(c, size, deviceId);

	// get the number of sms and then calculate the optimum number of blocks
	#define threads_per_block 512

    //printf("number of sms :%d \n", props.multiProcessorCount);
    int number_of_blocks = props.multiProcessorCount * multi;
	
	// create 2 streans
	cudaStream_t stream_a; cudaStreamCreate(&stream_a);
	cudaStream_t stream_b; cudaStreamCreate(&stream_b);
	
 	cudaError_t asyncErr;

	// execute 2 kernels on 2 different streams to fill the arrays
	fill<<<threads_per_block,number_of_blocks,0,stream_a>>>(a,ax);
	fill<<<threads_per_block,number_of_blocks,0,stream_b>>>(b,bx);
	
	// execute a kernel to fill C with the product of a[i] * b[i]
	// the sum of c is the answer to the dot product
	product<<<threads_per_block,number_of_blocks>>>(a,b,c);

	cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);	
	
	cudaFree(a); cudaFree(b); //clean up

	float *result;
   	cudaMallocManaged(&result,size);	
	
	cudaMemPrefetchAsync(result, size, deviceId);

	//C is now an array containing the product of [A] * [B] it must now be split to be added. 
	float my_result = GPUReduction<512,float>(c,elements);
	std::cout << "result for  "<< ax << " and "<< bx << " == " << my_result << std::endl;


}



int main(void)
{
	for(float i = 0.0; i < 0.227; i+=0.0005)
	{
		run(i, i+1);
	}
}