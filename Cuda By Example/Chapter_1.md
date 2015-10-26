###Hello World

```C
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void kernel(){}

int main()
{
   kernel<<<1,1>>>();
   printf("Hello Cuda!");
   return 0;
}
```

####__global__ : 
Keyword which specifies that this function is going to run on the DEVICE </br>
####kernel<<<1,1>>>() : 
Calling the function that needed to be run on DEVICE. Arguments for nvcc(Nvidia Cuda Compiler) 1,1 are passed at this state. Defines Blocks and Threads to be used. </br>

###First Program on DEVICE
In this code calculating slope of a line is implemented..

```C
#include "cuda_runtime.h"
#include <stdio.h>

__global__ void calculateSlope(int x1, int y1, int x2, int y2, double *slope)
{
	int numerator = y2-y1;
	int denominator = x2-x1;
	*slope = ((double)numerator/(double)denominator);
}

int main()
{
    double slope;
	double *slopePtr;

	cudaMalloc((void**)&slopePtr, sizeof(double));
	calculateSlope<<<1,1>>>(1, 1, 4, 5, slopePtr);
	cudaMemcpy(&slope, slopePtr, sizeof(double), cudaMemcpyDeviceToHost);
	printf("Answer : %f", slope);
	cudaFree(slopePtr);
    return 0;
}
```

####cudaMalloc :  
Allocates memory on GPU </br>
####cudaMemCpy : 
Copies data between HOST and DEVICE... There are 4 arguments...</br>
		1. Destination memory address : In our case slope variable</br>
		2. Source pointer : Which is slopePtr</br>
		3. size of the variable</br>
		4. type</br>
			..*cudaMemcpyDeviceToHost : Copy data from DEVICE to HOST</br>
			..*cudaMemcoyHostToDevice : Copy data from HOST to DEVICE</br>


####Fact :
You can never modify GPU memory from HOST. If you want to modify you will always have to use data copying in between HOST and DEVICE.


###Device Querying

Used to get details about the hardware and software perks of GPU.

```C
struct cudaDeviceProp{
char name[256];    		//Name of the GPU
size_t totalGlobalMem;    	//Global Memory in Bytes	    	
size_t sharedMemPerBlock;       //Max shared memory per block 
int regsPerBlock;        	//Number of 32bit Registerd per block
int warpSize;        		//Number of threads per warp
size_t memPitch;       		//Max pich allows for memory copies in Bytes
int maxThreadsPerBlock;        	//Max threads per block
int maxThreadsDim[3];        	//Max threads per each dimension in block
int maxGridSize[3];        	//Max threads per each dimension in grid
size_t totalConstMem;        	//Available constant memory
int major;			//Major version of Compute Engine
int minor;        		//Minor version of Compute Engine
int clockRate;        		//Clock speed of GPU
size_t textureAlignment;        //Device's requirement for texture alignment
int deviceOverlap;        	//Ability to perform cudaMemcpy and kernal execution at once in Bool
int multiProcessorCount;        //Number of multiprocesors
int kernelExecTimeoutEnabled;   //Having of run time limit for kernal executions in bool
int integrated;        		//Is GPU Integrated or Discrete in Bool
int canMapHostMemory;        	//Ability to map host memory to device address in bool
int computeMode;        	//GPU computing mode (default/exclusive/prohibited)
int maxTexture1D;        	//Max size for 1D textures
int maxTexture2D[2];        	//Max size for 2D textures
int maxTexture3D[3];        	//Max size for 3D textures
int maxTexture2DArray[3];      	//Max dimensions supported for 2D texture arrays
int concurrentKernels;		//Ability to execute multiple kernals within same context simultaneously in bool
}

cudaDeviceProp prop;

int deviceCount;
cudaGetDeviceCount(&deviceCount);	//check for SLI configs

for(int x=0; x<deviceCount; x++){
	//Accessing each GPU
	cudaGetDeviceProperties(&prop, x);
	printf("Device Name : %s\n", prop.name);
}
```

###Handling SLI Configs

```C
cudaSetDevice(<device_number>); //0-n
```




