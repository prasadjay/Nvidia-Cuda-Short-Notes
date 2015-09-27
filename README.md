# Nvidia-Cuda-Beginner-Guide
Provides comprehensive details of Nvidia Cuda API for Beginners.

##What is Nvidia CUDA?

Nvidia CUDA is a parallel programming platform which comes with it's own API developed by Nvidia. CUDA stands for Compute Unified Device Architecture. This API focus on GPGPU (General Purpose Graphics Processing Unit) programming. That in similar term, using gaming GPUs or general computation oriented GPUs such as Nvidia tesla GPUs to use for everyday programming tasks.
To program with this API either CUDA C (ANSI C combined with API interface) or Fortran can be used. In this cheat sheet C language is used. 

##TIPS!
It is required that you know fundamentals of C or C++ programming. If not you should read some text on C language and practise first. No need to be an expert, fundamentals, pointers and memory knowledge would be enough.
In this document some words will have exactly this meaning throughout the document.

HOST = CPU </br>
DEVICE = GPU
Fact = Some tip about specific segment or rule

###Hello World
Find this program in Root Folder as '1-HelloWorld'
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
Calling the function that needed to be run on DEVICE. Arguments for nvcc(Nvidia Cuda Compiler) 1,1 are passed at this state. Meaning of these arguments are explained later. </br>

###First Program on DEVICE
Find this program in Root Folder as '2-SlopeOfLine'</br>
In this code calculating slope of a line is implemented. I hope you know what slope of line is. If not you should be doing grade 7 mathematics before this. Seriously...

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




