#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>
#include <curand.h>
#include <math.h>

using namespace std;

#define nsamples 250000
#define threadsPerBlock 500
#define num_blocks 500

// function to count samples in circle using cpu
void count_samples_CPU(int samples) {

	long long cpu_start = clock();           // start time 
	long long count = 0;	                 // count samples in circle
	for (long i = 0; i < samples; i++) 	 // loop until nsamples
	{
		float x = float(rand()) / RAND_MAX;
		float y = float(rand()) / RAND_MAX;
		float r = x * x + y * y;
		if (r <= 1)
		{
			count++;
		}
	}
	float PI_CPU = 4.0 * float(count) / samples;   // estimated PI value
	clock_t cpu_stop = clock();                     // end time
	float CPU_TIME = float(cpu_stop - cpu_start);   // execution time

	cout << "PI = " << PI_CPU << endl << "Time = " << CPU_TIME << " ms\n\n" << endl;  // print data
}

// Create a kernel to estimate pi
__global__ void count_samples_GPU(float *d_X, float *d_Y, int *d_countInBlocks, int num_block, int samples)
{
	__shared__ int shared_blocks[500];            // shared memory for threads in the same block

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * num_block;

	int inCircle = 0;
	for (int i = index; i < samples; i += stride) {
		float xValue = d_X[i];
		float yValue = d_Y[i];

		if (xValue*xValue + yValue * yValue <= 1.0f) {
			inCircle++;
		}
	}

	shared_blocks[threadIdx.x] = inCircle;
	__syncthreads();                               //  prevent RAW/WAR/WAW hazards

	// Pick thread 0 for each block to collect all points from each Thread.
	if (threadIdx.x == 0)
	{
		int totalInCircleForABlock = 0;
		for (int j = 0; j < blockDim.x; j++)
		{
			totalInCircleForABlock += shared_blocks[j];
		}
		d_countInBlocks[blockIdx.x] = totalInCircleForABlock;
	}
}

int main(void) {

	cout << "\t\t\t*** CUDA TASK ***\n\t\t\t==================\n\n ";
	cout << "Monte Carlo for approximatting PI value :\n------------------------------------------\n\n";
	
	// allocate space to hold host random values    
	float h_randNumsX[nsamples];
	float h_randNumsY[nsamples];
	int * h_countInBlocks = new int[num_blocks];
	float GPU_TIME;

	//Initialize vector with random values from 0:1    
	for (int i = 0; i < h_randNumsX.size(); ++i)
	{
		h_randNumsX[i] = float(rand()) / RAND_MAX;
		h_randNumsY[i] = float(rand()) / RAND_MAX;
	}

	// device copies of random values
	float *d_randNumsX;
	float *d_randNumsY;
	int *d_countInBlocks;

	long long size = nsamples * sizeof(float);

	// allocate device data
	cudaMalloc((void **)&d_randNumsX, size);
	cudaMalloc((void **)&d_randNumsY, size);
	cudaMalloc((void **)&d_countInBlocks, num_blocks * sizeof(int));


	// copy data from host to device
	cudaMemcpy(d_randNumsX, h_randNumsX, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_randNumsY, h_randNumsY, size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;   // define 2 events     

	// create 2 events 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);   // begin START event

	// call kernal 
	count_samples_GPU << < num_blocks, threadsPerBlock >> > (d_randNumsX, d_randNumsY, d_countInBlocks, num_blocks, nsamples);

	cudaEventRecord(stop, 0);    // begin STOP event
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPU_TIME, start, stop); // calculate execution time

	// destroy 2 events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (cudaSuccess != cudaGetLastError())       // if there is any error in cuda running will print Error!
		cout << "Error!\n";

	// Return back the vector from device to host
	cudaMemcpy(h_countInBlocks, d_countInBlocks, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);

	// GPU calculation
	int nsamples_in_circle = 0;
	for (int i = 0; i < num_blocks; i++) {
		nsamples_in_circle += h_countInBlocks[i];
	}
	float PI_GPU = 4.0 * float(nsamples_in_circle) / nsamples;
	cout << "(1) GPU DATA:\n=============\n";
	cout << "PI = " << PI_GPU << "\nTime = " << GPU_TIME << " ms\n" << endl;

	// free device allocation
	cudaFree(d_randNumsX);
	cudaFree(d_randNumsY);
	cudaFree(d_countInBlocks);


	free(h_countInBlocks);

	// CPU calculation	
	cout << "(2) CPU DATA:\n=============\n";
	count_samples_CPU(nsamples);
}


