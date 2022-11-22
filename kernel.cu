/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#include "CPlusPlus_Common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <stdio.h>

#include "thrust/device_ptr.h"
#include "thrust/sort.h"

__global__ void
calculateDistance(int width, int height, double3 reference, cudaSurfaceObject_t positionsSurface, float4* positions, float* distance)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;
	float4 pos;
	int offset = y * width + x;
	surf2Dread(&pos, positionsSurface, x * (int)sizeof(float4), y, cudaBoundaryModeZero);
	float dist = (pos.x - reference.x) * (pos.x - reference.x) + (pos.y - reference.y) * (pos.y - reference.y) + (pos.z - reference.z) * (pos.z - reference.z);
	distance[offset] = -dist;
	positions[offset] = make_float4(pos.x, pos.y, pos.z, offset);
}


__global__ void
writeToSurface(int width, int height, float4* positions, cudaSurfaceObject_t output) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;
	int offset = y * width + x;
	float4 p = positions[offset];
	surf2Dwrite(make_float4(p.x,p.y,p.z,p.w), output, x * (int)sizeof(float4), y, cudaBoundaryModeZero);
}

int
divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}



cudaError_t
sort(int width, int height, double3 reference, cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	cudaError_t cudaStatus;
	float* distances;
	float4* positions;
	cudaMalloc(&distances, width * height * (int)sizeof(float));
	cudaMalloc(&positions, width * height * (int)sizeof(float4));

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(divUp(width, blockSize.x), divUp(height, blockSize.y), 1);
	

	calculateDistance << <gridSize, blockSize >> > (width, height, reference, input, positions, distances);

	thrust::sort_by_key(thrust::device_ptr < float > (distances),
		thrust::device_ptr<float>(distances + width*height),
		thrust::device_ptr<float4>(positions));

	writeToSurface << <gridSize, blockSize >> > (width, height, positions, output);


#ifdef _DEBUG
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	}
#else
	cudaStatus = cudaSuccess;
#endif
	cudaFree(distances);
	cudaFree(positions);
	return cudaStatus;
}

