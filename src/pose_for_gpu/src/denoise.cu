#include "gpucost_class.h"

using namespace std;
using namespace cv;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__global__ void initialize_vol_float(int nD, int height, int width, float a, float *dArr)
{
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
	if(idx<width && idy<height && idz<nD)
		dArr[idz*height*width + idy*width + idx] = a;
}

__global__ void initialize_vol_int(int nD, int height, int width, int a, int *dArr)
{
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
	if(idx<width && idy<height && idz<nD)
		dArr[idz*height*width + idy*width + idx] = a;
}

__global__ void initialize_image(int height, int width, float a, float *dArr)
{
  	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = yIndex*width+xIndex;
	dArr[idx] = a;
}


void gpucost_::initialize_variables(){
	int BLKXSIZE = 8;
	int BLKYSIZE = 8;
	int BLKZSIZE = 2;
	dim3 blockSize1(BLKXSIZE, BLKYSIZE, BLKZSIZE);
	dim3 gridSize1(((width+BLKXSIZE-1)/BLKXSIZE), ((height+BLKYSIZE-1)/BLKYSIZE), ((nDepth+BLKZSIZE-1)/BLKZSIZE));
	initialize_vol_float<<<gridSize1,blockSize1>>>(nDepth,height,width,0.0f,dCost);
	initialize_vol_float<<<gridSize1,blockSize1>>>(nDepth,height,width,0.0f,dCounter);
	
	dim3 blockSize2(8, 8, 1);
	dim3 gridSize2((width)/blockSize2.x, (height)/blockSize2.y, 1);
	initialize_image<<<gridSize2,blockSize2>>>(height,width,0.0f,dDepth);
	initialize_image<<<gridSize2,blockSize2>>>(height,width,0.0f,dConf);
	initialize_image<<<gridSize2,blockSize2>>>(height,width,1000.0f,dMinm);
}
