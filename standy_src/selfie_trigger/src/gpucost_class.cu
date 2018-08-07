#include "gpucost_class.h"

using namespace std;
using namespace cv;


//#define INDEXOFFSET
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

texture<float, 2, cudaReadModeElementType> tex_Im;
texture<float, 2, cudaReadModeElementType> tex_Ir;

gpucost::gpucost()
{

}

gpucost::~gpucost()
{
    if(dCost)
        cudaFree(dCost);
}

void gpucost::loadRefImg(cv::Mat img)
{
	Ir = img;
	height = Ir.rows;
	width = Ir.cols;
	tPx = height * width;
	int imgSz = Ir.cols*Ir.rows*sizeof(float);

	cudaCheckError(  cudaMalloc((void **)&dCost, nDepth*tPx*sizeof(float)) );
	// Allocate dIr
	cudaMalloc((void **)&dIr, imgSz);
	SAFE_CALL(cudaMemcpy(dIr,Ir.ptr(),imgSz,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");//
	// Allocate dCost
	cudaMalloc((void **)&dCost, nDepth*tPx*sizeof(float));
	SAFE_CALL(cudaMemset(dCost, 0, nDepth*tPx*sizeof(float)),"CUDA Memset Failed");
	// Allocate dCounter
	cudaMalloc((void **)&dCounter, nDepth*tPx*sizeof(float));
	SAFE_CALL(cudaMemset(dCounter, 0, nDepth*tPx*sizeof(float)),"CUDA Memset Failed");
	cudaMalloc((void **)&dIm, imgSz);
}

void gpucost::pushimg(cv::Mat Im)
{
	//costVolume<<<gridSize,blockSize>>>(dIr, dCost, dCounter, dDepth, dMinm, dLast, dConf, dest, dP, dB, height, width , maxOrd, nDepth);
}

/*
__global__ void costVolume( float *dIr, float *dCost, float *dCounter, float* dDepth, float* dMinm,bool dLast, float *dConf,float *dest ,float *P,float *B,int height,int width, int maxOrd, int nDepth)
{
  	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = yIndex*width+xIndex;
	float qmin = 0.1, qmax = 10.001;

	float maxRange=0.0;

	float step = qmax/(nDepth) ;
	dConf[idx]=0.0;
	dest[idx]=0.0;
	
	int order = 1;

	float dSum = 0.0;
	float dH[9];
	float dMax = 0.0;
	float dcostMin = 100000.0;
	float dEst = 1000.0;
	int countFail = 0;
	//dCounter[ 0 ] = dCounter[ 0 ] + 1;
	for(int di=0; di< nDepth ; di++)     //depth
	{
                  
    		float dval = qmin + di*step;
									//if(xIndex==100 && yIndex==100)	printf("%e\n",dval);
    	
    		for(int i = 0; i<9; i++)
			dH[i] = P[i] + dval * B[i];
									//if(xIndex==100 && yIndex==100){ for(int i = 0; i<9; i++) printf("%e ",dH[i]); printf("\n");} 
		float val = 0.0;
		//int patch_sz = (2*order + 1)*(2*order + 1);
		//int fail = 0;
		int countPass = 0;
		for(int patch_x = -1*order; patch_x <= order; patch_x++)
		{
			for(int patch_y = -1*order; patch_y <= order; patch_y++)
			{
				int nxIndex = xIndex + patch_x;
				int nyIndex = yIndex + patch_y;
				float u = dH[0] * (float)nyIndex + dH[1] * (float)nxIndex + dH[2];
				float v = dH[3] * (float)nyIndex + dH[4] * (float)nxIndex + dH[5];
				float w = dH[6] * (float)nyIndex + dH[7] * (float)nxIndex + dH[8];
				float neiImgCost = 0;
   				if(w>0 || w <0)
    				{
    					float neiIdx_y = u/w,neiIdx_x = v/w;
      					if(neiIdx_y > height || neiIdx_x > width || neiIdx_y < 0 || neiIdx_x < 0)
      						neiImgCost = 0;
      					else{
        					neiImgCost = abs(tex2D(tex_Im, neiIdx_x, neiIdx_y) - dIr[yIndex*width + xIndex]);//Shouldnt the second I value come from Ir rather than Im??
        					//fail = 255;
        					countPass++;
        				}
    				}
    	    
				val += neiImgCost;    //L1  
		    	}
		}


		if (countPass > 6 )
		{
			dCounter[ di*height*width + yIndex*width + xIndex ] ++;
		}
		dCost[ di*height*width + yIndex*width + xIndex ]+=val;
		


		dMax = max(dMax , val);
		dSum = dSum + val;
	}
	
	//delete[] dH;
}
*/
