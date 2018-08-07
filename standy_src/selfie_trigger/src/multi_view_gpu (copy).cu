#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <iostream>
#include <algorithm>
#include <limits>
using namespace cv;
using namespace std;

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

__global__ void cost_refinement( float *dCost, float *dCounter, float* dDepth, float* dMinm, int height,int width, int maxOrd, int nDepth)
{
  	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = yIndex*width+xIndex;
	float qmin = 0.1, qmax = 10.001;
	float step = qmax/(nDepth) ;
	dMinm[yIndex*width + xIndex] = 1000.0f;
	dDepth[yIndex*width + xIndex] = qmin;
	for(int di=0; di< nDepth ; di++)     //depth
	{

		if((dCounter[ di*height*width + yIndex*width + xIndex ]  > 0))
		{
			float min_cost = dMinm[yIndex*width + xIndex];
			float cur_cost = dCost[ di*height*width + yIndex*width + xIndex ]/(float)dCounter[ di*height*width + yIndex*width + xIndex ];
			if (min_cost > cur_cost)
				{
					dMinm[yIndex*width + xIndex] = cur_cost;
					dDepth[yIndex*width + xIndex] = qmin + di*step;
				}
		}
		else
			dCost[ di*height*width + yIndex*width + xIndex ] = 100.0;

	}
}

void function(Mat& Ir,vector<Mat>& Im, vector<Mat>& R,vector<Mat>& T,Mat& Nv,Mat& K,Mat& Ki,Mat& confMat, Mat &borderFlag, Mat &IrEstDepth, int height, int width,int order,float dMin,float stepSz, int nDepth)
{
	if( !( Ir.rows == height ) || !( Ir.cols == width ) )
	{
		cout << " Input image is not of  same dimensions as heignt and width. Return;. " << endl; return;
	}

	bool DEBUG = 0;
	Mat qmat=Mat::zeros(height,width,CV_32FC1);
	float *dIr;
	float *dIm;
	float *dCost;
	float *dCounter;	
	float *dDepth;
	float *dMinm;
	float *dest;
	float *dP;
	float *dB;
	float *dConf;
	//int *dBorderFlag;
	int maxOrd=order;
	const int tPx = height*width;
	const int imgSz = Ir.cols*Ir.rows*sizeof(float);
													
	cudaMalloc((void **)&dIr, imgSz);								
													
	SAFE_CALL(cudaMemcpy(dIr,Ir.ptr(),imgSz,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");//
	cudaMalloc((void **)&dCost, nDepth*tPx*sizeof(float));						
	SAFE_CALL(cudaMemset(dCost, 0, nDepth*tPx*sizeof(float)),"CUDA Memset Failed");			
	cudaMalloc((void **)&dCounter, nDepth*tPx*sizeof(float));					
	cudaMemset(dCounter, 0, nDepth*tPx*sizeof(int));
	cudaMalloc((void **)&dDepth, imgSz);								
	cudaMemset(dDepth, 0, tPx*sizeof(float));
	cudaMalloc((void **)&dMinm, tPx*sizeof(float));							
	cudaMemset(dMinm, std::numeric_limits<float>::infinity(), tPx*sizeof(float));//cout << " Value of dMinm :" << dMinm[0] <<endl;
													

	dim3 blSz(8,32,1);								
	dim3 thSz(nDepth/blSz.x, (tPx)/blSz.y, 1);
	dim3 blockSize(16, 16, 1);
	dim3 gridSize((Ir.cols)/blockSize.x, (Ir.rows)/blockSize.y, 1);	

	cudaMalloc((void **)&dIm, imgSz);						
	cudaMalloc((void **)&dConf, tPx*sizeof(float));					
	cudaMalloc((void **)&dest, tPx*sizeof(float));					
	cudaMalloc((void **)&dP, 3*3*sizeof(float));					
	cudaMalloc((void **)&dB, 3*3*sizeof(float));					
											

	size_t pitch;
	float* d_texture; //Device texture
											
	cudaMallocPitch((void**)&d_texture,&pitch, width * sizeof(float), height);
											
//--------------FOR LOOP............................
	for(int i = 0 ; i < Im.size() ; i++ )
		{
		Mat A = T[i]*Nv;//
		Mat P = K * R[i] * Ki;//
		Mat B = K * A * Ki;//
										
		cudaMemcpy(dP, P.ptr(), 3*3*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dB, B.ptr(), 3*3*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy2D(d_texture, pitch, Im[i].ptr(), width * sizeof(float), width *sizeof(float), height, cudaMemcpyHostToDevice);
		cudaBindTexture2D(NULL, tex_Im, d_texture, tex_Im.channelDesc, width, height, pitch) ;
		//-------COstVolume Calculation------------------------------------------------------------------------------------------
		bool dLast = (i == (Im.size() -1));				
		costVolume<<<gridSize,blockSize>>>(dIr, dCost, dCounter, dDepth, dMinm, dLast, dConf, dest, dP, dB, height, width , maxOrd, nDepth);
		//-----------------------------------------------------------------------------------------------------------------------
													
		double minE, maxE;
		cv::minMaxLoc(confMat, &minE, &maxE);				
										
		cudaUnbindTexture(tex_Im);					
		}
		cost_refinement<<<gridSize,blockSize>>>(dCost, dCounter, dDepth, dMinm, height, width , maxOrd, nDepth);
		DEBUG = 1;
											
		double minm, maxm;

		cv::Mat mDepth = cv::Mat::zeros(Ir.size(),IrEstDepth.type());
		cudaMemcpy(mDepth.ptr(), dDepth, tPx*sizeof(float), cudaMemcpyDeviceToHost);
		minMaxLoc(mDepth, &minm, &maxm);
											
		cv::Mat mMinm = cv::Mat::zeros(Ir.size(),IrEstDepth.type());
		cudaMemcpy(mMinm.ptr(), dMinm, tPx*sizeof(float), cudaMemcpyDeviceToHost);
		minMaxLoc(mMinm, &minm, &maxm);
		/*normalize(mDepth,mDepth,0,1,NORM_MINMAX);						if(DEBUG)cout<<"52 :minmax of mMinm : "<<minm<<" , "<<maxm<<endl;
		imshow("dDepth",mDepth);
		waitKey();*/
		IrEstDepth = mDepth.clone();
		cv::Mat mCost[nDepth];
		cv::Mat mCounter[nDepth];						
		cudaFree(dIm);								
		cudaFree(d_texture);
		cudaFree(dConf);							
		cudaFree(dest);								
		cudaFree(dP);								
		cudaFree(dB);								
		cudaFree(dIr);//printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));
		cudaFree(dCost);
		cudaFree(dCounter);
		cudaFree(dMinm);
		cudaFree(dDepth);	
}



