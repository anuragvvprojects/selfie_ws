#include "gpucost_class.h"
#include "qx_basic.h"
#include "qx_tree_upsampling.h"

using namespace std;
using namespace cv;
__global__ void costVolume( float *dIr, float *dCost, float *dCounter, float *P,float *B,int height,int width, int maxOrd, int nDepth);
__global__ void estimateDepth( float *dCost, float *dCounter, float *dMinm, float *dDepth, int nDepth, int height, int width );

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

void gpucost_::convertPose2T(geometry_msgs::PoseStamped msg, Mat &P_ref){
	float r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw;							
	r_x = (float)msg.pose.position.x;							
	r_y = (float)msg.pose.position.y;							
	r_z = (float)msg.pose.position.z;							
	r_qx = (float)msg.pose.orientation.x;						
	r_qy = (float)msg.pose.orientation.y;						
	r_qz = (float)msg.pose.orientation.z;						
	r_qw = (float)msg.pose.orientation.w;						
	
	Eigen::Matrix3f mat3 = Eigen::Quaternionf(r_qw, r_qx, r_qy, r_qz).toRotationMatrix();		
	Mat R_b2w;
	cv::eigen2cv(mat3,R_b2w);									

	Mat t_b2w =(Mat_<float>(3,1) << r_x,r_y,r_z);							
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++)
			P_ref.at<float>(i,j) = mat3(i,j);
		P_ref.at<float>(i,3) = t_b2w.at<float>(i);
	}
	P_ref.at<float>(3,3) = 1.0;			
	//cout<<P_ref<<endl;						
}

void gpucost_::reset(){
	R_ref = Mat::zeros(3,3,CV_32FC1); R_nei = R_ref.clone();
	T_ref = Mat::zeros(3,1,CV_32FC1); T_nei = T_ref.clone();
	P_ref = Mat::zeros(4,4,CV_32FC1); P_nei = P_ref.clone(); P_nei_inv = P_nei.clone();
/*
	cudaFree(dIr);
	cudaFree(dCost);
	cudaFree(dCounter);
	cudaFree(dDepth);
	cudaFree(dMinm);
	cudaFree(dP);								
	cudaFree(dB);	
	cudaFree(d_texture);								
*/
//	cudaFree(dConf);								


}

gpucost_::gpucost_()
{
	firstTime = true;
	R_ref = Mat::zeros(3,3,CV_32FC1); R_nei = R_ref.clone();
	T_ref = Mat::zeros(3,1,CV_32FC1); T_nei = T_ref.clone();
	P_ref = Mat::zeros(4,4,CV_32FC1); P_nei = P_ref.clone(); P_nei_inv = P_nei.clone();
    height = 360;
    width = 640;
	tPx = height * width;
	//cv::Mat temp = cv::Mat::zeros(height,width,CV_32FC1);
	imgSz = tPx*sizeof(float);
	
	order = 1;
	nDepth = 128;
	volSz = nDepth*imgSz;
	// Allocate dCost
	cudaCheckError(  cudaMalloc((void **)&dCost, volSz) );
	// Allocate dIr
	cudaCheckError(  cudaMalloc((void **)&dIr, imgSz) );
	// Allocate dCounter
	cudaCheckError(  cudaMalloc( (void **)&dCounter, volSz ) );
	//Allocate dDepth
	cudaCheckError(  cudaMalloc((void **)&dDepth, imgSz) );
	//Allocate dMinm
	cudaCheckError(  cudaMalloc((void **)&dMinm, imgSz) );
/*
	cudaCheckError(  cudaMalloc((void **)&dConf, imgSz);
*/
	cudaCheckError(  cudaMalloc((void **)&dP, 3*3*sizeof(float)) );
	cudaCheckError(  cudaMalloc((void **)&dB, 3*3*sizeof(float)) );
	cudaMallocPitch((void**)&d_texture,&pitch, width * sizeof(float), height);

	blockSize = dim3(16,16,1);
	gridSize = dim3((width+blockSize.x-1)/blockSize.x, (height+blockSize.y-1)/blockSize.y, 1);
//	gridSize = dim3(100/16,100/16);

}

void gpucost_::loadRefImg(Mat Ir, geometry_msgs::PoseStamped msg_ref,Mat K)
{
	convertPose2T(msg_ref, P_ref);
	
	P_ref_inv = P_ref.inv();
	cout<<"image size: "<<imgSz<<" "<<volSz<<" "<<Ir.size()<<" "<<Ir.type()<<endl;
	SAFE_CALL(cudaMemcpy(dIr,Ir.ptr(),imgSz,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");//

	SAFE_CALL(cudaMemset(dCost, 0, volSz),"CUDA Memset Failed at dCost");
	SAFE_CALL(cudaMemset(dCounter, 0, volSz),"CUDA Memset Failed at dCounter");
	SAFE_CALL(cudaMemset(dDepth, 0, imgSz),"CUDA Memset Failed at dDepth");
	SAFE_CALL(cudaMemset(dMinm, std::numeric_limits<float>::infinity(), imgSz),"CUDA Memset Failed at dMinm");
    cout << " loadRefImg Complete " << endl;

//	initialize_variables();
}

void gpucost_::downloadDepthmap(cv::Mat &depthMap, cv::Mat &minM){


	estimateDepth<<<gridSize,blockSize>>>( dCost, dCounter, dMinm, dDepth, nDepth, height, width );

	SAFE_CALL(cudaMemcpy(depthMap.ptr(), dDepth, imgSz, cudaMemcpyDeviceToHost),"Download depthmap: CUDA Memcpy Device To Host Failed");
	SAFE_CALL(cudaMemcpy(minM.ptr(), dMinm, imgSz, cudaMemcpyDeviceToHost),"Download minimum cost: CUDA Memcpy Device To Host Failed");
}

void gpucost_::pushimg(Mat Im, geometry_msgs::PoseStamped msg,Mat K)
{	

	convertPose2T(msg, P_nei);
	//cout<<P_nei<<endl<<P_nei.type()<<" "<<P_nei_inv.type()<<endl;
	P_nei_inv = P_nei.inv();
	
	Mat P_r2m = P_nei * P_ref_inv;			    
	Mat R = Mat::zeros(3,3,CV_32FC1);
	Mat T = Mat::zeros(3,1,CV_32FC1);
	for(size_t i=0;i<3;i++){
	for(size_t j=0;j<3;j++)
	    R.at<float>(i,j) = P_r2m.at<float>(i,j);
	T.at<float>(i) = P_r2m.at<float>(i,3);
	}
	Mat Nv = Mat::zeros(1,3,CV_32FC1);
	Nv.at<float>(0,2) = 1.0;
	//cout<<K.type()<<endl;
	Mat Ki = K.inv();

	/*
	Mat IrEstDepth=Mat::zeros(height, width, CV_32FC1);
	Mat entropyMat = Mat::zeros(height, width,CV_32FC1);
	Mat borderMat = Mat::zeros(height, width, CV_8UC1);
	*/

	Mat A = T*Nv;//
	Mat P = K * R * Ki;//
	Mat B = K * A * Ki;//
	cudaMemcpy(dP, P.ptr(), 3*3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B.ptr(), 3*3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_texture, pitch, Im.ptr(), width * sizeof(float), width *sizeof(float), height, cudaMemcpyHostToDevice);
	cudaBindTexture2D(NULL, tex_Im, d_texture, tex_Im.channelDesc, width, height, pitch) ;

/*
	cv::Mat tempImage;
	normalize(Im,tempImage,0.0,1.0,NORM_MINMAX);
	imshow("images",tempImage);
	waitKey(30);
*/	
	costVolume<<<gridSize,blockSize>>>(dIr, dCost, dCounter, dP, dB, height, width , order, nDepth);


}


__global__ void costVolume( float *dIr, float *dCost, float *dCounter,float *P,float *B,int height,int width, int maxOrd, int nDepth)
{
  	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if(xIndex<width && yIndex<height){


		int idx = yIndex*width+xIndex;

		float qmin = 0.1, qmax = 10.001;

		float step = qmax/(nDepth) ;

	
		int order = 1;

		float dH[9];

		for(int di=0; di< nDepth ; di++)     //depth
		{
		          
	    		float dval = qmin + di*step;
										//if(xIndex==100 && yIndex==100)	printf("%e\n",dval);
	    		int volIdx =  di*height*width + idx;

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
							neiImgCost = abs(tex2D(tex_Im, neiIdx_x, neiIdx_y) - dIr[idx]);//Shouldnt the second I value come from Ir rather than Im??
							//fail = 255;
							countPass++;
						}
	    				}
	    	    
					val += neiImgCost;    //L1  
			    	}
			}


			if (countPass > 6 )
			{
				dCounter[ volIdx ] ++;
			}
			dCost[ volIdx ]+=val;

	//		if(xIndex<width && yIndex<height)
	//			dCost [ volIdx ] = dIr[idx];
		
		}
	}

	
//	delete[] dH;

}

__global__ void estimateDepth( float *dCost, float *dCounter, float *dMinm, float *dDepth, int nDepth, int height, int width ){

  	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if(xIndex<width && yIndex<height){
		int idx = yIndex*width+xIndex;
		float qmin = 0.1, qmax = 10.001;
		float step = qmax/(nDepth) ;
		dMinm[idx] = 1000.0f;
		dDepth[idx] = qmin;
		for(int di=0; di< nDepth ; di++)     //depth
		{
			int volIdx = di*height*width + idx;
			if((dCounter[ volIdx ]  > 0))
			{
				float min_cost = dMinm[idx];
				float cur_cost = dCost[ volIdx ]/(float)dCounter[ volIdx ];
				if (min_cost > cur_cost)
					{
						dMinm[idx] = cur_cost;
						dDepth[idx] = qmin + di*step;
					}
			}
		}

	}

}


gpucost_::~gpucost_()
{	
	cudaFree(dCost);
	cudaFree(dIr);
	cudaFree(dCounter);
	cudaFree(dDepth);
	cudaFree(dMinm);
//	cudaFree(dConf);								
	cudaFree(dP);									
	cudaFree(dB);	
	cudaFree(d_texture);								
}

void gpucost_::denoise(cv::Mat &refinedResult, cv::Mat refImg, cv::Mat result){//, cv::Mat normalizedConf){

  int row = refImg.rows, col = refImg.cols;
  qx_tree_upsampling m_tree_upsampling;
  m_tree_upsampling.init(row,col,255,0.2);
  double **disparity = new double*[row];
/*
  std::cout<<normalizedConf.type()<<" "<<result.type()<<std::endl;

  
    float thresh = 0.9;
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(normalizedConf.at<float>(i,j)<thresh){
                result.at<float>(i,j) = 0.0f;
            }
        }
    }
*/   
    double mm, MM;
  cv::minMaxLoc(result, &mm, &MM);
    float factor = 255.0f/(float)MM;
    std::cout<<mm<<" "<<MM<<" Factor is : "<<factor<<std::endl;
    for(int i=0;i<row;i++){
        disparity[i] = new double[col];
        for(int j=0;j<col;j++){
            disparity[i][j] = (double)((result.at<float>(i,j))*factor);
            if(disparity[i][j]<=2)
                disparity[i][j] = 0;
        }
    }
    cv::Mat disp = result.clone();
  cv::minMaxLoc(result, &mm, &MM);
    //std::cout<<mm<<" "<<MM<<std::endl;
  normalize(result,disp,0.0,1.0,cv::NORM_MINMAX);
  //cv::imshow("disparity", disp);


    unsigned char ***guidance_img_=new unsigned char**[1];
  guidance_img_[0]=new unsigned char*[1];
  guidance_img_[0][0]=new unsigned char[row*col*3];
    guidance_img_[0][0] = refImg.data;

    m_tree_upsampling.build_minimum_spanning_tree(guidance_img_);
    m_tree_upsampling.disparity_upsampling(disparity);
    refinedResult = result.clone();
    for(int i=0;i<row;i++)
        for(int j=0;j<col;j++)
            refinedResult.at<float>(i,j) = disparity[i][j]/factor + 0.5;

}


