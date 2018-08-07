#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cstdio>
#include<iostream>
#include<cmath>
#include<vector_types.h>
#include <sstream>
#include<fstream>

using namespace std;
using namespace cv;
int freq[100000]={0};
float cum_freq[100000]={0.0};
extern void function(Mat&,Mat&,Mat&,Mat&,Mat&,Mat&,Mat&,Mat&,Mat&,Mat&,int,int,int,float,float,int);

void convertQuat2Rot(vector<float> q, Mat &R, Mat &T, Mat &P, Mat &P_inv){
	float sum=0.0;
	for(int i=0;i<4;i++){
		sum=sum+q[i]*q[i];
	}
	//cout<<sum<<endl;
	sum=1.0;//sqrt(sum);
    float w = q[1]/sum;
    float x = q[2]/sum;
    float y = q[3]/sum;
    float z = q[0]/sum;
    R.at<float>(0,0) = 1 - 2*y*y - 2*z*z;
    R.at<float>(0,1) = 2*x*y - 2*w*z;
    R.at<float>(0,2) = 2*x*z + 2*w*y;
    R.at<float>(1,0) = 2*x*y + 2*w*z;
    R.at<float>(1,1) = 1 - 2*x*x - 2*z*z;
    R.at<float>(1,2) = 2*y*z - 2*w*x;
    R.at<float>(2,0) = 2*x*z - 2*w*y;
    R.at<float>(2,1) = 2*y*z + 2*w*x;
    R.at<float>(2,2) = 1 - 2*x*x - 2*y*y;
    
	T.at<float>(0) = q[4];
    T.at<float>(1) = q[5];
    T.at<float>(2) = q[6];

//	R = -1 * R;   
// 	T = -1 * T;
    for(size_t i=0;i<3;i++){
        for(size_t j=0;j<3;j++)
            P.at<float>(i,j) = R.at<float>(i,j);
        P.at<float>(i,3) = T.at<float>(i);
    }
    P.at<float>(3,3) = 1.0;
    P_inv = P.inv();
}

void readProjectionmat(Mat& m1,string s,Mat& P_inv){
    ifstream myfile(s.c_str());
    string tmp;
    myfile>>tmp;
    for(int i=0;i<3;i++){
        for(int j=0;j<4;j++){
            myfile>>m1.at<float>(i,j);
        }
    }
    myfile.close();
	P_inv=m1.inv();
}

void readDepthfile(Mat& m1, char * s){
    ifstream myfile(s);
    float z;
    for(int i = 0;i < m1.rows; i++){
        for(int j = 0; j < m1.cols; j++){
            myfile>>z;
	    m1.at<float>(i, j) = z/100.0f;
        }
    }
    myfile.close();
}


int main(int argc, char* argv[]){
	string datasetPath = "../test_data/";
    string imagePath = datasetPath + "images/";

    int numRef = atoi(argv[1]), numNei = atoi(argv[2]);
    string fileName = datasetPath + "first_200_frames_traj_over_table_input_sequence.txt";	
	cout<<fileName<<endl;
    ifstream fp;
    fp.open(fileName.c_str());
    string tempS, ImString, IrString;
    float tempD;
    vector<float> IrProj, ImProj;
    for(size_t i=0;i<200;i++){
        if(i==numRef){        
            fp>>tempS;
            IrString = imagePath + tempS;
            //cout<<tempS<<' '<<IrString<<endl;
            for(size_t j=0;j<7;j++){
                fp>>tempD;
                IrProj.push_back(tempD);
            }
        }
        else if(i==numNei){
            fp>>tempS;
            ImString = imagePath + tempS;
            //cout<<tempS<<' '<<ImString<<endl;
            for(size_t j=0;j<7;j++){
                fp>>tempD;
                ImProj.push_back(tempD);
            }
        }
        else{
            fp>>tempS;
            for(size_t j=0;j<7;j++){
                fp>>tempD;
            }
        }        
    }
    Mat R_ref = Mat::zeros(3,3,CV_32FC1), R_nei = R_ref.clone();
    Mat T_ref = Mat::zeros(3,1,CV_32FC1), T_nei = T_ref.clone();
    Mat P_ref = Mat::zeros(4,4,CV_32FC1), P_nei = P_ref.clone();
    Mat P_ref_inv, P_nei_inv;
    convertQuat2Rot(IrProj,R_ref,T_ref,P_ref,P_ref_inv);
    convertQuat2Rot(ImProj,R_nei,T_nei,P_nei,P_nei_inv);
//	cout<<P_ref<<endl;
//	readProjectionmat(P_ref,"1.txt",P_ref_inv);
//	readProjectionmat(P_nei,"2.txt",P_nei_inv);
	Mat K = Mat::zeros(3,3,CV_32FC1);
    K.at<float>(0,0) = 481.2;
    K.at<float>(0,2) = 319.5;
    K.at<float>(1,1) = -480.0;
    K.at<float>(1,2) = 239.5;
    K.at<float>(2,2) = 1.0;
    Mat Ki = K.inv();

    Mat P_r2m = P_nei * P_ref_inv;
  //cout<<P_r2m<<endl; 
    
    Mat Ir = imread(IrString.c_str(),0);
    Ir.convertTo(Ir,CV_32F);
//    Ir = Ir/255.0;
    Mat Im = imread(ImString.c_str(),0);
    Im.convertTo(Im,CV_32F);
//    Im = Im/255.0;
    #if 0
    imshow("Reference Image",Ir);
    imshow("Neighborhood Image",Im);
    waitKey(0);
    #endif
    
    Mat R = Mat::zeros(3,3,CV_32FC1);
    Mat T = Mat::zeros(3,1,CV_32FC1);
    Mat Nv = Mat::zeros(1,3,CV_32FC1);
    for(size_t i=0;i<3;i++){
        for(size_t j=0;j<3;j++)
            R.at<float>(i,j) = P_r2m.at<float>(i,j);
        T.at<float>(i) = P_r2m.at<float>(i,3);
    }
    Nv.at<float>(0,2) = 1.0;
	//cout<<Nv<<endl; 
    int height = Ir.rows, width = Ir.cols;
    Mat entropyMat = Mat::zeros(Ir.size(),CV_32FC1);
    Mat borderMat = Mat::zeros(Ir.size(), CV_8UC1);

	char IrDepthfile[100];
	Mat IrDepth=Mat::zeros(height,width,CV_32FC1);
	sprintf(IrDepthfile, "../test_data/depthmaps/scene_%03d.depth", atoi(argv[1]));
	readDepthfile(IrDepth,IrDepthfile);
    
    int order = atoi(argv[3]);
    float dmin = 0.001;
    float stepSz = 0.10;
    int nDepth = 128;
    Mat IrEstDepth=Mat::zeros(height, width, CV_32FC1);
	function(Ir,Im,R,T,Nv,K,Ki,entropyMat,borderMat,IrEstDepth,height,width,order,dmin,stepSz,nDepth);

#if 1
    double minE, maxE;
    cv::minMaxLoc(IrDepth, &minE, &maxE);
    cout<<"Real depth "<<minE<<' '<<maxE<<endl;
    cv::minMaxLoc(IrEstDepth, &minE, &maxE);
    cout<<"Estimated depth "<<minE<<' '<<maxE<<endl;

	Mat Error=entropyMat.mul(abs(IrDepth-IrEstDepth));
 	cv::minMaxLoc(Error, &minE, &maxE);
//    cout<<minE<<' '<<maxE<<endl;
	if(abs(minE)>maxE)
		maxE = abs(minE);

	normalize(entropyMat, entropyMat, 0.0, 1.0,NORM_MINMAX);
    Mat writeIm0 = entropyMat * 255;
    Mat writeIm = Mat::zeros(height, width, CV_8UC1);
    normalize(writeIm0, writeIm, 0, 255, NORM_MINMAX);
    char str[100] = "";
    sprintf(str, "%d.jpg", order);
    imwrite(str, writeIm);
	float step=0.01;
	ofstream file;
	
	int maxNum = (height-2*order)*(width-2*order);
	for(int i=order;i<height-order;i++){
		for(int j=order;j<width-order;j++){
			freq[(int)(Error.at<float>(i,j)/step)]++;
		}
	}
	cum_freq[0]=float(freq[0])/float(maxNum);
	char fname[100];
	sprintf(fname,"out_%d.txt",order);
	file.open(fname);
	for(int i=0;i<=maxE/step;i++){
		if(i>0)
			cum_freq[i]=float(freq[i])/float(maxNum)+cum_freq[i-1];
//		cout<<cum_freq[i]<<' ';
		file<<(float)i*step<<" "<<freq[i]<<" "<<cum_freq[i]<<endl;
	}
	file.close();
#endif
    return 0;
	
}
