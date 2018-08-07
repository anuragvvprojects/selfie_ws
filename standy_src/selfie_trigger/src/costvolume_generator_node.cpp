#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>


#include<cstdio>
#include<iostream>
#include<cmath>
#include<vector_types.h>
#include <sstream>
#include<fstream>

//#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <poseimagearraymsg/poseImageArray.h>

#include "Yang_cvpr12/qx_basic.h"
#include "Yang_cvpr12/qx_tree_upsampling.h"


using namespace std;
using namespace cv;

char folder[1024];
char filepath[1024];
Mat K, D;
Mat t_b2co, R_b2co;
Mat t_b2cb, R_b2cb;



std::vector<Mat> cost_vol_images;
std::vector<geometry_msgs::PoseStamped> cost_vol_poses;
extern void function(Mat& Ir,vector<Mat>& Im,vector<Mat>& R, vector<Mat>& T,Mat& Nv,Mat& K,Mat& Ki,Mat& confMat, Mat &borderFlag, Mat &IrEstDepth, int height, int width,int order,float dMin,float stepSz, int nDepth);

void convertPose2T(int i_ref, const poseimagearraymsg::poseImageArrayConstPtr msg, Mat &P_ref);

void denoise(cv::Mat &refinedResult, cv::Mat refImg, cv::Mat result);//, cv::Mat normalizedConf);


void convertPose2T(int i_ref, const poseimagearraymsg::poseImageArrayConstPtr msg, Mat &P_ref){
	bool DEBUG = 0;
	float r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw;							if(DEBUG)cout << " ------------------------------------ " << endl;
	r_x = msg->poseArray.at(i_ref).pose.position.x;									if(DEBUG)cout << "\nr_x " << r_x; 
	r_y = msg->poseArray.at(i_ref).pose.position.y;									if(DEBUG)cout << "\nr_y " << r_y; 
	r_z = msg->poseArray.at(i_ref).pose.position.z;									if(DEBUG)cout << "\nr_z " << r_z; 
	r_qx = msg->poseArray.at(i_ref).pose.orientation.x;								if(DEBUG)cout << "\nr_qx " << r_qx; 
	r_qy = msg->poseArray.at(i_ref).pose.orientation.y;								if(DEBUG)cout << "\nr_qy " << r_qy;
	r_qz = msg->poseArray.at(i_ref).pose.orientation.z;								if(DEBUG)cout << "\nr_qz " << r_qz;
	r_qw = msg->poseArray.at(i_ref).pose.orientation.w;								if(DEBUG)cout << "\nr_qw" << r_qw;
	
	Eigen::Matrix3f mat3 = Eigen::Quaternionf(r_qw, r_qx, r_qy, r_qz).toRotationMatrix();			if(DEBUG)cout << "mat3 :\n" << mat3 <<endl<< endl;
	Mat R_b2w;
	cv::eigen2cv(mat3,R_b2w);										if(DEBUG)cout << "R_b2w :\n" << R_b2w <<endl<< endl;

	//R_b2w.convertTo(R_b2w, CV_64F);										
	Mat t_b2w =(Mat_<float>(3,1) << r_x,r_y,r_z);								if(DEBUG)cout << "t_b2w :\n" << t_b2w <<endl<< endl;
/*
	Mat R_w2cb = R_b2cb * R_b2w.t();								if(DEBUG)cout << "R_w2cb :\n" << R_w2cb <<endl<< endl;
if(DEBUG)cout << "Types :" << "t_b2cb : " << t_b2cb.type() << " R_b2cb : " << R_b2cb.type() << " R_b2w : " << R_b2w.type() << " t_b2w : " << t_b2w.type() << endl;
	Mat t_w2cb = t_b2cb + R_b2cb*(-R_b2w.t()*t_b2w);						if(DEBUG)cout << "t_w2cb :\n" << t_w2cb <<endl<< endl;
	
	Mat R_ref = R_w2cb.clone();										if(DEBUG)cout << "R_ref :\n" << R_ref <<endl<< endl;
	Mat T_ref = t_w2cb.clone();										if(DEBUG)cout << "T_ref :\n" << T_ref <<endl<< endl;
*/
	//R_ref.convertTo(R_ref, CV_32F);	
	//T_ref.convertTo(T_ref, CV_32F);									
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++)
			P_ref.at<float>(i,j) = mat3(i,j);
		P_ref.at<float>(i,3) = t_b2w.at<float>(i);
	}
	P_ref.at<float>(3,3) = 1.0;									if(DEBUG)cout << "P_ref :\n" << P_ref <<endl<< endl;
}

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


void bundle_callback(const poseimagearraymsg::poseImageArrayConstPtr msg)
{
	cout << "callback fired----------------------------" << endl;
//--------copied from cost_vol_node
    
    bool DEBUG = 0;
	int i_ref = (int)(msg->imageArray.size()/2);
    //cv::namedWindow("cost_vol_batch");
    for(int i = 0 ; i < msg->imageArray.size() ; i++)
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg->imageArray.at(i), sensor_msgs::image_encodings::BGR8);
																		if(DEBUG){imshow("cost_vol_batch",cv_ptr->image);waitKey(3);}
	Mat imageTemp,imageFloat;
	cvtColor(cv_ptr->image,imageTemp,cv::COLOR_BGR2GRAY);
	imageTemp.convertTo(imageTemp,CV_32F);
	normalize(imageTemp,imageFloat,0,1,NORM_MINMAX);
        
	cost_vol_images.push_back(imageFloat);
	cost_vol_poses.push_back(msg->poseArray.at(i));
    }
	Mat refImage;
	cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(msg->imageArray.at(i_ref), sensor_msgs::image_encodings::BGR8);
	refImage = cv_ptr->image.clone();
//-------------------------------------
				// ------- code from arnab
			    	    Mat R_ref = Mat::zeros(3,3,CV_32FC1), R_nei = R_ref.clone();
				    Mat T_ref = Mat::zeros(3,1,CV_32FC1), T_nei = T_ref.clone();
				    Mat P_ref = Mat::zeros(4,4,CV_32FC1), P_nei = P_ref.clone();
				    Mat P_ref_inv, P_nei_inv;
				    vector<float> IrProj, ImProj;
				    
				    //---------------------------TODO Check the transformation matrices calculation (Its better to do this at the other package, where you are calculating your camera pose)
				        // ----- For reference image
					
					convertPose2T(i_ref, msg, P_ref);

						P_ref_inv = P_ref.inv();									if(DEBUG)cout << "P_ref_inv :\n" << P_ref_inv <<endl<< endl;
			vector<Mat> Rvect;
			vector<Mat> Tvect;
			for( int i =0; i < cost_vol_images.size() ; i++)
			   {																if(DEBUG)cout << "For i = " << i << "--------"<<endl;
				    //--------------------------
						// ----- For mth image
				convertPose2T(i, msg, P_nei);
/*
						r_x = msg->poseArray.at(i).pose.position.x;									if(DEBUG)cout << "\nr_x " << r_x; 
						r_y = msg->poseArray.at(i).pose.position.y;									if(DEBUG)cout << "\nr_y " << r_y; 
						r_z = msg->poseArray.at(i).pose.position.z;									if(DEBUG)cout << "\nr_z " << r_z; 
						r_qx = msg->poseArray.at(i).pose.orientation.x;								if(DEBUG)cout << "\nr_qx " << r_qx; 
						r_qy = msg->poseArray.at(i).pose.orientation.y;								if(DEBUG)cout << "\nr_qy " << r_qy; 
						r_qz = msg->poseArray.at(i).pose.orientation.z;								if(DEBUG)cout << "\nr_qz " << r_qz; 
						r_qw = msg->poseArray.at(i).pose.orientation.w;								if(DEBUG)cout << "\nr_qw" << r_qw; 
						
						mat3 = Eigen::Quaternionf(r_qw, r_qx, r_qy, r_qz).toRotationMatrix();			if(DEBUG)cout << "mat3 :\n" << mat3 <<endl<< endl;
						cv::eigen2cv(mat3,R_b2w);										if(DEBUG)cout << "R_b2w :\n" << R_b2w <<endl<< endl;

						//R_b2w.convertTo(R_b2w, CV_64F);
						t_b2w =(Mat_<float>(3,1) << r_x,r_y,r_z);								if(DEBUG)cout << "t_b2w :\n" << t_b2w <<endl<< endl;

						R_w2cb = R_b2cb * R_b2w.t();								if(DEBUG)cout << "R_w2cb :\n" << R_w2cb <<endl<< endl;
						t_w2cb = t_b2cb + R_b2cb*(-R_b2w.t()*t_b2w);						if(DEBUG)cout << "t_w2cb :\n" << t_w2cb <<endl<< endl;
						// w2cb * cbref2w 
						R_nei = R_w2cb * R_ref.t();
						T_nei = t_w2cb + R_w2cb*( -R_ref.t() * T_ref );
						
						for(int i=0;i<3;i++){
							for(int j=0;j<3;j++)
								P_nei.at<float>(i,j) = R_nei.at<float>(i,j);
							P_nei.at<float>(i,3) = T_nei.at<float>(i);
						}
						P_nei.at<float>(3,3) = 1.0;
*/									if(DEBUG)cout << "P_ref :\n" << P_ref <<endl<< endl;
						P_nei_inv = P_nei.inv();									if(DEBUG)cout << "P_ref_inv :\n" << P_ref_inv <<endl<< endl;

				    //convertQuat2Rot(IrProj,R_ref,T_ref,P_ref,P_ref_inv);
				    //convertQuat2Rot(ImProj,R_nei,T_nei,P_nei,P_nei_inv);
				//	cout<<P_ref<<endl;
				//	readProjectionmat(P_ref,"1.txt",P_ref_inv);
				//	readProjectionmat(P_nei,"2.txt",P_nei_inv);

				    Mat P_r2m = P_nei * P_ref_inv;									if(DEBUG)cout << "P_r2m :\n" << P_r2m <<endl<< endl;
				    //cout<<P_nei<<endl<<P_ref<<endl<<endl;
				  //cout<<P_r2m<<endl; 
				    
				 //   Ir.convertTo(Ir,CV_32F);
				//    Ir = Ir/255.0;
				//    Im = Im/255.0;
				    #if 0
				    imshow("Reference Image",Ir);
				    imshow("Neighborhood Image",Im);
				    waitKey(0);
				    #endif
				    
				    Mat R = Mat::zeros(3,3,CV_32FC1);
				    Mat T = Mat::zeros(3,1,CV_32FC1);
				    for(size_t i=0;i<3;i++){
					for(size_t j=0;j<3;j++)
					    R.at<float>(i,j) = P_r2m.at<float>(i,j);
					T.at<float>(i) = P_r2m.at<float>(i,3);
				    }
				    Rvect.push_back(R);
				    Tvect.push_back(T);
					//cout<<Nv<<endl; 
				    

					//char IrDepthfile[100];
					//Mat IrDepth=Mat::zeros(height,width,CV_32FC1);
					//sprintf(IrDepthfile, "../test_data/depthmaps/scene_%03d.depth", atoi(argv[1]));
					//readDepthfile(IrDepth,IrDepthfile);
			   }

			    int order = 1;
			    float dmin = 0.001;
			    float stepSz = 0.10;
			    int nDepth = 128;
			    Mat Nv = Mat::zeros(1,3,CV_32FC1);
			    Nv.at<float>(0,2) = 1.0;

			    Mat Ki = K.inv();
			    
			    Mat Ir = cost_vol_images[i_ref].clone();int height = Ir.rows, width = Ir.cols;
			    Mat IrEstDepth=Mat::zeros(height, width, CV_32FC1);
			    Mat entropyMat = Mat::zeros(Ir.size(),CV_32FC1);
			    Mat borderMat = Mat::zeros(Ir.size(), CV_8UC1);
				if(!DEBUG)cout<< "In cpp: size of cost_vol_images :" << cost_vol_images.size()<< endl;
			    function(Ir,cost_vol_images,Rvect,Tvect,Nv,K,Ki,entropyMat,borderMat,IrEstDepth,height,width,order,dmin,stepSz,nDepth);
			Mat refinedResult, tempMat;
			denoise(refinedResult, refImage, IrEstDepth);
			normalize(refinedResult,tempMat,0,1,NORM_MINMAX);
			imshow("refinedResult",tempMat);
			waitKey(30);
				    
				    //imshow("InverseDepth",IrEstDepth);waitKey(8);
cost_vol_images.clear();
cost_vol_poses.clear();
	

}










int main(int argc, char* argv[]){

//---- copied from dfusmc_node---------------
  bool DEBUG = 0;
  ros::init(argc, argv, "costvolume_node");
  ros::NodeHandle n;
  //pub = n.advertise< sensor_msgs::PointCloud2 > ("/cost_vol_cloud", 1);
  ros::Subscriber sub = n.subscribe("/poseimagebundle",1,bundle_callback);

//-------------------------------------------
  sprintf(folder,"/home/anurag/mother_ws/cost_volume_ws/src/costvolume_generator/dataset");
  
  //Read Intrinsic parameters
  sprintf(filepath,"%s/internal_calib.yml",folder);if(DEBUG)cout << " -12- " << endl;
  FileStorage fs_i(filepath, FileStorage::READ);if(DEBUG)cout << " -13- " << endl;
  fs_i["camera_matrix"] >> K;K.convertTo(K, CV_32F);if(DEBUG)cout << " -14- " << endl;
  fs_i["distortion_coefficients"] >> D;D.convertTo(D, CV_32F);if(DEBUG)cout << " -15- " << endl;
  
  //Read have R_b2c,t_b2c from file
  sprintf(filepath,"%s/RT_b2co.yml",folder);if(DEBUG)cout << " -16- " << endl;
  FileStorage fs_o(filepath, FileStorage::READ);if(DEBUG)cout << " -17- " << endl;
  fs_o["R_b2co"] >> R_b2co;R_b2co.convertTo(R_b2co, CV_32F);if(DEBUG)cout << " -18- " << endl;
  fs_o["T_b2co"] >> t_b2co;t_b2co.convertTo(t_b2co, CV_32F);if(DEBUG)cout << " -19- " << endl;

  sprintf(filepath,"%s/RT_b2cb.yml",folder);if(DEBUG)cout << " -16- " << endl;
  FileStorage fs_b(filepath, FileStorage::READ);if(DEBUG)cout << " -17- " << endl;
  fs_b["R_b2cb"] >> R_b2cb;R_b2cb.convertTo(R_b2cb, CV_32F);if(DEBUG)cout << " -18- " << endl;
  fs_b["T_b2cb"] >> t_b2cb;t_b2cb.convertTo(t_b2cb, CV_32F);if(DEBUG)cout << " -19- " << endl;
  cout << "\nt_b2cb\n"<<t_b2cb << endl;
	
	
	ros::spin();
}



















void denoise(cv::Mat &refinedResult, cv::Mat refImg, cv::Mat result){//, cv::Mat normalizedConf){

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


