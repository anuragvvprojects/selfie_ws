#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <essentials.hpp>

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


//#include "Yang_cvpr12/qx_basic.h"
//#include "Yang_cvpr12/qx_tree_upsampling.h"


using namespace std;
using namespace cv;


int main()
{




	// Given R, T 
		Mat R;
		Mat t;
	vector<Mat> Rvect;
	vector<Mat> tvect;
	Rvect.push_back(R);
	tvect.push_back(t);
	getrandomposes(R, t, Rvect, tvect);



}
