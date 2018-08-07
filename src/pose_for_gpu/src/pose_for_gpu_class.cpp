/*#include <image_transport/image_transport.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write

#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/tf.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <deque>

#include "gpucost_class.h"
*/

#include "headers.h"
#include "ros_msgs.hpp"
#include "dataset_functions.hpp"

using namespace sensor_msgs;
using namespace message_filters;
using namespace cv;
using namespace std;

std::deque<Mat> dfusmc_images;
std::deque<geometry_msgs::PoseStamped> dfusmc_poses;
ros::Publisher pub;


char folder[1024];
char filepath[1024];
char bebop[] = "ardrone_hull";
Mat K, D;
Mat t_b2cb, R_b2cb;
int counter = 0;
bool buildUpon = true;
bool firstImage = true;

tf2_ros::TransformListener *listener;
tf2_ros::Buffer * tfBuffer;
ros::Time startTime, nowTime;

gpucost_ *GPC = new gpucost_();

void from_b2co(float* msg_in , float* msg_out);


void generate_pose(geometry_msgs::PoseStamped msg, geometry_msgs::PoseStamped &actual_pose);

void generate_pose(geometry_msgs::PoseStamped msg, geometry_msgs::PoseStamped &actual_pose){
    bool DEBUG = 1;
    Mat R_b2w,t_b2w;

    PoseStamped2matRT(msg,R_b2w,t_b2w);
    R_b2w.convertTo(R_b2w, CV_64F);
    t_b2w.convertTo(t_b2w, CV_64F);

    //cout<<R_b2cb.type()<<" "<<R_b2w.type()<<endl;
    Mat R_w2cb = R_b2cb * R_b2w.t();if(DEBUG)cout << " 4 " << endl;
    Mat t_w2cb = t_b2cb + R_b2cb*(-R_b2w.t()*t_b2w);if(DEBUG)cout << " 5 " << endl;if(DEBUG)cout << " -17- " << endl;

    //---------------------------------------

    /// Obtaining tf cd2co; Only for bebop; Not applicable to ardrone;
        //Eigen::Affine3d T_cb2co;
            // geometry_msgs::TransformStamped tf_cb2co;

         //    try
         //    {
         //        ros::Time now = ros::Time::now();
            // tf_cb2co = tfBuffer->lookupTransform( "camera_optical","camera_base_link", ros::Time(0) );
            // //cout<<tf_cb2co<<endl;
         //    }
         //    catch (tf2::TransformException ex){
         //    	cout << " failure" << endl;
         //    	cout << " No luck YET" << endl;
         //    	ROS_ERROR("%s",ex.what());
         //    	return;
            // }

    Mat R_cb2co = Mat::eye(3,3,R_w2cb.type());
    Mat t_cb2co = Mat::zeros(3,1,t_w2cb.type());
    Mat R_w2co = R_cb2co * R_w2cb;
    Mat t_w2co = t_cb2co + R_cb2co*t_w2cb;

    R_w2co = R_cb2co * R_b2cb * R_b2w.t();
    t_w2cb = t_b2cb + R_b2cb*(-R_b2w.t()*t_b2w);
    t_w2co = t_cb2co + R_cb2co*(t_w2cb);
    //---------------------------------------


    matRT2PoseStamped(R_w2co,t_w2co,actual_pose);
    actual_pose.header.stamp = msg.header.stamp;

}

void from_b2co(float* msg_in , float* msg_out)
{
    bool DEBUG = 0;if(DEBUG)cout << "from_b2co" << endl;

    Mat R_b2w, t_b2w;
    floatarray2matRT(msg_in,R_b2w,t_b2w);
    R_b2w.convertTo(R_b2w, CV_64F);
    t_b2w.convertTo(t_b2w, CV_64F);

    if(DEBUG)cout << "R_b2w : \n" << R_b2w << endl << "t_b2w : \n" << t_b2w << endl;
    if(DEBUG)cout << " 2 " << endl;
    Mat R_w2cb = R_b2cb * R_b2w.t();
    Mat t_w2cb = t_b2cb + R_b2cb*(-R_b2w.t()*t_b2w);
    Mat R_cb2w = R_w2cb.inv();
    Mat t_cb2w = -R_cb2w*t_w2cb;
    /// Obtaining tf cd2co; Only for bebop; Not applicable to ardrone;
        //    if(DEBUG)cout << " 3 " << endl;
        //    // Listening to T_co2cb
        //    tf::TransformListener listener;
        //    tf::StampedTransform tf_co2cb;
        //    geometry_msgs::TransformStamped tf_co2cb_gm;
        //    Eigen::Affine3d  T_co2cb;
        //    if(DEBUG)cout << " 4 " << endl;
        //    try{
        //        ros::Time now = ros::Time::now();
        //           tf_co2cb_gm = tfBuffer->lookupTransform( "camera_base_link","camera_optical", now );
        //        }
        //    catch (tf::TransformException ex){
        // //        ROS_ERROR("%s",ex.what());
        //    int a = 1;
        //    }

        // if(DEBUG)cout << " 5 " << endl;
        //    //tf::transformTFToEigen(tf_co2cb,T_co2cb);
        //    Eigen::Matrix3d mat3_co2cb = Eigen::Quaterniond(tf_co2cb_gm.transform.rotation.w,
        //                                                    tf_co2cb_gm.transform.rotation.x,
        //                                                    tf_co2cb_gm.transform.rotation.y,
        //                                                    tf_co2cb_gm.transform.rotation.z ).toRotationMatrix();
        //   // mat3 = T_co2cb.matrix().block(0,0,3,3);
        // if(DEBUG)cout << " 5 " << endl;
        //    Mat R_co2cb;
        //    cv::eigen2cv(mat3_co2cb,R_co2cb);
        //    //Mat t_co2cb = (Mat_<double>(3,1) << (float)T_co2cb(0,3),(float)T_co2cb(1,3),(float)T_co2cb(2,3));
        //    Mat t_co2cb =  (Mat_<double>(3,1) << (double)tf_co2cb_gm.transform.translation.x,
        //                                         (double)tf_co2cb_gm.transform.translation.y,
        //                                         (double)tf_co2cb_gm.transform.translation.z );


    Mat R_cb2co = Mat::eye(3,3,R_w2cb.type());
    Mat t_cb2co = Mat::zeros(3,1,t_w2cb.type());

    if(DEBUG)cout << " 6 " << endl;
    Mat R_co2w = R_cb2w * R_cb2co ;
    Mat t_co2w = R_cb2co * t_cb2w + t_cb2co ;

    matRT2floatarray(R_co2w,t_co2w,msg_out);
}



void callback(const geometry_msgs::PoseStampedConstPtr &vrpn_pose,const sensor_msgs::ImageConstPtr &image_msg){
    bool DEBUG = 1;
    //if(1)cout << "Callback fired---------- \n"; 

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    float msg_in[7];
    geometry_msgs::PoseStamped opti_pose = *vrpn_pose;
    PoseStamped2floatarray(opti_pose,msg_in);



    float msg_out[7];
    from_b2co(msg_in , msg_out);

	geometry_msgs::PoseStamped actual_pose;
	generate_pose(*vrpn_pose, actual_pose);


    geometry_msgs::PoseStamped cam_pose = actual_pose;
    cam_pose.header = (*vrpn_pose).header;
	
	cv::Mat greyImage;
	cvtColor(cv_ptr->image,greyImage,COLOR_BGR2GRAY);
	greyImage.convertTo(greyImage,CV_32F);

	if(buildUpon){
		if(firstImage){
			GPC->loadRefImg(greyImage, cam_pose, K);
													if(DEBUG)cout << " Reference Image " << endl;
			startTime = vrpn_pose->header.stamp;
			firstImage = false;
		}
		else{
			nowTime = vrpn_pose->header.stamp;
													//if(DEBUG)cout << " Neighborhood Image " << endl;
			GPC->pushimg(greyImage, cam_pose, K);
			ros::Duration diff = (nowTime-startTime);
			if(diff.sec > 3.0)
				buildUpon = false;
		}
	}
	else{
		Mat depthMap = Mat::zeros(greyImage.size(),greyImage.type()), minM = depthMap.clone();
		GPC->downloadDepthmap(depthMap, minM);
		GPC->reset();

													if(DEBUG)cout << " Doing reset " << endl;
		normalize(minM,depthMap,0.0, 1.0, NORM_MINMAX);
		imshow("initial_depthmap",depthMap);
		waitKey(30);
		firstImage = true;
		buildUpon = true;
	}

}


int main(int argc, char** argv)
{
	bool DEBUG = 0;
    sprintf(folder,"/home/anurag/mother_ws/ardrone_calibration_ws/src/solve_pnp_calib/dataset");
//    sprintf(filepath,"%s/RT_b2co_ardrone_hull.yml",folder);
//    FileStorage fs_e(filepath, FileStorage::READ);
//    fs_e["R_b2co"] >> R_b2cb;
//    fs_e["T_b2co"] >> t_b2cb;

    getmarkerRT("ardrone_hull",R_b2cb,t_b2cb);

/**/
    getcameraKD("ardrone_hull",K,D);
    K.convertTo(K, CV_32F);
    D.convertTo(D, CV_32F);
    //if(1) cout << "R_b2cb" << endl << R_b2cb << endl << "t_b2cb" << t_b2cb << endl;

	ros::init(argc, argv, "pose_for_dfusmc_node");

	ros::NodeHandle nh;

	tf2_ros::Buffer tempBuffer;
	tf2_ros::TransformListener tfListener(tempBuffer);
	tfBuffer = &tempBuffer;
	listener = &tfListener;

	// --- Time Synchronized Messages
	char quad_topic[1024] = ""; sprintf(quad_topic,"/vrpn_client_node/%s/pose",bebop);
	message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub(nh, quad_topic, 1);
	//message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub2(nh, "/green", 1);
	message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/ardrone/image_raw", 1);
	typedef sync_policies::ApproximateTime<geometry_msgs::PoseStamped,sensor_msgs::Image > MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), vrpn_sub,image_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2));

	pub = nh.advertise< sensor_msgs::Image > ("/initial_depthmap_optitrack", 1);
	ros::spin();
	return 0;
}


