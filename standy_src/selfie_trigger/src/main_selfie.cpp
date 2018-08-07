#include <image_transport/image_transport.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <poseimagearraymsg/poseImageArray.h>

#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/tf.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <deque>


using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace cv;

ros::Publisher pub;

int itr = 0;
ofstream myfile;
char folder[1024];
char filepath[1024];
char bebop[] = "bebop_battery_1";
tf::TransformListener* listener;

void callback(const geometry_msgs::PoseStampedConstPtr &vrpn_pose_quad,
              const geometry_msgs::PoseStampedConstPtr &vrpn_pose1,
              const sensor_msgs::ImageConstPtr &image_msg){
    cout << " Callback fired" << endl;
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    Mat img = cv_ptr->image;

    imshow("Selfie",img);
    waitKey();

	
    // Launch pose_for_dfusmc node
	// Senf green signal
	geometry_msgs::PoseStamped green;
	pub.publish(green);
    // And send bundle to cost_volume
    // Cost_volume 

}



int main(int argc, char** argv)
{
cout << "programmstarted" << endl;
  ros::init(argc, argv, "selfie_main");

  ros::NodeHandle nh;
  sprintf(folder,"/home/anurag/mother_ws/cost_volume_ws/src/costvolume_generator/dataset");
  
  tf::TransformListener sub_l;
  listener = &sub_l;
  // --- Time Synchronized Messages
  char quad_topic[1024] = ""; sprintf(quad_topic,"/vrpn_client_node/%s/pose",bebop);
  message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub_quad(nh, quad_topic, 1);
//  message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub1(nh, "/vrpn_client_node/self/pose", 1);
  message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub2(nh, "/vrpn_client_node/hand01/pose", 1);
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/bebop/image_raw", 1);

  typedef sync_policies::ApproximateTime<geometry_msgs::PoseStamped,
                     //geometry_msgs::PoseStamped,
                     geometry_msgs::PoseStamped,
                     sensor_msgs::Image > MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), vrpn_sub_quad, vrpn_sub2, image_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));
  pub = nh.advertise <geometry_msgs::PoseStamped>("/green",1);

  //Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), vrpn_sub_quad, image_sub);
  //sync.registerCallback(boost::bind(&callback, _1, _2));
  ros::spin();
  return 0;
}
