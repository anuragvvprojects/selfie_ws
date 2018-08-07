#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <deque>

#include <poseimagearraymsg/poseImageArray.h>


using namespace std;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_publisher");
  ros::NodeHandle nh;
}
