//
// Created by smher on 17-9-29.
//

#ifndef _HEADERS_H_
#define _HEADERS_H_

#include <iostream>
#include <vector>
#include "ctime"
#include <cassert>
#include "stdexcept"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/edge_filter.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <image_transport/image_transport.h>
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

using namespace std;
using namespace cv;

#define BLOCKSIZE 16

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)

inline void __cudaCheckError(cudaError_t err, const char *file, int line)
{
    if(err != cudaSuccess)
    {
        cout << err << " in " << file << " at " << line << " line." << endl;
    }
}

#endif //GUIDEDFILTER_HEADERS_H
