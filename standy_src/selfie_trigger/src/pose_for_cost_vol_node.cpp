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

#include <poseimagearraymsg/poseImageArray.h>

using namespace sensor_msgs;
using namespace message_filters;
using namespace cv;
using namespace std;

std::deque<Mat> dfusmc_images;
std::deque<geometry_msgs::PoseStamped> dfusmc_poses;
ros::Publisher pub;
poseimagearraymsg::poseImageArray batch;

char folder[1024];
char filepath[1024];
char bebop[] = "bebop_battery_1";
Mat K, D;
Mat t_b2cb, R_b2cb;
int counter = 0;

tf2_ros::TransformListener *listener;
tf2_ros::Buffer * tfBuffer;

void publish_dfusmc();
void from_b2co(float* msg_in , float* msg_out);
void angle_betwee_q(float* q1, float* q2, float*q );


void angle_betwee_q(float* q1, float* q2, float* ang )
{
   *ang = acos(   q1[0]*q2[0]
            -  q1[1]*q2[1]
            -  q1[2]*q2[2]
            -  q1[3]*q2[3]);
}

void from_b2co(float* msg_in , float* msg_out)
{
    bool DEBUG = 1;if(DEBUG)cout << "from_b2co" << endl;
    float r_x, r_y, r_z, r_qx, r_qy, r_qz, r_qw;
    r_x = msg_in[1];
    r_y = msg_in[2];
    r_z = msg_in[3];
    r_qw = msg_in[4];
    r_qx = msg_in[5];
    r_qy = msg_in[6];
    r_qz = msg_in[7];
if(DEBUG)cout << " 1 " << endl;
    Eigen::Matrix3d mat3 = Eigen::Quaterniond(r_qw, r_qx, r_qy, r_qz).toRotationMatrix();
    Mat R_b2w;
    cv::eigen2cv(mat3,R_b2w);
    Mat t_b2w =(Mat_<double>(3,1) << r_x,r_y,r_z);
    if(DEBUG)cout << "R_b2w : \n" << R_b2w << endl << "t_b2w : \n" << t_b2w << endl;
if(DEBUG)cout << " 2 " << endl;
    Mat R_w2cb = R_b2cb * R_b2w.t();
    Mat t_w2cb = t_b2cb + R_b2cb*(-R_b2w.t()*t_b2w);
    Mat R_cb2w = R_w2cb.inv();
    Mat t_cb2w = -R_cb2w*t_w2cb;
if(DEBUG)cout << " 3 " << endl;
    // Listening to T_co2cb
    tf::TransformListener listener;
    tf::StampedTransform tf_co2cb;
    geometry_msgs::TransformStamped tf_co2cb_gm;
    Eigen::Affine3d  T_co2cb;
if(DEBUG)cout << " 4 " << endl;
    try{
        ros::Time now = ros::Time::now();
           tf_co2cb_gm = tfBuffer->lookupTransform( "camera_base_link","camera_optical", now );
        }
    catch (tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
    }

if(DEBUG)cout << " 5 " << endl;
    //tf::transformTFToEigen(tf_co2cb,T_co2cb);
    Eigen::Matrix3d mat3_co2cb = Eigen::Quaterniond(tf_co2cb_gm.transform.rotation.w,
                                                    tf_co2cb_gm.transform.rotation.x,
                                                    tf_co2cb_gm.transform.rotation.y,
                                                    tf_co2cb_gm.transform.rotation.z ).toRotationMatrix();
   // mat3 = T_co2cb.matrix().block(0,0,3,3);
if(DEBUG)cout << " 5 " << endl;
    Mat R_co2cb;
    cv::eigen2cv(mat3_co2cb,R_co2cb);
    //Mat t_co2cb = (Mat_<double>(3,1) << (float)T_co2cb(0,3),(float)T_co2cb(1,3),(float)T_co2cb(2,3));
    Mat t_co2cb =  (Mat_<double>(3,1) << (double)tf_co2cb_gm.transform.translation.x,
                                         (double)tf_co2cb_gm.transform.translation.y,
                                         (double)tf_co2cb_gm.transform.translation.z );
if(DEBUG)cout << " 6 " << endl;
    Mat R_co2w = R_cb2w * R_co2cb ;
    Mat t_co2w = R_co2cb * t_cb2w + t_co2cb ;
    if(DEBUG)cout << "R_co2cb : \n" << R_co2cb << endl << "t_co2cb : \n" << t_co2cb << endl;
if(DEBUG)cout << " 7 " << endl;
    Eigen::Matrix3d mat3_out;
    cv::cv2eigen(R_co2w,mat3_out);
if(DEBUG)cout << " 8 " << endl;
    Eigen::Quaterniond Q_out(mat3_out);
    msg_out[0] = t_co2w.at<double>(0,0);
    msg_out[1] = t_co2w.at<double>(1,0);
    msg_out[2] = t_co2w.at<double>(2,0);
    msg_out[3] = Q_out.w();
    msg_out[4] = Q_out.x();
    msg_out[5] = Q_out.y();
    msg_out[6] = Q_out.z();

}



void callback(const geometry_msgs::PoseStampedConstPtr &vrpn_pose,const geometry_msgs::PoseStampedConstPtr &green,const sensor_msgs::ImageConstPtr &image_msg){
    bool DEBUG = 0;
    if(1)cout << "Callback fired---------- \n";

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    float msg_in[7];
    msg_in[0] = (*vrpn_pose).pose.position.x;
    msg_in[1] = (*vrpn_pose).pose.position.y;
    msg_in[2] = (*vrpn_pose).pose.position.z;
    msg_in[3] = (*vrpn_pose).pose.orientation.w;
    msg_in[4] = (*vrpn_pose).pose.orientation.x;
    msg_in[5] = (*vrpn_pose).pose.orientation.y;
    msg_in[6] = (*vrpn_pose).pose.orientation.z;

    float msg_out[7];
    from_b2co(msg_in , msg_out);
    geometry_msgs::PoseStamped cam_pose;
    cam_pose.header = (*vrpn_pose).header;
    cam_pose.pose.position.x = msg_out[0];
    cam_pose.pose.position.y = msg_out[1];
    cam_pose.pose.position.z = msg_out[2];
    cam_pose.pose.orientation.w = msg_out[3];
    cam_pose.pose.orientation.x = msg_out[4];
    cam_pose.pose.orientation.y = msg_out[5];
    cam_pose.pose.orientation.z = msg_out[6];

    if(dfusmc_poses.size()  == 0){
        dfusmc_images.push_back(cv_ptr->image);
        dfusmc_poses.push_back(cam_pose);
    }
    else
    {
        ros::Duration diff = vrpn_pose->header.stamp - dfusmc_poses.front().header.stamp;
        if (diff.sec  <  1.0)
        {
            dfusmc_images.push_back(cv_ptr->image);
            dfusmc_poses.push_back(*vrpn_pose);
        }
        else
        {
            float dist = pow(  (  pow(dfusmc_poses.back().pose.position.x - dfusmc_poses.front().pose.position.x , 2)
                                + pow(dfusmc_poses.back().pose.position.y - dfusmc_poses.front().pose.position.y , 2)
                                + pow(dfusmc_poses.back().pose.position.z - dfusmc_poses.front().pose.position.z , 2)
                               ), 0.5 );

            float q2[4] = {         dfusmc_poses.back().pose.orientation.w,
                                    dfusmc_poses.back().pose.orientation.x,
                                    dfusmc_poses.back().pose.orientation.y,
                                    dfusmc_poses.back().pose.orientation.z  };

            float q1[4] = {         dfusmc_poses.front().pose.orientation.w,
                                    dfusmc_poses.front().pose.orientation.x,
                                    dfusmc_poses.front().pose.orientation.y,
                                    dfusmc_poses.front().pose.orientation.z  };
            float ang;
            angle_betwee_q(q1, q2, &ang);

            if(DEBUG)cout << "ang : " << ang << " , dist : " << dist << endl;
            if ( (dist < 100) && (ang < 100)  )
            {
                if(DEBUG)cout << " IP IP HYRREY-----" << endl;
                publish_dfusmc();
                dfusmc_images.clear();
                dfusmc_poses.clear();
            }
            else
            {
                dfusmc_images.push_back(cv_ptr->image);
                dfusmc_poses.push_back(*vrpn_pose);

                dfusmc_images.pop_front();
                dfusmc_poses.pop_front();
                 if(DEBUG)cout << "dfusmc_poses.size()  : " << dfusmc_poses.size() << endl;
                 if(DEBUG)cout << "dfusmc_poses.Time   : " << dfusmc_poses.front().header.stamp.sec << "  " << dfusmc_poses.back().header.stamp.sec << endl;
            }
        }
    }
}


void publish_dfusmc()//
{
    bool DEBUG = 0;
    counter++;

    if(1)cout << "Publishing" << endl;
    sprintf(folder,"/home/arindam/mother_ws/anurag_ws/dfusmc_ws/dfusmc_node_Dataset");
    sprintf(filepath,"%s/for_dfusmc_%d.avi",folder,counter);

    //VideoWriter outputVideo;Size frameSize(dfusmc_images[0].size());
    //outputVideo.open( filepath, CV_FOURCC('M','P','E','G'), 30 , frameSize, true);
    //if (!outputVideo.isOpened())
    //{
    //    cout  << "Could not open the output video for write: \n" << filepath << endl;
    //    return ;
    //}

    batch.imageArray.clear();
    batch.poseArray.clear();
    for(int i = 0; i < dfusmc_images.size() ; i++)
    {
        //outputVideo << dfusmc_images[i];

        cv_bridge::CvImage out_msg;
        out_msg.image = dfusmc_images[i];
        out_msg.header.stamp = ros::Time::now();
        out_msg.header.frame_id = "/asd";
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;

        batch.imageArray.push_back(*(out_msg.toImageMsg()));
        batch.poseArray.push_back( dfusmc_poses[i] );

    }
    pub.publish(batch);
    //outputVideo.release();
}


int main(int argc, char** argv)
{
    bool DEBUG = 1;
    sprintf(folder,"/home/anurag/mother_ws/cost_volume_ws/src/costvolume_generator/dataset");  
    sprintf(filepath,"%s/RT_b2cb_%s.yml",folder,bebop);
    FileStorage fs_e(filepath, FileStorage::READ);
    fs_e["R_b2cb"] >> R_b2cb;
    fs_e["T_b2cb"] >> t_b2cb;
    if(DEBUG) cout << "R_b2cb" << endl << R_b2cb << endl << "t_b2cb" << t_b2cb << endl;



  ros::init(argc, argv, "pose_for_dfusmc_node");

  ros::NodeHandle nh;

      tf2_ros::Buffer tempBuffer;
      tf2_ros::TransformListener tfListener(tempBuffer);
      tfBuffer = &tempBuffer;
      listener = &tfListener;

  // --- Time Synchronized Messages
  char quad_topic[1024] = ""; sprintf(quad_topic,"/vrpn_client_node/%s/pose",bebop);
  message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub(nh, quad_topic, 1);
  message_filters::Subscriber<geometry_msgs::PoseStamped> vrpn_sub2(nh, "/green", 1);
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/bebop/image_raw", 1);
  typedef sync_policies::ApproximateTime<geometry_msgs::PoseStamped,geometry_msgs::PoseStamped,sensor_msgs::Image > MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), vrpn_sub,vrpn_sub2, image_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));
  pub = nh.advertise <poseimagearraymsg::poseImageArray>("/poseimagebundle",1);

  ros::spin();
  return 0;
}


