#ifndef _gpucost__CLASS_H_
#define _gpucost__CLASS_H_

#include "headers.h"

//class GFilter : public BFilter
class gpucost_
{
	public:

		gpucost_();
		~gpucost_();


		float *dCost;
		float *dIr;
		float *dCounter;
		float *dDepth;
		float *dMinm;
		float *dest;
		float *dP;
		float *dB;
		float *dConf;
		float *d_texture;
		size_t pitch;


		int tPx;
		int height;
		int width;
		int nDepth;
		int order;
		int imgSz, volSz;
		bool firstTime;

		cv::Mat R_ref;cv::Mat R_nei;
		cv::Mat T_ref;cv::Mat T_nei;
		cv::Mat P_ref;cv::Mat P_nei;
		cv::Mat P_ref_inv, P_nei_inv;

		dim3 blockSize, gridSize;



		void loadRefImg(cv::Mat img, geometry_msgs::PoseStamped msg_ref,cv::Mat K);
		void pushimg(cv::Mat Im, geometry_msgs::PoseStamped msg,cv::Mat K);
		void convertPose2T(geometry_msgs::PoseStamped msg, cv::Mat &P_ref);
		void downloadDepthmap(cv::Mat &depthMap, cv::Mat &minM);
		void reset();
		void initialize_variables();
		void denoise(cv::Mat &refinedResult, cv::Mat refImg, cv::Mat result);

};







#endif //GUIDEDFILTER_GFILTER_H
