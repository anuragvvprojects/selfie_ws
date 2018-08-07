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
//#include<random>
#include<cmath>
#include<ctime>

using namespace std;
using namespace cv;
void getNumber(double &v, double min, double max);
void getVector(vector<double> &v, vector<double> min, vector<double> max);
void getrandomposes(Mat& R, Mat &t, vector<Mat>& Rvect, vector<Mat>& tvect)
{

	ofstream fp1,fp2,fp3;
	fp1.open("workable_points.dat");
	fp2.open("all_points.dat");
	fp3.open("actual_point.dat");
	srand( time( NULL ) );
	int N = 100; //Number of random poses.
/*
	vector<double> v(6,0.0),min(6,0.0),max(6,0.1);
	for(int i=0;i<N;i++){
		getVector(v,min,max);
		for(int j=0;j<v.size();j++)
			cout<<v[j]<<" ";
		cout<<endl;
	}
*/

	double v = 0.0, theta, phi, x, y, z, m = 0.5;
	for (int i = 0; i < N; i++) 
	{	// incorrect way
		getNumber(v,-m,m);
		theta = v;
		getNumber(v,-m,m);
		phi = v;
		x = sin(phi) * cos(theta);
		y = sin(phi) * sin(theta);
		z = cos(phi);
		fp1 << x  << " "<< y << " "<< z << endl; 
		getNumber(v,M_PI/2-m,M_PI/2+m);
		theta = v;
		getNumber(v,-m,m);
		phi = v;
		x = sin(phi) * cos(theta);
		y = sin(phi) * sin(theta);
		z = cos(phi);
		fp1 << x  << " "<< y << " "<< z << endl; 
	}
	N = 1000;
	for (int i = 0; i < N; i++) 
	{	// incorrect way
		getNumber(v,0,2*M_PI);
		theta = v;
		getNumber(v,0,M_PI);
		phi = v;
		x = sin(phi) * cos(theta);
		y = sin(phi) * sin(theta);
		z = cos(phi);
		fp2 << x  << " "<< y << " "<< z << endl; 
	}
	theta = 0;
	phi = 0;
		x = sin(phi) * cos(theta);
		y = sin(phi) * sin(theta);
		z = cos(phi);
	fp3<< x  << " "<< y << " "<< z << endl; 
	
	fp1.close();
	fp2.close();
	fp3.close();


	// Generate random R
		// Generate very small random rotation in any form (axis-angle or eular, axis-angle would be better i guess)
		// Get R_small from the above
		// Multiply with R_ref

}

void getVector(vector<double> &v, vector<double> min, vector<double> max){
	for(int i=0;i<v.size();i++){
		double v1 = abs((double)rand()/ (RAND_MAX+1));
		v[i] = (v1) * (max[i]-min[i]) + min[i];
//		cout<<v1<<" "<<min[i]<<" "<<max[i]<<" ";
	}
//	cout<<endl;
}

void getNumber(double &v, double min, double max){
	double v1 = abs((double)rand()/ (RAND_MAX+1));
	v = (v1) * (max-min) + min;
}
