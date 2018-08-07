#ifndef _GPUCOST_CLASS_H_
#define _GPUCOST_CLASS_H_

#include "headers.h"

//class GFilter : public BFilter
class gpucost
{
public:

    //GFilter() = default;
    gpucost();
    ~gpucost();
        
    int ndepth;
    cv::Mat Ir;
    float *dCost;
    float *dIr;
    float *dIm;
    float *dCounter;

    int tPx;
    int height;
    int width;
    int nDepth;
    void loadRefImg(cv::Mat img);
    void pushimg(cv::Mat Im);

};





#endif //GUIDEDFILTER_GFILTER_H
