#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "imgPro.h"
#include "laserPro.h"
//#include <opencv2/ximgproc.hpp>
//#include "D:\\2_MyProjects\\SnapBox\\hessianDemo\\hessianMatrix.h"
using namespace std;
using namespace cv;

cv::Mat getDoG(const cv::Mat &img);

//背景暗，目标亮
int hessianEnhance(cv::Mat srcImage, cv::Mat &imOut);
int grayImg_hessian(const cv::Mat& img_in, cv::Mat& img_out, int window, float sigma);

cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);
void filterOver(cv::Mat thinSrc);
std::vector<cv::Point> getPoints(const cv::Mat &thinSrc, unsigned int raudis = 4,
	unsigned int thresholdMax = 6, unsigned int thresholdMin = 4);

int skeleton_endpoint(const cv::Mat &skeletonImg, vector<pair<cv::Point, cv::Point>> &endPoints);
int skeleton_endpoint(const cv::Mat &skeletonImg, vector<cv::Point> &endPoints);

int skeleton_endpoint2(cv::Mat skeletonImg, vector<cv::Point> &endPoints);
cv::Mat searchPtImg(cv::Mat binSrc, cv::Point startPt, vector<cv::Point> pts);
std::pair<cv::Point, cv::Point> getLinePtInImg(cv::Mat imgIn, cv::Vec4f linePara, cv::Point newStartPt = cv::Point(-1, -1));
cv::Mat scratch_extend(cv::Mat scratchImag, cv::Mat binRefImg, cv::Rect scratchRect);

//分开相互连接的划痕,输入骨架图
vector<cv::Mat> scratch_split(cv::Mat skeletonImgIn, double angleMergeThresh);
int scratch_connection(cv::Mat binImgIn, cv::Mat &connImg, vector<pair<cv::Point, cv::Point>> &vecPairPt);
int scratch_connection2(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> &vecPairPt, int angleFrmEndPtLen= 13,double distThreshLow = 15,double distThreshHighHigh=30);
vector<cv::Mat> scratch_merge(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> vecPtPair, bool outputAllOrder = false);


//获取旋转矩形的长边点对和短边点对，方向和起点方位均一致
pair<pair<cv::Point, cv::Point>, pair<cv::Point, cv::Point>> getRotRectPairPt(cv::RotatedRect rotRect,bool longEdge);

int getHalfRotRect( cv::RotatedRect bigRotRect,cv::RotatedRect rotCentRect, cv::RotatedRect &halfRotRect1,int chooseID);

//cnt：重建的次数
void morRebuild(cv::Mat src, cv::Mat toMor, cv::Mat &outPut,int cnt=-1);

vector<cv::Point> uniqueVecPts(vector<cv::Point> vecPts);
bool ptInVec(cv::Point pt, vector<cv::Point> vecPts);
cv::Mat removeIntersection(cv::Mat srcBin, double  &scratchLen);
int skeleton_intersecPoint(cv::Mat skeletonImg, vector<cv::Point> &intersecPts);
//由某点开始遍历图像，遇到vec中任一点停止，然后输出该点
//骨架图像，用形态学重建的方法查找,255,0
cv::Point searchPt(cv::Mat binSrc, cv::Point startPt, vector<cv::Point> pts);
double anglePostive(cv::Point pt1, cv::Point pt2);
double angle2PtNorm(cv::Point lineStartPt, cv::Point lineEndPt);
double angle(cv::Point pt1, cv::Point pt2, cv::Point ptStart);

//返回骨架轨迹追踪图
cv::Mat getSkeletonFromPt(cv::Point pt1, cv::Point pt2, cv::Mat skeletonImg);

cv::Mat connectSkeletonImg(cv::Mat skeletonImg,double distThresh=4);

//获得划痕周边区域，用于计算灰度
cv::Mat getScratchAround(cv::Mat srcSratch, cv::Mat roundScratch);

double angleFromEndPt(cv::Mat binImg, cv::Point endPt, cv::Point &endPtNew, int cntThresh = 7);
int skeleton_removeIntersectionLine(cv::Mat skeleImgIn, cv::Mat &binOut);

cv::Mat  skeleton_removeBranchs(cv::Mat skeleImgIn, double distThresh, double endAngleThresh);//有缺陷，主体较长时适用

cv::Mat  skeleton_removeBranchsPlus(cv::Mat skeleImgIn, double distThresh, double endAngleThresh);
//cv::Mat getRawScratchRound(cv::Mat scratchRaw);
cv::Mat getRawScratchRound(cv::Mat scratchRaw, int nearDist1 = 3, int nearDist2 = 5);

//连接所有间隔小于distThresh的点 ，以小线段为基本单元进行划痕合并
int biImg_connection(cv::Mat binImgIn, cv::Mat & connectedImg, double distThresh);

//连接区域，距离小均连接，距离大的时候需要其一个待连接满足面积较大的条件
int biImg_connection(cv::Mat binImgIn, cv::Mat & connectedImg, double distThreshLow,double distThreshHigh,double areaThreshLow);

/*
labelImg:标记图像
mapVecPts:输出标记和标记对应的点集
excludeLabel：不包含的标记
includeZero:是否包括标记零
*/
int labelImg2vectorPts(cv::Mat labelImg, map<int, vector<cv::Point>> &mapVecPts, int excludeLabel, bool includeZero=false);

pair<double, cv::Point> getMinDistInPtAndRegion(cv::Point pt, vector<cv::Point> vecPts);
pair<int, pair<cv::Point, cv::Point>> distInPtsAndMap(vector<cv::Point> vecPts, map<int, vector<cv::Point>> mapPts);

//功能：删除给定的分叉，
//branchPts:分叉两个端点，其中一个是与主线的点
cv::Mat skeleton_removeBranch(cv::Mat skeletonImgIn, vector<pair<cv::Point, cv::Point>> branchPts);
cv::Mat set8Neib(cv::Mat binIn, cv::Point pt);

//snapbox

double getDist2Pts(cv::Vec4f lineSeg);
cv::Point getFarEndPt(cv::Vec4f lineSeg, cv::Point pt);
cv::Point getNearEndPt(cv::Vec4f lineSeg, cv::Point pt);

//
cv::Mat phoneInvalidAreaL1(cv::Mat grayIn, int type);
cv::Mat phoneValidAreaL2(cv::Mat grayIn, cv::Mat &L1InvalidMaskImg);
cv::Mat getHessianFlashDot(cv::Mat grayInImg);
cv::Mat removeScreenEdge(cv::Mat binIn, cv::Mat &edgeImg);
cv::Mat removeScreenEdgeLineSeg(cv::Mat &srcImg, cv::Mat &validMask, cv::Mat lineLabelImg, vector<cv::Vec4f>& lineSegs);
cv::Mat removeSpot(cv::Mat binImg, cv::Mat &dotImg);
cv::Mat camScreenEdge(cv::Mat &srcImg, cv::Mat validMask, cv::Mat &edgeAboveCamiImg);
cv::Mat screenCornerEdge(cv::Mat &srcImg, cv::Mat validMask, cv::Rect roi);
bool screenBorderIsWhite(cv::Mat &srcImg);
cv::Mat removeLineSeg(cv::Mat lineLableImg, vector<cv::Vec4f>& lineSegs);

double getHessianSigma(cv::Mat & src);
cv::Mat nearInvalidRegion(cv::Mat &invalidMask, int dSize, int eSize, bool isCorner);

vector<cv::Mat> snapBox_splitScratch(cv::Mat &skeletonImgIn, double angleThreshVal);
cv::Mat  biImg_fillLineCrossGap(cv::Mat binImgIn);

vector<cv::Rect> rect_merge(cv::Size imgSize, vector<cv::Rect> vecRect, int extLen,int maxLenThresh);

int removeHorLine(cv::Mat &srcImg, cv::Mat &outImg);
cv::RotatedRect rotRectFromSeg(vector<cv::Vec4f> &vecSeg);
double getSigmaL1(cv::Mat &src);
double getSigmaL2(cv::Mat raw, cv::Mat &validMaskImg);
cv::Rect removeScreenEdgeLineSegL2(cv::Mat &srcImg, cv::Mat &validMask, cv::Mat lineLabelImg, vector<cv::Vec4f>& lineSegs);

int removeHorLineSeg(cv::Mat &src, cv::Mat &hessianImg, cv::Mat &lineLabelImg, vector<cv::Vec4f> &lineSegs);
cv::Rect getShiftRectL1(cv::Rect roiRect, int label, double rotAngle, map<string, vector<double>> &mapFeats,
	map<int, int>&mapLabelIndex, cv::Point centPt);
cv::Rect getShiftRectL2(cv::Rect roiRect, int label, double rotAngle, map<string, vector<double>> &mapFeats,
	map<int, int>&mapLabelIndex, cv::Point centPt);
int mergeRect(vector<cv::Rect> vecLightRect, vector<cv::Rect>& vecCheckRect);
cv::Mat removeSpotL2(cv::Mat binImg, cv::Mat &dotImg);
