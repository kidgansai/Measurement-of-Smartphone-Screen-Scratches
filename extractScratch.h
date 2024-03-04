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

//��������Ŀ����
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

//�ֿ��໥���ӵĻ���,����Ǽ�ͼ
vector<cv::Mat> scratch_split(cv::Mat skeletonImgIn, double angleMergeThresh);
int scratch_connection(cv::Mat binImgIn, cv::Mat &connImg, vector<pair<cv::Point, cv::Point>> &vecPairPt);
int scratch_connection2(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> &vecPairPt, int angleFrmEndPtLen= 13,double distThreshLow = 15,double distThreshHighHigh=30);
vector<cv::Mat> scratch_merge(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> vecPtPair, bool outputAllOrder = false);


//��ȡ��ת���εĳ��ߵ�ԺͶ̱ߵ�ԣ��������㷽λ��һ��
pair<pair<cv::Point, cv::Point>, pair<cv::Point, cv::Point>> getRotRectPairPt(cv::RotatedRect rotRect,bool longEdge);

int getHalfRotRect( cv::RotatedRect bigRotRect,cv::RotatedRect rotCentRect, cv::RotatedRect &halfRotRect1,int chooseID);

//cnt���ؽ��Ĵ���
void morRebuild(cv::Mat src, cv::Mat toMor, cv::Mat &outPut,int cnt=-1);

vector<cv::Point> uniqueVecPts(vector<cv::Point> vecPts);
bool ptInVec(cv::Point pt, vector<cv::Point> vecPts);
cv::Mat removeIntersection(cv::Mat srcBin, double  &scratchLen);
int skeleton_intersecPoint(cv::Mat skeletonImg, vector<cv::Point> &intersecPts);
//��ĳ�㿪ʼ����ͼ������vec����һ��ֹͣ��Ȼ������õ�
//�Ǽ�ͼ������̬ѧ�ؽ��ķ�������,255,0
cv::Point searchPt(cv::Mat binSrc, cv::Point startPt, vector<cv::Point> pts);
double anglePostive(cv::Point pt1, cv::Point pt2);
double angle2PtNorm(cv::Point lineStartPt, cv::Point lineEndPt);
double angle(cv::Point pt1, cv::Point pt2, cv::Point ptStart);

//���عǼܹ켣׷��ͼ
cv::Mat getSkeletonFromPt(cv::Point pt1, cv::Point pt2, cv::Mat skeletonImg);

cv::Mat connectSkeletonImg(cv::Mat skeletonImg,double distThresh=4);

//��û����ܱ��������ڼ���Ҷ�
cv::Mat getScratchAround(cv::Mat srcSratch, cv::Mat roundScratch);

double angleFromEndPt(cv::Mat binImg, cv::Point endPt, cv::Point &endPtNew, int cntThresh = 7);
int skeleton_removeIntersectionLine(cv::Mat skeleImgIn, cv::Mat &binOut);

cv::Mat  skeleton_removeBranchs(cv::Mat skeleImgIn, double distThresh, double endAngleThresh);//��ȱ�ݣ�����ϳ�ʱ����

cv::Mat  skeleton_removeBranchsPlus(cv::Mat skeleImgIn, double distThresh, double endAngleThresh);
//cv::Mat getRawScratchRound(cv::Mat scratchRaw);
cv::Mat getRawScratchRound(cv::Mat scratchRaw, int nearDist1 = 3, int nearDist2 = 5);

//�������м��С��distThresh�ĵ� ����С�߶�Ϊ������Ԫ���л��ۺϲ�
int biImg_connection(cv::Mat binImgIn, cv::Mat & connectedImg, double distThresh);

//�������򣬾���С�����ӣ�������ʱ����Ҫ��һ����������������ϴ������
int biImg_connection(cv::Mat binImgIn, cv::Mat & connectedImg, double distThreshLow,double distThreshHigh,double areaThreshLow);

/*
labelImg:���ͼ��
mapVecPts:�����Ǻͱ�Ƕ�Ӧ�ĵ㼯
excludeLabel���������ı��
includeZero:�Ƿ���������
*/
int labelImg2vectorPts(cv::Mat labelImg, map<int, vector<cv::Point>> &mapVecPts, int excludeLabel, bool includeZero=false);

pair<double, cv::Point> getMinDistInPtAndRegion(cv::Point pt, vector<cv::Point> vecPts);
pair<int, pair<cv::Point, cv::Point>> distInPtsAndMap(vector<cv::Point> vecPts, map<int, vector<cv::Point>> mapPts);

//���ܣ�ɾ�������ķֲ棬
//branchPts:�ֲ������˵㣬����һ���������ߵĵ�
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
