#pragma once
#include "laserPro.h"
#include <math.h>
#include <numeric>
#include "opencv.hpp"
#include <map>
#include <string>
#include <vector>

using namespace  std;

class LaserPro
{
public:
	LaserPro(void);
	~LaserPro(void);

	
	//-----------------------------------------------------------------
	// img 模块
	//static int img_getLaser(lua_State * L);											// 获取整副图像的激光位置
	//static int img_findPt(lua_State * L);												// 获取指定列的激光位置
	//static int img_drawLaser(lua_State *L);
	//static int img_combineLaser(lua_State *L);									//激光线条拼接

	//static int laser_getPt(lua_State * L);	
 //   static int laser_sub(lua_State *);                                                   //轮廓相减
 //   static int laser_add(lua_State *L);                                                 //轮廓相加
 //   static int laser_copy(lua_State *L);                                                //轮廓复制
	/*
	 功能：轮廓平滑
	 weights:size()必须等于奇数
	 size =1 时,
	*/
	static int laser_smooth(const vector<int> &vecPtsIn, vector<int> &vecPtsOut,vector<double> weithts,int times=1);
	static int laser_drawAsHist(const vector<int> &vecPtsIn, cv::Mat &histImg, bool down = true);
	static int laser_drawAsHist(const vector<int> &vecPtsIn, cv::Mat &histImg, vector<int> &vecNormPtsOut,bool down = true);
	static int laser_drawAsPoint(const vector<int> &vecPtsIn, cv::Mat &histImg,bool down = true);

	//获取局部极值点
	static int laser_getLocalPts(const vector<int> &vecPts,vector<int> &locs, int heightLowThresh);						//获取激光轮廓局部极值和位置
	static int laser_getGeneralMinPts(const vector<int> &vecPts, vector<int> &locs, int heightLowThresh);

	static bool polynomialCurveFit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);


	///////////////////////////////////////////////////////////////////////////////////////
	// skeleton 表示骨架图像处理

	//绘制点集
	static int img_drawPoint(cv::Mat& imgInPutOutput, vector<cv::Point> ptVec, cv::Scalar color,
			cv::MarkerTypes mt, int markerSize =1, int thickness=1);

	// 获取骨架（细化后）的点集, 
	// inver : 默认左上端点 - > 右下端点
	static int skeleton_getAllPt(cv::Mat biImg, vector<cv::Point>& cont,int inver);

	// 获取骨架线条所有末端点
	static int skeleton_getEndPt(const cv::Mat& skeletonImg, vector<cv::Point>& endPoints);

	// 获取骨架线条末端点，每个区域单独存放
	static int skeleton_getEndPt(cv::Mat biImg, vector <vector<cv::Point> > & ptPairVec);

	// 得到线条交点
	static int skeleton_getIntersecPt(cv::Mat biImg, vector < pair<cv::Point, cv::Point>> & ptPairVec);

	// 去除分支，保留线条主体
	static int skeleton_removeBranch(cv::Mat biImg, cv::Mat reBinImg);

	// 交叉线条切分， 按长度降序输出，无二次连接
	static int skeleton_split(cv::Mat biImg, vector<cv::Mat>& imgVec);

	// 交叉线条切分，再依据方向，长度合并线条， 按长度降序输出
	static int skeleton_merge(cv::Mat biImg, vector<cv::Mat>& imgVec);


	//static int laser_fromImage(lua_State *L);									//二维图像外边缘转换为轮廓
	//static int laser_getAverHeight(lua_State *L);								//获取平均像素位置高度（平均row值）
 //   static int laser_getPolyline(lua_State *L);                                   //获取轮廓的逼近多段线
 //   static int laser_filterByLen(lua_State *L);                                    //通过子轮廓长度筛选有效轮廓
 //   static int laser_getEndpoint(lua_State *L);                                 //获取轮廓中连续轮廓的端点
	//-----------------------------------------------------------------
	// 测量相关


    
    //多线程模块，对deque实现的操作
    //static int shared_pushMat2Que(lua_State *L);            //写入采集到的图片到队列
    //static int shared_getMatAndLaser(lua_State *L);         //从队列获取图片并计算激光位置到队列
    //static int shared_getLaserAndSendTable(lua_State *L);   
    //static int shared_getLaserAndSendString(lua_State *L);

    



};
