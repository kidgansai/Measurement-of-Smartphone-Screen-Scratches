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
	// img ģ��
	//static int img_getLaser(lua_State * L);											// ��ȡ����ͼ��ļ���λ��
	//static int img_findPt(lua_State * L);												// ��ȡָ���еļ���λ��
	//static int img_drawLaser(lua_State *L);
	//static int img_combineLaser(lua_State *L);									//��������ƴ��

	//static int laser_getPt(lua_State * L);	
 //   static int laser_sub(lua_State *);                                                   //�������
 //   static int laser_add(lua_State *L);                                                 //�������
 //   static int laser_copy(lua_State *L);                                                //��������
	/*
	 ���ܣ�����ƽ��
	 weights:size()�����������
	 size =1 ʱ,
	*/
	static int laser_smooth(const vector<int> &vecPtsIn, vector<int> &vecPtsOut,vector<double> weithts,int times=1);
	static int laser_drawAsHist(const vector<int> &vecPtsIn, cv::Mat &histImg, bool down = true);
	static int laser_drawAsHist(const vector<int> &vecPtsIn, cv::Mat &histImg, vector<int> &vecNormPtsOut,bool down = true);
	static int laser_drawAsPoint(const vector<int> &vecPtsIn, cv::Mat &histImg,bool down = true);

	//��ȡ�ֲ���ֵ��
	static int laser_getLocalPts(const vector<int> &vecPts,vector<int> &locs, int heightLowThresh);						//��ȡ���������ֲ���ֵ��λ��
	static int laser_getGeneralMinPts(const vector<int> &vecPts, vector<int> &locs, int heightLowThresh);

	static bool polynomialCurveFit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);


	///////////////////////////////////////////////////////////////////////////////////////
	// skeleton ��ʾ�Ǽ�ͼ����

	//���Ƶ㼯
	static int img_drawPoint(cv::Mat& imgInPutOutput, vector<cv::Point> ptVec, cv::Scalar color,
			cv::MarkerTypes mt, int markerSize =1, int thickness=1);

	// ��ȡ�Ǽܣ�ϸ���󣩵ĵ㼯, 
	// inver : Ĭ�����϶˵� - > ���¶˵�
	static int skeleton_getAllPt(cv::Mat biImg, vector<cv::Point>& cont,int inver);

	// ��ȡ�Ǽ���������ĩ�˵�
	static int skeleton_getEndPt(const cv::Mat& skeletonImg, vector<cv::Point>& endPoints);

	// ��ȡ�Ǽ�����ĩ�˵㣬ÿ�����򵥶����
	static int skeleton_getEndPt(cv::Mat biImg, vector <vector<cv::Point> > & ptPairVec);

	// �õ���������
	static int skeleton_getIntersecPt(cv::Mat biImg, vector < pair<cv::Point, cv::Point>> & ptPairVec);

	// ȥ����֧��������������
	static int skeleton_removeBranch(cv::Mat biImg, cv::Mat reBinImg);

	// ���������з֣� �����Ƚ���������޶�������
	static int skeleton_split(cv::Mat biImg, vector<cv::Mat>& imgVec);

	// ���������з֣������ݷ��򣬳��Ⱥϲ������� �����Ƚ������
	static int skeleton_merge(cv::Mat biImg, vector<cv::Mat>& imgVec);


	//static int laser_fromImage(lua_State *L);									//��άͼ�����Եת��Ϊ����
	//static int laser_getAverHeight(lua_State *L);								//��ȡƽ������λ�ø߶ȣ�ƽ��rowֵ��
 //   static int laser_getPolyline(lua_State *L);                                   //��ȡ�����ıƽ������
 //   static int laser_filterByLen(lua_State *L);                                    //ͨ������������ɸѡ��Ч����
 //   static int laser_getEndpoint(lua_State *L);                                 //��ȡ���������������Ķ˵�
	//-----------------------------------------------------------------
	// �������


    
    //���߳�ģ�飬��dequeʵ�ֵĲ���
    //static int shared_pushMat2Que(lua_State *L);            //д��ɼ�����ͼƬ������
    //static int shared_getMatAndLaser(lua_State *L);         //�Ӷ��л�ȡͼƬ�����㼤��λ�õ�����
    //static int shared_getLaserAndSendTable(lua_State *L);   
    //static int shared_getLaserAndSendString(lua_State *L);

    



};
