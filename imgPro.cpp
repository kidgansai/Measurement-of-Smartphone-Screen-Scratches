

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
#include "imgPro.h"
#include <iterator>
#include <opencv.hpp>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
//#include <timeapi.h>
using namespace std;
//using namespace cv;

#define PI 3.14159265358979323846
#define  INVALID_POINT cv::Point2d(-100000000.,-100000000.)
#define  INVALID_NUM -1000
enum PRO_ORIENTATION
{
	HORIZONTAL = 0,
	VERTICAL
};

enum PRO_MODE
{
	COUNTNOZERO = 0,
	ACCPIXELVAL
};



// 轮廓缓存
std::vector<std::vector<cv::Point>> g_contours;
std::vector<cv::Vec3f> g_circles;
std::vector<cv::Vec2f> g_points;

// 线数据缓存
vector<cv::Vec2f> g_Lines;
vector<cv::Vec4d> g_linesP;

// 线激光的轮廓数据
std::map<string, vector<cv::Point2d>> mapProfile;


//区域处理数据,string-labelImg name  , vector<vector<float>> -  7 features of region
std::map<string, vector<vector<float>>> g_regionFeature;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void splitProfile(vector<cv::Point2d> profile, vector<vector<cv::Point2d>> & conProfile, int distThresh)
{
	vector<cv::Point2d> object;
	vector<int> vecDist;

	for (int i = 0; i < profile.size() - 1; i++)
	{
		//vecDist.push_back(abs(profile[i].x - profile[i + 1].x) + abs(profile[i].y - profile[i + 1].y));
		int dist = abs(profile[i].x - profile[i + 1].x) + abs(profile[i].y - profile[i + 1].y);
		if (dist < distThresh)
		{
			object.push_back(profile[i]);
		}
		else
		{
			if (object.size() > 0)
			{
				conProfile.push_back(object);
			}
			object.clear();
		}
		if (i == profile.size() - 2)
		{
			conProfile.push_back(object);
		}
	}

}



void getContinuousProfile(vector<cv::Point2d> profile, vector<vector<cv::Point>> & conProfile)
{
	vector<cv::Point> object;
	for (int i = 0; i < profile.size(); i++)
	{
		if (profile[i].x>0)
		{
			object.push_back(profile[i]);
		}
		else
		{
			if (object.size() > 0)
			{
				conProfile.push_back(object);
			}
			object.clear();
		}
		if (i == profile.size() - 1 && profile.back().x >0)  //有效点在图片右边界
		{
			conProfile.push_back(object);
		}
	}
}

//**************************以下三个函数主要用于为RANSAC_Circle函数服务****************************************//
float verifyCircle(cv::Mat dt, cv::Point2f center, float radius, std::vector<cv::Point2f> & inlierSet)
{
	unsigned int counter = 0;
	unsigned int inlier = 0;
	float minInlierDist = 2.0f;
	float maxInlierDistMax = 100.0f;
	float maxInlierDist = radius / 25.0f;
	if (maxInlierDist<minInlierDist) maxInlierDist = minInlierDist;
	if (maxInlierDist>maxInlierDistMax) maxInlierDist = maxInlierDistMax;

	// choose samples along the circle and count inlier percentage
	for (float t = 0; t < 2 * 3.14159265359f; t += 0.05f)
	{
		counter++;
		float cX = radius*cos(t) + center.x;
		float cY = radius*sin(t) + center.y;

		if (cX < dt.cols)
			if (cX >= 0)
				if (cY < dt.rows)
					if (cY >= 0)
						if (dt.at<float>(cY, cX) < maxInlierDist)
						{
							inlier++;
							inlierSet.push_back(cv::Point2f(cX, cY));
						}
	}

	return (float)inlier / float(counter);
}


inline void getCircle(cv::Point2f& p1, cv::Point2f& p2, cv::Point2f& p3, cv::Point2f& center, float& radius)
{
	float x1 = p1.x;
	float x2 = p2.x;
	float x3 = p3.x;

	float y1 = p1.y;
	float y2 = p2.y;
	float y3 = p3.y;

	// PLEASE CHECK FOR TYPOS IN THE FORMULA :)
	center.x = (x1*x1 + y1*y1)*(y2 - y3) + (x2*x2 + y2*y2)*(y3 - y1) + (x3*x3 + y3*y3)*(y1 - y2);
	center.x /= (2 * (x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2));

	center.y = (x1*x1 + y1*y1)*(x3 - x2) + (x2*x2 + y2*y2)*(x1 - x3) + (x3*x3 + y3*y3)*(x2 - x1);
	center.y /= (2 * (x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2));

	radius = sqrt((center.x - x1)*(center.x - x1) + (center.y - y1)*(center.y - y1));
}

std::vector<cv::Point2f> getPointPositions(cv::Mat binaryImage)
{
	std::vector<cv::Point2f> pointPositions;

	for (unsigned int y = 0; y < binaryImage.rows; ++y)
	{
		//unsigned char* rowPtr = binaryImage.ptr<unsigned char>(y);
		for (unsigned int x = 0; x<binaryImage.cols; ++x)
		{
			//if(rowPtr[x] > 0) pointPositions.push_back(cv::Point2i(x,y));
			if (binaryImage.at<unsigned char>(y, x) > 0) pointPositions.push_back(cv::Point2f(x, y));
		}
	}

	return pointPositions;
}
//*************************************************以上***********************************************************************//



int findContours(cv::Mat & src, int iMode, int iMethod, int iMax, int iMin, std::vector<std::vector<cv::Point>> & contours)
{
	try
	{
		contours.clear();
		vector<cv::Vec4i> hierarchy;
		cv::findContours(src, contours, hierarchy, iMode, iMethod);

		std::vector<std::vector<cv::Point>> c;
		std::vector<std::vector<cv::Point>>::iterator itor = contours.begin();
		while (itor != contours.end())
		{
			if ((int)(*itor).size() >= iMin && (int)(*itor).size() <= iMax)
			{
				c.push_back(*itor);
			}
			itor++;
		}
		contours.clear();
		contours = c;
		return 0;
	}
	catch (cv::Exception e)
	{
		//m_strLastErr = e.err;
		return -1;
	}
	catch (...)
	{
		//m_strLastErr = "findContours error.";
		return -1;
	}
}

int prewitt(cv::Mat & src, cv::Mat & dst)
{
	try
	{
		cv::Mat & mIn = src;
		cv::Mat & mOut = dst;

		//float w = 0.7;

		cv::Mat m1, m2, m3, m4;
		cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
		filter2D(mIn, m1, mIn.depth(), kernel);
		kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
		filter2D(mIn, m2, mIn.depth(), kernel);
		//addWeighted(m1, w, m2, w, 0, m3);
		//bitwise_or(m1, m2, m3);

		kernel = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
		filter2D(mIn, m3, mIn.depth(), kernel);
		kernel = (cv::Mat_<float>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
		filter2D(mIn, m4, mIn.depth(), kernel);
		//addWeighted(m1, w, m2, w, 0, m4);
		//bitwise_or(m1, m2, m4);

		//addWeighted(m3, w, m4, w, 0, mOut);
		//bitwise_or(m3, m4, mOut);

		int nr = m1.rows;
		int nc = m1.cols;
		if (m1.isContinuous() && m2.isContinuous() && m3.isContinuous() && m4.isContinuous())
		{
			nr = 1;
			nc = nc*m1.rows*m1.channels();
		}

		for (int i = 0; i<nr; i++)
		{
			const uchar* m1Data = m1.ptr<uchar>(i);
			const uchar* m2Data = m2.ptr<uchar>(i);
			const uchar* m3Data = m3.ptr<uchar>(i);
			const uchar* m4Data = m4.ptr<uchar>(i);
			uchar * outData = mOut.ptr<uchar>(i);
			for (int j = 0; j<nc; j++)
			{
				uchar u1 = *m1Data > *m2Data ? *m1Data : *m2Data;
				uchar u2 = *m3Data > *m4Data ? *m3Data : *m4Data;
				*outData = u1 > u2 ? u1 : u2;
				m1Data++;
				m2Data++;
				m3Data++;
				m4Data++;
				outData++;
			}
		}
		return 0;
	}
	catch (cv::Exception e)
	{
		//m_strLastErr = e.err;
		return -1;
	}
	catch (...)
	{
		//sm_strLastErr = "prewitt error.";
		return -1;
	}
}
int  C_Region2vector(const cv::Mat &labelImg, vector<cv::Point> & vecPts, int index)
{
	try
	{
		vecPts.clear();
		const int * data = nullptr;
		int rows = labelImg.rows;
		int cols = labelImg.cols;
		int  label = index;

		for (int row = 0; row < rows; row++)
		{
			data = labelImg.ptr<int>(row);
			for (int col = 0; col < cols; col++)
			{
				if (data[col] == label)
				{
					vecPts.push_back(cv::Point(col, row));
				}
			}
		}
		return 0;
	}
	catch (...)
	{
		return -1;
	}
}

//从不同方向获取图像外沿轮廓
int C_GetProfile(const cv::Mat & imgIn, vector<cv::Point> & profileOut, int iOrient)
{
	try
	{
		const int IMG_WIDTH = imgIn.cols;
		const int IMG_HEIGHT = imgIn.rows;

		switch (iOrient)
		{
		case 0://从上往下
			for (int x = 0; x < IMG_WIDTH; x++)
			{
				for (int y = 0; y < IMG_HEIGHT; y++)
				{
					if (imgIn.at<int>(y, x)>0)
					{
						profileOut.push_back(cv::Point(x, y));
						break;
					}
				}
			}
			break;
		case 1://从下往上
			for (int x = 0; x < IMG_WIDTH; x++)
			{
				for (int y = IMG_HEIGHT - 1; y >= 0; y--)
				{
					if (imgIn.at<int>(y, x)>0)
					{
						profileOut.push_back(cv::Point(x, y));
						break;
					}
				}
			}
			break;
		case 2:   //从左往右
			for (int r = 0; r < IMG_HEIGHT; r++)
			{
				for (int c = 0; c <IMG_WIDTH; c++)
				{
					if (imgIn.at<int>(r, c) > 0)
					{
						profileOut.push_back(cv::Point(c, r));
						break;
					}
				}
			}
			break;
		case 3: //从右往左
			for (int r = 0; r < IMG_HEIGHT; r++)
			{
				for (int c = IMG_WIDTH - 1; c >= 0; c--)
				{
					if (imgIn.at<int>(r, c) > 0)
					{
						profileOut.push_back(cv::Point(c, r));
						break;
					}
				}
			}
			break;
		default:
			break;
		}
		return 0;
	}
	catch (...)
	{
		return -1;
	}
}


//获取轮廓的曲率
std::vector<double> getCurvature(std::vector<cv::Point> const& vecContourPoints, int step)
{
	std::vector< double > vecCurvature(vecContourPoints.size());

	if (vecContourPoints.size() < step)
		return vecCurvature;

	auto frontToBack = vecContourPoints.front() - vecContourPoints.back();
	bool isClosed = ((int)(std::max)(std::abs((float)frontToBack.x), std::abs((float)frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < vecContourPoints.size(); i++)
	{
		const cv::Point2f& pos = vecContourPoints[i];

		int maxStep = step;
		if (!isClosed)
		{
			maxStep = (std::min)((std::min)(step, i), (int)vecContourPoints.size() - 1 - i);
			if (maxStep == 0)
			{
				vecCurvature[i] = std::numeric_limits<double>::infinity();
				continue;
			}
		}
		int iminus = i - maxStep;
		int iplus = i + maxStep;
		pminus = vecContourPoints[iminus < 0 ? iminus + vecContourPoints.size() : iminus];
		pplus = vecContourPoints[iplus >= vecContourPoints.size() ? iplus - vecContourPoints.size() : iplus];

		double curvature2D;

		f1stDerivative.x = (pplus.x - pminus.x) / (iplus - iminus);
		f1stDerivative.y = (pplus.y - pminus.y) / (iplus - iminus);
		f2ndDerivative.x = (pplus.x - 2 * pos.x + pminus.x) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);
		f2ndDerivative.y = (pplus.y - 2 * pos.y + pminus.y) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);

		double divisor = f1stDerivative.x*f1stDerivative.x + f1stDerivative.y*f1stDerivative.y;
		if (std::abs(divisor) > 10e-8)
		{
			curvature2D = std::abs(f2ndDerivative.y*f1stDerivative.x - f2ndDerivative.x*f1stDerivative.y) /
				pow(divisor, 3.0 / 2.0);
		}
		else
		{
			curvature2D = std::numeric_limits<double>::infinity();
		}
		vecCurvature[i] = curvature2D;
	}
	return vecCurvature;
}

//对线段进行切割,seg为切割结果是多少块
vector<cv::Point2d> divideLineSeg(cv::Point ptStart, cv::Point ptEnd, int seg)
{
	vector<cv::Point2d> vecRes;
	for (int i = 0; i < seg; i++)
	{
		double x = ptStart.x + (int)((double)(ptEnd.x - ptStart.x) / (double)seg) * i;
		double y = ptStart.y + (int)((double)(ptEnd.y - ptStart.y) / (double)seg) * i;

		vecRes.push_back(cv::Point2d(x, y));
	}
	vecRes.push_back(ptEnd);
	return vecRes;
}

//对线段进行切割,seg为每次切割的起点和距离
vector<cv::Point2d> divideLineSeg(cv::Point ptStart, cv::Point ptEnd, vector<float> seg)
{
	vector<cv::Point2d> vecRes;
	for (size_t i = 0; i < seg.size() - 1; i = i + 2)
	{
		float startPtRatio = seg[i];
		float cutLenRatio = seg[i + 1];

		double startX = ptStart.x + double(ptEnd.x - ptStart.x)*startPtRatio;
		double startY = ptStart.y + double(ptEnd.y - ptStart.y)*startPtRatio;

		double endX = startX + double(ptEnd.x - ptStart.x)*cutLenRatio;
		double endY = startY + double(ptEnd.y - ptStart.y)*cutLenRatio;

		vecRes.push_back(cv::Point2d(startX, startY));
		vecRes.push_back(cv::Point2d(endX, endY));
	}
	return vecRes;
}
//根据曲率筛选轮廓
int C_FilterProfileByCurv(const vector<cv::Point>& profileIn, vector<cv::Point> &profileOut, int step)
{
	try
	{
		vector<double> vecCurv;
		double averCurv = 0.;
		profileOut = profileIn;
		vecCurv = getCurvature(profileIn, step);

		averCurv = (double)std::accumulate(vecCurv.begin() + 1, vecCurv.end() - 1, 0.0) / vecCurv.size();
		for (int i = 0; i < profileIn.size(); i++)
		{
			if (vecCurv[i] > averCurv)
			{
				profileOut[i] = INVALID_POINT;
			}
		}
		profileOut.erase(remove_if(profileOut.begin(), profileOut.end(), [](cv::Point pt) {return  pt.x < -100; }), profileOut.end());

		return 0;
	}
	catch (...)
	{
		return -1;
	}
}

//计算点到直线的距离，像素单位
double distPt2Line(cv::Point lp1, cv::Point lp2, cv::Point pt)
{
	double a, b, c, dis;
	a = lp2.y - lp1.y;
	b = lp1.x - lp2.x;
	c = lp2.x * lp1.y - lp1.x * lp2.y;
	// 距离公式为d = |A*x0 + B*y0 + C|/√(A^2 + B^2)
	dis = fabs(float((a * pt.x + b * pt.y + c) / std::sqrt(float(a * a + b * b))));
	return dis;
}
// End _imgPro


//计算两个二值图像的IOU
float calcRatio(cv::Mat &imgIn1, cv::Mat &imgIn2, float &ratio)
{
	if (imgIn1.empty() || imgIn2.empty())
	{
		cout << "no img" << endl;
		return -1;
	}
	cv::Mat imgIn2Bin, imgIn1Bin;       //二值图像用于计算面积比
	cv::threshold(imgIn1, imgIn1Bin, 100, 255, cv::THRESH_BINARY);
	cv::threshold(imgIn2, imgIn2Bin, 100, 255, cv::THRESH_BINARY);

	//用外轮廓来填补图像孔洞，
	cv::Mat resImgFill = cv::Mat::zeros(imgIn1.size(), CV_8UC1);
	cv::Mat glassImgFill = cv::Mat::zeros(imgIn2.size(), CV_8UC1);

	vector<vector<cv::Point>>  conts;
	cv::findContours(imgIn1Bin, conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::drawContours(resImgFill, conts, -1, cv::Scalar(255, 255, 255), cv::FILLED);

	cv::findContours(imgIn2Bin, conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::drawContours(glassImgFill, conts, -1, cv::Scalar(255, 255, 255), cv::FILLED);

	//对图像边缘进行平滑
	cv::Mat morResImg, morGlassImg;
	dilate(resImgFill, morResImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
	erode(morResImg, morResImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));//比获取的区域扩大2个像素的边缘，因为获取图像是边缘有明显空隙

	dilate(imgIn2Bin, morGlassImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(19, 19)));
	erode(morGlassImg, morGlassImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17)));

	cv::Mat imgAnd;
	bitwise_and(morResImg, morGlassImg, imgAnd);  //取实验结果图和玻璃上框图的相交的图像部分
	if (countNonZero(morGlassImg) == 0)
	{
		cout << "相交面积为0" << endl;
		return -1;
	}

	int cntImgAdd = countNonZero(imgAnd);
	int cntGlassImg = countNonZero(morGlassImg);
	return float(cntImgAdd) / cntGlassImg;

}

int  LeastSquareFittingCircle(vector<cv::Point> temp_coordinates, double &center_x, double &center_y, double & radius)//高斯消元法直接求解方程组
{
	try
	{
		center_x = 0.0f;
		center_y = 0.0f;
		radius = 0.0f;
		if (temp_coordinates.size() < 3)
		{
			return -1;
		}

		double sum_x = 0.0f, sum_y = 0.0f;
		double sum_x2 = 0.0f, sum_y2 = 0.0f;
		double sum_x3 = 0.0f, sum_y3 = 0.0f;
		double sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;

		int N = temp_coordinates.size();
		for (int i = 0; i < N; i++)
		{
			double x = temp_coordinates[i].x;
			double y = temp_coordinates[i].y;
			double x2 = x * x;
			double y2 = y * y;
			sum_x += x;
			sum_y += y;
			sum_x2 += x2;
			sum_y2 += y2;
			sum_x3 += x2 * x;
			sum_y3 += y2 * y;
			sum_xy += x * y;
			sum_x1y2 += x * y2;
			sum_x2y1 += x2 * y;
		}

		double C, D, E, G, H;
		double a, b, c;

		C = N * sum_x2 - sum_x * sum_x;
		D = N * sum_xy - sum_x * sum_y;
		E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
		G = N * sum_y2 - sum_y * sum_y;
		H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
		a = (H * D - E * G) / (C * G - D * D);
		b = (H * C - E * D) / (D * D - G * C);
		c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

		center_x = a / (-2);
		center_y = b / (-2);
		radius = sqrt(a * a + b * b - 4 * c) / 2;
		return 0;
	}
	catch (cv::Exception e)
	{
		return -2;
	}
	catch (...)
	{
		return -1;
	}
}

//***************************************************/
//int imgPro::grayImg_compareHist(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 3
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: grayImg_compareHist 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		string strIn1 = lua_tostring(L, 1);	// 第1个参数为输入1
//		string strIn2 = lua_tostring(L, 2);	// 第2个参数为输入2
//		int iMethod = (int)lua_tonumber(L, 3);
//
//		// 参数检查
//		if (g_MapHist.find(strIn1) == g_MapHist.end() || g_MapHist.find(strIn2) == g_MapHist.end())
//		{
//			string strErr = "imgPro: img_copy 有输入直方图 ";
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		//CWatch w
//		double d = cv::compareHist(g_MapHist[strIn1], g_MapHist[strIn2], iMethod);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, d);
//
//		//double d = w.Stop();
//
//		
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: grayImg_compareHist 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		return 2;
//	}
//	return 2;
//
//}
//
//
//
///**************************************************
//iErr,dMin,dMax,dRange = grayImg_geMinMaxValue(strImgIn, iRowTL,iColTL,iRowBR,iRowBR)
//功能：
//获取灰度图像指定区域的灰度最小值，最大值以及最小值和最大值之间的差值
//限制：
//无
//参数：
//strImgIn - 输入图像
//iRowTL - 指定矩形区域左上点行坐标
//iColTL - 指定矩形区域左上点列坐标
//iRowBR -指定矩形区域右下点行坐标
//iRowBR - 指定矩形区域右下点列坐标
//返回值：
//iErr - 0,正常； 非0，有错误
//dMin - 灰度最小值
//dMax - 灰度最大值
//dRange - 灰度最小值和最大值的范围
//***************************************************/
//
//int imgPro::grayImg_getMinMaxValue(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//		if (iCount != 5 
//			|| lua_type(L,1) != LUA_TSTRING
//			|| lua_type(L,2) != LUA_TNUMBER
//			|| lua_type(L,3) != LUA_TNUMBER
//			||lua_type(L,4) != LUA_TNUMBER
//			|| lua_type(L,5) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: grayImg_getMinMaxValue函数参数错误!";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 4;
//		}
//
//		string strImgIn = lua_tostring(L, 1);
//		int tlRow = (int)lua_tonumber(L, 2);	//top-left point
//		int tlCol = (int)lua_tonumber(L, 3);
//		int brRow = (int)lua_tonumber(L, 4);//bottom-right point
//		int brCol = (int)lua_tonumber(L, 5);
//
//		if (g_pMapImage->find(strImgIn) == g_pMapImage->end())
//		{
//			string strErr = "imgPro: grayImg_getMinMaxValue 输入图像错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 4;
//		}
//		double minVal = 0.;
//		double maxVal = 0.;
//		cv::Mat &imgSrc = (*g_pMapImage)[strImgIn];
//		cv::minMaxLoc(imgSrc(cv::Rect(cv::Point(tlCol, tlRow), cv::Point(brCol, brRow))), &minVal, &maxVal);
//
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, minVal);
//		lua_pushnumber(L, maxVal);
//		lua_pushnumber(L, maxVal-minVal);
//		return 4;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: grayImg_getMinMaxValue 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		return 4;
//	}
//}
//
///**************************************************
//iErr,dMean = grayImg_getMeanValue(imgIn,iRow,iCol,iWidth,iHeight,iVal)
//功能：
//获取一个矩形区域的平均灰度值
//限制：
//无
//参数：
//imgIn - 输入图像
//iRow - 矩形左上角点行坐标
//iCol - 矩形左上角点列坐标
//iWidth - 矩形宽度
//iHeight - 矩形高度
//iVal   -  大于该阈值参与计算平均灰度
//返回值：
//iErr - 0,正常； 非0，有错误
//dMean - 平均灰度值
//***************************************************/
//int imgPro::grayImg_getMeanValue(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//            || lua_type(L, 5) != LUA_TNUMBER
//            || lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: grayImg_getMeanValue 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 
//		int iRow = (int)lua_tonumber(L, 2);//
//		int iCol = (int)lua_tonumber(L, 3); //
//		int iWidth = (int)lua_tonumber(L, 4); // 
//		int iHeight = (int)lua_tonumber(L, 5); // 
//        int iVal = (int)lua_tonumber(L, 6);
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() )
//		{
//			string strErr = "imgPro: grayImg_getMeanValue 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		if ( iRow + iHeight > mIn.rows  || iCol+iWidth > mIn.cols)
//		{
//			string strErr = "imgPro: grayImg_getMeanValue 输入范围错误 ";
//
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		double dSum = 0;
//        int cnt = 0;
//		for (int r = iRow; r < iRow + iHeight; r++)
//		{
//			for (int c = iCol; c < iCol + iWidth; c++)
//			{
//                if (mIn.at<uchar>(r, c) >iVal)
//                {
//				    dSum += mIn.at<uchar>(r, c);
//                    cnt++;
//                }
//			}
//		}
//		if (cnt != 0)
//		{
//			lua_pushinteger(L, 0);
//			lua_pushnumber(L, dSum / (cnt));
//		}
//		else
//		{
//			lua_pushinteger(L, 0);
//			lua_pushnumber(L, 0);
//
//		}
//		return 2;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: grayImg_getMeanValue 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		return 2;
//	}
//}
//
////RANSAC_Circle
/*


*/
double otsu_8u_with_mask(const cv::Mat src, const cv::Mat& mask)
{
	const int n = 256;
	int m = 0;
	int i, j, h[n] = { 0 };
	for (i = 0; i < src.rows; i++)
	{
		const uchar* psrc = src.ptr(i);
		const uchar* pmask = mask.ptr(i);
		for (j = 0; j < src.cols; j++)
		{
			if (mask.at<uchar>(i, j) > 0)
			{
				h[src.at<uchar>(i, j)]++;
				++m;
			}
		}
	}

	double mu = 0, scale = 1. / (m);
	for (i = 0; i < n; i++)
		mu += i * (double)h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < n; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if ((std::min)(q1, q2) < FLT_EPSILON || (std::max)(q1, q2) > 1. - FLT_EPSILON)//FLT_EPSILON 
			continue;

		mu1 = (mu1 + i * p_i) / q1;
		mu2 = (mu - q1 * mu1) / q2;
		sigma = q1 * q2*(mu1 - mu2)*(mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	return max_val;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ZERO_AREA 0.000001
#define IMG_MIN_NUM 100
#define FILTER_AREA_MIN_NUM 10
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 算子模块

imgPro::imgPro(void)
{

}

imgPro::~imgPro(void)
{

}


bool imgPro::bZero(double dData)
{
	return dData < ZERO_AREA && dData > -ZERO_AREA ? true : false;
}

// 求两空间向量的夹角，返回值单位：cos值
double imgPro::Get2VecAngleCos(double dLineDirVec1[], double dLineDirVec2[])
{
	return (dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
		/ (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
			*sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2]));
}

// 求两空间向量的夹角，返回值单位：弧度
double imgPro::Get2VecAngle(double dLineDirVec1[], double dLineDirVec2[])
{
	return acos((dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
		/ (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
			*sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2])));
}

//dPluVec = dVector1 叉乘 dVector2
void imgPro::GetPluVecForPlane(double dVector1[], double dVector2[], double dPluVec[])
{
	dPluVec[0] = dVector2[1] * dVector1[2] - dVector2[2] * dVector1[1];
	dPluVec[1] = dVector2[2] * dVector1[0] - dVector2[0] * dVector1[2];
	dPluVec[2] = dVector2[0] * dVector1[1] - dVector2[1] * dVector1[0];
}

void imgPro::checkZero(double dTemp[], int iNum)
{
	while (iNum > 0)
	{
		if (dTemp[iNum - 1]<ZERO_AREA && dTemp[iNum - 1]>-ZERO_AREA)
		{
			dTemp[iNum - 1] = 0;
		}
		iNum--;
	}
}

//求 沿直线方向向量dLineDirVec、离已知线上的点dKnownPointCoo距离为d2PointDistance的未知点坐标dUnknowPointCoo
void imgPro::GetPointCooOnLine(double dLineDirVec[], double dKnownPointCoo[], double d2PointDistance, double dUnknowPointCoo[])
{
	if (bZero(d2PointDistance))
	{
		dUnknowPointCoo[0] = dKnownPointCoo[0];
		dUnknowPointCoo[1] = dKnownPointCoo[1];
		dUnknowPointCoo[2] = dKnownPointCoo[2];
		return;
	}
	checkZero(dLineDirVec, 3);
	checkZero(dKnownPointCoo, 3);

	double dAxisDirVec[3] = { 1, 0, 0 };	//X轴的方向向量
	dUnknowPointCoo[0] = Get2VecAngleCos(dLineDirVec, dAxisDirVec)*d2PointDistance + dKnownPointCoo[0];
	dAxisDirVec[0] = 0; dAxisDirVec[1] = 1; dAxisDirVec[2] = 0;	//Y轴的方向向量
	dUnknowPointCoo[1] = Get2VecAngleCos(dLineDirVec, dAxisDirVec)*d2PointDistance + dKnownPointCoo[1];
	dAxisDirVec[0] = 0; dAxisDirVec[1] = 0; dAxisDirVec[2] = 1;	//Z轴的方向向量
	dUnknowPointCoo[2] = Get2VecAngleCos(dLineDirVec, dAxisDirVec)*d2PointDistance + dKnownPointCoo[2];
}

//求空间两直线的交点，dLine1DirVec为直线1方向向量，dLine1Point为直线1上一点（直线2类似）
//dIntersectionPoint为交点坐标
void imgPro::GetIntersectionFor2Line(double dLine1DirVec[], double dLine1Point[], double dLine2DirVec[],
	double dLine2Point[], double dIntersectionPoint[])
{
	double bta;	//直线方程参数法的参数
	double dTem;
	int GC_Index1(0), GC_Index2(0), GC_i;
	for (GC_i = 0; GC_i<3; GC_i++)
	{
		if (!bZero(dLine1DirVec[GC_i]))
		{
			GC_Index1 = GC_i;
			break;
		}
	}
	for (GC_i = 0; GC_i<3; GC_i++)
	{
		if (GC_i != GC_Index1)
		{
			GC_Index2 = GC_i;
			dTem = dLine1DirVec[GC_Index2] * dLine2DirVec[GC_Index1] / dLine1DirVec[GC_Index1] - dLine2DirVec[GC_Index2];
			if (bZero(dTem)) continue;
			break;
		}
	}
	bta = (dLine2Point[GC_Index2] + dLine1DirVec[GC_Index2] * dLine1Point[GC_Index1] / dLine1DirVec[GC_Index1]
		- dLine1Point[GC_Index2] - dLine1DirVec[GC_Index2] * dLine2Point[GC_Index1] / dLine1DirVec[GC_Index1])
		/ (dLine1DirVec[GC_Index2] * dLine2DirVec[GC_Index1] / dLine1DirVec[GC_Index1] - dLine2DirVec[GC_Index2]);	//参数法
	dIntersectionPoint[0] = dLine2DirVec[0] * bta + dLine2Point[0];
	dIntersectionPoint[1] = dLine2DirVec[1] * bta + dLine2Point[1];
	dIntersectionPoint[2] = dLine2DirVec[2] * bta + dLine2Point[2];
}

void imgPro::getLinePoints(double dRho, double dTheta, int iWidth, int iHeight, double & dP1Row, double & dP1Col, double & dP2Row, double & dP2Col)
{
	if (dTheta < PI / 4 || dTheta > 3.*PI / 4)//处理接近于垂直的直线
	{
		dP1Col = dRho / cos(dTheta);
		dP1Row = 0;
		dP2Col = (dRho - iHeight*sin(dTheta)) / cos(dTheta);
		dP2Row = iHeight;
	}
	else
	{
		dP1Col = 0;
		dP1Row = dRho / sin(dTheta);
		dP2Col = iWidth;
		dP2Row = (dRho - iWidth*cos(dTheta)) / sin(dTheta);
	}

}

void imgPro::getClosestPointP2L(double dPoint[], double dLinePoint[], double dLineVec[], double dPointOut[])
{
	double dVec[3], dPointVec[3];
	dVec[0] = dLineVec[0]; dVec[1] = dLineVec[1]; dVec[2] = 100;
	GetPluVecForPlane(dLineVec, dVec, dPointVec);
	GetIntersectionFor2Line(dLineVec, dLinePoint, dPointVec, dPoint, dPointOut);
}


void imgPro::optimizeLines(vector<cv::Vec2f> & lines, double dR, double dT)
{
	vector<cv::Vec2f> linesIn = lines;
	lines.clear();
	vector<cv::Vec2f>::iterator itor = linesIn.begin();
	while (itor != linesIn.end())
	{
		vector<cv::Vec2f> temp;
		itor = linesIn.begin();
		temp.push_back(*itor);
		cv::Vec2f lineRef = *itor;
		linesIn.erase(itor);

		itor = linesIn.begin();
		while (itor != linesIn.end())
		{
			if (abs((*itor)[0] - lineRef[0])<dR && abs((*itor)[1] - lineRef[1])<dT)
			{
				temp.push_back(*itor);
				itor = linesIn.erase(itor);
			}
			else
			{
				itor++;
			}
		}

		double dRhoSum = 0;
		double dThetaSum = 0;
		for (vector<cv::Vec2f>::iterator itorT = temp.begin();
			itorT != temp.end(); itorT++)
		{
			dRhoSum += (*itorT)[0];
			dThetaSum += (*itorT)[1];
		}

		cv::Vec2f line;
		line[0] = dRhoSum / temp.size();
		line[1] = dThetaSum / temp.size();
		lines.push_back(line);

		itor = linesIn.begin();
	}
}
//查找二维局部峰值作为模板匹配结果坐标
vector<cv::Vec3f> imgPro::filterTemplateResPoints(const cv::Mat  & res, double dThresh, int rowDist, int colDist)
{
	int width = res.cols;
	int height = res.rows;
	float maxVal = 0.;
	vector<cv::Vec3f>  vecRes;   //结果table元素子table以score,row,col排列
	vector<cv::Vec3f>  colMaxTemp;
	int row, col;

	cv::Mat threshRes;
	cv::threshold(res, threshRes, dThresh, 0., cv::THRESH_TOZERO);
	for (col = 0; col < width; col++)
	{
		for (row = 0; row < height; row++)//先找列的局部峰值点
		{
			float valPixel = res.at<float>(row, col);
			int rowPlus = row + 1;
			int rowMinus = row - 1;
			if (rowMinus >= 0 && rowPlus<height &&valPixel>dThresh)
			{
				if (valPixel >= res.at<float>(rowMinus, col) && valPixel >= res.at<float>(rowPlus, col))
				{
					colMaxTemp.push_back(cv::Vec3f(valPixel, row, col));
				}
			}
			else if (rowMinus<0 && valPixel>dThresh)
			{
				if (valPixel >= res.at<float>(rowPlus, col))
				{
					colMaxTemp.push_back(cv::Vec3f(valPixel, row, col));
				}
			}
			else if (rowPlus >= height && valPixel > dThresh)
			{
				if (valPixel >= res.at<float>(rowMinus, col))
				{
					colMaxTemp.push_back(cv::Vec3f(valPixel, row, col));
				}
			}
		}
	}

	for (int i = 0; i<colMaxTemp.size(); i++)
	{
		int tempMaxVal = colMaxTemp[i][0];
		int tempRow = colMaxTemp[i][1];
		int tempCol = colMaxTemp[i][2];
		int colPlus = tempCol + 1;
		int colMinus = tempCol - 1;
		if (tempMaxVal > dThresh)  //判断列峰值点左右数值，找到局部峰值点
		{
			if (colMinus >= 0 && colPlus < width)
			{
				if (res.at<float>(tempRow, tempCol) >res.at<float>(tempRow, colPlus) && res.at<float>(tempRow, tempCol) > res.at<float>(tempRow, colMinus))
				{
					vecRes.push_back(cv::Vec3f(tempMaxVal, tempRow, tempCol));
				}
			}
			else if (colMinus <0)
			{
				if (res.at<float>(tempRow, tempCol)  > res.at<float>(tempRow, colPlus))
				{
					vecRes.push_back(cv::Vec3f(tempMaxVal, tempRow, tempCol));
				}
			}
			else if (colPlus >= width)
			{
				if (res.at<float>(tempRow, tempCol) > res.at<float>(tempRow, colMinus))
				{
					vecRes.push_back(cv::Vec3f(tempMaxVal, tempRow, tempCol));
				}
			}
		}
	}

	//去除临近结果点
	vector<cv::Vec3f>::iterator ite;
	while (true)
	{
		ite = adjacent_find(vecRes.begin(), vecRes.end(), [rowDist, colDist](cv::Vec3f elem1, cv::Vec3f elem2)
		{
			return abs(elem1[1] - elem2[1]) < rowDist &&
				abs(elem1[2] - elem2[2]) < colDist;
		});

		if (ite != vecRes.end())
		{
			if ((*ite)[0]>(*(ite + 1))[0])
			{
				vecRes.erase(ite + 1);
			}
			else
			{
				vecRes.erase(ite);
			}
		}
		else
		{
			break;
		}
	}
	return vecRes;
}



/**************************************************
iErr = img_getPartR(imgIn,imgOut,dRow,dCol,dAngle,dL1,dL2)
功能：
获取图像的指定旋转矩形部分
限制：
无
参数：
imgIn - 输入图像
imgOut - 输出图像
dRow - 矩形中心点行坐标
dCol - 矩形中心点列坐标
dAngle - 矩形旋转角度（单位：°）
dL1 - 矩形宽度
dL2 - 矩形高度
返回值：
iErr - 1,正常； 非1，有错误
***************************************************/
int imgPro::img_getPartR(cv::Mat &imgIn, cv::Mat &imgOut, int row, int col, double dAngle, int L1, int L2)
{
	try
	{
		if (imgIn.empty())
		{
			return -1;
		}
		cv::Mat m1 = cv::Mat::zeros(imgIn.size(), imgIn.type());

		cv::Mat v;
		boxPoints(cv::RotatedRect(cv::Point2f(col, row), cv::Size2f(L1, L2), dAngle), v);
		line(m1, cv::Point(v.at<float>(0, 0), v.at<float>(0, 1)), cv::Point(v.at<float>(1, 0), v.at<float>(1, 1)), cv::Scalar(255, 255, 255));
		line(m1, cv::Point(v.at<float>(1, 0), v.at<float>(1, 1)), cv::Point(v.at<float>(2, 0), v.at<float>(2, 1)), cv::Scalar(255, 255, 255));
		line(m1, cv::Point(v.at<float>(2, 0), v.at<float>(2, 1)), cv::Point(v.at<float>(3, 0), v.at<float>(3, 1)), cv::Scalar(255, 255, 255));
		line(m1, cv::Point(v.at<float>(0, 0), v.at<float>(0, 1)), cv::Point(v.at<float>(3, 0), v.at<float>(3, 1)), cv::Scalar(255, 255, 255));
		floodFill(m1, cv::Point(col, row), cv::Scalar(255, 255, 255));

		cv::Mat mResult;
		bitwise_and(m1, imgIn, mResult);
		imgOut = mResult.clone();

	}
	catch (...)
	{
		string strErr = "imgPro: img_getPartR 捕获到C++异常！";
		return -2;
	}
	return 1;
}

int imgPro::img_getPartR(cv::Mat & imgIn, cv::Mat & imgOut, cv::RotatedRect rotRect)
{
	try
	{
		if (imgIn.empty())
		{
			return -1;
		}
		cv::Mat m1 = cv::Mat::zeros(imgIn.size(), imgIn.type());

		cv::Mat v;
		boxPoints(rotRect,v);
		line(m1, cv::Point(v.at<float>(0, 0), v.at<float>(0, 1)), cv::Point(v.at<float>(1, 0), v.at<float>(1, 1)), cv::Scalar::all(255));
		line(m1, cv::Point(v.at<float>(1, 0), v.at<float>(1, 1)), cv::Point(v.at<float>(2, 0), v.at<float>(2, 1)), cv::Scalar::all(255));
		line(m1, cv::Point(v.at<float>(2, 0), v.at<float>(2, 1)), cv::Point(v.at<float>(3, 0), v.at<float>(3, 1)), cv::Scalar::all(255));
		line(m1, cv::Point(v.at<float>(0, 0), v.at<float>(0, 1)), cv::Point(v.at<float>(3, 0), v.at<float>(3, 1)), cv::Scalar::all(255));
		floodFill(m1, rotRect.center, cv::Scalar::all(255));

		cv::Mat mResult;
		bitwise_and(m1, imgIn, mResult);
		imgOut = mResult.clone();

	}
	catch (...)
	{
		string strErr = "imgPro: img_getPartR 捕获到C++异常！";
		return -2;
	}

	return 0;
}

/******************************************************************
iErr = imgPro.img_rotate(strImgIn, strImgOut, dAngle, iFillValue, dCenterRow, dCenterCol)
功能：
图形按照指定的中心(iCenterRow, iCenterCol)旋转特定的角度dAngle(单位：度)，逆时针为正。
限制：
无
参数：
strImgIn - 图像输入
strImgOut - 图像输出
dAngle - 旋转角度
iFillValue - 图像外区域填充值
dCenterRow - 旋转中心行数，y坐标
dCenterCol - 旋转中心列数，x坐标
返回值：
iErr - 1, 正常； 非1，有错误
***************************************************/
int imgPro::img_rotate(cv::Mat &imgIn, cv::Mat &imgOut, double dAngle, uchar iFillValue, int centerRow, int centerCol)
{
	try
	{
		// 如果输出图像缓冲不存在，则创建一个缓冲
		if (imgIn.empty())
		{
			return -1;
		}

		//旋转中心为图像中心  
		cv::Point2f center;
		center.x = centerCol;
		center.y = centerRow;
		//计算二维旋转的仿射变换矩阵  
		cv::Mat M = cv::getRotationMatrix2D(center, dAngle, 1);

		//变换图像，并用黑色填充其余值
		//cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 
		cv::warpAffine(imgIn, imgOut, M, imgIn.size(), cv::INTER_LINEAR || cv::WARP_FILL_OUTLIERS, 0, cv::Scalar(iFillValue));
		//double d = w.Stop();
		return 1;

	}
	catch (...)
	{
		string strErr = "imgPro: img_rotate 捕获到C++异常！";
		return -1;
	}
}
/******************************************************************
iErr = imgPro.img_translate(strImgIn, strImgOut, iRowTransl,iColTransl, iFillValue)
功能：
图形平移
限制：
无
参数：
strImgIn - 图像输入
strImgOut - 图像输出
iRowTransl - 行平移值，使图像上下平移
iColTransl   - 列平移值，使图像左右平移
iFillValue - 图像外区域填充值
返回值：
iErr - 0, 正常； 非0，有错误
***************************************************/
int imgPro::img_translate(cv::Mat &imgIn, cv::Mat &imgOut, int iRowTransl, int iColTransl, uchar iFillValue)
{
	try
	{
		if (imgIn.empty())
		{
			return -1;
		}

		cv::Mat t_mat = cv::Mat::zeros(2, 3, CV_32FC1);
		t_mat.at<float>(0, 0) = 1;
		t_mat.at<float>(0, 2) = iColTransl; //水平(列)平移量
		t_mat.at<float>(1, 1) = 1;
		t_mat.at<float>(1, 2) = iRowTransl; //竖直(行)平移量

											//根据平移矩阵进行仿射变换
		cv::warpAffine(imgIn, imgOut, t_mat, imgIn.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS, 0, cv::Scalar(iFillValue));
		//double d = w.Stop();
		return 1;

	}
	catch (...)
	{
		string strErr = "imgPro: img_translate 捕获到C++异常！";
		return -1;
	}
}

int imgPro::img_enhanceGM(cv::Mat &imgIn, cv::Mat &imgOut, double gamma, double c)
{
	try
	{
		if (imgIn.empty())
		{
			return 1;
		}
		cv::Mat lut_matrix(1, 256, CV_8UC1);
		uchar * ptr = lut_matrix.ptr();
		for (int i = 0; i < 256; i++)
			ptr[i] = (int)(c*(pow((double)i / 255.0, gamma) * 255.0));

		cv::Mat result;
		LUT(imgIn, lut_matrix, imgOut);
		return 0;
	}
	catch (...)
	{
		return -1;
	}
}

int imgPro::img_enhanceLog(cv::Mat &imgIn, cv::Mat &imgOut, double c /*= 1.0*/)
{
	try
	{
		if (imgIn.empty())
		{
			return 1;
		}
		cv::Mat srcImage = imgIn.clone();
		cv::Mat resultImage = cv::Mat::zeros(srcImage.size(), srcImage.type());
		cv::add(srcImage, cv::Scalar(1.0), srcImage);  //计算 r+1
		srcImage.convertTo(srcImage, CV_32F);  //转化为32位浮点型
		cv::log(srcImage, resultImage);            //计算log(1+r)
		resultImage = c * resultImage;
		//归一化处理
		cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX);
		cv::convertScaleAbs(resultImage, imgOut);

		return 0;
	}
	catch (...)
	{
		return -1;
	}

}

/************************************************************************/
/*  图像等比例缩小         
如果图像长边小于Size中最大值，那么对他们进行等比例缩小。
*/
/************************************************************************/
int imgPro::img_resize_keepRatio(cv::Mat &imgIn, cv::Mat &imgOut, cv::Size size)
{
	if (imgIn.empty())
	{
		return -1;
	}

	int sizeMax = std::max(size.width, size.height);
	int imgMax = std::max(imgIn.rows, imgIn.cols);

	if (imgMax > sizeMax)
	{
		double scale = double(sizeMax) / imgMax;
		resize(imgIn, imgOut, cv::Size(std::floor(imgIn.cols*scale), std::floor(imgIn.rows*scale)));

	}
	else
	{
		imgOut = imgIn.clone();
	}
}


///**************************************************
//iErr,Points = img_getContourPoints(imgIn, iIndex)
//功能：
//获取二值化图像中指定轮廓包含的点
//限制：
//无
//参数：
//imgIn - 输入图像
//iMode - 轮廓索引值，从1开始
//返回值：
//iErr - 0,正常； 非0，有错误
//Points - 包含构成指定轮廓的点的Table，双层Table,每个point下还有一个Table存放对应的row,col
//***************************************************/
//int imgPro::img_getContourPoints(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		// 参数检查
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_getContourPoints 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		int iIndex = (int)lua_tonumber(L, 2);// 第二个参数为索引值
//
//		iIndex -= 1;    //设置索引值从1开始
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() || iIndex < 0 || iIndex >= g_contours.size())
//		{
//			string strErr = "imgPro: img_getContourPoints 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！或索引值错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];  //所以mIn是不是用不用都没有差别
//
//		lua_pushinteger(L, 0);
//		lua_createtable(L, g_contours[iIndex].size(), 0);
//		for (int i = 0; i < g_contours[iIndex].size(); i++)
//		{
//			lua_pushnumber(L, i + 1);
//			lua_createtable(L, 0, 2);
//			lua_pushnumber(L, g_contours[iIndex][i].y);
//			lua_setfield(L, -2, "row");
//			lua_pushnumber(L, g_contours[iIndex][i].x);
//			lua_setfield(L, -2, "col");
//			lua_settable(L, -3);
//		}
//
//		return 2;
//
//	}
//	catch (...)
//	{
//		g_contours.clear();
//		string strErr = "imgPro: img_getContourPoints 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}
//
//
///**************************************************
//iErr = img_drawContours(imgIn,imgOut,iIndex,iRed,iGreen,iBlue,iThickness)
//功能：
//在图像中绘制轮廓
//限制：
//无
//参数：
//imgIn - 输入图像
//imgOut- 输出图像
//iIndex - 轮廓序号: 从1开始
//iRed - 红色分量
//iGreen - 绿色分量
//iBlue - 蓝色分量
//iThickness - 轮廓宽度
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
//int imgPro::img_drawContours(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 7
//			|| lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_drawContours 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//        string strOut = lua_tostring(L, 2);	// 第一个参数为输入图像
//		int iIndex = (int)lua_tonumber(L, 3);// 第二个参数为 轮廓序号
//		int iRed = (int)lua_tonumber(L, 4); // 
//		int iGreen = (int)lua_tonumber(L, 5); // 
//		int iBlue = (int)lua_tonumber(L, 6); // 
//		int iThickness = (int)lua_tonumber(L, 7); // 显示宽度
//
//		iIndex -= 1;
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  
//			|| iIndex<0 || iIndex>=g_contours.size())
//		{
//			string strErr = "imgPro: img_drawContours 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！或索引范围错误";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//
//			return 1;
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//        if (g_pMapImage->find(strOut) == g_pMapImage->end())
//        {
//            (*g_pMapImage)[strOut] = cv::Mat();
//        }
//
//        cv::Mat &mOut = (*g_pMapImage)[strOut];
//        if ((mIn.channels() == 1))
//        {
//            cvtColor(mIn, mOut, CV_GRAY2BGR);
//        }
//        else
//        {
//            mOut = mIn.clone();
//        }
//        cv::drawContours(mOut, g_contours, iIndex, cv::Scalar(iBlue, iGreen, iRed), iThickness);
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_drawContours 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//	lua_pushinteger(L, 0);
//	return 1;
//}
//
//
///**************************************************
//iErr,iNum = img_findHarris(imgIn,imgOut,iBlock,iAperture,dK,iThresh)
//功能：
//    查找图像中Harris角点
//限制：
//无
//参数：
//imgIn - 输入图像
//imgOut - 输出图像
//iBlock - 邻域窗口大小
//iAperture - sobel边缘检测窗口大小
//dK - 系数
//iThresh - 角点筛选阈值
//返回值：
//iErr - 0,正常； 非0，有错误
//iNum - 角点个数
//***************************************************/
//int imgPro::img_findHarris(lua_State* L)
//{
//	try
//	{
//		//------------------------- 参数输入和检查 --------------------------
//		int iCount = lua_gettop(L);      // 参数个数
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_findHarris 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第2个参数为输出图像
//		int iBlock = (int)lua_tonumber(L, 3); //邻域窗口大小
//		int iAperture = (int)lua_tonumber(L, 4);   //孔径大小
//		double dK = lua_tonumber(L, 5);
//		int iThresh = lua_tonumber(L, 6);
//
//		if ((*g_pMapImage).find(strIn) == (*g_pMapImage).end())	//如果没有图像
//		{
//			string strErr = "imgPro: img_findHarris 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		if ((*g_pMapImage).find(strOut) == (*g_pMapImage).end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat();
//			(*g_pMapImage)[strOut].create(mIn.size(), mIn.type());
//		}
//		//------------------------- 算法实现部分 --------------------------
//		cv::Mat & mOut = (*g_pMapImage)[strOut];
//		cv::Mat mTemp = mIn.clone();
//		cv::Mat dst_norm = mIn.clone();
//		cv::Mat dst_norm_scaled;
//		cornerHarris(mIn, mTemp, iBlock, iAperture, dK);
//		normalize(mTemp, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
//		convertScaleAbs(dst_norm, dst_norm_scaled);
//
//		mOut = dst_norm_scaled.clone();
//		g_points.clear();
//		for (int j = 0; j < dst_norm.rows; j++)
//		{
//			for (int i = 0; i < dst_norm.cols; i++)
//			{
//				int iCc = (int)dst_norm.at<float>(j, i);
//				if ((int)dst_norm.at<float>(j, i) > iThresh)
//				{
//					cv::Vec2f v;
//					v[0] = i;
//					v[1] = j;
//					g_points.push_back(v);
//				}
//			}
//		}
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_findHarris 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//		return 2;
//	}
//
//	//------------------------- 返回部分 --------------------------	
//
//	lua_pushinteger(L, 0);
//	lua_pushinteger(L, g_points.size());
//	return 2;
//
//}
//
///*************************************************************
//iErr = img_fillRectR(strImgIn,strImgOut,RectRow,RectCol,dAngle,RectWidth,RectHeigth)
//功能：
//	填充矩形区域为白色（灰度值255）
//参数：
//	strImgIn：输入图像
//	strImgOut：输出图像	
//	RectRow ：旋转矩形中心行坐标
//	RectCol	：旋转矩形中心列坐标
//	dAngle	：旋转矩形角度
//	RectWidth：旋转矩形宽
//	RectHeigth：旋转矩形高
//
//***************************************************************/
//
//int imgPro::img_fillRectR(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 7
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_fillRectR 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第二个参数为输出图像
//		float row = (float)lua_tonumber(L, 3);
//		float col = (float)lua_tonumber(L, 4);
//		float angle = (float)lua_tonumber(L, 5);
//		float L1 = (float)lua_tonumber(L, 6);
//		float L2 = (float)lua_tonumber(L, 7);
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end())
//		{
//			string strErr = "imgPro: img_fillRectR 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat();
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		cv::Mat & mOut = (*g_pMapImage)[strOut];
//
//		cv::Mat m1 = cv::Mat::zeros(mIn.size(), mIn.type());
//
//		cv::Mat v;
//		boxPoints(cv::RotatedRect(cv::Point2f(col, row), cv::Size2f(L1, L2), angle), v);
//		line(m1, cv::Point(v.at<float>(0, 0), v.at<float>(0, 1)), cv::Point(v.at<float>(1, 0), v.at<float>(1, 1)), cv::Scalar(255, 255, 255));
//		line(m1, cv::Point(v.at<float>(1, 0), v.at<float>(1, 1)), cv::Point(v.at<float>(2, 0), v.at<float>(2, 1)), cv::Scalar(255, 255, 255));
//		line(m1, cv::Point(v.at<float>(2, 0), v.at<float>(2, 1)), cv::Point(v.at<float>(3, 0), v.at<float>(3, 1)), cv::Scalar(255, 255, 255));
//		line(m1, cv::Point(v.at<float>(0, 0), v.at<float>(0, 1)), cv::Point(v.at<float>(3, 0), v.at<float>(3, 1)), cv::Scalar(255, 255, 255));
//		floodFill(m1, cv::Point(col, row), cv::Scalar(255, 255, 255));
//
//		cv::Mat mResult;
//		bitwise_or(m1, mIn, mResult);
//		mOut = mResult.clone();
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_fillRectR 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//	lua_pushinteger(L, 0);
//	return 1;
//
//}
//
///**************************************************
//iErr = img_filter(imgIn, imgOut,
//d00,d01, d02,
//d10, d11, d12,
//d20, d21, d22)   //3X3
//iErr = img_filter(imgIn, imgOut,
//d00,d01, d02, d03, d04,
//d10, d11, d12, d13, d14,
//d20, d21, d22, d23, d24,
//d30, d31, d32, d33, d34,
//d40, d41, d42, d43, d44)   //5X5
//功能：
//自定义滤波器
//限制：
//无
//参数：
//imgIn - 图像输入
//imgOut - 图像输出
//其他为算子数据
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
//int imgPro::img_filter(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//
//		if (iCount != 11 && iCount != 27
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: img_filter 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第二个参数为输出图像
//		double d00, d01, d02, d03, d04;
//		double d10, d11, d12, d13, d14;
//		double d20, d21, d22, d23, d24;
//		double d30, d31, d32, d33, d34;
//		double d40, d41, d42, d43, d44;
//
//		cv::Mat kernel;
//		if (iCount == 11) // 3x3
//		{
//			if(lua_type(L, 3) != LUA_TNUMBER
//				|| lua_type(L, 4) != LUA_TNUMBER
//				|| lua_type(L, 5) != LUA_TNUMBER
//				|| lua_type(L, 6) != LUA_TNUMBER
//				|| lua_type(L, 7) != LUA_TNUMBER
//				|| lua_type(L, 8) != LUA_TNUMBER
//				|| lua_type(L, 9) != LUA_TNUMBER
//				|| lua_type(L, 10) != LUA_TNUMBER
//				|| lua_type(L, 11) != LUA_TNUMBER)
//			{
//				string strErr = "imgPro: img_filter 3*3模板参数错误！";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//				lua_pushinteger(L, -100);
//				return 1;
//			}
//
//			d00 = lua_tonumber(L, 3);		// kernel : 00
//			d01 = lua_tonumber(L, 4);		// kernel : 01
//			d02 = lua_tonumber(L, 5);		// kernel : 02
//			d10 = lua_tonumber(L, 6);		// kernel : 10
//			d11 = lua_tonumber(L, 7);		// kernel : 11
//			d12 = lua_tonumber(L, 8);		// kernel : 12
//			d20 = lua_tonumber(L, 9);		// kernel : 20
//			d21 = lua_tonumber(L, 10);		// kernel : 21
//			d22 = lua_tonumber(L, 11);		// kernel : 22
//			kernel = (cv::Mat_<float>(3, 3) << d00, d01, d02, d10, d11, d12, d20, d21, d22);
//		}
//
//		if (iCount == 27) // 5x5
//		{
//			if(lua_type(L, 3) != LUA_TNUMBER
//				|| lua_type(L, 4) != LUA_TNUMBER
//				|| lua_type(L, 5) != LUA_TNUMBER
//				|| lua_type(L, 6) != LUA_TNUMBER
//				|| lua_type(L, 7) != LUA_TNUMBER
//				|| lua_type(L, 8) != LUA_TNUMBER
//				|| lua_type(L, 9) != LUA_TNUMBER
//				|| lua_type(L, 10) != LUA_TNUMBER
//				|| lua_type(L, 11) != LUA_TNUMBER
//				|| lua_type(L, 12) != LUA_TNUMBER
//				|| lua_type(L, 13) != LUA_TNUMBER
//				|| lua_type(L, 14) != LUA_TNUMBER
//				|| lua_type(L, 15) != LUA_TNUMBER
//				|| lua_type(L, 16) != LUA_TNUMBER
//				|| lua_type(L, 17) != LUA_TNUMBER
//				|| lua_type(L, 18) != LUA_TNUMBER
//				|| lua_type(L, 19) != LUA_TNUMBER
//				|| lua_type(L, 20) != LUA_TNUMBER
//				|| lua_type(L, 21) != LUA_TNUMBER
//				|| lua_type(L, 22) != LUA_TNUMBER
//				|| lua_type(L, 23) != LUA_TNUMBER
//				|| lua_type(L, 24) != LUA_TNUMBER
//				|| lua_type(L, 25) != LUA_TNUMBER
//				|| lua_type(L, 26) != LUA_TNUMBER
//				|| lua_type(L, 27) != LUA_TNUMBER)
//			{
//				string strErr = "imgPro: img_filter 5*5模板参数错误！";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//				lua_pushinteger(L, -100);
//				return 1;
//			}
//
//			d00 = lua_tonumber(L, 3);		// kernel : 00
//			d01 = lua_tonumber(L, 4);		// kernel : 01
//			d02 = lua_tonumber(L, 5);		// kernel : 02
//			d03 = lua_tonumber(L, 6);		// kernel : 10
//			d04 = lua_tonumber(L, 7);		// kernel : 11
//			d10 = lua_tonumber(L, 8);		// kernel : 00
//			d11 = lua_tonumber(L, 9);		// kernel : 01
//			d12 = lua_tonumber(L, 10);		// kernel : 02
//			d13 = lua_tonumber(L, 11);		// kernel : 10
//			d14 = lua_tonumber(L, 12);		// kernel : 11
//			d20 = lua_tonumber(L, 13);		// kernel : 00
//			d21 = lua_tonumber(L, 14);		// kernel : 01
//			d22 = lua_tonumber(L, 15);		// kernel : 02
//			d23 = lua_tonumber(L, 16);		// kernel : 10
//			d24 = lua_tonumber(L, 17);		// kernel : 11
//			d30 = lua_tonumber(L, 18);		// kernel : 00
//			d31 = lua_tonumber(L, 19);		// kernel : 01
//			d32 = lua_tonumber(L, 20);		// kernel : 02
//			d33 = lua_tonumber(L, 21);		// kernel : 10
//			d34 = lua_tonumber(L, 22);		// kernel : 11
//			d40 = lua_tonumber(L, 23);		// kernel : 00
//			d41 = lua_tonumber(L, 24);		// kernel : 01
//			d42 = lua_tonumber(L, 25);		// kernel : 02
//			d43 = lua_tonumber(L, 26);		// kernel : 10
//			d44 = lua_tonumber(L, 27);		// kernel : 11
//			kernel = (cv::Mat_<float>(5, 5) << d00, d01, d02, d03, d04,
//				d10, d11, d12, d13, d14,
//				d20, d21, d22, d23, d24,
//				d30, d31, d32, d33, d34,
//				d40, d41, d42, d43, d44);
//		}
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//
//			string strErr = "imgPro: img_filter 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat();
//		}
//
//		//CWatch w;
//		if (strOut == strIn)
//		{
//			cv::Mat mTemp;
//			filter2D((*g_pMapImage)[strIn], mTemp, (*g_pMapImage)[strIn].depth(), kernel);
//			mTemp.copyTo((*g_pMapImage)[strOut]);
//		}
//		else
//		{
//			filter2D((*g_pMapImage)[strIn], (*g_pMapImage)[strOut], (*g_pMapImage)[strIn].depth(), kernel);
//		}
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_filter 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//	lua_pushinteger(L, 0);
//	return 1;
//}
//
//
///**************************************************
//iErr = img_sobel(imgIn,imgOut,iDX,iDY)
//功能：
//对图像进行sobel边缘检测
//限制：
//无
//参数：
//imgIn - 输入图像
//imgOut - 输出图像
//iDX - X方向取1或0
//iDY - Y方向取1或0
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
//int imgPro::img_sobel(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 4
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_sobel 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第二个参数为输出图像
//		int iDX = (int)lua_tonumber(L, 3); // X方向取1或0
//		int iDY = (int)lua_tonumber(L, 4); // Y方向取1或0
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: img_sobel 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat();
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		cv::Mat & mOut = (*g_pMapImage)[strOut];
//        cv::Mat temp;
//		cv::Sobel(mIn, temp, CV_16SC1, iDX, iDY, 3, 1, 1, cv::BORDER_DEFAULT);
//        cv::convertScaleAbs(temp, mOut);
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_sobel 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//	lua_pushinteger(L, 0);
//	return 1;
//}
//
///**************************************************
//iErr = img_prewitt(imgIn,imgOut)
//功能：
//对图像进行prewitt边缘检测
//限制：
//无
//参数：
//imgIn - 输入图像
//imgOut - 输出图像
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
//int imgPro::img_prewitt(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: img_prewitt 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第二个参数为输出图像
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: img_prewitt 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat();
//			(*g_pMapImage)[strOut].create((*g_pMapImage)[strIn].size(), (*g_pMapImage)[strIn].type());
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		cv::Mat & mOut = (*g_pMapImage)[strOut];
//
//		if (C_Prewitt(mIn, mOut) != 0)
//		{
//			string strErr = "imgPro: img_prewitt 子程序错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_prewitt 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//	lua_pushinteger(L, 0);
//	return 1;
//}
//
//
///**************************************************
//iErr = img_LOG(imgIn,imgOut)
//功能：
//对图像进行高斯拉普拉斯边缘检测
//限制：
//无
//参数：
//imgIn - 输入图像
//imgOut - 输出图像
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
//int imgPro::img_LOG(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: img_LOG 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第二个参数为输出图像
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: img_LOG 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat();
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		cv::Mat & mOut = (*g_pMapImage)[strOut];
//
//		cv::Mat kernel = (cv::Mat_<float>(5, 5) << -2, -4, -4, -4, -2,
//			-4, 0, 8, 0, -4,
//			-4, 8, 24, 8, -4,
//			-4, 0, 8, 0, -4,
//			-2, -4, -4, -4, -2);
//		filter2D(mIn, mOut, mIn.depth(), kernel);
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_LOG 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//	lua_pushinteger(L, 0);
//	return 1;
//}
//
//
///**************************************************
//iErr,tableRes = img_projection(strImgIn, strImgProj,iOrit,iMode,iCalcImgProj)
//功能：
//	图像投影,部分功能可用reduce函数替代
//限制：
//无
//参数：
//imgIn - 图像输入
//strImgProj - 投影图像输出
//iOrit  - 计算投影的方向，0 - 按行计算 , 1 - 按列计算
//iMode - 计算方式，0 : 该行或列非0像素的个数，1： 该行或列像素值和 ，2 - 备用
//iCalcImgProj - 是否输出投影图像,0 :不输出  1，输出
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/

int imgPro::img_projection(cv::Mat& imgIn, cv::Mat &imgProj, std::vector<int> &vecProjVal,int iOrit, int iMode,bool iCalcImgProj)
{
	try
	{
		if (imgIn.empty() )
		{
			string strErr = "imgPro:img_projection 输入图像不存在！";
			return -2;
		}
		if (cv::countNonZero(imgIn) < 10)
		{
			return 1;
		}

		vecProjVal.clear();
		if (PRO_ORIENTATION::HORIZONTAL == iOrit)  //0为水平方向
		{
			if (PRO_MODE::COUNTNOZERO == iMode)//非0像素个数
			{
				for (int rowi = 0; rowi < imgIn.rows;rowi++)
				{
					vecProjVal.push_back(cv::countNonZero(imgIn.rowRange(rowi, rowi + 1)));
				}
			} 
			else //像素值和
			{
				for (int rowi = 0; rowi < imgIn.rows; rowi++)
				{
					vecProjVal.push_back(cv::sum(imgIn.rowRange(rowi, rowi + 1)).val[0]);
				}
			}
		} 
		else
		{
			if (PRO_MODE::COUNTNOZERO == iMode)//非0像素个数
			{
				for (int coli = 0; coli < imgIn.cols; coli++)
				{
					vecProjVal.push_back(cv::countNonZero(imgIn.colRange(coli, coli + 1)));
				}
			}
			else //像素值和
			{
				for (int coli = 0; coli < imgIn.cols; coli++)
				{
					vecProjVal.push_back(cv::sum(imgIn.colRange(coli, coli + 1)).val[0]);
				}
			}

		}
		int img_rows = imgIn.rows;
		int img_cols = imgIn.cols;
		if (iCalcImgProj > 0 )//计算输出映射图
		{
			imgProj.create(imgIn.rows,imgIn.cols, CV_8UC1);
			imgProj.setTo(0);

			if (PRO_ORIENTATION::HORIZONTAL == iOrit)
				{
					if (PRO_MODE::COUNTNOZERO == iMode)
					{
						for (int i = 0; i < img_rows;i++)
						{
							imgProj(cv::Range(i, i + 1), cv::Range(0, vecProjVal[i])).setTo(255);
						}
					} 
					else   
					{
						vector<int>::iterator maxIte = std::max_element(vecProjVal.begin(), vecProjVal.end());
						for (int i = 0; i < img_rows; i++)
						{
							imgProj(cv::Range(i, i + 1), cv::Range(0, vecProjVal[i]*img_cols/(*maxIte))).setTo(255);
						}
					}
				}
				else
				{
					if (PRO_MODE::COUNTNOZERO == iMode)
					{
						for (int i = 0; i < img_cols; i++)
						{
							imgProj(cv::Range(img_rows- vecProjVal[i],img_rows),cv::Range(i, i + 1)).setTo(255);
						}
					} 
					else
					{
						vector<int>::iterator maxIte = std::max_element(vecProjVal.begin(), vecProjVal.end());
						for (int i = 0; i < img_cols; i++)
						{
							imgProj(cv::Range(img_rows - vecProjVal[i]*img_rows/(*maxIte), img_rows), cv::Range(i, i + 1)).setTo(255);
						}
					}
			}
		}
		//点的个数归一化到与图形列同范围
		//std::pair<vector<int>::iterator,vector<int>::iterator> iteMinMaxVal = std::minmax_element(vecProjVal.begin(), vecProjVal.end());
		//int minVal = *iteMinMaxVal.first;
		//int maxVal = *iteMinMaxVal.second;

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro:img_projection 捕获到C++异常！";
		return -2;

	}
}
//
int imgPro::img_drawMask(const cv::Mat & src, cv::Mat & imgOut, const cv::Mat & mask,cv::Scalar c, float percent)
{
	try
	{
		if (src.empty())
		{
			return 1;
		}
		if (src.size()!= mask.size())
		{
			return 1;
		}
		cv::Mat srcTmpF, maskTmpF,colorF;
		cv::Mat srcTmp = src.clone();
		cv::Mat maskTmp = mask.clone();
		if (src.channels() ==1)
		{
			cvtColor(srcTmp, srcTmp, cv::COLOR_GRAY2BGR);
			srcTmp.convertTo(srcTmpF, CV_32F);
			srcTmpF /= 255;

		}
		else
		{
			srcTmp.convertTo(srcTmpF, CV_32F);  //not CV_32FC3
			srcTmpF /= 255;

		}
		double maxVal = 0;
		cv::minMaxLoc(maskTmp, 0, &maxVal, 0, 0);
		if (maxVal > 1)
			maskTmp /= 255;
	
		cvtColor(maskTmp, maskTmp, cv::COLOR_GRAY2BGR);
		maskTmp.convertTo(maskTmpF, CV_32F); 
		maskTmpF *= percent;
		cv::Mat color = cv::Mat::zeros(mask.size(), CV_8UC3);
		color.setTo(c);
		color.convertTo(colorF, CV_32F);
		colorF /= 255;
		cv::Mat imgOutF;
		cv::Mat cm = colorF.mul(maskTmpF);
		cv::Mat subm = cv::Scalar::all(1) - maskTmpF;
		cv::Mat muls =  srcTmpF .mul(subm);
		imgOutF = cm + muls;

		//normalize(imgOut, imgOut, 255, cv::NORM_MINMAX);
		imgOutF *= 255;

		imgOutF.convertTo(imgOut, CV_8U);
		return 0;

	}
	catch (...)
	{
		return -1;
	}

}
int imgPro::img_drawRect(cv::Mat & src, cv::Mat & imgOut, cv::Point &tlOut, cv::Point centerPt, cv::Size rectSize, double angle, cv::Scalar color, int thickness)
{
	
	try
	{
		int halfWidth = rectSize.width / 2;
		int halfHeight = rectSize.height / 2;

		cv::Point tl = cv::Point(centerPt.x-halfWidth,centerPt.y-halfHeight);
		cv::Point tr = cv::Point(centerPt.x+halfWidth,centerPt.y-halfHeight);
		cv::Point dl = cv::Point(centerPt.x-halfWidth,centerPt.y+halfHeight);
		cv::Point dr = cv::Point(centerPt.x+halfWidth,centerPt.y+halfHeight);

		vector<cv::Point> vecPts{tl,tr,dr,dl};
		vector<cv::Point> vecPtsRot;
		
		cv::Point ptOut;
		for (cv::Point &pt : vecPts)
		{
			pt_rotate(src.size(),pt,centerPt,ptOut,angle);
			vecPtsRot.push_back(ptOut);
		}
		cv::Mat show = src.clone();
		//if (show.channels() <3)
		//{
		//	cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
		//}
		tlOut = vecPtsRot[0];
		if (thickness>0)
		{
			cv::polylines(show,vecPtsRot,true,color,thickness);
		} 
		else
		{			
			std::vector<std::vector<cv::Point>> contours{ vecPtsRot };
			cv::drawContours(show, contours, 0, color, -1);
		}

		imgOut = show;
		return 0;

	}
	catch (const std::exception& e)
	{
		string err = "img_drawRect:发生异常, " + string(e.what());
		return -1;
	}


	return 0;
}
int imgPro::img_drawRect(cv::Mat & src, cv::Mat & imgOut, cv::RotatedRect &rotRectOut, cv::Point centerPt, cv::Size rectSize, double angle, cv::Scalar color, int thickness)
{
	
	try
	{
		int halfWidth = rectSize.width / 2;
		int halfHeight = rectSize.height / 2;

		cv::Point tl = cv::Point(centerPt.x-halfWidth,centerPt.y-halfHeight);
		cv::Point tr = cv::Point(centerPt.x+halfWidth,centerPt.y-halfHeight);
		cv::Point dl = cv::Point(centerPt.x-halfWidth,centerPt.y+halfHeight);
		cv::Point dr = cv::Point(centerPt.x+halfWidth,centerPt.y+halfHeight);

		vector<cv::Point> vecPts{tl,tr,dr,dl};
		vector<cv::Point> vecPtsRot;
		
		cv::Point ptOut;
		for (cv::Point &pt : vecPts)
		{
			pt_rotate(src.size(),pt,centerPt,ptOut,angle);
			vecPtsRot.push_back(ptOut);
		}
		cv::Mat show = src.clone();
		//if (show.channels() <3)
		//{
		//	cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
		//}
		rotRectOut =cv::minAreaRect(vecPtsRot);
		if (thickness>0)
		{
			cv::polylines(show,vecPtsRot,true,color,thickness);
		} 
		else
		{			
			std::vector<std::vector<cv::Point>> contours{ vecPtsRot };
			cv::drawContours(show, contours, 0, color, -1);
		}

		imgOut = show;
		return 0;

	}
	catch (const std::exception& e)
	{
		string err = "img_drawRect:发生异常, " + string(e.what());
		return -1;
	}


	return 0;
}
int imgPro::img_drawRect(cv::Mat src, cv::Mat & imgOut, cv::RotatedRect rotRect, cv::Scalar color, int thickness)
{
	cv::Point pt;
	img_drawRect(src, imgOut, pt, rotRect.center, rotRect.size, -rotRect.angle, color, thickness);
	return 0;
}

//通道数由src确定
//输入：cv::Vec4f line为fitLine的拟合结果
//输出：cv::Vec4f 为直线在图像中首末点
cv::Vec4f imgPro::img_drawLine(cv::Mat src, cv::Mat & imgOut, cv::Vec4f line, cv::Scalar color, int thickness)
{

	imgOut = src.clone();


	double cos_theta = line[0];
	double sin_theta = line[1];
	double x0 = line[2], y0 = line[3];

	double theta = atan2(sin_theta, cos_theta) + PI / 2.0;//angle
	double rho = y0 * cos_theta - x0 * sin_theta;
	cv::Point pt1, pt2;
	if (theta < PI / 4. || theta > 3.*PI / 4.)// ~vertical line
	{
		pt1 = cv::Point(rho / cos(theta), 0);
		pt2 = cv::Point((rho - src.rows * sin(theta)) / cos(theta), src.rows-1);
		cv::line(imgOut, pt1, pt2, color, thickness);
	}
	else
	{
		pt1 = cv::Point(0, rho / sin(theta));
		pt2 = cv::Point(src.cols-1, (rho - src.cols * cos(theta)) / sin(theta));
		cv::line(imgOut, pt1, pt2, color, thickness);
	}
	return cv::Vec4f(pt1.x,pt1.y,pt2.x,pt2.y);
}

int imgPro::img_drawSegments(cv::Mat &srcImg,cv::Mat &drawImg, vector<cv::Vec4f> lines, cv::Scalar color)
{
	try
	{
		if (srcImg.channels() == 1)
		{
			drawImg = srcImg.clone();
		}
		else if (srcImg.channels() == 3)
		{
			cvtColor(srcImg, drawImg, cv::COLOR_BGR2GRAY);
		}
		// Create a 3 channel image in order to draw colored lines
		std::vector<cv::Mat> planes;
		planes.push_back(drawImg);
		planes.push_back(drawImg);
		planes.push_back(drawImg);

		merge(planes, drawImg);

		double gap = 10.0;
		double arrow_angle = 30.0;

		// Draw segments
		for (int i = 0; i < lines.size(); ++i)
		{
			const cv::Vec4f& v = lines[i];
			cv::Point2f b(v[0], v[1]);
			cv::Point2f e(v[2], v[3]);
			cv::line(drawImg, b,e, color, 1);
		}
		return 0;
	}
	catch (const std::exception&)
	{
		return 1;
	}
}
int imgPro::img_drawSegmentsWithLabel(cv::Mat & srcImg, cv::Mat & drawImg, vector<cv::Vec4f> lines)
{
	try
	{
		if (srcImg.channels() == 1)
		{
			drawImg = srcImg.clone();
		}
		else if (srcImg.channels() == 3)
		{
			cvtColor(srcImg, drawImg, cv::COLOR_BGR2GRAY);
		}
		drawImg.convertTo(drawImg, CV_32FC1);

		// Draw segments
		for (int i = 0; i < lines.size(); ++i)
		{
			const cv::Vec4f& v = lines[i];
			cv::Point2f b(v[0], v[1]);
			cv::Point2f e(v[2], v[3]);
			cv::line(drawImg, b, e, cv::Scalar(i+1), 1);
		}
		return 0;
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
int imgPro::img_getPoints(cv::Mat imgIn, vector<cv::Point>& vecPt,int specificVal)
{
	try
	{
		if (imgIn.empty())
		{
			return 1;
		}
		vecPt.clear();
		for (int r = 0; r < imgIn.rows; r++)
		{
			for (int c = 0; c < imgIn.cols; c++)
			{
				if (specificVal == -1)
				{
					if (imgIn.type() == CV_8UC1)
					{
						if (imgIn.at<uchar>(r, c) > 0)
						{
							vecPt.push_back(cv::Point(c, r));
						}
					}
					else if (imgIn.type() == CV_32SC1 || imgIn.type() == CV_16SC1)
					{
						if (imgIn.at<int>(r, c) > 0)
						{
							vecPt.push_back(cv::Point(c, r));
						}

					}
				}
				else
				{
					if (imgIn.type() == CV_8UC1)
					{
						if (imgIn.at<uchar>(r, c) == specificVal)
						{
							vecPt.push_back(cv::Point(c, r));
						}
					}
					else if (imgIn.type() == CV_32SC1 || imgIn.type() == CV_16SC1)
					{
						if (imgIn.at<int>(r, c) == specificVal)
						{
							vecPt.push_back(cv::Point(c, r));
						}

					}

				}
			}
		}
	}
	catch (...)
	{
		return -1;
	}
	return 0;
}

int imgPro::img_getGrayLevel(cv::Mat imgIn, vector<int>& vecGrayLevel, bool includeZero/* = true*/, cv::Mat maskIn /*= cv::Mat()*/)
{
	try
	{
		if (imgIn.empty())
		{
			return 1;
		}
		vecGrayLevel.clear();
		set<int> setVal;
		for (int r = 0; r < imgIn.rows; r++)
		{
			for (int c = 0; c < imgIn.cols; c++)
			{
				if (imgIn.type() == CV_8UC1)
				{
					if(maskIn.empty())
						setVal.insert(imgIn.at<uchar>(r, c));
					else
						if (maskIn.at<uchar>(r,c) >0)
							setVal.insert(imgIn.at<uchar>(r, c));
				}
				else if (imgIn.type() == CV_32SC1 || imgIn.type() == CV_16SC1)
				{
					if (maskIn.empty())
						setVal.insert(imgIn.at<int>(r, c));
					else
						if (maskIn.at<uchar>(r, c) > 0)
							setVal.insert(imgIn.at<int>(r, c));

				}
			}
		}
		if (!includeZero && setVal.find(0) != setVal.end())
		{
			setVal.erase(0);
		}

		vecGrayLevel = vector<int>(setVal.begin(), setVal.end());
	}
	catch (...)
	{
		string err = "imgPro::img_getGrayLevel 中发生错误！";
		return -1;
	}
	return 0;
}


//
///******************************************************************
//iErr,resTable =find_template(strImg,strTemplImg,iMethod, dThreshRes,iRowDistFilter,iColDistFilter,strMaskImg)
//功能：
//	在一副图像中查找模板图像
//限制：
//	脚本调用
//参数：
//strImg			  - 图像输入,用于查找模板
//strImgTempl - 图像模板名称
//iMethod		  - 模板查找方法,0-5共6种
//dThreshRes  - 结果筛选阈值，范围为0-1,数值越大越相似,为1时则取最大值
//iRowDistFilter，iColDistFilter - 间距小于iRowDistFilter或iColDistFilter的结果中，只保留匹配分数大的结果
//strMaskImg  - 掩膜图像,尺寸必须与模板图像相等，可选参数。
//返回值：
//iErr		   - 0, 正常； 非0，有错误
//resTable - 返回table，元素顺序以score值降序排列。
//				形式为{ [1] = {[score] = val1, [row] = val2, [col] = val3 },
//							    [2] = {[score] = val1, [row] = val2, [col] = val3 },
//								 ...}
//***************************************************/
int imgPro::img_findTemplate(cv::Mat &img,cv::Mat &templ,vector<cv::Vec3f> &vecRes,int iMethod,double dThreshRes,int resRowDist,int resColDist,cv::Mat maskImg)
{
	try
	{
		vecRes.clear();
		if (img.empty() || templ.empty())
		{
			string strErr = "imgPro: img_findTemplate 输入图像或模板为空！";
			return 1;

		}
		cv::Mat tempRes;

		int result_cols = img.cols - templ.cols + 1;
		int result_rows = img.rows - templ.rows + 1;		

		tempRes.create(result_rows, result_cols, CV_32FC1);
		bool method_accepts_mask = (cv::TM_SQDIFF == iMethod || iMethod == cv::TM_CCORR_NORMED);

		if (cv::countNonZero(maskImg) > 10 && method_accepts_mask)
		{
			matchTemplate(img, templ, tempRes, iMethod, maskImg);
		}
		else
		{
			matchTemplate(img, templ, tempRes, iMethod);
		}



		if ((dThreshRes-1.0 )<0.001)
		{
			double maxVal,minVal;
			cv::Point maxLoc,minLoc;
			cv::minMaxLoc(tempRes, &minVal,& maxVal, &minLoc, &maxLoc);
			if (iMethod == cv::TM_SQDIFF || iMethod == cv::TM_SQDIFF_NORMED)  //取最小值
			{
				vecRes.push_back(cv::Vec3f(minVal, minLoc.y+templ.rows/2, minLoc.x+templ.cols/2));

			} 
			else
			{
				vecRes.push_back(cv::Vec3f(maxVal, maxLoc.y+templ.rows/2, maxLoc.x+templ.cols/2));
			}

		}
		else
		{
			//cv::normalize(tempRes, tempRes, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
			vecRes = filterTemplateResPoints(tempRes,dThreshRes , resRowDist, resColDist);
			sort(vecRes.begin(), vecRes.end(), [](cv::Vec3f& res1, cv::Vec3f & res2)
																				{
																					return res1[0] > res2[0];
																				});
		}

		return 0;

	}
	catch (...)
	{
		string strErr = "imgPro: img_rotate 捕获到C++异常！";
		return 1;
	}
}



int imgPro::grayImg_sobel(cv::Mat &grayImgIn, cv::Mat &sobelImg)
{
	try
	{
		if (grayImgIn.empty()|| grayImgIn.channels()!=1)
		{
			return 1;
		}
		cv::Mat x, y,resX, resY;
		Sobel(grayImgIn, x, CV_16S, 1, 0);
		Sobel(grayImgIn, y, CV_16S, 0, 1);

		convertScaleAbs(x, resX);
		convertScaleAbs(y, resY);

		addWeighted(resX, 0.5, resY, 0.5, 0, sobelImg);
		return 0;

	}
	catch (...)
	{
		return -1;
	}
}

///**************************************************
//iErr = grayImg_getHist(strIn, strOut)
//功能：
//获取灰度图像的灰度直方图数据
//限制：
//无
//参数：
//strIn - 输入图像
//strOut - 输出数据结果
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
int imgPro::grayImg_getHist(cv::Mat &grayImgIn,cv::Mat &histImgOut)
{
	try
	{
		// 参数检查
		if (grayImgIn.empty()  )
		{
			string strErr = "imgPro: grayImg_getHist 输入图像 ";
			return -1;
		}

		//CWatch w
		int iChannels[] = {0,1};
		int iHistSize = 255;

		float fH_ranges[] = {0, 255};
		const float * ranges[] = {fH_ranges};

		//cv::calcHist(&(*g_pMapImage)[strIn], 1, 0, cv::Mat(), g_MapHist[strOut], 1, &iHistSize, ranges);

		//double d = w.Stop();

	}
	catch (...)
	{
		string strErr = "imgPro: img_copy 捕获到C++异常！";
		return -1;
	}
	return 0;
}
///*******************************************************
//iErr = rgbImg_threshold(strImgIn, strImgOut, dBcoef,dGcoef,dRcoef,iThresh, iMax,iType)
//功能：
//	对彩色图像各像素进行阈值处理，B*dBcoef+G*dGcoef+R*dRcoef > iThresh 将被设置为iMax或0
//限制：
//	彩色图像输入
//参数：
//	strImgIn - 图像输入
//	strImgOut - 图像输出
//	dBcoef,dGcoef,dRcoef - 彩色图像像素点B,G,R对应的运算系数,B*dBcoef+G*dGcoef+R*dRcoef值将于iThresh进行比较。
//	iThresh - 阈值
//	iMax - 最大值
//	iType - 阈值处理类型: 0 : 大于iThresh 像素设置为iMax，否则设置为0, 1 : 大于iThresh 像素设置为0，否则设置为iMax。
//返回值：
//	iErr - 0, 正常； 非0，有错误
//
//***************************************************/
//int imgPro::rgbImg_threshold(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		if (iCount != 8
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER
//			|| lua_type(L, 8) != LUA_TNUMBER)
//
//		{
//			string strErr = "imgPro: rgbImg_threshold 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第2个参数为输出图像
//		double dBcoef = lua_tonumber(L, 3);
//		double dGcoef = lua_tonumber(L, 4);
//		double dRcoef = lua_tonumber(L, 5);
//		int iThresh = (int)lua_tonumber(L, 6);		// 第6个参数为阈值
//		int iMax = (int)lua_tonumber(L, 7);		// 第7个参数是设置的处理后灰度值
//		int iType = (int)lua_tonumber(L, 8);		// 第8个参数是阈值处理类型,
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() || iMax <0 || iMax >255 || iType<0)
//		{
//			string strErr = "imgPro: rgbImg_threshold 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！或输入参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		cv::Mat imgIn = (*g_pMapImage)[strIn];
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat(imgIn.size(), CV_8UC1);
//		}
//
//		cv::Mat imgOut = (*g_pMapImage)[strOut];
//		int nRows = imgIn.rows;
//		int nCols = imgIn.cols*imgIn.channels();
//		if (1==iType)   //大于threshVal,设置为Max
//		{
//			for (int i = 0; i < nRows; i++)
//			{
//				uchar *pImgIn = imgIn.ptr<uchar>(i);
//				uchar *pimgOut = imgOut.ptr<uchar>(i);
//				for (int k = 0, j = 0; j < nCols; j = j + 3, k++)
//				{
//					uchar B = pImgIn[j];
//					uchar G = pImgIn[j + 1];
//					uchar R = pImgIn[j + 2];
//					if (abs(dBcoef*B + G*dGcoef + R*dRcoef )> iThresh)
//					{
//						pimgOut[k] = iMax;
//					}
//					else
//					{
//						pimgOut[k] = 0;
//					}
//				}
//			}
//		} 
//		else 
//		{
//			for(int i = 0; i < nRows; i++)
//			{
//				uchar *pImgIn = imgIn.ptr<uchar>(i);
//				uchar *pimgOut = imgOut.ptr<uchar>(i);
//				for (int k = 0, j = 0; j < nCols; j = j + 3, k++)
//				{
//					uchar B = pImgIn[j];
//					uchar G = pImgIn[j + 1];
//					uchar R = pImgIn[j + 2];
//					if (abs(dBcoef*B + G*dGcoef + R*dRcoef) > iThresh)
//					{
//						pimgOut[k] = 0;
//					}
//					else
//					{
//						pimgOut[k] = iMax;
//					}
//				}
//			}
//		}
//
//		lua_pushinteger(L, 0);
//		return 1;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: rgbImg_threshold 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//}
//
///*******************************************************
//iErr = rgbImg_getColor(strImgIn, strImgOut, iB,iG,iR,iDistB,iDistG,iDistR, iMax,iType)
//功能：
//	提取特定的颜色
//限制：
//	彩色图像输入
//参数：
//	imgIn - 图像输入
//	imgOut - 图像输出
//	iB,iG,iR - 待提取彩色像素点的R,G,B值
//	iDistB,iDistG,iDistR - B,G,R分别对应的动态范围
//	iThresh - 阈值
//	iMax - 最大值
//	iType - 阈值处理类型: 0 : THRESH_BINARY, 1 : THRESH_BINARY_INV,
//返回值：
//	iErr - 0, 正常； 非0，有错误
//
//***************************************************/
//int imgPro::rgbImg_getColor(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		if (iCount != 10
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER
//			|| lua_type(L, 8) != LUA_TNUMBER 
//			|| lua_type(L, 9) != LUA_TNUMBER
//			|| lua_type(L, 10) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: rgbImg_getColor 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第2个参数为输出图像
//		int iB = (int)lua_tonumber(L, 3);
//		int iG = (int)lua_tonumber(L, 4);
//		int iR = (int)lua_tonumber(L, 5);
//		int iDistB = (int)lua_tonumber(L, 6);
//		int iDistG = (int)lua_tonumber(L, 7);
//		int iDistR = (int)lua_tonumber(L, 8);
//		int iMax = (int)lua_tonumber(L, 9);		// 第9个参数是设置的处理后灰度值
//		int iType = (int)lua_tonumber(L,10);		// 第10个参数是阈值处理类型,
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() || iType<-1)
//		{
//			string strErr = "imgPro: rgbImg_getColor 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！或阈值处理类型错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		cv::Mat imgIn = (*g_pMapImage)[strIn];
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat(imgIn.size(), CV_8UC1);
//		}
//
//		cv::Mat imgOut = (*g_pMapImage)[strOut];
//		int nRows = imgIn.rows;
//		int nCols = imgIn.cols*imgIn.channels();
//		if (0 == iType)  
//		{
//			for (int i = 0; i < nRows; i++)
//			{
//				uchar *pImgIn = imgIn.ptr<uchar>(i);
//				uchar *pimgOut = imgOut.ptr<uchar>(i);
//				for (int k = 0, j = 0; j < nCols; j = j + 3, k++)
//				{
//					uchar B = pImgIn[j];
//					uchar G = pImgIn[j + 1];
//					uchar R = pImgIn[j + 2];
//					if ((abs(B - iB)<iDistB) && abs(G - iG)<iDistG &&abs(R-iR)<iDistR)
//					{
//						pimgOut[k] = iMax;
//					}
//					else
//					{
//						pimgOut[k] = 0;
//					}
//				}
//			}
//		}
//		else
//		{
//			for (int i = 0; i < nRows; i++)
//			{
//				uchar *pImgIn = imgIn.ptr<uchar>(i);
//				uchar *pimgOut = imgOut.ptr<uchar>(i);
//				for (int k = 0, j = 0; j < nCols; j = j + 3, k++)
//				{
//					uchar B = pImgIn[j];
//					uchar G = pImgIn[j + 1];
//					uchar R = pImgIn[j + 2];
//					if ((abs(B - iB) < iDistB) && abs(G - iG) < iDistG &&abs(R - iR) < iDistR)
//					{
//						pimgOut[k] = 0;
//					}
//					else
//					{
//						pimgOut[k] = iMax;
//					}
//				}
//			}
//		}
//
//		lua_pushinteger(L, 0);
//		return 1;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: rgbImg_getColor 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//}
//
///**************************************************
//iErr = rgbImg_colorIntensity(imgIn, imgOut, iB,iG ,iR,iType )
//功能：
//	获取图像指定颜色的灰度映射图像，灰度越大颜色越接近
//限制：
//	无
//参数：
//imgIn - 输入图像
//imgOut - 输出图像
//iB,iG,iR - 获取的指定颜色
// iType  -颜色映射类型,0:按偏差绝对值最大值计算；
//
//返回值：
//iErr - 0,正常； 非0，有错误
//***************************************************/
//
//int imgPro::rgbImg_colorIntensity(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//
//		{
//			string strErr = "imgPro: rgbImg_colorIntensity 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		int bians;
//		string strIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 第2个参数为输出图像
//		int iB = (int)lua_tonumber(L, 3);
//		int iG = (int)lua_tonumber(L, 4);
//		int iR = (int)lua_tonumber(L, 5);
//		int iType = (int)lua_tonumber(L, 6);		// 第6个参数是阈值处理类型,
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() 	||iType != 0)
//		{
//			string strErr = "imgPro: rgbImg_colorIntensity 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！或输入参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		cv::Mat imgIn = (*g_pMapImage)[strIn];
//
//		// 如果输出图像缓冲不存在，则创建一个缓冲
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat(imgIn.size(), CV_8UC1);
//		}
//
//		cv::Mat imgOut = (*g_pMapImage)[strOut];
//		int nRows = imgIn.rows;
//		int nCols = imgIn.cols*imgIn.channels();
//
//		if (0 == iType)   //计算偏差值获取像素距离
//		{
//			for (int i = 0; i < nRows; i++)
//			{
//				uchar *pImgIn = imgIn.ptr<uchar>(i);
//				uchar *pimgOut = imgOut.ptr<uchar>(i);
//				for (int k = 0, j = 0; j < nCols; j = j + 3, k++)
//				{
//					uchar B = pImgIn[j];
//					uchar G = pImgIn[j + 1];
//					uchar R = pImgIn[j + 2];
//
//					bians = max(max(abs(B - iB), abs(G - iG)), abs(R - iR));
//					pimgOut[k] = cv::saturate_cast<uchar>(255-bians);
//				}
//			}
//		}
//		lua_pushinteger(L, 0);
//		return 1;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: rgbImg_colorIntensity 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//}
//

// model :0 :从低亮度到高亮度计算比列P或数量，1:从高亮度到低亮度计算比列P

int imgPro::grayImg_thresholdPTile(cv::Mat & grayimgIn, cv::Mat & binImgOut, double p,int model ,bool withZero,int threshType)
{
	if (grayimgIn.empty() || grayimgIn.channels() != 1 || p<0. || p >1.0)
		return -1;

	vector<int> vecPtsOfPiexl(256, 0);
	//grayimgIn.forEach([vecPtsOfPiexl](uchar ptVal) {vecPtsOfPiexl[ptVal]++; });
	for (int i = 0; i < grayimgIn.rows; ++i)
	{
		for(int j = 0; j < grayimgIn.cols; ++j)
		{
			vecPtsOfPiexl[grayimgIn.at<uchar>(i, j)]++;			
		}
	}
	double cntWithZero = grayimgIn.rows*grayimgIn.cols;
	double cntNoZero = cv::countNonZero(grayimgIn);
	int sum = 0;
	int threshVal = 0;
	if ( 1 ==model )
	{
		if (withZero)
		{
			for (int i = vecPtsOfPiexl.size()-1; i >=0; --i)
			{
				sum += vecPtsOfPiexl[i];
				if (double(sum) / cntWithZero >= p )
				{
					threshVal = i-1;
					break;
				}
			}
		}
		else
		{
			for (int i = vecPtsOfPiexl.size()-1; i >0; --i)
			{
				sum += vecPtsOfPiexl[i];
				if (double(sum) / cntNoZero >= p)
				{
					threshVal = i-1;
					break;
				}
			}

		}

	}
	else
	{
		if (withZero)
		{
			for (int i = 0; i < vecPtsOfPiexl.size(); ++i)
			{
				sum += vecPtsOfPiexl[i];
				if (double(sum) / cntWithZero >= p)
				{
					threshVal = i;
					break;
				}
			}
		}
		else
		{
			for (int i = 1; i < vecPtsOfPiexl.size(); ++i)
			{
				sum += vecPtsOfPiexl[i];
				if (double(sum) / cntNoZero >= p)
				{
					threshVal = i;
					break;
				}
			}

		}

	}

	cv::threshold(grayimgIn, binImgOut, threshVal, 255, threshType);	

	return threshVal;
}

//
///**************************************************
//iErr,dSim = grayImg_compareHist(strIn1, strIn2，iMethod)
//功能：
//比较两幅图像的灰度直方图
//限制：
//无
//参数：
//strIn1 - 输入数据1
//strIn2 - 输入数据2
//iMethod - 比较方式(推荐默认为0，0:CV_COMP_CORREL,1:CV_COMP_CHISQR,2:CV_COMP_INTERSECT ,3:CV_COMP_BHATTACHARYYA)
//返回值：
//iErr - 0,正常； 非0，有错误

int imgPro::grayImg_thresholdWithMask(cv::Mat & src, cv::Mat & dst, double thresh, double maxval, int type, const cv::Mat & mask)
{

	if (mask.empty() || (mask.rows == src.rows && mask.cols == src.cols && countNonZero(mask) == src.rows * src.cols))
	{
		// If empty mask, or all-white mask, use cv::threshold
		thresh = cv::threshold(src, dst, thresh, maxval, type);
	}
	else
	{
		// Use mask
		bool use_otsu = (type & cv::ThresholdTypes::THRESH_OTSU) != 0;
		if (use_otsu)
		{
			// If OTSU, get thresh value on mask only
			thresh = otsu_8u_with_mask(src, mask);
			// Remove THRESH_OTSU from type
			type &= cv::ThresholdTypes::THRESH_MASK;
		}

		// Apply cv::threshold on all image
		thresh = cv::threshold(src, dst, thresh, maxval, type);

		// Copy original image on inverted mask
		src.copyTo(dst, ~mask);
	}
	return thresh;

}
double imgPro::grayImg_getMean(const cv::Mat & src,double ratio, double start, double end,cv::Mat mask)
{
	double meanVal = 0.;
	vector<int> vecPtVal;
	try
	{
		if (src.empty() || ratio > src.total())
		{
			return -1;
		}
		for (int r = 0; r < src.rows; r++)
		{
			for (int c = 0; c < src.cols; c++)
			{
				int val = src.at<uchar>(r, c);
				if ( (start <= val && val < end)  || (end <= val && val < start) )
				{
					if (mask.empty())
					{
						vecPtVal.push_back(val);
					} 
					else
					{
						if (mask.at<uchar>(r,c)>0)
						{
							vecPtVal.push_back(val);
						}
					}
				}
			}
		}
		if (start < end)
		{
			sort(vecPtVal.begin(), vecPtVal.end());
		} 
		else
		{
			sort(vecPtVal.begin(), vecPtVal.end(),greater<int>());
		}
		if (ratio <= 1.001)
		{
			int cnt = vecPtVal.size()*ratio;
			meanVal = std::accumulate(vecPtVal.begin(), vecPtVal.begin() + cnt, 0) / double(cnt);
		}
		else
		{
			ratio = (std::min)(int(ratio), int(vecPtVal.size()));
			meanVal = std::accumulate(vecPtVal.begin(), vecPtVal.begin() + ratio, 0) / ratio;

		}
	}
	catch (const std::exception&)
	{
		return -1;
	}

	return meanVal;
}
int imgPro::grayImg_ransacCircle(cv::Mat &grayImgIn,double cannyThresh1,double cannyThresh2 ,Circle &cir)
{
	try
	{

		cv::Mat imgGray = grayImgIn.clone();
		if (grayImgIn.channels() != 1)
		{
			cv::cvtColor(grayImgIn, imgGray, cv::COLOR_BGR2GRAY);
		}
	
		if (cv::countNonZero(imgGray)< 20)
		{
			string strErr = "imgPro: grayImg_ransacCircle 图像包含点数过少！";
			return 1;
		}
		cv::Mat mask;

		cv::Mat canny;
		cv::Canny(imgGray, canny, cannyThresh1, cannyThresh2);

		mask = canny;
		std::vector<cv::Point2f> edgePositions;
		edgePositions = getPointPositions(mask);

		// create distance transform to efficiently evaluate distance to nearest edge
		cv::Mat dt;
		cv::distanceTransform(255 - mask, dt, cv::DistanceTypes::DIST_L1, 3);

		//TODO: maybe seed random variable for real random numbers.

		unsigned int nIterations = 0;

		cv::Point2f bestCircleCenter;
		float bestCircleRadius;
		float bestCirclePercentage = 0;
		float minRadius = 10;   // TODO: ADJUST THIS PARAMETER TO YOUR NEEDS, otherwise smaller circles wont be detected or "small noise circles" will have a high percentage of completion

		//float minCirclePercentage = 0.2f;
		float minCirclePercentage = 0.05f;  // at least 5% of a circle must be present? maybe more...

		int maxNrOfIterations = edgePositions.size();   // TODO: adjust this parameter or include some real ransac criteria with inlier/outlier percentages to decide when to stop

		cv::Point2f center; float radius;
		for (unsigned int its = 0; its < maxNrOfIterations; ++its)
		{
			//RANSAC: randomly choose 3 point and create a circle:
			//TODO: choose randomly but more intelligent, 
			//so that it is more likely to choose three points of a circle. 
			//For example if there are many small circles, it is unlikely to randomly choose 3 points of the same circle.
			unsigned int idx1 = rand() % edgePositions.size();
			unsigned int idx2 = rand() % edgePositions.size();
			unsigned int idx3 = rand() % edgePositions.size();

			// we need 3 different samples:
			if (idx1 == idx2) continue;
			if (idx1 == idx3) continue;
			if (idx3 == idx2) continue;

			// create circle from 3 points:
			//cv::Point2f center; float radius;
			getCircle(edgePositions[idx1], edgePositions[idx2], edgePositions[idx3], center, radius);

			// inlier set unused at the moment but could be used to approximate a (more robust) circle from alle inlier
			std::vector<cv::Point2f> inlierSet;

			//verify or falsify the circle by inlier counting:
			float cPerc = verifyCircle(dt, center, radius, inlierSet);

			// update best circle information if necessary
			if (cPerc >= bestCirclePercentage)
			if (radius >= minRadius)
			{
				bestCirclePercentage = cPerc;
				bestCircleRadius = radius;
				bestCircleCenter = center;
			}
		}
		
		cir.x = bestCircleCenter.x;
		cir.y = bestCircleCenter.y;
		cir.radius = bestCircleRadius;
		return 0;
	}
	

	catch (...)
	{
		string strErr = "imgPro: grayImg_ransacCircle 捕获到C++异常！";
		return 1;
	}
}



/**************************************************
iErr = biImg_filterByArea(strImgIn, strImgOut,iAreaThreshVal,iMode)
功能：
使用面积大小筛选图像
限制：
无
参数：
strImgIn - 输入图像
strImgOut - 输出图像
iAreaThreshVal - 面积阈值
iMode - 处理模式,1 - 阈值之间设为0，2 - 阈值之外设为0
返回值：
iErr - 0,正常； 非0，有错误
***************************************************/
int imgPro::biImg_filterByArea( cv::Mat &imgIn, cv::Mat &imgOut, int iAreaThreshLow, int iAreaThreshHigh, int iMode, int connection)
{
	try
	{
		if (imgIn.empty())
		{
			string strErr = "imgPro: grayImg_filterByArea 输入图像错误！";
			return -1;
		}
		if (imgOut.empty())
		{
			imgOut.create(imgIn.size(), CV_8UC1);
			imgOut.setTo(0);
		}
		cv::Mat labels, status, centroids;

		int labelCnt = cv::connectedComponentsWithStats(imgIn, labels, status, centroids, connection);

		vector<int> colors(labelCnt, 255);
		colors[0] = 0;
		for (int i = 1; i < labelCnt; i++)
		{
			if (1 == iMode)   //阈值之间设为0
			{
				if ( iAreaThreshLow < status.at<int>(i, cv::CC_STAT_AREA) && status.at<int>(i, cv::CC_STAT_AREA) < iAreaThreshHigh)
				{
					colors[i] = 0;
				}
			}
			else
			{
				if (  status.at<int>(i, cv::CC_STAT_AREA) < iAreaThreshLow || iAreaThreshHigh <status.at<int>(i, cv::CC_STAT_AREA))
				{
					colors[i] = 0;
				}
			}
		}
		for (int i = 0; i < imgIn.rows; i++)
		{
			for (int j = 0; j < imgIn.cols; j++)
			{
				int label = labels.at<int>(i, j);
				imgOut.at<uchar>(i, j) = colors[label];
			}
		}

		return 1;
	}
	catch (...)
	{
		string strErr = "imgPro: grayImg_filterByArea 捕获到C++异常！";
		return -1;
	}
}

int imgPro::biImg_delRegionOnboundary(cv::Mat & imgIn, cv::Mat & imgOut, int iMode, int connection)
{
	try
	{
		if (imgIn.empty())
		{
			return -1;
		}
		imgOut = imgIn.clone();
		const int nr = imgIn.rows;
		const int nc = imgIn.cols;
		cv::Mat edge[4];
		
		edge[0] = imgIn.row(0);    //up
		edge[1] = imgIn.row(nr - 1); //bottom
		edge[2] = imgIn.col(0);    //left
		edge[3] = imgIn.col(nc - 1); //right

		std::vector<cv::Point> edgePts;
		const int minLength = std::min(nr, nc) / 4;
		for (int i = 0; i < 4; ++i)
		{
			std::vector<cv::Point> line;
		    cv::Mat_<uchar>::const_iterator iter = edge[i].begin<uchar>();       //当前像素
			cv::Mat_<uchar>::const_iterator nextIter = edge[i].begin<uchar>() + 1; //下一个像素
			while (nextIter != edge[i].end<uchar>())
			{
				if (*iter == 255)
				{
					if (*nextIter == 255)
					{
						cv::Point pt = iter.pos();
						if (i == 1)
							pt.y = nr - 1;
						if (i == 3)
							pt.x = nc - 1;

						edgePts.push_back(pt);
					}
				}
				++iter;
				++nextIter;
			}
		}

		for (int n = 0; n < edgePts.size(); ++n)
			if (imgOut.at<uchar>(edgePts[n]) == 0)
				continue;
			else
				floodFill(imgOut, edgePts[n], 0,0,0,0,8);//漫水填充法

	}
	catch (...)
	{
		return -1;
	}

	return 0;
}
/*

*/
//iMode:0:
int imgPro::biImg_delMaxArea(cv::Mat &imgIn, cv::Mat &imgOut)
{
	try
	{
		if (imgIn.empty() || imgIn.total() < FILTER_AREA_MIN_NUM)
		{
			string strErr = "imgPro: biImg_delMaxArea 输入图像错误！";
			return -1;
		}
		imgOut.create(imgIn.size(), CV_8UC1);
		imgOut.setTo(0);
		cv::Mat labels, status, centroids;

		int labelCnt = cv::connectedComponentsWithStats(imgIn, labels, status, centroids);

		vector<int> colors(labelCnt, 255);
		colors[0] = 0;
		int maxAreaLabel = -1;
		int maxArea = -1;
		for (int i = 1; i < labelCnt; i++)
		{
			if (status.at<int>(i, cv::CC_STAT_AREA) > maxArea) //找到最大值区域
			{
				maxArea = status.at<int>(i, cv::CC_STAT_AREA);
				maxAreaLabel = i;
			}
		}
		if (maxAreaLabel >0)
		{
			colors[maxAreaLabel] = 0;   //最大值区域赋值0
		}
		if (labelCnt > 1)
		{
			for (int i = 0; i < imgIn.rows; i++)
			{
				for (int j = 0; j < imgIn.cols; j++)
				{
					int label = labels.at<int>(i, j);
					imgOut.at<uchar>(i, j) = colors[label];
				}
			}
		}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: grayImg_filterByArea 捕获到C++异常！";
		return -1;
	}
}
//iMode ==0,阈值之间设置为0， =1，阈值之外设置为0
int imgPro::biImg_getMaxArea(cv::Mat & imgIn, cv::Mat & imgOut)
{
	try
	{
		if (imgIn.empty() || imgIn.total() < FILTER_AREA_MIN_NUM)
		{
			string strErr = "imgPro: biImg_delMaxArea 输入图像错误！";
			return -1;
		}
		imgOut.create(imgIn.size(), CV_8UC1);
		imgOut.setTo(0);
		cv::Mat labels, status, centroids;

		int labelCnt = cv::connectedComponentsWithStats(imgIn, labels, status, centroids);

		vector<int> colors(labelCnt, 0);
		colors[0] = 0;
		int maxAreaLabel = -1;
		int maxArea = -1;
		for (int i = 1; i < labelCnt; i++)
		{
			if (status.at<int>(i, cv::CC_STAT_AREA) > maxArea) //找到最大值区域
			{
				maxArea = status.at<int>(i, cv::CC_STAT_AREA);
				maxAreaLabel = i;
			}

		}
		if (maxAreaLabel > 0)
		{
			colors[maxAreaLabel] = 255;   //最大值区域赋值255
		}
		if (labelCnt > 1)
		{
			for (int i = 0; i < imgIn.rows; i++)
			{
				for (int j = 0; j < imgIn.cols; j++)
				{
					int label = labels.at<int>(i, j);
					imgOut.at<uchar>(i, j) = colors[label];
				}
			}
		}

		return 1;
	}
	catch (...)
	{
		string strErr = "imgPro: grayImg_filterByArea 捕获到C++异常！";
		return -1;
	}

}

/**************************************************
iErr = biImg_thinImg(imgIn,imgOut,iter)
功能：
对图像中区域提取骨架
限制：
无
参数：
imgIn - 输入图像，该函数不改变原始图像
返回值：
iErr            - 0, 正常； 非0，有错误
***************************************************/
int imgPro::biImg_thinImg(cv::Mat imgIn, cv::Mat &imgOut, int maxIterations)
{
    try
    {
		if (imgIn.channels() > 1 || cv::countNonZero(imgIn) < IMG_MIN_NUM)
		{
			string strErr = "imgPro: biImg_thinImg 输入图像错误！";
			return 1;
		}

		cv::Mat dst;
		cv::threshold(imgIn, dst, 0, 1, cv::THRESH_BINARY);

		int width = imgIn.cols;
		int height = imgIn.rows;
		//src.copyTo(dst);
		int count = 0;  //记录迭代次数  
		while (true)
		{
			count++;
			if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达  
				break;
			std::vector<uchar*> mFlag; //用于标记需要删除的点  
			//对点标记  
			for (int i = 0; i < height; ++i)
			{
				uchar* p = dst.ptr<uchar>(i);
				for (int j = 0; j < width; ++j)
				{
					//如果满足四个条件，进行标记  
					//  p9 p2 p3  
					//  p8 p1 p4  
					//  p7 p6 p5  
					uchar p1 = p[j];
					if (p1 != 1) continue;
					uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
					uchar p8 = (j == 0) ? 0 : *(p + j - 1);
					uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
					uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
					uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
					uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
					uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
					uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
					{
						int ap = 0;
						if (p2 == 0 && p3 == 1) ++ap;
						if (p3 == 0 && p4 == 1) ++ap;
						if (p4 == 0 && p5 == 1) ++ap;
						if (p5 == 0 && p6 == 1) ++ap;
						if (p6 == 0 && p7 == 1) ++ap;
						if (p7 == 0 && p8 == 1) ++ap;
						if (p8 == 0 && p9 == 1) ++ap;
						if (p9 == 0 && p2 == 1) ++ap;

						if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
						{
							//标记  
							mFlag.push_back(p + j);
						}
					}
				}
			}

			//将标记的点删除  
			for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
			{
				**i = 0;
			}

			//直到没有点满足，算法结束  
			if (mFlag.empty())
			{
				break;
			}
			else
			{
				mFlag.clear();//将mFlag清空  
			}

			//对点标记  
			for (int i = 0; i < height; ++i)
			{
				uchar* p = dst.ptr<uchar>(i);
				for (int j = 0; j < width; ++j)
				{
					//如果满足四个条件，进行标记  
					//  p9 p2 p3  
					//  p8 p1 p4  
					//  p7 p6 p5  
					uchar p1 = p[j];
					if (p1 != 1) continue;
					uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
					uchar p8 = (j == 0) ? 0 : *(p + j - 1);
					uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
					uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
					uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
					uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
					uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
					uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
					{
						int ap = 0;
						if (p2 == 0 && p3 == 1) ++ap;
						if (p3 == 0 && p4 == 1) ++ap;
						if (p4 == 0 && p5 == 1) ++ap;
						if (p5 == 0 && p6 == 1) ++ap;
						if (p6 == 0 && p7 == 1) ++ap;
						if (p7 == 0 && p8 == 1) ++ap;
						if (p8 == 0 && p9 == 1) ++ap;
						if (p9 == 0 && p2 == 1) ++ap;

						if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
						{
							//标记  
							mFlag.push_back(p + j);
						}
					}
				}
			}

			//将标记的点删除  
			for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
			{
				**i = 0;
			}

			//直到没有点满足，算法结束  
			if (mFlag.empty())
			{
				break;
			}
			else
			{
				mFlag.clear();//将mFlag清空  
			}
		}
		imgOut =  dst * 255;

		return 0;
	}
    catch (...)
    {
        string strErr = "imgPro: biImg_thinImg 捕获到C++异常！";
        return -1;
    }

}

/**************************************************
iErr = biImg_fillup(imgIn, imgOut, iType)
功能：
二值图像孔洞填充
限制：
无
参数：
imgIn - 输入图像
imgOut - 输出图像
iType - 填充方法的选择.1:完全填充检测到的外轮廓，2：填充阈值之间的区域
***************************************************/
int imgPro::biImg_fillup(const cv::Mat &imgIn, cv::Mat &imgOut, int iType,int fillAreaLow,int fillAreaHigh)
{
    try
    {
        if (imgIn.empty() ||imgIn.total()<IMG_MIN_NUM)
        {
            string strErr = "imgPro: region_Dilation 输入图像 ";
            strErr += " 不存在！";
            return -1;
        }
		if (imgOut.empty())
		{
			imgOut.create(imgIn.size(), CV_8UC1);
		}
        if (imgIn.type() != CV_8UC1)
        {
            string strErr = "imgPro: biImg_fillup 输入图像不是二值图像！";
            return -1;
        }
		if (iType==1)
		{
			vector<vector<cv::Point>>  conts;
			cv::findContours(imgIn, conts, cv::RETR_EXTERNAL,cv:: CHAIN_APPROX_NONE);
			imgOut.setTo(0);
			cv::drawContours(imgOut, conts, -1, cv::Scalar(255), cv::FILLED);
		}
		else if(iType ==2)
		{
			cv::Mat notImg,filterImg,andImg;
			cv::bitwise_not(imgIn, notImg);
			biImg_filterByArea(notImg, filterImg, fillAreaLow, fillAreaHigh, 2,4);
			cv::bitwise_or(imgIn, filterImg, imgOut);
				
		}
        
        return 0;
    }
    catch (...)
    {
        string strErr = "imgPro: biImg_fillup 捕获到C++异常！";
        return -1;
    }

}


///**************************************************
//iErr, iNum = region_houghLines(imgIn, dRho, dTheta, iThre, dR, dT)
//功能：
//查找图片中的直线
//限制：
//无
//参数：
//imgIn - 输入图像
//dRho - 搜索步长精度（单位：像素）
//dTheta - 搜索角度精度（单位：弧度）
//iThre - 阈值
//dR - 直线合并的宽度
//dT - 直线合并的角度, dR、dT都为0时，不合并
//返回值：
//iErr - 0,正常； 非0，有错误
//iNum - 直线个数
//***************************************************/
//int imgPro::biImg_houghLines(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: region_HoughLines 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		string imgIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		double dRho = lua_tonumber(L, 2);   // 搜索步长精度（单位：像素）
//		double dTheta = lua_tonumber(L, 3)*PI/180; // 搜索角度精度（转换为弧度）
//		int iThre = (int)lua_tonumber(L, 4);
//		double dR = lua_tonumber(L, 5);        // 直线合并的宽度
//		double dT = lua_tonumber(L, 6)*PI/180; // 直线合并的角度, dR、dT都为0时，不合并
//
//		// 参数检查
//		if (g_pMapImage->find(imgIn) == g_pMapImage->end() || (*g_pMapImage)[imgIn].total()<IMG_MIN_NUM)
//		{
//			string strErr = "imgPro: region_HoughLines 输入图像 ";
//			strErr += imgIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		int iWidth = (*g_pMapImage)[imgIn].cols;
//		int iHeight = (*g_pMapImage)[imgIn].rows;
//		vector<cv::Vec2f> lines;
// 		cv::HoughLines((*g_pMapImage)[imgIn], lines, dRho, dTheta, iThre, 0, 0);
// 		if (!bZero(dR) || !bZero(dT)) optimizeLines(lines, dR, dT);
//
//		g_linesP.clear();
//		for (vector<cv::Vec2f>::iterator itor = lines.begin();
//			itor != lines.end(); itor++)
//		{
//			double dP1Row, dP1Col, dP2Row, dP2Col;
//			getLinePoints((*itor)[0], (*itor)[1], iWidth, iHeight, dP1Row, dP1Col, dP2Row, dP2Col);
//			cv::Vec4d v;
//			v[0] = dP1Col;
//			v[1] = dP1Row;
//			v[2] = dP2Col;
//			v[3] = dP2Row;
//			g_linesP.push_back(v);
//		}
//
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, g_linesP.size());
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: region_HoughLines 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//	}
//	return 2;
//}
//
///**************************************************
//iErr，iNum = region_houghLinesP(imgIn, dRho, dTheta, iThre, minLineLength, maxLineGap)
//功能：
//查找图片中的线段
//限制：
//无
//参数：
//imgIn - 输入图像
//dRho - 搜索步长精度（单位：像素）
//dTheta - 搜索角度精度（单位：弧度）
//iThre - 阈值
//minLineLength - 线段最小长度
//maxLineGap - 最大直线间隙
//返回值：
//iErr - 0,正常； 非0，有错误
//iNum - 直线的数量
//***************************************************/
//int imgPro::biImg_houghLinesP(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: region_HoughLinesP 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		double dRho = lua_tonumber(L, 2);	// 搜索步长精度
//		double dTheta = lua_tonumber(L, 3)*PI / 180;  // 搜索角度精度
//		int iThre = (int)lua_tonumber(L, 4); //
//		double minLineLength = lua_tonumber(L, 5);
//		double maxLineGap = lua_tonumber(L, 6);
//
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: region_HoughLinesP 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		g_linesP.clear();
//
//		cv::HoughLinesP((*g_pMapImage)[strIn], g_linesP, dRho, dTheta, iThre, minLineLength, maxLineGap);
//
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, g_linesP.size());
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: region_HoughLinesP 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//	}
//	return 2;
//}
//
///**************************************************
//iErr = biImg_getEdge(imgIn,edgeImgOut,strEdgeContOut,iDirection,iProStart,iStartPos,iEndPos，iDistOffset,iIsConnected)
//功能：
//	 从图像的某一方向获取该方向的边缘，如从上至下，从下至上，从左往右，从右往左
//限制：
//无
//参数：
//	imgIn - 输入图像
//	edgeImgOut - 输出的边缘图像
//	strEdgeContOut - 输出的边缘轮廓
//	iDirection - 获取边缘的方向 0：从上往下 ，1：从下往上 ，2：从左往右 ，3：从右往左
//    iProStart - 开始投影的位置
//	iStartPos - 开始获取边缘位置
//	iEndPos   - 结束获取边缘位置
//	iDistOffset - 设置找到的边缘点与上一个的偏移范围，超过该阈值则断开连接
//	iIsConnected - 是否将这些边缘点连接起来，除了超过iDistOffset的点，其他断开的边缘点将用直线连接
//返回值：
//	iErr - 0,正常； 非0，有错误
//***************************************************/
//
//int imgPro::biImg_getEdge(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//
//		if (iCount != 9
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TSTRING
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER
//			|| lua_type(L, 8) != LUA_TNUMBER
//			|| lua_type(L, 9) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: biImg_getEdge 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strImgIn = lua_tostring(L, 1);
//		string strEdgeImgOut = lua_tostring(L, 2);
//		string strEdgeContOut = lua_tostring(L, 3);
//		int iOrient = int(lua_tonumber(L, 4));
//		int iProStart = int(lua_tonumber(L, 5));
//		int startPos = int(lua_tonumber(L, 6));
//		int endPos = int(lua_tonumber(L, 7));
//		int iDistOffset = int(lua_tonumber(L, 8));
//		int iIsConnected = int(lua_tonumber(L, 9));
//
//		if (g_pMapImage->find(strImgIn) == g_pMapImage->end() || iOrient<0 || iOrient>3)
//		{
//			string strErr = "imgPro: biImg_getEdge 中输入图像不存在或提取方向错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		if (g_pMapImage->find(strEdgeImgOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strEdgeImgOut] = cv::Mat();
//		}
//
//		if (mapProfile.find(strEdgeContOut) == mapProfile.end())
//		{
//			(mapProfile)[strEdgeContOut] = vector<cv::Point2d>();
//		}
//		mapProfile[strEdgeContOut].clear();
//
//		const cv::Mat& imgIn = (*g_pMapImage)[strImgIn];
//		cv::Mat& edgeImgOut = (*g_pMapImage)[strEdgeImgOut];
//		edgeImgOut.create(imgIn.size(), CV_8UC1);
//		edgeImgOut.setTo(0);
//		if (iOrient == 1 || iOrient == 0)
//		{
//			if (startPos< 0 || endPos >imgIn.cols - 1 || startPos >endPos)
//			{
//				string strErr = "imgPro: biImg_getEdge 中输入提取范围错误！";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//				lua_pushinteger(L, -100);
//				return 1;
//			}
//		}
//		else
//		{
//			if (startPos< 0 || endPos >imgIn.rows - 1 || startPos >endPos)
//			{
//				string strErr = "imgPro: biImg_getEdge 中输入提取范围错误！";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//				lua_pushinteger(L, -100);
//				return 1;
//			}
//		}
//		vector<cv::Point> edge;
//		const int IMG_WIDTH = imgIn.cols;
//		const int IMG_HEIGHT = imgIn.rows;
//
//		switch (iOrient)
//		{
//		case 0://从上往下
//			for (int x = startPos; x < endPos; x++)
//			{
//				for (int y = iProStart; y < IMG_HEIGHT; y++)
//				{
//					if (imgIn.at<uchar>(y, x)>0)
//					{
//						edge.push_back(cv::Point(x, y));
//						break;
//					}
//				}
//			}
//			break;
//		case 1://从下往上
//			for (int x = startPos; x < endPos; x++)
//			{
//				for (int y = iProStart; y >= 0; y--)
//				{
//					if (imgIn.at<uchar>(y, x)>0)
//					{
//						edge.push_back(cv::Point(x, y));
//						break;
//					}
//				}
//			}
//			break;
//		case 2:   //从左往右
//			for (int r = startPos; r < endPos; r++)
//			{
//				for (int c = iProStart; c <IMG_WIDTH; c++)
//				{
//					if (imgIn.at<uchar>(r, c) > 0)
//					{
//						edge.push_back(cv::Point(c, r));
//						break;
//					}
//				}
//			}
//			break;
//		case 3: //从右往左
//			for (int r = startPos; r < endPos; r++)
//			{
//				for (int c = iProStart; c >= 0; c--)
//				{
//					if (imgIn.at<uchar>(r, c) > 0)
//					{
//						edge.push_back(cv::Point(c,r));
//						break;
//					}
//				}
//			}
//			break;
//		default:
//			break;
//		}
//		if (edge.size() <10)
//		{
//			string strErr = "imgPro: biImg_getEdge 拟合点数量小于10！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//
//		}
//				
//
//		for (size_t i = 0; i < edge.size(); i++)
//		{
//			cv::Point2d pt;
//			pt.x = edge[i].x;
//			pt.y = edge[i].y;
//			mapProfile[strEdgeContOut].push_back(pt);
//		}
//
//		std::vector<vector<cv::Point2d>> conts;
//		splitProfile(mapProfile[strEdgeContOut], conts, 12);
//		mapProfile[strEdgeContOut].clear();
//		for (size_t i = 0; i < conts.size(); i++)
//		{
//			if (conts[i].size() > 10)
//			{
//				mapProfile[strEdgeContOut].insert(mapProfile[strEdgeContOut].end(), conts[i].begin(), conts[i].end());
//			}
//			
//		}
//		vector<cv::Point2d> temp = mapProfile[strEdgeContOut];
//		for (size_t i = 0; i < mapProfile[strEdgeContOut].size() - 1; i++)
//		{
//			if (abs(mapProfile[strEdgeContOut][i].x - mapProfile[strEdgeContOut][i + 1].x) + abs(mapProfile[strEdgeContOut][i].y - mapProfile[strEdgeContOut][i + 1].y) < iDistOffset)  //街区距离判断点的距离
//			{
//				cv::line(edgeImgOut, mapProfile[strEdgeContOut][i], mapProfile[strEdgeContOut][i + 1], cv::Scalar(255, 255, 255), 1);
//			}
//		}
//
//		lua_pushinteger(L, 0);
//		return 1;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: biImg_getEdge 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//}
//
//
//
///**************************************************
//iErr,centRow,centCol,dAngle,width,height = biImg_getRotRect(imgIn)
//功能：
//	检测二值图像所有区域的的外接旋转矩形
//限制：
//无
//参数：
//imgIn - 输入图像
//
//返回值：
//iErr - 0,正常； 非0，有错误
//others -旋转矩形的参数
//***************************************************/

int imgPro::biImg_getRotRect(cv::Mat &imgIn,cv::RotatedRect & rotRect)
{
	try
	{

			//输入为点集
		int cnt = countNonZero(imgIn);
		vector<cv::Point> pts(cnt);  //二值图像中的所有点
		int i = 0;
		for (int row = 0; row < imgIn.rows; row++)
		{
			for (int col = 0; col < imgIn.cols; col++)
			{
				if (imgIn.at<uchar>(row, col) >0)
					pts[i++] = (cv::Point(col, row));
			}
		}

		rotRect = cv::minAreaRect(pts);		

		return 1;
		
	}
	catch(cv::Exception e)
	{
		string str = e.err;
		return -1;

	}
}
int imgPro::biImg_getBoundingRect(cv::Mat &imgIn,cv::Rect & rect)
{
	try
	{

			//输入为点集
		int cnt = countNonZero(imgIn);
		vector<cv::Point> pts(cnt);  //二值图像中的所有点
		int i = 0;
		for (int row = 0; row < imgIn.rows; row++)
		{
			for (int col = 0; col < imgIn.cols; col++)
			{
				if (imgIn.at<uchar>(row, col) >0)
					pts[i++] = (cv::Point(col, row));
			}
		}

		rect = cv::boundingRect(pts);		

		return 1;
		
	}
	catch(cv::Exception e)
	{
		string str = e.err;
		return -1;

	}
}

std::vector<cv::Point> imgPro::biImg_getBoundingRectPts(cv::Rect & rect)
{
	try
	{
		if (rect.empty())
		{
			return std::vector<cv::Point>();
		}
		else
		{
			return vector<cv::Point>{rect.tl(),cv::Point(rect.tl().x+rect.width,rect.tl().y),
									rect.br(),cv::Point(rect.tl().x,rect.tl().y+rect.height)
			};
		}
	}
	catch (const std::exception&)
	{
		return std::vector<cv::Point>();
	}
}

///**************************************************
//iErr,iNum = biImg_houghCircles(imgIn,iMinDis,iMinR,iMaxR)
//功能：
//霍夫圆检测
//限制：
//无
//参数：
//imgIn - 输入图像
//iMinDis - 最小距离
//iMinR - 最小半径
//iMinR - 最大半径
//返回值：
//iErr - 0,正常； 非0，有错误
//iNum - 检测出圆的个数
//***************************************************/

int imgPro::biImg_houghCircles(cv::Mat &imgIn,vector<cv::Vec3f> &circles,int iMinDis,int iMinR,int iMaxR)
{
	try
	{
		// 参数检查
		if (imgIn.empty() || imgIn.channels() != 1)
		{
			string strErr = "imgPro: region_houghCircles 输入图像 ";
			strErr += " 不存在！";
			return -1;
		}

		circles.clear();
		//cv::Mat g;
		//GaussianBlur( mIn, g, Size(5, 5), 2, 2 );
		cv::HoughCircles(imgIn, circles, cv::HOUGH_GRADIENT, 1, iMinDis, 100, 100, iMinR, iMaxR);

	}
	catch (...)
	{
		string strErr = "imgPro: region_houghCircles 捕获到C++异常！";
		return -1;
	}

	return 1;
}

int imgPro::img_showHoughCircles(cv::Mat &imgIn, cv::Mat &imgOut, vector<cv::Vec3f> circles)
{
	if (imgIn.empty())
	{
		return -1;
	}
	imgIn.copyTo(imgOut);
	for (size_t i = 0; i < circles.size(); i++)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		cv::circle(imgOut, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		cv::circle(imgOut, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
	}
	return 1;
}

/**************************************************
iErr,iCnt = biImg_createRegion(strImgIn,strLabelImgOut,iAreaThrehold)
功能：
分割出二值图像的区域并计算各自特征
限制：
无
参数：
strImgIn               - 输入的图像，
features          - 输出的标记图像及其对应的特征，特征为map<string,vector<double>>,string 包括area,,,
iAreaThrehold    - 面积筛选阈值，大于该阈值的显示
返回值：
iErr  - 0,正常； 非0，有错误
iCnt  - 分割出二值区域的个数

***************************************************/
int imgPro::biImg_createRegion(const cv::Mat &imgIn, cv::Mat &imgLabel, map<string, vector<double>> &features, int iAreaThreshLow, int iAreaThreshHigh)
{
	try
	{
		if (imgIn.empty() || imgIn.total() < FILTER_AREA_MIN_NUM)
		{
			string strErr = "imgPro: biImg_createRegion 输入图像 ";
			strErr += " 不存在或为空白";
			return -1;
		}
		cv::Mat  status, centroids;
		int validCnt = 0;
		features.clear();
		int labelCnt = cv::connectedComponentsWithStats(imgIn, imgLabel, status, centroids);
		for (int cnt = 1; cnt < labelCnt; cnt++)//不考虑背景
		{
			if (iAreaThreshLow < status.at<int>(cnt, cv::CC_STAT_AREA) && status.at<int>(cnt, cv::CC_STAT_AREA) < iAreaThreshHigh)
			{
				validCnt++;
			}
		}
		//空间换时间
		map<int, vector<cv::Point>> mapRegionPts;
		for (int r=0;r<imgIn.rows;r++)
		{
			for (int c = 0; c < imgIn.cols; c++)
			{
				mapRegionPts[imgLabel.at<int>(r, c)].push_back(cv::Point(c,r));
			}
		}
		//将status和centorid中的值放到map中方便管理，i=0为背景
		features["label"].assign(labelCnt, INT_MIN);
		features["left"].assign(labelCnt, INT_MIN);
		features["top"].assign(labelCnt, INT_MIN);
		features["width"].assign(labelCnt, INT_MIN);
		features["height"].assign(labelCnt, INT_MIN);
		features["area"].assign(labelCnt, INT_MIN);
		features["row"].assign(labelCnt, INT_MIN);
		features["col"].assign(labelCnt, INT_MIN);
		features["polyPts"].assign(labelCnt, INT_MIN);
		features["circle"].assign(labelCnt, INT_MIN);//
		features["rect"].assign(labelCnt, INT_MIN);
		features["rotW"].assign(labelCnt, INT_MIN);
		features["rotH"].assign(labelCnt, INT_MIN);
		features["rotA"].assign(labelCnt, INT_MIN);
		features["rotWidth"].assign(labelCnt, INT_MIN);
		features["rotHeight"].assign(labelCnt, INT_MIN);
		features["rotHWRatio"].assign(labelCnt, INT_MIN);
		features["rotRecArea"].assign(labelCnt, INT_MIN);
		features["rotAngle"].assign(labelCnt, INT_MIN);

		int procsCnt = omp_get_num_procs();
#pragma omp parallel for num_threads(procsCnt-2)
		for (int i = 1; i < labelCnt; i++)
		{
			if (iAreaThreshLow < status.at<int>(i, cv::CC_STAT_AREA) && status.at<int>(i, cv::CC_STAT_AREA) < iAreaThreshHigh)
			{
				features["label"][i] = i;
				features["left"][i] = (status.at<int>(i, cv::CC_STAT_LEFT));  //插入特征，不包括背景，vector索引从0开始。
				features["top"][i] = (status.at<int>(i, cv::CC_STAT_TOP));
				features["width"][i] = (status.at<int>(i, cv::CC_STAT_WIDTH));
				features["height"][i] = (status.at<int>(i, cv::CC_STAT_HEIGHT));
				features["area"][i] = (status.at<int>(i, cv::CC_STAT_AREA));
				features["row"][i] = (centroids.at<double>(i, 1));    // row of center
				features["col"][i] = (centroids.at<double>(i, 0));    // col of center
				vector<vector<cv::Point>> conts;
				vector<cv::Point> cont;
				vector<cv::Point> regionTemp;
				cv::Mat tempROI,labelImgAll;
				tempROI.create(status.at<int>(i, cv::CC_STAT_HEIGHT), status.at<int>(i, cv::CC_STAT_WIDTH), CV_8UC1);
				tempROI.setTo(0);
				regionTemp.clear();
				regionTemp = mapRegionPts[i];
				int left = status.at<int>(i, cv::CC_STAT_LEFT);
				int top = status.at<int>(i, cv::CC_STAT_TOP);
				for (cv::Point pt : regionTemp)
				{
					tempROI.at<uchar>(pt - cv::Point(left, top)) = 255;
				}
				cv::findContours(tempROI, conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE,
					cv::Point(status.at<int>(i, cv::CC_STAT_LEFT), status.at<int>(i, cv::CC_STAT_TOP)));
				cont = conts[0];

				//插入逼近轮廓的点数量
				vector<cv::Point> polyPts;
				cv::approxPolyDP(cont, polyPts, 8, true);
				cv::Mat show = tempROI.clone();
				//show.convertTo(show, CV_8UC3);
				//cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
				//cv::polylines(show, polyPts, true, cv::Scalar(0, 0, 255), 2);
				features["polyPts"][i] = (polyPts.size());
				//插入圆相似度特征
				//double likeCircle = 4 * CV_PI*status.at<int>(i, cv::CC_STAT_AREA) / (cont.size()*cont.size());
				double likeCircle = 4 * CV_PI*cv::contourArea(cont) / (pow(cv::arcLength(cont, true), 2));

				features["circle"][i] = (likeCircle);    // 
															//插入最小旋转矩形的长宽比
				float HWRatio = 0.;
				cv::RotatedRect roRect = minAreaRect(cont);//regionTemp
				double likeRect = status.at<int>(i, cv::CC_STAT_AREA) / roRect.size.area();
				features["rect"][i] = (likeRect);    //

				features["rotW"][i] = (roRect.size.width);
				features["rotH"][i] = (roRect.size.height);
				features["rotA"][i] = (roRect.angle);

				if (roRect.size.height > roRect.size.width)
				{
					features["rotWidth"][i] = (roRect.size.height);  //插入最小旋转矩形的长边
					features["rotHeight"][i] = (roRect.size.width);  //插入最小旋转矩形的边
					HWRatio = roRect.size.height / roRect.size.width;
				}
				else
				{
					features["rotWidth"][i] = (roRect.size.width);
					features["rotHeight"][i] = (roRect.size.height);
					HWRatio = roRect.size.width / roRect.size.height;
				}

				features["rotHWRatio"][i] = (HWRatio);    // 9															 
				features["rotRecArea"][i] = (roRect.size.area()); //插外接旋转矩形的面积
				vector<cv::Point2f> pts;
				float angle;
				rect_getRotRectPts(roRect, pts, angle);
				features["rotAngle"][i] = (angle);    //   矩形角度		
			}
		}
		for (auto &pairItem : features)
		{
			pairItem.second.erase(remove_if(pairItem.second.begin(), pairItem.second.end(), [](auto it1) {return it1 < INT_MIN + 2; }),
				pairItem.second.end());
		}
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: biImg_createRegion 捕获到C++异常！";
		return -1;
	}
}



/**************************************************
iErr = region_toImg(strLabelImg,strImgOut,iIndex)
功能：
按标记特征 向量索引输出区域的特征
限制：
无
参数：
strLabelImg        - 输入的标记图像，INT32型
strImgOut           - 输出的图像，CV_8UC1类型，标记区域为255，背景区域为0
iIndex                  - 输入的索引，从1开始
返回值：
iErr  - 0,正常； 非0，有错误

***************************************************/
int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, int index)
{
	try
	{

		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImg 输入图像 ";
			strErr += " 不存在或";
			return -1;
		}
		double minVal, maxVal;
		//cv::minMaxIdx(labelImg, &minVal, &maxVal);
		//if (maxVal < index)
		//{
		//	return -1;
		//}
		if (imgOut.empty())
		{
			imgOut = cv::Mat::zeros(labelImg.size(), CV_8UC1);
		}
		else
		{
			imgOut.setTo(cv::Scalar(0));
		}
		vector<cv::Point> pts;
		if (C_Region2vector(labelImg, pts, index) != -1)
		{
			for (cv::Point pt : pts)
			{
				imgOut.at<uchar>(pt) = 255;
			}
			return 1;
		}
		//else
		//{
		//	string strErr = "imgPro: region_toImg 函数中region2vector运行错误！";
		//	return -1;
		//}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: region_toImg 捕获到C++异常！";
		return -1;
	}
}

int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, const vector<cv::Point> &vecPt)
{
	try
	{
		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImg 输入图像 ";
			strErr += " 不存在或";
			return -1;
		}
		if (imgOut.empty())
		{
			imgOut = cv::Mat::zeros(labelImg.size(), CV_8UC1);
		}
		else
		{
			imgOut.setTo(cv::Scalar(0));
		}

		for (cv::Point pt : vecPt)
		{
			imgOut.at<uchar>(pt) = 255;
		}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: region_toImg 捕获到C++异常！";
		return -1;
	}

}

int imgPro::region_toImg_ROI(const cv::Size size, cv::Mat &imgOut, const vector<cv::Point> &vecPt, const cv::Point tl)
{
	try
	{
		if (imgOut.empty())
		{
			imgOut = cv::Mat::zeros(size, CV_8UC1);
		}
		else
		{
			imgOut.setTo(cv::Scalar(0));
		}

		for (cv::Point pt : vecPt)
		{
			imgOut.at<uchar>(pt-tl) = 255;
		}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: region_toImg 捕获到C++异常！";
		return -1;
	}

}
/**************************************************
iErr = region_toImgAll(strLabelImg,strImgOut,num)
功能：
按标记特征 向量索引输出区域的特征
限制：
无
参数：
strLabelImg        - 输入的标记图像，INT32型
strImgOut           - 输出的图像，CV_8UC1类型，标记区域为255，背景区域为0
num                  - label的总数
返回值：
iErr  - 0,正常； 非0，有错误

***************************************************/
int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut,const vector<int> &vecIndex)
{
	try
	{
		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImgAll 输入图像 ";
			return -1;
		}
		if (imgOut.empty())
		{
			imgOut = cv::Mat::zeros(labelImg.size(), CV_8UC1);
		}
		else
		{
			imgOut.setTo(cv::Scalar(0));
		}
		int procsCnt = omp_get_num_procs();
#pragma omp parallel num_threads(procsCnt-2) 
		{
#pragma omp parallel for 
			for (int i = 0; i < vecIndex.size(); i++)
			{
				vector<cv::Point> pts;
				int ret = C_Region2vector(labelImg, pts, vecIndex[i]);
#pragma omp critical
				{
					if (ret != -1)
					{
						for (int pi = 0; pi < pts.size(); pi++)
						{
							cv::Point pt = pts[pi];
							imgOut.at<uchar>(pt) = 255;
						}
					}
				}
			}
		}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: region_toImgAll 捕获到C++异常！";
		return -1;
	}
}
int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut,const vector<vector<cv::Point>> &vvPt)
{
	try
	{
		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImgAll 输入图像 ";
			return -1;
		}
		if (imgOut.empty())
		{
			imgOut = cv::Mat::zeros(labelImg.size(), CV_8UC1);
		}
		else
		{
			imgOut.setTo(cv::Scalar(0));
		}

		for (int i = 0; i < vvPt.size(); i++)
		{
			for (int c=0;c<vvPt[i].size();c++)
			{
				imgOut.at<uchar>(vvPt[i][c]) = 255;
			}
		}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: region_toImgAll 捕获到C++异常！";
		return -1;
	}
}
//
///**************************************************
//iErr = region_sortByFeature(strLabelImg,strFeature,iCnt)
//功能：
//    按标记特征 向量索引输出区域的特征
//限制：
//    无
//参数：
//strLabelImg        - 输入的标记图像，INT32型
//strFeature           - 输入的用于排序的特征，现支持面积“AREA”,高度"H",宽度"W",圆度"CIRCLE",高宽比"HW",外接旋转矩形面积“minRectArea”,
//                                按区域中心位置排序
//                                            -LR  从左至右排序
//                                            -RL  从右至左排序
//                                            -UD 从上至下排序
//                                            -DU 从下至上排序
//                               均降序排列
//iCnt			--保留前iCnt个区域
//返回值：
//iErr  - 0,正常； 非0，有错误
//Value  - 特征对应的值
//
//***************************************************/
//int imgPro::region_sortByFeature(lua_State *L)
//{
//	try
//	{
//		std::set<string> features{ "AREA", "H", "W", "CIRCLE", "HW", "LR", "RL", "UD", "DU", "minRectArea", "RECT" };
//		int iCount = lua_gettop(L);
//		if ((iCount == 2) && (lua_type(L, 1) != LUA_TSTRING || lua_type(L, 2) != LUA_TSTRING))
//		{
//			string strErr = "imgPro: region_sortByFeature 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//		if (iCount == 3 && (lua_type(L, 1) != LUA_TSTRING || lua_type(L, 2) != LUA_TSTRING || lua_type(L, 3) != LUA_TNUMBER))
//		{
//			string strErr = "imgPro: region_sortByFeature 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		} 
//		string strImgLabel = lua_tostring(L, 1);
//		string strFeat = lua_tostring(L, 2);
//		int cnt = 0;
//		if (iCount == 3)
//		{
//			cnt = (int)lua_tonumber(L, 3);
//		}
//		if (g_pMapImage->find(strImgLabel) == g_pMapImage->end() || (g_regionFeature)[strImgLabel].size()<1 ||
//						features.find(strFeat) == features.end())
//        {
//            string strErr = "imgPro: region_sortByFeature 输入图像 ";
//            strErr += strImgLabel;
//            strErr += " 不存在或特征";
//            strErr+= "不存在";
//			if (iCount == 3 && cnt > g_regionFeature[strImgLabel].size())
//			{
//				strErr += "或特征个数错误";
//			}
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//		vector<vector<float>> & vvTemp = g_regionFeature[strImgLabel];
//		if (strFeat == "AREA")
//		{
//			sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//			{
//				return va[5] > vb[5];   //area
//			});
//		}
//		else if (strFeat == "H")
//		{
//			sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//			{
//				return va[4] > vb[4];   //H
//			});
//
//        }
//        else if (strFeat == "W")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//                                                                            {
//                                                                                return va[3] > vb[3];   //W
//                                                                            });
//        }
//        else if (strFeat == "CIRCLE")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//            {
//                return va[8] > vb[8];   //W
//            });
//        }
//        else if (strFeat == "HW")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//            {
//                return va[9] > vb[9];   //W
//            });
//        }
//        else if (strFeat == "RL")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//            {
//                return va[7] > vb[7];   //COL from  Right To Left 
//            });
//        }
//        else if (strFeat == "LR")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//            {
//                return va[7] < vb[7];   //
//            });
//        }
//        else if (strFeat == "DU")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//            {
//                return va[6] > vb[6];   //
//            });
//        }
//        else if (strFeat == "UD")
//        {
//            sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//            {
//                return va[6] < vb[6];   //
//            });
//        }
//		else if (strFeat == "minRectArea")
//		{
//			sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//			{
//				return va[10] > vb[10];   //
//			});
//
//		}
//		else if (strFeat == "RECT")
//		{
//			sort(vvTemp.begin(), vvTemp.end(), [](vector<float> & va, vector<float> &vb)
//			{
//				return va[11] > vb[11];
//			});
//		}
//		else
//		{
//			string strErr = "imgPro: region_sortByFeature 输入特征不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//		if (iCount == 3)
//		{
//			g_regionFeature[strImgLabel].resize(cnt);  
//		}
//		lua_pushinteger(L, 0);
//		lua_newtable(L);
//		lua_pushnumber(L, -1);
//		lua_rawseti(L, -2, 0);
//		////////
//		if (strFeat == "AREA")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][5]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "H")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][4]);
//				lua_rawseti(L, -2, i + 1);
//			}
//
//		}
//		else if (strFeat == "W")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][3]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "CIRCLE")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][8]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "HW")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][9]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "RL")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][7]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "LR")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][7]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "DU")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][6]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "UD")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][6]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		else if (strFeat == "minRectArea")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][10]);
//				lua_rawseti(L, -2, i + 1);
//			}
//
//		}
//		else if (strFeat == "RECT")
//		{
//			for (int i = 0; i < vvTemp.size(); i++)
//			{
//				lua_pushnumber(L, vvTemp[i][11]);
//				lua_rawseti(L, -2, i + 1);
//			}
//		}
//		
//		return 2;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: region_sortByFeature 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}

///**************************************************
//iErr,startRow,startCol,endRow,endCol = region_fitLine(strLabelImg,iIndex,iDistType,iDiff)
//功能：
//    按索引对区域进行直线拟合，对离群点很敏感,拟合之前会依据斜率筛选区域上的点
//限制：
//无
//参数：
//strLabelImgIn        - 输入的标记图像，INT32型
//iIndex                     - 索引，从1开始
//iDistType               - 直线拟合方法, 常用1,2
//                                CV_DIST_L1      =1,   < distance = |x1-x2| + |y1-y2| 
//                                CV_DIST_L2 = 2,   < the simple euclidean distance
//                                CV_DIST_C = 3,   < distance = max(|x1-x2|,|y1-y2|)
//                                CV_DIST_L12 = 4,   < L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
//                                CV_DIST_FAIR = 5,   < distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998 
//                                CV_DIST_WELSCH = 6, < distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
//                                CV_DIST_HUBER = 7    < distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345 
//isVertical:  获取直线的类型，0 :拟合直线， 1:拟合直线中点处的垂线
//

//***************************************************/
int imgPro::region_fitLine(cv::Mat labelImg, int index, int distType, cv::Point &startPt, cv::Point &endPt,int isVertical)
{
	try
	{
		if (labelImg.empty())
		{
			return -1;
		}
		vector<cv::Point> pts = vector<cv::Point>();
		vector<cv::Point> ptsFilter = vector<cv::Point>();
		vector<cv::Point> profileSrc = vector<cv::Point>();
		vector<cv::Point> profileFilter = vector<cv::Point>();

		if (C_Region2vector(labelImg, pts, index) != -1)
		{
			cv::Vec4f line_para;
			cv::fitLine(pts, line_para, distType, 0, 0.01, 0.01);

			cv::Point point0;
			point0.x = line_para[2];
			point0.y = line_para[3];
			double k = line_para[1] / line_para[0];

			//找出直线在图像内的端点
			vector<cv::Point > vecPts;
			cv::Point point1, point2, point3, point4;
			// cv::Point startPt(-1, -1), endPt(-1, -1);
			int xEdge = labelImg.cols - 1;
			int yEdge = labelImg.rows - 1;
			point1.x = 0;
			point2.x = xEdge;
			point1.y = k * (point1.x - point0.x) + point0.y;
			point2.y = k * (point2.x - point0.x) + point0.y;

			point3.y = 0;
			point4.y = yEdge;
			point3.x = (point3.y - point0.y) / k + point0.x;
			point4.x = (point4.y - point0.y) / k + point0.x;

			vecPts.push_back(point1);
			vecPts.push_back(point2);
			vecPts.push_back(point3);
			vecPts.push_back(point4);

			vecPts.erase(remove_if(vecPts.begin(), vecPts.end(), [xEdge, yEdge](cv::Point pt1)
			{return (pt1.x == INT_MAX || pt1.x == INT_MIN || pt1.x < 0 || pt1.x > xEdge) || (pt1.y == INT_MAX || pt1.y == INT_MIN || pt1.y < 0 || pt1.y > yEdge); }), vecPts.end());
			if (vecPts.size() != 2)
			{
				string strErr = "imgPro: region_fitLine 函数中第一次寻找绘制点错误！";
				return -1;
			}
			startPt = vecPts[0];
			endPt = vecPts[1];
			if (isVertical)
			{
				vecPts.clear();
				point0.x = (startPt.x+endPt.x)/2;
				point0.y = (startPt.y + endPt.y) / 2;

				k = -1. / k;
				point1.x = 0;
				point2.x = xEdge;
				point1.y = k * (point1.x - point0.x) + point0.y;
				point2.y = k * (point2.x - point0.x) + point0.y;

				point3.y = 0;
				point4.y = yEdge;
				point3.x = (point3.y - point0.y) / k + point0.x;
				point4.x = (point4.y - point0.y) / k + point0.x;

				vecPts.push_back(point1);
				vecPts.push_back(point2);
				vecPts.push_back(point3);
				vecPts.push_back(point4);

				vecPts.erase(remove_if(vecPts.begin(), vecPts.end(), [xEdge, yEdge](cv::Point pt1)
				{return (pt1.x == INT_MAX || pt1.x == INT_MIN || pt1.x < 0 || pt1.x > xEdge) || (pt1.y == INT_MAX || pt1.y == INT_MIN || pt1.y < 0 || pt1.y > yEdge); }), vecPts.end());
				if (vecPts.size() != 2)
				{
					string strErr = "imgPro: region_fitLine 函数中第一次寻找绘制点错误！";
					return -1;
				}
				startPt = vecPts[0];
				endPt = vecPts[1];
			}
			return 0;
		}
	}
	catch (...)
	{
		return -2;
	}
}


//int imgPro::region_fitLine(lua_State*L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//        if (iCount != 4
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER
//            || lua_type(L, 3) != LUA_TNUMBER
//            || lua_type(L, 4) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: region_fitLine 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 5;
//        }
//
//    string strLabelImg = lua_tostring(L, 1);
//    int iIndex = (int)lua_tonumber(L, 2);
//    int  iDistType= (int)lua_tonumber(L, 3);
//    int iDiff = (int)lua_tonumber(L, 4);
//
//    iIndex -= 1;
//	if (g_pMapImage->find(strLabelImg) == g_pMapImage->end() || g_regionFeature[strLabelImg].size()< iIndex + 1 )
//    {
//        string strErr = "imgPro: region_fitLine 输入图像 ";
//        strErr += strLabelImg;
//        strErr += " 不存在或索引小于1 !";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        return 5;
//    }
//    vector<cv::Point> pts = vector<cv::Point>();    
//    vector<cv::Point> ptsFilter = vector<cv::Point>();
//    vector<cv::Point> profileSrc = vector<cv::Point>();
//    vector<cv::Point> profileFilter = vector<cv::Point>();
//
//    const cv::Mat&  labelImg = (*g_pMapImage)[strLabelImg];
//   
//    if (C_Region2vector(strLabelImg, pts, iIndex) != -1)
//    {
//        cv::Vec4f line_para;
//        cv::fitLine(pts, line_para, iDistType, 0, 0.01, 0.01);
//
//        cv::Point point0;
//        point0.x = line_para[2];
//        point0.y = line_para[3];
//        double k = line_para[1] / line_para[0];
//
//        //找出直线在图像内的端点
//		vector<cv::Point > vecPts;
//        cv::Point point1, point2, point3, point4;
//        cv::Point startPt(-1, -1), endPt(-1, -1);
//        int xEdge = labelImg.cols-1;
//        int yEdge = labelImg.rows-1;
//        point1.x = 0;
//        point2.x = xEdge;
//        point1.y = k * (point1.x - point0.x) + point0.y;
//        point2.y = k * (point2.x - point0.x) + point0.y;
//
//        point3.y = 0;
//        point4.y = yEdge;
//        point3.x = (point3.y - point0.y) / k + point0.x;
//        point4.x = (point4.y - point0.y) / k + point0.x;
//
//		vecPts.push_back(point1);
//		vecPts.push_back(point2);
//		vecPts.push_back(point3);
//		vecPts.push_back(point4);
//		 
//		vecPts.erase(remove_if(vecPts.begin(), vecPts.end(), [xEdge, yEdge](cv::Point pt1)
//		{return (pt1.x == INT_MAX || pt1.x == INT_MIN || pt1.x < 0 || pt1.x > xEdge) || (pt1.y == INT_MAX || pt1.y == INT_MIN ||pt1.y < 0 || pt1.y > yEdge); }), vecPts.end());
//		if (vecPts.size() != 2)
//		{
//			string strErr = "imgPro: region_fitLine 函数中第一次寻找绘制点错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			return 5;
//
//		}
//		startPt = vecPts[0];
//		endPt = vecPts[1];
//      //  line(out, startPt, endPt, cv::Scalar(0, 255, 0), 1);
//        //对远离直线的点进行删除
//        for (int i = 0; i<pts.size();i++)
//        {
//            cv::Point  pt = pts[i];
//            double dist = distPt2Line(startPt,endPt,pt);
//            if (dist > iDiff)
//            {
//                pts[i]= cv::Point(-1,-1);
//            }
//        }
//        pts.erase(remove_if(pts.begin(), pts.end(), [](cv::Point pt){return pt.x < 0; }),pts.end());       
//
//        /*  cv::Mat out = cv::Mat::zeros(labelImg.size(), CV_8UC3);
//        for (int i = 0; i < pts.size() - 1; i++)
//        {
//            line(out, pts[i], pts[i + 1], cv::Scalar(0, 255, 0), 1);
//        }*/
//
//        cv::fitLine(pts, line_para, iDistType, 0, 0.01, 0.01);
//       
//        point0.x = line_para[2];
//        point0.y = line_para[3];
//         k = line_para[1] / line_para[0];
//
//        //找出直线在图像内的端点
//		 vecPts.clear();
//        startPt = cv::Point(-1,-1), endPt = cv::Point(-1,-1);
//         xEdge = labelImg.cols;
//         yEdge = labelImg.rows;
//        point1.x = 0;
//        point2.x = xEdge;
//        point1.y = k * (point1.x - point0.x) + point0.y;
//        point2.y = k * (point2.x - point0.x) + point0.y;
//
//        point3.y = 0;
//        point4.y = yEdge;
//        point3.x = (point3.y - point0.y) / k + point0.x;
//        point4.x = (point4.y - point0.y) / k + point0.x;        
//
//		vecPts.push_back(point1);
//		vecPts.push_back(point2);
//		vecPts.push_back(point3);
//		vecPts.push_back(point4);
//
//		vecPts.erase(remove_if(vecPts.begin(), vecPts.end(), [xEdge, yEdge](cv::Point pt1)
//		{ return (pt1.x == INT_MAX || pt1.x == INT_MIN || pt1.x < 0 || pt1.x > xEdge) || (pt1.y == INT_MAX || pt1.y == INT_MIN || pt1.y < 0 || pt1.y > yEdge); }), vecPts.end());
//		if (vecPts.size() != 2)
//		{
//			string strErr = "imgPro: region_fitLine 函数中第二次寻找绘制点错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, -100);
//			return 5;
//
//		}
//		startPt = vecPts[0];
//		endPt = vecPts[1];
//
//        lua_pushinteger(L, 0);
//        lua_pushinteger(L, startPt.y);
//        lua_pushinteger(L, startPt.x);
//        lua_pushinteger(L, endPt.y);
//        lua_pushinteger(L, endPt.x);
//        return 5;
//    }
//    else
//    {
//        string strErr = "imgPro: region_fitLine 函数中C_FilterProfileByCurv运行错误！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        return 5;
//    }
//   }
//catch (...)
//{
//    string strErr = "imgPro: region_fitLine 捕获到C++异常！";
//    ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//    lua_pushinteger(L, -100);
//    lua_pushinteger(L, -100);
//    lua_pushinteger(L, -100);
//    lua_pushinteger(L, -100);
//    lua_pushinteger(L, -100);
//    return 5;
//}}
//
///**************************************************
//iErr,iCenterRow,iCenterCol,iWidth,iHeight,dAngle = region_smallestRectR(strLabelImg,iIndex)
//功能：
//    获取区域中对应索引的最小外接矩形,     注意KitB的Angle与imgPro的是相反的。
//无
//参数：
//strLabelImg        - 输入的标记图像，INT32型
//iIndex                  - 对应的索引
//
//返回值：
//iErr  - 0,正常； 非0，有错误
//
//***************************************************/
//int imgPro::region_smallestRectR(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//        if (iCount !=2
//            ||lua_type(L,1)!= LUA_TSTRING
//            ||lua_type(L,2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: region_smallestRectR 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 6;
//        }
//
//        string strLabelImg = lua_tostring(L, 1);
//        int iLabelIndex = (int)lua_tonumber(L, 2);
//
//        iLabelIndex -= 1;
//		if (g_pMapImage->find(strLabelImg) == g_pMapImage->end() || g_regionFeature[strLabelImg].size()<iLabelIndex+1 )
//        {
//            string strErr = "imgPro: region_fitLine 输入图像 ";
//            strErr += strLabelImg;
//            strErr += " 不存在或索引不存在 !";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 6;
//        }
//        vector<cv::Point> pts = vector<cv::Point>();
//        if (0 == C_Region2vector(strLabelImg,pts,iLabelIndex))
//        {
//            cv::RotatedRect roRect = cv::minAreaRect(pts);
//
//            int longEdge, shortEdge;
//            float angle = -roRect.angle;
//            if (roRect.size.width > roRect.size.height)
//            {
//                longEdge = roRect.size.width;
//                shortEdge = roRect.size.height;
//            } 
//            else
//            {
//                shortEdge = roRect.size.width;
//                longEdge = roRect.size.height;
//                angle = 90 + angle;
//            }          
//
//            lua_pushinteger(L,0);
//            lua_pushinteger(L, roRect.center.y);   // row
//            lua_pushinteger(L, roRect.center.x);    //col
//            lua_pushinteger(L, longEdge);  //long edge is width
//            lua_pushinteger(L, shortEdge);
//            lua_pushnumber(L, angle);
//            return 6;
//        } 
//        else
//        {
//            string strErr = "imgPro: region_smallestRectR 函数中C_Region2vector运行错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 6;
//        }
//    }
//    catch (...)
//    {
//        string strErr = "imgPro: region_smallestRectR 函数中捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        return 6;
//    }
//}
//
//
///**************************************************
//iErr,dRow,dCol,dAngle,dL1,dL2 = cont_smallestRectR(iIndex)
//功能：
//获取轮廓的最小外接旋转矩形
//限制：
//无
//参数：
//iIndex - 轮廓序号,从序号1开始
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow - 矩形中心点行坐标
//dCol - 矩形中心点列坐标
//dAngle - 矩形旋转角度
//dL1 - 矩形长边的一半
//dL2 - 矩形短边的一半
//***************************************************/
//int imgPro::cont_smallestRectR(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_smallestRectR 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 6;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);// 第二个参数为 轮廓序号,从1开始
//		iIndex -= 1;
//		// 参数检查
//		if (iIndex < 0 || iIndex >= int(g_contours.size()))
//		{
//			string strErr = "imgPro: cont_smallestRectR 索引区间错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 6;
//		}
//
//		std::vector<std::vector<cv::Point>>::iterator itor = g_contours.begin();
//		for (int i=0; i<iIndex; i++)
//		{
//			itor++;
//		}
//
//		cv::RotatedRect r = cv::minAreaRect(*itor);
//		
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, r.center.y);
//		lua_pushnumber(L, r.center.x);
//		lua_pushnumber(L, -r.angle);
//		lua_pushnumber(L, r.size.width/2);
//		lua_pushnumber(L, r.size.height/2);
//// 		Point2f pt;
//// 		float r;
//// 		minEnclosingCircle(*itor, pt, r);
//// 		lua_pushinteger(L, 0);
//// 		lua_pushnumber(L, pt.y);
//// 		lua_pushnumber(L, pt.x);
//// 		lua_pushnumber(L, 0);
//// 		lua_pushnumber(L, r);
//// 		lua_pushnumber(L, r);
//		return 6;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: cont_smallestRectR 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		return 6;
//	}
//
//}
//
//
///**************************************************
//iErr,dRow,dCol,dWidth,dHeight = cont_smallestRect(iIndex)
//功能：
//计算指定轮廓的最小外接矩形(无旋转)
//限制：
//无
//参数：
//iIndex - 轮廓序号
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow - 矩形左上角点行坐标
//dCol - 矩形左上角点列坐标
//dWidth - 矩形宽度
//dHeight - 矩形高度
//***************************************************/
//int imgPro::cont_smallestRect(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_smallestRect 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 5;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);// 第二个参数为 轮廓序号
//		iIndex -= 1;
//		// 参数检查
//		if (iIndex < 0 || iIndex >= int(g_contours.size()))
//		{
//			string strErr = "imgPro: cont_smallestRect 索引范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 5;
//		}
//
//		std::vector<std::vector<cv::Point>>::iterator itor = g_contours.begin();
//		for (int i = 0; i<iIndex; i++)
//		{
//			itor++;
//		}
//
//		cv::Rect r = cv::boundingRect(*itor);
//
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, r.y);
//		lua_pushnumber(L, r.x);
//		lua_pushnumber(L, r.width);
//		lua_pushnumber(L, r.height);
//
//		return 5;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: cont_smallestRect 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		return 5;
//	}
//
//}
//
///**************************************************
//iErr,iArea,dRow,dCol = cont_areaCenter(iIndex)
//功能：
//计算轮廓的面积和中心点坐标
//限制：
//无
//参数：
//iIndex - 轮廓序号
//返回值：
//iErr - 0,正常； 非0，有错误
//iArea - 轮廓面积
//dRow - 轮廓中心点行坐标
//dCol - 轮廓中心点列坐标
//***************************************************/
//int imgPro::cont_areaCenter(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		int iIndex = (int)lua_tonumber(L, 1);// 第二个参数为 轮廓序号
//		iIndex -= 1;
//		// 参数检查
//		if (iIndex < 0 || iIndex >= int(g_contours.size()) || iCount != 1)
//		{
//			string strErr = "imgPro: cont_areaCenter 参数范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 4;
//		}
//
//		std::vector<std::vector<cv::Point>>::iterator itor = g_contours.begin();
//		itor += iIndex;
//		int iLength = itor->size();
//		double dSumRow = 0, dSumCol = 0, dRow = 0, dCol = 0;
//		if (0 == iLength)
//		{
//			string strErr = "imgPro: cont_areaCenter 该索引对应轮廓长度为0！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 4;
//		}
//		for (int i = 0; i < iLength; ++i)
//		{
//			cv::Point p = (*itor)[i];
//			dSumRow += p.y;
//			dSumCol += p.x;
//		}
//		dRow = dSumRow / iLength;
//		dCol = dSumCol / iLength;
//
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, cv::contourArea(*itor));
//		lua_pushnumber(L, dRow);
//		lua_pushnumber(L, dCol);
//		return 4;
//		
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: cont_areaCenter 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		return 4;
//	}
//
//}
//
///**************************************************
//iErr,iNum = cont_unionContByDist(dDis)
//功能：
//根据距离联合轮廓
//限制：
//无
//参数：
//dDis - 最小距离
//返回值：
//iErr - 0,正常； 非0，有错误
//iNum - 轮廓个数
//***************************************************/
//int imgPro::cont_unionContByDist(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 1 
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_unionContByDist 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		double dDis = lua_tonumber(L, 1);// 最大距离
//		double dDistance = dDis > 0 ? dDis*dDis : -dDis*dDis;	// 计算距离平方值，后面比较距离不需要开方
//
//		std::vector<std::vector<cv::Point>>::iterator itor = g_contours.begin();
//		while (itor != g_contours.end())
//		{
//		a:			std::vector<std::vector<cv::Point>>::iterator itor2 = itor;
//			itor2++;
//			while (itor2 != g_contours.end())
//			{
//				for (std::vector<cv::Point>::iterator itorPt = itor->begin();
//					itorPt != itor->end(); itorPt++)
//				{
//					for (std::vector<cv::Point>::iterator itorPt2 = itor2->begin();
//						itorPt2 != itor2->end(); itorPt2++)
//					{
//						//double d = (itorPt->x-itorPt2->x)*(itorPt->x-itorPt2->x)
//						//	+(itorPt->y-itorPt2->y)*(itorPt->y-itorPt2->y);
//						double d = abs(itorPt->x - itorPt2->x) + abs(itorPt->y - itorPt2->y);	//算街道距离速度很快
//						if (d < dDistance)
//						{
//							(*itor).reserve(itor->size() + itor2->size());
//							(*itor).insert((*itor).end(), (*itor2).begin(), (*itor2).end());
//							g_contours.erase(itor2);
//							goto a;
//						}
//					}
//				}
//				itor2++;
//			}
//			itor++;
//		}
//
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, (int)g_contours.size());
//		return 2;
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: cont_unionContByDist 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, (int)g_contours.size());
//		return 2;
//	}
//}
//
///**************************************************
//iErr,dRow1,dCol1,dRow2,dCol2 = cont_fitLine(imgIn,iIndex,iType)
//功能：
//根据轮廓拟合直线（不是线段）
//限制：
//无
//参数：
//imgIn - 输入图像
//iIndex - 轮廓序号，从1开始
//iType - 拟合类型(1到7，1：CV_DIST_L1，2：CV_DIST_L2，3：CV_DIST_C，4：CV_DIST_L12)
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow1 - 直线起始点行坐标
//dCol1 - 直线起始点列坐标
//dRow2 - 直线末端点行坐标
//dCol2 - 直线末端点列坐标
//***************************************************/
//int imgPro::cont_fitLine(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 3
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_fitLine 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 5;
//		}
//
//		string strIn = lua_tostring(L, 1);// 第二个参数为 轮廓序号
//		int iIndex = (int)lua_tonumber(L, 1);// 第二个参数为 轮廓序号
//		int iType = (int)lua_tonumber(L, 2);//
//		iIndex -= 1;
//		// 参数检查
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()   
//				|| iIndex >= g_contours.size() ||iIndex<0)
//		{
//			string strErr = "imgPro: cont_fitLine 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！或索引范围错误";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 5;
//		}
//
//		cv::Vec4f v;
//		cv::fitLine(g_contours[iIndex], v, iType, 0, 0.01, 0.01);
//
//		int iWidth = (*g_pMapImage)[strIn].cols;
//
//		//获取点斜式的点和斜率  
//		cv::Point point0;
//		point0.x = v[2];
//		point0.y = v[3];
//
//		double k = v[1] / v[0];
//
//		//计算直线的端点(y = k(x - x0) + y0)  
//		cv::Point point1, point2;
//		point1.x = 0;
//		point1.y = k * (0 - point0.x) + point0.y;
//		point2.x = iWidth;
//		point2.y = k * (iWidth - point0.x) + point0.y;
//
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, point1.y);
//		lua_pushnumber(L, point1.x);
//		lua_pushnumber(L, point2.y);
//		lua_pushnumber(L, point2.x);
//		return 5;
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: cont_fitLine 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		return 5;
//	}
//}
//
///**************************************************
//iErr,dRow,dCol,dWidth,dHeight,dAngle = cont_fitEllipse(imgIn,iIndex)
//功能：
//根据轮廓拟合椭圆
//限制：
//无
//参数：
//imgIn - 输入图像
//iIndex - 轮廓序号，从1开始
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow    - 椭圆中心点行坐标
//dCol      - 椭圆中心点列坐标
//dWidth  - 椭圆宽
//dHeight - 椭圆高
//dAngle   -椭圆角度
//***************************************************/
//
//int imgPro::cont_fitEllipse(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);      // 参数个数
//
//        // 参数检查
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: cont_fitEllipse 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 6;
//        }
//
//        string strIn = lua_tostring(L, 1);// 第二个参数为 轮廓序号
//        int iIndex = (int)lua_tonumber(L, 2);// 第二个参数为 轮廓序号
//        iIndex -= 1;
//        // 参数检查
//        if (g_pMapImage->find(strIn) == g_pMapImage->end()
//            || iIndex >= g_contours.size() || iIndex < 0)
//        {
//            string strErr = "imgPro: cont_fitEllipse 输入图像 ";
//            strErr += strIn;
//            strErr += " 不存在！或索引范围错误";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 6;
//        }
//
//        cv::RotatedRect rectEllipse = cv::fitEllipse(g_contours[iIndex]);
//
//        lua_pushinteger(L, 0);
//        lua_pushnumber(L, rectEllipse.center.y);
//        lua_pushnumber(L, rectEllipse.center.x);
//        lua_pushnumber(L, rectEllipse.size.width/2.);
//        lua_pushnumber(L, rectEllipse.size.height/2);
//        lua_pushnumber(L, rectEllipse.angle);
//        return 6;
//    }
//    catch (...)
//    {
//        string strErr = "imgPro: cont_fitEllipse 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        return 6;
//    }
//
//}
//
//
///**************************************************
//iErr,dRow,dCol,dR= cont_fitCircle(imgIn,iIndex)
//功能：
//根据轮廓拟合圆,最小二乘法
//限制：
//无
//参数：
//imgIn - 输入图像
//iIndex - 轮廓序号，从1开始
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow    - 圆中心点行坐标
//dCol      - 圆中心点列坐标
//dR         - 圆半径
//***************************************************/
//
//int imgPro::cont_fitCircle(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);      // 参数个数
//
//        // 参数检查
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: cont_fitCircle 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        string strIn = lua_tostring(L, 1);// 第二个参数为 轮廓序号
//        int iIndex = (int)lua_tonumber(L, 2);// 第二个参数为 轮廓序号
//        iIndex -= 1;
//        // 参数检查
//        if (g_pMapImage->find(strIn) == g_pMapImage->end()
//            || iIndex >= g_contours.size() || iIndex < 0)
//        {
//            string strErr = "imgPro: cont_fitCircle 输入图像 ";
//            strErr += strIn;
//            strErr += " 不存在！或索引范围错误";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        double row, col, r;
//        if (0 != LeastSquareFittingCircle(g_contours[iIndex], col, row, r))
//        {
//            string strErr = "imgPro: cont_fitCircle 执行回归函数异常！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        lua_pushinteger(L, 0);
//        lua_pushnumber(L, row);
//        lua_pushnumber(L, col);
//        lua_pushnumber(L, r);
//        return 4;
//    }
//    catch (...)
//    {
//        string strErr = "imgPro: cont_fitCircle 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        return 4;
//    }
//
//}
//
///**************************************************
//dAngle = angle2Lines()
//功能：
//获取两条线段的夹角
//限制：
//无
//参数：
//返回值：
//dAngle - 线段夹角,用角度表示
//***************************************************/
int imgPro::angle2Lines(cv::Point line1StartPt, cv::Point line1EndPt,cv::Point line2StartPt, cv::Point line2EndPt,double &angle)
{
	try
	{
		double dA[3], dB[3];
		dA[0] = line1EndPt.x-line1StartPt.x;
		dA[1] = line1EndPt.y - line1StartPt.y;
		dA[2] = 0;
		dB[0] = line2EndPt.x - line2StartPt.x;
		dB[1] = line2EndPt.y - line2StartPt.y;
		dB[2] = 0;
		
		angle =  Get2VecAngle(dA, dB)/PI*180.0; //输出以角度表示
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro::angle2Lines 捕获到C++异常！";
		return 1 ;
	}
}
int imgPro::angle2Lines(cv::Point2f line1StartPt, cv::Point2f line1EndPt,cv::Point2f line2StartPt, cv::Point2f line2EndPt,double &angle)
{
	try
	{
		double dA[3], dB[3];
		dA[0] = line1EndPt.x-line1StartPt.x;
		dA[1] = line1EndPt.y - line1StartPt.y;
		dA[2] = 0;
		dB[0] = line2EndPt.x - line2StartPt.x;
		dB[1] = line2EndPt.y - line2StartPt.y;
		dB[2] = 0;
		
		angle =  Get2VecAngle(dA, dB)/PI*180.0; //输出以角度表示
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro::angle2Lines 捕获到C++异常！";
		return 1 ;
	}
}
//
///**************************************************
//dDist = dist2Pts(dP1Row,dP1Col,dP2Row,dP2Col)
//功能：
//获取两个点之间的距离
//限制：
//无
//参数：

//返回值：
//iErr - 0,正常； 非0，有错误
//dDist - 两点之间距离
//***************************************************/
double imgPro::dist2Pts(cv::Point pt1,cv::Point pt2)
{
	try
	{

		double dP1Row = pt1.y;	// 
		double dP1Col = pt1.x;	// 
		double dP2Row = pt2.y;	// 
		double dP2Col = pt2.x;	// 

		return sqrt((dP1Row - dP2Row)*(dP1Row - dP2Row) + (dP1Col - dP2Col)*(dP1Col - dP2Col));
	
	}
	catch (...)
	{
		string strErr = "imgPro: dist2Pts 捕获到C++异常！";
		return -1;
	}
}

int imgPro::angle2Pts(cv::Point ptStart, cv::Point ptEnd,double &angle)
{
	try
	{
		cv::Point ptEndHor = ptStart;
		ptEndHor.x = ptEnd.x + 10;

		angle2Lines(ptStart, ptEnd, ptStart, ptEndHor, angle);//返回空间向量夹角
		return 0;

	}
	catch (...)
	{
		return -1;
	}
}

int imgPro::angle2Pts(cv::Point2f ptStart, cv::Point2f ptEnd, double & angle)
{
	try
	{
		cv::Point2f ptEndHor = ptStart;
		ptEndHor.x = ptEnd.x + 10.;

		angle2Lines(ptStart, ptEnd, ptStart, ptEndHor, angle);//返回空间向量夹角
		return 0;

	}
	catch (...)
	{
		return -1;
	}
}


//
///**************************************************
//iErr,tableCenter = math_splitRect(iCenterRow,iCenterCol,iWidth,iHeight,dAngle,iM,iN)
//功能：
//将矩形切割成m*n个小矩形(row-->m, col-->n)
//限制：
//无
//参数：
//	iCenterRow,iCenterCol,iWidth,iHeight,dAngle：旋转矩形参数
//	iM	：矩形的行分割为M份
//	iN	：矩形的列分割为N份
//返回值：
//iErr - 0,正常； 非0，有错误
//tableCenter
//***************************************************/
//int imgPro::math_splitRect(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//		if (iCount != 7
//			|| lua_type(L, 1) != LUA_TNUMBER
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: math_splitRect 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//		int iCenterRow = (int)lua_tonumber(L, 1);
//		int iCenterCol = (int)lua_tonumber(L, 2);
//		int iWidth = (int)lua_tonumber(L, 3);
//		int iHeight = (int)lua_tonumber(L, 4);
//		int dAngle = lua_tonumber(L, 5);
//		int iM = lua_tointeger(L, 6);
//		int iN = lua_tointeger(L, 7);
//
//		cv::Point2f pts[4];
//		cv::RotatedRect rotRect(cv::Point(iCenterCol, iCenterRow), cv::Size(iWidth, iHeight), dAngle);
//		rotRect.points(pts);
//		
//		vector<cv::Point2f> vecPts(begin(pts), end(pts));
//		sort(vecPts.begin(), vecPts.end(), [](cv::Point2f pt1, cv::Point2f pt2){return pt1.x < pt2.x; });
//		sort(vecPts.begin(), vecPts.begin() + 2, [](cv::Point2f pt1, cv::Point2f pt2){return pt1.y < pt2.y; });
//		sort(vecPts.begin() + 2, vecPts.end(), [](cv::Point2f pt1, cv::Point2f pt2){return pt1.y < pt2.y; });
//
//		cv::Point TL = vecPts[0];
//		cv::Point BL = vecPts[1];
//
//		cv::Point TR = vecPts[2];
//		cv::Point BR = vecPts[3];
//
//
//		vector<cv::Point2d> LPts = divideLineSeg(TL, BL, iN);
//		vector<cv::Point2d> RPts = divideLineSeg(TR, BR, iN);
//
//		vector<vector<cv::Point2d>> grid;
//		vector<cv::Point2d> nowPts, nextPts;
//		cv::Point2d gridCent;
//		vector<cv::Point2d> gridCents;
//
//		for (int i = 0; i < LPts.size(); i++)
//		{
//			vector<cv::Point2d> temp = divideLineSeg(LPts[i], RPts[i], iM);
//			grid.push_back(temp);
//		}
//
//		for (int i = 0; i < grid.size() - 1; i++)
//		{
//			nowPts = grid[i];
//			nextPts = grid[i + 1];
//			for (int i = 0; i < nowPts.size() - 1; i++)
//			{
//				gridCent = (nowPts[i] + nextPts[i + 1]) / 2;
//				gridCents.push_back(gridCent);
//			}
//		}
//
//		lua_pushinteger(L, 0);
//		lua_createtable(L, gridCents.size(), 0);
//		for (int i = 0; i < gridCents.size(); i++)
//		{
//			lua_pushnumber(L, i + 1);
//			lua_createtable(L, 0, 2);
//			lua_pushnumber(L, gridCents[i].y);
//			lua_setfield(L, -2, "row");
//			lua_pushnumber(L, gridCents[i].x);
//			lua_setfield(L, -2, "col");
//			lua_settable(L, -3);
//		}
//		return 2;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: math_splitRect 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}
///**************************************************
//iErr,dRow,dCol = intersecBy2Lines(dRow1,dCol1,dRow2,dCol2,dRow3,dCol3,dRow4,dCol4)
//功能：
//查找两条线段的交点
//限制：
//无
//参数：

//返回值：
//iErr - 0,正常； 非0，有错误
//dRow - 交点的行坐标
//dCol - 交点的列坐标
//***************************************************/

double imgPro::angle_segX(cv::Point2f ptStart, cv::Point2f ptEnd)
{	
	return -std::atan2(ptEnd.y-ptStart.y,ptEnd.x-ptStart.y)/CV_PI*180;
}

double imgPro::angle_segX(cv::Vec4f lineSeg)
{
	return  -std::atan2(lineSeg[3] - lineSeg[1], lineSeg[2] - lineSeg[0]) / CV_PI * 180;
}

int imgPro::intersecBy2Lines(cv::Point line1StartPt, cv::Point line1EndPt,
	cv::Point line2StartPt, cv::Point line2EndPt, cv::Point &intersecPt)
{
	try
	{

		double dRow1 = line1StartPt.y;	// 
		double dCol1 = line1StartPt.x;	// 
		double dRow2 = line1EndPt.x;	// 
		double dCol2 = line1EndPt.y;	// 
		double dRow3 = line2StartPt.y;	// 
		double dCol3 = line2StartPt.x;	// 
		double dRow4 = line2EndPt.x;	// 
		double dCol4 = line2EndPt.y;	// 



		double A[3], vecA[3], B[3], vecB[3], iPoint[3];
		A[0]=dCol1; A[1]=dRow1; A[2]=0; 
		vecA[0]=dCol2-dCol1; vecA[1]=dRow2-dRow1; vecA[2]=0; 
		B[0]=dCol3; B[1]=dRow3; B[2]=0; 
		vecB[0]=dCol4-dCol3; vecB[1]=dRow4-dRow3; vecB[2]=0; 

		GetIntersectionFor2Line(vecA, A, vecB, B, iPoint);
		
		intersecPt = cv::Point(iPoint[0], iPoint[1]);
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: math_intersectionBy2Lines 捕获到C++异常！";
		return 1;
	}
}
//
///**************************************************
//ret = closestPt2L(dPtRow,dPtCol,dLRow1,dLCol1,dLRow2,dLCol2)
//功能：
//计算直线上离点距离最近的点，同时返回点与直线的距离
//限制：
//无
//参数：

//返回值：

//***************************************************/
int imgPro::closestPt2Line(cv::Point offLinePt, cv::Point linePt1, cv::Point linePt2, cv::Point &nearestLinePt, double &dist)
{
	try
	{

		double dPointRow = offLinePt.y;	// 
		double dPointCol = offLinePt.x;	// 
		double dLineRow1 = linePt1.y;	// 
		double dLineCol1 = linePt1.x;	// 
		double dLineRow2 = linePt2.y;	// 
		double dLineCol2 = linePt2.x;	// 

		double A[3], vecA[3], dPoint[3], iPoint[3];
		dPoint[0]=dPointCol; dPoint[1]=dPointRow; dPoint[2]=0; 
 		A[0]=dLineCol1; A[1]=dLineRow1; A[2]=0; 
 		vecA[0]=dLineCol2-dLineCol1; vecA[1]=dLineRow2-dLineRow1; vecA[2]=0; 
		getClosestPointP2L(dPoint, A, vecA, iPoint);
		
		nearestLinePt =cv::Point(iPoint[0],iPoint[1]);	
		dist = sqrt((iPoint[0]-dPoint[0])*(iPoint[0]-dPoint[0])+(iPoint[1]-dPoint[1])*(iPoint[1]-dPoint[1]));
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: img_copy 捕获到C++异常！";
		return 1;
	}
}
double imgPro::closestPt2Line(cv::Point offLinePt, cv::Vec4f line, cv::Point & nearestLinePt)
{
	double dist = 0.0;
	closestPt2Line(offLinePt, cv::Point(std::round(line[0]),std::round(line[1])),
							  cv::Point(std::round(line[2]), std::round(line[3])), nearestLinePt, dist);
	return dist;
}
double imgPro::closestPt2LineSeg(cv::Mat &canvImg, cv::Point offLinePt, cv::Vec4f line, cv::Point & nearestLinePt)
{
	try
	{
		cv::LineIterator lineIte(canvImg,cv::Point(std::round(line[0]), std::round(line[1])) ,cv::Point(std::round(line[2]), std::round(line[3])));
		double minDist = INT_MAX;
		for (int i=0;i<lineIte.count;i++, lineIte++)
		{
			cv::Point t = lineIte.pos();

			double distTmp = imgPro::dist2Pts(offLinePt, t);
			if (distTmp < minDist)
			{
				minDist = distTmp;
				nearestLinePt = t;
			}
		}

		return minDist;
	}
	catch (const std::exception&)
	{
		return -1;
	}

}
//
///**************************************************
//iErr,dMinDist,dMaxDist,dMinRow,dMinCol,dMaxRow,dMaxCol = math_distPtCont(dPtRow,dPtCol,iIndex)
//功能：
//计算指定轮廓与指定点的最小距离点和最大距离点，并返回最小、最大距离
//限制：
//无
//参数：
//dPtRow - 点的行坐标
//dPtCol - 点的列坐标
//iIndex - 轮廓序号
//返回值：
//iErr - 0,正常； 非0，有错误
//dMinDist - 轮廓到点的最小距离
//dMaxDist - 轮廓到点的最大距离
//dMinRow - 轮廓与点距离最小点的行坐标
//dMinCol - 轮廓与点距离最小点的列坐标
//dMaxRow - 轮廓与点距离最大点的行坐标
//dMaxCol - 轮廓与点距离最大点的列坐标
//***************************************************/
//int imgPro::math_distPtCont(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 3 
//			|| lua_type(L, 1) != LUA_TNUMBER
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: math_distPtCont 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 7;
//		}
//
//		double dPointRow = lua_tonumber(L, 1);	// 
//		double dPointCol = lua_tonumber(L, 2);	// 
//		int iIndex = (int)lua_tonumber(L, 3);	// 
//		iIndex -= 1;
//		// 参数检查
//		if (iIndex >= g_contours.size()|| iIndex<0)
//		{
//			string strErr = "imgPro: math_distPtCont 索引范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 7;
//		}
//
//		std::vector<cv::Point> & c = g_contours[iIndex];
//		cv::Point ptMin(c[0]), ptMax(c[0]);
//		double dMin(INT_MAX), dMax(0);
//		for (std::vector<cv::Point>::iterator itor = c.begin();
//			itor != c.end(); itor++)
//		{
//			cv::Point & p = (*itor);
//			double dLen = sqrt(((p.x-dPointCol)*(p.x-dPointCol)+(p.y-dPointRow)*(p.y-dPointRow)));
//			if (dLen < dMin)
//			{
//				ptMin = p;
//				dMin = dLen;
//			}
//			if (dLen > dMax)
//			{
//				ptMax = p;
//				dMax = dLen;
//			}
//		}
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, dMin);
//		lua_pushnumber(L, dMax);
//		lua_pushnumber(L, ptMin.y);
//		lua_pushnumber(L, ptMin.x);
//		lua_pushnumber(L, ptMax.y);
//		lua_pushnumber(L, ptMax.x);
//		return 7;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: math_distPtCont 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		return 7;
//	}
//}
//
///**************************************************
//iErr,dMinDist,dMaxDist,dMinRow,dMinCol,dMaxRow,dMaxCol = math_distLineCont(dLRow1,dLCol1,dLRow2,dLCol2,iIndex)
//功能：
//计算指定轮廓与指定线的最小距离点和最大距离点，并输出最小、最大距离
//限制：
//无
//参数：
//dLRow1 - 线段起始点的行坐标
//dLCol1  - 线段起始点的列坐标
//dLRow2 - 线段末端点的行坐标
//dLCol2  - 线段末端点的列坐标
//iIndex    - 轮廓序号
//返回值：
//iErr - 0,正常； 非0，有错误
//dMinDist - 轮廓到线段的最小距离
//dMaxDist - 轮廓到线段的最大距离
//dMinRow - 轮廓与线段距离最小点的行坐标
//dMinCol - 轮廓与线段距离最小点的列坐标
//dMaxRow - 轮廓与线段距离最大点的行坐标
//dMaxCol - 轮廓与线段距离最大点的列坐标
//***************************************************/
//int imgPro::math_distLineCont(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 5 
//			|| lua_type(L, 1) != LUA_TNUMBER
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: math_distLineCont 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 7;
//		}
//
//		double dLineRow1 = lua_tonumber(L, 1);	// 
//		double dLineCol1 = lua_tonumber(L, 2);	// 
//		double dLineRow2 = lua_tonumber(L, 3);	// 
//		double dLineCol2 = lua_tonumber(L, 4);	// 
//		int iIndex = (int)lua_tonumber(L, 5);	// 
//		iIndex -= 1;
//
//		double dLineVec[3], dLinePoint[3];
//		dLineVec[0] = dLineCol2 - dLineCol1;
//		dLineVec[1] = dLineRow2 - dLineRow1;
//		dLineVec[2] = 0;
//		dLinePoint[0] = dLineCol1;
//		dLinePoint[1] = dLineRow1;
//		dLinePoint[2] = 0;
//		
//		// 参数检查
//		if (iIndex >= g_contours.size() || iIndex<0)
//		{
//			string strErr = "imgPro: math_distLineCont 索引范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 7;
//		}
//
//		std::vector<cv::Point> & c = g_contours[iIndex];
//		cv::Point ptMin(c[0]), ptMax(c[0]);
//		double dMin(INT_MAX), dMax(0);
//		for (std::vector<cv::Point>::iterator itor = c.begin();
//			itor != c.end(); itor++)
//		{
//
//			cv::Point & p = (*itor);
//			double dPoint[3], dPtOut[3];
//			dPoint[0] = p.x;
//			dPoint[1] = p.y;
//			dPoint[2] = 0;
//			getClosestPointP2L(dPoint, dLinePoint, dLineVec, dPtOut);
//			double dLen = sqrt(((p.x-dPtOut[0])*(p.x-dPtOut[0])+(p.y-dPtOut[1])*(p.y-dPtOut[1])));
//			if (dLen < dMin)
//			{
//				ptMin = p;
//				dMin = dLen;
//			}
//			if (dLen > dMax)
//			{
//				ptMax = p;
//				dMax = dLen;
//			}
//		}
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, dMin);
//		lua_pushnumber(L, dMax);
//		lua_pushnumber(L, ptMin.y);
//		lua_pushnumber(L, ptMin.x);
//		lua_pushnumber(L, ptMax.y);
//		lua_pushnumber(L, ptMax.x);
//		return 7;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: math_distLineCont 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		lua_pushnumber(L, 0);
//		return 7;
//	}
//}
//
//
//
///**************************************************
//iErr = rect_getBoxPts(dRow,dCol,dAngle,dL1,dL2)
//功能：
//获取旋转矩形的四个顶点和角度，四个顶点的顺序依次是左上，右上，右下，左下，注意角度与halcon类型一致[]

//限制：

cv::RotatedRect imgPro::rect_build(cv::Size size, cv::Point centerPt, cv::Size rectSize, double angle)
{
	try
	{
		int halfWidth = rectSize.width / 2;
		int halfHeight = rectSize.height / 2;

		cv::Point tl = cv::Point(centerPt.x - halfWidth, centerPt.y - halfHeight);
		cv::Point tr = cv::Point(centerPt.x + halfWidth, centerPt.y - halfHeight);
		cv::Point dl = cv::Point(centerPt.x - halfWidth, centerPt.y + halfHeight);
		cv::Point dr = cv::Point(centerPt.x + halfWidth, centerPt.y + halfHeight);

		vector<cv::Point> vecPts{ tl,tr,dr,dl };
		vector<cv::Point> vecPtsRot;

		cv::Point ptOut;
		for (cv::Point &pt : vecPts)
		{
			pt_rotate(size, pt, centerPt, ptOut, angle);
			vecPtsRot.push_back(ptOut);
		}
		return cv::minAreaRect(vecPtsRot);
	}
	catch (const std::exception& e)
	{
		string err = "img_drawRect:发生异常, " + string(e.what());
		return cv::RotatedRect();
	}
}
//矩形四个顶点
vector<cv::Point> imgPro::rect_pts(cv::Size size, cv::Point centerPt, cv::Size rectSize, double angle)
{
	try
	{
		int halfWidth = rectSize.width / 2;
		int halfHeight = rectSize.height / 2;

		cv::Point tl = cv::Point(centerPt.x - halfWidth, centerPt.y - halfHeight);
		cv::Point tr = cv::Point(centerPt.x + halfWidth, centerPt.y - halfHeight);
		cv::Point dl = cv::Point(centerPt.x - halfWidth, centerPt.y + halfHeight);
		cv::Point dr = cv::Point(centerPt.x + halfWidth, centerPt.y + halfHeight);

		vector<cv::Point> vecPts{ tl,tr,dr,dl };
		vector<cv::Point> vecPtsRot;

		cv::Point ptOut;
		for (cv::Point &pt : vecPts)
		{
			pt_rotate(size, pt, centerPt, ptOut, angle);
			vecPtsRot.push_back(ptOut);
		}
		return vecPtsRot;
	}
	catch (const std::exception& e)
	{
		string err = "img_drawRect:发生异常, " + string(e.what());
		return vector<cv::Point>();
	}
}

//***************************************************/
int imgPro::rect_getRotRectPts(const cv::RotatedRect rotRect,vector<cv::Point2f> &vecPts,float &angle)
{
	try
	{

		cv::Mat v;
		cv::Point2f pts[4];

		rotRect.points(pts);
		vecPts.clear();
		vecPts.insert(vecPts.begin(),begin(pts),end(pts));
		
		if (abs(vecPts[0].x - vecPts[2].x) < abs(vecPts[0].y- vecPts[2].y))
		{
			sort(vecPts.begin(), vecPts.end(), [](cv::Point2f pt1, cv::Point2f pt2) {return pt1.y < pt2.y; });
			sort(vecPts.begin(), vecPts.begin() + 2, [](cv::Point2f pt1, cv::Point2f pt2) {return pt1.x < pt2.x; });
			sort(vecPts.begin() + 2, vecPts.end(), [](cv::Point2f pt1, cv::Point2f pt2) {return pt1.x < pt2.x; });
			swap(vecPts[2], vecPts[3]);
		}
		else
		{
			sort(vecPts.begin(), vecPts.end(), [](cv::Point2f pt1, cv::Point2f pt2){return pt1.x < pt2.x; });
			sort(vecPts.begin(), vecPts.begin() + 2, [](cv::Point2f pt1, cv::Point2f pt2){return pt1.y < pt2.y; });
			sort(vecPts.begin() + 2, vecPts.end(), [](cv::Point2f pt1, cv::Point2f pt2){return pt1.y < pt2.y; });
			swap(vecPts[1], vecPts[2]);
			swap(vecPts[2], vecPts[3]);

		}



		float angleTemp = rotRect.angle;
		if (rotRect.size.width < rotRect.size.height)
		{
			angle = -(90 + angleTemp);
		}
		else
		{
			angle = -angleTemp;
		}
		return 1;

	}
	catch (...)
	{
		string strErr = "imgPro: rect_getBoxPts 捕获到C++异常！";
		return -1;
	}

}
int imgPro::rect_getRotRectPts(const cv::RotatedRect rotRect,vector<cv::Point> &vecPts,float &angle)
{
	try
	{

		cv::Mat v;
		cv::Point2f pts[4];

		rotRect.points(pts);
		vecPts.clear();
		vecPts.insert(vecPts.begin(),begin(pts),end(pts));
		
		if (abs(vecPts[0].x - vecPts[2].x) < abs(vecPts[0].y- vecPts[2].y))
		{
			sort(vecPts.begin(), vecPts.end(), [](cv::Point pt1, cv::Point pt2) {return pt1.y < pt2.y; });
			sort(vecPts.begin(), vecPts.begin() + 2, [](cv::Point pt1, cv::Point pt2) {return pt1.x < pt2.x; });
			sort(vecPts.begin() + 2, vecPts.end(), [](cv::Point pt1, cv::Point pt2) {return pt1.x < pt2.x; });
			swap(vecPts[2], vecPts[3]);
		}
		else
		{
			sort(vecPts.begin(), vecPts.end(), [](cv::Point pt1, cv::Point pt2){return pt1.x < pt2.x; });
			sort(vecPts.begin(), vecPts.begin() + 2, [](cv::Point pt1, cv::Point pt2){return pt1.y < pt2.y; });
			sort(vecPts.begin() + 2, vecPts.end(), [](cv::Point pt1, cv::Point pt2){return pt1.y < pt2.y; });
			swap(vecPts[1], vecPts[2]);
			swap(vecPts[2], vecPts[3]);

		}



		float angleTemp = rotRect.angle;
		if (rotRect.size.width < rotRect.size.height)
		{
			angle = -(90 + angleTemp);
		}
		else
		{
			angle = -angleTemp;
		}
		return 1;

	}
	catch (...)
	{
		string strErr = "imgPro: rect_getBoxPts 捕获到C++异常！";
		return -1;
	}

}
//len:半边扩大的尺寸
cv::Rect imgPro::rect_enlarge(cv::Size imgSize, cv::Rect rect, int len)
{
	try
	{
		cv::Rect r(rect.tl().x - len, rect.tl().y - len, rect.width + 2 * len, rect.height + 2 * len);
		cv::Rect rnorm = imgPro::rect_normInImg(r, imgSize);
		return rnorm;
	}
	catch (const std::exception&)
	{
		return cv::Rect();
	}
}

cv::Rect imgPro::rect_enlarge(cv::Size imgSize, cv::Rect rect, int widthLen,int heightLen)
{
	try
	{
		cv::Rect r(rect.tl().x - widthLen, rect.tl().y - heightLen, rect.width + 2 * widthLen, rect.height + 2 * heightLen);
		cv::Rect rnorm = imgPro::rect_normInImg(r, imgSize);
		return rnorm;
	}
	catch (const std::exception&)
	{
		return cv::Rect();
	}
}

cv::Rect imgPro::rect_normInImg(const cv::Rect rect, cv::Size size)
{
	try
	{
		cv::Rect rectTmp = rect;
		if (rect.x < 0)
		{
			rectTmp.width = rectTmp.width - abs(rect.x);
			rectTmp.x = 0;
		}
		if (rect.y < 0)
		{
			rectTmp.height = rectTmp.height - abs(rect.y);
			rectTmp.y = 0;
		}
		if (rectTmp.x + rect.width > size.width) rectTmp.width = size.width - rectTmp.x ;
		if (rectTmp.y + rect.height > size.height) rectTmp.height = size.height - rectTmp.y ;

		
		return rectTmp;
	}
	catch (...)
	{
		return cv::Rect(-1,-1,-1,-1);
	}
}

int imgPro::rect_intersection(const cv::Rect rectA, const cv::Rect rectB)
{
	try
	{
		if (rectA.x > rectB.x + rectB.width) { return 0.; }
		if (rectA.y > rectB.y + rectB.height) { return 0.; }
		if ((rectA.x + rectA.width) < rectB.x) { return 0.; }
		if ((rectA.y + rectA.height) < rectB.y) { return 0.; }
		float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
		float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
		float intersection = colInt * rowInt;

		return intersection;

	}
	catch (...)
	{
		return -1;
	}
	return 0;
}


//随机颜色
cv::Scalar imgPro::randomColor(cv::RNG& rng)
{
	int icolor = (unsigned)rng;
	return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

//theta :单位为角度
int imgPro::pt_rotate(cv::Size size, cv::Point ptIn, cv::Point rotCenter, cv::Point & ptOut, double theta)
{
	try
	{
		int row = size.height;
		int col = size.width;
		int x1 = ptIn.x;
		int y1 = ptIn.y;
		int x2 = rotCenter.x;
		int y2 = rotCenter.y;
		int x = ptOut.x;
		int y = ptOut.y;

		x1 = x1;
		y1 = row - y1;
		x2 = x2;
		y2 = row - y2;
		x = (x1 - x2)*cos(CV_PI / 180.0 * theta) - (y1 - y2)*sin(CV_PI / 180.0 * theta) + x2;
		y = (x1 - x2)*sin(CV_PI / 180.0 * theta) + (y1 - y2)*cos(CV_PI / 180.0 * theta) + y2;
		x = x;
		y = row - y;

		ptOut.x = x;
		ptOut.y = y;

		return 0;	
	}
	catch (...)
	{
		string strErr = "imgPro: pt_rotate 捕获到C++异常！";
		return -1;
	}

}

cv::Point imgPro::segment_extend(cv::Size imgSize, cv::Vec4f lineSeg, cv::Point extendPt, double len)
{
	cv::Point newPt,farPt;
	cv::Point pt1(lineSeg[0], lineSeg[1]);
	cv::Point pt2(lineSeg[2], lineSeg[3]);
	if (imgPro::dist2Pts(pt1,extendPt) > imgPro::dist2Pts(pt2, extendPt))
	{
		farPt = pt1;
	}
	else
	{
		farPt = pt2;
	}
	double segLen = imgPro::dist2Pts(pt1,pt2);
	newPt.x = extendPt.x + (extendPt.x - farPt.x) / segLen * len;
	newPt.y = extendPt.y + (extendPt.y - farPt.y) / segLen * len;
	if (newPt.x < 0 || newPt.x >= imgSize.width || newPt.y < 0 || newPt.y >= imgSize.height)
	{
		return cv::Point(-1, -1);
	}
	return newPt;
}

//
//
///**************************************************
//iErr,dRow,dCol,dR = circle_getCircleInfo(iIndex)
//功能：
//获取圆的参数
//限制：
// 无
//参数：
//iIndex - 圆序号
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow - 圆中心点行坐标
//dCol - 圆中心点列坐标
//dR - 圆的半径
//***************************************************/
//int imgPro::circle_getCircleInfo(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 1 
//			|| lua_type (L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: circle_getCircleInfo 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 4;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);//
//		iIndex -= 1;
//		// 参数检查
//		if (iIndex >= g_circles.size()||iIndex<0)
//		{
//			string strErr = "imgPro: circle_getCircleInfo 索引范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 4;
//		}
//
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, g_circles[iIndex][0]);
//		lua_pushinteger(L, g_circles[iIndex][1]);
//		lua_pushinteger(L, g_circles[iIndex][2]);
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: circle_getCircleInfo 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		return 4;
//	}
//	return 4;
//}
//
///**************************************************
//iErr,dRow,dCol = point_getPointInfo(iIndex)
//功能：
//获取点的参数
//限制：
//无
//参数：
//iIndex - 点序号，从1开始
//返回值：
//iErr - 0,正常； 非0，有错误
//dRow - 点行坐标
//dCol - 点列坐标
//***************************************************/
//int imgPro::point_getPointInfo(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		// 参数检查
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: point_getPointInfo 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 3;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);//
//		iIndex -= 1;
//		// 参数检查
//		if (iIndex >= g_points.size()|| iIndex<0)
//		{
//			string strErr = "imgPro: point_getPointInfo 索引范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 3;
//		}
//
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, g_points[iIndex][1]);
//		lua_pushinteger(L, g_points[iIndex][0]);
//
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: point_getPointInfo 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//		lua_pushinteger(L, 0);
//		return 3;
//	}
//	return 3;
//}
//
///**********************************************************
//nRet = laser_fromImage(strImgIn,strLaserOut,iOrientation,int startPos ,int endPos)
//
//输入
//strImgIn		   -  输入二值图像
//strImgOut	   -  输出的轮廓
//iOrientation   -  扫描方向（图像四周）,0-由上往下，1-由下往上，2 - 由左往右，3 - 由右往左
//startPos       -  开始找点的位置
//endPos         -  结束找点的位置
//**********************************************************/
int imgPro::laser_fromImage(cv::Mat &imgIn,vector<cv::Point> &profileOut,int oritation,int startPos,int endPos,bool withZero,bool clearInvalidPt)
{
    try
    {
        if (imgIn.empty() || oritation <0 || oritation>3)
        {
            string strErr = "imgPro: laser_fromImage 中输入图像不存在或提取方向错误！";
            return -1;
        }
		profileOut.clear();

		if (oritation == 1 || oritation ==0)
		{
			if (startPos< 0 || endPos >imgIn.cols || startPos >endPos)
			{
				string strErr = "imgPro: laser_fromImage 中输入提取范围错误！";
				return -1;
			}
		}
		else
		{
			if (startPos< 0 || endPos >imgIn.rows || startPos >endPos)
			{
				string strErr = "imgPro: laser_fromImage 中输入提取范围错误！";
				return -1;
			}
		}
	
        const int IMG_WIDTH = imgIn.cols;
        const int IMG_HEIGHT = imgIn.rows;
		if (oritation < 2)//上下扫描
		{
			profileOut.assign(IMG_WIDTH, INVALID_POINT);
		}
		else
		{
			profileOut.assign(IMG_HEIGHT, INVALID_POINT);
		}


        switch (oritation)
        {
        case 0://从上往下
            for (int x = startPos; x < endPos; x++)
            {
                for (int y = 0; y < IMG_HEIGHT; y++)
                {
                    if (imgIn.at<uchar>(y, x)>0)
                    {
                        profileOut[x] = cv::Point(x, y);
                        break;
                    }
					if (y == IMG_HEIGHT-1 && withZero)
					{
						profileOut[x] = cv::Point(x, 0);
					}
                }

            }
            break;
        case 1://从下往上
            for (int x = startPos; x < endPos; x++)
            {
                for (int y = IMG_HEIGHT - 1; y >= 0; y--)
                {
                    if (imgIn.at<uchar>(y, x)>0)
                    {
                        profileOut[x] = cv::Point(x, y);
                        break;
                    }
					if (y == 0 && withZero)
					{
						profileOut[x] = cv::Point(x, IMG_HEIGHT - 1);
					}

                }
            }
            break;
        case 2:   //从左往右
            profileOut.resize(IMG_HEIGHT);
            for (int r = startPos; r < endPos; r++)
            {
                for (int c = 0; c <IMG_WIDTH; c++)
                {
                    if (imgIn.at<uchar>(r, c) > 0)
                    {
                        profileOut[r] = cv::Point(c, r);
                        break;
                    }
					if (c == IMG_WIDTH - 1 && withZero)
					{
						profileOut[r] = cv::Point(IMG_WIDTH-1, r);
					}
                }
            }
            break;
        case 3: //从右往左
            profileOut.resize(IMG_HEIGHT);
            for (int r = startPos; r < endPos; r++)
            {
                for (int c = IMG_WIDTH - 1; c >= 0; c--)
                {
                    if (imgIn.at<uchar>(r, c) > 0)
                    {
                        profileOut[r] = cv::Point(c, r);
                        break;
                    }
					if (c == 0 && withZero)
					{
						profileOut[r] = cv::Point(0,r );
					}
                }
            }
            break;
        default:
            break;
        }
		if (clearInvalidPt)
		{
			for (auto ite=profileOut.begin();ite != profileOut.end();)
			{
				if (ite->x < 0)
				{
					ite = profileOut.erase(ite);
					
				}
				else
				{
					ite++;
				}
			}
		}
        return 0;
    }
    catch (...)
    {
        string strErr = "imgPro: laser_fromImage 捕获到C++异常！";
        return -2;
    }
}
int imgPro::laser_fromImage(cv::Mat & imgIn, vector<cv::Point>& laserPts, cv::Range rowRange, cv::Range colRange, bool withZero, bool clearInvalidPt)
{
	try
	{
		int startPos = min(colRange.start, colRange.end);
		int endPos = max(colRange.start, colRange.end);

		//if (rowRange.start > rowRange.end && colRange.start > colRange.end)//
		//{
		//	imgPro::laser_fromImage(imgIn, laserPts, , startPos, endPos, withZero, clearInvalidPt);
		//}
		//else if (rowRange.start > rowRange.end && colRange.start < colRange.end)
		//{
		//	imgPro::laser_fromImage(imgIn, laserPts, , startPos, endPos, withZero, clearInvalidPt);
		//}
		//else if (rowRange.start < rowRange.end && colRange.start > colRange.end)
		//{
		//	imgPro::laser_fromImage(imgIn, laserPts,3 , startPos, endPos, withZero, clearInvalidPt);
		//}
		//else
		//{
		//	imgPro::laser_fromImage(imgIn, laserPts, , startPos, endPos, withZero, clearInvalidPt);
		//}
		return 0;
	}
	catch (const std::exception&)
	{
		return 1;
	}

}
//
///**************************************************
//nRet = img_drawLaser(strImgIn,strImgOut,strProfileIn,iThickness)
//功能：
//绘制指定轮廓线到指定输出图像
//限制：
//脚本调用
//参数：
//strImgIn    - 输入图像
//strImgOut - 输出图像
//strProfileIn	  - 输入轮廓
//iThickness - 绘制宽度
//返回值：
//nRet  - 0,正常； 非0，有错误
//***************************************************/
int imgPro::img_drawLaser(const cv::Mat &imgIn, cv::Mat &imgOut, const vector<cv::Point2d>& profileIn, int thickness)
{
	try
	{
		if (imgIn.empty()|| profileIn.empty() || thickness<0)
        {
            return -1;
        }

        if ( imgIn.channels() == 1)
        {
            cvtColor(imgIn, imgOut, cv::COLOR_GRAY2BGR);
        }
		else
		{
			imgOut = imgIn.clone();
		}

        vector < vector <cv::Point >> conProfile;
        getContinuousProfile(profileIn, conProfile);

        for (int i = 0; i < conProfile.size(); i++)
        {
            vector<cv::Point> temp(conProfile[i].begin(), conProfile[i].end());
            cv::polylines(imgOut, temp, false, cv::Scalar(0, 0, 255), thickness);
        }

        return 0;
    }
    catch (...)
    {
        string strErr = "imgPro: img_drawLaser 捕获到C++异常！";
        return -2;
    }
}

int imgPro::img_drawLaser(const cv::Mat &imgIn, cv::Mat &imgOut, const vector<cv::Point>& profileIn, int thickness)
{
	try
	{
		if (imgIn.empty() || profileIn.empty() || thickness < 0)
		{
			return -1;
		}

		if (imgIn.channels() == 1)
		{
			cvtColor(imgIn, imgOut, cv::COLOR_GRAY2BGR);
		}
		else
		{
			imgOut = imgIn.clone();
		}

		vector < vector <cv::Point >> conProfile;
		//getContinuousProfile(profileIn, conProfile);

			vector<cv::Point> temp(profileIn.begin(), profileIn.end());
			temp.erase(remove_if(temp.begin(), temp.end(), [](auto pt) {return pt.x < 0; }),temp.end());
			cv::polylines(imgOut, temp, false, cv::Scalar(0, 0, 255), thickness);

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: img_drawLaser 捕获到C++异常！";
		return -2;
	}
}
//
//
///*
//nRet,tablePoints = laser_getPolyline(strLaserIn,dEpsilon,iIsClosed)
//输入：
//strLaserIn -  输入轮廓，点可以是浮点型
//dEpsilon   - 指定逼近精度， 是原始曲线与逼近线段的最大距离
//iIsClosed  - 是否把轮廓看做首尾相连的闭合轮廓，一般用于处理旋转扫描物体
//输出：
//tablePoints - 线段折弯点
//*/
int imgPro::laser_getPolyline(const vector<cv::Point2d> &profileIn, vector<cv::Point2d> &polyPts,double epsilon,bool isClosed)
{
    try
    {

        if (profileIn.empty() || epsilon < 0)
        {
            string strErr = "imgPro: laser_getLocalExtrepoint 输入轮廓不存在或精度参数错误！";
            return -1;
        }
        vector < vector <cv::Point>> conProfile;
		vector <cv::Point> temp;
        //getContinuousProfile(profileIn, conProfile);
		polyPts.clear();
		for (cv::Point pt:profileIn)
		{
			temp.push_back(cv::Point(pt.x,pt.y));
		}
		conProfile.push_back(temp);
        vector<vector<cv::Point>> profilePoints(conProfile.size());
        for (int i = 0; i < conProfile.size(); i++)
        {
            cv::approxPolyDP(conProfile[i], profilePoints[i], epsilon, isClosed);
        }
        for (int i = 0; i < profilePoints.size(); i++)
        {
			polyPts.insert(polyPts.end(), profilePoints[i].begin(), profilePoints[i].end());
        }
        return 0;
    }
    catch (...)
    {
        string strErr = "imgPro: laser_getPolyline 捕获到C++异常！";
        return -2;

    }
}

//重载
int imgPro::laser_getPolyline(const vector<cv::Point> &profileIn, vector<cv::Point> &polyPts, double epsilon, bool isClosed)
{
	try
	{

		if (profileIn.empty() || epsilon < 0)
		{
			string strErr = "imgPro: laser_getLocalExtrepoint 输入轮廓不存在或精度参数错误！";
			return -1;
		}
		vector < vector <cv::Point>> conProfile;
		vector <cv::Point> temp;
		//getContinuousProfile(profileIn, conProfile);
		polyPts.clear();
		for (cv::Point pt : profileIn)
		{
			temp.push_back(cv::Point(pt.x, pt.y));
		}
		conProfile.push_back(temp);
		vector<vector<cv::Point>> profilePoints(conProfile.size());
		for (int i = 0; i < conProfile.size(); i++)
		{
			cv::approxPolyDP(conProfile[i], profilePoints[i], epsilon, isClosed);
		}
		for (int i = 0; i < profilePoints.size(); i++)
		{
			polyPts.insert(polyPts.end(), profilePoints[i].begin(), profilePoints[i].end());
		}
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: laser_getPolyline 捕获到C++异常！";
		return -2;

	}
}

//// mode 0:随机采样一致性算法  1:最小二乘法
//int imgPro::laser_fitCircle(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);      // 参数个数
//        // 参数检查
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: laser_fitCircle 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        string strIn = lua_tostring(L, 1);// 第二个参数为 轮廓
//        int mode = (int)lua_tonumber(L, 2);// 第二个参数为 模式
//        // 参数检查
//        if (mapProfile.find(strIn) == mapProfile.end() || mode <0 ||  2<mode)
//        {
//            string strErr = "imgPro: laser_fitCircle 输入轮廓 ";
//            strErr += strIn;
//            strErr += " 不存在！或模式范围错误";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        double row, col, r;
//        vector<cv::Point> laser;
//        cv::Mat(mapProfile[strIn]).convertTo(laser, CV_32S);
//
//        if (1 == mode)
//        {
//	        if (0 != LeastSquareFittingCircle(laser, col, row, r))
//	        {
//	            string strErr = "imgPro: cont_fitCircle 执行回归函数异常！";
//	            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//	            lua_pushinteger(L, -100);
//	            lua_pushnumber(L, 0);
//	            lua_pushnumber(L, 0);
//	            lua_pushnumber(L, 0);
//	            return 4;
//	        }
//        } 
//        else if (0== mode)
//        {
//			
//        }
//		else if (2 == mode)
//		{
//			cv::Point2f minEnclosingCenter;
//			float minClosingRadius;
//			cv::minEnclosingCircle(laser, minEnclosingCenter, minClosingRadius);
//			row = minEnclosingCenter.y;
//			col = minEnclosingCenter.x;
//			r = minClosingRadius;
//		}
//
//        lua_pushinteger(L, 0);
//        lua_pushnumber(L, row);
//        lua_pushnumber(L, col);
//        lua_pushnumber(L, r);
//        return 4;
//    }
//    catch (...)
//    {
//        string strErr = "imgPro: cont_fitCircle 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        lua_pushnumber(L, 0);
//        return 4;
//    }
//
//}
//
////nRet ,cnt = imgPro.laser_concatenate(laserIn1, laserIn2 , laserOut)
//int imgPro::laser_concatenate(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//        if (iCount != 3
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TSTRING
//            || lua_type(L, 3) != LUA_TSTRING)
//        {
//            string strErr = "imgPro: laser_concatenate 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 2;
//        }
//
//        string strLaserIn1 = lua_tostring(L, 1);
//        string strLaserIn2 = lua_tostring(L, 2);
//        string strLaserOut = lua_tostring(L, 3);
//
//        if (mapProfile.find(strLaserIn1) == mapProfile.end())
//        {
//            string strErr = "imgPro: laser_concatenate 输入轮廓1不存在!";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 2;
//        }
//        if (mapProfile.find(strLaserIn2) == mapProfile.end())
//        {
//            string strErr = "imgPro: laser_concatenate 输入轮廓2不存在!";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 2;
//        }
//
//        const  vector<cv::Point2d>  &laser1 = (mapProfile)[strLaserIn1];
//        const  vector<cv::Point2d>  &laser2 = (mapProfile)[strLaserIn2];
//
//        if (mapProfile.find(strLaserOut) == mapProfile.end())
//        {
//            mapProfile[strLaserOut] = vector<cv::Point2d>();
//        }
//
//        vector<cv::Point2d> & laserOut = mapProfile[strLaserOut];
//        laserOut.clear();
//        laserOut = laser1;
//        laserOut.insert(laserOut.end(), laser2.begin(), laser2.end());
//
//        lua_pushinteger(L, 0);
//        lua_pushinteger(L, laserOut.size());
//        return 2;
//    }
//    catch (...)
//    {
//        string strErr = "imgPro: laser_concatenate 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        return 2;
//    }
//}
//
//
///**************************************************
//iErr,Points = laser_getPoints(laserContIn)
//功能：
//获取二值化图像中指定轮廓包含的点
//限制：
//无
//参数：
//laserContIn - 输入轮廓
//返回值：
//iErr - 0,正常； 非0，有错误
//Points - 包含构成指定轮廓的点的Table，双层Table,每个point下还有一个Table存放对应的row,col
//***************************************************/
//int imgPro::laser_getPoints(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//		//参数检查
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: laser_getPoints 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//		string strLaserIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//
//		if (mapProfile.find(strLaserIn) == mapProfile.end())
//		{
//			string strErr = "imgPro: laser_concatenate 输入轮廓不存在!";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//		const  vector<cv::Point2d>  &laserIn = (mapProfile)[strLaserIn];
//
//		lua_pushinteger(L, 0);
//		lua_createtable(L, laserIn.size(), 0);
//		for (int i = 0; i < laserIn.size(); i++)
//		{
//			lua_pushnumber(L, i + 1);
//			lua_createtable(L, 0, 2);
//			lua_pushnumber(L, laserIn[i].y);
//			lua_setfield(L, -2, "row");
//			lua_pushnumber(L, laserIn[i].x);
//			lua_setfield(L, -2, "col");
//			lua_settable(L, -3);
//		}
//
//		return 2;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: laser_getPoints 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}
//
//
//
//
//
////nRet,time = getTime()
////获取系统当前时间
//int imgPro::sys_getTime(lua_State *L)
//{
//	try
//	{
//		time_t t = time(0);
//		char tmp[32] = { NULL };
//		strftime(tmp, sizeof(tmp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
//
//		lua_pushinteger(L, 0);
//		lua_pushstring(L, tmp);
//		return 2;
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: sys_getTime 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushstring(L,"");
//		return 2;
//
//	}
//}
//
///**************************************************
//int test(lua_State* L)
//功能：
//测试用函数
//限制：
//无
//参数：
//见代码注释
//返回值：
//无
//***************************************************/
//int test(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//
//		string strIn = lua_tostring(L, 1);	// 第1个参数为输入图像
//		string strOut = lua_tostring(L, 2);	// 
//		(*g_pMapImage)[strOut] = cv::Mat();
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		cv::Mat & mOut = (*g_pMapImage)[strOut];
//		cv::Mat mRGB = cv::Mat::zeros(mIn.size(), mIn.type());
//		cv::cvtColor(mIn, mRGB, cv::COLOR_GRAY2RGB);
//		std::vector<cv::Vec3f> circles;
//		cv::HoughCircles(mIn, circles, CV_HOUGH_GRADIENT, 2, 50, 200, 100, 40, 83);
//		std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
//		while (itc != circles.end())
//		{
//			cv::circle(mRGB, cv::Point((*itc)[0], (*itc)[1]), (*itc)[2], cv::Scalar(0, 0, 255), 2);
//			line(mRGB, cv::Point((*itc)[0] + 20, (*itc)[1]), cv::Point((*itc)[0] - 20, (*itc)[1]), cv::Scalar(255, 0, 0));
//			line(mRGB, cv::Point((*itc)[0], (*itc)[1] + 20), cv::Point((*itc)[0], (*itc)[1] - 20), cv::Scalar(255, 0, 0));
//			++itc;
//		}
//		mOut = mRGB.clone();
//
//
//	}
//	catch (cv::Exception e)
//	{
//		int i = 0;
//		i = 1;
//	}
//	catch (...)
//	{
//		int i = 0;
//		i = 1;
//	}
//
//
//	return 0;
//}

//void sobelXY()
//{
//
//	cvtColor(src, gray, CV_BGR2GRAY);
//	// Canny(src, res, 50, 200, 3);
//	Sobel(gray, x, CV_16S, 1, 0);
//	Sobel(gray, y, CV_16S, 0, 1);
//
//	convertScaleAbs(x, resX);
//	convertScaleAbs(y, resY);
//
//	addWeighted(resX, 0.5, resY, 0.5, 0, res);
//}


cv::Scalar Color::RED = cv::Scalar(0, 0, 255);
cv::Scalar Color::BLUE = cv::Scalar(255, 0, 0);
cv::Scalar Color::GREEN = cv::Scalar(0, 255, 0);
cv::Scalar Color::getRandomColor()
{
	cv::RNG cousRng((unsigned)time(NULL));

	return cv::Scalar(cousRng.uniform(0, 255), cousRng.uniform(0, 255), cousRng.uniform(0, 255));
}