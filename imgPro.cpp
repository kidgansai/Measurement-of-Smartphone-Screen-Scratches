

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



// ��������
std::vector<std::vector<cv::Point>> g_contours;
std::vector<cv::Vec3f> g_circles;
std::vector<cv::Vec2f> g_points;

// �����ݻ���
vector<cv::Vec2f> g_Lines;
vector<cv::Vec4d> g_linesP;

// �߼������������
std::map<string, vector<cv::Point2d>> mapProfile;


//����������,string-labelImg name  , vector<vector<float>> -  7 features of region
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
		if (i == profile.size() - 1 && profile.back().x >0)  //��Ч����ͼƬ�ұ߽�
		{
			conProfile.push_back(object);
		}
	}
}

//**************************��������������Ҫ����ΪRANSAC_Circle��������****************************************//
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
//*************************************************����***********************************************************************//



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

//�Ӳ�ͬ�����ȡͼ����������
int C_GetProfile(const cv::Mat & imgIn, vector<cv::Point> & profileOut, int iOrient)
{
	try
	{
		const int IMG_WIDTH = imgIn.cols;
		const int IMG_HEIGHT = imgIn.rows;

		switch (iOrient)
		{
		case 0://��������
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
		case 1://��������
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
		case 2:   //��������
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
		case 3: //��������
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


//��ȡ����������
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

//���߶ν����и�,segΪ�и����Ƕ��ٿ�
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

//���߶ν����и�,segΪÿ���и�����;���
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
//��������ɸѡ����
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

//����㵽ֱ�ߵľ��룬���ص�λ
double distPt2Line(cv::Point lp1, cv::Point lp2, cv::Point pt)
{
	double a, b, c, dis;
	a = lp2.y - lp1.y;
	b = lp1.x - lp2.x;
	c = lp2.x * lp1.y - lp1.x * lp2.y;
	// ���빫ʽΪd = |A*x0 + B*y0 + C|/��(A^2 + B^2)
	dis = fabs(float((a * pt.x + b * pt.y + c) / std::sqrt(float(a * a + b * b))));
	return dis;
}
// End _imgPro


//����������ֵͼ���IOU
float calcRatio(cv::Mat &imgIn1, cv::Mat &imgIn2, float &ratio)
{
	if (imgIn1.empty() || imgIn2.empty())
	{
		cout << "no img" << endl;
		return -1;
	}
	cv::Mat imgIn2Bin, imgIn1Bin;       //��ֵͼ�����ڼ��������
	cv::threshold(imgIn1, imgIn1Bin, 100, 255, cv::THRESH_BINARY);
	cv::threshold(imgIn2, imgIn2Bin, 100, 255, cv::THRESH_BINARY);

	//�����������ͼ��׶���
	cv::Mat resImgFill = cv::Mat::zeros(imgIn1.size(), CV_8UC1);
	cv::Mat glassImgFill = cv::Mat::zeros(imgIn2.size(), CV_8UC1);

	vector<vector<cv::Point>>  conts;
	cv::findContours(imgIn1Bin, conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::drawContours(resImgFill, conts, -1, cv::Scalar(255, 255, 255), cv::FILLED);

	cv::findContours(imgIn2Bin, conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::drawContours(glassImgFill, conts, -1, cv::Scalar(255, 255, 255), cv::FILLED);

	//��ͼ���Ե����ƽ��
	cv::Mat morResImg, morGlassImg;
	dilate(resImgFill, morResImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
	erode(morResImg, morResImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));//�Ȼ�ȡ����������2�����صı�Ե����Ϊ��ȡͼ���Ǳ�Ե�����Կ�϶

	dilate(imgIn2Bin, morGlassImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(19, 19)));
	erode(morGlassImg, morGlassImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17)));

	cv::Mat imgAnd;
	bitwise_and(morResImg, morGlassImg, imgAnd);  //ȡʵ����ͼ�Ͳ����Ͽ�ͼ���ཻ��ͼ�񲿷�
	if (countNonZero(morGlassImg) == 0)
	{
		cout << "�ཻ���Ϊ0" << endl;
		return -1;
	}

	int cntImgAdd = countNonZero(imgAnd);
	int cntGlassImg = countNonZero(morGlassImg);
	return float(cntImgAdd) / cntGlassImg;

}

int  LeastSquareFittingCircle(vector<cv::Point> temp_coordinates, double &center_x, double &center_y, double & radius)//��˹��Ԫ��ֱ����ⷽ����
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
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 3
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: grayImg_compareHist ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		string strIn1 = lua_tostring(L, 1);	// ��1������Ϊ����1
//		string strIn2 = lua_tostring(L, 2);	// ��2������Ϊ����2
//		int iMethod = (int)lua_tonumber(L, 3);
//
//		// �������
//		if (g_MapHist.find(strIn1) == g_MapHist.end() || g_MapHist.find(strIn2) == g_MapHist.end())
//		{
//			string strErr = "imgPro: img_copy ������ֱ��ͼ ";
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: grayImg_compareHist ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ȡ�Ҷ�ͼ��ָ������ĻҶ���Сֵ�����ֵ�Լ���Сֵ�����ֵ֮��Ĳ�ֵ
//���ƣ�
//��
//������
//strImgIn - ����ͼ��
//iRowTL - ָ�������������ϵ�������
//iColTL - ָ�������������ϵ�������
//iRowBR -ָ�������������µ�������
//iRowBR - ָ�������������µ�������
//����ֵ��
//iErr - 0,������ ��0���д���
//dMin - �Ҷ���Сֵ
//dMax - �Ҷ����ֵ
//dRange - �Ҷ���Сֵ�����ֵ�ķ�Χ
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
//			string strErr = "imgPro: grayImg_getMinMaxValue������������!";
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
//			string strErr = "imgPro: grayImg_getMinMaxValue ����ͼ�����";
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
//		string strErr = "imgPro: grayImg_getMinMaxValue ����C++�쳣��";
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
//���ܣ�
//��ȡһ�����������ƽ���Ҷ�ֵ
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//iRow - �������Ͻǵ�������
//iCol - �������Ͻǵ�������
//iWidth - ���ο��
//iHeight - ���θ߶�
//iVal   -  ���ڸ���ֵ�������ƽ���Ҷ�
//����ֵ��
//iErr - 0,������ ��0���д���
//dMean - ƽ���Ҷ�ֵ
//***************************************************/
//int imgPro::grayImg_getMeanValue(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//            || lua_type(L, 5) != LUA_TNUMBER
//            || lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: grayImg_getMeanValue ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() )
//		{
//			string strErr = "imgPro: grayImg_getMeanValue ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			return 2;
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		if ( iRow + iHeight > mIn.rows  || iCol+iWidth > mIn.cols)
//		{
//			string strErr = "imgPro: grayImg_getMeanValue ���뷶Χ���� ";
//
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: grayImg_getMeanValue ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
// ����ģ��

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

// �����ռ������ļнǣ�����ֵ��λ��cosֵ
double imgPro::Get2VecAngleCos(double dLineDirVec1[], double dLineDirVec2[])
{
	return (dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
		/ (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
			*sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2]));
}

// �����ռ������ļнǣ�����ֵ��λ������
double imgPro::Get2VecAngle(double dLineDirVec1[], double dLineDirVec2[])
{
	return acos((dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
		/ (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
			*sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2])));
}

//dPluVec = dVector1 ��� dVector2
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

//�� ��ֱ�߷�������dLineDirVec������֪���ϵĵ�dKnownPointCoo����Ϊd2PointDistance��δ֪������dUnknowPointCoo
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

	double dAxisDirVec[3] = { 1, 0, 0 };	//X��ķ�������
	dUnknowPointCoo[0] = Get2VecAngleCos(dLineDirVec, dAxisDirVec)*d2PointDistance + dKnownPointCoo[0];
	dAxisDirVec[0] = 0; dAxisDirVec[1] = 1; dAxisDirVec[2] = 0;	//Y��ķ�������
	dUnknowPointCoo[1] = Get2VecAngleCos(dLineDirVec, dAxisDirVec)*d2PointDistance + dKnownPointCoo[1];
	dAxisDirVec[0] = 0; dAxisDirVec[1] = 0; dAxisDirVec[2] = 1;	//Z��ķ�������
	dUnknowPointCoo[2] = Get2VecAngleCos(dLineDirVec, dAxisDirVec)*d2PointDistance + dKnownPointCoo[2];
}

//��ռ���ֱ�ߵĽ��㣬dLine1DirVecΪֱ��1����������dLine1PointΪֱ��1��һ�㣨ֱ��2���ƣ�
//dIntersectionPointΪ��������
void imgPro::GetIntersectionFor2Line(double dLine1DirVec[], double dLine1Point[], double dLine2DirVec[],
	double dLine2Point[], double dIntersectionPoint[])
{
	double bta;	//ֱ�߷��̲������Ĳ���
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
		/ (dLine1DirVec[GC_Index2] * dLine2DirVec[GC_Index1] / dLine1DirVec[GC_Index1] - dLine2DirVec[GC_Index2]);	//������
	dIntersectionPoint[0] = dLine2DirVec[0] * bta + dLine2Point[0];
	dIntersectionPoint[1] = dLine2DirVec[1] * bta + dLine2Point[1];
	dIntersectionPoint[2] = dLine2DirVec[2] * bta + dLine2Point[2];
}

void imgPro::getLinePoints(double dRho, double dTheta, int iWidth, int iHeight, double & dP1Row, double & dP1Col, double & dP2Row, double & dP2Col)
{
	if (dTheta < PI / 4 || dTheta > 3.*PI / 4)//����ӽ��ڴ�ֱ��ֱ��
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
//���Ҷ�ά�ֲ���ֵ��Ϊģ��ƥ��������
vector<cv::Vec3f> imgPro::filterTemplateResPoints(const cv::Mat  & res, double dThresh, int rowDist, int colDist)
{
	int width = res.cols;
	int height = res.rows;
	float maxVal = 0.;
	vector<cv::Vec3f>  vecRes;   //���tableԪ����table��score,row,col����
	vector<cv::Vec3f>  colMaxTemp;
	int row, col;

	cv::Mat threshRes;
	cv::threshold(res, threshRes, dThresh, 0., cv::THRESH_TOZERO);
	for (col = 0; col < width; col++)
	{
		for (row = 0; row < height; row++)//�����еľֲ���ֵ��
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
		if (tempMaxVal > dThresh)  //�ж��з�ֵ��������ֵ���ҵ��ֲ���ֵ��
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

	//ȥ���ٽ������
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
���ܣ�
��ȡͼ���ָ����ת���β���
���ƣ�
��
������
imgIn - ����ͼ��
imgOut - ���ͼ��
dRow - �������ĵ�������
dCol - �������ĵ�������
dAngle - ������ת�Ƕȣ���λ���㣩
dL1 - ���ο��
dL2 - ���θ߶�
����ֵ��
iErr - 1,������ ��1���д���
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
		string strErr = "imgPro: img_getPartR ����C++�쳣��";
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
		string strErr = "imgPro: img_getPartR ����C++�쳣��";
		return -2;
	}

	return 0;
}

/******************************************************************
iErr = imgPro.img_rotate(strImgIn, strImgOut, dAngle, iFillValue, dCenterRow, dCenterCol)
���ܣ�
ͼ�ΰ���ָ��������(iCenterRow, iCenterCol)��ת�ض��ĽǶ�dAngle(��λ����)����ʱ��Ϊ����
���ƣ�
��
������
strImgIn - ͼ������
strImgOut - ͼ�����
dAngle - ��ת�Ƕ�
iFillValue - ͼ�����������ֵ
dCenterRow - ��ת����������y����
dCenterCol - ��ת����������x����
����ֵ��
iErr - 1, ������ ��1���д���
***************************************************/
int imgPro::img_rotate(cv::Mat &imgIn, cv::Mat &imgOut, double dAngle, uchar iFillValue, int centerRow, int centerCol)
{
	try
	{
		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
		if (imgIn.empty())
		{
			return -1;
		}

		//��ת����Ϊͼ������  
		cv::Point2f center;
		center.x = centerCol;
		center.y = centerRow;
		//�����ά��ת�ķ���任����  
		cv::Mat M = cv::getRotationMatrix2D(center, dAngle, 1);

		//�任ͼ�񣬲��ú�ɫ�������ֵ
		//cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) ); 
		cv::warpAffine(imgIn, imgOut, M, imgIn.size(), cv::INTER_LINEAR || cv::WARP_FILL_OUTLIERS, 0, cv::Scalar(iFillValue));
		//double d = w.Stop();
		return 1;

	}
	catch (...)
	{
		string strErr = "imgPro: img_rotate ����C++�쳣��";
		return -1;
	}
}
/******************************************************************
iErr = imgPro.img_translate(strImgIn, strImgOut, iRowTransl,iColTransl, iFillValue)
���ܣ�
ͼ��ƽ��
���ƣ�
��
������
strImgIn - ͼ������
strImgOut - ͼ�����
iRowTransl - ��ƽ��ֵ��ʹͼ������ƽ��
iColTransl   - ��ƽ��ֵ��ʹͼ������ƽ��
iFillValue - ͼ�����������ֵ
����ֵ��
iErr - 0, ������ ��0���д���
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
		t_mat.at<float>(0, 2) = iColTransl; //ˮƽ(��)ƽ����
		t_mat.at<float>(1, 1) = 1;
		t_mat.at<float>(1, 2) = iRowTransl; //��ֱ(��)ƽ����

											//����ƽ�ƾ�����з���任
		cv::warpAffine(imgIn, imgOut, t_mat, imgIn.size(), cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS, 0, cv::Scalar(iFillValue));
		//double d = w.Stop();
		return 1;

	}
	catch (...)
	{
		string strErr = "imgPro: img_translate ����C++�쳣��";
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
		cv::add(srcImage, cv::Scalar(1.0), srcImage);  //���� r+1
		srcImage.convertTo(srcImage, CV_32F);  //ת��Ϊ32λ������
		cv::log(srcImage, resultImage);            //����log(1+r)
		resultImage = c * resultImage;
		//��һ������
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
/*  ͼ��ȱ�����С         
���ͼ�񳤱�С��Size�����ֵ����ô�����ǽ��еȱ�����С��
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
//���ܣ�
//��ȡ��ֵ��ͼ����ָ�����������ĵ�
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//iMode - ��������ֵ����1��ʼ
//����ֵ��
//iErr - 0,������ ��0���д���
//Points - ��������ָ�������ĵ��Table��˫��Table,ÿ��point�»���һ��Table��Ŷ�Ӧ��row,col
//***************************************************/
//int imgPro::img_getContourPoints(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//		// �������
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_getContourPoints ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//		int iIndex = (int)lua_tonumber(L, 2);// �ڶ�������Ϊ����ֵ
//
//		iIndex -= 1;    //��������ֵ��1��ʼ
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() || iIndex < 0 || iIndex >= g_contours.size())
//		{
//			string strErr = "imgPro: img_getContourPoints ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�������ֵ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];  //����mIn�ǲ����ò��ö�û�в��
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
//		string strErr = "imgPro: img_getContourPoints ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}
//
//
///**************************************************
//iErr = img_drawContours(imgIn,imgOut,iIndex,iRed,iGreen,iBlue,iThickness)
//���ܣ�
//��ͼ���л�������
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//imgOut- ���ͼ��
//iIndex - �������: ��1��ʼ
//iRed - ��ɫ����
//iGreen - ��ɫ����
//iBlue - ��ɫ����
//iThickness - �������
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/
//int imgPro::img_drawContours(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
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
//			string strErr = "imgPro: img_drawContours ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//        string strOut = lua_tostring(L, 2);	// ��һ������Ϊ����ͼ��
//		int iIndex = (int)lua_tonumber(L, 3);// �ڶ�������Ϊ �������
//		int iRed = (int)lua_tonumber(L, 4); // 
//		int iGreen = (int)lua_tonumber(L, 5); // 
//		int iBlue = (int)lua_tonumber(L, 6); // 
//		int iThickness = (int)lua_tonumber(L, 7); // ��ʾ���
//
//		iIndex -= 1;
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  
//			|| iIndex<0 || iIndex>=g_contours.size())
//		{
//			string strErr = "imgPro: img_drawContours ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ���������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: img_drawContours ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//    ����ͼ����Harris�ǵ�
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//imgOut - ���ͼ��
//iBlock - ���򴰿ڴ�С
//iAperture - sobel��Ե��ⴰ�ڴ�С
//dK - ϵ��
//iThresh - �ǵ�ɸѡ��ֵ
//����ֵ��
//iErr - 0,������ ��0���д���
//iNum - �ǵ����
//***************************************************/
//int imgPro::img_findHarris(lua_State* L)
//{
//	try
//	{
//		//------------------------- ��������ͼ�� --------------------------
//		int iCount = lua_gettop(L);      // ��������
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_findHarris ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// ��2������Ϊ���ͼ��
//		int iBlock = (int)lua_tonumber(L, 3); //���򴰿ڴ�С
//		int iAperture = (int)lua_tonumber(L, 4);   //�׾���С
//		double dK = lua_tonumber(L, 5);
//		int iThresh = lua_tonumber(L, 6);
//
//		if ((*g_pMapImage).find(strIn) == (*g_pMapImage).end())	//���û��ͼ��
//		{
//			string strErr = "imgPro: img_findHarris ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		//------------------------- �㷨ʵ�ֲ��� --------------------------
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
//		string strErr = "imgPro: img_findHarris ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//		return 2;
//	}
//
//	//------------------------- ���ز��� --------------------------	
//
//	lua_pushinteger(L, 0);
//	lua_pushinteger(L, g_points.size());
//	return 2;
//
//}
//
///*************************************************************
//iErr = img_fillRectR(strImgIn,strImgOut,RectRow,RectCol,dAngle,RectWidth,RectHeigth)
//���ܣ�
//	����������Ϊ��ɫ���Ҷ�ֵ255��
//������
//	strImgIn������ͼ��
//	strImgOut�����ͼ��	
//	RectRow ����ת��������������
//	RectCol	����ת��������������
//	dAngle	����ת���νǶ�
//	RectWidth����ת���ο�
//	RectHeigth����ת���θ�
//
//***************************************************************/
//
//int imgPro::img_fillRectR(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 7
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER
//			|| lua_type(L, 7) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_fillRectR ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// �ڶ�������Ϊ���ͼ��
//		float row = (float)lua_tonumber(L, 3);
//		float col = (float)lua_tonumber(L, 4);
//		float angle = (float)lua_tonumber(L, 5);
//		float L1 = (float)lua_tonumber(L, 6);
//		float L2 = (float)lua_tonumber(L, 7);
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end())
//		{
//			string strErr = "imgPro: img_fillRectR ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
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
//		string strErr = "imgPro: img_fillRectR ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//�Զ����˲���
//���ƣ�
//��
//������
//imgIn - ͼ������
//imgOut - ͼ�����
//����Ϊ��������
//����ֵ��
//iErr - 0,������ ��0���д���
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
//			string strErr = "imgPro: img_filter ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// �ڶ�������Ϊ���ͼ��
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
//				string strErr = "imgPro: img_filter 3*3ģ���������";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//				string strErr = "imgPro: img_filter 5*5ģ���������";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//
//			string strErr = "imgPro: img_filter ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
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
//		string strErr = "imgPro: img_filter ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ͼ�����sobel��Ե���
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//imgOut - ���ͼ��
//iDX - X����ȡ1��0
//iDY - Y����ȡ1��0
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/
//int imgPro::img_sobel(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 4
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: img_sobel ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// �ڶ�������Ϊ���ͼ��
//		int iDX = (int)lua_tonumber(L, 3); // X����ȡ1��0
//		int iDY = (int)lua_tonumber(L, 4); // Y����ȡ1��0
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: img_sobel ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
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
//		string strErr = "imgPro: img_sobel ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ͼ�����prewitt��Ե���
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//imgOut - ���ͼ��
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/
//int imgPro::img_prewitt(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: img_prewitt ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// �ڶ�������Ϊ���ͼ��
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: img_prewitt ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
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
//			string strErr = "imgPro: img_prewitt �ӳ������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//	}
//	catch (...)
//	{
//		string strErr = "imgPro: img_prewitt ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ͼ����и�˹������˹��Ե���
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//imgOut - ���ͼ��
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/
//int imgPro::img_LOG(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: img_LOG ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// �ڶ�������Ϊ���ͼ��
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: img_LOG ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
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
//		string strErr = "imgPro: img_LOG ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//	ͼ��ͶӰ,���ֹ��ܿ���reduce�������
//���ƣ�
//��
//������
//imgIn - ͼ������
//strImgProj - ͶӰͼ�����
//iOrit  - ����ͶӰ�ķ���0 - ���м��� , 1 - ���м���
//iMode - ���㷽ʽ��0 : ���л��з�0���صĸ�����1�� ���л�������ֵ�� ��2 - ����
//iCalcImgProj - �Ƿ����ͶӰͼ��,0 :�����  1�����
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/

int imgPro::img_projection(cv::Mat& imgIn, cv::Mat &imgProj, std::vector<int> &vecProjVal,int iOrit, int iMode,bool iCalcImgProj)
{
	try
	{
		if (imgIn.empty() )
		{
			string strErr = "imgPro:img_projection ����ͼ�񲻴��ڣ�";
			return -2;
		}
		if (cv::countNonZero(imgIn) < 10)
		{
			return 1;
		}

		vecProjVal.clear();
		if (PRO_ORIENTATION::HORIZONTAL == iOrit)  //0Ϊˮƽ����
		{
			if (PRO_MODE::COUNTNOZERO == iMode)//��0���ظ���
			{
				for (int rowi = 0; rowi < imgIn.rows;rowi++)
				{
					vecProjVal.push_back(cv::countNonZero(imgIn.rowRange(rowi, rowi + 1)));
				}
			} 
			else //����ֵ��
			{
				for (int rowi = 0; rowi < imgIn.rows; rowi++)
				{
					vecProjVal.push_back(cv::sum(imgIn.rowRange(rowi, rowi + 1)).val[0]);
				}
			}
		} 
		else
		{
			if (PRO_MODE::COUNTNOZERO == iMode)//��0���ظ���
			{
				for (int coli = 0; coli < imgIn.cols; coli++)
				{
					vecProjVal.push_back(cv::countNonZero(imgIn.colRange(coli, coli + 1)));
				}
			}
			else //����ֵ��
			{
				for (int coli = 0; coli < imgIn.cols; coli++)
				{
					vecProjVal.push_back(cv::sum(imgIn.colRange(coli, coli + 1)).val[0]);
				}
			}

		}
		int img_rows = imgIn.rows;
		int img_cols = imgIn.cols;
		if (iCalcImgProj > 0 )//�������ӳ��ͼ
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
		//��ĸ�����һ������ͼ����ͬ��Χ
		//std::pair<vector<int>::iterator,vector<int>::iterator> iteMinMaxVal = std::minmax_element(vecProjVal.begin(), vecProjVal.end());
		//int minVal = *iteMinMaxVal.first;
		//int maxVal = *iteMinMaxVal.second;

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro:img_projection ����C++�쳣��";
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
		string err = "img_drawRect:�����쳣, " + string(e.what());
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
		string err = "img_drawRect:�����쳣, " + string(e.what());
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

//ͨ������srcȷ��
//���룺cv::Vec4f lineΪfitLine����Ͻ��
//�����cv::Vec4f Ϊֱ����ͼ������ĩ��
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
		string err = "imgPro::img_getGrayLevel �з�������";
		return -1;
	}
	return 0;
}


//
///******************************************************************
//iErr,resTable =find_template(strImg,strTemplImg,iMethod, dThreshRes,iRowDistFilter,iColDistFilter,strMaskImg)
//���ܣ�
//	��һ��ͼ���в���ģ��ͼ��
//���ƣ�
//	�ű�����
//������
//strImg			  - ͼ������,���ڲ���ģ��
//strImgTempl - ͼ��ģ������
//iMethod		  - ģ����ҷ���,0-5��6��
//dThreshRes  - ���ɸѡ��ֵ����ΧΪ0-1,��ֵԽ��Խ����,Ϊ1ʱ��ȡ���ֵ
//iRowDistFilter��iColDistFilter - ���С��iRowDistFilter��iColDistFilter�Ľ���У�ֻ����ƥ�������Ľ��
//strMaskImg  - ��Ĥͼ��,�ߴ������ģ��ͼ����ȣ���ѡ������
//����ֵ��
//iErr		   - 0, ������ ��0���д���
//resTable - ����table��Ԫ��˳����scoreֵ�������С�
//				��ʽΪ{ [1] = {[score] = val1, [row] = val2, [col] = val3 },
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
			string strErr = "imgPro: img_findTemplate ����ͼ���ģ��Ϊ�գ�";
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
			if (iMethod == cv::TM_SQDIFF || iMethod == cv::TM_SQDIFF_NORMED)  //ȡ��Сֵ
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
		string strErr = "imgPro: img_rotate ����C++�쳣��";
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
//���ܣ�
//��ȡ�Ҷ�ͼ��ĻҶ�ֱ��ͼ����
//���ƣ�
//��
//������
//strIn - ����ͼ��
//strOut - ������ݽ��
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/
int imgPro::grayImg_getHist(cv::Mat &grayImgIn,cv::Mat &histImgOut)
{
	try
	{
		// �������
		if (grayImgIn.empty()  )
		{
			string strErr = "imgPro: grayImg_getHist ����ͼ�� ";
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
		string strErr = "imgPro: img_copy ����C++�쳣��";
		return -1;
	}
	return 0;
}
///*******************************************************
//iErr = rgbImg_threshold(strImgIn, strImgOut, dBcoef,dGcoef,dRcoef,iThresh, iMax,iType)
//���ܣ�
//	�Բ�ɫͼ������ؽ�����ֵ����B*dBcoef+G*dGcoef+R*dRcoef > iThresh ��������ΪiMax��0
//���ƣ�
//	��ɫͼ������
//������
//	strImgIn - ͼ������
//	strImgOut - ͼ�����
//	dBcoef,dGcoef,dRcoef - ��ɫͼ�����ص�B,G,R��Ӧ������ϵ��,B*dBcoef+G*dGcoef+R*dRcoefֵ����iThresh���бȽϡ�
//	iThresh - ��ֵ
//	iMax - ���ֵ
//	iType - ��ֵ��������: 0 : ����iThresh ��������ΪiMax����������Ϊ0, 1 : ����iThresh ��������Ϊ0����������ΪiMax��
//����ֵ��
//	iErr - 0, ������ ��0���д���
//
//***************************************************/
//int imgPro::rgbImg_threshold(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
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
//			string strErr = "imgPro: rgbImg_threshold ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// ��2������Ϊ���ͼ��
//		double dBcoef = lua_tonumber(L, 3);
//		double dGcoef = lua_tonumber(L, 4);
//		double dRcoef = lua_tonumber(L, 5);
//		int iThresh = (int)lua_tonumber(L, 6);		// ��6������Ϊ��ֵ
//		int iMax = (int)lua_tonumber(L, 7);		// ��7�����������õĴ����Ҷ�ֵ
//		int iType = (int)lua_tonumber(L, 8);		// ��8����������ֵ��������,
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() || iMax <0 || iMax >255 || iType<0)
//		{
//			string strErr = "imgPro: rgbImg_threshold ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ��������������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		cv::Mat imgIn = (*g_pMapImage)[strIn];
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat(imgIn.size(), CV_8UC1);
//		}
//
//		cv::Mat imgOut = (*g_pMapImage)[strOut];
//		int nRows = imgIn.rows;
//		int nCols = imgIn.cols*imgIn.channels();
//		if (1==iType)   //����threshVal,����ΪMax
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
//		string strErr = "imgPro: rgbImg_threshold ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//}
//
///*******************************************************
//iErr = rgbImg_getColor(strImgIn, strImgOut, iB,iG,iR,iDistB,iDistG,iDistR, iMax,iType)
//���ܣ�
//	��ȡ�ض�����ɫ
//���ƣ�
//	��ɫͼ������
//������
//	imgIn - ͼ������
//	imgOut - ͼ�����
//	iB,iG,iR - ����ȡ��ɫ���ص��R,G,Bֵ
//	iDistB,iDistG,iDistR - B,G,R�ֱ��Ӧ�Ķ�̬��Χ
//	iThresh - ��ֵ
//	iMax - ���ֵ
//	iType - ��ֵ��������: 0 : THRESH_BINARY, 1 : THRESH_BINARY_INV,
//����ֵ��
//	iErr - 0, ������ ��0���д���
//
//***************************************************/
//int imgPro::rgbImg_getColor(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
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
//			string strErr = "imgPro: rgbImg_getColor ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// ��2������Ϊ���ͼ��
//		int iB = (int)lua_tonumber(L, 3);
//		int iG = (int)lua_tonumber(L, 4);
//		int iR = (int)lua_tonumber(L, 5);
//		int iDistB = (int)lua_tonumber(L, 6);
//		int iDistG = (int)lua_tonumber(L, 7);
//		int iDistR = (int)lua_tonumber(L, 8);
//		int iMax = (int)lua_tonumber(L, 9);		// ��9�����������õĴ����Ҷ�ֵ
//		int iType = (int)lua_tonumber(L,10);		// ��10����������ֵ��������,
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() || iType<-1)
//		{
//			string strErr = "imgPro: rgbImg_getColor ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�����ֵ�������ʹ���";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		cv::Mat imgIn = (*g_pMapImage)[strIn];
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
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
//		string strErr = "imgPro: rgbImg_getColor ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//}
//
///**************************************************
//iErr = rgbImg_colorIntensity(imgIn, imgOut, iB,iG ,iR,iType )
//���ܣ�
//	��ȡͼ��ָ����ɫ�ĻҶ�ӳ��ͼ�񣬻Ҷ�Խ����ɫԽ�ӽ�
//���ƣ�
//	��
//������
//imgIn - ����ͼ��
//imgOut - ���ͼ��
//iB,iG,iR - ��ȡ��ָ����ɫ
// iType  -��ɫӳ������,0:��ƫ�����ֵ���ֵ���㣻
//
//����ֵ��
//iErr - 0,������ ��0���д���
//***************************************************/
//
//int imgPro::rgbImg_colorIntensity(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//
//		{
//			string strErr = "imgPro: rgbImg_colorIntensity ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		int bians;
//		string strIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
//		string strOut = lua_tostring(L, 2);	// ��2������Ϊ���ͼ��
//		int iB = (int)lua_tonumber(L, 3);
//		int iG = (int)lua_tonumber(L, 4);
//		int iR = (int)lua_tonumber(L, 5);
//		int iType = (int)lua_tonumber(L, 6);		// ��6����������ֵ��������,
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end() 	||iType != 0)
//		{
//			string strErr = "imgPro: rgbImg_colorIntensity ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ��������������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		cv::Mat imgIn = (*g_pMapImage)[strIn];
//
//		// ������ͼ�񻺳岻���ڣ��򴴽�һ������
//		if (g_pMapImage->find(strOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strOut] = cv::Mat(imgIn.size(), CV_8UC1);
//		}
//
//		cv::Mat imgOut = (*g_pMapImage)[strOut];
//		int nRows = imgIn.rows;
//		int nCols = imgIn.cols*imgIn.channels();
//
//		if (0 == iType)   //����ƫ��ֵ��ȡ���ؾ���
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
//		string strErr = "imgPro: rgbImg_colorIntensity ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//
//}
//

// model :0 :�ӵ����ȵ������ȼ������P��������1:�Ӹ����ȵ������ȼ������P

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
//iErr,dSim = grayImg_compareHist(strIn1, strIn2��iMethod)
//���ܣ�
//�Ƚ�����ͼ��ĻҶ�ֱ��ͼ
//���ƣ�
//��
//������
//strIn1 - ��������1
//strIn2 - ��������2
//iMethod - �ȽϷ�ʽ(�Ƽ�Ĭ��Ϊ0��0:CV_COMP_CORREL,1:CV_COMP_CHISQR,2:CV_COMP_INTERSECT ,3:CV_COMP_BHATTACHARYYA)
//����ֵ��
//iErr - 0,������ ��0���д���

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
			string strErr = "imgPro: grayImg_ransacCircle ͼ������������٣�";
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
		string strErr = "imgPro: grayImg_ransacCircle ����C++�쳣��";
		return 1;
	}
}



/**************************************************
iErr = biImg_filterByArea(strImgIn, strImgOut,iAreaThreshVal,iMode)
���ܣ�
ʹ�������Сɸѡͼ��
���ƣ�
��
������
strImgIn - ����ͼ��
strImgOut - ���ͼ��
iAreaThreshVal - �����ֵ
iMode - ����ģʽ,1 - ��ֵ֮����Ϊ0��2 - ��ֵ֮����Ϊ0
����ֵ��
iErr - 0,������ ��0���д���
***************************************************/
int imgPro::biImg_filterByArea( cv::Mat &imgIn, cv::Mat &imgOut, int iAreaThreshLow, int iAreaThreshHigh, int iMode, int connection)
{
	try
	{
		if (imgIn.empty())
		{
			string strErr = "imgPro: grayImg_filterByArea ����ͼ�����";
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
			if (1 == iMode)   //��ֵ֮����Ϊ0
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
		string strErr = "imgPro: grayImg_filterByArea ����C++�쳣��";
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
		    cv::Mat_<uchar>::const_iterator iter = edge[i].begin<uchar>();       //��ǰ����
			cv::Mat_<uchar>::const_iterator nextIter = edge[i].begin<uchar>() + 1; //��һ������
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
				floodFill(imgOut, edgePts[n], 0,0,0,0,8);//��ˮ��䷨

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
			string strErr = "imgPro: biImg_delMaxArea ����ͼ�����";
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
			if (status.at<int>(i, cv::CC_STAT_AREA) > maxArea) //�ҵ����ֵ����
			{
				maxArea = status.at<int>(i, cv::CC_STAT_AREA);
				maxAreaLabel = i;
			}
		}
		if (maxAreaLabel >0)
		{
			colors[maxAreaLabel] = 0;   //���ֵ����ֵ0
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
		string strErr = "imgPro: grayImg_filterByArea ����C++�쳣��";
		return -1;
	}
}
//iMode ==0,��ֵ֮������Ϊ0�� =1����ֵ֮������Ϊ0
int imgPro::biImg_getMaxArea(cv::Mat & imgIn, cv::Mat & imgOut)
{
	try
	{
		if (imgIn.empty() || imgIn.total() < FILTER_AREA_MIN_NUM)
		{
			string strErr = "imgPro: biImg_delMaxArea ����ͼ�����";
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
			if (status.at<int>(i, cv::CC_STAT_AREA) > maxArea) //�ҵ����ֵ����
			{
				maxArea = status.at<int>(i, cv::CC_STAT_AREA);
				maxAreaLabel = i;
			}

		}
		if (maxAreaLabel > 0)
		{
			colors[maxAreaLabel] = 255;   //���ֵ����ֵ255
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
		string strErr = "imgPro: grayImg_filterByArea ����C++�쳣��";
		return -1;
	}

}

/**************************************************
iErr = biImg_thinImg(imgIn,imgOut,iter)
���ܣ�
��ͼ����������ȡ�Ǽ�
���ƣ�
��
������
imgIn - ����ͼ�񣬸ú������ı�ԭʼͼ��
����ֵ��
iErr            - 0, ������ ��0���д���
***************************************************/
int imgPro::biImg_thinImg(cv::Mat imgIn, cv::Mat &imgOut, int maxIterations)
{
    try
    {
		if (imgIn.channels() > 1 || cv::countNonZero(imgIn) < IMG_MIN_NUM)
		{
			string strErr = "imgPro: biImg_thinImg ����ͼ�����";
			return 1;
		}

		cv::Mat dst;
		cv::threshold(imgIn, dst, 0, 1, cv::THRESH_BINARY);

		int width = imgIn.cols;
		int height = imgIn.rows;
		//src.copyTo(dst);
		int count = 0;  //��¼��������  
		while (true)
		{
			count++;
			if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������  
				break;
			std::vector<uchar*> mFlag; //���ڱ����Ҫɾ���ĵ�  
			//�Ե���  
			for (int i = 0; i < height; ++i)
			{
				uchar* p = dst.ptr<uchar>(i);
				for (int j = 0; j < width; ++j)
				{
					//��������ĸ����������б��  
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
							//���  
							mFlag.push_back(p + j);
						}
					}
				}
			}

			//����ǵĵ�ɾ��  
			for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
			{
				**i = 0;
			}

			//ֱ��û�е����㣬�㷨����  
			if (mFlag.empty())
			{
				break;
			}
			else
			{
				mFlag.clear();//��mFlag���  
			}

			//�Ե���  
			for (int i = 0; i < height; ++i)
			{
				uchar* p = dst.ptr<uchar>(i);
				for (int j = 0; j < width; ++j)
				{
					//��������ĸ����������б��  
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
							//���  
							mFlag.push_back(p + j);
						}
					}
				}
			}

			//����ǵĵ�ɾ��  
			for (std::vector<uchar*>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
			{
				**i = 0;
			}

			//ֱ��û�е����㣬�㷨����  
			if (mFlag.empty())
			{
				break;
			}
			else
			{
				mFlag.clear();//��mFlag���  
			}
		}
		imgOut =  dst * 255;

		return 0;
	}
    catch (...)
    {
        string strErr = "imgPro: biImg_thinImg ����C++�쳣��";
        return -1;
    }

}

/**************************************************
iErr = biImg_fillup(imgIn, imgOut, iType)
���ܣ�
��ֵͼ��׶����
���ƣ�
��
������
imgIn - ����ͼ��
imgOut - ���ͼ��
iType - ��䷽����ѡ��.1:��ȫ����⵽����������2�������ֵ֮�������
***************************************************/
int imgPro::biImg_fillup(const cv::Mat &imgIn, cv::Mat &imgOut, int iType,int fillAreaLow,int fillAreaHigh)
{
    try
    {
        if (imgIn.empty() ||imgIn.total()<IMG_MIN_NUM)
        {
            string strErr = "imgPro: region_Dilation ����ͼ�� ";
            strErr += " �����ڣ�";
            return -1;
        }
		if (imgOut.empty())
		{
			imgOut.create(imgIn.size(), CV_8UC1);
		}
        if (imgIn.type() != CV_8UC1)
        {
            string strErr = "imgPro: biImg_fillup ����ͼ���Ƕ�ֵͼ��";
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
        string strErr = "imgPro: biImg_fillup ����C++�쳣��";
        return -1;
    }

}


///**************************************************
//iErr, iNum = region_houghLines(imgIn, dRho, dTheta, iThre, dR, dT)
//���ܣ�
//����ͼƬ�е�ֱ��
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//dRho - �����������ȣ���λ�����أ�
//dTheta - �����ǶȾ��ȣ���λ�����ȣ�
//iThre - ��ֵ
//dR - ֱ�ߺϲ��Ŀ��
//dT - ֱ�ߺϲ��ĽǶ�, dR��dT��Ϊ0ʱ�����ϲ�
//����ֵ��
//iErr - 0,������ ��0���д���
//iNum - ֱ�߸���
//***************************************************/
//int imgPro::biImg_houghLines(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: region_HoughLines ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		string imgIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
//		double dRho = lua_tonumber(L, 2);   // �����������ȣ���λ�����أ�
//		double dTheta = lua_tonumber(L, 3)*PI/180; // �����ǶȾ��ȣ�ת��Ϊ���ȣ�
//		int iThre = (int)lua_tonumber(L, 4);
//		double dR = lua_tonumber(L, 5);        // ֱ�ߺϲ��Ŀ��
//		double dT = lua_tonumber(L, 6)*PI/180; // ֱ�ߺϲ��ĽǶ�, dR��dT��Ϊ0ʱ�����ϲ�
//
//		// �������
//		if (g_pMapImage->find(imgIn) == g_pMapImage->end() || (*g_pMapImage)[imgIn].total()<IMG_MIN_NUM)
//		{
//			string strErr = "imgPro: region_HoughLines ����ͼ�� ";
//			strErr += imgIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: region_HoughLines ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//	}
//	return 2;
//}
//
///**************************************************
//iErr��iNum = region_houghLinesP(imgIn, dRho, dTheta, iThre, minLineLength, maxLineGap)
//���ܣ�
//����ͼƬ�е��߶�
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//dRho - �����������ȣ���λ�����أ�
//dTheta - �����ǶȾ��ȣ���λ�����ȣ�
//iThre - ��ֵ
//minLineLength - �߶���С����
//maxLineGap - ���ֱ�߼�϶
//����ֵ��
//iErr - 0,������ ��0���д���
//iNum - ֱ�ߵ�����
//***************************************************/
//int imgPro::biImg_houghLinesP(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		if (iCount != 6
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER
//			|| lua_type(L, 6) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: region_HoughLinesP ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		string strIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
//		double dRho = lua_tonumber(L, 2);	// ������������
//		double dTheta = lua_tonumber(L, 3)*PI / 180;  // �����ǶȾ���
//		int iThre = (int)lua_tonumber(L, 4); //
//		double minLineLength = lua_tonumber(L, 5);
//		double maxLineGap = lua_tonumber(L, 6);
//
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()  )
//		{
//			string strErr = "imgPro: region_HoughLinesP ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: region_HoughLinesP ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, 0);
//	}
//	return 2;
//}
//
///**************************************************
//iErr = biImg_getEdge(imgIn,edgeImgOut,strEdgeContOut,iDirection,iProStart,iStartPos,iEndPos��iDistOffset,iIsConnected)
//���ܣ�
//	 ��ͼ���ĳһ�����ȡ�÷���ı�Ե����������£��������ϣ��������ң���������
//���ƣ�
//��
//������
//	imgIn - ����ͼ��
//	edgeImgOut - ����ı�Եͼ��
//	strEdgeContOut - ����ı�Ե����
//	iDirection - ��ȡ��Ե�ķ��� 0���������� ��1���������� ��2���������� ��3����������
//    iProStart - ��ʼͶӰ��λ��
//	iStartPos - ��ʼ��ȡ��Եλ��
//	iEndPos   - ������ȡ��Եλ��
//	iDistOffset - �����ҵ��ı�Ե������һ����ƫ�Ʒ�Χ����������ֵ��Ͽ�����
//	iIsConnected - �Ƿ���Щ��Ե���������������˳���iDistOffset�ĵ㣬�����Ͽ��ı�Ե�㽫��ֱ������
//����ֵ��
//	iErr - 0,������ ��0���д���
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
//			string strErr = "imgPro: biImg_getEdge ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//			string strErr = "imgPro: biImg_getEdge ������ͼ�񲻴��ڻ���ȡ�������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//				string strErr = "imgPro: biImg_getEdge ��������ȡ��Χ����";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//				lua_pushinteger(L, -100);
//				return 1;
//			}
//		}
//		else
//		{
//			if (startPos< 0 || endPos >imgIn.rows - 1 || startPos >endPos)
//			{
//				string strErr = "imgPro: biImg_getEdge ��������ȡ��Χ����";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		case 0://��������
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
//		case 1://��������
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
//		case 2:   //��������
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
//		case 3: //��������
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
//			string strErr = "imgPro: biImg_getEdge ��ϵ�����С��10��";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//			if (abs(mapProfile[strEdgeContOut][i].x - mapProfile[strEdgeContOut][i + 1].x) + abs(mapProfile[strEdgeContOut][i].y - mapProfile[strEdgeContOut][i + 1].y) < iDistOffset)  //���������жϵ�ľ���
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
//		string strErr = "imgPro: biImg_getEdge ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//}
//
//
//
///**************************************************
//iErr,centRow,centCol,dAngle,width,height = biImg_getRotRect(imgIn)
//���ܣ�
//	����ֵͼ����������ĵ������ת����
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//
//����ֵ��
//iErr - 0,������ ��0���д���
//others -��ת���εĲ���
//***************************************************/

int imgPro::biImg_getRotRect(cv::Mat &imgIn,cv::RotatedRect & rotRect)
{
	try
	{

			//����Ϊ�㼯
		int cnt = countNonZero(imgIn);
		vector<cv::Point> pts(cnt);  //��ֵͼ���е����е�
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

			//����Ϊ�㼯
		int cnt = countNonZero(imgIn);
		vector<cv::Point> pts(cnt);  //��ֵͼ���е����е�
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
//���ܣ�
//����Բ���
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//iMinDis - ��С����
//iMinR - ��С�뾶
//iMinR - ���뾶
//����ֵ��
//iErr - 0,������ ��0���д���
//iNum - ����Բ�ĸ���
//***************************************************/

int imgPro::biImg_houghCircles(cv::Mat &imgIn,vector<cv::Vec3f> &circles,int iMinDis,int iMinR,int iMaxR)
{
	try
	{
		// �������
		if (imgIn.empty() || imgIn.channels() != 1)
		{
			string strErr = "imgPro: region_houghCircles ����ͼ�� ";
			strErr += " �����ڣ�";
			return -1;
		}

		circles.clear();
		//cv::Mat g;
		//GaussianBlur( mIn, g, Size(5, 5), 2, 2 );
		cv::HoughCircles(imgIn, circles, cv::HOUGH_GRADIENT, 1, iMinDis, 100, 100, iMinR, iMaxR);

	}
	catch (...)
	{
		string strErr = "imgPro: region_houghCircles ����C++�쳣��";
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
���ܣ�
�ָ����ֵͼ������򲢼����������
���ƣ�
��
������
strImgIn               - �����ͼ��
features          - ����ı��ͼ�����Ӧ������������Ϊmap<string,vector<double>>,string ����area,,,
iAreaThrehold    - ���ɸѡ��ֵ�����ڸ���ֵ����ʾ
����ֵ��
iErr  - 0,������ ��0���д���
iCnt  - �ָ����ֵ����ĸ���

***************************************************/
int imgPro::biImg_createRegion(const cv::Mat &imgIn, cv::Mat &imgLabel, map<string, vector<double>> &features, int iAreaThreshLow, int iAreaThreshHigh)
{
	try
	{
		if (imgIn.empty() || imgIn.total() < FILTER_AREA_MIN_NUM)
		{
			string strErr = "imgPro: biImg_createRegion ����ͼ�� ";
			strErr += " �����ڻ�Ϊ�հ�";
			return -1;
		}
		cv::Mat  status, centroids;
		int validCnt = 0;
		features.clear();
		int labelCnt = cv::connectedComponentsWithStats(imgIn, imgLabel, status, centroids);
		for (int cnt = 1; cnt < labelCnt; cnt++)//�����Ǳ���
		{
			if (iAreaThreshLow < status.at<int>(cnt, cv::CC_STAT_AREA) && status.at<int>(cnt, cv::CC_STAT_AREA) < iAreaThreshHigh)
			{
				validCnt++;
			}
		}
		//�ռ任ʱ��
		map<int, vector<cv::Point>> mapRegionPts;
		for (int r=0;r<imgIn.rows;r++)
		{
			for (int c = 0; c < imgIn.cols; c++)
			{
				mapRegionPts[imgLabel.at<int>(r, c)].push_back(cv::Point(c,r));
			}
		}
		//��status��centorid�е�ֵ�ŵ�map�з������i=0Ϊ����
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
				features["left"][i] = (status.at<int>(i, cv::CC_STAT_LEFT));  //����������������������vector������0��ʼ��
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

				//����ƽ������ĵ�����
				vector<cv::Point> polyPts;
				cv::approxPolyDP(cont, polyPts, 8, true);
				cv::Mat show = tempROI.clone();
				//show.convertTo(show, CV_8UC3);
				//cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
				//cv::polylines(show, polyPts, true, cv::Scalar(0, 0, 255), 2);
				features["polyPts"][i] = (polyPts.size());
				//����Բ���ƶ�����
				//double likeCircle = 4 * CV_PI*status.at<int>(i, cv::CC_STAT_AREA) / (cont.size()*cont.size());
				double likeCircle = 4 * CV_PI*cv::contourArea(cont) / (pow(cv::arcLength(cont, true), 2));

				features["circle"][i] = (likeCircle);    // 
															//������С��ת���εĳ����
				float HWRatio = 0.;
				cv::RotatedRect roRect = minAreaRect(cont);//regionTemp
				double likeRect = status.at<int>(i, cv::CC_STAT_AREA) / roRect.size.area();
				features["rect"][i] = (likeRect);    //

				features["rotW"][i] = (roRect.size.width);
				features["rotH"][i] = (roRect.size.height);
				features["rotA"][i] = (roRect.angle);

				if (roRect.size.height > roRect.size.width)
				{
					features["rotWidth"][i] = (roRect.size.height);  //������С��ת���εĳ���
					features["rotHeight"][i] = (roRect.size.width);  //������С��ת���εı�
					HWRatio = roRect.size.height / roRect.size.width;
				}
				else
				{
					features["rotWidth"][i] = (roRect.size.width);
					features["rotHeight"][i] = (roRect.size.height);
					HWRatio = roRect.size.width / roRect.size.height;
				}

				features["rotHWRatio"][i] = (HWRatio);    // 9															 
				features["rotRecArea"][i] = (roRect.size.area()); //�������ת���ε����
				vector<cv::Point2f> pts;
				float angle;
				rect_getRotRectPts(roRect, pts, angle);
				features["rotAngle"][i] = (angle);    //   ���νǶ�		
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
		string strErr = "imgPro: biImg_createRegion ����C++�쳣��";
		return -1;
	}
}



/**************************************************
iErr = region_toImg(strLabelImg,strImgOut,iIndex)
���ܣ�
��������� ��������������������
���ƣ�
��
������
strLabelImg        - ����ı��ͼ��INT32��
strImgOut           - �����ͼ��CV_8UC1���ͣ��������Ϊ255����������Ϊ0
iIndex                  - �������������1��ʼ
����ֵ��
iErr  - 0,������ ��0���д���

***************************************************/
int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, int index)
{
	try
	{

		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImg ����ͼ�� ";
			strErr += " �����ڻ�";
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
		//	string strErr = "imgPro: region_toImg ������region2vector���д���";
		//	return -1;
		//}

		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro: region_toImg ����C++�쳣��";
		return -1;
	}
}

int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, const vector<cv::Point> &vecPt)
{
	try
	{
		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImg ����ͼ�� ";
			strErr += " �����ڻ�";
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
		string strErr = "imgPro: region_toImg ����C++�쳣��";
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
		string strErr = "imgPro: region_toImg ����C++�쳣��";
		return -1;
	}

}
/**************************************************
iErr = region_toImgAll(strLabelImg,strImgOut,num)
���ܣ�
��������� ��������������������
���ƣ�
��
������
strLabelImg        - ����ı��ͼ��INT32��
strImgOut           - �����ͼ��CV_8UC1���ͣ��������Ϊ255����������Ϊ0
num                  - label������
����ֵ��
iErr  - 0,������ ��0���д���

***************************************************/
int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut,const vector<int> &vecIndex)
{
	try
	{
		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImgAll ����ͼ�� ";
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
		string strErr = "imgPro: region_toImgAll ����C++�쳣��";
		return -1;
	}
}
int imgPro::region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut,const vector<vector<cv::Point>> &vvPt)
{
	try
	{
		if (labelImg.empty())
		{
			string strErr = "imgPro: region_toImgAll ����ͼ�� ";
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
		string strErr = "imgPro: region_toImgAll ����C++�쳣��";
		return -1;
	}
}
//
///**************************************************
//iErr = region_sortByFeature(strLabelImg,strFeature,iCnt)
//���ܣ�
//    ��������� ��������������������
//���ƣ�
//    ��
//������
//strLabelImg        - ����ı��ͼ��INT32��
//strFeature           - ����������������������֧�������AREA��,�߶�"H",���"W",Բ��"CIRCLE",�߿��"HW",�����ת���������minRectArea��,
//                                ����������λ������
//                                            -LR  ������������
//                                            -RL  ������������
//                                            -UD ������������
//                                            -DU ������������
//                               ����������
//iCnt			--����ǰiCnt������
//����ֵ��
//iErr  - 0,������ ��0���д���
//Value  - ������Ӧ��ֵ
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
//			string strErr = "imgPro: region_sortByFeature ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//		if (iCount == 3 && (lua_type(L, 1) != LUA_TSTRING || lua_type(L, 2) != LUA_TSTRING || lua_type(L, 3) != LUA_TNUMBER))
//		{
//			string strErr = "imgPro: region_sortByFeature ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//            string strErr = "imgPro: region_sortByFeature ����ͼ�� ";
//            strErr += strImgLabel;
//            strErr += " �����ڻ�����";
//            strErr+= "������";
//			if (iCount == 3 && cnt > g_regionFeature[strImgLabel].size())
//			{
//				strErr += "��������������";
//			}
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//			string strErr = "imgPro: region_sortByFeature �������������ڣ�";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: region_sortByFeature ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}

///**************************************************
//iErr,startRow,startCol,endRow,endCol = region_fitLine(strLabelImg,iIndex,iDistType,iDiff)
//���ܣ�
//    ���������������ֱ����ϣ�����Ⱥ�������,���֮ǰ������б��ɸѡ�����ϵĵ�
//���ƣ�
//��
//������
//strLabelImgIn        - ����ı��ͼ��INT32��
//iIndex                     - ��������1��ʼ
//iDistType               - ֱ����Ϸ���, ����1,2
//                                CV_DIST_L1      =1,   < distance = |x1-x2| + |y1-y2| 
//                                CV_DIST_L2 = 2,   < the simple euclidean distance
//                                CV_DIST_C = 3,   < distance = max(|x1-x2|,|y1-y2|)
//                                CV_DIST_L12 = 4,   < L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
//                                CV_DIST_FAIR = 5,   < distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998 
//                                CV_DIST_WELSCH = 6, < distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
//                                CV_DIST_HUBER = 7    < distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345 
//isVertical:  ��ȡֱ�ߵ����ͣ�0 :���ֱ�ߣ� 1:���ֱ���е㴦�Ĵ���
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

			//�ҳ�ֱ����ͼ���ڵĶ˵�
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
				string strErr = "imgPro: region_fitLine �����е�һ��Ѱ�һ��Ƶ����";
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
					string strErr = "imgPro: region_fitLine �����е�һ��Ѱ�һ��Ƶ����";
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
//            string strErr = "imgPro: region_fitLine ��������";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: region_fitLine ����ͼ�� ";
//        strErr += strLabelImg;
//        strErr += " �����ڻ�����С��1 !";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        //�ҳ�ֱ����ͼ���ڵĶ˵�
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
//			string strErr = "imgPro: region_fitLine �����е�һ��Ѱ�һ��Ƶ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        //��Զ��ֱ�ߵĵ����ɾ��
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
//        //�ҳ�ֱ����ͼ���ڵĶ˵�
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
//			string strErr = "imgPro: region_fitLine �����еڶ���Ѱ�һ��Ƶ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: region_fitLine ������C_FilterProfileByCurv���д���";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//    string strErr = "imgPro: region_fitLine ����C++�쳣��";
//    ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//    ��ȡ�����ж�Ӧ��������С��Ӿ���,     ע��KitB��Angle��imgPro�����෴�ġ�
//��
//������
//strLabelImg        - ����ı��ͼ��INT32��
//iIndex                  - ��Ӧ������
//
//����ֵ��
//iErr  - 0,������ ��0���д���
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
//            string strErr = "imgPro: region_smallestRectR ��������";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//            string strErr = "imgPro: region_fitLine ����ͼ�� ";
//            strErr += strLabelImg;
//            strErr += " �����ڻ����������� !";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//            string strErr = "imgPro: region_smallestRectR ������C_Region2vector���д���";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: region_smallestRectR �����в���C++�쳣��";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ȡ��������С�����ת����
//���ƣ�
//��
//������
//iIndex - �������,�����1��ʼ
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow - �������ĵ�������
//dCol - �������ĵ�������
//dAngle - ������ת�Ƕ�
//dL1 - ���γ��ߵ�һ��
//dL2 - ���ζ̱ߵ�һ��
//***************************************************/
//int imgPro::cont_smallestRectR(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_smallestRectR ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 6;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);// �ڶ�������Ϊ �������,��1��ʼ
//		iIndex -= 1;
//		// �������
//		if (iIndex < 0 || iIndex >= int(g_contours.size()))
//		{
//			string strErr = "imgPro: cont_smallestRectR �����������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: cont_smallestRectR ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//����ָ����������С��Ӿ���(����ת)
//���ƣ�
//��
//������
//iIndex - �������
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow - �������Ͻǵ�������
//dCol - �������Ͻǵ�������
//dWidth - ���ο��
//dHeight - ���θ߶�
//***************************************************/
//int imgPro::cont_smallestRect(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_smallestRect ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 5;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);// �ڶ�������Ϊ �������
//		iIndex -= 1;
//		// �������
//		if (iIndex < 0 || iIndex >= int(g_contours.size()))
//		{
//			string strErr = "imgPro: cont_smallestRect ������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: cont_smallestRect ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//������������������ĵ�����
//���ƣ�
//��
//������
//iIndex - �������
//����ֵ��
//iErr - 0,������ ��0���д���
//iArea - �������
//dRow - �������ĵ�������
//dCol - �������ĵ�������
//***************************************************/
//int imgPro::cont_areaCenter(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//		int iIndex = (int)lua_tonumber(L, 1);// �ڶ�������Ϊ �������
//		iIndex -= 1;
//		// �������
//		if (iIndex < 0 || iIndex >= int(g_contours.size()) || iCount != 1)
//		{
//			string strErr = "imgPro: cont_areaCenter ������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//			string strErr = "imgPro: cont_areaCenter ��������Ӧ��������Ϊ0��";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: cont_areaCenter ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//���ݾ�����������
//���ƣ�
//��
//������
//dDis - ��С����
//����ֵ��
//iErr - 0,������ ��0���д���
//iNum - ��������
//***************************************************/
//int imgPro::cont_unionContByDist(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 1 
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_unionContByDist ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			return 2;
//		}
//
//		double dDis = lua_tonumber(L, 1);// ������
//		double dDistance = dDis > 0 ? dDis*dDis : -dDis*dDis;	// �������ƽ��ֵ������ȽϾ��벻��Ҫ����
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
//						double d = abs(itorPt->x - itorPt2->x) + abs(itorPt->y - itorPt2->y);	//��ֵ������ٶȺܿ�
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
//		string strErr = "imgPro: cont_unionContByDist ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_pushinteger(L, (int)g_contours.size());
//		return 2;
//	}
//}
//
///**************************************************
//iErr,dRow1,dCol1,dRow2,dCol2 = cont_fitLine(imgIn,iIndex,iType)
//���ܣ�
//�����������ֱ�ߣ������߶Σ�
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//iIndex - ������ţ���1��ʼ
//iType - �������(1��7��1��CV_DIST_L1��2��CV_DIST_L2��3��CV_DIST_C��4��CV_DIST_L12)
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow1 - ֱ����ʼ��������
//dCol1 - ֱ����ʼ��������
//dRow2 - ֱ��ĩ�˵�������
//dCol2 - ֱ��ĩ�˵�������
//***************************************************/
//int imgPro::cont_fitLine(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 3
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: cont_fitLine ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			lua_pushnumber(L, 0);
//			return 5;
//		}
//
//		string strIn = lua_tostring(L, 1);// �ڶ�������Ϊ �������
//		int iIndex = (int)lua_tonumber(L, 1);// �ڶ�������Ϊ �������
//		int iType = (int)lua_tonumber(L, 2);//
//		iIndex -= 1;
//		// �������
//		if (g_pMapImage->find(strIn) == g_pMapImage->end()   
//				|| iIndex >= g_contours.size() ||iIndex<0)
//		{
//			string strErr = "imgPro: cont_fitLine ����ͼ�� ";
//			strErr += strIn;
//			strErr += " �����ڣ���������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		//��ȡ��бʽ�ĵ��б��  
//		cv::Point point0;
//		point0.x = v[2];
//		point0.y = v[3];
//
//		double k = v[1] / v[0];
//
//		//����ֱ�ߵĶ˵�(y = k(x - x0) + y0)  
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
//		string strErr = "imgPro: cont_fitLine ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//�������������Բ
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//iIndex - ������ţ���1��ʼ
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow    - ��Բ���ĵ�������
//dCol      - ��Բ���ĵ�������
//dWidth  - ��Բ��
//dHeight - ��Բ��
//dAngle   -��Բ�Ƕ�
//***************************************************/
//
//int imgPro::cont_fitEllipse(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);      // ��������
//
//        // �������
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: cont_fitEllipse ��������";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 6;
//        }
//
//        string strIn = lua_tostring(L, 1);// �ڶ�������Ϊ �������
//        int iIndex = (int)lua_tonumber(L, 2);// �ڶ�������Ϊ �������
//        iIndex -= 1;
//        // �������
//        if (g_pMapImage->find(strIn) == g_pMapImage->end()
//            || iIndex >= g_contours.size() || iIndex < 0)
//        {
//            string strErr = "imgPro: cont_fitEllipse ����ͼ�� ";
//            strErr += strIn;
//            strErr += " �����ڣ���������Χ����";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: cont_fitEllipse ����C++�쳣��";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//�����������Բ,��С���˷�
//���ƣ�
//��
//������
//imgIn - ����ͼ��
//iIndex - ������ţ���1��ʼ
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow    - Բ���ĵ�������
//dCol      - Բ���ĵ�������
//dR         - Բ�뾶
//***************************************************/
//
//int imgPro::cont_fitCircle(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);      // ��������
//
//        // �������
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: cont_fitCircle ��������";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        string strIn = lua_tostring(L, 1);// �ڶ�������Ϊ �������
//        int iIndex = (int)lua_tonumber(L, 2);// �ڶ�������Ϊ �������
//        iIndex -= 1;
//        // �������
//        if (g_pMapImage->find(strIn) == g_pMapImage->end()
//            || iIndex >= g_contours.size() || iIndex < 0)
//        {
//            string strErr = "imgPro: cont_fitCircle ����ͼ�� ";
//            strErr += strIn;
//            strErr += " �����ڣ���������Χ����";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//            string strErr = "imgPro: cont_fitCircle ִ�лع麯���쳣��";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: cont_fitCircle ����C++�쳣��";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ȡ�����߶εļн�
//���ƣ�
//��
//������
//����ֵ��
//dAngle - �߶μн�,�ýǶȱ�ʾ
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
		
		angle =  Get2VecAngle(dA, dB)/PI*180.0; //����ԽǶȱ�ʾ
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro::angle2Lines ����C++�쳣��";
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
		
		angle =  Get2VecAngle(dA, dB)/PI*180.0; //����ԽǶȱ�ʾ
		return 0;
	}
	catch (...)
	{
		string strErr = "imgPro::angle2Lines ����C++�쳣��";
		return 1 ;
	}
}
//
///**************************************************
//dDist = dist2Pts(dP1Row,dP1Col,dP2Row,dP2Col)
//���ܣ�
//��ȡ������֮��ľ���
//���ƣ�
//��
//������

//����ֵ��
//iErr - 0,������ ��0���д���
//dDist - ����֮�����
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
		string strErr = "imgPro: dist2Pts ����C++�쳣��";
		return -1;
	}
}

int imgPro::angle2Pts(cv::Point ptStart, cv::Point ptEnd,double &angle)
{
	try
	{
		cv::Point ptEndHor = ptStart;
		ptEndHor.x = ptEnd.x + 10;

		angle2Lines(ptStart, ptEnd, ptStart, ptEndHor, angle);//���ؿռ������н�
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

		angle2Lines(ptStart, ptEnd, ptStart, ptEndHor, angle);//���ؿռ������н�
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
//���ܣ�
//�������и��m*n��С����(row-->m, col-->n)
//���ƣ�
//��
//������
//	iCenterRow,iCenterCol,iWidth,iHeight,dAngle����ת���β���
//	iM	�����ε��зָ�ΪM��
//	iN	�����ε��зָ�ΪN��
//����ֵ��
//iErr - 0,������ ��0���д���
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
//			string strErr = "imgPro: math_splitRect ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: math_splitRect ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}
///**************************************************
//iErr,dRow,dCol = intersecBy2Lines(dRow1,dCol1,dRow2,dCol2,dRow3,dCol3,dRow4,dCol4)
//���ܣ�
//���������߶εĽ���
//���ƣ�
//��
//������

//����ֵ��
//iErr - 0,������ ��0���д���
//dRow - �����������
//dCol - �����������
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
		string strErr = "imgPro: math_intersectionBy2Lines ����C++�쳣��";
		return 1;
	}
}
//
///**************************************************
//ret = closestPt2L(dPtRow,dPtCol,dLRow1,dLCol1,dLRow2,dLCol2)
//���ܣ�
//����ֱ��������������ĵ㣬ͬʱ���ص���ֱ�ߵľ���
//���ƣ�
//��
//������

//����ֵ��

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
		string strErr = "imgPro: img_copy ����C++�쳣��";
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
//���ܣ�
//����ָ��������ָ�������С������������㣬��������С��������
//���ƣ�
//��
//������
//dPtRow - ���������
//dPtCol - ���������
//iIndex - �������
//����ֵ��
//iErr - 0,������ ��0���д���
//dMinDist - �����������С����
//dMaxDist - ���������������
//dMinRow - ������������С���������
//dMinCol - ������������С���������
//dMaxRow - ���������������������
//dMaxCol - ���������������������
//***************************************************/
//int imgPro::math_distPtCont(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 3 
//			|| lua_type(L, 1) != LUA_TNUMBER
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: math_distPtCont ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		// �������
//		if (iIndex >= g_contours.size()|| iIndex<0)
//		{
//			string strErr = "imgPro: math_distPtCont ������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: math_distPtCont ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//����ָ��������ָ���ߵ���С������������㣬�������С��������
//���ƣ�
//��
//������
//dLRow1 - �߶���ʼ���������
//dLCol1  - �߶���ʼ���������
//dLRow2 - �߶�ĩ�˵��������
//dLCol2  - �߶�ĩ�˵��������
//iIndex    - �������
//����ֵ��
//iErr - 0,������ ��0���д���
//dMinDist - �������߶ε���С����
//dMaxDist - �������߶ε�������
//dMinRow - �������߶ξ�����С���������
//dMinCol - �������߶ξ�����С���������
//dMaxRow - �������߶ξ��������������
//dMaxCol - �������߶ξ��������������
//***************************************************/
//int imgPro::math_distLineCont(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 5 
//			|| lua_type(L, 1) != LUA_TNUMBER
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: math_distLineCont ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		// �������
//		if (iIndex >= g_contours.size() || iIndex<0)
//		{
//			string strErr = "imgPro: math_distLineCont ������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: math_distLineCont ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ȡ��ת���ε��ĸ�����ͽǶȣ��ĸ������˳�����������ϣ����ϣ����£����£�ע��Ƕ���halcon����һ��[]

//���ƣ�

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
		string err = "img_drawRect:�����쳣, " + string(e.what());
		return cv::RotatedRect();
	}
}
//�����ĸ�����
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
		string err = "img_drawRect:�����쳣, " + string(e.what());
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
		string strErr = "imgPro: rect_getBoxPts ����C++�쳣��";
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
		string strErr = "imgPro: rect_getBoxPts ����C++�쳣��";
		return -1;
	}

}
//len:�������ĳߴ�
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


//�����ɫ
cv::Scalar imgPro::randomColor(cv::RNG& rng)
{
	int icolor = (unsigned)rng;
	return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

//theta :��λΪ�Ƕ�
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
		string strErr = "imgPro: pt_rotate ����C++�쳣��";
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
//���ܣ�
//��ȡԲ�Ĳ���
//���ƣ�
// ��
//������
//iIndex - Բ���
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow - Բ���ĵ�������
//dCol - Բ���ĵ�������
//dR - Բ�İ뾶
//***************************************************/
//int imgPro::circle_getCircleInfo(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 1 
//			|| lua_type (L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: circle_getCircleInfo ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 4;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);//
//		iIndex -= 1;
//		// �������
//		if (iIndex >= g_circles.size()||iIndex<0)
//		{
//			string strErr = "imgPro: circle_getCircleInfo ������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: circle_getCircleInfo ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//���ܣ�
//��ȡ��Ĳ���
//���ƣ�
//��
//������
//iIndex - ����ţ���1��ʼ
//����ֵ��
//iErr - 0,������ ��0���д���
//dRow - ��������
//dCol - ��������
//***************************************************/
//int imgPro::point_getPointInfo(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		// �������
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TNUMBER)
//		{
//			string strErr = "imgPro: point_getPointInfo ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_pushinteger(L, 0);
//			lua_pushinteger(L, 0);
//			return 3;
//		}
//
//		int iIndex = (int)lua_tonumber(L, 1);//
//		iIndex -= 1;
//		// �������
//		if (iIndex >= g_points.size()|| iIndex<0)
//		{
//			string strErr = "imgPro: point_getPointInfo ������Χ����";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: point_getPointInfo ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//����
//strImgIn		   -  �����ֵͼ��
//strImgOut	   -  ���������
//iOrientation   -  ɨ�跽��ͼ�����ܣ�,0-�������£�1-�������ϣ�2 - �������ң�3 - ��������
//startPos       -  ��ʼ�ҵ��λ��
//endPos         -  �����ҵ��λ��
//**********************************************************/
int imgPro::laser_fromImage(cv::Mat &imgIn,vector<cv::Point> &profileOut,int oritation,int startPos,int endPos,bool withZero,bool clearInvalidPt)
{
    try
    {
        if (imgIn.empty() || oritation <0 || oritation>3)
        {
            string strErr = "imgPro: laser_fromImage ������ͼ�񲻴��ڻ���ȡ�������";
            return -1;
        }
		profileOut.clear();

		if (oritation == 1 || oritation ==0)
		{
			if (startPos< 0 || endPos >imgIn.cols || startPos >endPos)
			{
				string strErr = "imgPro: laser_fromImage ��������ȡ��Χ����";
				return -1;
			}
		}
		else
		{
			if (startPos< 0 || endPos >imgIn.rows || startPos >endPos)
			{
				string strErr = "imgPro: laser_fromImage ��������ȡ��Χ����";
				return -1;
			}
		}
	
        const int IMG_WIDTH = imgIn.cols;
        const int IMG_HEIGHT = imgIn.rows;
		if (oritation < 2)//����ɨ��
		{
			profileOut.assign(IMG_WIDTH, INVALID_POINT);
		}
		else
		{
			profileOut.assign(IMG_HEIGHT, INVALID_POINT);
		}


        switch (oritation)
        {
        case 0://��������
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
        case 1://��������
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
        case 2:   //��������
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
        case 3: //��������
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
        string strErr = "imgPro: laser_fromImage ����C++�쳣��";
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
//���ܣ�
//����ָ�������ߵ�ָ�����ͼ��
//���ƣ�
//�ű�����
//������
//strImgIn    - ����ͼ��
//strImgOut - ���ͼ��
//strProfileIn	  - ��������
//iThickness - ���ƿ��
//����ֵ��
//nRet  - 0,������ ��0���д���
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
        string strErr = "imgPro: img_drawLaser ����C++�쳣��";
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
		string strErr = "imgPro: img_drawLaser ����C++�쳣��";
		return -2;
	}
}
//
//
///*
//nRet,tablePoints = laser_getPolyline(strLaserIn,dEpsilon,iIsClosed)
//���룺
//strLaserIn -  ����������������Ǹ�����
//dEpsilon   - ָ���ƽ����ȣ� ��ԭʼ������ƽ��߶ε�������
//iIsClosed  - �Ƿ������������β�����ıպ�������һ�����ڴ�����תɨ������
//�����
//tablePoints - �߶������
//*/
int imgPro::laser_getPolyline(const vector<cv::Point2d> &profileIn, vector<cv::Point2d> &polyPts,double epsilon,bool isClosed)
{
    try
    {

        if (profileIn.empty() || epsilon < 0)
        {
            string strErr = "imgPro: laser_getLocalExtrepoint �������������ڻ򾫶Ȳ�������";
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
        string strErr = "imgPro: laser_getPolyline ����C++�쳣��";
        return -2;

    }
}

//����
int imgPro::laser_getPolyline(const vector<cv::Point> &profileIn, vector<cv::Point> &polyPts, double epsilon, bool isClosed)
{
	try
	{

		if (profileIn.empty() || epsilon < 0)
		{
			string strErr = "imgPro: laser_getLocalExtrepoint �������������ڻ򾫶Ȳ�������";
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
		string strErr = "imgPro: laser_getPolyline ����C++�쳣��";
		return -2;

	}
}

//// mode 0:�������һ�����㷨  1:��С���˷�
//int imgPro::laser_fitCircle(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);      // ��������
//        // �������
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER)
//        {
//            string strErr = "imgPro: laser_fitCircle ��������";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            lua_pushnumber(L, 0);
//            return 4;
//        }
//
//        string strIn = lua_tostring(L, 1);// �ڶ�������Ϊ ����
//        int mode = (int)lua_tonumber(L, 2);// �ڶ�������Ϊ ģʽ
//        // �������
//        if (mapProfile.find(strIn) == mapProfile.end() || mode <0 ||  2<mode)
//        {
//            string strErr = "imgPro: laser_fitCircle �������� ";
//            strErr += strIn;
//            strErr += " �����ڣ���ģʽ��Χ����";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//	            string strErr = "imgPro: cont_fitCircle ִ�лع麯���쳣��";
//	            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: cont_fitCircle ����C++�쳣��";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//            string strErr = "imgPro: laser_concatenate ��������";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//            string strErr = "imgPro: laser_concatenate ��������1������!";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//            lua_pushinteger(L, -100);
//            lua_pushinteger(L, -100);
//            return 2;
//        }
//        if (mapProfile.find(strLaserIn2) == mapProfile.end())
//        {
//            string strErr = "imgPro: laser_concatenate ��������2������!";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//        string strErr = "imgPro: laser_concatenate ����C++�쳣��";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//        lua_pushinteger(L, -100);
//        lua_pushinteger(L, -100);
//        return 2;
//    }
//}
//
//
///**************************************************
//iErr,Points = laser_getPoints(laserContIn)
//���ܣ�
//��ȡ��ֵ��ͼ����ָ�����������ĵ�
//���ƣ�
//��
//������
//laserContIn - ��������
//����ֵ��
//iErr - 0,������ ��0���д���
//Points - ��������ָ�������ĵ��Table��˫��Table,ÿ��point�»���һ��Table��Ŷ�Ӧ��row,col
//***************************************************/
//int imgPro::laser_getPoints(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//		//�������
//		if (iCount != 1
//			|| lua_type(L, 1) != LUA_TSTRING)
//		{
//			string strErr = "imgPro: laser_getPoints ��������";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//		string strLaserIn = lua_tostring(L, 1);	// ��һ������Ϊ����ͼ��
//
//		if (mapProfile.find(strLaserIn) == mapProfile.end())
//		{
//			string strErr = "imgPro: laser_concatenate ��������������!";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
//		string strErr = "imgPro: laser_getPoints ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
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
////��ȡϵͳ��ǰʱ��
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
//		string strErr = "imgPro: sys_getTime ����C++�쳣��";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // ���ʹ������
//		lua_pushinteger(L, -100);
//		lua_pushstring(L,"");
//		return 2;
//
//	}
//}
//
///**************************************************
//int test(lua_State* L)
//���ܣ�
//�����ú���
//���ƣ�
//��
//������
//������ע��
//����ֵ��
//��
//***************************************************/
//int test(lua_State* L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // ��������
//
//		string strIn = lua_tostring(L, 1);	// ��1������Ϊ����ͼ��
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