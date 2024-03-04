///////////////////////////////////////////////////////////////////////////////////////////////
////
///////////////////////////////////////////////////////////////////////////////////////////////
#include "laserpro.h"
#include <stack>

#define  invalid_point point2d(-100000000.,-100000000.)

#define  val_unvalid  -1
#define  VAL_VALID  0 


int getLocalMin(vector<int> &vecVal,vector<int> &vecLoc,vector<int> &vecValOut,vector<int> &vecLocOut)
{
	vecValOut.clear();
	vecLocOut.clear();
	for (int i=0;i<vecVal.size();i++)
	{
		if (i==0)
		{
			if (vecVal[i] < vecVal[i+1])
			{
				vecValOut.push_back(vecVal[i]);
				vecLocOut.push_back(vecLoc[i]);
			}
		}
		else if (i == vecVal.size()-1)
		{
			if (vecVal[i] < vecVal[i-1])
			{
				vecValOut.push_back(vecVal[i]);
				vecLocOut.push_back(vecLoc[i]);
			}

		}
		else
		{
			if (2*vecVal[i] < vecVal[i-1]+vecVal[i+1])
			{
				vecValOut.push_back(vecVal[i]);
				vecLocOut.push_back(vecLoc[i]);
			}
		}
	}
	return 0;
}
int getLocalMax(vector<int> &vecVal,vector<int> &vecLoc,vector<int> &vecValOut,vector<int> &vecLocOut)
{
	vecValOut.clear();
	vecLocOut.clear();
	for (int i=0;i<vecVal.size();i++)
	{
		if (i==0)
		{
			if (vecVal[i] > vecVal[i+1])
			{
				vecValOut.push_back(vecVal[i]);
				vecLocOut.push_back(vecLoc[i]);
			}
		}
		else if (i == vecVal.size()-1)
		{
			if (vecVal[i] > vecVal[i-1])
			{
				vecValOut.push_back(vecVal[i]);
				vecLocOut.push_back(vecLoc[i]);
			}

		}
		else
		{
			if (2*vecVal[i] > vecVal[i-1]+vecVal[i+1])
			{
				vecValOut.push_back(vecVal[i]);
				vecLocOut.push_back(vecLoc[i]);
			}
		}
	}
	return 0;
}


// 获取4邻域点是否存在，存在则返回
vector<cv::Point> get4NeiPt(cv::Mat biImg, cv::Point pt)
{
	cv::Point Lpt, Tpt, Bpt, Rpt;
	
	Lpt = cv::Point(pt.x - 1, pt.y);
	Tpt = cv::Point(pt.x, pt.y + 1);
	Bpt = cv::Point(pt.x, pt.y - 1);
	Rpt = cv::Point(pt.x + 1, pt.y);
	
	vector<cv::Point> nPtVec{Lpt, Tpt ,Bpt, Rpt};
	vector<cv::Point> resVec;
	for(auto pt : nPtVec  )
	{
		if (0 <= pt.x && pt.x < biImg.cols &&
			0 <= pt.y && pt.y < biImg.rows &&
			biImg.at<uchar>(pt) > 0
			)
		{
			resVec.push_back(pt);
		}
	}

	return resVec;
}

// 获取8领域点
vector<cv::Point> get8NeiPt(cv::Mat biImg, cv::Point pt)
{
	vector<cv::Point> resVec;;

	for (int x = pt.x - 1; x < pt.x + 2; x++)
	{
		for (int y = pt.y - 1; y < pt.y + 2; y++)
		{
			if (0 <= pt.x && pt.x < biImg.cols &&
				0 <= pt.y && pt.y < biImg.rows &&
				(pt.x != x && pt.y != y) &&
				biImg.at<uchar>(y,x)>0
				)
			{
				resVec.push_back(cv::Point(x,y));
			}
		}
	}

	return resVec;
}


LaserPro::LaserPro(void)
{
}

LaserPro::~LaserPro(void)
{
}

int LaserPro::laser_smooth(const vector<int>& vecPtsIn, vector<int>& vecPtsOut, vector<double> weithts,int times)
{
	try
	{
		int len = vecPtsIn.size() - 1;
		vecPtsOut.assign(256,0);
		if (weithts.size() >1 ) //todo
		{
			double midWeight = weithts[(weithts.size() - 1) / 2];
		}

		vector<int> vecPtsInTmp(vecPtsIn.begin(), vecPtsIn.end());

		for (int t=0;t<times;t++)
		{
			vecPtsOut[0] =( vecPtsInTmp[0] + vecPtsInTmp[0] + vecPtsInTmp[1]) / 3;
			vecPtsOut[len-1] =( vecPtsInTmp[len-1] + vecPtsInTmp[len-1] + vecPtsInTmp[len-2]) / 3;
			for (int i = 1; i < vecPtsInTmp.size()-1; i++)
			{
				vecPtsOut[i] = vecPtsInTmp[i - 1] + vecPtsInTmp[i] + vecPtsInTmp[i + 1];
			}
			vecPtsInTmp.assign(vecPtsOut.begin(), vecPtsOut.end());
		}
		return 0;
	}
	catch(...)
	{
		string err = "laser_smooth发生异常！";
		return -1;
	}

}

int LaserPro::laser_drawAsHist(const vector<int>& vecPtsIn, cv::Mat & histImg,vector<int> &vecNormPtsOut, bool down)
{
	try
	{
		int H = 256;
		vector<double> vecHistTmp(vecPtsIn.begin(), vecPtsIn.end());
		cv::Mat canvs = cv::Mat::zeros(256, 256, CV_8UC1);
		canvs.setTo(255);
		//int maxVal;
		double maxVal = *max_element(vecHistTmp.begin(), vecHistTmp.end());
		vector<int> normHist(256, 0);
		for_each(vecHistTmp.begin(), vecHistTmp.end(), [maxVal](double &val) {val = val / maxVal * 256.; });
		vecNormPtsOut.assign(vecHistTmp.begin(),vecHistTmp.end());
		//绘制到画布上
		for (int c = 0; c < canvs.cols; c++)
		{
			int len = vecHistTmp[c];
			for (int r = canvs.rows - 1; r > H - len; r--)
			{
				canvs.at<uchar>(r, c) = 0;
			}
		}
		histImg = canvs;
		return 0;
	}
	catch (...)
	{
		string msg = "laser_drawAsHist:发生异常";
		return -1;
	}

}
int LaserPro::laser_drawAsHist(const vector<int>& vecPtsIn, cv::Mat & histImg, bool down)
{
	try
	{
		int H = 256;
		vector<double> vecHistTmp(vecPtsIn.begin(), vecPtsIn.end());
		cv::Mat canvs = cv::Mat::zeros(256, 256, CV_8UC1);
		canvs.setTo(255);
		//int maxVal;
		double maxVal = *max_element(vecHistTmp.begin(), vecHistTmp.end());
		vector<int> normHist(256, 0);
		for_each(vecHistTmp.begin(), vecHistTmp.end(), [maxVal](double &val) {val = val / maxVal * 256.; });
		//绘制到画布上
		for (int c = 0; c < canvs.cols; c++)
		{
			int len = vecHistTmp[c];
			for (int r = canvs.rows - 1; r > H - len; r--)
			{
				canvs.at<uchar>(r, c) = 0;
			}
		}
		histImg = canvs;
		return 0;
	}
	catch (...)
	{
		string msg = "laser_drawAsHist:发生异常";
		return -1;
	}

}

int LaserPro::laser_getLocalPts(const vector<int> &vecVal, vector<int> &locs, int heightLowThresh)
{
	try
	{
		cv::Mat vecPtsImg;
		laser_drawAsHist(vecVal, vecPtsImg);

		//取局部极值，注意高原问题
		vector<int> maximaLoc;
		vector<int> maximaVal;
		vector<int> minimaLoc;
		vector<int> minimaVal;
		vector<int> extremaLoc;
		vector<int> extremaVal;
		vector<int> extremaLab; //1 maxima, 0 minma

		vector<int> vecPts;
		vecPts.assign(vecVal.begin(),vecVal.end());
		//1st,获取局部极值
		for (int i=0;i<vecPts.size();i++)
		{
			if (i==0)
			{
				if (vecPts[i] < vecPts[i+1])
				{
					minimaLoc.push_back(i);
					minimaVal.push_back(vecPts[i]);
				}
				else if (vecPts[i] > vecPts[i + 1])
				{
					maximaLoc.push_back(i);
					maximaVal.push_back(vecPts[i]);
				}
			}
			else if (i == vecPts.size() - 1)
			{
				if (vecPts[i] < vecPts[i - 1])
				{
					minimaLoc.push_back(i);
					minimaVal.push_back(vecPts[i]);
				}
				else if (vecPts[i] > vecPts[i -1])
				{
					maximaLoc.push_back(i);
					maximaVal.push_back(vecPts[i]);
				}

			}
			else
			{
				if (vecPts[i-1] +vecPts[i+1] > 2*vecPts[i] && (vecPts[i - 1] >= vecPts[i] && vecPts[i] <= vecPts[i + 1]))
				{
					minimaLoc.push_back(i);
					minimaVal.push_back(vecPts[i]);
				}
				else if ( vecPts[i - 1] + vecPts[i + 1] < 2 * vecPts[i] && (vecPts[i - 1] <= vecPts[i] && vecPts[i] >= vecPts[i + 1]))
				{
					maximaLoc.push_back(i);
					maximaVal.push_back(vecPts[i]);
				}
			}
		}
		cv::Mat showTmp = vecPtsImg.clone();
		cv::cvtColor(showTmp, showTmp, cv::COLOR_GRAY2BGR);
		for (int i = 0; i < minimaLoc.size(); i++)
		{
			cv::line(showTmp, cv::Point(minimaLoc[i], 255), cv::Point(minimaLoc[i], 250), cv::Scalar(0, 0, 255), 1);
		}
		for (int i = 0; i < maximaLoc.size(); i++)
		{
			cv::line(showTmp, cv::Point(maximaLoc[i], 255), cv::Point(maximaLoc[i], 250), cv::Scalar(0, 255, 0), 1);
		}

		int ite = 3;
		for (int iteCnt=0;iteCnt<ite;iteCnt++)
		{
			//迭代最小值
			vector<int> minimaLocTmp;
			vector<int> minimaValTmp;
			vector<int> maximaLocTmp;
			vector<int> maximaValTmp;
			getLocalMax(maximaVal, maximaLoc, maximaValTmp, maximaLocTmp);
			getLocalMin(minimaVal, minimaLoc, minimaValTmp, minimaLocTmp);

			maximaVal.assign(maximaValTmp.begin(), maximaValTmp.end());
			maximaLoc.assign(maximaLocTmp.begin(), maximaLocTmp.end());
			minimaVal.assign(minimaValTmp.begin(), minimaValTmp.end());
			minimaLoc.assign(minimaLocTmp.begin(), minimaLocTmp.end());

			maximaValTmp.clear();
			maximaLocTmp.clear();
			minimaValTmp.clear();
			minimaLocTmp.clear();

			//2nd,删除重复的元素
			for (int i = 0; i < maximaLoc.size(); i++)
			{
				if (std::count(minimaLoc.begin(), minimaLoc.end(), maximaLoc[i]) == 0)
				{
					maximaLocTmp.push_back(maximaLoc[i]);
					maximaValTmp.push_back(maximaVal[i]);
				}
			}
			maximaLoc.assign(maximaLocTmp.begin(), maximaLocTmp.end());
			maximaVal.assign(maximaValTmp.begin(), maximaValTmp.end());
			for (int i = 0; i < minimaLoc.size(); i++)
			{
				if (std::count(maximaLoc.begin(), maximaLoc.end(), minimaLoc[i]) == 0)
				{
					minimaLocTmp.push_back(minimaLoc[i]);
					minimaValTmp.push_back(minimaVal[i]);
				}
			}
			minimaLoc.assign(minimaLocTmp.begin(), minimaLocTmp.end());
			minimaVal.assign(minimaValTmp.begin(), minimaValTmp.end());

			//show
			//cv::Mat showTmp = vecPtsImg.clone();
			cv::cvtColor(showTmp, showTmp, cv::COLOR_GRAY2BGR);
			for (int i = 0; i < minimaLoc.size(); i++)
			{
				cv::line(showTmp, cv::Point(minimaLoc[i], 255), cv::Point(minimaLoc[i], 250), cv::Scalar(0, 0, 255), 1);
			}
			for (int i = 0; i < maximaLoc.size(); i++)
			{
				cv::line(showTmp, cv::Point(maximaLoc[i], 255), cv::Point(maximaLoc[i], 250), cv::Scalar(0, 255, 0), 1);
			}


		}
		//开始迭代
		





		return 0;
	}
	catch (const std::exception& e)
	{
		return -1;
	}

}
int LaserPro::laser_getGeneralMinPts(const vector<int> &vecPts, vector<int> &locs, int grayDiff)
{
	try
	{
		cv::Mat vecPtsImg;
		laser_drawAsHist(vecPts, vecPtsImg);
		vector<int> localMin;

		int valMax = vecPts[0];
		int valMin = vecPts[0];
		int locMax = 0;
		int locMin = 0;

		//局部极小值



		return 0;
	}
	catch (const std::exception& e)
	{
		return -1;
	}

}
/*****
功能：多项式拟合点集

n: 多项式最高次数
A：结果矩阵，Y = A[0,0]+A[1,0]*X+A[2,0]*X^2+...

*****/
bool LaserPro::polynomialCurveFit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();

	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}

	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}

int LaserPro::img_drawPoint(cv::Mat& imgInPutOutput, vector<cv::Point> ptVec, cv::Scalar color,
		cv::MarkerTypes mt, int markerSize, int thickness)
{
	try
	{
		for (auto pt : ptVec)
		{
			cv::drawMarker(imgInPutOutput, pt, color, mt, markerSize, thickness);
		}

		return 0;
	}
	catch (const std::exception&)
	{
		return -1;
	}
}

int LaserPro::skeleton_getAllPt(cv::Mat biImg, vector<cv::Point>& cont, int inver)
{
	try
	{
		vector<cv::Point> endPtVec;
		skeleton_getEndPt(biImg, endPtVec);
		if (endPtVec.size() <= inver)
		{
			return 1;
		}

		cv::Mat trackTmp = biImg.clone();
		// 遍历追踪
		//vector<cv::Point> cont
		stack<cv::Point> tmpPtStack;
		tmpPtStack.push(endPtVec[inver]);
		while (! tmpPtStack.empty())
		{
			cv::Point tmpPt = tmpPtStack.top();
			tmpPtStack.pop();
			trackTmp.at<uchar>(tmpPt) = 0;
			cont.push_back(tmpPt);

			vector<cv::Point> pt4N = get4NeiPt(trackTmp, tmpPt);
			if (!pt4N.empty())
			{
				tmpPtStack.push(pt4N.front());
				continue;
			}
			else
			{
				vector<cv::Point> pt8N = get8NeiPt(trackTmp, tmpPt);
				if (pt8N.empty())
				{
					break;
				}
				else
				{
					tmpPtStack.push(pt8N.front());
				}
			}
		}
		return 0;

	}
	catch (const std::exception&)
	{

		return -1;
	}
}

int LaserPro::skeleton_getEndPt(cv::Mat biImg, vector<vector<cv::Point>>& ptVVec)
{
	try
	{
		vector<cv::Point> vecPts;
		cv::Mat thinTmp;
		cv::Mat labelImg;
		// thinTmp = thinImage(skeletonImg);
		thinTmp = biImg;
		skeleton_getEndPt(thinTmp, vecPts);
		map<string, vector<double>> feats;
		//biImg_createRegion(skeletonImg, labelImg, feats, 0, skeletonImg.total());
		cv::connectedComponents(biImg, labelImg);

		map<int, vector<cv::Point>> mapEndPts;
		for (auto pt : vecPts)
		{
			mapEndPts[labelImg.at<int>(pt)].push_back(pt);
		}
		for (auto ite : mapEndPts)
		{
			ptVVec.push_back(ite.second);
		}

		return 0;
	}
	catch (const std::exception&)
	{
		return -1;
	}
}

int LaserPro::skeleton_getEndPt(const cv::Mat& skeletonImg, vector<cv::Point>& endPoints)
{
	try
	{
		cv::Mat zerOneImg, padImg;
		endPoints.clear();
		cv::copyMakeBorder(skeletonImg, padImg, 1, 1, 1, 1, CV_8UC1, cv::Scalar(0));

		cv::threshold(padImg, zerOneImg, 0, 1, cv::THRESH_BINARY);
	#pragma region kernerl
		//http ://www.imagemagick.org/Usage/morphology/#linejunctions

		cv::Mat kernel1 = (cv::Mat_<int>(3, 3) << -1, -1, 0,
			-1, 1, 1,
			-1, -1, 0);

		cv::Mat kernel2 = (cv::Mat_<int>(3, 3) << -1, -1, -1,
			-1, 1, -1,
			0, 1, 0);

		cv::Mat kernel3 = (cv::Mat_<int>(3, 3) << 0, -1, -1,
			1, 1, -1,
			0, -1, -1);

		cv::Mat kernel4 = (cv::Mat_<int>(3, 3) << 0, 1, 0,
			-1, 1, -1,
			-1, -1, -1);

		cv::Mat kernel5 = (cv::Mat_<int>(3, 3) << -1, -1, -1,
			-1, 1, -1,
			-1, -1, 1);

		cv::Mat kernel6 = (cv::Mat_<int>(3, 3) << -1, -1, -1,
			-1, 1, -1,
			1, -1, -1);

		cv::Mat kernel7 = (cv::Mat_<int>(3, 3) << 1, -1, -1,
			-1, 1, -1,
			-1, -1, -1);

		cv::Mat kernel8 = (cv::Mat_<int>(3, 3) << -1, -1, 1,
			-1, 1, -1,
			-1, -1, -1);

	#pragma endregion
		cv::Mat endPointsImg, hitmisImg;
		//zerOneImg.convertTo(zerOneImg, CV_32S);
		endPointsImg.create(padImg.size(), CV_8UC1);
		endPointsImg.setTo(0);
		vector<cv::Mat> kerners{ kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7,kernel8 };
		for (cv::Mat ker : kerners)
		{
			cv::morphologyEx(zerOneImg, hitmisImg, cv::MorphTypes::MORPH_HITMISS, ker);

			cv::bitwise_or(hitmisImg, endPointsImg, endPointsImg);
		}

		endPointsImg = endPointsImg * 255;
		for (int r = 0; r < endPointsImg.rows; r++)
		{
			for (int c = 0; c < endPointsImg.cols; c++)
			{
				if (endPointsImg.at<uchar>(r, c) > 0)
				{
					endPoints.push_back(cv::Point(c - 1, r - 1));
				}
			}
		}

		return 0;
	}
	catch (const std::exception&)
	{
		return -1;
	}
}

int LaserPro::skeleton_split(cv::Mat biImg, vector<cv::Mat>& imgVec)
{

	return 0;
}

//double similarityLaser(vector<Point2d> & vecPts, vector<Point2d> &vecTemplate, int errTolerance)
//{
//	vector<Point2d> input(vecPts);
//	vector<Point2d> templatePts(vecTemplate);
//	double ptsCnt = 0.;
//	int  greatSize, smallSize;
//
//	if (input.size() > templatePts.size())
//	{
//		greatSize = input.size();
//		smallSize = templatePts.size();
//	}
//	else
//	{
//		smallSize = input.size();
//		greatSize = templatePts.size();
//	}
//	for (int i = 0; i <smallSize; i++)
//	{
//		double diff = abs(input[i].y - templatePts[i].y);
//		if (diff <= errTolerance)
//		{
//			ptsCnt++;
//		}
//	}
//	return ptsCnt / greatSize;
//}
//double similarityKeyPoints(vector<Point2d> & vecPts, vector<Point2d> &vecTemplate ,int  baseHeight)
//{
//	double objHeight = (vecPts[0].y + vecPts[1].y) / 2. - baseHeight;
//	double objBigDiameter = abs(vecPts[0].x - vecPts[1].x);
//
//	double tempHeight = (vecTemplate[0].y + vecTemplate[1].y) / 2 - baseHeight;
//	double tempBigDiameter = abs(vecTemplate[0].x - vecTemplate[1].x);
//
//	double diff = (abs(objHeight - tempHeight) / tempHeight )+ abs(objBigDiameter - tempBigDiameter) / tempBigDiameter;
//
//	return (1.0-(diff/2.));
//}
//vector<Point2d> getKeyPointsOfProfile(vector<Point2d> & vecPts)
//{
//	vector<Point2d>::iterator realPoint = std::remove_if(vecPts.begin(), vecPts.end(), [](Point2d m){return m.x<0; });
//	pair<vector<Point2d>::iterator ,
//		     vector<Point2d>::iterator>	   //选出x点最值，为窝口最外沿点
//			 ptPair = std::minmax_element(vecPts.begin(), realPoint, [](Point2d m, Point2d n){return m.x<n.x ; });
//
//	return vector<Point2d>{(*ptPair.first), (*ptPair.second)};// {ptPair.first, ptPair.second}是迭代器初始化
//}
//
//int clearLaserNoise(Mat &inImg, Mat &outImg, int roiRowBegin, int morEleWidth, int morEleHeight, int threshValue)
//{
//	outImg.create(inImg.size(), CV_8UC1);
//	outImg.setTo(Scalar(0));
//	Mat src(inImg.rowRange(roiRowBegin, inImg.rows));
//	Mat sobelImg;
//	Sobel(src, sobelImg, CV_16SC1, 0, 1, 3);
//	double minVal, maxVal;
//	Mat yimg, dst1, dst2;
//	cv::minMaxLoc(sobelImg, &minVal, &maxVal);
//	sobelImg.convertTo(yimg, CV_8UC1, 255 / maxVal);
//
//	Mat kernel1 = (Mat_<char>(3, 3) << 2, -1, -1,
//		-1, 2, -1,
//		-1, -1, 2);
//	Mat kernel2 = (Mat_<char>(3, 3) << -1, -1, 2,
//		-1, 2, -1,
//		2, -1, -1);
//	filter2D(src, dst1, CV_16U, kernel1);
//	filter2D(src, dst2, CV_16U, kernel2);
//
//	double max = 0, max2 = 0;
//	minMaxLoc(dst1, 0, &max);
//	minMaxLoc(dst2, 0, &max2);
//
//	Mat dst1_8u, dst3_8u;
//	dst1.convertTo(dst1_8u, CV_8U, 255. / max);
//	dst2.convertTo(dst3_8u, CV_8U, 255. / max);
//
//	Mat res = dst1_8u + dst3_8u + yimg;
//	Mat morImg;
//	Mat thre;
//	Mat mor = getStructuringElement(MORPH_RECT, Size(morEleWidth, morEleHeight));
//	threshold(res, thre, threshValue, 255, THRESH_BINARY);
//
//	morphologyEx(thre, morImg, MORPH_DILATE, mor, Point(), 2);
//
//	vector < vector < Point >> contours;
//	vector<Vec4i> v4i;
//	findContours(morImg, contours, RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, roiRowBegin));//
//	std::nth_element(contours.begin(), contours.begin() + 3, contours.end(),
//		[](vector<Point> &m, vector<Point> &n){return m.size() > n.size(); });
//	for (int i = 0; i < 4; i++)
//	{
//		drawContours(outImg, contours, i, Scalar(255), -1);
//	}
//	return 0;
//}
//
//
////从激光轮廓中获取有效的连续子轮廓
////轮廓X坐标连续，高度Y有跳跃，Hoffset为高度跳跃阈值
//void getContinuousProfile(vector<cv::Point2d> profile,vector<vector<Point2d>> & conProfile, int Hoffset)
//{
//	vector<Point2d> object;
//	for (int i = 0; i < profile.size()-1; i++)
//	{
//		if (profile[i].x>0 )
//		{
//			object.push_back(profile[i]);
//            if (abs(profile[i].y - profile[i + 1].y) > Hoffset )
//            {
//                conProfile.push_back(object);
//                object.clear();
//            }
//		}
//		else
//		{
//			if (object.size() > 0)
//			{
//				conProfile.push_back(object);
//			}
//			object.clear();
//		}
//		if (i == profile.size() - 1 && profile.back().x > 0)  //处理有效点在图片右边界的情况
//		{
//			conProfile.push_back(object);
//		}
//	}
//}
////重载
//void getContinuousProfile(vector<cv::Point2d> profile, vector<vector<Point>> & conProfile,int Hoffset)
//{
//    vector<Point> object;
//    for (int i = 0; i < profile.size(); i++)
//    {
//        if (profile[i].x>0)
//        {
//            object.push_back(profile[i]);
//            if (abs(profile[i].y - profile[i + 1].y) > Hoffset)
//            {
//                conProfile.push_back(object);
//                object.clear();
//            }
//        }
//        else
//        {
//            if (object.size() > 0)
//            {
//                conProfile.push_back(object);
//            }
//            object.clear();
//        }
//        if (i == profile.size() - 1 && profile.back().x > 0)  //有效点在图片右边界
//        {
//            conProfile.push_back(object);
//        }
//    }
//}
//
////在连续子轮廓中获取它们的端点
//vector<pair<Point2d, Point2d>> getEndPoints(vector<vector<Point2d>> objects)
//{
//    vector<pair<Point2d, Point2d>> endPoints;
//    pair<Point2d, Point2d> twoPoints;
//    for (int i = 0; i < objects.size(); i++)
//    {
//        twoPoints = make_pair(objects[i].front(), objects[i].back());
//        endPoints.push_back(twoPoints);
//    }
//    return endPoints;
//}
//
//
//vector<vector<Point2d>> getHeightestPointsOfProfile(vector<Point2d> profile, int nei, int Hthresh,int isClosed)
//{
//	vector<Point2d> res;
//	vector<vector<Point2d>> reses;
//	int vecProfileSize = profile.size();
//	if (0 == isClosed)
//	{
//			for (int i = nei; i < vecProfileSize - nei - 1; i++)
//			{
//				int pMinus = i - nei;
//				int pPlus = i + nei;
//
//				if ((profile[i].y - profile[pMinus].y <Hthresh) && (profile[i].y - profile[pMinus].y >0) &&
//					(profile[i].y - profile[pPlus].y <Hthresh) && (profile[i].y - profile[pPlus].y >0) )//&&abs(profile[pPlus].y - profile[pMinus].y)<5)用于限制轮廓对称性
//				{
//					res.push_back(profile[i]);
//				}
//				else
//				{
//					if (res.size()>0)
//					{
//						reses.push_back(res);
//						res.clear();
//					}
//				}
//			}
//	} 
//	else
//	{
//		for (int i = 0; i < vecProfileSize;i++)
//		{
//			int iPlus = i + nei;
//			int iMinus = i - nei;
//			Point2d	pMinus = profile[iMinus < 0 ? iMinus + profile.size() : iMinus];
//			Point2d	pPlus = profile[iPlus >= profile.size() ? iPlus - profile.size() : iPlus];
//
//			if ((profile[i].y - pMinus.y <Hthresh) && (profile[i].y - pMinus.y >0) &&
//				(profile[i].y - pPlus.y <Hthresh) && (profile[i].y - pPlus.y >0))//&&abs(profile[pPlus].y - profile[pMinus].y)<5)用于限制轮廓对称性
//			{
//				res.push_back(profile[i]);
//			}
//			else
//			{
//				if (res.size()>0)
//				{
//					reses.push_back(res);
//					res.clear();
//				}
//			}
//
//
//		}
//	}
//	return reses;
//}
//vector<vector<Point2d>> getLowestPointsOfProfile(vector<Point2d> profile, int nei, int Hthresh,int isClosed)
//{
//	vector<Point2d> res;
//	vector<vector<Point2d>> reses;
//	int vecProfileSize = profile.size();
//	if (0 == isClosed)
//	{	
//		for (int i = nei; i < vecProfileSize - nei - 1; i++)
//			{
//				int pMinus = i - nei;
//				int pPlus = i + nei;
//
//				if ((profile[pMinus].y - profile[i].y <Hthresh) && ( profile[pMinus].y  -profile[i].y>0) &&
//					( profile[pPlus].y - profile[i].y<Hthresh) && ( profile[pPlus].y- profile[i].y >0))// &&abs(profile[pPlus].y - profile[pMinus].y)<5)
//				{
//					res.push_back(profile[i]);
//				}
//				else
//				{
//					if (res.size()>0)
//					{
//						reses.push_back(res);
//						res.clear();
//					}
//				}
//			}
//			}
//	else
//	{
//		for (int i = 0; i < vecProfileSize; i++)
//		{
//			int iPlus = i + nei;
//			int iMinus = i - nei;
//			Point2d	pMinus = profile[iMinus < 0 ? iMinus + profile.size() : iMinus];
//			Point2d	pPlus = profile[iPlus >= profile.size() ? iPlus - profile.size() : iPlus];
//
//			if (( pMinus.y- profile[i].y < Hthresh) && ( pMinus.y - profile[i].y>0) &&
//				  (pPlus.y -  profile[i].y  < Hthresh) && ( pPlus.y-profile[i].y  >0))//&&abs(profile[pPlus].y - profile[pMinus].y)<5)用于限制轮廓对称性
//			{
//				res.push_back(profile[i]);
//			}
//			else
//			{
//				if (res.size()>0)
//				{
//					reses.push_back(res);
//					res.clear();
//				}
//			}
//		}
//	}
//	return reses;
//}
//
//vector<vector<Point2d>> getExtrePointOfProfile(vector<Point2d> profile, int nei, int hThresh,int isClosed)
//{
//	vector<Point2d> res;
//	vector<vector<Point2d>> reses;
//	int vecProfileSize = profile.size();
//	if (0 == isClosed)
//	{
//			for (int i = nei; i < vecProfileSize - nei - 1; i++)
//			{
//				int pMinus = i - nei;
//				int pPlus = i + nei;
//
//				double yMDist = profile[i].y - profile[pMinus].y;
//				double yPDist = profile[i].y - profile[pPlus].y;
//				if (        yMDist*yPDist >0	 						//两边同号为极值点
//					&&  abs(yMDist )< hThresh 	&&	abs(yPDist) <hThresh ) 		//&&abs(profile[pPlus].y - profile[pMinus].y)<5),是否要求两边对称
//				{
//					res.push_back(profile[i]);						  //满足端点条件的点，可能不止一个
//				}
//				else
//				{
//					if (res.size()>0)
//					{
//						reses.push_back(res);
//						res.clear();
//					}
//				}
//			}
//	}
//	else
//	{
//		for (int i = 0; i < vecProfileSize; i++)
//		{
//			int iPlus = i + nei;
//			int iMinus = i - nei;
//			Point2d	pMinus = profile[iMinus < 0 ? iMinus + profile.size() : iMinus];
//			Point2d	pPlus = profile[iPlus >= profile.size() ? iPlus - profile.size() : iPlus];
//
//			double yMDist = profile[i].y - pMinus.y;
//			double yPDist = profile[i].y - pPlus.y;
//			if (yMDist*yPDist >0	 						//两边同号为极值点
//				&& abs(yMDist) < hThresh 	&&	abs(yPDist) <hThresh) 		//&&abs(profile[pPlus].y - profile[pMinus].y)<5),是否要求两边对称
//			{
//				res.push_back(profile[i]);						  //满足端点条件的点，可能不止一个
//			}
//			else
//			{
//				if (res.size()>0)
//				{
//					reses.push_back(res);
//					res.clear();
//				}
//			}
//		}
//	}
//	return reses;
//}
//
////////////////////////////////////////////////////////////////
//;
//
//LaserPro::LaserPro(void)
//{
//}
//
//
//LaserPro::~LaserPro(void)
//{
//	
//}
//
//
///**************************************************
//nRet, dRow, dCol, iWidth = KitL.img_findPt(strImgIn, iCol, iMode)
//功能：
//  计算指定列的激光位置
//限制：
//  脚本调用
//参数：
//  strImgIn - 图像输入
//  iCol - 指定列
//  iMode - 模式选择
//返回值：
//  nRet - 0,正常； 非0，有错误；
//  dRow - 激光的行位置
//  dCol - 激光的列位置
//  iWidth - 激光宽度，0：表示没有找到激光
//***************************************************/
//int KitL::img_findPt(lua_State * L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入图像
//		int iCol = int(lua_tonumber(L, 2));	// 
//		int iMode = int(lua_tonumber(L, 3));
//
//
//		// 参数检查
//		if (iCount != 3
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "KitL: img_findPt 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, -1);
//			lua_pushnumber(L, -1);
//			lua_pushnumber(L, -1);
//			return 4;
//		}
//
//		if (g_pMapImage->find(strIn) == g_pMapImage->end())
//		{
//			string strErr = "KitL: img_findPt 输入图像 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, -1);
//			lua_pushnumber(L, -1);
//			lua_pushnumber(L, -1);
//			return 4;
//		}
//
//		cv::Mat & mIn = (*g_pMapImage)[strIn];
//		//----- 功能实现 -----
//		cv::Point2d Pt;
//		int iWidth;
//		if (m_Profile.findPt(mIn, iCol, Pt, iWidth, iMode) == 0)
//		{
//			lua_pushinteger(L, 0);
//			lua_pushnumber(L, Pt.y);	// y 对应 row
//			lua_pushnumber(L, Pt.x);	// x 对应 col
//			lua_pushnumber(L, iWidth);
//			return 4;
//		}
//		else
//		{
//			string strErr = "findPt 错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, -1);
//			lua_pushnumber(L, -1);
//			lua_pushnumber(L, -1);
//			return 4;
//		}
//
//	}
//	catch (...)
//	{
//		string strErr = "KitL: img_findPt 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, -1);
//		lua_pushnumber(L, -1);
//		lua_pushnumber(L, -1);
//		return 4;
//	}
//}
//
//
///**************************************************
//nRet, x, y = KitL.laser_getPt(strLaser, iIndex)
//功能：
//  获取轮廓指定序号点的数据
//限制：
//  脚本调用
//参数：
//  iIndex - 点序号,从1开始
//返回值：
//  nRet - 0,正常； 非0，有错误；
//  x - 点x的值，如果是未标定轮廓，则为 col
//  y - 点y的值，如果是未标定轮廓，则为 row
//***************************************************/
//int KitL::laser_getPt(lua_State * L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);      // 参数个数
//		string strIn = lua_tostring(L, 1);	// 第一个参数为输入轮廓
//		int iIndex = int(lua_tonumber(L, 2));	// 第二个参数为第几个
//        iIndex -= 1;
//
//		// 参数检查
//		if (iCount != 2
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER)
//		{
//			string strErr = "KitL: laser_getPt 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, PT_NA);
//			lua_pushnumber(L, PT_NA);
//			return 3;
//		}
//		std::map<string, vector<Point2d>>::iterator itor = g_pMapProfile->find(strIn);
//		if (itor == g_pMapProfile->end())
//		{
//			string strErr = "KitL: laser_getPt 输入轮廓 ";
//			strErr += strIn;
//			strErr += " 不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, PT_NA);
//			lua_pushnumber(L, PT_NA);
//			return 3;
//		}
//
//		if (iIndex < 0 || iIndex >= itor->second.size())
//		{
//			string strErr = "KitL: laser_getPt 输入个数超范围 ";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, PT_NA);
//			lua_pushnumber(L, PT_NA);
//			return 3;
//		}
//
//		
//		lua_pushinteger(L, 0);
//		lua_pushnumber(L, (itor->second)[iIndex].x);
//		lua_pushnumber(L, (itor->second)[iIndex].y);
//		return 3;
//		
//
//	}
//	catch (...)
//	{
//		string strErr = "KitL: laser_getPt 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, PT_NA);
//		lua_pushnumber(L, PT_NA);
//		return 3;
//	}
//}
//
///**********************************************************
//nRet = laser_fromImage(strImgIn,strLaserOut,iOrientation)
//
//输入
//strImgIn		   -  输入二值图像
//strImgOut	   -  输出的轮廓
//iOrientation   -  提取方向,0-由上往下，1-由下往上，2 - 由左往右，3 - 由右往左
//**********************************************************/
//int KitL::laser_fromImage(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//
//		if (iCount != 3
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "KitL: laser_fromImage 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		string strImgIn = lua_tostring(L, 1);
//		string strProfileOut = lua_tostring(L, 2);
//		int iOrient = int(lua_tonumber(L, 3));
//
//		if (g_pMapImage->find(strImgIn) == g_pMapImage->end() || iOrient<0 || iOrient>3)
//		{
//				string strErr = "KitL: laser_fromImage 中输入图像不存在或提取方向错误！";
//				::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//				lua_pushinteger(L, -100);
//				return 1;
//		}
//		if (g_pMapProfile->find(strProfileOut) == g_pMapProfile->end())
//		{
//			(*g_pMapProfile)[strProfileOut] = vector<Point2d>();
//		}
//		
//		const cv::Mat& imgIn = (*g_pMapImage)[strImgIn];
//		vector<Point2d> & profileOut = (*g_pMapProfile)[strProfileOut];
//		const int IMG_WIDTH = imgIn.cols;
//		const int IMG_HEIGHT = imgIn.rows;
//		profileOut.assign(IMG_WIDTH, INVALID_POINT);
//
//		switch (iOrient)
//		{
//		case 0://从上往下
//			for (int x = 0; x < IMG_WIDTH; x++)
//			{
//				for (int y = 0; y < IMG_HEIGHT; y++)
//				{
//					if (imgIn.at<uchar>(y,x)>0)
//					{
//						profileOut[x] = Point2d(x, y);
//						break;
//					}
//				}
//			}
//			break;
//		case 1://从下往上
//			for (int x = 0; x < IMG_WIDTH; x++)
//			{
//				for (int y = IMG_HEIGHT - 1; y >= 0; y--)
//				{
//					if (imgIn.at<uchar>(y, x)>0)
//					{
//						profileOut[x] = Point2d(x, y);
//						break;
//					}
//				}
//			}
//			break;
//		case 2:   //从左往右
//			profileOut.resize(IMG_HEIGHT);
//			for (int r = 0; r <IMG_HEIGHT; r++)
//			{
//				for (int c = 0; c <IMG_WIDTH; c++)
//				{
//					if (imgIn.at<uchar>(r, c) > 0)
//					{
//						profileOut[r] = Point2d(c, r);
//						break;
//					}
//				}
//			}
//			break;
//		case 3: //从右往左
//			profileOut.resize(IMG_HEIGHT);
//			for (int r = 0; r < IMG_HEIGHT; r++)
//			{
//				for (int c = IMG_WIDTH - 1; c >= 0; c--)
//				{
//					if (imgIn.at<uchar>(r, c) > 0)
//					{
//						profileOut[r] = Point2d(c, r);
//						break;
//					}
//				}
//			}
//			break;
//		default:
//			break;
//		}
//			lua_pushinteger(L, 0);
//			return 1;
//		}
//	catch (...)
//	{
//		string strErr = "KitL: laser_fromImage 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//}
//
///*
//nRet,dAverHeight = laser_getAverHeight(strProfileIn,iColStart,iColEnd)
//输入:
//	strProfileIn  - 输入轮廓
//
//输出:
//	nRet - 函数运行结果 0-正常，1- 异常
//	dAverHeight - 平均高度，即轮廓点的平均row高
//
//*/
//int KitL::laser_getAverHeight(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//
//		if (iCount != 3
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TNUMBER
//            || lua_type(L, 3) != LUA_TNUMBER)
//		{
//			string strErr = "KitL: laset_getAverHeight 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, -100);
//			return 2;
//		}
//
//		string strLaserIn = lua_tostring(L, 1);
//        int iColStart = (int)lua_tointeger(L, 2);
//        int iColEnd = (int)lua_tointeger(L, 3);
//		if (g_pMapProfile->find(strLaserIn) == g_pMapProfile->end())
//		{
//			string strErr = "KitL: laset_getAverHeight 输入轮廓不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_pushnumber(L, -100);
//			return 2;
//		}
//
//		const vector<Point2d> &vecProfile = (*g_pMapProfile)[strLaserIn];
//        if (iColEnd - iColStart < 4 || iColStart> iColEnd || iColEnd>vecProfile.size()-1)
//        {
//            string strErr = "KitL: laset_getAverHeight 输入的截取范围错误";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, -100);
//            return 2;
//        }
//        if (vecProfile.size()>4)
//        {
//            //去除无效点
//            vector<cv::Point2d> vecTemp(vecProfile.begin() + iColStart, vecProfile.begin() + iColEnd+1);
//            vecTemp.erase(std::remove_if(vecTemp.begin(), vecTemp.end(), [](cv::Point2d pt1){return pt1 == INVALID_POINT; }),vecTemp.end());
//            //排序
//            std::sort(vecTemp.begin(), vecTemp.end(), [](cv::Point2d pt1, cv::Point2d pt2){return pt1.y > pt2.y; });
//            int size = vecTemp.size();
//		    double sumH = 0.;
//		    int pointCnt=0;
//		    double averH=0.;
//		    for (int i = size/4; i < size-size/4; i++)
//		    {
//                sumH += vecTemp[i].y;
//				    pointCnt++;
//		    }
//		    averH = sumH / pointCnt;
//
//		    lua_pushinteger(L, 0);
//		    lua_pushnumber(L, averH);
//		    return 2;
//        } 
//        else
//        {
//            string strErr = "KitL: laset_getAverHeight 轮廓点数量小于4！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_pushnumber(L, -100);
//            return 2;
//        }
//	}
//	catch (...)
//	{
//		string strErr = "KitL: laset_getAverHeight 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_pushnumber(L, -100);
//		return 2;
//	}
//}
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
//int KitL::laser_getPolyline(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//         if (iCount !=3
//             ||lua_type(L,1) != LUA_TSTRING
//             ||lua_type(L,2)!=LUA_TNUMBER
//             ||lua_type(L,3)!=LUA_TNUMBER)
//         {
//             string strErr = "KitL: laser_getPolyline 参数错误！";
//             ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//             lua_pushinteger(L, -100);
//             lua_createtable(L, 0, 0);
//             return 2;
//
//         }
//
//         string strProfile = lua_tostring(L, 1);
//         double dEpsilon = lua_tonumber(L, 2);
//         int iIsClosed = (int)lua_tointeger(L, 3);
//         bool isClosed = false;
//
//         iIsClosed > 0 ? isClosed = true : isClosed = false;
//
//         if (g_pMapProfile->find(strProfile) == g_pMapProfile->end() || dEpsilon<0)
//         {
//             string strErr = "KitL: laser_getLocalExtrepoint 输入轮廓不存在或精度参数错误！";
//             ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//             lua_pushinteger(L, -100);
//             lua_createtable(L, 0, 0);
//             return 2;
//
//         }
//
//         const  vector<cv::Point2d>  &profileIn = (*g_pMapProfile)[strProfile];
//
//         vector < vector <cv::Point>> conProfile;
//         getContinuousProfile(profileIn, conProfile,g_Hoffset);
//
//         vector<vector<cv::Point>> profilePoints(conProfile.size());
//         for (int i = 0; i < conProfile.size();i++)
//         {
//             cv::approxPolyDP(conProfile[i], profilePoints[i], dEpsilon, isClosed);
//         }
//         vector<cv::Point> allPolyPts;
//         for (int i = 0; i < profilePoints.size();i++)
//         {
//             allPolyPts.insert(allPolyPts.end(), profilePoints[i].begin(), profilePoints[i].end());
//         }
//
//         lua_pushinteger(L, 0);
//         lua_createtable(L, allPolyPts.size(), 0);	//create parent table of size vecResPoint.size() array elements
//         for (int i = 0; i < allPolyPts.size(); i++)
//         {
//             lua_pushnumber(L, i + 1);							// puts key of the first child table on-top of lua VM stack
//             lua_createtable(L, 0, 2);							//create first child table of size 3 non-array elements
//             lua_pushnumber(L, allPolyPts[i].y);			//fills the first child table
//             lua_setfield(L, -2, "row");
//             lua_pushnumber(L, allPolyPts[i].x);	//fills the first child table
//             lua_setfield(L, -2, "col");        			//setfield() pops the calue from lua VM stack
//             lua_settable(L, -3);								//Remember,child table is on-top of the stack.
//             //lua_settable() pops key,value pair form lua VM stack
//         }
//
//         return 2;
//    }
//    catch (...)
//    {
//        string strErr = "KitL: laser_getPolyline 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_createtable(L, 0, 0);
//        return 2;
//
//    }
//}
//
///*
//nRet = laser_filterByLen(strLaserIn,strLaserOut,minLen,maxLen)
//输入：
//strLaserIn     -  输入轮廓
//strLaserOut  -  输出轮廓
//minLen         - 筛选的最小轮廓长度
//maxLen        - 筛选的最长轮廓长度
//输出：
//nRet
//*/
//int KitL::laser_filterByLen(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//
//        if (iCount != 4
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TSTRING
//            || lua_type(L, 3) != LUA_TNUMBER
//            || lua_type(L, 4) != LUA_TNUMBER
//            )
//        {
//            string strErr = "KitL: laser_filterByLen 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//        string strLaserIn = lua_tostring(L, 1);
//        string strLaserOut = lua_tostring(L, 2);
//        int  minLen = int(lua_tonumber(L, 3));
//        int maxLen = int(lua_tonumber(L, 4));
//
//        if (g_pMapProfile->find(strLaserIn) == g_pMapProfile->end()
//            || minLen >(*g_pMapProfile)[strLaserIn].size()
//            || minLen  > maxLen )
//        {
//            string strErr = "KitL: laser_filterByLen 中输入轮廓不存在或长度范围错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//
//        if (g_pMapProfile->find(strLaserOut) == g_pMapProfile->end())
//        {
//            (*g_pMapProfile)[strLaserOut] = vector<Point2d>();
//        }
//        (*g_pMapProfile)[strLaserOut] = (*g_pMapProfile)[strLaserIn];
//
//        const vector<cv::Point2d> & laserIn = (*g_pMapProfile)[strLaserIn];
//         vector<cv::Point2d> & laserOut = (*g_pMapProfile)[strLaserOut];
//
//        vector < vector <Point2d >> conProfile;
//        vector<Point2d>   delPts;                //记录要删除的点
//        getContinuousProfile(laserIn, conProfile,g_Hoffset);
//
//        for (int i = 0; i < conProfile.size(); i++)
//        {
//            if (conProfile[i].size() <minLen || conProfile[i].size()>maxLen)
//            {
//                delPts.insert(delPts.end(), conProfile[i].begin(), conProfile[i].end());
//            }
//        }
//
//        if (delPts.size()>0)
//        {
//            for (int i = laserOut.size()-1; i >= 0; i--)
//            {
//                if (laserOut[i] == delPts.back())
//                {
//                    laserOut[i] = INVALID_POINT;
//                    delPts.pop_back();
//                    if (delPts.size()== 0)
//                        break;
//                }
//            }
//        } 
//        if (delPts.size() == 0)
//        {
//            lua_pushinteger(L, 0);
//            return 1;
//        }
//        else
//        {
//            string strErr = "KitL: laser_filterByLen 删除点错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//
//    }
//    catch (...)
//    {
//        string strErr = "KitL: laser_filterByLen 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        return 1;
//    }
//}
//
///*
//nRet,tablePoints = laser_getEndpoint(strLaserIn)
//输入：
//strLaserIn -  输入轮廓
//输出：
//nRet
//tablePoints - 极值与坐标系表
//*/
//int KitL::laser_getEndpoint(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//
//        if (iCount != 1
//            || lua_type(L, 1) != LUA_TSTRING)
//        {
//            string strErr = "KitL: laser_getEndpoint 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_createtable(L, 0, 0);
//            return 2;
//        }
//
//        string strLaserIn = lua_tostring(L, 1);
//        if (g_pMapProfile->find(strLaserIn) == g_pMapProfile->end())
//        {
//            string strErr = "KitL: laser_getEndpoint 输入轮廓不存在！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            lua_createtable(L, 0, 0);
//            return 2;
//        }
//
//        const vector<Point2d> &vecProfile = (*g_pMapProfile)[strLaserIn];
//        vector<vector<Point2d>> conProfile;
//        vector<pair<Point2d, Point2d>> endPts;
//        getContinuousProfile(vecProfile, conProfile,g_Hoffset);
//        if (conProfile.size()>1)
//        {
//            endPts = getEndPoints(conProfile);
//
//            lua_pushinteger(L, 0);
//            lua_createtable(L, endPts.size(), 0);
//            for (int i = 0; i < endPts.size(); i++)
//            {
//                lua_pushnumber(L, i + 1);
//                lua_createtable(L, 0, 4);
//                lua_pushnumber(L, endPts[i].first.y);
//                lua_setfield(L, -2,"startRow");
//                lua_pushnumber(L, endPts[i].first.x);
//                lua_setfield(L, -2, "startCol");
//                lua_pushnumber(L, endPts[i].second.y);
//                lua_setfield(L, -2, "endRow");
//                lua_pushnumber(L, endPts[i].second.x);
//                lua_setfield(L, -2, "endCol");
//                lua_settable(L, -3);
//            }
//            return 2;
//        } 
//        else
//        {
//            lua_pushinteger(L, 0);
//            lua_createtable(L, 0, 0);
//            return 2;
//        }
//    }
//    catch (...)
//    {
//        string strErr = "KitL: laser_getEndpoint 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        lua_createtable(L, 0, 0);
//        return 2;
//    }
//}
//
///*
//nRet = laser_sub(strLaserIn1,strLaserIn2, strLaserOut,iOffset)
//输入：
//strLaserIn1  - 输入轮廓1
//strLaserIn2  - 输入轮廓2 
//strLaserOut - 输出轮廓
//iOffset         - row偏移范围在 [-Offset,Offset]内的轮廓点认为是相等的，用于处理设备轻微位移
//
//输出：
//nRet-
// */
//int KitL::laser_sub(lua_State * L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//
//        if (iCount != 4
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TSTRING
//            || lua_type(L, 3) != LUA_TSTRING
//            || lua_type(L, 4) != LUA_TNUMBER
//            )
//        {
//            string strErr = "KitL: laser_sub 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//        string strLaserIn1 = lua_tostring(L, 1);
//        string strLaserIn2 = lua_tostring(L, 2);
//        string strLaserOut = lua_tostring(L, 3);
//        int offset = int(lua_tonumber(L, 4));
//
//        if (g_pMapProfile->find(strLaserIn1) == g_pMapProfile->end()
//            || g_pMapProfile->find(strLaserIn2) == g_pMapProfile->end())
//        {
//            string strErr = "KitL: laser_sub 中输入轮廓不存在！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//
//        if (g_pMapProfile->find(strLaserOut) == g_pMapProfile->end())
//        {
//            (*g_pMapProfile)[strLaserOut] = vector<Point2d>();
//        }
//        (*g_pMapProfile)[strLaserOut] = (*g_pMapProfile)[strLaserIn1];
//
//        const vector<cv::Point2d> & laserIn1 = (*g_pMapProfile)[strLaserIn1];
//        const vector<cv::Point2d> & laserIn2 = (*g_pMapProfile)[strLaserIn2];
//        if (laserIn1.size() != laserIn2.size())
//        {
//            string strErr = "KitL: laser_sub 中输入轮廓size不相等！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//        vector<cv::Point2d> & laserOut = (*g_pMapProfile)[strLaserOut];
//
//        for (int i = 0; i < laserIn1.size();i++)
//        {
//            if (laserIn1[i] != INVALID_POINT)
//            {
//                if (abs(laserIn1[i].y-laserIn2[i].y)<offset)
//                {
//                    laserOut[i] = INVALID_POINT;
//                }
//            }
//        }
//        lua_pushinteger(L, 0);
//        return 1;
//    }
//    catch (...)
//    {
//        string strErr = "KitL: laser_sub 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        return 1;
//    }
//}
//
///*
//nRet = laser_sub(strLaserIn1,strLaserIn2, strLaserOut,iMode)
//输入：
//strLaserIn1  -  输入轮廓1
//strLaserIn2  -  输入轮廓2
//strLaserOut -  输出轮廓
//iMode          - 当轮廓边缘重叠时,0 取最大值，1取最小值
//输出：
//nRet-
//*/
//int KitL::laser_add(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//        if (iCount != 4
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TSTRING
//            || lua_type(L, 3) != LUA_TSTRING    
//            || lua_type(L, 4) != LUA_TNUMBER    )
//
//        {
//            string strErr = "KitL: laser_add 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//        string strLaserIn1 = lua_tostring(L, 1);
//        string strLaserIn2 = lua_tostring(L, 2);
//        string strLaserOut = lua_tostring(L, 3);
//        int iMode = (int)lua_tonumber(L, 4);
//        if (g_pMapProfile->find(strLaserIn1) == g_pMapProfile->end()
//            || g_pMapProfile->find(strLaserIn2) == g_pMapProfile->end())
//        {
//            string strErr = "KitL: laser_add 中输入轮廓不存在！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//
//        if (g_pMapProfile->find(strLaserOut) == g_pMapProfile->end())
//        {
//            (*g_pMapProfile)[strLaserOut] = vector<Point2d>();
//        }
//
//        const vector<cv::Point2d> & laserIn1 = (*g_pMapProfile)[strLaserIn1];
//        const vector<cv::Point2d> & laserIn2 = (*g_pMapProfile)[strLaserIn2];
//        if (laserIn1.size() != laserIn2.size())
//        {
//            string strErr = "KitL: laser_add 中输入轮廓size不相等！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//        vector<cv::Point2d> & laserOut = (*g_pMapProfile)[strLaserOut];
//        laserOut.assign(laserIn1.size(), INVALID_POINT);
//        for (int i = 0; i < laserIn1.size(); i++)
//        {
//            if (laserIn1[i] != INVALID_POINT || laserIn2[i] !=INVALID_POINT)
//            {
//                if (laserIn1[i] !=INVALID_POINT && laserIn2[i]== INVALID_POINT)
//                {
//                    laserOut[i] = laserIn1[i];
//                }
//                else if (laserIn1[i] == INVALID_POINT && laserIn2[i] != INVALID_POINT)
//                {
//                    laserOut[i] = laserIn2[i];
//                }
//                else
//                {
//                    if (0 == iMode)
//                    {
//                        laserIn1[i].y > laserIn2[i].y ? laserOut[i] = laserIn1[i]: laserOut[i] = laserIn2[i];
//                    } 
//                    else
//                    {
//                        laserIn1[i].y < laserIn2[i].y ? laserOut[i] = laserIn1[i] : laserOut[i] = laserIn2[i];
//                    }
//                }
//            }
//        }
//        lua_pushinteger(L, 0);
//        return 1;
//    }
//    catch (...)
//    {
//        string strErr = "KitL: laser_add 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        return 1;
//    }
//}
//
///*
//nRet = laser_copy(strLaserIn, strLaserOut)
//输入：
//strLaserIn  - 输入轮廓
//strLaserOut - 输出轮廓
//
//输出：
//nRet-
//*/
//int KitL::laser_copy(lua_State *L)
//{
//    try
//    {
//        int iCount = lua_gettop(L);
//
//        if (iCount != 2
//            || lua_type(L, 1) != LUA_TSTRING
//            || lua_type(L, 2) != LUA_TSTRING
//            )
//        {
//            string strErr = "KitL: laser_copy 参数错误！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//        string strLaserIn = lua_tostring(L, 1);
//        string strLaserOut = lua_tostring(L, 2);
//
//        if (g_pMapProfile->find(strLaserIn) == g_pMapProfile->end())
//        {
//            string strErr = "KitL: laser_copy 中输入轮廓不存在！";
//            ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//            lua_pushinteger(L, -100);
//            return 1;
//        }
//
//        if (g_pMapProfile->find(strLaserOut) == g_pMapProfile->end())
//        {
//            (*g_pMapProfile)[strLaserOut] = vector<Point2d>();
//        }
//
//        const vector<cv::Point2d> & laserIn = (*g_pMapProfile)[strLaserIn];
//
//        vector<cv::Point2d> & laserOut = (*g_pMapProfile)[strLaserOut];
//        laserOut.assign( laserIn.begin(),laserIn.end());
//
//        lua_pushinteger(L, 0);
//        return 1;
//    }
//    catch (...)
//    {
//        string strErr = "KitL: laser_copy 捕获到C++异常！";
//        ::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//        lua_pushinteger(L, -100);
//        return 1;
//    }
//}
//
///*
//nRet,tablePoints = laser_getLocalExtrepoint(strLaserIn,iMode,iNeighbor,iHightThresh,iIsClosed)
//输入：
//strLaserIn -  输入轮廓
//iMode    -  极值模式，0 - 局部极大值，1 - 局部最小值 ，2 - 局部极值
//iNeighbor	   - 筛选邻域，在筛选邻域内只取一个极值
//iHightThresh - 未超过高度差阈值为有效极值点，防止轮廓噪声过大产生干扰。
//iIsClosed     - 是否把轮廓看做首尾相连的闭合轮廓，一般用于处理旋转扫描物体
//输出：
//tablePoints - 极值与坐标系表
//*/
//
//int KitL::laser_getLocalExtrepoint(vector<)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//
//		if (iCount != 5
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TNUMBER
//			|| lua_type(L, 3) != LUA_TNUMBER
//			|| lua_type(L, 4) != LUA_TNUMBER
//			|| lua_type(L, 5) != LUA_TNUMBER)
//		{
//			string strErr = "KitL: laser_getLocalExtrepoint 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0,0);
//			return 2;
//		}
//		string strLaserIn = lua_tostring(L, 1);
//		int  mode= (int)lua_tonumber(L, 2);
//		int  nei= (int)lua_tonumber(L, 3);
//		int hThresh = (int)lua_tonumber(L, 4);
//		int isClosed = (int)lua_tonumber(L, 5);
//
//		if (g_pMapProfile->find(strLaserIn) == g_pMapProfile->end()|| mode<0 || mode>2)
//		{
//			string strErr = "KitL: laser_getLocalExtrepoint 输入轮廓不存在！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			lua_createtable(L, 0, 0);
//			return 2;
//		}
//
//		const vector<Point2d> &vecProfile = (*g_pMapProfile)[strLaserIn];
//		vector<vector<Point2d>> contProf;
//		vector<vector<Point2d>> tempProf;
//		vector<Point2d> extrePoints;
//
//		getContinuousProfile(vecProfile, contProf,g_Hoffset);
//		switch (mode)
//		{
//		case 0://局部极大值
//			for (int i = 0; i < contProf.size(); i++)
//			{
//				tempProf = getHeightestPointsOfProfile(contProf[i], nei, hThresh,isClosed);
//
//				for (int j = 0; j < tempProf.size(); j++)
//				{
//					extrePoints.push_back(tempProf[j][tempProf[j].size() / 2]);   //极值位于平峰处会出现多个极值点
//				}
//			}
//			break;
//		case 1://局部极小值
//			for (int i = 0; i < contProf.size(); i++)
//			{
//				tempProf = getLowestPointsOfProfile(contProf[i], nei, hThresh,isClosed);
//
//				for (int j = 0; j < tempProf.size(); j++)
//				{
//					extrePoints.push_back(tempProf[j][tempProf[j].size() / 2]);
//				}
//			}
//			break;
//		case 2://局部极值
//			for (int i = 0; i < contProf.size(); i++)
//			{
//				tempProf = getExtrePointOfProfile(contProf[i], nei, hThresh,isClosed);
//
//				for (int j = 0; j < tempProf.size(); j++)
//				{
//					extrePoints.push_back(tempProf[j][tempProf[j].size() / 2]);
//				}
//			}
//			break;
//		default:
//			break;
//		}
//
//		lua_pushinteger(L, 0);
//		lua_createtable(L, extrePoints.size(), 0);	//create parent table of size vecResPoint.size() array elements
//		for (int i = 0; i < extrePoints.size(); i++)
//		{
//			lua_pushnumber(L, i + 1);							// puts key of the first child table on-top of lua VM stack
//			lua_createtable(L, 0, 2);							//create first child table of size 3 non-array elements
//			lua_pushnumber(L, extrePoints[i].y);			//fills the first child table
//			lua_setfield(L, -2, "row");
//			lua_pushnumber(L, extrePoints[i].x);	//fills the first child table
//			lua_setfield(L, -2, "col");        			//setfield() pops the calue from lua VM stack
//			lua_settable(L, -3);								//Remember,child table is on-top of the stack.
//																		//lua_settable() pops key,value pair form lua VM stack
//		}
//		
//		return 2;
//	}
//	catch (...)
//	{
//		string strErr = "KitL: laser_getLocalExtrepoint 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		lua_createtable(L, 0, 0);
//		return 2;
//	}
//}
//
///**************************************************
//nRet = img_drawLaser(strImgIn,strImgOut,strProfileIn,iThickness)
//功能：
//	绘制指定轮廓线到指定输出图像
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
//int KitL::img_drawLaser(lua_State *L)
//{
//	try
//	{
//		int iCount = lua_gettop(L);
//
//		if (iCount != 4
//			|| lua_type(L, 1) != LUA_TSTRING
//			|| lua_type(L, 2) != LUA_TSTRING
//			|| lua_type(L, 3) != LUA_TSTRING
//			|| lua_type(L, 4) != LUA_TNUMBER
//			)
//		{
//			string strErr = "KitL: img_drawLaser 参数错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//		string strImgIn = lua_tostring(L, 1);
//		string strImgOut = lua_tostring(L, 2);
//		string strProfileIn = lua_tostring(L, 3);
//		int thickness = int(lua_tonumber(L, 4));
//
//		if (g_pMapImage->find(strImgIn) == g_pMapImage->end() 
//			|| g_pMapProfile->find(strProfileIn) == g_pMapProfile->end()
//			|| thickness<0 )
//		{
//			string strErr = "KitL: img_drawLaser 中输入图像或轮廓不存在或线宽范围错误！";
//			::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//			lua_pushinteger(L, -100);
//			return 1;
//		}
//
//		if (g_pMapImage->find(strImgOut) == g_pMapImage->end())
//		{
//			(*g_pMapImage)[strImgOut] = cv::Mat();
//		}	
//
//		const cv::Mat& imgIn = (*g_pMapImage)[strImgIn];
//		cv::Mat & imgOut = (*g_pMapImage)[strImgOut];
//		vector<Point2d> & profileIn = (*g_pMapProfile)[strProfileIn];
//        if ( imgIn.channels() ==1)
//        {
//		    cvtColor(imgIn, imgOut, CV_GRAY2BGR);
//        }
//        else
//        {
//            imgOut = imgIn.clone();
//        }
//		
//		vector < vector <Point2d >> conProfile;
//		getContinuousProfile(profileIn, conProfile,g_Hoffset);
//
//		for (int i = 0; i < conProfile.size();i++)
//		{
//			vector<Point> temp(conProfile[i].begin(), conProfile[i].end());
//			cv::polylines(imgOut, temp, false, Scalar(0, 0, 255), thickness);
//		}
//
//		lua_pushinteger(L, 0);
//		return 1;
//	}
//	catch (...)
//	{
//		string strErr = "KitL: img_drawLaser 捕获到C++异常！";
//		::SendMessageA(g_winHandle, 1002, -100, (LPARAM)strErr.c_str());   // 发送错误代码
//		lua_pushinteger(L, -100);
//		return 1;
//	}
//}


