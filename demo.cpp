#include <iostream>
#include <map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "extractScratch.h"
//#include "imgPro.h"

//#include "labelRotRect.h"

using namespace cv;
using namespace std;

Mat srcImg, srcImg2Crop, srcImg2Show;
string srcPath;
//int procImg(cv::Mat cropImg, cv::RotatedRect rotROI);
int procImg(cv::Mat cropImg, cv::Rect rect);

int binImg2vectorPt(cv::Mat binImg, vector<cv::Point>& vecPt);
int getConnPairPt(cv::Mat labelImg, int centerRegionLabel, vector<cv::Point> vecCentPts, vector<pair<cv::Point, cv::Point> > &vecPairPt);

cv::Point rectTLPt, rectRDPt;
cv::Point lineStartPt, lineEndPt;
bool rectSetFlag = false;
bool mouseMove = false;
bool rectStartFlag = false;
static void onMouseSrcImg(int event, int x, int y, int, void*)
{
	if (event == EVENT_LBUTTONDOWN)
	{
			rectTLPt = cv::Point(x, y);
			rectStartFlag = true;
			mouseMove = true;
	}
	if (event == EVENT_MOUSEMOVE && (mouseMove))
	{
		Mat showImg;
		showImg = srcImg2Show.clone();
		if (rectStartFlag)
		{
			rectRDPt = cv::Point(x, y);
			rectangle(showImg, Rect(rectTLPt, rectRDPt), Scalar(0, 255, 0), 1);
		}
		imshow("srcImgWin", showImg);
	}

	if (event == EVENT_LBUTTONUP)
	{
		Mat dst, dstMask;
		if (rectStartFlag)
		{
			rectangle(srcImg2Show, Rect(rectTLPt, rectRDPt), Scalar(0, 255, 0), 1);
			
			rectStartFlag = false;
			mouseMove = false;
			imshow("srcImgWin", srcImg2Show);
			procImg(srcImg, Rect(rectTLPt, rectRDPt));  // procImg 实际工作的函数 
		}
		//处理

	}

}
//static int draw_symbol = 0;
//static void onMouseSrcImg(int event, int x, int y, int, void* )
//{
//	static Point longEdgePt1;
//	static Point longEdgePt2;
//	static  Point thirdPt;
//	int longEdge = 0, shortEdge = 0;
//
//
//	if (event == EVENT_LBUTTONUP)
//	{
//		switch (draw_symbol)
//		{
//		case  0:
//			draw_symbol = 1;
//			longEdgePt1 = Point(x, y);
//			break;
//		case 1:
//			draw_symbol = 2;
//			longEdgePt2 = Point(x, y);
//			break;
//		case 2:
//			draw_symbol = 3;
//			thirdPt = Point(x, y);
//			break;
//
//		default:
//			break;
//		}
//
//		char srcSavePath[100];
//		char maskSavePath[100];
//		//sprintf_s(srcSavePath, "F:\\MyData\\CrackForest-dataset-master\\myCrack\\image\\%3d.jpg", numCnt);
//		//sprintf_s(maskSavePath, "F:\\MyData\\CrackForest-dataset-master\\myCrack\\label\\%3d.jpg", numCnt);
//
//		//imwrite(srcSavePath, dst);
//		//imwrite(maskSavePath, dstMask);
//
//	}
//	if (event == EVENT_MOUSEMOVE)
//	{
//		Mat showRectImg;
//		showRectImg = srcImg.clone();
//		if (draw_symbol == 3)
//		{
//			draw_symbol = 0;
//			vector<Point>  vecpts = drawRotRect1(showRectImg, longEdgePt1, longEdgePt2, thirdPt);
//			outInfo rotRectData = drawCVRotRect(showRectImg, vecpts[0], vecpts[1], vecpts[2]);
//
//			cv::RotatedRect cropRotRect(cv::Point(int(rotRectData.centerX),int(rotRectData.centerY)),
//										cv::Size(rotRectData.shortEdge,rotRectData.longEdge),rotRectData.angle);
//			cv::RotatedRect rotRect = cv::minAreaRect(vecpts);
//			procImg(srcImg, rotRect);
//			//srcImg = showRectImg.clone();
//		}
//		if (draw_symbol == 1)
//		{
//			line(showRectImg, longEdgePt1, Point(x, y), Scalar(0, 255, 0), 1);
//		}
//		if (draw_symbol == 2)
//		{
//			drawRotRect1(showRectImg, longEdgePt1, longEdgePt2, Point(x, y));
//		}
//
//		imshow("srcImgWin", showRectImg);
//	}
//}


cv::Mat resShowImg;
int main(int argc, char** argv)
{
	namedWindow("srcImgWin", WINDOW_AUTOSIZE );
	vector<cv::String> vecSrcPathes, vecMaskPathes;
	//glob(srcPathes, vecSrcPathes);
	{
		//
		//srcPath = "E:\\5_Snapbox\\scratch_crop_hessian_5_bin_crop1_1.png";
		srcPath = "C:/C++Projects/ScratchExtract1210/test3.png";//31-lcd
		//srcPath = "E:\\5_Snapbox\\前期增强验证\\0079-220-1-168-5900-A1 - 副本.jpg";
		string path = srcPath;//vecSrcPathes[i]
		srcImg = imread(path, 1);
		resShowImg = srcImg.clone();
		srcImg2Crop = srcImg.clone();
		srcImg2Show = srcImg.clone();
		cout << srcImg.size() << endl;
		if (srcImg.empty())
		{
			std::cout << "image doesnt exist!!" << endl;
			return 0;
		}
		imshow("srcImgWin", srcImg);
		cv::Mat toProc;
		cvtColor(srcImg, toProc, COLOR_BGR2GRAY);
		//while (true)
		{
			setMouseCallback("srcImgWin", onMouseSrcImg, 0);  // 实际工作的函数onMouseSrcImg
			//std::cout << "after callback  :" << getImgCropFlag << endl;
		}
		if (waitKey(0) == 32)  //32 is 
			return 1;
	}
	std::cout << "over!" << endl;
	cv::waitKey();
	return 0;
}
int region_connnetion(cv::Mat binImg, cv::Mat &connImg, cv::RotatedRect rotRect);
int procImg1(cv::Mat srcImg, cv::RotatedRect rotROI)
{
	cv::Mat enhanceImg, binImg,cropImg;
	if (srcImg.channels() >1)
	{
		cv::cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
	}

	cv::Mat srcImgInverse = 255 - srcImg;
	hessianEnhance(srcImgInverse, enhanceImg);

	cv::threshold(enhanceImg, binImg, 20, 255, CV_8UC1);
	imgPro::img_getPartR(binImg, cropImg, rotROI);

	//图像细化，骨骼化  
	Mat zerOneImg;
	cv::threshold(cropImg, zerOneImg, 10, 1, cv::THRESH_BINARY);
	cv::Mat thinImg = thinImage(zerOneImg);

	thinImg = thinImg * 255;

	cv::Mat showThinImg;
	cvtColor(cropImg, showThinImg, COLOR_GRAY2BGR);

	vector<cv::Point> vecImgPts;
	for (int row = 0; row < cropImg.rows; row++)
	{
		for (int col = 0; col < cropImg.cols; col++)
		{
			if (thinImg.at<uchar>(row, col) > 0)
				vecImgPts.push_back(cv::Point(col, row));
		}
	}

	Vec4f linePara;
	cv::fitLine(vecImgPts, linePara, DIST_FAIR, 0, 0.01, 0.01);

	cv::Point point0;
	point0.x = linePara[2];
	point0.y = linePara[3];
	double k = linePara[1] / linePara[0];

	//找出直线在图像内的端点
	vector<cv::Point > vecPts;
	cv::Point point1, point2, point3, point4;
	// cv::Point startPt(-1, -1), endPt(-1, -1);
	int xEdge = cropImg.cols - 1;
	int yEdge = cropImg.rows - 1;
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

	vector<cv::Point> vecPtsFilter;
	vecPtsFilter = uniqueVecPts(vecPts);


	if (vecPtsFilter.size() != 2)
	{
		string strErr = "imgPro: region_fitLine 函数中第一次寻找绘制点错误！";
		cout << strErr << endl;
		return -1;
	}
	cv::Point  startPt = vecPtsFilter[0];
	cv::Point  endPt = vecPtsFilter[1];

	//cv::line(showThinImg, startPt, endPt, Scalar(0, 0, 255), 1);
	//找出原图在直线的区域

	cv::Mat canv = cv::Mat::zeros(cropImg.size(), CV_8UC1);
	cv::line(canv, startPt, endPt, Scalar(255), 5);

	//形态学重建 to do
	cv::Mat validImg, rebuildImg;
	cv::bitwise_and(canv, cropImg, validImg);

	morRebuild(cropImg, validImg, rebuildImg);
	//将重建区域连接成一条线
	cv::Mat connImg, maskShowImg;
	cv::RotatedRect rotRect;
	imgPro::biImg_getRotRect(rebuildImg, rotRect);
	cv::Point t;
	imgPro::img_drawRect(rebuildImg, showThinImg, t, rotRect.center, rotRect.size, -rotRect.angle, Scalar(0, 0, 255));

	region_connnetion(rebuildImg, connImg, rotRect);

	//conImg去除交叉点
	double scratchLen;
	cv::Mat resImg = removeIntersection(connImg, scratchLen);
	//计算划痕与周边对比度
	cv::Mat regionAroundScratch = getScratchAround(connImg, resImg);

	imgPro::img_drawMask(srcImg, maskShowImg, connImg, Scalar(0, 0, 255), 0.3);

	return 0;
}
int procImg2(cv::Mat srcImg, cv::RotatedRect rotROI)
{
	cv::Mat enhanceImg, srcBinImg,cropImg;
	if (srcImg.channels() >1)
	{
		cv::cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
	}

	cv::Mat srcImgInverse = 255 - srcImg;
	hessianEnhance(srcImgInverse, enhanceImg);
	cv::threshold(enhanceImg, srcBinImg, 10, 255, CV_8UC1);
	imgPro::img_getPartR(srcBinImg, cropImg, rotROI);
	cv::Rect scratchRectRaw = rotROI.boundingRect();	
	cv::Rect scratchRect =  imgPro::rect_normInImg(scratchRectRaw, srcImg.size());
	cv::Mat scratchRectImg;
	cropImg(scratchRect).copyTo(scratchRectImg);
	cv::Mat scratchRectImg2, scratchRectImg3;
	cv::morphologyEx(scratchRectImg, scratchRectImg2, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	cv::Mat thin1 = thinImage(scratchRectImg2);
	cv::bitwise_or(thin1, scratchRectImg, scratchRectImg);

	//cv::morphologyEx(thin1, scratchRectImg3, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	//cv::Mat thin2 = thinImage(scratchRectImg3);
	//图像细化，骨骼化  

	cv::Mat thinImg = thinImage(scratchRectImg);

	cv::Mat labelImg, showThinImg, maxScratchImgRaw;
	map<string, vector<double>> feats;
	cvtColor(scratchRectImg2, showThinImg, COLOR_GRAY2BGR);
	cv::Mat maxScratchImg, noMaxScratchImg;
	cv::Mat toConnImg = thinImg.clone();
	vector<Point> vecMaxScratchEnds;
	vector<pair<Point, Point>> vecPairPt;

	cv::Mat preImg = toConnImg.clone();
	cv::Mat nowImg = toConnImg.clone();
	double maxScratchLen = 0;
	while (true)
	{
		cv::Mat maxScratchImg;
		imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, thinImg.total());   //labelImg作为标记使用
		imgPro::biImg_getMaxArea(nowImg, maxScratchImg);
		//去除交点
	//	skeleton_removeIntersection(maxScratchImgRaw,maxScratchImg);

		skeleton_endpoint(maxScratchImg, vecMaxScratchEnds);

		for (auto endPt:vecMaxScratchEnds)
		{
			double distThresh = 12;
			double dist = 0;
			map<int, vector<cv::Point>> mapVecPts;
			int maxRegionLabel = labelImg.at<int>(endPt);
			cv::Point endPtNew;
			while (true)
			{
				//回溯7个像素计算角度
				double angleEnd = angleFromEndPt(maxScratchImg, endPt,endPtNew,13);
				//计算该点本区域外最近的点，所在区域的角度，距离，连线角度
					labelImg2vectorPts(labelImg, mapVecPts, maxRegionLabel);
					vector<Point>	vecPts{ endPt };
				pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecPts, mapVecPts);
			
				double distPtPair = imgPro::dist2Pts(res.second.first, res.second.second);
				//连接点对与末端像素段的向量夹角角度，与180之差应该小于35度。
				double anglePtPairWithPartMaxScratch;
				imgPro::angle2Lines(endPt, endPtNew, res.second.first, res.second.second, anglePtPairWithPartMaxScratch);
				//imgPro::lineseg_degree(endPt, endPtNew, res.second.first, res.second.second, anglePtPairWithPartMaxScratch);
				if (distPtPair > distThresh && (180 - anglePtPairWithPartMaxScratch) >15)
				{
					break;
				}

				double connectedRegionAngle;
				int connectedRegionIndex = labelImg.at<int>(res.second.second);

				cv::Mat regionImg;
				imgPro::region_toImg(labelImg, regionImg, connectedRegionIndex);
				if (cv::countNonZero(regionImg) > 3)
				{
					vector<Point> ptsTmp;
					skeleton_endpoint(regionImg, ptsTmp);
					connectedRegionAngle = anglePostive(ptsTmp[0], ptsTmp[1]);
				} 
				else
				{
					connectedRegionAngle = angleEnd;
				}
			
				if ((180 - anglePtPairWithPartMaxScratch) < 15		||
					((180 - anglePtPairWithPartMaxScratch) < 35 && abs(connectedRegionAngle - angleEnd)<35 )
					)
				{
					//可连接点对
					vecPairPt.push_back(res.second);
					//连接图像
					cv::line(nowImg, vecPairPt.back().first, vecPairPt.back().second, Scalar(255));
					break;
				}
				else
				{
					//移除该label,继续连接
					cv::Mat region2Remove;
					imgPro::region_toImg(labelImg, region2Remove,labelImg.at<int>(res.second.second) );

					nowImg = nowImg - region2Remove;
					//将label中的label去掉
					vector<Point> pt2Remmove;
					imgPro::img_getPoints(region2Remove, pt2Remmove);
					for (Point pt : pt2Remmove)
					{
						labelImg.at<int>(pt) = 0;
					}
				}
			}
		}

		cv::Mat diff =(nowImg - preImg);
		if (countNonZero(diff) == 0)  //不再增加连接区域则跳出
		{
			break;
		} 
		else {
			preImg = nowImg.clone();
		}
	}
	
	//conImg去除交叉点
	double scratchLen;
	//cv::Mat resImg = removeIntersection(connImg, scratchLen);
	cv::Mat resImg, showImg, connImg,thinScratchImg, scratchMask;
	cv::bitwise_or(nowImg, scratchRectImg2, connImg);
	//划痕长度
	imgPro::biImg_getMaxArea(nowImg, thinScratchImg);
	vector<Point> vecThinPts;
	imgPro::img_getPoints(thinScratchImg, vecThinPts);
	 scratchLen = vecThinPts.size();

	 imgPro::biImg_getMaxArea(connImg, resImg);
	 cv::Mat scratchMaskTmp = cv::Mat::zeros(srcImg.size(), CV_8UC1);
	 resImg.copyTo(scratchMaskTmp(scratchRect));
	 cv::bitwise_and(scratchMaskTmp, srcBinImg, scratchMask);

	 //划痕对比度
	 cv::Mat srctchRoundMask =  getRawScratchRound(scratchMaskTmp);
	 cv::Mat scratchMeanImg, scratchStdImg;
	 cv::Mat roundMeanImg, roundStdImg;
	 cv::meanStdDev(srcImg, scratchMeanImg, scratchStdImg, scratchMask);
	 cv::meanStdDev(srcImg, roundMeanImg, roundStdImg, srctchRoundMask);

	 double scratchMean = scratchMeanImg.at<double>(0, 0);
	 double roundScratchMean = roundMeanImg.at<double>(0, 0);
	 double meanDiff = abs(scratchMean - roundScratchMean);
	 double scratchStd = scratchStdImg.at<double>(0, 0);

	 string scratchInfo = to_string(scratchLen).substr(0,5) + "_" + to_string(meanDiff).substr(0, 5) + "_" + to_string(scratchStd).substr(0, 5);

	imgPro::img_drawMask(srcImg, showImg, scratchMaskTmp, Scalar(0, 0, 255));    //
	imgPro::img_drawMask(showImg, showImg, scratchMask, Scalar(255, 0, 255));    //
	imgPro::img_drawMask(showImg, showImg, srctchRoundMask, Scalar(0, 255, 2505));

	vector<Point> rotPts;
	float angleTmp;
	imgPro::rect_getRotRectPts(rotROI, rotPts, angleTmp);
	cv::Point showTxtPt = (rotPts[0] + rotPts[1]) / 2;
	 cv::putText(showImg,scratchInfo,showTxtPt,1,1,Scalar(0,0,255));
	 imgPro::img_drawRect(showImg, showImg, rotROI, Scalar(0, 255, 0));
	 
	cv::imshow("res", showImg);
	cv::waitKey();
	return 0;
}
int procImg3(cv::Mat srcImg, cv::RotatedRect rotROI)
{
	cv::Mat enhanceImg, srcBinImg,cropImg;
	if (srcImg.channels() >1)
	{
		cv::cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
	}

	cv::Mat srcImgInverse = 255 - srcImg;
	hessianEnhance(srcImgInverse, enhanceImg);
	
	cv::threshold(enhanceImg, srcBinImg, 10, 255, CV_8UC1);
	imgPro::img_getPartR(srcBinImg, cropImg, rotROI);
	cv::Rect scratchRectRaw = rotROI.boundingRect();	
	cv::Rect scratchRect =  imgPro::rect_normInImg(scratchRectRaw, srcImg.size());
	cv::Mat scratchRectImg;
	cropImg(scratchRect).copyTo(scratchRectImg);
	cv::Mat scratchRectImgCopy,mor1, scratchRectImg2;
	scratchRectImgCopy = scratchRectImg.clone();
	cv::morphologyEx(scratchRectImgCopy, mor1, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	cv::Mat thin1 = thinImage(mor1);
	//cv::bitwise_or(thin1, scratchRectImg, scratchRectImg2);
	scratchRectImg2 = thin1;

	//cv::morphologyEx(thin1, scratchRectImg3, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	//cv::Mat thin2 = thinImage(scratchRectImg3);
	//图像细化，骨骼化  

	cv::Mat thinImg = thinImage(scratchRectImg2);

	cv::Mat labelImg, showThinImg, maxScratchImgRaw;
	map<string, vector<double>> feats;
	cvtColor(scratchRectImg2, showThinImg, COLOR_GRAY2BGR);
	cv::Mat maxScratchImg, noMaxScratchImg;
	cv::Mat toConnImg = thinImg.clone();
	vector<Point> vecMaxScratchEnds;
	vector<pair<Point, Point>> vecPairPt;

	cv::Mat preImg = toConnImg.clone();
	cv::Mat nowImg = toConnImg.clone();

	double maxScratchLen = 0;
	while (true)
	{
		cv::Mat maxScratchImg;
		imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, thinImg.total());   //labelImg作为标记使用
		imgPro::biImg_getMaxArea(nowImg, maxScratchImg);
		//去除交点
	//	skeleton_removeIntersection(maxScratchImgRaw,maxScratchImg);

		skeleton_endpoint(maxScratchImg, vecMaxScratchEnds);

		for (auto endPt:vecMaxScratchEnds)
		{
			double distThresh = 50;
			double dist = 0;
			map<int, vector<cv::Point>> mapVecPts;
			int maxRegionLabel = labelImg.at<int>(endPt);
			cv::Point endPtNew;
			while (true)
			{
				//回溯像素计算角度
				double angleEnd = angleFromEndPt(maxScratchImg, endPt,endPtNew,13);
				//计算该点本区域外最近的点，所在区域的角度，距离，连线角度
				labelImg2vectorPts(labelImg, mapVecPts, maxRegionLabel);
				vector<Point>	vecPts{ endPt };
				pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecPts, mapVecPts);
			
				double distPtPair = imgPro::dist2Pts(res.second.first, res.second.second);
				//连接点对与末端像素段的向量夹角角度，与180之差应该小于35度。
				double anglePtPairWithPartMaxScratch;
				imgPro::angle2Lines(endPt, endPtNew, res.second.first, res.second.second, anglePtPairWithPartMaxScratch);
				if (distPtPair > distThresh && (180 - anglePtPairWithPartMaxScratch) >15)
				{
					break;
				}

				double connectedRegionAngle;
				int connectedRegionIndex = labelImg.at<int>(res.second.second);

				cv::Mat regionImg;
				imgPro::region_toImg(labelImg, regionImg, connectedRegionIndex);
				if (cv::countNonZero(regionImg) > 3)
				{
					vector<Point> ptsTmp;
					skeleton_endpoint(regionImg, ptsTmp);
					if (ptsTmp.size() < 2)
					{
						break;
					}
					connectedRegionAngle = anglePostive(ptsTmp[0], ptsTmp[1]);
				} 
				else
				{
					connectedRegionAngle = angleEnd;
				}
			
				if ((180 - anglePtPairWithPartMaxScratch) < 15		||
					((180 - anglePtPairWithPartMaxScratch) < 35 && abs(connectedRegionAngle - angleEnd)<35 )
					)
				{
					//可连接点对
					vecPairPt.push_back(res.second);
					//连接图像
					cv::line(nowImg, vecPairPt.back().first, vecPairPt.back().second, Scalar(255));
					break;
				}
				else
				{
					//移除该label,继续连接
					cv::Mat region2Remove;
					imgPro::region_toImg(labelImg, region2Remove,labelImg.at<int>(res.second.second) );

					nowImg = nowImg - region2Remove;
					//将label中的label去掉
					vector<Point> pt2Remmove;
					imgPro::img_getPoints(region2Remove, pt2Remmove);
					for (Point pt : pt2Remmove)
					{
						labelImg.at<int>(pt) = 0;
					}
				}
			}
		}

		cv::Mat diff =(nowImg - preImg);
		if (countNonZero(diff) == 0)  //不再增加连接区域则跳出
		{
			break;
		} 
		else {
			preImg = nowImg.clone();
		}
	}
	
	//conImg去除交叉点
	double scratchLen;
	//cv::Mat resImg = removeIntersection(connImg, scratchLen);
	cv::Mat resImg, showImg, connImg,thinScratchImg, scratchMask;
	cv::bitwise_or(nowImg, scratchRectImg2, connImg);
	//划痕长度
	imgPro::biImg_getMaxArea(nowImg, thinScratchImg);
	vector<Point> vecThinPts;
	imgPro::img_getPoints(thinScratchImg, vecThinPts);
	 scratchLen = vecThinPts.size();

	 imgPro::biImg_getMaxArea(connImg, resImg);
	 cv::Mat scratchMaskTmp = cv::Mat::zeros(srcImg.size(), CV_8UC1);
	 resImg.copyTo(scratchMaskTmp(scratchRect));
	 cv::bitwise_and(scratchMaskTmp, srcBinImg, scratchMask);

	 //划痕对比度
	 cv::Mat srctchRoundMask =  getRawScratchRound(scratchMaskTmp);
	 cv::Mat scratchMeanImg, scratchStdImg;
	 cv::Mat roundMeanImg, roundStdImg;
	 cv::meanStdDev(srcImg, scratchMeanImg, scratchStdImg, scratchMask);
	 cv::meanStdDev(srcImg, roundMeanImg, roundStdImg, srctchRoundMask);

	 double scratchMean = scratchMeanImg.at<double>(0, 0);
	 double roundScratchMean = roundMeanImg.at<double>(0, 0);
	 double meanDiff = abs(scratchMean - roundScratchMean);
	 double scratchStd = scratchStdImg.at<double>(0, 0);

	 string scratchInfo = to_string(scratchLen).substr(0,5) + "_" + to_string(meanDiff).substr(0, 5) + "_" + to_string(scratchStd).substr(0, 5);

	imgPro::img_drawMask(srcImg, showImg, scratchMaskTmp, Scalar(0, 0, 255));    //
	imgPro::img_drawMask(showImg, showImg, scratchMask, Scalar(255, 0, 255));    //
	imgPro::img_drawMask(showImg, showImg, srctchRoundMask, Scalar(0, 255, 2505));

	vector<Point> rotPts;
	float angleTmp;
	imgPro::rect_getRotRectPts(rotROI, rotPts, angleTmp);
	cv::Point showTxtPt = (rotPts[0] + rotPts[1]) / 2;
	 cv::putText(showImg,scratchInfo,showTxtPt,1,1,Scalar(0,0,255));
	 imgPro::img_drawRect(showImg, showImg, rotROI, Scalar(0, 255, 0));
	 
	cv::imshow("res", showImg);
	cv::waitKey();
	return 0;
}

//不连接断线，只用现有的线段进行灰度差值计算
//int procImg(cv::Mat srcImg, cv::RotatedRect rotROI)
//{
//	cv::Mat enhanceImg, scratchRectImgBin,cropImg, connImg;
//	if (srcImg.channels() >1)
//	{
//		cv::cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
//	}
//	cv::Mat scratchRectImg, srcRectBinMean;
//	cv::Rect scratchRectRaw = rotROI.boundingRect();	
//	cv::Rect scratchRect =  imgPro::rect_normInImg(scratchRectRaw, srcImg.size());
//	srcImg(scratchRect).copyTo(scratchRectImg);
//
//	cv::Mat scratchRectImgInverse = 255 - scratchRectImg;
//	hessianEnhance(scratchRectImgInverse, enhanceImg);
//	cv::threshold(enhanceImg, scratchRectImgBin, 6, 255, CV_8UC1);
//	int c = 2;
//
//	cv::adaptiveThreshold(scratchRectImg, srcRectBinMean, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C,
//		cv::ThresholdTypes::THRESH_BINARY, 37, -c);
//
//	cv::RotatedRect rotRectShift = rotROI;
//	rotRectShift.center = cv::Point2f(rotRectShift.center.x - scratchRect.tl().x,rotRectShift.center.y - scratchRect.tl().y);
//	imgPro::img_getPartR(scratchRectImgBin, cropImg, rotRectShift);
//	cv::Mat rebuildImg;
//	
//	//判断最大倾斜矩形的长边
//	morRebuild(srcRectBinMean, cropImg, rebuildImg);
//
//	biImg_connection(rebuildImg, connImg, 4);
//	
//	cv::Mat filterImg;
//
//	imgPro::biImg_filterByArea(connImg, filterImg, 5, connImg.total(), 2);
//
//	cv::Mat binImg = thinImage(filterImg);
//	cv::Mat filterBinImg;
//	imgPro::biImg_filterByArea(binImg, filterBinImg, 2, filterImg.total(), 2);
//	
//	//提取最大划痕区域，去除分支
//	
//	vector<Point> vecMaxScratchEnds;
//	vector<pair<Point, Point>> vecPairPt;
//
//	cv::Mat preImg, nowImg,scratchThinImg;
//	nowImg  = filterBinImg.clone();
//	preImg  = filterBinImg.clone();
//
//	while (true)
//	{
//		cv::Mat labelImg;
//		map<string, vector<double>> feats;
//		cv::Mat maxScratchImg, maxScratchImgRaw;
//		imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, nowImg.total());   //labelImg作为标记使用
//		imgPro::biImg_getMaxArea(nowImg, maxScratchImgRaw);
//		//去除交点
//		skeleton_removeIntersectionLine(maxScratchImgRaw,maxScratchImg);
//
//		skeleton_endpoint(maxScratchImg, vecMaxScratchEnds);
//
//		for (auto endPt : vecMaxScratchEnds)
//		{
//			double distThresh = 50;
//			double dist = 0;
//			map<int, vector<cv::Point>> mapVecPts;
//			int maxRegionLabel = labelImg.at<int>(endPt);
//			cv::Point endPtNew;
//			while (true)
//			{
//				//回溯像素计算角度
//				double angleEnd = angleFromEndPt(maxScratchImg, endPt, endPtNew, 13);
//				//计算该点本区域外最近的点，所在区域的角度，距离，连线角度
//				labelImg2vectorPts(labelImg, mapVecPts, maxRegionLabel);
//				vector<Point>	vecPts{ endPt };
//				pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecPts, mapVecPts);
//
//				double distPtPair = imgPro::dist2Pts(res.second.first, res.second.second);
//				//连接点对与末端像素段的向量夹角角度，与180之差应该小于35度。
//				double anglePtPairWithPartMaxScratch;
//				imgPro::angle2Lines(endPt, endPtNew, res.second.first, res.second.second, anglePtPairWithPartMaxScratch);
//				if ( (distPtPair > distThresh && (180 - anglePtPairWithPartMaxScratch) > 15 )|| (180 - anglePtPairWithPartMaxScratch) > 145)
//				{
//					break;
//				}
//
//				double connectedRegionAngle;
//				int connectedRegionIndex = labelImg.at<int>(res.second.second);
//
//				cv::Mat regionImg;
//				imgPro::region_toImg(labelImg, regionImg, connectedRegionIndex);
//				if (cv::countNonZero(regionImg) > 3)
//				{
//					vector<Point> ptsTmp;
//					skeleton_endpoint(regionImg, ptsTmp);
//					if (ptsTmp.size() < 2)
//					{
//						break;
//					}
//					connectedRegionAngle = anglePostive(ptsTmp[0], ptsTmp[1]);
//				}
//				else
//				{
//					connectedRegionAngle = angleEnd;
//				}
//
//				if ((180 - anglePtPairWithPartMaxScratch) < 15 ||
//					((180 - anglePtPairWithPartMaxScratch) < 35 && abs(connectedRegionAngle - angleEnd) < 35)
//					)
//				{
//					//可连接点对
//					vecPairPt.push_back(res.second);
//					//连接图像
//					cv::line(nowImg, vecPairPt.back().first, vecPairPt.back().second, Scalar(255));
//					break;
//				}
//				else
//				{
//					//移除该label,继续连接
//					cv::Mat region2Remove;
//					imgPro::region_toImg(labelImg, region2Remove, labelImg.at<int>(res.second.second));
//
//					nowImg = nowImg - region2Remove;
//					//将label中的label去掉
//					vector<Point> pt2Remmove;
//					imgPro::img_getPoints(region2Remove, pt2Remmove);
//					for (Point pt : pt2Remmove)
//					{
//						labelImg.at<int>(pt) = 0;
//					}
//				}
//			}
//		}
//
//		cv::Mat diff = (nowImg - preImg);
//		if (countNonZero(diff) == 0)  //不再增加连接区域则跳出
//		{
//			break;
//		}
//		else {
//			preImg = nowImg.clone();
//		}
//	}
//
//	cv::Mat resImg,maxThinImg, connImg2;
//	cv::Mat showImg;
////获取最终结果图像
//	imgPro::biImg_getMaxArea(nowImg, maxThinImg);
//	cv::Mat getLenImg;
//	skeleton_removeIntersectionLine(maxThinImg, getLenImg);
//	int scratchLen = cv::countNonZero(getLenImg);
//	//int scratchLen = max(rotROI.size.height,rotROI.size.width);
//	for (auto pairPt:vecPairPt)
//	{
//		cv::line(connImg, pairPt.first, pairPt.second, Scalar(255),1);
//	}
//
//	cv::Mat morLenImg;
//	cv::dilate(getLenImg, morLenImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));
//	cv::bitwise_and(connImg, morLenImg, connImg2);
//	imgPro::biImg_getMaxArea(connImg2, resImg);
//
//	cv::Mat scratchMaskTmp = cv::Mat::zeros(srcImg.size(), CV_8UC1);
//	resImg.copyTo(scratchMaskTmp(scratchRect));
//	imgPro::img_drawMask(srcImg, showImg, scratchMaskTmp, Scalar(0, 0, 255));
//
//	//取划痕内部高亮度 70% 的平均灰度
//	cv::Mat morScratchMask;
//
//	//cv::dilate(scratchMaskTmp, morScratchMask, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
//	double scratchMeanVal = imgPro::grayImg_getMean(srcImg, 0.3, 255, 0, scratchMaskTmp);
//
//	cv::Mat roundMask =  getRawScratchRound(scratchMaskTmp);
//	double groundMeanVal = imgPro::grayImg_getMean(srcImg, 1.0, 0, 255);
//
//	//imgPro::img_drawMask(showImg, showImg, roundMask, Scalar(0, 250, 255));
//	double meanDiff = scratchMeanVal - groundMeanVal;
//
//	vector<Point> rotPts;
//	float angleTmp;
//	string scratchInfo = to_string(scratchLen).substr(0, 5) + "_" + to_string(meanDiff).substr(0, 5);
//
//	imgPro::rect_getRotRectPts(rotROI, rotPts, angleTmp);
//	cv::Point showTxtPt = (rotPts[0] + rotPts[1]) / 2;
//	cv::putText(showImg, scratchInfo, showTxtPt, 1, 1, Scalar(0, 0, 255));
//	imgPro::img_drawRect(showImg, showImg, rotROI, Scalar(0, 255, 0));
//
//	cv::imshow("res", showImg);
//	cv::waitKey();
//
//	return 0;
//}
int scratch_connection(cv::Mat binImgIn, cv::Mat &connImg, vector<pair<cv::Point, cv::Point>> &vecPairPt);

//从最大区域开始连接
int procImg4(cv::Mat srcImg, cv::Rect scratchRect)
{
	cv::Mat enhanceImg, scratchHessianBin,cropImg, connImg;
	if (scratchRect.area() < 10)
	{
		cout << "矩形太小" << endl;
		return 1;
	}
	if (srcImg.channels() >1)
	{
		cv::cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
	}
	cv::Mat scratchRectImg, srcRectBinMean, rebuildImg;
	srcImg(scratchRect).copyTo(scratchRectImg);

	cv::Mat scratchRectImgInverse = 255 - scratchRectImg;
	hessianEnhance(scratchRectImgInverse, enhanceImg);
	cv::threshold(enhanceImg, scratchHessianBin, 8, 255, CV_8UC1);
	int c = 2;

	cv::adaptiveThreshold(scratchRectImg, srcRectBinMean, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C,
		cv::ThresholdTypes::THRESH_BINARY, 37, -c);

	cv::morphologyEx(srcRectBinMean, srcRectBinMean, cv::MorphTypes::MORPH_CLOSE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);

	//判断最大倾斜矩形的长边
	morRebuild(srcRectBinMean, scratchHessianBin, rebuildImg);

	cv::Mat maxRegion;
	cv::RotatedRect rotRect;
	imgPro::biImg_getMaxArea(rebuildImg, maxRegion);
	imgPro::biImg_getRotRect(maxRegion, rotRect);
	if (max(rotRect.size.width,rotRect.size.height)/double(max(rebuildImg.rows, rebuildImg.cols) < 0.3 ))
	{
		rebuildImg = srcRectBinMean;
	}

	biImg_connection(rebuildImg, connImg, 3);
	//connImg = srcRectBinMean;
 	cv::Mat filterImg, filterImgMor;

	imgPro::biImg_filterByArea(connImg, filterImg, 8, srcRectBinMean.total(), 2);

	//cv::morphologyEx(filterImg, filterImgMor, cv::MorphTypes::MORPH_CLOSE,
	//	cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)),cv::Point(-1,-1),2);
	imgPro::biImg_fillup(filterImg, filterImgMor, 2, 0, 20);
	cv::Mat binImg = thinImage(filterImgMor);
	cv::Mat filterBinImg;
	imgPro::biImg_filterByArea(binImg, filterBinImg, 2, filterImg.total(), 2);
	
	//提取最大划痕区域，去除分支
	cv::Mat resImg,maxThinImg, connImg2, nowImg;
	cv::Mat showImg;
	//获取最终结果图像
	vector<pair<cv::Point, cv::Point>> vecPairPt;
	scratch_connection(filterBinImg, nowImg,vecPairPt);

	imgPro::biImg_getMaxArea(nowImg, maxThinImg);
	cv::Mat getLenImg = skeleton_removeBranchs(maxThinImg,15,135 );
	//getLenImg = skeleton_removeBranchs(getLenImg,20,145 );
	vector<cv::Mat> scratchImgs = scratch_split(getLenImg, 145);

	int scratchLen = cv::countNonZero(getLenImg);
	//int scratchLen = max(rotROI.size.height,rotROI.size.width);
	for (auto pairPt:vecPairPt)
		cv::line(connImg, pairPt.first, pairPt.second, Scalar(255),1);

	
	cv::Mat scratchExtendImg = scratch_extend(getLenImg, srcRectBinMean, scratchRect);//端点进行弱划痕延伸，依据局部均值图
	//cv::Mat scratchExtendImg = getLenImg;

	cv::Mat morLenImg;
	cv::Mat scratchMaskTmp = cv::Mat::zeros(srcImg.size(), CV_8UC1);

	cv::bitwise_or(scratchExtendImg, scratchHessianBin, connImg2);
	imgPro::biImg_getMaxArea(connImg2, resImg);

	scratchExtendImg.copyTo(scratchMaskTmp(scratchRect));
	//resImg.copyTo(scratchMaskTmp(scratchRect));
	imgPro::img_drawMask(resShowImg, resShowImg, scratchMaskTmp, Scalar(0, 0, 255));

	//取划痕内部高亮度 70% 的平均灰度
	cv::Mat morScratchMask;

	//cv::dilate(scratchMaskTmp, morScratchMask, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	double scratchMeanVal = imgPro::grayImg_getMean(srcImg, 0.3, 255, 0, scratchMaskTmp);
	cv::Mat roundMask =  getRawScratchRound(scratchMaskTmp);
	double groundMeanVal = imgPro::grayImg_getMean(srcImg, 1.0, 0, 255);
	double meanDiff = scratchMeanVal - groundMeanVal;

	vector<Point> rotPts;
	float angleTmp;
	string scratchInfo = to_string(scratchLen).substr(0, 5) + "_" + to_string(meanDiff).substr(0, 5);

	cv::Point showTxtPt = scratchRect.tl();
	cv::putText(resShowImg, scratchInfo, showTxtPt, 1, 1, Scalar(0, 0, 255));
	cv::rectangle(resShowImg, scratchRect, cv::Scalar(0, 255, 0));
	cv::imshow("res", resShowImg);
	cv::waitKey();

	return 0;
}

//断开交点，全部进行连接
int procImg(cv::Mat srcImg, cv::Rect scratchRect)
{
	cv::Mat enhanceImg, scratchHessianBin,cropImg, connImg;
	if (scratchRect.area() < 10)
	{
		cout << "矩形太小" << endl;
		return 1;
	}
	if (srcImg.channels() >1)
	{
		cv::cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
	}
	cv::Mat scratchRectImg, srcRectBinMean, rebuildImg;
	srcImg(scratchRect).copyTo(scratchRectImg);

	cv::Mat scratchRectImgInverse = 255 - scratchRectImg;
	hessianEnhance(scratchRectImgInverse, enhanceImg);  // 功能： 增强线性特征
	//grayImg_hessian(scratchRectImgInverse, enhanceImg, 5, 1.0);
	cv::threshold(enhanceImg, scratchHessianBin, 8, 255, CV_8UC1);
	int c = 2;
	cv::adaptiveThreshold(scratchRectImg, srcRectBinMean, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_MEAN_C,
		cv::ThresholdTypes::THRESH_BINARY, 37, -c);
	cv::morphologyEx(srcRectBinMean, srcRectBinMean, cv::MorphTypes::MORPH_CLOSE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);
	// 形态学重建，以海森矩阵增强后的阈值图scratchHessianBin作为种子图像，
	// 在srcRectBinMean进行生长，得到确定的划痕图像，srcRectBinMean会受噪声干扰，无法直接提取划痕。
	morRebuild(srcRectBinMean, scratchHessianBin, rebuildImg);   // rebuildImg：大概率包含了划痕
	cv::Mat maxRegion;
	cv::RotatedRect rotRect;
	imgPro::biImg_getMaxArea(rebuildImg, maxRegion);
	imgPro::biImg_getRotRect(maxRegion, rotRect);
	if (max(rotRect.size.width,rotRect.size.height)/double(max(rebuildImg.rows, rebuildImg.cols) < 0.3 ))
	{
		rebuildImg = srcRectBinMean;
	}

	biImg_connection(rebuildImg, connImg, 3);  // 小间断（<3个像素）划痕合并
	//connImg = srcRectBinMean;
 	cv::Mat filterImg, filterBinImg,filterImgMor;
	imgPro::biImg_filterByArea(connImg, filterImg, 8, srcRectBinMean.total(), 2);  // 去除小面积噪声

	imgPro::biImg_fillup(filterImg, filterImgMor, 2, 0, 20);  // 划痕内部空洞填充，方便后续细化算法（长度计算）
	cv::Mat binImg = thinImage(filterImgMor);
	imgPro::biImg_filterByArea(binImg, filterBinImg, 2, filterImg.total(), 2);  // 
	
	cv::Mat resImg,maxThinImg, connImg2, nowImg;
	cv::Mat showImg, intersecBrokenImg, toConnImg;
	vector<pair<cv::Point, cv::Point>> vecPairPt;
	vector<cv::Point> vecIntersecPt;

	skeleton_intersecPoint(filterBinImg, vecIntersecPt);
	intersecBrokenImg = filterBinImg.clone();
	for (auto pt : vecIntersecPt)
	{
		intersecBrokenImg = set8Neib(intersecBrokenImg, pt); //将pt8邻域内置为0，查找该邻域内相对交点
	}
	imgPro::biImg_filterByArea(intersecBrokenImg, toConnImg, 3, filterBinImg.total(), 2);
	scratch_connection2(toConnImg,vecPairPt,15,30);
	//处理连接区域
	showImg = toConnImg.clone();
	cv::cvtColor(showImg, showImg, COLOR_GRAY2BGR);
		RNG rng(0xFFFFFFFF);
	for (auto it:vecPairPt)
	{
		line(showImg, it.first, it.second,imgPro::randomColor(rng) );
	}
	//去除同一端点连出的多条线段

	//提取几条长的划痕，
	 vector<cv::Mat> scratchImgs =  scratch_merge(toConnImg, vecPairPt);
	//再连接，输出划痕参数
	 int offsetY = 20;
	 for (int i=0;i < scratchImgs.size();i++ )
	 {
		cv::Mat scraImg = scratchImgs[i];
		cv::Mat toExtImg = skeleton_removeBranchs(scraImg, 15, 150);
		cv::Mat scratchExtendImg = scratch_extend(toExtImg, srcRectBinMean, scratchRect);//端点进行弱划痕延伸，依据局部均值图
		cv::Mat morLenImg;
		cv::Mat scratchMaskTmp = cv::Mat::zeros(srcImg.size(), CV_8UC1);

		cv::bitwise_or(scratchExtendImg, scratchHessianBin, connImg2);
		imgPro::biImg_getMaxArea(connImg2, resImg);
		cv::Mat thinImgTmp = thinImage(resImg);
		cv::Mat resThinImg = skeleton_removeBranchs(thinImgTmp, 15, 150);
		 float scratchLen = cv::countNonZero(resThinImg);

		//scratchExtendImg.copyTo(scratchMaskTmp(scratchRect));
		resImg.copyTo(scratchMaskTmp(scratchRect));
		cv::Scalar color = imgPro::randomColor(rng);
		imgPro::img_drawMask(resShowImg, resShowImg, scratchMaskTmp, color);

		//取划痕内部高亮度 70% 的平均灰度
		cv::Mat morScratchMask;

		//cv::dilate(scratchMaskTmp, morScratchMask, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
		double scratchMeanVal = imgPro::grayImg_getMean(srcImg, 0.3, 255, 0, scratchMaskTmp);
		cv::Mat roundMask = getRawScratchRound(scratchMaskTmp);
		double groundMeanVal = imgPro::grayImg_getMean(srcImg, 1.0, 0, 255);
		double meanDiff = scratchMeanVal - groundMeanVal;

		vector<Point> rotPts;
		float angleTmp;
		string scratchInfo = to_string(scratchLen).substr(0, 5) + "_" + to_string(meanDiff).substr(0, 5);

		cv::Point showTxtPt = cv::Point(scratchRect.tl().x ,scratchRect.tl().y+i*offsetY);
		cv::putText(resShowImg, scratchInfo, showTxtPt, 1, 1, color);
	 }
	
	cv::rectangle(resShowImg, scratchRect, cv::Scalar(0, 255, 0));
	cv::imshow("res", resShowImg);
	cv::waitKey();

	return 0;
}


int linePair_filter(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> vecPtPairIn, vector<pair<cv::Point, cv::Point>> vecPtPairOut)
{
	//找出所有出现2次的点

	//比较被连接线的长度
	return 0;
}

int region_connnetion(cv::Mat binImg, cv::Mat &connImg, cv::RotatedRect rotRect)
{
	//提取中心位置region
	cv::Mat labelImg;
	map<string, vector<double>> feat;
	connImg = binImg.clone();
	imgPro::biImg_createRegion(binImg, labelImg, feat, 0, binImg.total());
	int centerX = rotRect.center.x;
	int centerY = rotRect.center.y;
	cv::Point imgCent(centerX, centerY);
	vector<double> regionCenterRows = feat["row"];
	vector<double> regionCenterCols = feat["col"];
	vector<double> regionLabel = feat["label"];
	if (feat["label"].size() <2)
	{
		connImg = binImg.clone();
		return 0;
	}

	vector<double> dist2ImgCenter;
	for (int i = 0; i < regionCenterRows.size(); i++)
	{
		int x = regionCenterCols[i];
		int y = regionCenterRows[i];

		double dist = imgPro::dist2Pts(cv::Point(x, y), imgCent);

		dist2ImgCenter.push_back(dist);
	}

	auto smallest = std::min_element(std::begin(dist2ImgCenter), std::end(dist2ImgCenter));
	int minIndex = std::distance(std::begin(dist2ImgCenter), smallest);

	int centerRegionLabel = regionLabel[minIndex];
	//提取中心region所有点
	cv::Mat regionCentImg,morRegionCentImg;
	imgPro::region_toImg(labelImg, regionCentImg, centerRegionLabel);
	cv::dilate(regionCentImg, morRegionCentImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	vector<cv::Point> vecCentPts;
	binImg2vectorPt(morRegionCentImg, vecCentPts);


	cv::RotatedRect rotCentRect = cv::minAreaRect(vecCentPts);
	cv::Point t;
	cv::Mat showImg;
	imgPro::img_drawRect(binImg, showImg, t, rotRect.center, rotRect.size, -rotRect.angle, Scalar(0, 0, 255));
	imgPro::img_drawRect(showImg, showImg, t, rotCentRect.center, rotCentRect.size, -rotCentRect.angle, Scalar(0, 0, 255));

	//提取labelImg一半矩形区域，中心区域包含两次
	cv::RotatedRect halfRotRect1, halfRotRect2;
	getHalfRotRect(rotRect, rotCentRect, halfRotRect1, 0);
	getHalfRotRect(rotRect, rotCentRect, halfRotRect2, 1);	
	
	cv::Point ptTemp1, ptTemp2;
	double dist1, dist2;
	//	切分旋转矩形
	imgPro::img_drawRect(showImg, showImg, t, halfRotRect1.center, halfRotRect1.size, -halfRotRect1.angle, Scalar(0, 255, 0));
	imgPro::img_drawRect(showImg, showImg, t, halfRotRect2.center, halfRotRect2.size, -halfRotRect2.angle, Scalar(254, 0, 0));

	cv::Mat halfLabelImg1, halfLabelImg2;
	imgPro::img_getPartR(labelImg, halfLabelImg1, halfRotRect1);
	imgPro::img_getPartR(labelImg, halfLabelImg2, halfRotRect2);

	vector<pair<cv::Point, cv::Point> > vecPairPt,vecPairPt1, vecPairPt2;
	getConnPairPt(halfLabelImg1, centerRegionLabel, vecCentPts, vecPairPt1);
	getConnPairPt(halfLabelImg2, centerRegionLabel, vecCentPts, vecPairPt2);

	vecPairPt.insert(vecPairPt.end(), vecPairPt1.begin(), vecPairPt1.end());
	vecPairPt.insert(vecPairPt.end(), vecPairPt2.begin(), vecPairPt2.end());

	for (auto pts : vecPairPt)
	{
		line(connImg, pts.first, pts.second, Scalar(255), 1);
	}

	return 0;
}

int getConnPairPt(cv::Mat labelImg, int centerRegionLabel,vector<cv::Point> vecCentPts, vector<pair<cv::Point, cv::Point> > &vecPairPt)
{
	map<int, vector<cv::Point>> mapVecPts;
	vecPairPt.clear();

	labelImg2vectorPts(labelImg, mapVecPts, centerRegionLabel);

	//开始循环连接，获取pair<cv::Point,cv::Point> 连接对
	while (mapVecPts.size() > 0)
	{
		pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecCentPts, mapVecPts);
		vecPairPt.push_back(res.second);

		vecCentPts.swap(mapVecPts[res.first]);
		mapVecPts.erase(res.first);
	}

	return 0;
}



int binImg2vectorPt(cv::Mat binImg, vector<cv::Point>& vecPt)
{
	vecPt.clear();

	for (int r = 0; r < binImg.rows; r++)
	{
		for (int c = 0; c < binImg.cols; c++)
		{
			if (binImg.at<uchar>(r, c) > 0)
			{
				vecPt.push_back(cv::Point(c, r));
			}
		}
	}
	return 0;

}


