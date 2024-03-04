#include "extractScratch.h"

#include <math.h>

using namespace cv;
using namespace std ;

int hessianEnhance(cv::Mat srcImage, cv::Mat &imOut)
{
	if (srcImage.empty())
	{
		cout << "图像未被读入";
		return 1;
	}
	if (srcImage.channels() != 1)
	{
		cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
	}
	int width = srcImage.cols;
	int height = srcImage.rows;

	Mat outImage(height, width, CV_8UC1, Scalar::all(0));
	int W = 5;            // 海森矩阵窗口参数, 划痕粗细去调整
	float sigma = 1.;     // 根据图像本身的噪声大小，；亮度差来计算均值和方差， 方差* alpha = sigma
	Mat xxGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, Scalar::all(0));
	Mat xyGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, Scalar::all(0));
	Mat yyGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, Scalar::all(0));

	//构建高斯二阶偏导数模板
	for (int i = -W; i <= W; i++)
	{
		for (int j = -W; j <= W; j++)
		{
			xxGauKernel.at<float>(i + W, j + W) = (1 - (i*i) / (sigma*sigma))*exp(-1 * (i*i + j * j) / (2 * sigma*sigma))*(-1 / (2 * CV_PI*pow(sigma, 4)));
			yyGauKernel.at<float>(i + W, j + W) = (1 - (j*j) / (sigma*sigma))*exp(-1 * (i*i + j * j) / (2 * sigma*sigma))*(-1 / (2 * CV_PI*pow(sigma, 4)));
			xyGauKernel.at<float>(i + W, j + W) = ((i*j))*exp(-1 * (i*i + j * j) / (2 * sigma*sigma))*(1 / (2 * CV_PI*pow(sigma, 6)));
		}
	}
	//for (int i = 0; i < (2 * W + 1); i++)
	//{
	//	for (int j = 0; j < (2 * W + 1); j++)
	//	{
	//		cout << xxGauKernel.at<float>(i, j) << "  ";
	//	}
	//	cout << endl;
	//}
	Mat xxDerivae(height, width, CV_32FC1, Scalar::all(0));
	Mat yyDerivae(height, width, CV_32FC1, Scalar::all(0));
	Mat xyDerivae(height, width, CV_32FC1, Scalar::all(0));
	//图像与高斯二阶偏导数模板进行卷积
	filter2D(srcImage, xxDerivae, xxDerivae.depth(), xxGauKernel);
	filter2D(srcImage, yyDerivae, yyDerivae.depth(), yyGauKernel);
	filter2D(srcImage, xyDerivae, xyDerivae.depth(), xyGauKernel);

	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{

			//map<int, float> best_step;

		/*	int HLx = h - STEP; if (HLx < 0){ HLx = 0; }
			int HUx = h + STEP; if (HUx >= height){ HUx = height - 1; }
			int WLy = w - STEP; if (WLy < 0){ WLy = 0; }
			int WUy = w + STEP; if (WUy >= width){ WUy = width - 1; }
			float fxx = srcImage.at<uchar>(h, WUy) + srcImage.at<uchar>(h, WLy) - 2 * srcImage.at<uchar>(h, w);
			float fyy = srcImage.at<uchar>(HLx, w) + srcImage.at<uchar>(HUx, w) - 2 * srcImage.at<uchar>(h, w);
			float fxy = 0.25*(srcImage.at<uchar>(HUx, WUy) + srcImage.at<uchar>(HLx, WLy) - srcImage.at<uchar>(HUx, WLy) - srcImage.at<uchar>(HLx, WUy));*/


			float fxx = xxDerivae.at<float>(h, w);
			float fyy = yyDerivae.at<float>(h, w);
			float fxy = xyDerivae.at<float>(h, w);


			float myArray[2][2] = { { fxx, fxy }, { fxy, fyy } };          //构建矩阵，求取特征值

			Mat Array(2, 2, CV_32FC1, myArray);
			Mat eValue;
			Mat eVector;

			eigen(Array, eValue, eVector);                               //矩阵是降序排列的
			float a1 = eValue.at<float>(0, 0);
			float a2 = eValue.at<float>(1, 0);

			//if ((a1 > 0) && (abs(a1) > (1 + abs(a2))))             //根据特征向量判断线性结构 -根据特征值判断
			if ((a1 > 0) && (abs(a1) > (0.5 + abs(a2))))             //根据特征向量判断线性结构 -根据特征值判断
			{
				outImage.at<uchar>(h, w) = pow((abs(a1) - abs(a2)), 4);
				//outImage.at<uchar>(h, w) = pow((ABS(a1) / ABS(a2))*(ABS(a1) - ABS(a2)), 1.5);
			}
		}
	}

	//----------做一个闭操作
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	cv::morphologyEx(outImage, outImage, MORPH_CLOSE, element);

	imOut = outImage;
}


int grayImg_hessian(const cv::Mat& img_in, cv::Mat& img_out, int window, float sigma) {
	if (img_in.empty()) {
		std::cout << "输入图像异常" << std::endl;
		return 1;
	}
	if (img_in.channels() != 1)
		cvtColor(img_in, img_in, cv::COLOR_BGR2GRAY);

	cv::Mat invImg = 255 - img_in;
	int width = img_in.cols;
	int height = img_in.rows;

	cv::Mat outImage(height, width, CV_8UC1, cv::Scalar::all(0));
	int W = window;   // 
	//float sigma = 1.23;  
	cv::Mat xxGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat xyGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat yyGauKernel(2 * W + 1, 2 * W + 1, CV_32FC1, cv::Scalar::all(0));
	cv::Mat maxV = cv::Mat::zeros(img_in.size(), CV_32FC1);
	cv::Mat minV = cv::Mat::zeros(img_in.size(), CV_32FC1);
	cv::Mat feat = cv::Mat::zeros(img_in.size(), CV_32FC1);
	cv::Mat lineImg = cv::Mat::zeros(img_in.size(), CV_32FC1);
	cv::Mat eigenMat = cv::Mat::zeros(img_in.size(), CV_32FC2);

	//构建高斯二阶偏导数模板
	for (int i = -W; i <= W; i++) {
		for (int j = -W; j <= W; j++) {
			xxGauKernel.at<float>(i + W, j + W) = (1. - (i * i) / (sigma * sigma)) * expf(-1. * (i * i + j * j) / (2. * sigma * sigma)) * (-1. / (2. * CV_PI * powf(sigma, 4.)));
			yyGauKernel.at<float>(i + W, j + W) = (1. - (j * j) / (sigma * sigma)) * expf(-1. * (i * i + j * j) / (2. * sigma * sigma)) * (-1. / (2. * CV_PI * powf(sigma, 4.)));
			xyGauKernel.at<float>(i + W, j + W) = ((i * j)) * expf(-1. * (i * i + j * j) / (2. * sigma * sigma)) * (1. / (2. * CV_PI * powf(sigma, 6.)));
		}
	}

	cv::Mat xxDerivae(height, width, CV_32FC1, cv::Scalar::all(0));
	cv::Mat yyDerivae(height, width, CV_32FC1, cv::Scalar::all(0));
	cv::Mat xyDerivae(height, width, CV_32FC1, cv::Scalar::all(0));
	//图像与高斯二阶偏导数模板进行卷积
	cv::filter2D(img_in, xxDerivae, xxDerivae.depth(), xxGauKernel);
	cv::filter2D(img_in, yyDerivae, yyDerivae.depth(), yyGauKernel);
	cv::filter2D(img_in, xyDerivae, xyDerivae.depth(), xyGauKernel);

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			float fxx = xxDerivae.at<float>(h, w);
			float fyy = yyDerivae.at<float>(h, w);
			float fxy = xyDerivae.at<float>(h, w);

			float myArray[2][2] = { { fxx, fxy }, { fxy, fyy } };          //构建矩阵，求取特征值

			cv::Mat Array(2, 2, CV_32FC1, myArray);
			cv::Mat eValue;
			cv::Mat eVector;

			eigen(Array, eValue, eVector);                               //矩阵是降序排列的
			float a1 = eValue.at<float>(0, 0);  // 比较特征值大小，输出不同特征
			float a2 = eValue.at<float>(1, 0);
			maxV.at<float>(h, w) = a1;
		}
	}
	img_out = maxV;
	return 0;
}

#include <vector>  


cv::Mat getDoG(const cv::Mat &img)
{
	cv::Mat img_G0,img_G1;
	GaussianBlur(img, img_G0, Size(3, 3), 0);
	GaussianBlur(img_G0, img_G1, Size(3, 3), 0);
	Mat img_DoG = img_G0 - img_G1;
	normalize(img_DoG, img_DoG, 255, 0, NORM_MINMAX);

	return img_DoG;

}

/**
* @brief 对输入图像进行细化,骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，最终使用的元素中只有0与1,1代表有元素，0代表为空白
* @param maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
cv::Mat thinImage(const cv::Mat& src, const int maxIterations /*= -1*/)
{
	assert(src.type() == CV_8UC1);
	cv::Mat dst;
	cv::threshold(src, dst, 0, 1, cv::THRESH_BINARY);

	int width = src.cols;
	int height = src.rows;
	//src.copyTo(dst);
	int count = 0;  //记录迭代次数  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达  
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点  
		//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
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
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
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
			uchar * p = dst.ptr<uchar>(i);
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
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
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
	return dst*255;
}

/**
* @brief 对骨骼化图数据进行过滤，实现两个点之间至少隔一个空白像素
* @param thinSrc为输入的骨骼化图像,8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
*/
void filterOver(cv::Mat thinSrc)
{
	assert(thinSrc.type() == CV_8UC1);
	int width = thinSrc.cols;
	int height = thinSrc.rows;
	for (int i = 0; i < height; ++i)
	{
		uchar * p = thinSrc.ptr<uchar>(i);
		for (int j = 0; j < width; ++j)
		{
			// 实现两个点之间至少隔一个像素
			//  p9 p2 p3  
			//  p8 p1 p4  
			//  p7 p6 p5  
			uchar p1 = p[j];
			if (p1 != 1) continue;
			uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
			uchar p8 = (j == 0) ? 0 : *(p + j - 1);
			uchar p2 = (i == 0) ? 0 : *(p - thinSrc.step + j);
			uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - thinSrc.step + j + 1);
			uchar p9 = (i == 0 || j == 0) ? 0 : *(p - thinSrc.step + j - 1);
			uchar p6 = (i == height - 1) ? 0 : *(p + thinSrc.step + j);
			uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + thinSrc.step + j + 1);
			uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + thinSrc.step + j - 1);
			if (p2 + p3 + p8 + p9 >= 1)
			{
				p[j] = 0;
			}
		}
	}
}

/**
* @brief 从过滤后的骨骼化图像中寻找端点和交叉点
* @param thinSrc为输入的过滤后骨骼化图像,8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param raudis卷积半径，以当前像素点位圆心，在圆范围内判断点是否为端点或交叉点
* @param thresholdMax交叉点阈值，大于这个值为交叉点
* @param thresholdMin端点阈值，小于这个值为端点
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
std::vector<cv::Point> getPoints(const cv::Mat &thinSrc, unsigned int raudis /*= 4*/, unsigned int thresholdMax/* = 6*/, unsigned int thresholdMin/* = 4*/)
{
	assert(thinSrc.type() == CV_8UC1);
	int width = thinSrc.cols;
	int height = thinSrc.rows;
	cv::Mat tmp;
	thinSrc.copyTo(tmp);
	std::vector<cv::Point> points;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (*(tmp.data + tmp.step * i + j) == 0)
			{
				continue;
			}
			int count = 0;
			for (int k = i - raudis; k < i + raudis + 1; k++)
			{
				for (int l = j - raudis; l < j + raudis + 1; l++)
				{
					if (k < 0 || l < 0 || k>height - 1 || l>width - 1)
					{
						continue;

					}
					else if (*(tmp.data + tmp.step * k + l) == 1)
					{
						count++;
					}
				}
			}

			if (count > thresholdMax || count < thresholdMin)
			{
				Point point(j, i);
				points.push_back(point);
			}
		}
	}
	return points;
}

int skeleton_endpoint(const cv::Mat &skeletonImg, vector<cv::Point> &endPoints)
{
	cv::Mat zerOneImg, padImg;
	endPoints.clear();
	cv::copyMakeBorder(skeletonImg, padImg, 1, 1, 1, 1, CV_8UC1, Scalar(0));

	cv::threshold(padImg, zerOneImg, 0, 1, cv::THRESH_BINARY);
#pragma region kernerl
	//http ://www.imagemagick.org/Usage/morphology/#linejunctions

	cv::Mat kernel1 = (Mat_<int>(3, 3) << -1, -1, 0,
											-1, 1, 1,
											-1, -1, 0);

	cv::Mat kernel2 = (Mat_<int>(3, 3) << -1, -1, -1,
											-1, 1, -1,
											0, 1, 0);

	cv::Mat kernel3 = (Mat_<int>(3, 3) << 0, -1, -1,
											1, 1, -1,
											0, -1, -1);

	cv::Mat kernel4 = (Mat_<int>(3, 3) << 0, 1, 0,
											-1, 1, -1,
											-1, -1, -1);

	cv::Mat kernel5 = (Mat_<int>(3, 3) << -1, -1, -1,
											-1, 1, -1,
											-1, -1, 1);
													
	cv::Mat kernel6 = (Mat_<int>(3, 3) << -1, -1, -1,
											-1, 1, -1,
											1, -1, -1);
													
	cv::Mat kernel7 = (Mat_<int>(3, 3) << 1, -1, -1,
											-1, 1, -1,
											-1, -1, -1);
													
	cv::Mat kernel8 = (Mat_<int>(3, 3) << -1, -1, 1,
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

//
//int main(int argc, char*argv[])
//{
//	cv::Mat src;
//	//获取图像  
//	if (argc != 2)
//	{
//		src = cv::imread("src.jpg", cv::IMREAD_GRAYSCALE);
//	}
//	else
//	{
//		src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
//	}
//	if (src.empty())
//	{
//		std::cout << "读取文件失败！" << std::endl;
//		return -1;
//	}
//
//	//将原图像转换为二值图像  
//	cv::threshold(src, src, 128, 1, cv::THRESH_BINARY);
//	//图像细化，骨骼化  
//	cv::Mat dst = thinImage(src);
//	//过滤细化后的图像
//	filterOver(dst);
//	//查找端点和交叉点  
//	std::vector<cv::Point> points = getPoints(dst, 6, 9, 6);
//	//二值图转化成灰度图，并绘制找到的点
//	dst = dst * 255;
//	src = src * 255;
//	vector<cv::Point>::iterator it = points.begin();
//	for (; it != points.end(); it++)
//	{
//		circle(dst, *it, 4, 255, 1);
//	}
//	imwrite("dst.jpg", dst);
//	//显示图像  
//	cv::namedWindow("src1", CV_WINDOW_AUTOSIZE);
//	cv::namedWindow("dst1", CV_WINDOW_AUTOSIZE);
//	cv::imshow("src1", src);
//	cv::imshow("dst1", dst);
//	cv::waitKey(0);
//}

//skeletonImg :二值骨架图像，最好四连通  |废弃,只能获得单一邻域的点
//endPoints:所有端点值,

//int skeleton_endpoint2(cv::Mat skeletonImg, vector<cv::Point> &endPoints)
//{
//	cv::Mat zerOneImg;
//	endPoints.clear();
//	cv::threshold(skeletonImg, zerOneImg, 0, 1, cv::THRESH_BINARY);
//
//	cv::Mat kernel = (Mat_<uchar>(3,3)<<1,1,1,
//										1,100,1,
//										1,1,1);
//
//	cv::Mat endPointImg, padImg;
//	cv::copyMakeBorder(zerOneImg, padImg, 1, 1, 1, 1, CV_8UC1, Scalar(0));
//
//	cv::filter2D(padImg, endPointImg, CV_8UC1, kernel);
//	uchar endPointVal = kernel.at<uchar>(1, 1) + kernel.at<uchar>(1, 0);
//	//邻域数量为1的点
//	for (int r=0;r<endPointImg.rows;r++)
//	{
//		for (int c=0;c<endPointImg.cols;c++)
//		{
//			if (endPointImg.at<uchar>(r,c) == endPointVal)
//			{
//				endPoints.push_back(cv::Point(c-1,r-1));
//			}
//		}
//	}
//	return 0;
//}
//skeletonImg :二值骨架图像，最好四连通
//endPoints:所有端点值
//int skeleton_endpoint(cv::Mat skeletonImg, vector<cv::Point> &endPoints)
//{
//	cv::Mat zerOneImg, padImg;
//	cv::copyMakeBorder(skeletonImg, padImg, 1, 1, 1, 1, CV_8UC1, Scalar(0));
//
//	cv::threshold(padImg, zerOneImg, 0, 1, cv::THRESH_BINARY);
//#pragma region kernerl
//	//http ://www.imagemagick.org/Usage/morphology/#linejunctions
//
//	cv::Mat kernel1 = (Mat_<uchar>(3, 3) << -1, -1, 0,
//											-1, 1, 1,
//											-1, -1, 0);
//
//	cv::Mat kernel2 = (Mat_<uchar>(3, 3) << -1, -1, -1,
//											-1, 1, -1,
//											0, 1, 0);
//
//	cv::Mat kernel3 = (Mat_<uchar>(3, 3) << 0, -1, -1,
//											1, 1, -1,
//											0, -1, -1);
//
//	cv::Mat kernel4 = (Mat_<uchar>(3, 3) << 0, 1, 0,
//											-1, 1, -1,
//											-1, -1, -1);
//
//	cv::Mat kernel5 = (Mat_<uchar>(3, 3) << -1, -1,-1,
//											-1, 1,-1,
//											-1, -1,1);
//												
//	cv::Mat kernel6 = (Mat_<uchar>(3, 3) << -1, -1,-1,
//											-1, 1,-1,
//											1, -1,-1);
//												
//	cv::Mat kernel7 = (Mat_<uchar>(3, 3) << 1, -1,-1,
//											-1, 1,-1,
//											-1, -1,-1);
//												
//	cv::Mat kernel8 = (Mat_<uchar>(3, 3) << -1, -1,1,
//											-1, 1,-1,
//											-1, -1,-1);
//#pragma endregion
//	cv::Mat endImg, hitmisImg;
//	endImg.create(padImg.size(), CV_8UC1);
//	endImg.setTo(0);
//	vector<cv::Mat> kerners{ kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7,kernel8 };
//	for (cv::Mat ker : kerners)
//	{
//		cv::morphologyEx(zerOneImg, hitmisImg, cv::MorphTypes::MORPH_HITMISS, ker);
//
//		cv::bitwise_or(hitmisImg, endImg, endImg);
//	}
//
//	endImg = endImg * 255;
//
//	for (int r = 0; r < endImg.rows; r++)
//	{
//		for (int c = 0; c < endImg.cols; c++)
//		{
//			if (endImg.at<uchar>(r, c) > 0)
//			{
//				endPoints.push_back(cv::Point(c - 1, r - 1));
//			}
//		}
//	}
//
//	return 0;
//}
//

//获取同一骨架区域点对
int skeleton_endpoint(const cv::Mat &skeletonImg, vector<pair<cv::Point, cv::Point>> &endPoints)
{
	vector<cv::Point> vecPts;
	cv::Mat thinTmp = thinImage(skeletonImg);

	skeleton_endpoint(thinTmp, vecPts);
	cv::Mat labelImg;
	map<string, vector<double>> feats;
	imgPro::biImg_createRegion(skeletonImg, labelImg, feats, 0, skeletonImg.total());
	map<int, vector<cv::Point>> mapEndPts;


	for (auto pt : vecPts)
	{
		mapEndPts[labelImg.at<int>(pt)].push_back(pt);
	}
	for (auto ite : mapEndPts)
	{
		endPoints.push_back(make_pair(ite.second[0], ite.second[1]));
	}

	return 0;
}

int skeleton_intersecPoint(cv::Mat skeletonImg, vector<cv::Point> &intersecPts)
{
	cv::Mat zerOneImg, padImg;
	intersecPts.clear();
	cv::copyMakeBorder(skeletonImg, padImg, 1, 1, 1, 1, CV_8UC1, Scalar(0));

	cv::threshold(padImg, zerOneImg, 0, 1, cv::THRESH_BINARY);
#pragma region kernerl
//http ://www.imagemagick.org/Usage/morphology/#linejunctions

	cv::Mat kernel1 = (Mat_<uchar>(3, 3) << 1, 0, 1,
											0, 1, 0,
											0, 1, 0);

	cv::Mat kernel2 = (Mat_<uchar>(3, 3) << 0, 1, 0,
											0, 1, 1,
											1, 0, 0);	

	cv::Mat kernel3 = (Mat_<uchar>(3, 3) << 0, 0, 1,
											1, 1, 0,
											0, 0, 1);

	cv::Mat kernel4 = (Mat_<uchar>(3, 3) << 1, 0, 0,
											0, 1, 1,
											0, 1, 0);

	cv::Mat kernel5 = (Mat_<uchar>(3, 3) << 0, 1, 0,
											0, 1, 0,
											1, 0, 1);

	cv::Mat kernel6 = (Mat_<uchar>(3, 3) << 0, 0, 1,
											1, 1, 0,
											0, 1, 0);

	cv::Mat kernel7 = (Mat_<uchar>(3, 3) << 1, 0, 0,
											0, 1, 1,
											1, 0, 0);
	
	cv::Mat kernel8 = (Mat_<uchar>(3, 3) << 0, 1, 0,
											1, 1, 0,
											0, 0, 1);
	
	cv::Mat kernel9 = (Mat_<uchar>(3, 3) << 1, 0, 0,
											0, 1, 0,
											1, 0, 1);
	
	cv::Mat kernel10 = (Mat_<uchar>(3, 3) <<1, 0, 1,
											0, 1, 0,
											1, 0, 0);	
	
	cv::Mat kernel11 = (Mat_<uchar>(3, 3) <<1, 0, 1,
											0, 1, 0,
											0, 0, 1);
	
	cv::Mat kernel12 = (Mat_<uchar>(3, 3) <<0, 0, 1,
											0, 1, 0,
											1, 0, 1);
#pragma endregion
	cv::Mat intersecImg,hitmisImg;
	intersecImg.create(padImg.size(), CV_8UC1);
	intersecImg.setTo(0);
	vector<cv::Mat> kerners{ kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7,kernel8,kernel9,kernel10,kernel11,kernel12 };
	for (cv::Mat ker:kerners)
	{
		cv::morphologyEx(zerOneImg, hitmisImg, cv::MorphTypes::MORPH_HITMISS, ker);

		cv::bitwise_or(hitmisImg, intersecImg, intersecImg);
	}

	intersecImg = intersecImg * 255;
	
	for (int r = 0; r < intersecImg.rows; r++)
	{
		for (int c = 0; c < intersecImg.cols; c++)
		{
			if (intersecImg.at<uchar>(r, c) > 0)
			{
				intersecPts.push_back(cv::Point(c - 1, r - 1));
			}
		}
	}

	return 0;
}


pair<pair<cv::Point, cv::Point>, pair<cv::Point, cv::Point>> getRotRectPairPt(cv::RotatedRect rotRect, bool longEdge)
{
	vector<cv::Point> vecPt;
	float angle;
	imgPro::rect_getRotRectPts(rotRect,vecPt,angle);
	//左上点
	cv::Point ltPt = vecPt[0];
	cv::Point rtPt = vecPt[1];
	cv::Point rdPt = vecPt[2];
	cv::Point ldPt = vecPt[3];
	pair<cv::Point, cv::Point> pair1, pair2;
	if (longEdge)
	{
		double dist1 = imgPro::dist2Pts(ltPt, rtPt);
		double dist2 = imgPro::dist2Pts(ltPt, ldPt);
		if (dist1 > dist2)
		{
			 pair1 = make_pair(ltPt, rtPt);
			 pair2 = make_pair(ldPt, rdPt);
		}
		else
		{
			pair1 = make_pair(ltPt, ldPt);
			pair2 = make_pair(rtPt, rdPt);
		}

	}
	else
	{
		double dist1 = imgPro::dist2Pts(ltPt, rtPt);
		double dist2 = imgPro::dist2Pts(ltPt, ldPt);
		if (dist1 < dist2)
		{
			pair1 = make_pair(ltPt, rtPt);
			pair2 = make_pair(ldPt, rdPt);
		}
		else
		{
			pair1 = make_pair(ltPt, ldPt);
			pair2 = make_pair(rtPt, rdPt);
		}
	}
	return pair<pair<cv::Point, cv::Point>, pair<cv::Point, cv::Point>>(pair1,pair2);
}

int getHalfRotRect(cv::RotatedRect rotRect, cv::RotatedRect rotCentRect, cv::RotatedRect & halfRotRect1, int chooseID)
{
	vector<Point> centRegionPts, rotPts;
	float centRegionAngle, rotAngle;
	imgPro::rect_getRotRectPts(rotCentRect, centRegionPts, centRegionAngle);
	imgPro::rect_getRotRectPts(rotRect, rotPts, rotAngle);

	auto bigRotLongEdgePair = getRotRectPairPt(rotRect, true);
	auto bigRotShortEdgePair = getRotRectPairPt(rotRect, false);

	pair<cv::Point, cv::Point> longEdgePts1 = bigRotLongEdgePair.first;
	pair<cv::Point, cv::Point> longEdgePts2 = bigRotLongEdgePair.second;
	pair<cv::Point, cv::Point> shortEdgePts1 = bigRotShortEdgePair.first;
	pair<cv::Point, cv::Point> shortEdgePts2 = bigRotShortEdgePair.second;

	auto shortEdgePair = getRotRectPairPt(rotCentRect, false);
	cv::Point cropShortEdge1CentPt1;
	if (chooseID == 0)
	{
		pair<cv::Point, cv::Point> cropShortEdgePt1 = shortEdgePair.first;
		cropShortEdge1CentPt1 = (cropShortEdgePt1.first + cropShortEdgePt1.second) / 2;
	} 
	else
	{
		pair<cv::Point, cv::Point> cropShortEdgePt2 = shortEdgePair.second;
		cropShortEdge1CentPt1 = (cropShortEdgePt2.first + cropShortEdgePt2.second) / 2;
	}
	cv::Point interPt1, interPt2;
	double distTmp;
	imgPro::closestPt2Line(cropShortEdge1CentPt1, longEdgePts1.first, longEdgePts1.second, interPt1, distTmp);
	imgPro::closestPt2Line(cropShortEdge1CentPt1, longEdgePts2.first, longEdgePts2.second, interPt2, distTmp);
	cv::Point pt3, pt4;

	if (longEdgePts2.first == shortEdgePts1.first)
	{
		pt3 = shortEdgePts1.second;
	}
	else if (longEdgePts2.first == shortEdgePts1.second)
	{
		pt3 = shortEdgePts1.first;
	}
	else if (longEdgePts2.first == shortEdgePts2.first)
	{
		pt3 = shortEdgePts2.second;
	}
	else
	{
		pt3 = shortEdgePts2.first;
	}
	if (longEdgePts2.second == shortEdgePts1.first)
	{
		pt4 = shortEdgePts1.second;
	}
	else if (longEdgePts2.second == shortEdgePts1.second)
	{
		pt4 = shortEdgePts1.first;
	}
	else if (longEdgePts2.second == shortEdgePts2.first)
	{
		pt4 = shortEdgePts2.second;
	}
	else
	{
		pt4 = shortEdgePts2.first;
	}

	cv::RotatedRect halfTmp1 = cv::minAreaRect(vector<cv::Point>{interPt1, interPt2, longEdgePts2.first, pt3});
	cv::RotatedRect halfTmp2 = cv::minAreaRect(vector<cv::Point>{interPt1, interPt2, longEdgePts2.second, pt4});
	//cv::RotatedRect halfRotRect1, halfRotRect2;

	string srcPath = "E:\\5_Snapbox\\scratch_crop_hessian_5_bin_crop1_1.png";
	cv::Mat srcImg = imread(srcPath, 0);
	cv::Mat showThinImg;
	cv::Point t;
	imgPro::img_drawRect(srcImg, showThinImg, t, halfTmp1.center, halfTmp1.size, -halfTmp1.angle, Scalar(0, 0, 255));
	imgPro::img_drawRect(showThinImg, showThinImg, t, halfTmp2.center, halfTmp2.size, -halfTmp2.angle, Scalar(0, 0, 255));

	
	std::vector<cv::Point2f> rotRectInter1, rotRectInter2;
	cv::rotatedRectangleIntersection(halfTmp1, rotCentRect, rotRectInter1);
	cv::rotatedRectangleIntersection(halfTmp2, rotCentRect, rotRectInter2);

	double area1 = cv::contourArea(rotRectInter1);
	double area2 = cv::contourArea(rotRectInter2);

	if (area1 > area2)  //一半提取完成
		halfRotRect1 = halfTmp1;
	else
		halfRotRect1 = halfTmp2;

	return 0;
}

void morRebuild(cv::Mat src, cv::Mat toMor, cv::Mat & outPut,int cnt)
{
	//outPut.create(toMor.size(), toMor.type());
	//outPut.setTo(0);
	cv::Mat morImg,preImg,nowImg,orImg;
	preImg = toMor.clone();
	nowImg = preImg.clone();
	int cntTmp = 0;
	do
	{
		cv::dilate(preImg, morImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
		cv::bitwise_and(src, morImg, nowImg);
		cv::bitwise_xor(preImg,nowImg,orImg);
		if (cv::countNonZero(orImg) !=0)
		{
			 preImg = nowImg.clone();
		} 
		else
		{
			outPut = nowImg;
			break;
		}
		if (cnt > 0)
		{
			cntTmp++;
			if (cntTmp >= cnt)
			{
				outPut = nowImg;
				break;
			}

		}
	} while (true);


}

vector<cv::Point> uniqueVecPts(vector<cv::Point> vecPts)
{
	vector<cv::Point> vecPtTmp;

	for (auto pt :vecPts)
	{
		if (ptInVec(pt,vecPtTmp))
		{
			continue;
		} 
		else
		{
			vecPtTmp.push_back(pt);
		}
	}
	return vecPtTmp;
}

bool ptInVec(cv::Point pt,vector<cv::Point> vecPts)
{
	if (vecPts.empty())
	{
		return false;
	} 
	else
	{
		for (auto p:vecPts)
		{
			if (pt == p)
			{
				return true;
			}
		}
	}
	return false;
}

cv::Mat removeIntersection(cv::Mat srcBin,double  &scratchLen)
{
	cv::Mat zerOneImg;
	cv::Mat output = srcBin.clone();
	cv::threshold(srcBin, zerOneImg, 10, 1, cv::THRESH_BINARY);
	cv::Mat thinImg = thinImage(zerOneImg);

	thinImg = thinImg * 255;

	vector<Point> vecEndPt,vecIntersecPt;

	skeleton_endpoint(thinImg,vecEndPt);
	skeleton_intersecPoint(thinImg, vecIntersecPt);
	vector<pair<cv::Point, cv::Point>> linePair, linePair2Remove;
	if (vecIntersecPt.size() >0)
	{
		for (auto p:vecEndPt)
		{
			cv::Point pt2 = searchPt(thinImg, p, vecIntersecPt);
				linePair.push_back(make_pair(p, pt2));
		}
	}
	
	//取出端点到交点的线段,计算距离和角度
	vector<pair<double, double>> vecDistAngle;
	for (auto ptPair:linePair)
	{
		int dist = imgPro::dist2Pts(ptPair.first, ptPair.second);
		double angle = anglePostive(ptPair.first, ptPair.second);
		vecDistAngle.push_back(make_pair(dist,angle));
	}
	//计算需要删除的线，角度阈值35  to do
	double angleThresh = 35;
	auto maxEle = std::max_element(vecDistAngle.begin(), vecDistAngle.end(), [](auto item1, auto item2) {return item1.first < item2.first; });
	
	int maxIndex = std::distance(std::begin(vecDistAngle), maxEle);

	for (int i=0;i<vecDistAngle.size();i++)
	{
		if (abs(vecDistAngle[i].second - maxEle->second) > 35)
			linePair2Remove.push_back(linePair[i]);
	}

	//绘制要删除的线和剩余的骨架
	cv::Mat skeletonNoIntersecImg = thinImg.clone();
	vector<cv::Mat> vecSkeleton2RemoveImg;
	for (auto item:linePair2Remove)
	{
		cv::Mat skeleton1 =  getSkeletonFromPt(item.first,item.second,thinImg);
		vecSkeleton2RemoveImg.push_back(skeleton1);
		cv::bitwise_xor(skeleton1, skeletonNoIntersecImg, skeletonNoIntersecImg);
	}
	//再次连接骨架
	cv::Mat scratchSkeletonImg = connectSkeletonImg(skeletonNoIntersecImg);
	scratchLen = cv::countNonZero(scratchSkeletonImg);
	//此处可输出长度和对比度

	//以要删除的线开始膨胀删除，但不能越过骨架
	cv::bitwise_or(srcBin, scratchSkeletonImg, srcBin);
	cv::Mat preImg = srcBin.clone();
	cv::Mat nowImg = preImg.clone();
	for (int i=0;i<vecSkeleton2RemoveImg.size();i++)
	{
		cv::Mat removeImg = vecSkeleton2RemoveImg[i];

		cv::Mat intersecRegion;
		cv::bitwise_and(removeImg, scratchSkeletonImg, intersecRegion);
		removeImg = removeImg - intersecRegion;
		cv::Mat morImg = removeImg.clone();

		while (true)
		{
			cv::morphologyEx(morImg, morImg, cv::MorphTypes::MORPH_DILATE,
				cv::getStructuringElement(cv::MorphShapes::MORPH_CROSS, cv::Size(3, 3)));
			cv::bitwise_and(morImg, scratchSkeletonImg, intersecRegion);
			morImg = morImg - intersecRegion;

			nowImg = preImg - morImg;

			cv::Mat orImg;
			cv::bitwise_xor(preImg, nowImg, orImg);
			if (cv::countNonZero(orImg) != 0)
			{
				preImg = nowImg.clone();
			} 
			else
			{
				break;
			}
		} 
	}

	return nowImg;

}
//由某点开始遍历图像，遇到vec中任一点停止，然后输出该点
//骨架图像，用形态学重建的方法查找,255,0
cv::Point searchPt(cv::Mat binSrc,cv::Point startPt, vector<cv::Point> pts)
{
	cv::Mat startImg,morImg;
	startImg.create(binSrc.size(), binSrc.type());
	startImg.setTo(0);
	startImg.at<uchar>(startPt) = 255;

	while (cv::countNonZero(binSrc - startImg) != 0)
	{
		cv::dilate(startImg, morImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
		cv::bitwise_and(binSrc, morImg, startImg);
		for (auto p:pts)
		{
			if (startImg.at<uchar>(p) != 0)
			{
				return p;
			}
		}
	}
	return cv::Point();
}

//返回找到该点的图像
cv::Mat searchPtImg(cv::Mat binSrc, cv::Point startPt, vector<cv::Point> pts)
{
	cv::Mat startImg, morImg;
	startImg.create(binSrc.size(), binSrc.type());
	startImg.setTo(0);
	startImg.at<uchar>(startPt) = 255;

	while (cv::countNonZero(binSrc - startImg) != 0)
	{
		cv::dilate(startImg, morImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
		cv::bitwise_and(binSrc, morImg, startImg);
		for (auto p : pts)
		{
			if (startImg.at<uchar>(p) != 0)
			{
				return startImg;
			}
		}
	}
	return cv::Mat();
}

//返回直线在图像边界交点，第一个点为距离指定点最近的点
pair<Point, Point> getLinePtInImg(cv::Mat imgIn, Vec4f linePara, cv::Point newStartPt/*=Point(-1,-1)*/)
{
	cv::Point point0;
	if (newStartPt.x == -1)
	{
		point0.x = linePara[2];
		point0.y = linePara[3];
	}
	else
	{
		point0.x = newStartPt.x;
		point0.y = newStartPt.y;

	}
	double k = linePara[1] / linePara[0];

	//找出直线在图像内的端点
	vector<cv::Point > vecPts;
	cv::Point point1, point2, point3, point4;
	// cv::Point startPt(-1, -1), endPt(-1, -1);
	int xEdge = imgIn.cols - 1;
	int yEdge = imgIn.rows - 1;
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
		return pair<cv::Point,cv::Point>();
	}
	cv::Point  startPt = vecPtsFilter[0];
	cv::Point  endPt = vecPtsFilter[1];


	if(imgPro::dist2Pts(startPt,point0) < imgPro::dist2Pts(endPt,point0))
		return pair<cv::Point, cv::Point>(point0,startPt);
	else
		return pair<cv::Point, cv::Point>(point0,endPt);

}

//暂没有全部删除交叉点，废弃
int skeleton_removeIntersectionLine(cv::Mat skeleImgIn, cv::Mat &binOut)
{
	cv::Mat zerOneImg;
	cv::Mat binIn = skeleImgIn.clone();
	cv::threshold(binIn, zerOneImg, 10, 1, cv::THRESH_BINARY);
	cv::Mat thinImg = thinImage(zerOneImg);

	thinImg = thinImg * 255;

	vector<Point> vecEndPt, vecIntersecPt;

	skeleton_endpoint(thinImg, vecEndPt);
	cv::Mat endShowImg = skeleImgIn.clone();
	cv::cvtColor(endShowImg, endShowImg, COLOR_GRAY2BGR);
	for (auto pt:vecEndPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0,255, 0);
	skeleton_intersecPoint(thinImg, vecIntersecPt);
	for (auto pt : vecIntersecPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0, 0, 255);

	vector<pair<cv::Point, cv::Point>> linePairRemove1, linePairRemoveTmp, linePairRemove2;
	vector<cv::Point> vecIntersecInLines;
	if (vecIntersecPt.size()==0)
	{
		binOut = skeleImgIn.clone();
		return 0;
	} 
	else if (vecIntersecPt.size() == 1)  
	{
		cout << "交点为1个" << endl;
		return 0;
	}

	else 
	{
		for (auto p : vecEndPt)
		{
			cv::Point pt2 = searchPt(thinImg, p, vecIntersecPt);
			linePairRemove1.push_back(make_pair(p, pt2));
			vecIntersecInLines.push_back(pt2);
		}
	}

	//1.删除没有共享交点的两个线段，
	for (auto item:linePairRemove1)
	{
		int cnt = std::count_if(vecIntersecInLines.begin(), vecIntersecInLines.end(), 
			[item](auto pt) {return pt.x == item.second.x && pt.y == item.second.y; });
		if (cnt == 1)
		{
			linePairRemove1.push_back(item);
		}
		else if(cnt <3)
		{
			linePairRemoveTmp.push_back(item);
		}
		else
		{
			cout << "交点为3或以上，特殊处理" << endl;
		}
	}
	cv::Mat noBranchImg1 = binIn.clone();
	if (linePairRemove1.size()>0)
	{
		cv::Mat noBranchImg1Broken = skeleton_removeBranch(binIn, linePairRemove1);
		noBranchImg1 = connectSkeletonImg(noBranchImg1Broken);	//再次连接骨架
	}

	//3.计算共享交点与两端endPoint的夹角，去除夹角较小的
	//共享交点向反方向遍历8个像素，先删除所有剩余线段，但保留交点，从每个交点遍历获得像素

	cv::Mat noBranchImg2 = skeleton_removeBranch(noBranchImg1, linePairRemoveTmp);
	for (auto ptPair:linePairRemoveTmp)
	{
		noBranchImg2.at<uchar>(ptPair.second) = 255;
	}
	vector<pair<cv::Point, cv::Point>> vecPtPairOnMainBranch;//first point is start point
	vector<Point> vecMainBrachEnd;
	skeleton_endpoint(noBranchImg2, vecMainBrachEnd);
	cv::Point endPtMain;
	for (auto pt:vecMainBrachEnd)
	{
		double angle = angleFromEndPt(noBranchImg2, pt, endPtMain, 8);
		/*vecPtPairOnMainBranch.push_back(make_pair(pt,endPtMain));*/
		//找到与pt距离最小的两个点，作为判断线段
		vector<pair<int,double>> vecIndDist;
		for (int pi=0;pi<linePairRemoveTmp.size();pi++)
		{
			vecIndDist.push_back(make_pair(pi, imgPro::dist2Pts(pt, linePairRemoveTmp[pi].second)));
		}
		std::sort(vecIndDist.begin(), vecIndDist.end(), [](auto it1, auto it2) {return it1.second < it2.second; });
		pair<Point, Point> lineIntersec1 =  linePairRemoveTmp[vecIndDist[0].first];
		pair<Point, Point> lineIntersec2 = linePairRemoveTmp[vecIndDist[1].first];

		double angle1,angle2;
		imgPro::angle2Lines(lineIntersec1.second,lineIntersec1.first,lineIntersec1.second, endPtMain,angle1);
		imgPro::angle2Lines(lineIntersec2.second,lineIntersec2.first,lineIntersec2.second, endPtMain,angle2);
		if (angle1 > angle2)
		{
			linePairRemove2.push_back(lineIntersec2);
		} 
		else
		{
			linePairRemove2.push_back(lineIntersec1);
		}

	}

	cv::Mat resImgTmp = skeleton_removeBranch(noBranchImg2, linePairRemove2);
	cv::Mat resImg = connectSkeletonImg(resImgTmp);
	binOut = resImg;
	return 0;

}


//删除小于一定距离的分支,但线条末端的分支按倾斜角度保留或全部删除，，//有缺陷，主体较长时适用
cv::Mat  skeleton_removeBranchs(cv::Mat skeleImgIn, double distThresh,double endAngleThresh)
{
	cv::Mat zerOneImg;
	vector<Point> vecEndPt, vecIntersecPt;
	cv::Mat binIn = skeleImgIn.clone();
	skeleton_endpoint(binIn, vecEndPt);
	cv::Mat endShowImg = skeleImgIn.clone();
	cv::cvtColor(endShowImg, endShowImg, COLOR_GRAY2BGR);

	for (auto pt : vecEndPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
	skeleton_intersecPoint(binIn, vecIntersecPt);
	for (auto pt : vecIntersecPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0, 0, 255);

	vector<pair<cv::Point, cv::Point>> linePairRemove1, linePairRemoveTmp, linePairRemove2;
	vector<pair<cv::Point, cv::Point>> linePair, linePair2Remove;
	if (vecIntersecPt.size() > 0)
	{
		for (auto p : vecEndPt)
		{
			cv::Point pt2 = searchPt(binIn, p, vecIntersecPt);
			linePair.push_back(make_pair(p, pt2));
		}
	}

	//取出端点到交点的线段,计算距离和角度
	vector<pair<double, double>> vecDistAngle;
	for (auto ptPair : linePair)
	{
		double dist = imgPro::dist2Pts(ptPair.first, ptPair.second);
		double angle = anglePostive(ptPair.first, ptPair.second);
		vecDistAngle.push_back(make_pair(dist, angle));
		if (dist < distThresh)
			linePairRemove1.push_back(ptPair);
	}

	cv::Mat skeletonBroken1 =  skeleton_removeBranch(skeleImgIn, linePairRemove1);
	//cv::Mat skeletonMinus =  connectSkeletonImg(skeletonBroken1);
	cv::Mat skeletonMinus;
	biImg_connection(skeletonBroken1, skeletonMinus, 10);

	vector<cv::Point> vecSkeletonMinusEndPts;
	skeleton_endpoint(skeletonMinus, vecSkeletonMinusEndPts);

	//如果末端同时删除两个线段，保留角度最接近的
	cv::Point endInLine;
	map<int, vector<pair<Point, Point>>> mapPtIndexAndLines;//端点和待连接的
	for (int ptCnt=0;ptCnt<vecSkeletonMinusEndPts.size();ptCnt++ )
	{
		for (int i=0;i< vecIntersecPt.size(); i++)  //所有交点坐标
		{
			double dist = imgPro::dist2Pts(vecSkeletonMinusEndPts[ptCnt], vecIntersecPt[i]);
			if (dist < 2.5)//交点位置
			{
				//找到secPt在的线段和角度
				for (int j=0;j<linePair.size();j++)
				{
					if (vecIntersecPt[i].x == linePair[j].second.x && vecIntersecPt[i].y == linePair[j].second.y)
					{
						mapPtIndexAndLines[ptCnt].push_back(linePair[j]);
					}
				}
			}
		}
	}

	vector<pair<Point, Point>> vecLineRepair;
	cv::Point endTmp;
	for (auto item:mapPtIndexAndLines)
	{
		cv::Point endPt = vecSkeletonMinusEndPts[item.first];
		vector<pair<cv::Point, cv::Point>> lines = item.second;  // first是端点
		vector<double> angles;
		double angle2Lines;
		double angleFromLine = angleFromEndPt(skeletonMinus, endPt, endTmp);;
		for (auto ite:lines)
		{
			//double angleTmp =   anglePostive(ite.first, ite.second);
			imgPro::angle2Lines(endPt, endTmp, endPt, ite.first,angle2Lines);
			angles.push_back(angle2Lines);
		}
		auto ite = max_element(angles.begin(), angles.end());
		if (*ite > endAngleThresh)
		{
			int index = std::distance(angles.begin(), ite);
			vecLineRepair.push_back(lines[index]);
		}
	}
	for (auto item:vecLineRepair)
	{
		cv::line(skeletonMinus, item.first, item.second, Scalar(255));
	}
	cv::Mat skeletonMinusCon;
	biImg_connection(skeletonMinus, skeletonMinusCon, 10);

	return skeletonMinusCon;
}
int skeleton_intersecPtCnt(cv::Mat skeleImgIn, double distThresh)
{
	vector<cv::Point> vecIntesecPt;
	skeleton_intersecPoint(skeleImgIn, vecIntesecPt);
	int intersecCnt = vecIntesecPt.size();
	//去除相近交点
	for (int c1 = 0; c1 < vecIntesecPt.size(); c1++)
	{
		for (int c2 = c1 + 1; c2 < vecIntesecPt.size(); c2++)
		{
			if (imgPro::dist2Pts(vecIntesecPt[c1], vecIntesecPt[c2]) < 2)
				intersecCnt--;
		}
	}

	return 0;
}


//删除骨架图分支，适应所有情况 todo
cv::Mat  skeleton_removeBranchsPlus(cv::Mat skeleImgIn, double distThresh,double endAngleThresh)
{
	cv::Mat zerOneImg;
	vector<Point> vecEndPt, vecIntersecPt;
	cv::Mat binIn = skeleImgIn.clone();
	skeleton_endpoint(binIn, vecEndPt);
	cv::Mat endShowImg = skeleImgIn.clone();
	cv::cvtColor(endShowImg, endShowImg, COLOR_GRAY2BGR);

	for (auto pt : vecEndPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
	skeleton_intersecPoint(binIn, vecIntersecPt);
	for (auto pt : vecIntersecPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0, 0, 255);

	vector<pair<cv::Point, cv::Point>> linePairRemove1, linePairRemoveTmp, linePairRemove2;
	vector<pair<cv::Point, cv::Point>> linePair, linePair2Remove;
	if (vecIntersecPt.size() > 0)
	{
		for (auto p : vecEndPt)
		{
			cv::Point pt2 = searchPt(binIn, p, vecIntersecPt);
			linePair.push_back(make_pair(p, pt2));
		}
	}

	//取出端点到交点的线段,计算距离和角度
	vector<pair<double, double>> vecDistAngle;
	for (auto ptPair : linePair)
	{
		double dist = imgPro::dist2Pts(ptPair.first, ptPair.second);
		double angle = anglePostive(ptPair.first, ptPair.second);
		vecDistAngle.push_back(make_pair(dist, angle));
		if (dist < distThresh)
			linePairRemove1.push_back(ptPair);
	}

	cv::Mat skeletonBroken1 = skeleton_removeBranch(skeleImgIn, linePairRemove1);
	//cv::Mat skeletonMinus =  connectSkeletonImg(skeletonBroken1);
	cv::Mat skeletonMinus;
	biImg_connection(skeletonBroken1, skeletonMinus, 10);

	vector<cv::Point> vecSkeletonMinusEndPts;
	skeleton_endpoint(skeletonMinus, vecSkeletonMinusEndPts);

	//如果末端同时删除两个线段，保留角度最接近的
	cv::Point endInLine;
	map<int, vector<pair<Point, Point>>> mapPtIndexAndLines;//端点和待连接的
	for (int ptCnt = 0; ptCnt < vecSkeletonMinusEndPts.size(); ptCnt++)
	{
		for (int i = 0; i < vecIntersecPt.size(); i++)  //所有交点坐标
		{
			double dist = imgPro::dist2Pts(vecSkeletonMinusEndPts[ptCnt], vecIntersecPt[i]);
			if (dist < 2.5)//交点位置
			{
				//找到secPt在的线段和角度
				for (int j = 0; j < linePair.size(); j++)
				{
					if (vecIntersecPt[i].x == linePair[j].second.x && vecIntersecPt[i].y == linePair[j].second.y)
					{
						mapPtIndexAndLines[ptCnt].push_back(linePair[j]);
					}
				}
			}
		}
	}

	vector<pair<Point, Point>> vecLineRepair;
	cv::Point endTmp;
	for (auto item : mapPtIndexAndLines)
	{
		cv::Point endPt = vecSkeletonMinusEndPts[item.first];
		vector<pair<cv::Point, cv::Point>> lines = item.second;  // first是端点
		vector<double> angles;
		double angle2Lines;
		double angleFromLine = angleFromEndPt(skeletonMinus, endPt, endTmp);;
		for (auto ite : lines)
		{
			//double angleTmp =   anglePostive(ite.first, ite.second);
			imgPro::angle2Lines(endPt, endTmp, endPt, ite.first, angle2Lines);
			angles.push_back(angle2Lines);
		}
		auto ite = max_element(angles.begin(), angles.end());
		if (*ite > endAngleThresh)
		{
			int index = std::distance(angles.begin(), ite);
			vecLineRepair.push_back(lines[index]);
		}
	}
	for (auto item : vecLineRepair)
	{
		cv::line(skeletonMinus, item.first, item.second, Scalar(255));
	}
	cv::Mat skeletonMinusCon;
	biImg_connection(skeletonMinus, skeletonMinusCon, 10);

	return skeletonMinusCon;



	return Mat();
}

//功能：删除给定的分叉，
//branchPts:分叉两个端点，其中一个是与主线的点
cv::Mat skeleton_removeBranch(cv::Mat skeletonImgIn, vector<pair<cv::Point, cv::Point>> branchPts)
{
	cv::Mat skeletonNoIntersecImg = skeletonImgIn.clone();
	vector<cv::Mat> branchImgs;
	for (auto item : branchPts)
	{
		cv::Mat skeleton1 = getSkeletonFromPt(item.first, item.second, skeletonImgIn);
		branchImgs.push_back(skeleton1);
	}
	for (auto item:branchImgs)
	{
		skeletonNoIntersecImg = skeletonNoIntersecImg - item;
	}
	return skeletonNoIntersecImg;
}

////angle统一到[0,180]
double angle2PtNorm(cv::Point lineStartPt, cv::Point lineEndPt)
{
	double angle;
	imgPro::angle2Pts(lineStartPt, lineEndPt, angle);
	
	double lineAngle = 0.0;
	if (angle < 0)
	{
		angle += 180;
	}
	//angle < 90 ? lineAngle = angle : lineAngle -= 90;

	return angle;
}

double angle(Point pt1, Point pt2, Point ptStart)
{
	double dx1 = pt1.x - ptStart.x;
	double dy1 = pt1.y - ptStart.y;
	double dx2 = pt2.x - ptStart.x;
	double dy2 = pt2.y - ptStart.y;
	double angle_line = (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
	return acos(angle_line) * 180 / 3.141592653;
}

double anglePostive(cv::Point pt1, cv::Point pt2)
{
	cv::Point ptStart, ptEnd;
	if (abs(pt1.y-pt2.y)<0.0001)   //水平
	{
		if (pt1.x > pt2.x)
		{
			ptStart = pt2;
			ptEnd = pt1;

		} 
		else
		{
			ptStart = pt1;
			ptEnd = pt2;

		}
	}
	else if (pt2.y > pt1.y)
	{
		ptStart = pt2;
		ptEnd = pt1;
	}
	else if (pt1.y > pt2.y)
	{
		ptStart = pt1;
		ptEnd = pt2;

	}

	cv::Point ptEnd2(ptStart.x + 10, ptStart.y);
	return angle(ptEnd2, ptEnd, ptStart);
}

//输入端点，交点
cv::Mat getSkeletonFromPt(cv::Point endPt, cv::Point intersecPt, cv::Mat skeletonImg)
{
	cv::Mat startImg, morImg,preImg, diffImg;
	startImg.create(skeletonImg.size(), skeletonImg.type());
	startImg.setTo(0);
	startImg.at<uchar>(endPt) = 255;

	while (cv::countNonZero(skeletonImg - startImg) != 0)
	{
		preImg = startImg.clone();
		cv::dilate(startImg, morImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
		cv::bitwise_and(skeletonImg, morImg, startImg);
		cv::bitwise_xor(preImg, startImg, diffImg);
		if (startImg.at<uchar>(intersecPt) != 0  || cv::countNonZero(diffImg) == 0)  //达到交点或startImg不再变化
		{
			return startImg;
		}		
	}
}
//输入：skeletonLabelImg:骨架图像
cv::Mat connectSkeletonImg(cv::Mat skeletonImg,double distThresh/*=4*/)
{
	vector<pair<cv::Point,cv::Point>> vecPairSkeletonEndPt,connPtPair;
	cv::Mat thinImgTmp = thinImage(skeletonImg);

	skeleton_endpoint(thinImgTmp, vecPairSkeletonEndPt);
	cv::Mat output = thinImgTmp.clone();
	int gapCnt = vecPairSkeletonEndPt.size() - 1;

	vector < pair<pair<cv::Point, cv::Point>, double>> vecPairPtDist;

	for (int i=0;i<vecPairSkeletonEndPt.size()-1;i++)  //遍历所有点对
	{
		cv::Point startPt1 = vecPairSkeletonEndPt[i].first;
		cv::Point startPt2 = vecPairSkeletonEndPt[i].second;
		for (int j = i+1; j < vecPairSkeletonEndPt.size(); j++)
		{
			cv::Point endPt1 = vecPairSkeletonEndPt[j].first;
			cv::Point endPt2 = vecPairSkeletonEndPt[j].second;
			
			vecPairPtDist.push_back(make_pair(make_pair(startPt1, endPt1), imgPro::dist2Pts(startPt1, endPt1)));
			vecPairPtDist.push_back(make_pair(make_pair(startPt1, endPt2), imgPro::dist2Pts(startPt1, endPt2)));
			vecPairPtDist.push_back(make_pair(make_pair(startPt2, endPt1), imgPro::dist2Pts(startPt2, endPt1)));
			vecPairPtDist.push_back(make_pair(make_pair(startPt2, endPt2), imgPro::dist2Pts(startPt2, endPt2)));
		}
		if (vecPairPtDist.size()>0)
		{
			auto minEle = min_element(vecPairPtDist.begin(), vecPairPtDist.end(), [](auto item1, auto item2) {return item1.second < item2.second; });
			if (minEle->second < distThresh)
			{
				connPtPair.push_back(minEle->first);
			}
			vecPairPtDist.clear();
		}
	}

	for (auto item:connPtPair)
	{
		cv::line(output, item.first, item.second, cv::Scalar(255), 1);
	}
	return output;
}

//获得划痕周边区域，用于计算灰度
cv::Mat getScratchAround(cv::Mat srcSratch, cv::Mat roundScratch)
{
	cv::Mat regionAroundSrcScratch, regionAroundSRoundScratch, mor5, mor7;
	int nearDist1 = 2;
	int nearDist2 = 3;

	morphologyEx(srcSratch, mor5, cv::MorphTypes::MORPH_DILATE, 
				cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(nearDist1*2+1, nearDist1*2+1)));
	morphologyEx(srcSratch, mor7, cv::MorphTypes::MORPH_DILATE,
				cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(nearDist2 * 2 + 1, nearDist2 * 2 + 1)));
	regionAroundSrcScratch = mor7 - mor5;

	morphologyEx(roundScratch, mor5, cv::MorphTypes::MORPH_DILATE, 
				cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(nearDist1 * 2 + 1, nearDist1 * 2 + 1)));
	morphologyEx(roundScratch, mor7, cv::MorphTypes::MORPH_DILATE, 
				cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(nearDist2 * 2 + 1, nearDist2* 2 + 1)));
	regionAroundSRoundScratch = mor7 - mor5;

	cv::Mat regionAnd;
	cv::bitwise_and(regionAroundSrcScratch, regionAroundSRoundScratch, regionAnd);

	return regionAnd;

}

cv::Mat getRawScratchRound(cv::Mat scratchRaw, int nearDist1 /*= 3 */ ,int nearDist2 /*= 5*/)
{
	cv::Mat regionAroundSrcScratch, regionAroundSRoundScratch, mor5, mor7;

	morphologyEx(scratchRaw, mor5, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(nearDist1 * 2 + 1, nearDist1 * 2 + 1)));
	morphologyEx(scratchRaw, mor7, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(nearDist2 * 2 + 1, nearDist2 * 2 + 1)));
	regionAroundSrcScratch = mor7 - mor5;

	return regionAroundSrcScratch;
}

//回溯端点 计算角度
double angleFromEndPt(cv::Mat binImg, cv::Point endPt,cv::Point &endPtNew, int cntThresh /*= 7*/)
{
	cv::Mat startImg, morImg;
	startImg.create(binImg.size(), binImg.type());
	startImg.setTo(0);
	startImg.at<uchar>(endPt) = 255;
	int cnt = 0;
	vector<Point> goBackPts;
	cv::Mat preImg, nowImg;
	while (cv::countNonZero(binImg - startImg) != 0 && cnt < cntThresh)
	{
		cv::dilate(startImg, morImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
		preImg = startImg.clone();
		cv::bitwise_and(binImg, morImg, startImg);
		nowImg = startImg.clone();

		cv::Mat diff;
		cv::bitwise_xor(preImg, nowImg, diff);
		vector<cv::Point> vecPts;
		imgPro::img_getPoints(diff, vecPts);
		goBackPts.insert(goBackPts.end(), vecPts.begin(), vecPts.end());
		cnt++;
	}	
		endPtNew = goBackPts.back();

	double angle = 0;
	return anglePostive(endPt, endPtNew);
}

// 功能：四舍五入（double），支持正负数
// dSrc ： 待四舍五入之数
// iBit ： 保留的小数位数。 0 - 不保留小数、1 - 保留一位小数
// 返回值：返回计算结果
// 

/*
功能：计算区域之间的最近点，若最近点距离小于distThresh认为是有效的
输入二值图像，可以是骨架图
*/
int biImg_connection(cv::Mat binImgIn, cv::Mat & connectedImg, double distThresh)
{
	cv::Mat labelImg;
	map<string, vector<double>> feats;
	cv::Mat preImg = binImgIn.clone();
	cv::Mat nowImg = binImgIn.clone();
	if (cv::countNonZero(binImgIn) < 3)
	{
		cout << "图像全黑" << endl;
		return 1;
	}
	map<int, vector<cv::Point>> mapVecPts;
	vector<pair<cv::Point, cv::Point>> vecPairPts;   //save valid point pairs	

	//循环迭代直至不在preImg,nowImg变化
	while (true)
	{
		imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, binImgIn.total());
		labelImg2vectorPts(labelImg, mapVecPts, -1);

		for (auto item1:mapVecPts)
		{
			map<int, vector<Point>> mapVecPtTmp(mapVecPts);
			mapVecPtTmp.erase(item1.first);
		
			vector<cv::Point> vecPts1 = item1.second;

			pair<int, pair<cv::Point, cv::Point>> res= distInPtsAndMap(vecPts1, mapVecPtTmp);
			double dist = imgPro::dist2Pts(res.second.first, res.second.second);

			if (dist < distThresh)
			{
				vecPairPts.push_back(res.second);

			}
		}
		for (auto pp:vecPairPts)
		{
			if (pp.first != cv::Point(0,0) && pp.second != cv::Point(0, 0))
			{
				cv::line(nowImg, pp.first, pp.second, Scalar(255), 1);
			}
		}
		cv::Mat orImg;
		cv::bitwise_xor(preImg, nowImg, orImg);
		if (cv::countNonZero(orImg) >0)
		{
			preImg = nowImg.clone();
		}
		else
		{
			connectedImg = nowImg.clone();
			break;
		}
	}
	
	return 0;
}

//线条末端进行延拓
cv::Mat scratch_extend(cv::Mat scratchImg, cv::Mat binRefImg, cv::Rect scratchRect)
{
	vector<Point> vecPtsExtend;
	cv::Mat extendLineShow = scratchImg.clone();
	cv::cvtColor(extendLineShow, extendLineShow, COLOR_GRAY2BGR);
	skeleton_endpoint(scratchImg, vecPtsExtend);
	for (auto pt : vecPtsExtend)
	{
		cv::Point ptNew;
		angleFromEndPt(scratchImg, pt, ptNew, 36);
		cv::Mat endSegImg = searchPtImg(scratchImg, pt, vector<cv::Point>{ptNew});
		vector<cv::Point> pts2fit;
		imgPro::img_getPoints(scratchImg, pts2fit);
		Vec4f linePara;
		cv::fitLine(pts2fit, linePara, DIST_FAIR, 0, 0.01, 0.01);

		pair<Point, Point> ptPair = getLinePtInImg(scratchImg, linePara, pt);
		cv::line(extendLineShow, ptPair.first, ptPair.second, Scalar(0, 0, 255), 3);
		cv::Mat canvs = scratchImg.clone();
		canvs.setTo(0);
		cv::line(canvs, ptPair.first, ptPair.second, Scalar(255), 3);
		cv::Mat andImg,toConnImg;
		cv::bitwise_and(canvs, binRefImg, andImg);

		imgPro::biImg_filterByArea(andImg, toConnImg,3,andImg.total(),2);

		if (cv::countNonZero(toConnImg) > 0)
		{
			vector<cv::Point> vecPts;
			imgPro::img_getPoints(toConnImg, vecPts);
			for (int i = 0; i < vecPts.size() - 1; i++)
			{
				
				cv::line(scratchImg, vecPts[i], vecPts[i + 1], Scalar(255));
			}
		}
	}

	return scratchImg;

}
cv::Mat set8Neib(cv::Mat binIn, cv::Point pt)
{
	cv::Mat imgOut = binIn.clone();
	for (int x=pt.x-1;x<pt.x+2;x++)
	{
		for (int  y = pt.y-1; y < pt.y+2; y++)
		{
			imgOut.at<uchar>(cv::Point(x,y)) = 0;
		}
	}

	return imgOut;
}

//按照一定规则进行断开划痕的连接，从最大区域开始连接
//binImgIn：输入二值图
//connImg：输出连接图
//vecPairPt：输出的连接点对
int scratch_connection(cv::Mat binImgIn, cv::Mat &connImg, vector<pair<cv::Point, cv::Point>> &vecPairPt)
{
	vector<Point> vecMaxScratchEnds;
	cv::Mat preImg, nowImg, scratchThinImg;

	vecPairPt.clear();
	nowImg = binImgIn.clone();
	preImg = binImgIn.clone();

	while (true)
	{
		cv::Mat labelImg;
		map<string, vector<double>> feats;
		cv::Mat maxScratchImg, maxScratchImgRaw;
		imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, nowImg.total());   //labelImg作为标记使用
		if (feats.size() < 1)
		{
			break;//图像只有一个区域
		}
		imgPro::biImg_getMaxArea(nowImg, maxScratchImgRaw);
		//去除交点
		maxScratchImg = skeleton_removeBranchs(maxScratchImgRaw, 15, 135);

		skeleton_endpoint(maxScratchImg, vecMaxScratchEnds);
		std::cout << "进入一次连接" << endl;

		for (auto endPt : vecMaxScratchEnds)
		{

			double distThreshLow = 25; //距离近允许的角度大，距离远允许的角度小
			double angleThreshLow = 13;
			//double distThreshHigh = 50;
			double angleThreshHigh = 30;
			double dist = 0;
			map<int, vector<cv::Point>> mapVecPts;
			int maxRegionLabel = labelImg.at<int>(endPt);
			cv::Point endPtNew;
			while (true)
			{
				//回溯像素计算角度,主动连接的大线段的末端角度
				double angleEnd = angleFromEndPt(maxScratchImg, endPt, endPtNew, 13);
				//计算该点本区域外最近的点，所在区域的角度，距离，连线角度
				labelImg2vectorPts(labelImg, mapVecPts, maxRegionLabel);
				if (mapVecPts.size()==0)
				{
					break;
				}
				vector<Point>	vecPts{ endPt };
				pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecPts, mapVecPts); //距离最近两点，和被连接点label

				double distPtPair = imgPro::dist2Pts(res.second.first, res.second.second);  //距离最近两点的距离
				
				double anglePtPairWithPartMaxScratch;//最近点对的连接线段 与 末端像素段 的向量夹角角度
				imgPro::angle2Lines(endPt, endPtNew, res.second.first, res.second.second, anglePtPairWithPartMaxScratch);

				if ((distPtPair > distThreshLow && (180 - anglePtPairWithPartMaxScratch) > angleThreshLow))//最近点距离超过阈值且偏角过大，跳出连接循环
				{
					break;
				}
				if (anglePtPairWithPartMaxScratch < 150) //跳过该区域，寻找下一个区域
				{
					//将该区域labelImg中去除，原图不能去除
										//移除该label,继续连接
					cv::Mat region2Remove;
					imgPro::region_toImg(labelImg, region2Remove, labelImg.at<int>(res.second.second));
					//将label中的label去掉
					vector<Point> pt2Remmove;
					imgPro::img_getPoints(region2Remove, pt2Remmove);
					for (Point pt : pt2Remmove)
					{
						labelImg.at<int>(pt) = 0;
					}
					continue;
				}

				double connectedRegionAngle; //被连接区域的角度
				int connectedRegionIndex = labelImg.at<int>(res.second.second);

				cv::Mat regionImg;
				imgPro::region_toImg(labelImg, regionImg, connectedRegionIndex);

				vector<double> vecLabel = feats["label"];
				auto iteLa = std::find(vecLabel.begin(), vecLabel.end(), connectedRegionIndex);
				int indexInVec = std::distance(vecLabel.begin(), iteLa);
				double rotHWRatio = feats["rotHWRatio"][indexInVec];
				if (cv::countNonZero(regionImg) > 3 && rotHWRatio >= 2.5) //长宽比
				{
					//计算被连接区域角度					
					vector<Point> ptsTmp;
					skeleton_endpoint(regionImg, ptsTmp);
					if (ptsTmp.size() < 2)
					{
						break;
					}
					//被连接最近点和其他端点的角度中最接近末端角度的区域
					vector<double> vecAngle, vecAngleMinus;
					for (auto pt : ptsTmp)
					{
						if (imgPro::dist2Pts(res.second.second, pt) > 2)
						{
							connectedRegionAngle = anglePostive(res.second.second, pt);
							vecAngle.push_back(connectedRegionAngle);
						}
					}
					vecAngleMinus.insert(vecAngleMinus.end(), vecAngle.begin(), vecAngle.end());
					for (double &val : vecAngleMinus)
					{
						val = abs(val - angleEnd);
					}
					//取最小差值
					auto iteLa = std::min_element(vecAngleMinus.begin(), vecAngleMinus.end());
					int indexInVec = std::distance(vecAngleMinus.begin(), iteLa);
					connectedRegionAngle = vecAngle[indexInVec];
				}
				else
				{
					connectedRegionAngle = angleEnd;
				}
				if (
					(abs(connectedRegionAngle - angleEnd) < 35)
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
					imgPro::region_toImg(labelImg, region2Remove, labelImg.at<int>(res.second.second));
					//cv::bitwise_xor(nowImg, region2Remove, nowImg); //原图中移除？

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
		std::cout << "连接一次完成" << endl;
		cv::Mat diff = (nowImg - preImg);
		if (countNonZero(diff) == 0)  //不再增加连接区域则跳出
		{
			break;
		}
		else {
			preImg = nowImg.clone();
		}
	}
	connImg = nowImg;
	return 0;
}



vector<cv::Mat> scratch_merge(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> vecPtPair,bool outputAllOrder)
{
	cv::Mat labelImg;
	map<string, vector<double>> feats;
	imgPro::biImg_createRegion(binImgIn, labelImg, feats, 0, binImgIn.total());
	vector<pair<int, int>> vecPtLabel;  //被连接的label
	for (auto ite : vecPtPair)
	{
		int labelVal1 = labelImg.at<int>(ite.first);
		int labelVal2 = labelImg.at<int>(ite.second);
		vecPtLabel.push_back(make_pair(labelVal1, labelVal2));
	}
	vector<int> vecLabelInLine;
	vector<vector<int>> vec2LineLabel;  //存在合并线条的直线
	vector<vector<pair<cv::Point,cv::Point>>> vecLineLabelPt;

	while (true)
	{
		vector<int> vecLineLabel;
		if (vecPtLabel.size() > 0)
		{
			vecLineLabel.push_back(vecPtLabel[0].first);
			vecLineLabel.push_back(vecPtLabel[0].second);
		}
		else
		{
			break;
		}
		int restartPos = 1;
		for (int i = 1; i < vecPtLabel.size(); i++)
		{
			if (std::count(vecLineLabel.begin(), vecLineLabel.end(), vecPtLabel[i].first) > 0 || std::count(vecLineLabel.begin(), vecLineLabel.end(), vecPtLabel[i].second) > 0)
			{
				vecLineLabel.push_back(vecPtLabel[i].first);
				vecLineLabel.push_back(vecPtLabel[i].second);
				vecLabelInLine.push_back(vecPtLabel[i].first);
				vecLabelInLine.push_back(vecPtLabel[i].second);

				if (i != restartPos)
				{
					swap(vecPtLabel[i], vecPtLabel[restartPos]);
					i = restartPos;
				}
				restartPos++;
			}
		}
		vec2LineLabel.push_back(vecLineLabel);
		vecPtLabel.erase(vecPtLabel.begin(), vecPtLabel.begin() + vecLineLabel.size() / 2);
		
	}
	vector<int> vecLabelAll(feats["label"].begin(), feats["label"].end());
	vector<int> vecSingleLine;   //单个区域构成的直线
	for (auto ite:vecLabelAll)
	{
		if (count(vecLabelInLine.begin(),vecLabelInLine.end(),ite) <1)
		{
			vecSingleLine.push_back(ite);
		}
	}
	vector<cv::Mat> vecMat;
	for (auto item:vec2LineLabel)
	{
		cv::Mat lineImg;
		imgPro::region_toImg(labelImg, lineImg, item);
		for (auto ptPair:vecPtPair)
		{
			if (lineImg.at<uchar>(ptPair.first)>0  )
			{
				cv::line(lineImg, ptPair.first, ptPair.second, cv::Scalar(255));
			}
		}
		if (cv::countNonZero(lineImg)> max(binImgIn.rows,binImgIn.cols)/2 || outputAllOrder)
		{
			vecMat.push_back(lineImg);
		}
	}

	for (auto item:vecSingleLine)
	{
		cv::Mat lineImg;
		imgPro::region_toImg(labelImg, lineImg, item);
		if (cv::countNonZero(lineImg) > max(binImgIn.rows, binImgIn.cols) / 2 || outputAllOrder)
		{
			vecMat.push_back(lineImg);
		}
	}
	vector<pair<int, int>> vecIndexArea;
	for (int i=0;i<vecMat.size();i++)
	{
		vecIndexArea.push_back(make_pair(i, cv::countNonZero(vecMat[i])));
	}
	sort(vecIndexArea.begin(), vecIndexArea.end(), [](auto it1, auto it2) {return it1.second < it2.second; });
	vector<Mat> orderMat;
	for (auto it:vecIndexArea)
	{
		orderMat.push_back(vecMat[it.first]);
	}
		return vecMat;

}




//vector<cv::Mat> scratch_merge(cv::Mat binImgIn, vector<pair<cv::Point, cv::Point>> vecPtPair)
//{
//	cv::Mat labelImg;
//	map<string, vector<double>> feats;
//	imgPro::biImg_createRegion(binImgIn, labelImg, feats, 0, binImgIn.total());
//	vector<pair<int, int>> vecPtLabel;  //被连接的label
//	for (auto ite : vecPtPair)
//	{
//		int labelVal1 = labelImg.at<int>(ite.first);
//		int labelVal2 = labelImg.at<int>(ite.second);
//		vecPtLabel.push_back(make_pair(labelVal1, labelVal2));
//	}
//	set<int> setLabelInLine;
//	vector<vector<int>> vecLineLabel;
//	vector<vector<pair<cv::Point,cv::Point>>> vecLineLabelPt;
//
//	//再次检查可否合并
//
//	vector<int> vecCnt;
//		vector<cv::Mat> vecMat;
//	cv::Mat binMaxCntImg, binMaxCntSingleImg;
//	for (int i=0;i<vecLineLabel.size();i++)
//	{
//		cv::Mat binTmp;
//		int maxCnt = 0;
//		cv::Rect rectTmp;
//		imgPro::region_toImg(labelImg, binTmp, vecLineLabel[i]);
//		for (auto it:vecLineLabelPt[i])
//		{
//			line(binTmp, it.first, it.second, cv::Scalar(255), 1);
//		}
//		int cnt = cv::countNonZero(binTmp);
//		if (cnt > maxCnt)
//		{
//			binMaxCntImg = binTmp.clone();
//		}
//		if (cnt / double(max(binTmp.rows,binTmp.cols)) > 0.5)
//		{
//			vecMat.push_back(binTmp);
//		}				
//	}
//	set<int> setLabelIndexAll(feats["label"].begin(), feats["label"].end());
//	vector<int> labelIndexSingleLine;  //未参加连接的独立的线
//	set_difference(setLabelIndexAll.begin(), setLabelIndexAll.end(), setLabelInLine.begin(), setLabelInLine.end(),
//		back_inserter(labelIndexSingleLine));
//	//剩下的单个划痕
//	for (auto it : labelIndexSingleLine)
//	{
//		cv::Mat binTmp;
//		int maxCnt = 0;
//		cv::Rect rectTmp;
//		imgPro::region_toImg(labelImg, binTmp, it);
//		int cnt = cv::countNonZero(binTmp);
//		if (cnt > maxCnt)
//		{
//			binMaxCntSingleImg = binTmp.clone();
//		}
//		if (cnt / double(max(binTmp.rows, binTmp.cols)) > 0.5)
//		{
//			vecMat.push_back(binTmp);
//		}
//	}
//	if (vecMat.size() < 1)
//	{
//		if (cv::countNonZero(binMaxCntSingleImg) >  cv::countNonZero(binMaxCntImg))
//		{
//			vecMat.push_back(binMaxCntSingleImg);
//		}
//		else
//		{
//			vecMat.push_back(binMaxCntImg);
//		}
//	}
//
//	return vecMat;
//
//}

//输入二值图，遍历所有端点，连接划痕
//一般输入去除交点的骨架图
int scratch_connection2(cv::Mat binImgIn,vector<pair<cv::Point, cv::Point>> &vecPairPt, int angleFrmEndPtLen,double distThreshLow/*=15*/,double distTheshHigh/*=30*/)
{
	vector<Point> vecMaxScratchEnds;
	cv::Mat preImg, nowImg, scratchThinImg;

	vecPairPt.clear();
	nowImg = binImgIn.clone();
	preImg = binImgIn.clone();

	while (true)
	{
		cv::Mat labelImg;
		map<string, vector<double>> feats;
		cv::Mat maxScratchImg, maxScratchImgRaw;	
		//maxScratchImg = skeleton_removeBranchs(nowImg, 15, 135);
		maxScratchImg = nowImg.clone();
		skeleton_endpoint(maxScratchImg, vecMaxScratchEnds);
		std::cout << "进入一次连接" << endl;
		for (auto endPt : vecMaxScratchEnds)
		{		
			//double angleThreshLow = 13;   
			double anglePtPairWithPartMaxScratchThreshFar = 180-13; //主动连接划痕与连接线的阈值,远距离允许的误差小
			double anglePtPairWithPartMaxScratchThreshNear = 180-25; //主动连接划痕与连接线的阈值，近距离允许的误差大
			double angleInRegionsThresh = 30;   //两个被连接划痕的角度的差 阈值

			double dist = 0;
			cv::Point endPtNew;
			map<int, vector<cv::Point>> mapVecPts;
			imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, nowImg.total());   //labelImg作为标记使用
			if (feats["label"].size() < 2)
				break;//图像只有一个区域

			int regionLabel = labelImg.at<int>(endPt);
			//int angleFrmEndPtLen = 13;

			while (true)
			{
				//回溯像素计算角度,主动连接的线段的末端角度
				double angleEnd = angleFromEndPt(maxScratchImg, endPt, endPtNew, angleFrmEndPtLen);
				//计算该点本区域外最近的点，所在区域的角度，距离，连线角度
				labelImg2vectorPts(labelImg, mapVecPts, regionLabel);
				if (mapVecPts.size()==0)
				{
					break;
				}
				vector<Point>	vecPts{ endPt };
				pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecPts, mapVecPts); //距离最近两点，和被连接点label

				double distPtPair = imgPro::dist2Pts(res.second.first, res.second.second);  //距离最近两点的距离
				//最近点对的连接线段 与 末端像素段 的向量夹角角度
				double anglePtPairWithPartMaxScratch;
				imgPro::angle2Lines(endPt, endPtNew, res.second.first, res.second.second, anglePtPairWithPartMaxScratch);

				if ((distPtPair > distTheshHigh /*&& (180 - anglePtPairWithPartMaxScratch) > angleThreshLow*/))//最近点距离超过阈值且偏角过大，跳出连接循环
				{
					break;
				}
				if ( ( distPtPair <= distThreshLow &&  anglePtPairWithPartMaxScratch < anglePtPairWithPartMaxScratchThreshNear ) ||
					 ( distPtPair > distThreshLow &&  anglePtPairWithPartMaxScratch < anglePtPairWithPartMaxScratchThreshFar)
					) //长短距离内均不符合连接条件，   跳过该区域，寻找下一个区域
				{
					//将该区域labelImg中去除，原图不能去除										
					cv::Mat region2Remove;
					imgPro::region_toImg(labelImg, region2Remove, labelImg.at<int>(res.second.second));
					//将label中的label去掉
					vector<Point> pt2Remmove;
					imgPro::img_getPoints(region2Remove, pt2Remmove);
					for (Point pt : pt2Remmove)
					{
						labelImg.at<int>(pt) = 0;
					}
					continue;
				}

				double connectedRegionAngle; //被连接区域的角度
				double angleInTwoRegion = 0;
				int connectedRegionIndex = labelImg.at<int>(res.second.second);

				cv::Mat regionImg;
				imgPro::region_toImg(labelImg, regionImg, connectedRegionIndex);

				vector<double> vecLabel = feats["label"];
				auto iteLa = std::find(vecLabel.begin(), vecLabel.end(), connectedRegionIndex);
				int indexInVec = std::distance(vecLabel.begin(), iteLa);
				double rotHWRatio = feats["rotHWRatio"][indexInVec];
				if ( (cv::countNonZero(regionImg) > 3 && rotHWRatio >= 2.5) || 
					  cv::countNonZero(regionImg) >= 5) //长宽比
				{
					//计算被连接区域角度					
					vector<Point> ptsTmp;
					skeleton_endpoint(regionImg, ptsTmp);
					double angleInTwoRegionTmp = 0;
					if (ptsTmp.size() < 2)
					{
						break;
					}
					for (auto pt : ptsTmp)
					{
						if (imgPro::dist2Pts(res.second.second, pt) > 2)
						{
							imgPro::angle2Lines(endPt, endPtNew, res.second.second, pt, angleInTwoRegionTmp);
							if (angleInTwoRegionTmp > angleInTwoRegion)
							{
								angleInTwoRegion = angleInTwoRegionTmp;
							}
						}
					}
				}
				else
				{
					angleInTwoRegion = 180;
				}
				if ( angleInTwoRegion   > 180- angleInRegionsThresh 	)
				{
					//可连接点对
					vecPairPt.push_back(res.second);
					//连接图像
					//cv::line(nowImg, vecPairPt.back().first, vecPairPt.back().second, Scalar(255));
					break;
				}
				else
				{
					//移除该label,继续连接
					cv::Mat region2Remove;
					imgPro::region_toImg(labelImg, region2Remove, labelImg.at<int>(res.second.second));
					//cv::bitwise_xor(nowImg, region2Remove, nowImg); //原图中移除？

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
		std::cout << "连接一次完成" << endl;
		cv::Mat diff = (nowImg - preImg);
		if (countNonZero(diff) == 0)  //不再增加连接区域则跳出
		{
			break;
		}
		else {
			preImg = nowImg.clone();
		}
	}
	//connImg = nowImg;
	return 0;
}


//将有交叉的划痕分开，按划痕长度大小，降序输出
// 超过angleMergeThresh，可以合并
vector<cv::Mat> scratch_split(cv::Mat skeletonImgIn, double angleMergeThresh)
{
	cv::Mat zerOneImg;
	vector<Point> vecEndPt, vecEndPtBroken,vecIntersecPt;
	cv::Mat binIn = skeletonImgIn.clone();
	skeleton_endpoint(skeletonImgIn, vecEndPt);
	skeleton_intersecPoint(binIn, vecIntersecPt);
	cv::Mat endShowImg = skeletonImgIn.clone();
	cv::cvtColor(endShowImg, endShowImg, COLOR_GRAY2BGR);
	for (auto pt : vecIntersecPt)
		endShowImg.at<Vec3b>(pt) = Vec3b(0, 255, 0);

	cv::Mat intersecBrokenImg;
	intersecBrokenImg = skeletonImgIn.clone();
	for (auto pt : vecIntersecPt)
	{		
		intersecBrokenImg = set8Neib(intersecBrokenImg, pt); //将pt8邻域内置为0，查找该邻域内相对交点
	}

	cv::Mat labelImg, connImg;
	map<string, vector<double>> feats;
	skeleton_endpoint(intersecBrokenImg, vecEndPtBroken);
	imgPro::biImg_createRegion(intersecBrokenImg, labelImg, feats, 0, intersecBrokenImg.total());
	vector<pair<cv::Point, cv::Point>> vecPtPair;
	scratch_connection2(intersecBrokenImg, vecPtPair);	
	
	return vector<cv::Mat>();
}


//输入：细化线条  
vector<cv::Mat> snapBox_splitScratch(cv::Mat &skeletonImgIn,double angleThreshVal)
{
	vector<pair<cv::Point, cv::Point>> vecPairPt;
	vector<cv::Point> vecIntersecPt;
	cv::Mat intersecBrokenImg, toConnImg, showImg;
	skeleton_intersecPoint(skeletonImgIn, vecIntersecPt);
	intersecBrokenImg = skeletonImgIn.clone();
	for (auto pt : vecIntersecPt)
	{
		intersecBrokenImg = set8Neib(intersecBrokenImg, pt); //将pt8邻域内置为0，查找该邻域内相对交点
	}
	imgPro::biImg_filterByArea(intersecBrokenImg, toConnImg, 3, skeletonImgIn.total(), 2);

	scratch_connection2(toConnImg, vecPairPt);

	//处理连接区域
	showImg = toConnImg.clone();
	cv::cvtColor(showImg, showImg, COLOR_GRAY2BGR);
	RNG rng(0xFFFFFFFF);
	for (auto it : vecPairPt)
	{
		line(showImg, it.first, it.second, imgPro::randomColor(rng));
	}

	vector<cv::Mat> scratchImgs = scratch_merge(toConnImg, vecPairPt);

	return scratchImgs;
}




//全局图像划痕连接,限制距离和连接区域大小
int biImg_connection(cv::Mat binImgIn, cv::Mat & connectedImg, double distThreshLow, double distThreshHigh, double areaThreshLow)
{
	cv::Mat labelImg;
	map<string, vector<double>> feats;
	cv::Mat preImg = binImgIn.clone();
	cv::Mat nowImg = binImgIn.clone();
	if (cv::countNonZero(binImgIn) < 3)
	{
		cout << "图像全黑" << endl;
		return 1;
	}
	map<int, vector<cv::Point>> mapVecPts;
	vector<pair<cv::Point, cv::Point>> vecPairPts;   //save valid point pairs	

	//循环迭代直至preImg,nowImg不再变化
	imgPro::biImg_createRegion(nowImg, labelImg, feats, 0, binImgIn.total());
	int distLarge =  distThreshLow;
	while (true)
	{
		labelImg2vectorPts(labelImg, mapVecPts, -1);

		for (auto item1 : mapVecPts)
		{
			map<int, vector<Point>> mapVecPtTmp;
			//取附近位置label
			cv::Point tlPt = cv::Point(feats["left"][item1.first]- distLarge, feats["top"][item1.first]-distLarge);
			int newWidth = feats["width"][item1.first] + 2 * distLarge;
			int newHeight= feats["height"][item1.first] + 2 * distLarge;
			cv::Rect rectLocal = cv::Rect(tlPt, cv::Size(newWidth, newHeight));
			cv::Rect rectNorm = imgPro::rect_normInImg(rectLocal,labelImg.size());
			cv::Mat localLabelImg = labelImg(rectNorm);

			vector<int> localGrayVals;
			imgPro::img_getGrayLevel(localLabelImg, localGrayVals, false);
			for (auto i:localGrayVals)
			{
				mapVecPtTmp.insert(make_pair(i, mapVecPts[i]));
			}			
			mapVecPtTmp.erase(item1.first);

			vector<cv::Point> vecPts1 = item1.second;

			pair<int, pair<cv::Point, cv::Point>> res = distInPtsAndMap(vecPts1, mapVecPtTmp);
			double dist = imgPro::dist2Pts(res.second.first, res.second.second);

			if (dist < distThreshLow)
			{
				vecPairPts.push_back(res.second);

			}
		}
		for (auto pp : vecPairPts)
		{
			cv::line(nowImg, pp.first, pp.second, Scalar(255), 1);
		}
		cv::Mat orImg;
		cv::bitwise_xor(preImg, nowImg, orImg);
		if (cv::countNonZero(orImg) > 0)
		{
			preImg = nowImg.clone();
			//将已经连接的标记 修为一致
		}
		else
		{
			connectedImg = nowImg.clone();
			break;
		}
	}

	return 0;
}

int labelImg2vectorPts(cv::Mat labelImg, map<int, vector<cv::Point>> &mapVecPts, int excludeLabel, bool includZero/*=false*/)
{
	mapVecPts.clear();

	for (int r = 0; r < labelImg.rows; r++)
	{
		for (int c = 0; c < labelImg.cols; c++)
		{
			if (labelImg.at<int>(r, c) != excludeLabel)
			{
				if (labelImg.at<int>(r, c) == 0)
				{
					if (includZero)
					{
						mapVecPts[labelImg.at<int>(r, c)].push_back(cv::Point(c, r));
					}
					else
					{
						continue;
					}
				}
				else
				{
					mapVecPts[labelImg.at<int>(r, c)].push_back(cv::Point(c, r));
				}
			}
		}
	}
	return 0;
}

//返回 label , 点对,第二个点是被连接的点
pair<int, pair<cv::Point, cv::Point>> distInPtsAndMap(vector<cv::Point> vecPts, map<int, vector<cv::Point>> mapPts)
{
	cv::Point pt1, pt2;
	int nearestLabelIndex = -1;
	double nearestDist = 10000.;
	for (auto pt : vecPts)
	{
		for (auto ite = mapPts.begin(); ite != mapPts.end(); ite++)
		{
			pair<double, cv::Point> res = getMinDistInPtAndRegion(pt, ite->second);
			if (res.first < nearestDist)
			{
				nearestDist = res.first;
				pt1 = pt;
				pt2 = res.second;
				nearestLabelIndex = ite->first;
			}
		}
	}
	return pair<int, pair<cv::Point, cv::Point>>(nearestLabelIndex, pair<cv::Point, cv::Point>(pt1, pt2));
}

//返回 最小距离和该点
pair<double, cv::Point> getMinDistInPtAndRegion(cv::Point pt, vector<cv::Point> vecPts)
{
	if (vecPts.size() < 1)
	{
		return pair<double, cv::Point>();
	}
	double nearestDist = 10000.;
	cv::Point nearestPt;
	for (auto p : vecPts)
	{
		double distTmp = imgPro::dist2Pts(pt, p);
		if (distTmp < nearestDist)
		{
			nearestDist = distTmp;
			nearestPt = p;
		}
	}

	return pair<double, cv::Point>(nearestDist, nearestPt);
}

double getDist2Pts(cv::Vec4f lineSeg)
{
	return imgPro::dist2Pts(cv::Point(lineSeg[0], lineSeg[1]), cv::Point(lineSeg[2], lineSeg[3]));
}

cv::Point getFarEndPt(cv::Vec4f lineSeg, cv::Point pt)
{
	cv::Point p1(cv::Point2f(lineSeg[0], lineSeg[1]));
	cv::Point p2(cv::Point2f(lineSeg[2], lineSeg[3]));
	if (imgPro::dist2Pts(p1, pt) > imgPro::dist2Pts(p2, pt))
	{
		return p1;
	}
	else
	{
		return p2;
	}

}
cv::Point getNearEndPt(cv::Vec4f lineSeg, cv::Point pt)
{
	cv::Point p1(cv::Point2f(lineSeg[0], lineSeg[1]));
	cv::Point p2(cv::Point2f(lineSeg[2], lineSeg[3]));
	if (imgPro::dist2Pts(p1, pt) > imgPro::dist2Pts(p2, pt))
	{
		return p2;
	}
	else
	{
		return p1;
	}

}
//弥补二值图像中一个像素交错的间隙，该间隙形态学无法弥补
cv::Mat  biImg_fillLineCrossGap(cv::Mat binImgIn)
{
	cv::Mat zerOneImg;
	cv::threshold(binImgIn, zerOneImg, 0, 1, cv::THRESH_BINARY);

	cv::Mat kernel1 = (Mat_<int>(3, 3) << -1, -1, 1,
		-1, -1, -1,
		1, -1, -1);

	cv::Mat kernel2 = (Mat_<int>(3, 3) << 1, -1, -1,
		-1, -1, -1,
		-1, -1, 1);

	cv::Mat kernel3 = (Mat_<int>(3, 3) << -1, 1, 0,
		-1, -1, 0,
		1, -1, -1);

	cv::Mat kernel4 = (Mat_<int>(3, 3) << 0, 0, -1,
		1, -1, -1,
		-1, -1, 1);

	cv::Mat kernel5 = (Mat_<int>(3, 3) << 0, 1, -1,
		0, -1, -1,
		-1, -1, 1);

	cv::Mat kernel6 = (Mat_<int>(3, 3) << -1, -1, 1,
		1, -1, -1,
		0, 0, -1);

#pragma endregion
	cv::Mat joingGapImg, hitmisImg;
	//zerOneImg.convertTo(zerOneImg, CV_32S);
	joingGapImg.create(binImgIn.size(), CV_8UC1);
	joingGapImg.setTo(0);
	vector<cv::Mat> kerners{ kernel1,kernel2,kernel3,kernel4,kernel5,kernel6 };
	for (cv::Mat ker : kerners)
	{
		cv::morphologyEx(zerOneImg, hitmisImg, cv::MorphTypes::MORPH_HITMISS, ker);

		cv::bitwise_or(hitmisImg, joingGapImg, joingGapImg);
	}
	vector<cv::Point> jointEndPts;
	joingGapImg = joingGapImg * 255;
	for (int r = 0; r < joingGapImg.rows; r++)
	{
		for (int c = 0; c < joingGapImg.cols; c++)
		{
			if (joingGapImg.at<uchar>(r, c) > 0)
			{
				jointEndPts.push_back(cv::Point(c - 1, r - 1));
			}
		}
	}

	cv::Mat resImg;
	cv::bitwise_or(binImgIn, joingGapImg, resImg);
	return resImg;
}

//
cv::Mat removeScreenEdge(cv::Mat binIn,cv::Mat &edgeImg)
{
	cv::Mat widthMorImg1, heightMorImg1, longImg, binIn2;
	cv::Mat labelImg, widthImg, heightImg, filterOut, cntImg;
	map<string, vector<double>> feats;
	cv::morphologyEx(binIn, binIn2, cv::MorphTypes::MORPH_CLOSE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);

	cv::morphologyEx(binIn2, widthMorImg1, cv::MorphTypes::MORPH_ERODE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 1)));
	cv::morphologyEx(widthMorImg1, widthMorImg1, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 1)), cv::Point(-1, -1), 3);

	//imgPro::biImg_createRegion(binIn, labelImg, feats, 20, 10000);
	//vector<int> vecCirLabel;
	//for (int i = 0; i < feats["label"].size(); i++)
	//{
	//	if (feats["circle"][i] > 0.6 && feats["width"][i] > 5)
	//	{
	//		vecCirLabel.push_back(feats["label"][i]);
	//	}
	//}
	//cv::Mat cirImg, toRemove;
	//imgPro::region_toImg(labelImg, cirImg, vecCirLabel);

	imgPro::biImg_createRegion(widthMorImg1, labelImg, feats, 100, widthMorImg1.total());
	vector<int> vecLabel, vecValidLabel, vecRow, veclongRegionLabel;
	for (int i = 0; i < feats["label"].size(); i++)
	{
		if (feats["area"][i] / 6 > widthMorImg1.size().width * 2 / 3)  // 连接较好的屏幕边缘
		{
			veclongRegionLabel.push_back(feats["label"][i]);
			continue;
		}
		else if (feats["width"][i] > 100 && (abs(feats["rotAngle"][i]) < 8))
		{
			vecLabel.push_back(feats["label"][i]);
			vecRow.push_back(feats["row"][i]);
		}
	}
	imgPro::region_toImg(labelImg, cntImg, vecLabel);
	for (int i = 0; i < vecRow.size(); i++)
	{
		int cnt = cv::countNonZero(cntImg(cv::Range(max(0, vecRow[i] - 7), min(cntImg.rows - 1, vecRow[i] + 7)), cv::Range::all()));
		//cout << "cnt " << cnt << endl;
		if (cnt / 2 > widthMorImg1.size().width / 4 * 3)
		{
			vecValidLabel.push_back(vecLabel[i]);
		}
	}
	vecValidLabel.insert(vecValidLabel.end(), veclongRegionLabel.begin(), veclongRegionLabel.end());
	imgPro::region_toImg(labelImg, widthImg, vecValidLabel);
	cv::morphologyEx(widthImg, widthImg, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);

	cv::morphologyEx(binIn2, heightMorImg1, cv::MorphTypes::MORPH_ERODE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(1, 7)));
	cv::morphologyEx(heightMorImg1, heightMorImg1, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(1, 7)), cv::Point(-1, -1), 3);

	imgPro::biImg_createRegion(heightMorImg1, labelImg, feats, 170, heightMorImg1.total());
	vecLabel.clear(); vecValidLabel.clear(); veclongRegionLabel.clear();
	vector<int> vecCol;
	for (int i = 0; i < feats["label"].size(); i++)
	{
		if (feats["area"][i] / 5 > widthMorImg1.size().height / 2)  // 去除连接较好的屏幕边缘
		{
			veclongRegionLabel.push_back(feats["label"][i]);
			continue;
		}

		if (feats["height"][i] > 170 && (abs(feats["rotAngle"][i]) > 90 - 8))
		{
			vecLabel.push_back(feats["label"][i]);
			vecCol.push_back(feats["col"][i]);
		}
	}
	imgPro::region_toImg(labelImg, cntImg, vecLabel);
	for (int i = 0; i < vecLabel.size(); i++)
	{
		int cnt = cv::countNonZero(cntImg(cv::Range::all(), cv::Range(max(0, vecCol[i] - 6), min(cntImg.cols - 1, vecCol[i] + 6))));
		if (cnt / 2 > heightMorImg1.size().height / 3 * 2)
		{
			vecValidLabel.push_back(vecLabel[i]);
		}
	}
	vecValidLabel.insert(vecValidLabel.end(), veclongRegionLabel.begin(), veclongRegionLabel.end());
	imgPro::region_toImg(labelImg, heightImg, vecValidLabel);
	cv::morphologyEx(heightImg, heightImg, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);
	
	edgeImg = widthImg + heightImg;
	return binIn - edgeImg;
}
//去除L2中的屏幕与边框边缘
cv::Mat removeScreenEdgeL2(cv::Mat &hessianImg,cv::Mat &edgeImg)
{
	cv::Mat binImg,widthMorImg1, heightMorImg1, longImg, binIn2;
	cv::Mat labelImg, widthImg, heightImg, filterOut, cntImg, horImg;
	map<string, vector<double>> feats;
	cv::threshold(hessianImg, binImg,15,255,cv::THRESH_BINARY);

	cv::morphologyEx(binImg, horImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 1)));
	int horizontal_size = 9;
	cv::Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
	cv::erode(horImg, horImg, horizontalStructure, Point(-1, -1));
	cv::dilate(horImg, horImg, horizontalStructure, Point(-1, -1));


	imgPro::biImg_createRegion(widthMorImg1, labelImg, feats, 100, widthMorImg1.total());
	vector<int> vecLabel, vecValidLabel, vecRow, veclongRegionLabel;
	for (int i = 0; i < feats["label"].size(); i++)
	{
		if (feats["area"][i] / 6 > widthMorImg1.size().width * 2 / 3)  // 连接较好的屏幕边缘
		{
			veclongRegionLabel.push_back(feats["label"][i]);
			continue;
		}
		else if (feats["width"][i] > 100 && (abs(feats["rotAngle"][i]) < 8))
		{
			vecLabel.push_back(feats["label"][i]);
			vecRow.push_back(feats["row"][i]);
		}
	}
	imgPro::region_toImg(labelImg, cntImg, vecLabel);
	for (int i = 0; i < vecRow.size(); i++)
	{
		int cnt = cv::countNonZero(cntImg(cv::Range(max(0, vecRow[i] - 7), min(cntImg.rows - 1, vecRow[i] + 7)), cv::Range::all()));
		//cout << "cnt " << cnt << endl;
		if (cnt / 2 > widthMorImg1.size().width / 4 * 3)
		{
			vecValidLabel.push_back(vecLabel[i]);
		}
	}
	vecValidLabel.insert(vecValidLabel.end(), veclongRegionLabel.begin(), veclongRegionLabel.end());
	imgPro::region_toImg(labelImg, widthImg, vecValidLabel);
	cv::morphologyEx(widthImg, widthImg, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);

	cv::morphologyEx(binIn2, heightMorImg1, cv::MorphTypes::MORPH_ERODE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(1, 7)));
	cv::morphologyEx(heightMorImg1, heightMorImg1, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(1, 7)), cv::Point(-1, -1), 3);

	imgPro::biImg_createRegion(heightMorImg1, labelImg, feats, 170, heightMorImg1.total());
	vecLabel.clear(); vecValidLabel.clear(); veclongRegionLabel.clear();
	vector<int> vecCol;
	for (int i = 0; i < feats["label"].size(); i++)
	{
		if (feats["area"][i] / 5 > widthMorImg1.size().height / 2)  // 去除连接较好的屏幕边缘
		{
			veclongRegionLabel.push_back(feats["label"][i]);
			continue;
		}

		if (feats["height"][i] > 170 && (abs(feats["rotAngle"][i]) > 90 - 8))
		{
			vecLabel.push_back(feats["label"][i]);
			vecCol.push_back(feats["col"][i]);
		}
	}
	imgPro::region_toImg(labelImg, cntImg, vecLabel);
	for (int i = 0; i < vecLabel.size(); i++)
	{
		int cnt = cv::countNonZero(cntImg(cv::Range::all(), cv::Range(max(0, vecCol[i] - 6), min(cntImg.cols - 1, vecCol[i] + 6))));
		if (cnt / 2 > heightMorImg1.size().height / 3 * 2)
		{
			vecValidLabel.push_back(vecLabel[i]);
		}
	}
	vecValidLabel.insert(vecValidLabel.end(), veclongRegionLabel.begin(), veclongRegionLabel.end());
	imgPro::region_toImg(labelImg, heightImg, vecValidLabel);
	cv::morphologyEx(heightImg, heightImg, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);
	
	edgeImg = widthImg + heightImg;
	return heightImg - edgeImg;
}

cv::Mat removeScreenEdgeLineSeg(cv::Mat &srcImg,cv::Mat &validMask, cv::Mat lineLabelImg, vector<cv::Vec4f>& lineSegs)
{
	const float angleDiffThresh = 4; //允许的角度误差
	const float distDiffThresh = 6;   // 领域距离
	vector<cv::Vec4f> vecHorLineSegUp, vecHorLineSegDown, vecVerLineSegR, vecVerLineSegL;
	cv::Vec4f verLineSeg(1, 1, 1, 10);
	cv::Vec4f horLineSeg(1, 1, 10, 1);
	cv::Mat lineInvalidImg;

	//查找去除线段范围
	cv::Mat lineBinImg, binLFilterImg;
	cv::threshold(lineLabelImg, lineBinImg, 0, 255, THRESH_BINARY);
	lineBinImg.convertTo(lineBinImg, CV_8UC1);
	//cv::bitwise_and(lineInvalidMaskImg, lineBinImg, lineInvalidImg);

	vector<int> vecHorUp;
	cv::Mat projImg;
	vector<int> vecProj;
	imgPro::img_projection(lineBinImg, projImg, vecProj, 0, 0,true);
	
	vector<int> vecProjMid(vecProj.begin() + 800, vecProj.end() - 800);
	std::sort(vecProjMid.begin(), vecProjMid.end(), greater<int>());
	int meanVal =std::accumulate(vecProjMid.begin(),vecProjMid.begin()+vecProjMid.size()/3,0)/(vecProjMid.size()/3);
	std::cout<<meanVal << endl;
	vector<int> vecHorEdgePtRow, vecVerEdgePtCol;
	vector<cv::Vec4f> vecDelLineSegs;  //待去除的线段
	for (int i=0;i<lineBinImg.rows;i++)
	{
		if (vecProj[i]> meanVal*3 && (i < 300 || srcImg.rows -300 <i))
		{
			vecHorEdgePtRow.push_back(i);
		}
	}
	imgPro::img_projection(lineBinImg, projImg, vecProj, 1, 0,true);
	 vecProjMid.assign(vecProj.begin() + 800, vecProj.end() - 800);
	std::sort(vecProjMid.begin(), vecProjMid.end(), greater<int>());
	meanVal = std::accumulate(vecProjMid.begin(), vecProjMid.begin() + vecProjMid.size() / 3, 0) / (vecProjMid.size() / 3);
	std::cout<<meanVal << endl;

	for (int i = 0; i < lineBinImg.cols; i++)
	{
		if (vecProj[i] >max(150, meanVal*3 )&& (i <600 || srcImg.cols-500 < i))
		{
			vecVerEdgePtCol.push_back(i);
		}
	}

	for (auto it =lineSegs.begin();it != lineSegs.end();)//删除满足条件的线段
	{
		double angleWithHorLine, angleWithVerLine;
		imgPro::angle2Lines(cv::Point2f(horLineSeg[0], horLineSeg[1]), cv::Point2f(horLineSeg[2], horLineSeg[3]),
							cv::Point2f((*it)[0], (*it)[1]), cv::Point2f((*it)[2], (*it)[3]), angleWithHorLine);

		imgPro::angle2Lines(cv::Point2f(verLineSeg[0], verLineSeg[1]), cv::Point2f(verLineSeg[2], verLineSeg[3]),
							cv::Point2f((*it)[0], (*it)[1]), cv::Point2f((*it)[2], (*it)[3]), angleWithVerLine);

		if (abs(angleWithHorLine-180) < angleDiffThresh || abs(angleWithHorLine) < angleDiffThresh) //水平
		{
			bool del = false;
			for (auto itRow:vecHorEdgePtRow)
			{
				if (abs((*it)[1] - itRow) < distDiffThresh)
				{
					vecDelLineSegs.push_back(*it);
					it = lineSegs.erase(it);
					del = true;
					break;
				}
			}
			if (!del)
			{
				it++;
			}
			
		}
		else if (abs(angleWithVerLine - 180) < angleDiffThresh || abs(angleWithVerLine) < angleDiffThresh)//竖直
		{
			bool del = false;
			for (auto itCol : vecVerEdgePtCol)
			{
				if (abs((*it)[0] - itCol) < distDiffThresh)
				{
					vecDelLineSegs.push_back(*it);
					it = lineSegs.erase(it);
					del = true;
					break;
				}
			}
			if (!del)
			{
				it++;
			}
		}
		else
		{
			it++;
		}

	}
	cv::Mat showImg, invalidMask;
	//cv::bitwise_not(validMask, invalidMask);
	//cv::Mat edgeImg = cv::Mat::zeros(lineBinImg.size(),CV_8UC1);
	//imgPro::img_drawSegments(edgeImg, edgeImg, lineSegs, cv::Scalar::all(255));
	//////四周无效区域
	//cv::Mat screenOuterEdgeRegionImg = nearInvalidRegion(invalidMask, 21, 5);
	//imgPro::img_drawMask(edgeImg, edgeImg, screenOuterEdgeRegionImg, Scalar(0, 0, 255), 0.15);
	
	//删除左侧刘海屏检测到的直线
	//cv::Mat screenEdgeResImg = camScreenEdge(srcImg, validMask, edgeAboveCamImg);
	//screenOuterEdgeRegionImg += edgeAboveCamImg;

	////右下，左上角
	////const int trDist = 600;
	////const int tlDist = 600;
	////cv::Mat trBinImg, fillImg;
	////cv::Rect trRect(srcImg.cols - trDist, 0, trDist, trDist);
	////cv::Rect brRect(srcImg.cols - tlDist-1, srcImg.rows-tlDist-1, tlDist, tlDist);
	////cv::Mat trResImg = screenCornerEdge(srcImg, validMask, trRect);
	////cv::Mat tlResImg = screenCornerEdge(srcImg, validMask, brRect);
	////screenEdgeResImg = screenEdgeResImg + trResImg + tlResImg;
	//vector<vector<cv::Point>> vecCont;
	//cv::findContours(screenEdgeResImg+ screenOuterEdgeRegionImg, vecCont, cv::RetrievalModes::RETR_LIST,cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	//showImg= lineBinImg.clone();
	//cv::cvtColor(showImg, showImg, cv::COLOR_GRAY2BGR);	
	//cv::drawContours(showImg, vecCont, -1, cv::Scalar(0, 0, 255));

	return showImg;
}
//返回 屏幕边缘矩形
cv::Rect removeScreenEdgeLineSegL2(cv::Mat &srcImg,cv::Mat &validMask, cv::Mat lineLabelImg, vector<cv::Vec4f>& lineSegs)
{
	const float angleDiffThresh = 3; //允许的角度误差
	const float distDiffThresh = 6;   // 领域距离
	vector<cv::Vec4f> vecHorLineSegUp, vecHorLineSegDown, vecVerLineSegR, vecVerLineSegL;
	cv::Vec4f verLineSeg(1, 1, 1, 10);
	cv::Vec4f horLineSeg(1, 1, 10, 1);
	cv::Mat lineInvalidImg;

	//查找去除线段范围
	cv::Mat lineBinImg, binLFilterImg;
	cv::threshold(lineLabelImg, lineBinImg, 0, 255, THRESH_BINARY);
	lineBinImg.convertTo(lineBinImg, CV_8UC1);

	vector<int> vecHorUp;
	cv::Mat projImg;
	vector<int> vecProj;
	//上下内边范围,找到validMask的内部上下边缘
	int cols = validMask.cols;
	imgPro::img_projection(validMask, projImg, vecProj, 0, 0, true);
	int upRowScreen = 0, dwRowScreen = 0;
	upRowScreen = std::distance(vecProj.begin(), std::find_if(vecProj.rbegin() + vecProj.size() / 2, vecProj.rend(),
		[cols](auto it) {return it > cols / 3; }).base());
	dwRowScreen = std::distance(vecProj.begin(), std::find_if(vecProj.begin() + vecProj.size() / 2, vecProj.end(),
		[cols](auto it) {return it > cols / 3; }));

	imgPro::img_projection(lineBinImg, projImg, vecProj, 0, 0,true);	
	vector<int> vecProjMid(vecProj.begin() + 800, vecProj.end() - 800);//用于求取平均水平线段的数量
	std::sort(vecProjMid.begin(), vecProjMid.end(), greater<int>());
	int meanVal =std::accumulate(vecProjMid.begin(),vecProjMid.begin()+vecProjMid.size()/3,0)/(vecProjMid.size()/3);
	std::cout<<meanVal*5 << endl;
	vector<int> vecHorEdgePtRow, vecVerEdgePtCol;
	vector<cv::Vec4f> vecDelLineSegs;  //待去除的线段
	int edgeRowUpCnt = *std::max_element(vecProj.begin(), vecProj.begin() + 800);
	int edgeRowUp = std::distance(vecProj.begin(), std::max_element(vecProj.begin(), vecProj.begin() + 800));
	if (edgeRowUpCnt > max(800, meanVal * 5))
		vecHorEdgePtRow.push_back(edgeRowUp); 
	else
		vecHorEdgePtRow.push_back(upRowScreen);

	int edgeRowDwCnt = *std::max_element(vecProj.end()-1000, vecProj.end());
	int edgeRowDw = std::distance(vecProj.begin(), std::max_element(vecProj.end() - 1000, vecProj.end()));
	if (edgeRowDwCnt > max(800, meanVal * 5))
		vecHorEdgePtRow.push_back(edgeRowDw);
	else
		vecHorEdgePtRow.push_back(dwRowScreen);

	//左右外边沿范围
	int rows = validMask.rows;
	imgPro::img_projection(validMask, projImg, vecProj, 1, 0, true);
	int leftColStart = 0, leftColEnd = 0, rColStart = 0, rColEnd = 0;
	leftColEnd = std::distance(vecProj.begin(), std::find_if(vecProj.rbegin() + vecProj.size() / 2, vecProj.rend(),
		[rows](auto it) {return it > rows / 3; }).base());
	rColStart = std::distance(vecProj.begin(), std::find_if(vecProj.begin() + vecProj.size() / 2, vecProj.end(),
		[rows](auto it) {return it > rows / 3; }));

	//找到最左和最右边沿
	imgPro::img_projection(lineBinImg, projImg, vecProj, 1, 0,true);
	 vecProjMid.assign(vecProj.begin(), vecProj.begin() + 800);
	 vecProjMid.insert(vecProjMid.end(), vecProj.end() - 800, vecProj.end());
	std::sort(vecProjMid.begin(), vecProjMid.end(), greater<int>());
	meanVal = std::accumulate(vecProjMid.begin(), vecProjMid.begin() + vecProjMid.size() / 3, 0) / (vecProjMid.size() / 3);
	//std::cout<<meanVal << endl;
	for (int i = 0; i < lineBinImg.cols; i++)
		if (vecProj[i] >max(300, meanVal*2 )&& (  (leftColEnd-200 <i && i < leftColEnd)))
			vecVerEdgePtCol.push_back(i);
	if (vecVerEdgePtCol.empty())
		vecVerEdgePtCol.push_back(leftColEnd);

	for (int i = 0; i < lineBinImg.cols; i++)
		if (vecProj[i] >max(300, meanVal*2 )&& (  (rColStart  <i && i < rColStart+200)))
			vecVerEdgePtCol.push_back(i);
	if (vecVerEdgePtCol.size() ==1 )
		vecVerEdgePtCol.push_back(rColStart);

	int edgeUpRow = *std::min_element(vecHorEdgePtRow.begin(), vecHorEdgePtRow.begin()+ vecHorEdgePtRow.size() / 2);
	int edgeDwRow = *std::max_element(vecHorEdgePtRow.begin()+vecHorEdgePtRow.size()/2, vecHorEdgePtRow.end());
	int edgeLeftCol = *std::min_element(vecVerEdgePtCol.begin(), vecVerEdgePtCol.begin() + vecVerEdgePtCol.size() / 2);
	int edgeRightCol = *std::max_element(vecVerEdgePtCol.begin()+ vecVerEdgePtCol.size()/2, vecVerEdgePtCol.end());
	const int extenLen = 6;
	cv::Rect rect(edgeLeftCol-extenLen,edgeUpRow-extenLen,edgeRightCol-edgeLeftCol+extenLen*2,edgeDwRow - edgeUpRow+extenLen*2);
	static cv::Mat showImg = srcImg.clone();
	vector<cv::Point> cont{rect.tl(),cv::Point(rect.tl().x+rect.width,rect.tl().y),rect.br(),cv::Point(rect.tl().x,rect.tl().y + rect.height)};
	//cv::polylines(showImg,(vector<vector<cv::Point>>{cont}), true, cv::Scalar(0, 0, 255), 2);	

	for (auto it =lineSegs.begin();it != lineSegs.end();)//删除满足条件的线段
	{
		double angleWithHorLine, angleWithVerLine;
		imgPro::angle2Lines(cv::Point2f(horLineSeg[0], horLineSeg[1]), cv::Point2f(horLineSeg[2], horLineSeg[3]),
			cv::Point2f((*it)[0], (*it)[1]), cv::Point2f((*it)[2], (*it)[3]), angleWithHorLine);

		imgPro::angle2Lines(cv::Point2f(verLineSeg[0], verLineSeg[1]), cv::Point2f(verLineSeg[2], verLineSeg[3]),
			cv::Point2f((*it)[0], (*it)[1]), cv::Point2f((*it)[2], (*it)[3]), angleWithVerLine);

		cv::Point2f centPt(((*it)[0] + (*it)[2])/2, ((*it)[1] + (*it)[3])/2);
		//if (lineLabelImg.at<float>(centPt) == 264 || lineLabelImg.at<float>(centPt) == 417|| lineLabelImg.at<float>(centPt) == 413 )
		//{
		//	cout << "for debug" << endl;
		//}

		if (  rect.tl().x < centPt.x && centPt.x < rect.br().x &&  
			  rect.tl().y < centPt.y && centPt.y < rect.br().y) //pt inside
		{
			it = lineSegs.erase(it);
		}
		else if ( (abs(angleWithHorLine - 180) < angleDiffThresh || abs(angleWithHorLine) < angleDiffThresh) &&//水平
					rect.tl().y - distDiffThresh < centPt.y && centPt.y < rect.br().y+ distDiffThresh)
		{
			it = lineSegs.erase(it);
		}
		else if ((abs(angleWithVerLine - 180) < angleDiffThresh || abs(angleWithVerLine) < angleDiffThresh) &&//竖直
			rect.tl().x - distDiffThresh < centPt.x && centPt.x < rect.br().x + distDiffThresh)
		{
			it = lineSegs.erase(it);
		}
		else 
		{
			it++;
		}
	}

	return rect;
}



cv::Mat removeLineSeg(cv::Mat lineLableImg, vector<cv::Vec4f> &lineSegs);
//移除特定线段
//dSize=21 eSize=5
cv::Mat nearInvalidRegion(cv::Mat &invalidMask,int dSize,int eSize,bool isCorner)
{
	cv::Mat allScreenEdgeDImg, allScreenEdgeEImg, screenOuterEdgeRegionImg, edgeAboveCamImg, invalidMaskMaxImg;
	imgPro::biImg_getMaxArea(invalidMask, invalidMaskMaxImg);
	cv::morphologyEx(invalidMaskMaxImg, allScreenEdgeDImg, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(dSize, dSize)));//扩大图像边缘
	cv::morphologyEx(invalidMaskMaxImg, allScreenEdgeEImg, cv::MorphTypes::MORPH_ERODE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(eSize, eSize)));//缩小图像边缘
	screenOuterEdgeRegionImg = allScreenEdgeDImg - allScreenEdgeEImg;
	int patchLen = 800;
	cv::Rect tl(0, 0, patchLen, patchLen);
	cv::Rect tr(invalidMask.cols-patchLen-1, 0, patchLen, patchLen);
	cv::Rect dl(0, invalidMask.rows - patchLen - 1, patchLen, patchLen);
	cv::Rect dr(invalidMask.cols - patchLen - 1, invalidMask.rows - patchLen - 1, patchLen, patchLen);
	cv::Mat cornerMaskImg = cv::Mat::zeros(invalidMask.size(), CV_8UC1);
	if (isCorner)
	{
		cornerMaskImg(tl).setTo(255);
		cornerMaskImg(tr).setTo(255);
		cornerMaskImg(dl).setTo(255);
		cornerMaskImg(dr).setTo(255);
		cornerMaskImg.setTo(0, (invalidMask));
		return cornerMaskImg;
	}
	return screenOuterEdgeRegionImg;
}



cv::Mat getCirInScreenBorder(cv::Mat &screenBorderImg, cv::Mat mask, cv::Mat invalidPatchMask);
//
cv::Mat camScreenEdge(cv::Mat &srcImg,cv::Mat validMask,cv::Mat &edgeAboveCamiImg )
{
	const int LDist = 700;
	const int trDist = 600;
	const int morCirleL = 30;
	cv::Mat binLImg, binLImgTmp, binRImg, invalidMask, lineInvalidMaskImg, screenLEdgeImgMor, binLFilterImg, binLImgReve;
	cv::bitwise_not(validMask, invalidMask);
	cv::Mat phoneLImg = srcImg(cv::Range::all(), cv::Range(0, LDist));
	cv::Mat phoneLValidMaskImg = validMask(cv::Range::all(), cv::Range(0, LDist));
	cv::Mat phoneLInvalidMaskImg = invalidMask(cv::Range::all(), cv::Range(0, LDist));
	cv::Mat edgeNearScreenImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);  //去除线段的区域

	imgPro::grayImg_thresholdWithMask(phoneLImg, binLImgTmp, 30, 255, cv::THRESH_OTSU | THRESH_BINARY_INV, phoneLValidMaskImg);
	//被提取区域在左侧
	if (cv::countNonZero(binLImgTmp(cv::Range::all(),cv::Range(binLImgTmp.cols-1, binLImgTmp.cols))) > binLImgTmp.rows/3)
	{
		cv::bitwise_not(binLImgTmp, binLImg, phoneLValidMaskImg);
	}
	else
	{
		binLImg = binLImgTmp;
	}
	cv::Mat binLMaxImg, binRMaxImg, binLMorD1, binRMorD1, binLMorE1, binRMorE1, binLMorC1;;
	binLImg.setTo(0, phoneLInvalidMaskImg);
	imgPro::biImg_filterByArea(binLImg, binLFilterImg, 0, 200, 1);

	binLFilterImg.copyTo(edgeNearScreenImg(cv::Range::all(), cv::Range(0, LDist)));
	cv::morphologyEx(edgeNearScreenImg, binLMorC1, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));//修补图像边缘
	cv::morphologyEx(binLMorC1, binLMorD1, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));//扩大边缘
	//提取图像最左侧边缘轮廓，相机上方
	cv::Mat leftEdgeImg, binLImgMorImg, leftEdgeImgMor;
	cv::morphologyEx(binLFilterImg, binLImgMorImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(9, 9)));//修补图像边缘
	cv::bitwise_not(binLFilterImg, leftEdgeImg);
	cv::morphologyEx(leftEdgeImg, leftEdgeImgMor, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(9, 9)));//修补图像边缘
	leftEdgeImg = leftEdgeImgMor - binLImgMorImg;
	cv::Mat labelImg;
	int cnt = cv::connectedComponents(leftEdgeImg, labelImg, 4);
	labelImg.convertTo(labelImg, CV_8UC1);
	cv::inRange(labelImg, 1, 1, leftEdgeImg);
	edgeAboveCamiImg.create(srcImg.size(), CV_8UC1);
	edgeAboveCamiImg.setTo(0);
	leftEdgeImg.copyTo(edgeAboveCamiImg(cv::Range::all(), cv::Range(0, LDist)));
	//相机下方刘海边沿区域
	vector<cv::Point> vecLEdgePts, vecLEdgeShiftPts;
	imgPro::laser_fromImage(binLMorD1, vecLEdgePts, 3, 1, binLMorD1.rows - 1, false, true);
	cv::Mat screenLEdgeImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);
	int upScreenColPosX = (vecLEdgePts.begin() + vecLEdgePts.size() / 2)->x; //
			//清除两端折弯区域
	cv::Point rightPtUp, rightPtDw;
	int clipUp = 0, clipDw = vecLEdgePts.size() - 1;
	for (int pti = vecLEdgePts.size() / 2; pti > 0; pti--)
	{
		if (vecLEdgePts[pti].x > upScreenColPosX + 100)
		{
			clipUp = pti;
			rightPtUp = vecLEdgePts[pti];
			break;
		}
	}

	for (int pti = vecLEdgePts.size() / 2; pti < vecLEdgePts.size(); pti++)
	{
		if (vecLEdgePts[pti].x > upScreenColPosX + 100)
		{
			clipDw = pti;
			rightPtDw = vecLEdgePts[pti];
			break;
		}
	}
	vector<cv::Point2d> vecLEdgePts2{ vecLEdgePts.begin() + clipUp, vecLEdgePts.begin() + clipDw + 1 };
	for (int ptCnt = 0; ptCnt < vecLEdgePts2.size() - 1; ptCnt++)
	{
		cv::line(screenLEdgeImg, cv::Point(vecLEdgePts2[ptCnt]), cv::Point(vecLEdgePts2[ptCnt + 1]), cv::Scalar::all(255));
	}
	vector<cv::Point> vecNoZeroPts;
	cv::findNonZero(screenLEdgeImg, vecNoZeroPts);
	screenLEdgeImgMor = screenLEdgeImg.clone();
	for (auto pti : vecNoZeroPts)
	{
		cv::circle(screenLEdgeImgMor, pti, morCirleL, cv::Scalar(255), -1);
	}
	cv::line(screenLEdgeImgMor, rightPtUp, rightPtUp + cv::Point(morCirleL, 0), cv::Scalar::all(0));
	cv::line(screenLEdgeImgMor, rightPtDw, rightPtDw + cv::Point(morCirleL, 0), cv::Scalar::all(0));
	screenLEdgeImgMor = screenLEdgeImgMor - screenLEdgeImg;
	cv::Mat  screenEdgeResImg;
	 cnt = cv::connectedComponents(screenLEdgeImgMor, labelImg, 4);
	labelImg.convertTo(labelImg, CV_8UC1);
	cv::inRange(labelImg, 1, 1, screenEdgeResImg);
	//获取圆区域
	cv::Mat getCirMask,getCirMaskFillImg;
	imgPro::biImg_getMaxArea(binLFilterImg, getCirMask);
	cv::Mat cirImg = getCirInScreenBorder(phoneLImg, getCirMask, phoneLInvalidMaskImg);

	(screenEdgeResImg(cv::Range::all(), cv::Range(0, LDist))) = cirImg + (screenEdgeResImg(cv::Range::all(), cv::Range(0, LDist)));


	return screenEdgeResImg;
}
//删除特定线，角度等
cv::Mat removeLineSeg(cv::Mat lineLableImg, vector<cv::Vec4f>& lineSegs)
{
	for (auto ite=lineSegs.begin();ite != lineSegs.end();)
	{
		double segAngle = imgPro::angle_segX(*ite);
		if (abs(segAngle) <3.5 ||  180-3.5 < abs(segAngle))
		{
			ite =lineSegs.erase(ite);
		}
		else
			ite++;
	}

	return cv::Mat();
}

cv::Mat getCirInScreenBorder(cv::Mat &screenBorderImg, cv::Mat mask,cv::Mat invalidPatchMask)
{
	//屏幕圆区域
	cv::Mat cirImg, getCirMaskImg,fillImg,cirImg1, filterImg, maskMor1Img;
	cv::morphologyEx(mask, maskMor1Img, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(11, 11)));
	imgPro::biImg_fillup(maskMor1Img, fillImg, 1, 1, 100);
	cv::morphologyEx(fillImg, maskMor1Img, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(11, 11)));

	getCirMaskImg = maskMor1Img - invalidPatchMask;

		//imgPro::grayImg_thresholdWithMask(screenBorderImg, cirImg, 30, 255, cv::THRESH_OTSU | THRESH_BINARY_INV, getCirMaskImg);
	int meanVal = imgPro::grayImg_getMean(screenBorderImg, 3000, 20, 80, getCirMaskImg);
	cv::threshold(screenBorderImg, cirImg, meanVal + 3, 255, cv::THRESH_BINARY_INV);
	cv::morphologyEx(cirImg, cirImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));
	cirImg += invalidPatchMask;
	imgPro::biImg_filterByArea(cirImg, cirImg1, 0, 1000, 1);
	imgPro::biImg_delMaxArea(cirImg1, filterImg);

	cv::Mat cirImgErode, cirImgDilate;
	cv::morphologyEx(filterImg, cirImgDilate, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(11, 11)));
	cv::morphologyEx(filterImg, cirImgErode, cv::MorphTypes::MORPH_ERODE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(9, 9)));

	return cirImgDilate - cirImgErode;
}
//0-右上角 1-右下角
//返回:无关直线的区域
cv::Mat screenCornerEdge(cv::Mat &srcImg, cv::Mat validMask, cv::Rect roi)
{
	cv::Mat invalidMask;
	const int morCirle = 25;
	cv::bitwise_not(validMask, invalidMask);
	cv::Mat patchImg = srcImg(roi).clone();
	cv::Mat patchInvalidMaskImg = invalidMask(roi).clone();
	cv::Mat patchValidMaskImg = validMask(roi).clone();
	bool isTRPatch = false;
	int rectW = roi.width;
	if (roi.br().y < srcImg.rows/2) //top right corner
	{
		vector<cv::Point> vecAdd{ cv::Point(0,rectW/2),cv::Point(0,rectW-1),cv::Point(rectW/2,rectW-1) };
		vector<vector<cv::Point>> vvecPts{ vecAdd };
		cv::drawContours(patchInvalidMaskImg, vvecPts, -1, Scalar::all(255), -1);
		cv::drawContours(patchValidMaskImg, vvecPts, -1, Scalar::all(0), -1);
		isTRPatch = true;
	}
	else
	{
		vector<cv::Point> vecAdd{ cv::Point(0,0),cv::Point(0,rectW/2),cv::Point(rectW/2,0) };
		vector<vector<cv::Point>> vvecPts{ vecAdd };
		cv::drawContours(patchInvalidMaskImg, vvecPts, -1, Scalar::all(255), -1);
		cv::drawContours(patchValidMaskImg, vvecPts, -1, Scalar::all(0), -1);
	}
	cv::Mat patchBinImg,fillImg,binFilterImg, binMorC1, binMorD1;
	imgPro::grayImg_thresholdWithMask(patchImg, patchBinImg, 30, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV, patchValidMaskImg);
	if (cv::countNonZero(patchBinImg(cv::Range::all(),cv::Range(0,1))) >patchImg.rows/3 )
	{
		cv::bitwise_not(patchBinImg, patchBinImg, patchValidMaskImg);
	}
	patchBinImg.setTo(0, patchInvalidMaskImg);
	imgPro::biImg_fillup(patchBinImg, fillImg, 2, 0, 2000);
	imgPro::biImg_filterByArea(fillImg, binFilterImg, 0, 200, 1);
	cv::morphologyEx(binFilterImg, binMorC1, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));//修补图像边缘
	cv::morphologyEx(binMorC1, binMorD1, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(17, 17)));//扩大边缘
	vector<cv::Point> vecEdgePts, vecEdgePtsFilter;
	imgPro::laser_fromImage(binMorD1, vecEdgePts, 2, 1, binMorD1.rows - 1, false, true);
	cv::Mat patchShowImg = fillImg.clone();
	patchShowImg.setTo(0);
	cv::Mat screenEdgeImg = patchShowImg.clone();
	//极端点
	if (isTRPatch)
	{
		int pos1 = 0;
		for (auto ite=vecEdgePts.rbegin();ite != vecEdgePts.rend();)
		{
			if (ite->x - 0 < 0.1)
			{
				ite = vector<cv::Point>::reverse_iterator( vecEdgePts.erase((++ite).base()));
				pos1 = std::distance( vecEdgePts.begin(), ite.base())-1;
				break;
			}
			else
				ite++;
		}
		vecEdgePtsFilter.assign(vecEdgePts.begin() + pos1, vecEdgePts.end());
	}
	else
	{
		int posEnd = vecEdgePts.size()-1;
		for (auto ite = vecEdgePts.begin(); ite != vecEdgePts.end();)
		{
			if (ite->x - 0 < 0.1)
			{
				ite = vecEdgePts.erase(ite);
				posEnd = std::distance(vecEdgePts.begin(), ite);
				break;
			}
			else
				ite++;
		}
		vecEdgePtsFilter.assign(vecEdgePts.begin(), vecEdgePts.begin()+posEnd);
	}

	for (int ptCnt = 0; ptCnt < vecEdgePtsFilter.size() - 1; ptCnt++)
	{
		cv::line(screenEdgeImg, cv::Point(vecEdgePtsFilter[ptCnt]), cv::Point(vecEdgePtsFilter[ptCnt + 1]), cv::Scalar::all(255));
	}
	vector<cv::Point> vecNoZeroPts;
	cv::findNonZero(screenEdgeImg, vecNoZeroPts);
	for (auto pti : vecNoZeroPts)
		cv::circle(patchShowImg, pti, morCirle, cv::Scalar(255), -1);
	int areaIndex = 0;
	if (isTRPatch)
	{
		cv::line(patchShowImg, vecEdgePtsFilter[0], vecEdgePtsFilter[0] - cv::Point(morCirle, 0), cv::Scalar::all(0));
		cv::line(patchShowImg, vecEdgePtsFilter.back(), vecEdgePtsFilter.back() + cv::Point(0, morCirle), cv::Scalar::all(0));
		areaIndex = 1;
	}
	else
	{
		cv::line(patchShowImg, vecEdgePtsFilter[0], vecEdgePtsFilter[0] - cv::Point(0,morCirle), cv::Scalar::all(0));
		cv::line(patchShowImg, vecEdgePtsFilter.back(), vecEdgePtsFilter.back() - cv::Point(morCirle, 0), cv::Scalar::all(0));
		areaIndex = 2;
	}
	cv::Mat circleMorImg;
	circleMorImg = patchShowImg - screenEdgeImg;
	cv::Mat labelImg, screenEdgeResImg;
	int cnt = cv::connectedComponents(circleMorImg, labelImg, 4);
	labelImg.convertTo(labelImg, CV_8UC1);
	cv::inRange(labelImg, areaIndex, areaIndex, screenEdgeResImg);

	cv::Mat resImg =cv::Mat::zeros(srcImg.size(),CV_8UC1);
	screenEdgeResImg.copyTo(resImg(roi));
	return resImg;
}
bool screenBorderIsWhite(cv::Mat &srcImg)
{
	cv::Mat dstImg;
	cv::threshold(srcImg, dstImg, 180, 255, THRESH_BINARY);
	if (cv::countNonZero(dstImg) > srcImg.rows * 1300)
	{
		return true;
	}
	else
	{
		return false;
	}

}

double getHessianSigma(cv::Mat & src)
{

	return 0.0;
}

cv::Mat getHessianFlashDot(cv::Mat grayInImg)
{
	cv::Mat binIn,binIn2;
	cv::Mat labelImg,dotMorImg;
	cv::Mat dotBin,dotAndNearBin,thinImg, cirImg;
	cv::threshold(grayInImg, binIn, 220, 255, THRESH_BINARY);	
	cv::threshold(grayInImg, dotAndNearBin, 100, 255, THRESH_BINARY);

	morRebuild(dotAndNearBin, binIn, dotMorImg);


	map<string, vector<double>> feats;
	cv::morphologyEx(binIn, binIn2, cv::MorphTypes::MORPH_CLOSE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 2);

	cv::Mat  thin2;
	//cv::ximgproc::thinning(binIn2, thinImg,1);

	imgPro::biImg_createRegion(thinImg, labelImg, feats, 0, 10000);
	vector<int> vecCirLabel;
	for (int i = 0; i < feats["label"].size(); i++)
	{
		if ( feats["area"][i] < 12) 
		{
			vecCirLabel.push_back(feats["label"][i]);
		}
	}

	imgPro::region_toImg(labelImg, cirImg, vecCirLabel);
	morRebuild(binIn2, cirImg,dotBin);
	cv::morphologyEx(dotBin, dotBin, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1, -1), 1);


	return dotBin;
}

cv::Mat fixCornerArea(cv::Mat &toFixImg, int type);
cv::Mat netCamSeg(cv::Mat &grayIn, cv::Mat &inValidMask, cv::Rect roi);
//返回无效区域
//type：在黑色背景时候使用，A1->type=1,A2->type=2
cv::Mat phoneInvalidAreaL1(cv::Mat grayIn,int type)
{
	double threshVal = 0;
	cv::Mat mask;
	if (screenBorderIsWhite(grayIn))//白色边框
	{
		cv::Mat binImg, lightImg, closeImg, maxImg, fillImg, binMorImg2;
		cv::threshold(grayIn, binImg, 200, 255, THRESH_BINARY_INV); //有效区域
		cv::threshold(grayIn, lightImg, 210, 255, THRESH_BINARY); //高亮无效区域
		imgPro::biImg_getMaxArea(binImg, maxImg);
		cv::morphologyEx(lightImg, lightImg, cv::MorphTypes::MORPH_DILATE,
						cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));


		imgPro::biImg_fillup(maxImg, fillImg, 1, 0, 500);
		cv::morphologyEx(fillImg, fillImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(13, 13)));
		cv::morphologyEx(fillImg, fillImg, cv::MorphTypes::MORPH_ERODE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(5, 5)));

		 mask = 255 - fillImg;
		 mask = mask + lightImg;
	}
	else
	{
		cv::Mat binImg, distImg, closeImg, fillImg, maxImg, highAreaImg, highAreaImg2,binMorImg2, binMorImg, lightImg;
		threshVal = mean(grayIn(cv::Range(grayIn.rows / 3, grayIn.rows * 2 / 3), cv::Range(grayIn.cols / 3, grayIn.cols * 2 / 3)))[0];
		cv::threshold(grayIn, binImg, threshVal+20, 255, THRESH_BINARY); //无效区域

		cv::threshold(grayIn, lightImg, 210, 255, THRESH_BINARY); //高亮无效区域
		cv::morphologyEx(lightImg, lightImg, cv::MorphTypes::MORPH_DILATE,
			cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));

		cv::Mat screenImg, extImg, lightMaxImg;
		//cv::bitwise_not(binImg, screenImg);
		//imgPro::biImg_getMaxArea(binImg, maxImg);
		imgPro::biImg_filterByArea(binImg, maxImg, binImg.rows * 3, INT_MAX, 2);
		cv::morphologyEx(maxImg, binMorImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(25, 25)));
		cv::morphologyEx(binMorImg, binMorImg2, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(5, 5)));
		cv::Mat binRevImg = 255 - binMorImg2;
		imgPro::biImg_delMaxArea(binRevImg, extImg);
		binMorImg2 += extImg;
		cv::Mat fixCornerImg = fixCornerArea(binMorImg2, type);

		//扬声器网和前置相机
		cv::Rect roi(0, grayIn.rows / 3, 600, grayIn.rows / 3);
		cv::Mat netCamImg = netCamSeg(grayIn, fixCornerImg,roi);
		cv::Mat canvs = cv::Mat::zeros(binMorImg2.size(), CV_8UC1);
		netCamImg.copyTo(canvs(roi));
		mask = canvs + fixCornerImg;
		mask += lightImg;
	}
	return mask;
}

//返回L2的有效计算区域
cv::Mat phoneValidAreaL2(cv::Mat grayIn, cv::Mat &L1InvalidMaskImg)
{
	cv::Mat binImg, lightImg;
	cv::threshold(grayIn, lightImg, 150, 255, THRESH_BINARY); //高亮无效区域
	//防止最外边缘反光，如iphone5
	lightImg(cv::Range::all(), cv::Range(0, 30)).setTo(0);
	lightImg(cv::Range::all(), cv::Range(lightImg.cols-30, lightImg.cols)).setTo(0);
	cv::morphologyEx(lightImg, lightImg, cv::MorphTypes::MORPH_DILATE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));

	int	threshVal = mean(grayIn(cv::Range(grayIn.rows / 3, grayIn.rows * 2 / 3), cv::Range(grayIn.cols / 3, grayIn.cols * 2 / 3)))[0];
	cv::threshold(grayIn, binImg, threshVal - 20, 255, THRESH_BINARY); //白色区域
	cv::resize(L1InvalidMaskImg, L1InvalidMaskImg, grayIn.size());
	cv::bitwise_and(L1InvalidMaskImg, binImg, binImg);
	cv::morphologyEx(binImg, binImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(11, 3)));
	cv::Mat maxImg,closeImg,erodeImg, fillImg,smallInvalidArea;
	binImg -= lightImg;
	imgPro::biImg_getMaxArea(binImg, maxImg);
	imgPro::biImg_fillup(binImg, fillImg, 2,1, L1InvalidMaskImg.cols*50);
	smallInvalidArea = fillImg - binImg;
	cv::morphologyEx(smallInvalidArea, smallInvalidArea, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(5, 5)));
	maxImg = maxImg  - smallInvalidArea;
	cv::morphologyEx(maxImg, closeImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));
	cv::morphologyEx(closeImg, erodeImg, cv::MorphTypes::MORPH_ERODE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(7, 7)));

	//去除外边沿
	const int radius = 7;
	vector<vector<cv::Point>> vvecPt;
	cv::findContours(erodeImg, vvecPt, cv::RetrievalModes::RETR_EXTERNAL, cv::ContourApproximationModes::CHAIN_APPROX_NONE);
	cv::Mat canvs = cv::Mat::zeros(erodeImg.size(), CV_8UC1);
	for (auto pt:vvecPt[0])
	{
		cv::circle(canvs, pt, radius, cv::Scalar::all(255), -1);
	}
	canvs(cv::Range::all(), cv::Range(500, canvs.cols - 500)).setTo(0);
	erodeImg -= canvs;

	return erodeImg;
}

cv::Mat netCamSeg(cv::Mat &grayIn,cv::Mat &inValidMaskImg,cv::Rect roi )
{
	//切割出扬声器网和前置相机
	cv::Mat patchBinImg, closeImg, highAreaImg;
	cv::Mat patch = grayIn(roi).clone();
	cv::Mat maxImg;

	patch.setTo(0, inValidMaskImg(roi));
	cv::threshold(patch, patchBinImg, 100, 255, THRESH_BINARY);  //不需要的区域
	cv::morphologyEx(patchBinImg, closeImg, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));	
	imgPro::biImg_filterByArea(closeImg, highAreaImg, 500, INT_MAX, 2);

	return highAreaImg;
}

//type: 与扫描方向有关
cv::Mat getSplitLineA1(cv::Mat &img, cv::Rect roi, int upOrDw, int LorR);
cv::Mat getSplitLineA2(cv::Mat &img, cv::Rect roi, int upOrDw, int LorR);
cv::Mat fixCornerArea(cv::Mat &toFixImg, int type)
{
	//修补4个角的缺漏,使用lightImg
	cv::Rect tlPart(0, 0, 600, 600);
	cv::Mat notImg = 255 - toFixImg;
	cv::Mat notMaxImg;
	cv::Mat resFixImg = toFixImg.clone();
	imgPro::biImg_getMaxArea(notImg, notMaxImg);
	cv::morphologyEx(notMaxImg, notMaxImg, cv::MorphTypes::MORPH_CLOSE,
		cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(15, 15)));
	//A1图从上向下扫描
	int scanLen = 650;
	if (type == 1) //A1图
	{
		//左上角
		cv::Rect tlPart(0, 0, scanLen*2, scanLen / 2);
		cv::Rect dlPart(0, notMaxImg.rows-scanLen/2-1, scanLen*2, scanLen / 2);
		cv::Rect trPart(notMaxImg.cols- scanLen*2-1, 0, scanLen*2, scanLen / 2);
		cv::Rect drPart(notMaxImg.cols - scanLen*2 - 1, notMaxImg.rows - scanLen / 2 - 1, scanLen*2, scanLen / 2);
		cv::Mat trPartImg = getSplitLineA1(notMaxImg, trPart, 1,2);
		cv::Mat drPartImg = getSplitLineA1(notMaxImg, drPart, 2,2);
		cv::Mat tlPartImg = getSplitLineA1(notMaxImg, tlPart, 1,1);
		cv::Mat dlPartImg = getSplitLineA1(notMaxImg, dlPart, 2,1);

		resFixImg(tlPart) += tlPartImg ;
		resFixImg(trPart) += trPartImg ;
		resFixImg(dlPart) += dlPartImg ;
		resFixImg(drPart) += drPartImg ;
	}
	else
	{
		int width = scanLen / 2;
		int height = scanLen * 2;
		cv::Rect tlPart(0, 0, width, height);
		cv::Rect dlPart(0, notMaxImg.rows - height - 1, width, height);
		cv::Rect trPart(notMaxImg.cols - scanLen/2 - 1, 0, width, height);
		cv::Rect drPart(notMaxImg.cols - scanLen/2 - 1, notMaxImg.rows - height - 1, width, height);
		cv::Mat trPartImg = getSplitLineA2(notMaxImg, trPart, 1, 2);
		cv::Mat tlPartImg = getSplitLineA2(notMaxImg, tlPart, 1, 1);
		cv::Mat dlPartImg = getSplitLineA2(notMaxImg, dlPart, 2, 1);
		cv::Mat drPartImg = getSplitLineA2(notMaxImg, drPart, 2, 2);
		resFixImg(tlPart) += tlPartImg;
		resFixImg(trPart) += trPartImg;
		resFixImg(dlPart) += dlPartImg;
		resFixImg(drPart) += drPartImg;
	}
	return resFixImg;
}

//A1图左侧切割线，upOrDw=1 ,上。 upOrDw=2，下。LorR=1，左
cv::Mat getSplitLineA1(cv::Mat &img, cv::Rect roi, int upOrDw,int LorR)
{
	vector<cv::Point> vecPt, vecPt2Fit;
	cv::Mat tlImg = img(roi);
	if (upOrDw == 1)
	{
		imgPro::laser_fromImage(tlImg, vecPt, 0, 0, roi.width , false, false);
	} 
	else
	{
		imgPro::laser_fromImage(tlImg, vecPt, 1, 0, roi.width , false, false);
	}
	cv::Mat tlCanv = cv::Mat::zeros(tlImg.size(), CV_8UC1);
	imgPro::img_drawLaser(tlCanv, tlCanv, vecPt, 1);
	cv::cvtColor(tlCanv, tlCanv, cv::COLOR_BGR2GRAY);
	cv::Mat projImg, noMaxPartImg;
	vector<int> vecProj;
	imgPro::img_projection(tlCanv, projImg, vecProj, 1, 0, 1);
	int maxVal = 0;
	int maxLoc = 0;
	for (int i = 1; i < vecProj.size() - 1; i++)
	{
		if (vecProj[i - 1] + vecProj[i] + vecProj[i + 1] > maxVal)
		{
			maxVal = vecProj[i - 1] + vecProj[i] + vecProj[i + 1];
			maxLoc = i;
		}
	}
	if (LorR == 1)
	{
		maxLoc -= 2;
		vecPt2Fit.assign(vecPt.begin()+vecPt.size()/2, vecPt.end());
	}
	else
	{
		maxLoc += 2;
		vecPt2Fit.assign(vecPt.begin(), vecPt.end()- vecPt.size() / 2);
	}
	cv::cvtColor(tlCanv, tlCanv, cv::COLOR_GRAY2BGR);
	cv::drawMarker(tlCanv, vecPt[maxLoc], Scalar(0, 0, 255),cv::MarkerTypes::MARKER_CROSS,5);
	cv::Vec4f lineFitted;
	//拟合直线点，防止由防撕标干扰,A2 fit竖直直线
	vector<int> vecPosY(img.rows, 0);
	for (int i = 0; i < vecPt2Fit.size(); i++)
		if (vecPt2Fit[i].y > 0)
		{
			vecPosY[vecPt2Fit[i].y]++;
		}
	int maxY = std::distance(vecPosY.begin(), std::max_element(vecPosY.begin(), vecPosY.end()));
	vecPt2Fit.erase(remove_if(vecPt2Fit.begin(), vecPt2Fit.end(), [maxY](auto pt) {return abs(pt.y - maxY) > 3; }), vecPt2Fit.end());

	cv::fitLine(vecPt2Fit, lineFitted, DIST_L1, 0, 0.01, 0.01);
	cv::Vec4f line =  imgPro::img_drawLine(tlCanv, tlCanv, lineFitted, cv::Scalar(0, 0, 255), 1);
	cv::Point edgePt;
	if (LorR == 1)
	{
		for (int i = maxLoc;i<vecPt.size();i++)
		{
			cv::Point nearstPt;
			if (imgPro::closestPt2Line(vecPt[i], line,nearstPt) <=2)//与直线一定差距的点
			{
				edgePt = vecPt[i];
				break;
			}
		}
	} 
	else
	{
		for (int i = maxLoc; i > 0; i--)
		{
			cv::Point nearstPt;
			if (imgPro::closestPt2Line(vecPt[i], line, nearstPt) <= 2)//与直线一定差距的点
			{
				edgePt = vecPt[i];
				break;
			}
		}
	}	
	cv::drawMarker(tlCanv, edgePt, Scalar(0, 0, 255), cv::MarkerTypes::MARKER_CROSS, 5);
	if (tlImg.channels() > 1)
		cv::cvtColor(tlImg, tlImg, cv::COLOR_BGR2GRAY);
	cv::line(tlImg, vecPt[maxLoc], edgePt, Scalar::all(0), 3);
	imgPro::biImg_delMaxArea(tlImg, noMaxPartImg);
	cv::dilate(noMaxPartImg, noMaxPartImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(5, 5)));
	return noMaxPartImg;
}

cv::Mat getSplitLineA2(cv::Mat &img, cv::Rect roi, int upOrDw,int LorR)
{
	vector<cv::Point> vecPt, vecPt2Fit;
	cv::Mat tlImg = img(roi);
	if (LorR == 1)
	{
		imgPro::laser_fromImage(tlImg, vecPt, 2, 0, roi.height , false, false);
		//防止被标签截断
	} 
	else
	{
		imgPro::laser_fromImage(tlImg, vecPt, 3, 0, roi.height , false, false);
		bool startSet = false;
		if (upOrDw ==1)
		{
			for (int r = 0; r < tlImg.rows; r++)
				if (tlImg.at<uchar>(r, 0) == 255 || startSet)
				{
					tlImg.at<uchar>(r, 0) = 255;
					startSet = true;
				}
		}
		else
		{
			for (int r = tlImg.rows-1; r >=0 ; r--)
				if (tlImg.at<uchar>(r, 0) == 255 || startSet)
				{
					tlImg.at<uchar>(r, 0) = 255;
					startSet = true;
				}
		}
	}
	cv::Mat tlCanv = cv::Mat::zeros(tlImg.size(), CV_8UC1);
	imgPro::img_drawLaser(tlCanv, tlCanv, vecPt, 1);
	cv::cvtColor(tlCanv, tlCanv, cv::COLOR_BGR2GRAY);
	cv::Mat projImg, noMaxPartImg;
	vector<int> vecProj;
	imgPro::img_projection(tlCanv, projImg, vecProj, 0, 0, 1);
	int maxVal = 0;
	int maxLoc = 0;
	if (upOrDw == 1)
	{
		for (int i = 1; i < vecProj.size() / 2 - 1; i++)
		{
			if (vecProj[i - 1] + vecProj[i] + vecProj[i + 1] > maxVal)
			{
				maxVal = vecProj[i - 1] + vecProj[i] + vecProj[i + 1];
				maxLoc = i;
			}
		}
		maxLoc -= 2;
		vecPt2Fit.assign(vecPt.begin() + vecPt.size() / 2, vecPt.end());
	}
	else
	{
		for (int i = vecProj.size() / 2 - 1; i < vecProj.size()-1 ; i++)
		{
			if (vecProj[i - 1] + vecProj[i] + vecProj[i + 1] > maxVal)
			{
				maxVal = vecProj[i - 1] + vecProj[i] + vecProj[i + 1];
				maxLoc = i;
			}
		}
		maxLoc += 2;
		vecPt2Fit.assign(vecPt.begin(), vecPt.end() - vecPt.size() / 2);
	}

	cv::cvtColor(tlCanv, tlCanv, cv::COLOR_GRAY2BGR);
	cv::drawMarker(tlCanv, vecPt[maxLoc], Scalar(0, 0, 255), cv::MarkerTypes::MARKER_CROSS, 5);
	cv::Vec4f lineFitted;
	//拟合直线点，防止由防撕标干扰,A2 fit竖直直线
	vector<int> vecPosX(img.cols,0);
	for (int i=0;i<vecPt2Fit.size();i++)
		if (vecPt2Fit[i].x >0)
		{
			vecPosX[vecPt2Fit[i].x]++;
		}
	int maxX =std::distance(vecPosX.begin(), std::max_element(vecPosX.begin(), vecPosX.end()));
	vecPt2Fit.erase(remove_if(vecPt2Fit.begin(), vecPt2Fit.end(), [maxX](auto pt) {return abs(pt.x - maxX) > 3; }), vecPt2Fit.end());

	cv::fitLine(vecPt2Fit, lineFitted, DIST_L1, 0, 0.01, 0.01);
	cv::Vec4f line = imgPro::img_drawLine(tlCanv, tlCanv, lineFitted, cv::Scalar(0, 0, 255), 1);
	cv::Point edgePt;
	if (upOrDw == 1)
	{
		for (int i = maxLoc; i < vecPt.size(); i++)
		{
			cv::Point nearstPt;
			if (imgPro::closestPt2Line(vecPt[i], line, nearstPt) <= 2)//与直线一定差距的点
			{
				edgePt = vecPt[i];
				break;
			}
		}
	}
	else
	{
		for (int i = maxLoc; i > 0; i--)
		{
			cv::Point nearstPt;
			if (imgPro::closestPt2Line(vecPt[i], line, nearstPt) <= 2)//与直线一定差距的点
			{
				edgePt = vecPt[i];
				break;
			}
		}
	}
	cv::drawMarker(tlCanv, edgePt, Scalar(0, 0, 255), cv::MarkerTypes::MARKER_CROSS, 5);
	cv::line(tlImg, vecPt[maxLoc], edgePt, Scalar::all(0), 3);
	if (tlImg.channels()>1)
		cv::cvtColor(tlImg, tlImg, cv::COLOR_BGR2GRAY);
	imgPro::biImg_delMaxArea(tlImg, noMaxPartImg);
	cv::dilate(noMaxPartImg, noMaxPartImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(5, 5)));
	return noMaxPartImg;
}


cv::Mat removeSpot(cv::Mat binImg,cv::Mat &dotImg)
{
	cv::Mat labelImg, labelImg2, closeImg, distThresImg;
	cv::Mat distImg, distThinImg, distImg32, dilateImg, filterImg;
	map<string, vector<double>> feats, feats2;
	imgPro::biImg_filterByArea(binImg, filterImg, 8, 300, 2);

	cv::distanceTransform(filterImg, distImg32, DIST_L1, 3);
	distImg32.convertTo(distImg, CV_8UC1);
	cv::threshold(distImg, distThresImg, 1, 255, THRESH_BINARY);  //会破坏线的结构
	cv::dilate(distThresImg, dilateImg, cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3)));
	imgPro::biImg_createRegion(dilateImg, labelImg, feats, 10, dilateImg.total());//hw <4
	vector<int> vecLabelKeep, vecLabelCir, vecLabelRect;
	map< int, cv::Point> mapIndexCentPt;//记录中心点和对应label

	for (int i = 0; i < feats["label"].size(); i++)
	{
		cv::RotatedRect rotRect;
		if ((12 < feats["area"][i] && feats["circle"][i] > 0.55)) //潜在原型亮点/擦痕区域
		{
			mapIndexCentPt[feats["label"][i]] = cv::Point(feats["col"][i], feats["row"][i]);
		}
		else if (13 < feats["area"][i] && feats["rect"][i] > 0.7  &&feats["rect"][i] < 0.9  && feats["rotWidth"][i] < 15 && feats["rotHWRatio"][i] < 2)//加凸限制
		{
			mapIndexCentPt[feats["label"][i]] = cv::Point(feats["col"][i], feats["row"][i]);
		}
		else
		{
			//vecLabelKeep.push_back(feats["label"][i]);
		}
	}
	imgPro::biImg_createRegion(filterImg, labelImg2, feats2, 10, dilateImg.total());
	map<int, vector<cv::Point>> mapIndexPts, mapIndexPtsRemove;
	vector<vector<cv::Point>> vvecPts;
	for (int r = 0; r < labelImg2.rows; r++)
	{
		for (int c = 0; c < labelImg2.cols; c++)
		{
			mapIndexPts[labelImg2.at<int>(r, c)].push_back(cv::Point(c, r));
		}
	}

	vector<int> labelToRemove;
	vector<int> vecLabelInt(feats["label"].begin(), feats["label"].end());
	vector<int> vecLabel2Int(feats2["label"].begin(), feats2["label"].end());
	cv::Mat removeImg2;
	for (auto it = mapIndexCentPt.begin(); it != mapIndexCentPt.end(); it++)
	{
		int labelInSpotImg = it->first;
		int labelInRawImg = labelImg2.at<int>(it->second);
		//分别找到两个面积，比较
		auto ite = find(vecLabelInt.begin(), vecLabelInt.end(), labelInSpotImg);
		int index1 = distance(vecLabelInt.begin(), ite);
		double rArea = feats["area"][index1];

		ite = find(vecLabel2Int.begin(), vecLabel2Int.end(), labelInRawImg);
		int index2 = distance(vecLabel2Int.begin(), ite);
		double rArea2 = feats2["area"][index2];
		if (rArea2 / rArea < 1.5)
		{
			//labelToRemove.push_back(feats2["label"][index2]);   //
			vvecPts.push_back(mapIndexPts[feats2["label"][index2]]);
		}
	}
	clock_t t1 = clock();
	imgPro::region_toImg(labelImg2, removeImg2, vvecPts);
	clock_t t2 = clock();
	//std::cout << "region2img  cost : " << (double)(t2 - t1) / CLOCKS_PER_SEC << "s" << endl;
	dotImg = removeImg2;
	return binImg - removeImg2;
}
cv::Mat removeSpotL2(cv::Mat binImg,cv::Mat &dotImg)
{
	cv::Mat labelImg, labelImg2, closeImg, distThresImg;
	cv::Mat distImg, distThinImg, distImg32, dilateImg, filterImg;
	map<string, vector<double>> feats, feats2;
	imgPro::biImg_filterByArea(binImg, filterImg, 4, 300, 2);

	cv::distanceTransform(filterImg, distImg32, DIST_L1, 3);
	distImg32.convertTo(distImg, CV_8UC1);
	cv::threshold(distImg, distThresImg, 1, 255, THRESH_BINARY);  //会破坏线的结构
	cv::dilate(distThresImg, dilateImg, cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3)));
	imgPro::biImg_createRegion(dilateImg, labelImg, feats, 4, dilateImg.total());//hw <4
	vector<int> vecLabelKeep, vecLabelCir, vecLabelRect;
	map< int, cv::Point> mapIndexCentPt;//记录中心点和对应label

	for (int i = 0; i < feats["label"].size(); i++)
	{
		cv::RotatedRect rotRect;
		if ((4 < feats["area"][i] && feats["circle"][i] > 0.55)) //潜在原型亮点/擦痕区域
		{
			mapIndexCentPt[feats["label"][i]] = cv::Point(feats["col"][i], feats["row"][i]);
		}
		else if (8 < feats["area"][i] && feats["rect"][i] > 0.7  &&feats["rect"][i] < 0.9  && feats["rotWidth"][i] < 15 && feats["rotHWRatio"][i] < 2)//加凸限制
		{
			mapIndexCentPt[feats["label"][i]] = cv::Point(feats["col"][i], feats["row"][i]);
		}
		else
		{
			//vecLabelKeep.push_back(feats["label"][i]);
		}
	}
	imgPro::biImg_createRegion(filterImg, labelImg2, feats2, 4, dilateImg.total());
	map<int, vector<cv::Point>> mapIndexPts, mapIndexPtsRemove;
	vector<vector<cv::Point>> vvecPts;
	for (int r = 0; r < labelImg2.rows; r++)
	{
		for (int c = 0; c < labelImg2.cols; c++)
		{
			mapIndexPts[labelImg2.at<int>(r, c)].push_back(cv::Point(c, r));
		}
	}

	vector<int> labelToRemove;
	vector<int> vecLabelInt(feats["label"].begin(), feats["label"].end());
	vector<int> vecLabel2Int(feats2["label"].begin(), feats2["label"].end());
	cv::Mat removeImg2;
	for (auto it = mapIndexCentPt.begin(); it != mapIndexCentPt.end(); it++)
	{
		int labelInSpotImg = it->first;
		int labelInRawImg = labelImg2.at<int>(it->second);
		//分别找到两个面积，比较
		auto ite = find(vecLabelInt.begin(), vecLabelInt.end(), labelInSpotImg);
		int index1 = distance(vecLabelInt.begin(), ite);
		double rArea = feats["area"][index1];

		ite = find(vecLabel2Int.begin(), vecLabel2Int.end(), labelInRawImg);
		int index2 = distance(vecLabel2Int.begin(), ite);
		double rArea2 = feats2["area"][index2];
		if (rArea2 / rArea < 1.5/* || (rArea < 10 && rArea2 <)*/)
		{
			//labelToRemove.push_back(feats2["label"][index2]);   //
			vvecPts.push_back(mapIndexPts[feats2["label"][index2]]);
		}
	}
	clock_t t1 = clock();
	imgPro::region_toImg(labelImg2, removeImg2, vvecPts);
	clock_t t2 = clock();
	//std::cout << "region2img  cost : " << (double)(t2 - t1) / CLOCKS_PER_SEC << "s" << endl;
	dotImg = removeImg2;
	return binImg - removeImg2;
}


vector<cv::Rect> rect_merge(cv::Size imgSize, vector<cv::Rect> vecRect,int extLen, int maxLenThresh)
{
	vector<cv::Rect> vecResRect, vecBigRect;
	cv::Mat canvs = cv::Mat::zeros(imgSize, CV_8UC1);
	for (auto r : vecRect)
	{
		if (max(r.size().height, r.size().width) < maxLenThresh)
		{
			cv::Rect extRect = imgPro::rect_enlarge(imgSize, r, extLen);
			cv::rectangle(canvs, extRect, Scalar::all(255), -1);
		}
		else
		{
			vecBigRect.push_back(r);
		}
	}
	cv::Mat mor1, mor2, mor3;
	//cv::morphologyEx(canvs, mor1, cv::MorphTypes::MORPH_CLOSE, cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7)));
	mor1 = canvs;
	cv::Mat labelImg;
	map<string, vector<double>> feats;
	imgPro::biImg_createRegion(mor1, labelImg, feats, 5, mor1.total());
	int offset = 0;
	for (int i = 0; i < feats["label"].size(); i++)
	{
		cv::Rect rect(feats["left"][i] - offset, (feats["top"][i] - offset), (feats["width"][i] + 2 * offset), (feats["height"][i] + 2 * offset));
		vecResRect.push_back(rect);
	}
	vecResRect.insert(vecResRect.end(), vecBigRect.begin(), vecBigRect.end());
	return vecResRect;
}

int removeIntersec4Nei(cv::Mat &thinImg, vector<cv::Point> vecInterPt, cv::Mat &outImg);
//移除水平线条，和屏幕边框得矩形边缘（如果有）
//int removeHorLine(cv::Mat &srcImg, cv::Mat &outImg)
//{
//	cv::Mat binImg, thinImg, thinImg2, xorImg;
//	cv::threshold(srcImg, binImg, 11, 255, cv::THRESH_BINARY);
//	cv::Mat horImg = binImg.clone();
//	//切分线条
//	clock_t c1 = clock();
//	//cv::ximgproc::thinning(binImg,thinImg2, cv::ximgproc::THINNING_GUOHALL);
//	thinning_block(binImg, thinImg, cv::ximgproc::THINNING_GUOHALL);
//	clock_t c2 = clock();
//	//std::cout << " thin cost : " << (double)(c2 - c1) / CLOCKS_PER_SEC << "s" << endl;
//	//cv::bitwise_xor(thinImg2, thinImg, xorImg);
//	//vector<cv::Point> vecPt;
//	//cv::findNonZero(xorImg, vecPt);
//	vector<cv::Point> intersecPts;
//	skeleton_intersecPoint(thinImg, intersecPts);
//
//
//	cv::Mat showImg = thinImg.clone();
//	cv::cvtColor(showImg, showImg, cv::COLOR_GRAY2BGR);
//	for (auto pt:intersecPts)
//	{
//		showImg.at<cv::Vec3b>(pt) = cv::Vec3b(0,0,255);
//	}
//	cv::morphologyEx(horImg,horImg, cv::MorphTypes::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,1)));
//	int horizontal_size = 9;
//	cv::Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
//	cv::erode(horImg, horImg, horizontalStructure, Point(-1, -1));
//	cv::dilate(horImg, horImg, horizontalStructure, Point(-1, -1));
//	cv::Mat splitImg, interImg;;
//	removeIntersec4Nei(thinImg, intersecPts, splitImg);
//
//	cv::Mat splitHorImg;
//	cv::bitwise_and(splitImg, horImg, splitHorImg);
//	cv::Mat labelImg;
//	map<string, vector<double>> mapFeats;
//	vector<int> vecInvalidLabel;
//	int cnt = imgPro::biImg_createRegion(splitHorImg, labelImg, mapFeats, 2, INT_MAX);
//
//	map<int, vector<cv::Point>> mapLabelVecPt;
//	for (int r = 0; r < labelImg.rows; r++)
//	{
//		for (int c = 0; c < labelImg.cols; c++)
//		{
//			int labelVal = labelImg.at<int>(r, c);
//			if (labelVal > 0)
//			{
//				mapLabelVecPt[labelVal].emplace_back(c, r);
//			}
//		}
//	}
//	
//	vector<vector<cv::Point>> vvecPt;
//	for (int i=0;i<mapFeats["label"].size();i++)
//	{
//		//cout << i << endl;
//		if (abs(mapFeats["rotAngle"][i]) < 4 || abs(mapFeats["rotAngle"][i] -180) < 4)
//		{
//			cv::Rect roi(mapFeats["left"][i], mapFeats["top"][i], mapFeats["width"][i], mapFeats["height"][i]);
//			double mVal = cv::mean(srcImg(roi), thinImg(roi))[0];
//			if (mVal < 41  )
//			{
//				vvecPt.push_back(mapLabelVecPt[mapFeats["label"][i]]);
//			}
//		}
//	}
//
//	cv::Mat toRemoveThinImg;
//	imgPro::region_toImg(labelImg, toRemoveThinImg, vvecPt);
//	cv::morphologyEx(toRemoveThinImg, toRemoveThinImg, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3)));
//	interImg = thinImg - splitImg;
//	cv::morphologyEx(interImg, interImg, cv::MorphTypes::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
//	toRemoveThinImg -= interImg;
//
//	outImg = srcImg - toRemoveThinImg;
//
//	return 0;
//}
//


//去除交点，及交点8领域区域内，存在8领域连接得点
int removeIntersec4Nei(cv::Mat &thinImg, vector<cv::Point> vecInterPt,cv::Mat &outImg)
{
	outImg = thinImg.clone();
	for (auto pt:vecInterPt)
	{
		//四联通区域内有两两相连
		outImg.at<uchar>(pt) = 0;
		if (thinImg.at<uchar>(pt.y - 1, pt.x) > 0 || thinImg.at<uchar>(pt.y+1,pt.x) >0 &&
			(thinImg.at<uchar>(pt.y, pt.x + 1) >0|| thinImg.at<uchar>(pt.y, pt.x - 1)>0))
		{
			outImg.at<uchar>(pt.y, pt.x + 1) = 0;
			outImg.at<uchar>(pt.y, pt.x - 1) = 0;
		}
	}
	return 0;
}



cv::RotatedRect rotRectFromSeg(vector<cv::Vec4f> &vecSeg)
{
	vector<cv::Point> vecPt;
	for (auto L1 : vecSeg)
	{
		cv::Point p1(L1[0], L1[1]);
		cv::Point p2(L1[2], L1[3]);
		vecPt.push_back(p1);
		vecPt.push_back(p2);
	}
	cv::RotatedRect rot = cv::minAreaRect(vecPt);
	return rot;
}

double getSigmaL1(cv::Mat &src)
{
	vector<int> vecGrayVal(256);
	vector<int> vecDistri;
	for (int r = src.rows / 3; r < src.rows * 2 / 3; r++)
		for (int c = src.cols / 3; c < src.cols * 2 / 3; c++)
			vecGrayVal[src.at<uchar>(r, c)] ++;

	for (auto it : vecGrayVal)
		if (it > 10000)
			vecDistri.push_back(it);

	//归一化
	int maxV = *std::max_element(vecDistri.begin(), vecDistri.end());
	vector<double> vecNormDistri;
	for (auto it : vecGrayVal)
	{
		vecNormDistri.push_back(it / double(maxV));
	}

	double sum = std::accumulate(vecNormDistri.begin(), vecNormDistri.end(), 0.);
	double mVal = sum / vecNormDistri.size();
	double accum = 0.;
	std::for_each(vecNormDistri.begin(), vecNormDistri.end(), [&](const double d) {
		accum += (d - mVal)*(d - mVal); });

	double stdev = sqrt(accum / (vecNormDistri.size() - 1));
	cout << "stdev = " << stdev << endl;
	//拟合直线方程
	double fitA, fitB, sigma;
	fitA = 1.34;
	fitB = 1.108;
	sigma = stdev * fitA + fitB;
	sigma = min(sigma, 1.24);
	//cout << "sigma = " << sigma << endl;
	return sigma;
}


//0.221367->1.27  ,0.251177->1.26,0.24-->1.26
double getSigmaL2(cv::Mat raw, cv::Mat &validMaskImg)
{
	vector<int> vecGrayVal(256);
	vector<int> vecDistri;
	vector<int> vecProj;
	cv::Mat projImg;
	int rows = validMaskImg.rows;
	imgPro::img_projection(validMaskImg, projImg, vecProj, 1, 0, true);
	int leftColStart=0, leftColEnd=0, rColStart=0, rColEnd=0;
	leftColStart = std::distance(vecProj.begin(), std::find_if(vecProj.begin(), vecProj.end(), [](auto it) {return it > 100; }));
	leftColEnd = std::distance(vecProj.begin(), std::find_if(vecProj.rbegin() + vecProj.size() / 2, vecProj.rend(),
		[rows](auto it) {return it > rows / 3; }).base());

	rColStart = std::distance(vecProj.begin(), std::find_if(vecProj.begin() + vecProj.size() / 2, vecProj.end(),
		[rows](auto it) {return it > rows / 3; }));
	rColEnd = std::distance(vecProj.begin(), std::find_if(vecProj.rbegin(), vecProj.rend(),
		[](auto it) {return it > 100; }).base());

	for (int r = 0; r < raw.rows; r++)
		for (int c = leftColStart; c < rColEnd; c++)
		{
			if ( leftColEnd -50 < c && c < rColStart+50)
				continue;
			else
			{
				if (validMaskImg.at<uchar>(r, c) > 0)
				{
					vecGrayVal[raw.at<uchar>(r, c)] ++;
				}
			}
		}

	for (auto it : vecGrayVal)
		if (it > 100000)
			vecDistri.push_back(it);

	//归一化
	int maxV = *std::max_element(vecDistri.begin(), vecDistri.end());
	vector<double> vecNormDistri;
	for (auto it : vecDistri)
	{
		vecNormDistri.push_back(it / double(maxV));
	}

	double sum = std::accumulate(vecNormDistri.begin(), vecNormDistri.end(), 0.);
	double mVal = sum / vecNormDistri.size();
	double accum = 0.;
	std::for_each(vecNormDistri.begin(), vecNormDistri.end(), [&](const double d) {
		accum += (d - mVal)*(d - mVal); });

	double stdev = sqrt(accum / (vecNormDistri.size() - 1));
	//cout << "stdev = " << stdev << endl;
	//拟合直线方程
	double fitA, fitB, sigma;
	fitA = -0.5;
	fitB = 1.38;
	sigma = stdev * fitA + fitB;
	sigma = max(1.26, min(sigma, 1.275));
	//cout << "sigma = " << sigma << endl;
	return sigma;
}

//去除hessianImg中低灰度水平线段
int removeHorLineSeg(cv::Mat &src, cv::Mat &hessianImg, cv::Mat &lineLabelImg, vector<cv::Vec4f> &lineSegs)
{
	cv::Mat lineBinImg;
	cv::threshold(lineLabelImg, lineBinImg, 0, 255, THRESH_BINARY);
	lineBinImg.convertTo(lineBinImg, CV_8UC1);
	cv::dilate(lineBinImg, lineBinImg, cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3)));
	cv::Mat showImg = lineBinImg.clone();
	cv::cvtColor(showImg, showImg, cv::COLOR_GRAY2BGR);
	for (auto ite = lineSegs.begin(); ite != lineSegs.end();)
	{
		double segAngle = imgPro::angle_segX(*ite);
		int segDist = getDist2Pts(*ite);
		cv::Point2f startPt((*ite)[0], (*ite)[1]);
		cv::Point2f endPt((*ite)[2], (*ite)[3]);
		int labelIndex = lineLabelImg.at<float>(startPt);
		cv::Rect roi(cv::Point(startPt )- cv::Point(3, 3), cv::Point(endPt )+ cv::Point(3, 3));
		//if (labelIndex== 13030 || labelIndex == 13030)
		//{
		//	cv::rectangle(showImg, roi, Scalar(0, 0, 255));
		//}
		if (abs(segAngle) < 3.5 || 180 - 3.5 < abs(segAngle))
		{
			double mVal = imgPro::grayImg_getMean(hessianImg(roi), segDist, 255, 0, lineBinImg(roi));
			if (mVal < 30)
			{
				//cout << "angle mVal " << mVal << "index "<<labelIndex<<endl;
				ite = lineSegs.erase(ite);
			}
			else
			{
				ite++;
			}

		}
		else if (segDist < 20)
		{
			double mVal = imgPro::grayImg_getMean(hessianImg(roi), segDist, 255, 0, lineBinImg(roi));
			if (mVal < 16)
			{
				//cout << "angle mVal " << mVal << "index " << labelIndex << endl;
				ite = lineSegs.erase(ite);
			}
			else
			{
				ite++;
			}

		}
		else
			ite++;
	}

	return 0;

}


//原始_有重叠部分
cv::Rect getShiftRect(cv::Rect roiRect, int label, double rotAngle, map<string, vector<double>> &mapFeats,
	map<int, int>&mapLabelIndex, cv::Point centPt)
{
	int imgCentRow = centPt.y;
	int imgCentCol = centPt.x;
	const int shiftDist = 8;
	const int oneShiftDist = 8;

	cv::Rect shiftRect;
	int objRow = mapFeats["row"][mapLabelIndex[label]];
	int objCol = mapFeats["col"][mapLabelIndex[label]];
	if (objRow < imgCentRow && objCol < imgCentCol) //左上
	{
		if (abs(rotAngle) < 45)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, oneShiftDist), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(shiftDist, 0), roiRect.size());
		}
		if (rotAngle > 0)
		{

		}
	}
	else if (objRow < imgCentRow && objCol > imgCentCol) // 右上
	{
		if (abs(rotAngle) < 45)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, oneShiftDist), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(-shiftDist, 0), roiRect.size());
		}
	}
	else if (objRow > imgCentRow && objCol < imgCentCol) //左下
	{
		if (abs(rotAngle) < 45)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, -oneShiftDist), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(shiftDist, 0), roiRect.size());
		}
	}

	else
	{
		if (abs(rotAngle) < 45)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, -oneShiftDist), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(-shiftDist, 0), roiRect.size());
		}

	}


	return shiftRect;
}


cv::Rect getShiftRectL1(cv::Rect roiRect, int label, double rotAngle, map<string, vector<double>> &mapFeats,
	map<int, int>&mapLabelIndex, cv::Point centPt)
{
	int imgCentRow = centPt.y;
	int imgCentCol = centPt.x;
	const int shiftDistX = 11;
	const int shiftDistY = 11;

	cv::Rect shiftRect;
	int objRow = mapFeats["row"][mapLabelIndex[label]];
	int objCol = mapFeats["col"][mapLabelIndex[label]];
	if (objRow < imgCentRow && objCol < imgCentCol) //左上
	{
		if (   -55 < rotAngle && (rotAngle) < -35)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, shiftDistY), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(shiftDistX, shiftDistY), roiRect.size());
		}

	}
	else if (objRow < imgCentRow && objCol > imgCentCol) // 右上
	{
		if (35 < rotAngle  && rotAngle < 55)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, shiftDistX), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(-shiftDistX, shiftDistY), roiRect.size());
		}
	}
	else if (objRow > imgCentRow && objCol < imgCentCol) //左下
	{
		if (  35 <rotAngle  && rotAngle < 55)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, -shiftDistY), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(shiftDistX, -shiftDistY), roiRect.size());
		}
	}

	else
	{
		if ( -55 < (rotAngle) && rotAngle < -35)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, -shiftDistY), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(-shiftDistX, -shiftDistY), roiRect.size());
		}

	}


	return shiftRect;
}

cv::Rect getShiftRectL2(cv::Rect roiRect, int label, double rotAngle, map<string, vector<double>> &mapFeats,
	map<int, int>&mapLabelIndex, cv::Point centPt)
{
	int imgCentRow = centPt.y;
	int imgCentCol = centPt.x;
	const int shiftDistX = 11;
	const int shiftDistY = 11;

	cv::Rect shiftRect;
	int objRow = mapFeats["row"][mapLabelIndex[label]];
	int objCol = mapFeats["col"][mapLabelIndex[label]];
	if (objRow < imgCentRow && objCol < imgCentCol) //左上
	{
		if (   -55 < rotAngle && (rotAngle) < -35)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, shiftDistY), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(shiftDistX, shiftDistY), roiRect.size());
		}

	}
	else if (objRow < imgCentRow && objCol > imgCentCol) // 右上
	{
		if (35 < rotAngle  && rotAngle < 55)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, shiftDistX), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(-shiftDistX, shiftDistY), roiRect.size());
		}
	}
	else if (objRow > imgCentRow && objCol < imgCentCol) //左下
	{
		if (  35 <rotAngle  && rotAngle < 55)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, -shiftDistY), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(shiftDistX, -shiftDistY), roiRect.size());
		}
	}

	else
	{
		if ( -55 < (rotAngle) && rotAngle < -35)
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(0, -shiftDistY), roiRect.size());
		}
		else
		{
			shiftRect = cv::Rect(roiRect.tl() + cv::Point(-shiftDistX, -shiftDistY), roiRect.size());
		}

	}


	return shiftRect;
}

//合并vecLightRect矩形至vecCheckRect
int mergeRect(vector<cv::Rect> vecLightRect, vector<cv::Rect> &vecCheckRect)
{
	//合并矩形
	  vector<pair<int, cv::Rect>> vecIou;
	for (auto r1Ite = vecLightRect.begin(); r1Ite != vecLightRect.end(); r1Ite++)//找到最大IOU矩形
	{
		double maxIOU = 0;
		int maxIndex = 0;
		for (auto r2Ite = vecCheckRect.begin(); r2Ite != vecCheckRect.end(); r2Ite++)
		{
			double r1IOU = imgPro::rect_intersection(*r1Ite, *r2Ite) / double(r1Ite->area());
			if (r1IOU > maxIOU)
			{
				maxIOU = r1IOU;
				maxIndex = std::distance(vecCheckRect.begin(), r2Ite);
			}
		}
		if (maxIOU < 0.001)//没有匹配到矩形
		{
			vecIou.emplace_back(-1, *r1Ite);
		}
		else
		{
			vecIou.emplace_back(maxIndex, *r1Ite);
		}
	}
	for (int i = 0; i < vecIou.size(); i++)//合并矩形，取并
	{
		if (vecIou[i].first >= 0)
		{
			cv::Rect checkR = vecCheckRect[vecIou[i].first];
			cv::Rect lightR = vecIou[i].second;
			cv::Rect mergeR(cv::Point(min(checkR.x, lightR.x), min(checkR.y, lightR.y)),
				cv::Point(max(checkR.br().x, lightR.br().x), max(checkR.br().y, lightR.br().y)));
			vecCheckRect[vecIou[i].first] = mergeR;
		}
		else
		{
			vecCheckRect.push_back(vecIou[i].second);
		}
	}
	return 0;
}
