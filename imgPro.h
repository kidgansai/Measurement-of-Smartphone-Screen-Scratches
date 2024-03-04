#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "opencv.hpp"
#include "omp.h"

using namespace std;

typedef cv::Point3_<uchar> colorPixel;
typedef struct Circle
{
	Circle() {};
	Circle(double _x, double _y)
	{
		x = _x;
		y = _y;
		radius = 3;
	}
	Circle(double _x, double _y, double _rad)
	{
		x = _x;
		y = _y;
		radius = _rad;
	}
	double x=0;
	double y = 0;
	double radius = 0;
}Circle;


class Color
{
public:
	Color() = default;
	~Color() = default;

	static cv::Scalar RED;
	static cv::Scalar BLUE;
	static cv::Scalar GREEN;

	static cv::Scalar getRandomColor( );
private:
};


class imgPro
{
public:
    imgPro(void);
    ~imgPro(void);


    // 通用功能模块
    //static int img_filter(lua_State* L);
    //static int img_fillRectR(lua_State* L);
    static int img_getPartR(cv::Mat &imgIn, cv::Mat &imgOut, int row, int col, double dAngle, int L1, int L2);
    static int img_getPartR(cv::Mat &imgIn, cv::Mat &imgOut, cv::RotatedRect rotRect);
    static int img_rotate(cv::Mat &imgIn, cv::Mat &imgOut, double dAngle, uchar iFillValue, int centerRow, int centerCol);    // 图像绕指定中心旋转任意角度
    static int img_translate(cv::Mat &imgIn, cv::Mat &imgOut, int iRowTransl, int iColTransl, uchar iFillValue);    // 图像平移
	static int img_enhanceGM(cv::Mat &imgIn, cv::Mat &imgOut, double gamma, double c = 1.0);    // 图像增强，伽马变换
	static int img_enhanceLog(cv::Mat &imgIn, cv::Mat &imgOut, double c = 1.0);    // 图像增强，对数变换
    //static int img_findContours(cv::Mat &imgIn, int iMode, int iMethod, int iMin, int iMax);
    //static int img_getContourPoints(lua_State* L);
    //static int img_drawContours(lua_State* L);


    // 图像等比例缩小
    static int img_resize_keepRatio(cv::Mat &imgIn, cv::Mat &imgOut, cv::Size size);
    //static int img_scale(lua_State* L);
    //static int img_smoothGauss(lua_State* L);
    //static int img_smoothMean(lua_State* L);
    //static int img_smoothMedian(lua_State* L);
    //static int img_findHarris(lua_State* L);

/*
Func: 图像投影,
Params:
    imgIn - 图像输入
    strImgProj - 投影图像输出
    iOrit - 计算投影的方向，0 - 水平 ，1 - 竖直
    iMode - 计算方式，0 - 该行或列非0像素的个数，1 - 该行或列像素值和，2 - 备用
    iCalcImgProj - 是否输出投影图像，0 - 不输出，1 - 输出
Return:
    iErr - 0 - 正常，非0 - 有错误
*/
    static int img_projection(cv::Mat& imgIn, cv::Mat &ImgProj, std::vector<int> &vecProjVal, int iOrit, int iMode, bool iCalcImgProj);
	static int img_drawSegments(cv::Mat & srcImg, cv::Mat & drawImg, vector<cv::Vec4f> lines, cv::Scalar color);
	static int img_drawSegmentsWithLabel(cv::Mat & srcImg, cv::Mat & drawImg, vector<cv::Vec4f> lines);

	/*
	输出图像的指定灰度点位置，默认输出所有非零点
	imgIn: CV_8U 或CV_32S/16S
	specificVal： -1：默认输出所有非零像素
	*/
	static int img_getPoints(cv::Mat imgIn, vector<cv::Point> &vecPt,int specificVal = -1);

	/*
	输出图像包含的灰度级,默认包括零
	imgIn: CV_8U 或CV_32S/16S
	*/
	static int img_getGrayLevel(cv::Mat imgIn, vector<int> &vecGrayLevel, bool includeZero = true,cv::Mat maskIn = cv::Mat() );

	/*绘制图像蒙皮
	params:
		mask - 0/1或0/255
	*/
	static int img_drawMask(const cv::Mat &src, cv::Mat &imgOut, const cv::Mat &mask, cv::Scalar c,float percent=0.3);

	/*绘制矩形，包括正矩形，旋转矩形,矩形角度与halcon一致
	*/
	static int img_drawRect(cv::Mat &src, cv::Mat &imgOut,cv::Point &tlOut, cv::Point centerPt,cv::Size rectSize, double angle,cv::Scalar color,int thickness=1);
	static int img_drawRect(cv::Mat &src, cv::Mat &imgOut,cv::RotatedRect &rotRectOut, cv::Point centerPt,cv::Size rectSize, double angle,cv::Scalar color,int thickness=1);
	static int img_drawRect(cv::Mat src, cv::Mat &imgOut,cv::RotatedRect rotRect, cv::Scalar color, int thickness = 1);
	//halcon方式创建一个opencv矩形
	static cv::RotatedRect rect_build(cv::Size size, cv::Point centerPt,cv::Size rectSize, double angle);
	//绘制fitLine拟合的直线
	static cv::Vec4f img_drawLine(cv::Mat src, cv::Mat &imgOut,cv::Vec4f line, cv::Scalar color, int thickness = 1);

    ////---------------------------------------------------------------
    ////模板匹配
	static int img_findTemplate(cv::Mat &img, cv::Mat &templ, vector<cv::Vec3f> &vecRes, int iMethod, double dThreshRes,
									int resRowDist, int resColDist, cv::Mat maskImg);


    ////---------------------------------------------------------------
    ////灰度图像处理
    //static int grayImg_threshold(lua_State* L);
    //static int grayImg_thresholdAuto(lua_State* L);
    //static int grayImg_getMeanValue(lua_State* L);
    //static int grayImg_thresholdRect(lua_State* L);
    //static int grayImg_gray2RGB(lua_State* L);
    //static int grayImg_canny(lua_State* L);
    static int grayImg_sobel(cv::Mat &grayImgIn, cv::Mat &sobelImg);
    static int grayImg_getHist(cv::Mat &grayImgIn, cv::Mat &histImgOut);
    //static int grayImg_compareHist(lua_State* L);
    //static int grayImg_getMinMaxValue(lua_State *L);    // 获取指定区域的灰度最大值和最小值
    //static int grayImg_enhanceEH(lua_State* L);    // 图像增强，直方图均衡化实现

/*
	Func: 百分比阈值分割
	Params:
    p - 小于等于分割阈值的像素数量与总像素的比例
	model :0 :从低亮度到高亮度计算比列P ，1:从高亮度到低亮度计算比列P
    withZero - 计算时是否考虑0像素
*/
    static int grayImg_thresholdPTile(cv::Mat &grayimgIn, cv::Mat &binImgOut, double p, int mode,bool withZero,int threshTypes);
    
/*
	Func: 带模板的阈值分割
*/
	static int grayImg_thresholdWithMask(cv::Mat& src, cv::Mat& dst, double thresh, double maxval, int type, const cv::Mat& mask);

/*
	Func: 取一定比列或数量的值的平均灰度
	ratio: <为比列， >1为数量
	start,end:参与计算的灰度范围[start,end)
			start < end :低灰度值向高灰度值计算
			start > end :高灰度值向低灰度值计算
	mask:感兴趣区域
*/
	static double grayImg_getMean(const cv::Mat & src, double ratio, double start, double end,cv::Mat mask=cv::Mat());	
	static int grayImg_ransacCircle(cv::Mat &grayImgIn, double cannyThresh1, double cannyThresh2, Circle &cir);
    ////-------------------------------------------------------
    ////彩色图像处理
    //static int rgbImg_toGray(lua_State* L);
    //static int rgbImg_threshold(lua_State* L);    // 彩色图像分割,获取二值图像
    //static int rgbImg_getColor(lua_State* L);    // 获取指定彩色范围的值
    //static int rgbImg_colorIntensity(lua_State *L);    // 彩色图像转颜色强度图，强度越大颜色越接近


    ////-------------------------------------------------------
    ////二值图像处理
	static int calcRatio(cv::Mat &imgIn1, cv::Mat &imgIn2, float &ratio);
    //iMode - 处理模式。1 - 阈值之间设为0，2 - 阈值之外设为0
	static int biImg_filterByArea( cv::Mat &imgIn, cv::Mat &imgOut, int iAreaThreshLow,int iAreaThreshHigh, int iMode, int connection = 8);			//使用面积大小阈值筛选二值图像区域
	static int biImg_delRegionOnboundary(cv::Mat &imgIn, cv::Mat &imgOut, int iMode=0, int connection = 8);			//使用面积大小阈值筛选二值图像区域
    static int biImg_delMaxArea(cv::Mat &imgIn, cv::Mat &imgOut);
	static int biImg_getMaxArea(cv::Mat &imgIn, cv::Mat &imgOut);
    static int biImg_thinImg(cv::Mat imgIn, cv::Mat &imgOut,int maxIter = -1);    //图像细化

	//iType - 填充方法的选择.1:完全填充检测到的外轮廓，2：填充阈值之间的区域
    static int biImg_fillup(const cv::Mat &imgIn, cv::Mat &imgOut, int iType, int fillAreaLow, int fillAreaHigh);    // 二值图像填充
    
    //static int biImg_houghLines(lua_State* L);
    //static int biImg_houghLinesP(lua_State* L);
    //static int biImg_dilation(lua_State* L);
    //static int biImg_erosion(lua_State* L);
    static int biImg_houghCircles(cv::Mat &imgIn, vector<cv::Vec3f> &circles, int iMinDis, int iMinR, int iMaxR);
    static int img_showHoughCircles(cv::Mat &imgIn, cv::Mat &imgOut, vector<cv::Vec3f> circles);

    //static int biImg_getEdge(lua_State *L);
    static int biImg_getRotRect(cv::Mat &imgIn, cv::RotatedRect & rotRect);
	static int biImg_getBoundingRect(cv::Mat &imgIn, cv::Rect & rect);
	static std::vector<cv::Point> biImg_getBoundingRectPts( cv::Rect & rect);

    ////-------------------------------------------------------
    ////区域处理
    static int biImg_createRegion(const cv::Mat &imgIn, cv::Mat &imgLabel, map<string, vector<double>> &features, int iAreaThreshLow,int iAeraThreshHigh);    //创建区域
    static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, int index);    //将区域转换为图像，对应label的区域转为图像
    static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, const vector<cv::Point> &vecPt);    //时间优化
    static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut,const vector<int> &vecIndex);    //将所有的区域转换为图像
	static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, const vector<vector<cv::Point>> &vvPt);  //时间优化
	static	int region_toImg_ROI(const cv::Size size, cv::Mat &imgOut, const vector<cv::Point> &vecPt, const cv::Point tl);
	//static int region_getSkeleton(lua_State *L);    //获取区域的骨架，起始点为左上角端点
	static int region_fitLine(cv::Mat labelImg, int index, int distType, cv::Point &startPt, cv::Point &endPt, int isVertical =0);                 //将区域拟合成直线
    //static int region_smallestRectR(lua_State *L);
    //static int region_fitEllipse(lua_State *L);    //拟合椭圆


    ////-------------------------------------------------------
    ////轮廓处理
    //static int cont_smallestRectR(lua_State* L);
    //static int cont_smallestRect(lua_State* L);
    //static int cont_areaCenter(lua_State* L);
    //static int cont_unionContByDist(lua_State* L);
    //static int cont_fitLine(lua_State* L);
    //static int cont_fitEllipse(lua_State *L);    //拟合椭圆
    //static int cont_fitCircle(lua_State *L);    //拟合圆

    //-------------------------------------------------------
    //数学计算
    static int angle2Lines(cv::Point line1StartPt, cv::Point line1EndPt, cv::Point line2StartPt, cv::Point line2EndPt,double &angle);
	static int angle2Lines(cv::Point2f line1StartPt, cv::Point2f line1EndPt, cv::Point2f line2StartPt, cv::Point2f line2EndPt, double &angle);

	//两个点组成直线与水平线夹角，
    static int angle2Pts(cv::Point ptStart,cv::Point ptEnd, double &angle);
	//两个点组成直线与水平线夹角
    static int angle2Pts(cv::Point2f ptStart,cv::Point2f ptEnd, double &angle);
	//线段相对X轴的角度[-90,90)
    static double angle_segX(cv::Point2f ptStart,cv::Point2f ptEnd);
	//线段相对X轴的角度[-90,90)
    static double angle_segX(cv::Vec4f lineSeg);
    static int intersecBy2Lines(cv::Point line1StartPt, cv::Point line1EndPt, 
								cv::Point line2StartPt, cv::Point line2EndPt,cv::Point &intersecPt);
    static int closestPt2Line(cv::Point offLinePt,cv::Point linePt1,cv::Point linePt2,cv::Point &nearestLinePt,double &dist);
	//line 为两点式
    static double closestPt2Line(cv::Point offLinePt,cv::Vec4f line, cv::Point &nearestLinePt);
	static double closestPt2LineSeg(cv::Mat &canvImg, cv::Point offLinePt, cv::Vec4f line, cv::Point & nearestLinePt);
		//static int math_distPtCont(lua_State* L);
    //static int math_distLineCont(lua_State* L);
    static double dist2Pts(cv::Point pt1,cv::Point pt2);
    //static int math_splitRect(lua_State *L);


    //------------------------------------------------------
    //基本图元信息获取
	//halcon方式构造矩形，返回顶点
	static vector<cv::Point> rect_pts(cv::Size size, cv::Point centerPt, cv::Size rectSize, double angle);
	//获取旋转矩形的四个顶点和角度，四个顶点的顺序依次是左上，右上，右下，左下，注意角度与halcon类型一致[-90,90]
	static int rect_getRotRectPts(cv::RotatedRect rotRect, std::vector<cv::Point2f> &pts, float &angle);
	static int rect_getRotRectPts(cv::RotatedRect rotRect, std::vector<cv::Point> &pts, float &angle);
	static cv::Rect rect_enlarge(cv::Size imgSize,cv::Rect rect,int len);
	static cv::Rect rect_enlarge(cv::Size imgSize, cv::Rect rect, int widthLen, int heightLen);
	/*
	//rect:待约束矩形
	//size:约束的范围
	*/
	static cv::Rect rect_normInImg(const cv::Rect rect,cv::Size size);
	static int rect_intersection(const cv::Rect rectA,const cv::Rect rectB);

	static cv::Scalar randomColor(cv::RNG& rng);
	/*
	//size ：图像尺寸
	//ptIn：旋转点
	//rotCenter：旋转中心点
	//ptOut：旋转后坐标
	//angle: 旋转角度
	*/
	static int pt_rotate(cv::Size size, cv::Point ptIn, cv::Point rotCenter, cv::Point &ptOut, double angle);

	//
	static cv::Point segment_extend(cv::Size imgSize, cv::Vec4f lineSeg, cv::Point extendPt, double len);
    //static int line_getLineInfo(lua_State* L);
    //static int circle_getCircleInfo(lua_State* L);
    //static int point_getPointInfo(lua_State* L);

/*
Func: 二维图像外边缘转换为轮廓
Params:
    imgIn - 输入二值图像
    laserPts - 输出的轮廓
    oritation - 提取方向。0 - 由上往下，1 - 由下往上，2 - 由左往右，3 - 由右往左
    startPos - 开始找点的位置
    endPos - 结束找点的位置
    withZero - 是否考虑边界上的点
*/
    static int laser_fromImage(cv::Mat &imgIn, vector<cv::Point> &laserPts, int oritation, int startPos, int endPos,bool withZero,bool clearInvalidPt);

	//Range :扫描间隔决定扫描方向
	static int laser_fromImage(cv::Mat &imgIn, vector<cv::Point> &laserPts, cv::Range rowRange,cv::Range colRange,bool withZero, bool clearInvalidPt);
    static int img_drawLaser(const cv::Mat &imgin, cv::Mat &imgOut, const vector<cv::Point2d>& profile, int thickness);
    static int img_drawLaser(const cv::Mat &imgIn, cv::Mat &imgOut, const vector<cv::Point>& profileIn, int thickness);

    // 获取轮廓的逼近多段线
    static int laser_getPolyline(const vector<cv::Point2d> &profileIn, vector<cv::Point2d> &polyPts, double epsilon, bool isClosed);
    static int laser_getPolyline(const vector<cv::Point> &profileIn, vector<cv::Point> &polyPts, double epsilon, bool isClosed);

    //static int laser_fitCircle(lua_State *L);    // 轮廓圆拟合
    //static int laser_concatenate(lua_State *L);    // 轮廓连接
    //static int laser_getPoints(lua_State *L);


    ////------------------------------------------------------
    ////多线程函数,写读一个图像队列
    //static int shared_pushMat2Que
    //static int shared_getMatFromQue

    //static int shared_pushMat2Que2
    //static int shared_getMatFromQue2


    ////-------------------------------------------------------
    ////测试函数
    //static int test(lua_State *L);
    //static int sys_getTime(lua_State *L);    //获取系统当前时间


    //-------------------------------------------------------
    //其他公用函数
	private:
    static bool bZero(double dData);
    static double Get2VecAngleCos(double dLineDirVec1[], double dLineDirVec2[]);    // 求两空间向量的夹角，返回值单位：cos值
    static double Get2VecAngle(double dLineDirVec1[], double dLineDirVec2[]);    // 求两空间向量的夹角，返回值单位：弧度
    static void GetPluVecForPlane(double dVector1[], double dVector2[], double dPluVec[]);    //dPluVec = dVector1 叉乘 dVector2
    static void checkZero(double dTemp[], int iNum);

    // 求沿直线方向向量dLineDirVec、离已知线上的点dKnownPointCoo距离为d2PointDistance的未知点坐标dUnknowPointCoo
    static void GetPointCooOnLine(double dLineDirVec[], double dKnownPointCoo[], double d2PointDistance, double dUnknowPointCoo[]);

    // 求空间两直线的交点，dLine1DirVec为直线1方向向量，dLine1Point为直线1上一点（直线2类似）
    //dIntersectionPoint为交点坐标
    static void GetIntersectionFor2Line(double dLine1DirVec[], double dLine1Point[], double dLine2DirVec[], double dLine2Point[],
        double dIntersectionPoint[]);

    static void getLinePoints(double dRho, double dTheta, int iWidth, int iHeight, double &dP1Row, double &dP1Col,
        double &dP2Row, double &dP2Col);

    static void getClosestPointP2L(double dPoint[], double dLinePoint[], double dLineVec[], double dPointOut[]);
    static void optimizeLines(vector<cv::Vec2f> &lines, double dR, double dT);

    static	vector<cv::Vec3f> filterTemplateResPoints(const cv::Mat &res, double dThresh, int rowDist, int colDist);
};


//void sobelXY();
