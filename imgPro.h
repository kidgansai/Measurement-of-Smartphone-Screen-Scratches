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


    // ͨ�ù���ģ��
    //static int img_filter(lua_State* L);
    //static int img_fillRectR(lua_State* L);
    static int img_getPartR(cv::Mat &imgIn, cv::Mat &imgOut, int row, int col, double dAngle, int L1, int L2);
    static int img_getPartR(cv::Mat &imgIn, cv::Mat &imgOut, cv::RotatedRect rotRect);
    static int img_rotate(cv::Mat &imgIn, cv::Mat &imgOut, double dAngle, uchar iFillValue, int centerRow, int centerCol);    // ͼ����ָ��������ת����Ƕ�
    static int img_translate(cv::Mat &imgIn, cv::Mat &imgOut, int iRowTransl, int iColTransl, uchar iFillValue);    // ͼ��ƽ��
	static int img_enhanceGM(cv::Mat &imgIn, cv::Mat &imgOut, double gamma, double c = 1.0);    // ͼ����ǿ��٤��任
	static int img_enhanceLog(cv::Mat &imgIn, cv::Mat &imgOut, double c = 1.0);    // ͼ����ǿ�������任
    //static int img_findContours(cv::Mat &imgIn, int iMode, int iMethod, int iMin, int iMax);
    //static int img_getContourPoints(lua_State* L);
    //static int img_drawContours(lua_State* L);


    // ͼ��ȱ�����С
    static int img_resize_keepRatio(cv::Mat &imgIn, cv::Mat &imgOut, cv::Size size);
    //static int img_scale(lua_State* L);
    //static int img_smoothGauss(lua_State* L);
    //static int img_smoothMean(lua_State* L);
    //static int img_smoothMedian(lua_State* L);
    //static int img_findHarris(lua_State* L);

/*
Func: ͼ��ͶӰ,
Params:
    imgIn - ͼ������
    strImgProj - ͶӰͼ�����
    iOrit - ����ͶӰ�ķ���0 - ˮƽ ��1 - ��ֱ
    iMode - ���㷽ʽ��0 - ���л��з�0���صĸ�����1 - ���л�������ֵ�ͣ�2 - ����
    iCalcImgProj - �Ƿ����ͶӰͼ��0 - �������1 - ���
Return:
    iErr - 0 - ��������0 - �д���
*/
    static int img_projection(cv::Mat& imgIn, cv::Mat &ImgProj, std::vector<int> &vecProjVal, int iOrit, int iMode, bool iCalcImgProj);
	static int img_drawSegments(cv::Mat & srcImg, cv::Mat & drawImg, vector<cv::Vec4f> lines, cv::Scalar color);
	static int img_drawSegmentsWithLabel(cv::Mat & srcImg, cv::Mat & drawImg, vector<cv::Vec4f> lines);

	/*
	���ͼ���ָ���Ҷȵ�λ�ã�Ĭ��������з����
	imgIn: CV_8U ��CV_32S/16S
	specificVal�� -1��Ĭ��������з�������
	*/
	static int img_getPoints(cv::Mat imgIn, vector<cv::Point> &vecPt,int specificVal = -1);

	/*
	���ͼ������ĻҶȼ�,Ĭ�ϰ�����
	imgIn: CV_8U ��CV_32S/16S
	*/
	static int img_getGrayLevel(cv::Mat imgIn, vector<int> &vecGrayLevel, bool includeZero = true,cv::Mat maskIn = cv::Mat() );

	/*����ͼ����Ƥ
	params:
		mask - 0/1��0/255
	*/
	static int img_drawMask(const cv::Mat &src, cv::Mat &imgOut, const cv::Mat &mask, cv::Scalar c,float percent=0.3);

	/*���ƾ��Σ����������Σ���ת����,���νǶ���halconһ��
	*/
	static int img_drawRect(cv::Mat &src, cv::Mat &imgOut,cv::Point &tlOut, cv::Point centerPt,cv::Size rectSize, double angle,cv::Scalar color,int thickness=1);
	static int img_drawRect(cv::Mat &src, cv::Mat &imgOut,cv::RotatedRect &rotRectOut, cv::Point centerPt,cv::Size rectSize, double angle,cv::Scalar color,int thickness=1);
	static int img_drawRect(cv::Mat src, cv::Mat &imgOut,cv::RotatedRect rotRect, cv::Scalar color, int thickness = 1);
	//halcon��ʽ����һ��opencv����
	static cv::RotatedRect rect_build(cv::Size size, cv::Point centerPt,cv::Size rectSize, double angle);
	//����fitLine��ϵ�ֱ��
	static cv::Vec4f img_drawLine(cv::Mat src, cv::Mat &imgOut,cv::Vec4f line, cv::Scalar color, int thickness = 1);

    ////---------------------------------------------------------------
    ////ģ��ƥ��
	static int img_findTemplate(cv::Mat &img, cv::Mat &templ, vector<cv::Vec3f> &vecRes, int iMethod, double dThreshRes,
									int resRowDist, int resColDist, cv::Mat maskImg);


    ////---------------------------------------------------------------
    ////�Ҷ�ͼ����
    //static int grayImg_threshold(lua_State* L);
    //static int grayImg_thresholdAuto(lua_State* L);
    //static int grayImg_getMeanValue(lua_State* L);
    //static int grayImg_thresholdRect(lua_State* L);
    //static int grayImg_gray2RGB(lua_State* L);
    //static int grayImg_canny(lua_State* L);
    static int grayImg_sobel(cv::Mat &grayImgIn, cv::Mat &sobelImg);
    static int grayImg_getHist(cv::Mat &grayImgIn, cv::Mat &histImgOut);
    //static int grayImg_compareHist(lua_State* L);
    //static int grayImg_getMinMaxValue(lua_State *L);    // ��ȡָ������ĻҶ����ֵ����Сֵ
    //static int grayImg_enhanceEH(lua_State* L);    // ͼ����ǿ��ֱ��ͼ���⻯ʵ��

/*
	Func: �ٷֱ���ֵ�ָ�
	Params:
    p - С�ڵ��ڷָ���ֵ�����������������صı���
	model :0 :�ӵ����ȵ������ȼ������P ��1:�Ӹ����ȵ������ȼ������P
    withZero - ����ʱ�Ƿ���0����
*/
    static int grayImg_thresholdPTile(cv::Mat &grayimgIn, cv::Mat &binImgOut, double p, int mode,bool withZero,int threshTypes);
    
/*
	Func: ��ģ�����ֵ�ָ�
*/
	static int grayImg_thresholdWithMask(cv::Mat& src, cv::Mat& dst, double thresh, double maxval, int type, const cv::Mat& mask);

/*
	Func: ȡһ�����л�������ֵ��ƽ���Ҷ�
	ratio: <Ϊ���У� >1Ϊ����
	start,end:�������ĻҶȷ�Χ[start,end)
			start < end :�ͻҶ�ֵ��߻Ҷ�ֵ����
			start > end :�߻Ҷ�ֵ��ͻҶ�ֵ����
	mask:����Ȥ����
*/
	static double grayImg_getMean(const cv::Mat & src, double ratio, double start, double end,cv::Mat mask=cv::Mat());	
	static int grayImg_ransacCircle(cv::Mat &grayImgIn, double cannyThresh1, double cannyThresh2, Circle &cir);
    ////-------------------------------------------------------
    ////��ɫͼ����
    //static int rgbImg_toGray(lua_State* L);
    //static int rgbImg_threshold(lua_State* L);    // ��ɫͼ��ָ�,��ȡ��ֵͼ��
    //static int rgbImg_getColor(lua_State* L);    // ��ȡָ����ɫ��Χ��ֵ
    //static int rgbImg_colorIntensity(lua_State *L);    // ��ɫͼ��ת��ɫǿ��ͼ��ǿ��Խ����ɫԽ�ӽ�


    ////-------------------------------------------------------
    ////��ֵͼ����
	static int calcRatio(cv::Mat &imgIn1, cv::Mat &imgIn2, float &ratio);
    //iMode - ����ģʽ��1 - ��ֵ֮����Ϊ0��2 - ��ֵ֮����Ϊ0
	static int biImg_filterByArea( cv::Mat &imgIn, cv::Mat &imgOut, int iAreaThreshLow,int iAreaThreshHigh, int iMode, int connection = 8);			//ʹ�������С��ֵɸѡ��ֵͼ������
	static int biImg_delRegionOnboundary(cv::Mat &imgIn, cv::Mat &imgOut, int iMode=0, int connection = 8);			//ʹ�������С��ֵɸѡ��ֵͼ������
    static int biImg_delMaxArea(cv::Mat &imgIn, cv::Mat &imgOut);
	static int biImg_getMaxArea(cv::Mat &imgIn, cv::Mat &imgOut);
    static int biImg_thinImg(cv::Mat imgIn, cv::Mat &imgOut,int maxIter = -1);    //ͼ��ϸ��

	//iType - ��䷽����ѡ��.1:��ȫ����⵽����������2�������ֵ֮�������
    static int biImg_fillup(const cv::Mat &imgIn, cv::Mat &imgOut, int iType, int fillAreaLow, int fillAreaHigh);    // ��ֵͼ�����
    
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
    ////������
    static int biImg_createRegion(const cv::Mat &imgIn, cv::Mat &imgLabel, map<string, vector<double>> &features, int iAreaThreshLow,int iAeraThreshHigh);    //��������
    static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, int index);    //������ת��Ϊͼ�񣬶�Ӧlabel������תΪͼ��
    static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, const vector<cv::Point> &vecPt);    //ʱ���Ż�
    static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut,const vector<int> &vecIndex);    //�����е�����ת��Ϊͼ��
	static int region_toImg(const cv::Mat &labelImg, cv::Mat &imgOut, const vector<vector<cv::Point>> &vvPt);  //ʱ���Ż�
	static	int region_toImg_ROI(const cv::Size size, cv::Mat &imgOut, const vector<cv::Point> &vecPt, const cv::Point tl);
	//static int region_getSkeleton(lua_State *L);    //��ȡ����ĹǼܣ���ʼ��Ϊ���ϽǶ˵�
	static int region_fitLine(cv::Mat labelImg, int index, int distType, cv::Point &startPt, cv::Point &endPt, int isVertical =0);                 //��������ϳ�ֱ��
    //static int region_smallestRectR(lua_State *L);
    //static int region_fitEllipse(lua_State *L);    //�����Բ


    ////-------------------------------------------------------
    ////��������
    //static int cont_smallestRectR(lua_State* L);
    //static int cont_smallestRect(lua_State* L);
    //static int cont_areaCenter(lua_State* L);
    //static int cont_unionContByDist(lua_State* L);
    //static int cont_fitLine(lua_State* L);
    //static int cont_fitEllipse(lua_State *L);    //�����Բ
    //static int cont_fitCircle(lua_State *L);    //���Բ

    //-------------------------------------------------------
    //��ѧ����
    static int angle2Lines(cv::Point line1StartPt, cv::Point line1EndPt, cv::Point line2StartPt, cv::Point line2EndPt,double &angle);
	static int angle2Lines(cv::Point2f line1StartPt, cv::Point2f line1EndPt, cv::Point2f line2StartPt, cv::Point2f line2EndPt, double &angle);

	//���������ֱ����ˮƽ�߼нǣ�
    static int angle2Pts(cv::Point ptStart,cv::Point ptEnd, double &angle);
	//���������ֱ����ˮƽ�߼н�
    static int angle2Pts(cv::Point2f ptStart,cv::Point2f ptEnd, double &angle);
	//�߶����X��ĽǶ�[-90,90)
    static double angle_segX(cv::Point2f ptStart,cv::Point2f ptEnd);
	//�߶����X��ĽǶ�[-90,90)
    static double angle_segX(cv::Vec4f lineSeg);
    static int intersecBy2Lines(cv::Point line1StartPt, cv::Point line1EndPt, 
								cv::Point line2StartPt, cv::Point line2EndPt,cv::Point &intersecPt);
    static int closestPt2Line(cv::Point offLinePt,cv::Point linePt1,cv::Point linePt2,cv::Point &nearestLinePt,double &dist);
	//line Ϊ����ʽ
    static double closestPt2Line(cv::Point offLinePt,cv::Vec4f line, cv::Point &nearestLinePt);
	static double closestPt2LineSeg(cv::Mat &canvImg, cv::Point offLinePt, cv::Vec4f line, cv::Point & nearestLinePt);
		//static int math_distPtCont(lua_State* L);
    //static int math_distLineCont(lua_State* L);
    static double dist2Pts(cv::Point pt1,cv::Point pt2);
    //static int math_splitRect(lua_State *L);


    //------------------------------------------------------
    //����ͼԪ��Ϣ��ȡ
	//halcon��ʽ������Σ����ض���
	static vector<cv::Point> rect_pts(cv::Size size, cv::Point centerPt, cv::Size rectSize, double angle);
	//��ȡ��ת���ε��ĸ�����ͽǶȣ��ĸ������˳�����������ϣ����ϣ����£����£�ע��Ƕ���halcon����һ��[-90,90]
	static int rect_getRotRectPts(cv::RotatedRect rotRect, std::vector<cv::Point2f> &pts, float &angle);
	static int rect_getRotRectPts(cv::RotatedRect rotRect, std::vector<cv::Point> &pts, float &angle);
	static cv::Rect rect_enlarge(cv::Size imgSize,cv::Rect rect,int len);
	static cv::Rect rect_enlarge(cv::Size imgSize, cv::Rect rect, int widthLen, int heightLen);
	/*
	//rect:��Լ������
	//size:Լ���ķ�Χ
	*/
	static cv::Rect rect_normInImg(const cv::Rect rect,cv::Size size);
	static int rect_intersection(const cv::Rect rectA,const cv::Rect rectB);

	static cv::Scalar randomColor(cv::RNG& rng);
	/*
	//size ��ͼ��ߴ�
	//ptIn����ת��
	//rotCenter����ת���ĵ�
	//ptOut����ת������
	//angle: ��ת�Ƕ�
	*/
	static int pt_rotate(cv::Size size, cv::Point ptIn, cv::Point rotCenter, cv::Point &ptOut, double angle);

	//
	static cv::Point segment_extend(cv::Size imgSize, cv::Vec4f lineSeg, cv::Point extendPt, double len);
    //static int line_getLineInfo(lua_State* L);
    //static int circle_getCircleInfo(lua_State* L);
    //static int point_getPointInfo(lua_State* L);

/*
Func: ��άͼ�����Եת��Ϊ����
Params:
    imgIn - �����ֵͼ��
    laserPts - ���������
    oritation - ��ȡ����0 - �������£�1 - �������ϣ�2 - �������ң�3 - ��������
    startPos - ��ʼ�ҵ��λ��
    endPos - �����ҵ��λ��
    withZero - �Ƿ��Ǳ߽��ϵĵ�
*/
    static int laser_fromImage(cv::Mat &imgIn, vector<cv::Point> &laserPts, int oritation, int startPos, int endPos,bool withZero,bool clearInvalidPt);

	//Range :ɨ��������ɨ�跽��
	static int laser_fromImage(cv::Mat &imgIn, vector<cv::Point> &laserPts, cv::Range rowRange,cv::Range colRange,bool withZero, bool clearInvalidPt);
    static int img_drawLaser(const cv::Mat &imgin, cv::Mat &imgOut, const vector<cv::Point2d>& profile, int thickness);
    static int img_drawLaser(const cv::Mat &imgIn, cv::Mat &imgOut, const vector<cv::Point>& profileIn, int thickness);

    // ��ȡ�����ıƽ������
    static int laser_getPolyline(const vector<cv::Point2d> &profileIn, vector<cv::Point2d> &polyPts, double epsilon, bool isClosed);
    static int laser_getPolyline(const vector<cv::Point> &profileIn, vector<cv::Point> &polyPts, double epsilon, bool isClosed);

    //static int laser_fitCircle(lua_State *L);    // ����Բ���
    //static int laser_concatenate(lua_State *L);    // ��������
    //static int laser_getPoints(lua_State *L);


    ////------------------------------------------------------
    ////���̺߳���,д��һ��ͼ�����
    //static int shared_pushMat2Que
    //static int shared_getMatFromQue

    //static int shared_pushMat2Que2
    //static int shared_getMatFromQue2


    ////-------------------------------------------------------
    ////���Ժ���
    //static int test(lua_State *L);
    //static int sys_getTime(lua_State *L);    //��ȡϵͳ��ǰʱ��


    //-------------------------------------------------------
    //�������ú���
	private:
    static bool bZero(double dData);
    static double Get2VecAngleCos(double dLineDirVec1[], double dLineDirVec2[]);    // �����ռ������ļнǣ�����ֵ��λ��cosֵ
    static double Get2VecAngle(double dLineDirVec1[], double dLineDirVec2[]);    // �����ռ������ļнǣ�����ֵ��λ������
    static void GetPluVecForPlane(double dVector1[], double dVector2[], double dPluVec[]);    //dPluVec = dVector1 ��� dVector2
    static void checkZero(double dTemp[], int iNum);

    // ����ֱ�߷�������dLineDirVec������֪���ϵĵ�dKnownPointCoo����Ϊd2PointDistance��δ֪������dUnknowPointCoo
    static void GetPointCooOnLine(double dLineDirVec[], double dKnownPointCoo[], double d2PointDistance, double dUnknowPointCoo[]);

    // ��ռ���ֱ�ߵĽ��㣬dLine1DirVecΪֱ��1����������dLine1PointΪֱ��1��һ�㣨ֱ��2���ƣ�
    //dIntersectionPointΪ��������
    static void GetIntersectionFor2Line(double dLine1DirVec[], double dLine1Point[], double dLine2DirVec[], double dLine2Point[],
        double dIntersectionPoint[]);

    static void getLinePoints(double dRho, double dTheta, int iWidth, int iHeight, double &dP1Row, double &dP1Col,
        double &dP2Row, double &dP2Col);

    static void getClosestPointP2L(double dPoint[], double dLinePoint[], double dLineVec[], double dPointOut[]);
    static void optimizeLines(vector<cv::Vec2f> &lines, double dR, double dT);

    static	vector<cv::Vec3f> filterTemplateResPoints(const cv::Mat &res, double dThresh, int rowDist, int colDist);
};


//void sobelXY();
