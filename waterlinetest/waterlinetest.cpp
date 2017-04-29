// waterlinetest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<opencv2\opencv.hpp>
#include <iostream>
#include<cv.h>  
#include <opencv2\core\core.hpp>
#include<opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <math.h>  
#include <numeric>
#include <algorithm>
#include<vector>
#include <set>

using namespace cv;
using namespace std;

#define MIN_AREA 1000//定义最小有效的矩形面积

vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  

void colorReduce(Mat& image, int div)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			image.at<Vec3b>(i, j)[0] = 0;
			image.at<Vec3b>(i, j)[1] = 255;
			image.at<Vec3b>(i, j)[2] = 255;
		}
	}
}


void  rmHighlight(const Mat  &src, Mat &dst)
{
	Mat  srccopy,grayImage;
	src.copyTo(srccopy);
	cvtColor(srccopy, grayImage, CV_BGR2GRAY);//变为灰度图
//	imshow("grayImage", grayImage);
	int a1 = 0;
	dst = srccopy;
	int sl = 200, m = 3;
	int mc = 0, c = 1, fr1 = 0, fr2 = 0, fr3 = 0,a3 = 0, b3 = 0;
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{


			if (grayImage.at<uchar>(y, x) > sl)
			{

				mc = 0;
				c = 0;
				fr1=fr2=fr3 = 0;
			}
			while (c == 0 && fr1 == 0)
			{
				if (mc == 0)
				{
					for (a3 = -1; a3 < 2; a3++)
					{
						for (b3 = -1; b3 < 2; b3++)
						{
							int temp = grayImage.at<uchar>(y, x);


							if ((x + b3) < 1 || (x + b3) > (src.cols - 1) || (y + a3) < 1 || (y + a3) > (src.rows - 1))
							{
								continue;
							}

							if (grayImage.at<uchar>(y + a3, x + b3) <= sl)
							{

								c = c + 1;
								fr1= fr1 + src.at<Vec3b>(y + a3, x + b3)[0];
								fr2 = fr2 + src.at<Vec3b>(y + a3, x + b3)[1];
								fr3 = fr3 + src.at<Vec3b>(y + a3, x + b3)[2];

							}
						}
					}
				}
				else
				{

					for (a3 = -1 - mc; a3 < 2 + mc; a3++)
					{

						for (b3 = -1 - mc; b3 < 2 + mc; b3++)
						{

							if ((x + b3) < 1 || (x + b3) > (src.cols - 1) || (y + a3) < 1 || (y + a3) > (src.rows - 1))
							{
								continue;
							}

							if (grayImage.at<uchar>(y + a3, x + b3) <= sl)
							{

								if ((abs(a3)*abs(a3) + abs(b3)* abs(b3)) >= 4)
								{

									c = c + 1;
									fr1 = fr1 + src.at<Vec3b>(y + a3, x + b3)[0];
									fr2 = fr2 + src.at<Vec3b>(y + a3, x + b3)[1];
									fr3 = fr3 + src.at<Vec3b>(y + a3, x + b3)[2];
								}

							}
						}
					}
				}
				mc = mc + 2;
				int a5 = 0;
			//	a5 = fr / (c + 1);
				//cout <<"a5"<<a5<<endl;
				dst.at<Vec3b>(y, x)[0] = (fr1 + 190) / (c + 1);
				dst.at<Vec3b>(y, x)[1] = (fr2 + 190) / (c + 1);
				dst.at<Vec3b>(y, x)[2] = (fr3 + 190) / (c + 1);
			}

		}
	}

	// 把图像边缘像素设置为0
	//dst.row(0).setTo(Scalar(0));
	//dst.row(dst.rows - 1).setTo(Scalar(0));
	//dst.col(0).setTo(Scalar(0));
	//dst.col(dst.cols - 1).setTo(Scalar(0));


}
void mySobel1(Mat &image)
{
	//Calculate cl;
	Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY);
//	vector<vector<int>> gray = cl.Countgray(grayImage);
	int i,j;
	int t1 = 0;
	int	t2 = 0;
	int maxgray = 0;
	int mingray = 255;
	int gradientgray = 0;
	for (i = 2; i != grayImage.rows - 2; ++i)
	{
		for (j = 2; j != grayImage.cols - 2; ++j)
		{
		//	gradientgray = gray[i + 1][j - 1] + 2 * gray[i + 1][j] + gray[i + 1][j + 1]
		//		- gray[i - 1][j - 1] - 2 * gray[i - 1][j] - gray[i - 1][j + 1];
			gradientgray = grayImage.at<uchar>(i+ 1, j - 1) + 2 * grayImage.at<uchar>(i + 1, j) + grayImage.at<uchar>(i + 1, j+ 1)
			- grayImage.at<uchar>(i - 1, j - 1) - 2 * grayImage.at<uchar>(i - 1, j) - grayImage.at<uchar>(i - 1, j + 1);

			if (gradientgray >= maxgray)
			{
				maxgray = gradientgray;
			}
			if (gradientgray <= mingray)
			{
				mingray = gradientgray;
			}
		}
	}
	float k = 255 / float(maxgray - mingray);
	for (i = 2; i != grayImage.rows - 2; ++i)
	{
		uchar* outdata = grayImage.ptr<uchar>(i);
		for (j = 2; j != grayImage.cols - 2; ++j)
		{
		//	gradientgray = gray[i + 1][j - 1] + 2 * gray[i + 1][j] + gray[i + 1][j + 1]
				//- gray[i - 1][j - 1] - 2 * gray[i - 1][j] - gray[i - 1][j + 1];
			gradientgray = grayImage.at<uchar>(i + 1, j - 1) + 2 * grayImage.at<uchar>(i + 1, j) + grayImage.at<uchar>(i + 1, j + 1)
				- grayImage.at<uchar>(i - 1, j - 1) - 2 * grayImage.at<uchar>(i - 1, j) - grayImage.at<uchar>(i - 1, j + 1);
			outdata[j] = gradientgray * k + 255 - maxgray * k;
		}
	}
	cvtColor(grayImage, image, CV_GRAY2RGB);
}

void mySobel(Mat &image)
{
	//Calculate cl;
	Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY);
	imshow("image", image);
	//vector<vector<int>> gray = cl.Countgray(grayImage);
	int i, j;
	int y, x;
	int t1 = 0;
	int	t2 = 0;
	int maxgray = 0;
	int mingray = 255;
	int gradientgray = 0;
	//for (i = 2; i != grayImage.rows - 2; ++i)
	//{
	//	for (j = 2; j != grayImage.cols - 2; ++j)
	//	{
	//		gradientgray = gray[i + 1][j - 1] + 2 * gray[i + 1][j] + gray[i + 1][j + 1]- gray[i - 1][j - 1] - 2 * gray[i - 1][j] - gray[i - 1][j + 1];
	//		if (gradientgray >= maxgray)
	//		{
	//			maxgray = gradientgray;
	//		}
	//		if (gradientgray <= mingray)
	//		{
	//			mingray = gradientgray;
	//		}
	//	}
	//}

	for( y = 2; y != grayImage.rows - 2; ++y)
	{
		for (x = 2; x != grayImage.cols - 2; ++x)
		{
			gradientgray = grayImage.at<uchar>(y + 1, x - 1)+2*grayImage.at<uchar>(y + 1, x ) + grayImage.at<uchar>(y + 1, x +1) 
				- grayImage.at<uchar>(y -1, x - 1) -2* grayImage.at<uchar>(y -1, x ) - grayImage.at<uchar>(y - 1, x + 1)  ;
			if (gradientgray >= maxgray)
			{
				maxgray = gradientgray;
			}
			if (gradientgray <= mingray)
			{
				mingray = gradientgray;
			}
		}
	}

	float k = 255 / float(maxgray - mingray);
	for (y = 2; y != grayImage.rows - 2; ++y)
	{
		uchar* outdata = grayImage.ptr<uchar>(y);
		for (x = 2; x != grayImage.cols - 2; ++x)
		{
			gradientgray = grayImage.at<uchar>(y + 1, x - 1) + 2 * grayImage.at<uchar>(y + 1, x) + grayImage.at<uchar>(y + 1, x + 1)
				- grayImage.at<uchar>(y - 1, x - 1) - 2 * grayImage.at<uchar>(y - 1, x) - grayImage.at<uchar>(y - 1, x + 1);
			outdata[x] = gradientgray * k + 255 - maxgray * k;
		}
	}
	imshow("gray", grayImage);
	cvtColor(grayImage, image, CV_GRAY2RGB);
}

void meanShiftMy(const Mat  &src, Mat &dst)
{

	Mat res; //分割后图像  
	Mat grayImage;
	Mat out1;
	Mat out2;
	Mat out3;
	Mat out4;
	Mat out11;
	Mat out111;
	int a1 = 0;

	cvtColor(src, grayImage, CV_BGR2GRAY);//变为灰度图

//	rmHighlight(grayImage, out1);							  //高亮处理

	//cvtColor(out1, out11, CV_GRAY2BGR);//变为彩色图
	cvtColor(grayImage, out11, CV_GRAY2BGR);//变为彩色图
									   //imshow("caise", out11);
									   //获取自定义核  
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//进行闭运算操作  
	//morphologyEx(out11, out111, MORPH_CLOSE ,element);

	medianBlur(out11, out2, 5);//中值滤波

							   //imshow("test",out2);

							   //cvtColor(out2, out3, CV_BGR2Luv);//变为彩色图
	int spatialRad = 20;  //空间窗口大小  
	int colorRad = 20;   //色彩窗口大小  
	int maxPyrLevel = 2;  //金字塔层数  
	pyrMeanShiftFiltering(out2, dst, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波  
	Mat dst_x, dst_y;
	Mat  m_ResImg;
	//Sobel边缘检测
	Sobel(grayImage, dst_x, grayImage.depth(), 1, 0); //X方向梯度
	Sobel(grayImage, dst_y, grayImage.depth(), 0, 1); //Y方向梯度
	convertScaleAbs(dst_x, dst_x);
	convertScaleAbs(dst_y, dst_y);
	addWeighted(dst_x, 0.5, dst_y, 0.5, 0, m_ResImg);//合并梯度(近似)
//	imshow("Sobel边缘检测", m_ResImg);

}

void floodFillMy(const Mat  &src, Mat &dst)
{
	RNG  rng = theRNG();
	dst = src;
	Mat mask(src.rows + 2, src.cols + 2, CV_8UC1, Scalar::all(0));  //掩模  
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)  //非0处即为1，表示已经经过填充，不再处理  
			{
				Scalar newVal(rng(256), rng(256), rng(256));
				floodFill(dst, mask, Point(x, y), newVal, 0, Scalar::all(5), Scalar::all(5)); //执行漫水填充  
				dst.at<Vec3b>(y, x)[0] = 0;
				dst.at<Vec3b>(y, x)[1] =0;
				dst.at<Vec3b>(y, x)[2] = 0;
			}
		}
	}

}


int main()
{
	//读入图像，RGB三通道    
	Mat  srcImage = imread("4.jpg");
//	colorReduce(srcImage,254);
	Mat out1;
//	rmHighlight(srcImage,out1);
	//imshow("src", srcImage);

	mySobel1(srcImage);
	imshow("out", srcImage);
	//Mat grad_x,grad_y;
	//Mat abs_grad_x, abs_grad_y, dst;
	//Sobel(out1, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);
	//imshow("【效果图】 X方向Sobel", abs_grad_x);
	//Sobel(out1, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	//convertScaleAbs(grad_y, abs_grad_y);
	//imshow("【效果图】Y方向Sobel", abs_grad_y);
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	//imshow("【效果图】整体方向Sobel", dst);
	////进行水面目标的meanshift处理并进行目标识别
	Mat  outMeanshift;

	int spatialRad = 20;  //空间窗口大小  
	int colorRad = 20;   //色彩窗口大小  
	int maxPyrLevel = 2;  //金字塔层数  
	//pyrMeanShiftFiltering(out1,outMeanshift, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波  
	//meanShiftMy(srcImage, outMeanshift);
	//imshow("mean_shift", outMeanshift);

	//识别海天线
	Mat outwater;
	Point pmax1, pmax2, pout1, pout2;
//	waterLineFound(srcImage, pmax1, pmax2);
	//floodFillMy(outMeanshift,outwater);
	//imshow("line",outwater);
	//pout1.x = srcImage.cols;
	//pout1.y = pmax2.y - (pmax2.x - srcImage.cols)*(pmax2.y - pmax1.y) / (pmax2.x - pmax1.x);
	//pout2.y = pmax2.y - pmax2.x*(pmax2.y - pmax1.y) / (pmax2.x - pmax1.x);
	//pout2.x = 0;
	//line(srcImage, pout1, pout2, Scalar(0, 0, 255), 1, CV_AA);

	//imshow("outwater",srcImage);

	//直方图提取
	Mat histogram, test1;
	int hismin, hismax;
	//imshow("outmean", outMeanshift);
	waitKey(0);
    return 0;
}

