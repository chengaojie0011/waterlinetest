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
#include  <functional>
#include <math.h>
using namespace cv;
using namespace std;

#define MIN_AREA 1000//定义最小有效的矩形面积

vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
typedef  struct colorbox//声明一个结构体类型boxShore
{
	int num;
	int r;
	int g;
	int b ;
	int labl;
	int laba;
	int labb;
	int s;
}colorboxInfo;


typedef vector <colorboxInfo> colorboxInfoVec;

bool GreaterSort(colorboxInfo a, colorboxInfo b) { return (a.s > b.s); }
bool LessSort(colorboxInfo a, colorboxInfo b) { return (a.s < b.s); }

void colorReduce(Mat& image, int div)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int a = image.at<Vec3b>(i, j)[0];
			int b = a *div/ 256;
			image.at<Vec3b>(i, j)[0] = image.at<Vec3b>(i, j)[0] *div/ 256; 
			image.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1] *div/ 256;
			image.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2] *div/ 256;
		/*	cout << "r " << image.at<Vec3b>(i, j)[2] << endl;
			cout << "g" << image.at<Vec3b>(i, j)[1] << endl;
			cout << "b " << image.at<Vec3b>(i, j)[0] << endl;*/
		}
	}
}

void colorReduceLab(Mat& image)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			/*		int a = image.at<Vec3b>(i, j)[0];
					int b = a *div / 256;*/
			image.at<Vec3b>(i, j)[0] = image.at<Vec3b>(i, j)[0] ;
			image.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1] ;
			image.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2] ;
				cout << "l " <<int( image.at<Vec3b>(i, j)[2] )<< endl;
			cout << "a" <<int( image.at<Vec3b>(i, j)[1] )<< endl;
			cout << "b " <<int( image.at<Vec3b>(i, j)[0]) << endl;
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


void mySobel(Mat &image)
{

	Mat grayImage;
	Mat grayImagecopy;
	cvtColor(image, grayImage, CV_RGB2GRAY);
	//GaussianBlur(grayImage, grayImage, Size(5, 5), 2);   //高斯滤波  
	grayImage.copyTo(grayImagecopy);
	//imshow("gray", grayImage);
	int t1 = 0;
	int	t2 = 0;
	int maxgray = 0;
	int mingray = 255;
	int gradientgray = 0;

	for (int y = 1; y < grayImage.rows-1; y++)
	{
		for (int x = 1; x < grayImage.cols-1; x++)
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

	for (int y =1; y < grayImage.rows - 1; y++)
	{
		for (int x = 1; x < grayImage.cols - 1; x++)
		{
			gradientgray = grayImage.at<uchar>(y + 1, x - 1) + 2 * grayImage.at<uchar>(y + 1, x) + grayImage.at<uchar>(y + 1, x + 1)
				- grayImage.at<uchar>(y - 1, x - 1) - 2 * grayImage.at<uchar>(y - 1, x) - grayImage.at<uchar>(y - 1, x + 1);
		grayImagecopy.at<uchar>(y, x) = gradientgray * k + 255 - maxgray * k;
		}
	}


//	cvtColor(grayImagecopy, image, CV_GRAY2RGB);
		for (int y = 1; y < grayImage.rows - 1; y++)
		{
			for (int x = 1; x < grayImage.cols - 1; x++)
			{
				if (grayImagecopy.at<uchar>(y, x)>180)
				{
					grayImagecopy.at<uchar>(y, x) = 255;
				}
				else if  (grayImagecopy.at<uchar>(y, x)<110)
				{
					grayImagecopy.at<uchar>(y, x) = 0;
				}
				else {
					grayImagecopy.at<uchar>(y, x) = 125;
				}
			}
		}

		cvtColor(grayImagecopy, image, CV_GRAY2RGB);

}

void myWaterLine(const Mat  &src, const Mat  &sobelimage,Mat &dst)
{



		for (int x = 1; x < src.cols - 1; x++)
		{
			int  garymeanvalue=0;
			for (int y = 1; y < src.rows - 1; y++)
			{
				garymeanvalue = garymeanvalue + sobelimage.at<uchar>(y, x);
			}
		}
	
}

void meanShiftMy(const Mat  &src, Mat &dst)
{

	Mat res; //分割后图像  
	Mat grayImage;
	Mat out1;


	int a1 = 0;


	//获取自定义核  
	Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
	//进行闭运算操作  
	morphologyEx(src, out1, MORPH_CLOSE ,element);

	medianBlur(out1, out1, 5);//中值滤波

	int spatialRad = 20;  //空间窗口大小  
	int colorRad = 20;   //色彩窗口大小  
	int maxPyrLevel = 2;  //金字塔层数  
	pyrMeanShiftFiltering(out1, dst, spatialRad, colorRad, maxPyrLevel); //色彩聚类平滑滤波  


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

void print(colorboxInfoVec* colorboxinfovec) {
	for (int j = 0; j < (*colorboxinfovec).size(); j++)
	{
		std::cout <<
			(*colorboxinfovec)[j].num<< "\t" <<
			(*colorboxinfovec)[j].r << "\t" <<
			(*colorboxinfovec)[j].g << "\t" <<
			(*colorboxinfovec)[j].b << "\t" <<
			(*colorboxinfovec)[j].labl << "\t" << 
			(*colorboxinfovec)[j].laba << "\t" << 
			(*colorboxinfovec)[j].labb << "\t" << 
			(*colorboxinfovec)[j].s << "\t" << std::endl;
	}

	return;
}

void getHC(const Mat &src, Mat &dst)
{
	Mat labImage;
	Mat srccopy,srclab;
	src.copyTo(srccopy);
	colorReduce(srccopy,12);
	src.copyTo(srclab);
	src.copyTo(dst);
	cvtColor(dst, dst, CV_BGR2GRAY);
	cvtColor(srclab, labImage, CV_BGR2Lab);
	//colorReduceLab(labImage);
	colorboxInfo firstpixel = { 1,srccopy.at<Vec3b>(1, 1)[2],srccopy.at<Vec3b>(1, 1)[1],srccopy.at<Vec3b>(1, 1)[0] ,
		labImage.at<Vec3b>(1, 1)[0],	labImage.at<Vec3b>(1, 1)[1],	labImage.at<Vec3b>(1, 1)[2],0};
	colorboxInfo cherry = { 5,6,7,8 };
	colorboxInfoVec coloboxinfovec;
    coloboxinfovec.push_back(firstpixel);
	for (int y = 1; y < src.rows - 1; y++)
	{
		for (int x = 1; x < src.cols - 1; x++)
		{
	/*for (int y = 1; y < 100 - 1; y++)
	{
		for (int x = 1; x <100 - 1; x++)
		{*/
			if (x==1&&y==1)
			{
				continue;
			}
			int i = 0;
			/*	cout << "r " << srccopy.at<Vec3b>(y, x)[2] << endl;
				cout << "g" << srccopy.at<Vec3b>(y, x)[1] << endl;
				cout<<	"b " << srccopy.at<Vec3b>(y, x)[0] << endl;*/
			colorboxInfo changepixel = { 1,srccopy.at<Vec3b>(y, x)[2],srccopy.at<Vec3b>(y, x)[1],srccopy.at<Vec3b>(y, x)[0],
				labImage.at<Vec3b>(y, x)[0],	labImage.at<Vec3b>(y, x)[1],	labImage.at<Vec3b>(y, x)[2],0};
			for (int j = 0; j < coloboxinfovec.size(); j++)
			{
				if (srccopy.at<Vec3b>(y, x)[0] == coloboxinfovec[j].b &&srccopy.at<Vec3b>(y, x)[1] == coloboxinfovec[j].g&&
					srccopy.at<Vec3b>(y, x)[2] == coloboxinfovec[j].r)
				{
					i = 0;
					int hahah2 = changepixel.b;
					coloboxinfovec[j].num = coloboxinfovec[j].num + 1;
				}
				else
				{
					i++;
				}
			}
			if (i == coloboxinfovec.size())
			{
				int hahah = changepixel.b;
				coloboxinfovec.push_back(changepixel);
			}

		}
	}


	int pixelnum = (src.rows - 2)*(src.cols - 2);
	for (int i= 0; i< coloboxinfovec.size(); i++)
	{
			int signvalue = 0;
			for (int j = 0; j < coloboxinfovec.size(); j++)
			{
				int dlab= sqrt(pow(coloboxinfovec[i].labl- coloboxinfovec[j].labl, 2)+ pow(coloboxinfovec[i].laba- coloboxinfovec[j].laba, 2)
					+pow(coloboxinfovec[i].labb - coloboxinfovec[j].labb, 2));
					signvalue = signvalue + dlab*coloboxinfovec[j].num / pixelnum;
			}
			coloboxinfovec[i] .s= signvalue;
		}
	sort(coloboxinfovec.begin(), coloboxinfovec.end(), GreaterSort);//降序排列  
//	print(&coloboxinfovec);
	int maxs=0, mins=0;
	for (int i = 0; i < coloboxinfovec.size(); i++)
	{
		if (i == 0)
		{
			maxs= coloboxinfovec[i].s;
			int b = maxs;
		}
		if (i==coloboxinfovec.size()-1)
		{
			mins = coloboxinfovec[i].s;
		}
	}
	//cout << "maxs=" << maxs << endl;
	//cout << "mins=" << mins<< endl;
	int k = maxs - mins;
	for (int i = 0; i < coloboxinfovec.size(); i++)
	{
		coloboxinfovec[i].s= (coloboxinfovec[i].s-mins)*255/k;
	}
	//print(&coloboxinfovec);
	for (int y = 1; y < src.rows - 1; y++)
	{
		for (int x = 1; x < src.cols - 1; x++)
		{
			for (int j = 0; j < coloboxinfovec.size(); j++)
			{
				if (srccopy.at<Vec3b>(y, x)[0] == coloboxinfovec[j].b &&srccopy.at<Vec3b>(y, x)[1] == coloboxinfovec[j].g&&
					srccopy.at<Vec3b>(y, x)[2] == coloboxinfovec[j].r)
				{
					dst.at<uchar>(y, x) = coloboxinfovec[j].s;
				}
			}
		}
	}
	//for (int y = 1; y < src.rows - 1; y++)
	//{
	//	for (int x = 1; x < src.cols - 1; x++)
	//	{
	//		if (dst.at<uchar>(y, x)>100)
	//		{
	//			dst.at<uchar>(y, x) = 255;
	//		}
	//		else
	//		{
	//			dst.at<uchar>(y, x) = 0;

	//		}
	//	}
	//}
	imshow("dst", dst);
	imshow("lab", labImage);



}

void wtline(const Mat &src, Mat &dst)
{
	Mat imageGray;
	cvtColor(src, imageGray, CV_RGB2GRAY);//灰度转换  
	GaussianBlur(imageGray, imageGray, Size(5, 5), 2);   //高斯滤波  
	imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 30, 120);
	imshow("Canny Image", imageGray);

	//查找轮廓  

	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(src.size(), CV_8UC1);  //轮廓     
	Mat marks(src.size(), CV_32S);   //Opencv分水岭第二个矩阵参数  
	//marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点  
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	//我们来看一下传入的矩阵marks里是什么东西  
	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	imshow("marksShow", marksShows);
	imshow("轮廓", imageContours);
	watershed(src, marks);
	


}

int main()
{
	//读入图像，RGB三通道    
	Mat  srcImage = imread("12.jpg");
	imshow("src", srcImage);
//	colorReduce(srcImage,254);

	Mat out1;
 	rmHighlight(srcImage,out1);
	//Mat hcImage;
	//getHC(out1, hcImage);

	Mat  outMeanshift;

meanShiftMy(out1, outMeanshift);
	imshow("mean_shift", outMeanshift);

	Mat wtlineimg;
wtline(outMeanshift,wtlineimg);
	Mat  sobelimg;
	sobelimg = outMeanshift;
	//mySobel(sobelimg);
 //  imshow("out", sobelimg);
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

