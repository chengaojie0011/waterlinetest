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
vector<Point> maxLine;
vector<Point> minLine;
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

void colorgraychange(Mat& image, int div)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			image.at<uchar>(i, j) =div;
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


		/*for (int y = 1; y < grayImage.rows - 1; y++)
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
		}*/

		cvtColor(grayImagecopy, image, CV_GRAY2RGB);

}

void myWaterLine(const Mat  &sobelimage,Mat &dst)
{
	for (int x = 1; x < dst.cols-5; x = x + 5)
	{
		int  garymax = 0, garymin = 255, ymin = 0, ymax = 0;
		for (int y = 1; y < dst.rows - 1; y++)
		{
			//每个x的每行的值
			int  garymeanvalue = 0;
			for (int xin = x; xin < x + 5; xin++)
			{
				int color = sobelimage.at<uchar>(y, xin);
				garymeanvalue = garymeanvalue + int(sobelimage.at<uchar>(y, xin));
			}

			if (garymax <= garymeanvalue / 5)
			{
				garymax = garymeanvalue / 5;
				ymax = y;
			
			}

		}
		Point  pout1;
		pout1.x = x + 2;
		pout1.y = ymax;
		maxLine.push_back(pout1);

		for (int y = 1; y < dst.rows - 1; y++)
		{
			//每个x的每行的值
			int  garymeanvalue = 0;
			for (int xin = x; xin < x + 5; xin++)
			{
				int color = sobelimage.at<uchar>(y, xin);
				garymeanvalue = garymeanvalue + int(sobelimage.at<uchar>(y, xin));
			}


			if (garymin >= garymeanvalue / 5 && y<=ymax-10)
			{
				garymin = garymeanvalue / 5;
				ymin = y;
			}
			
		}

		Point  pout3;
		pout3.x = x+2;
		pout3.y = ymin;
		minLine.push_back(pout3);
	}


	for (int j = 3; j < minLine.size() - 3; j++)
	{
		if (abs(minLine[j - 1].y - minLine[j].y) > 10&& abs(minLine[j + 1].y - minLine[j].y) > 10)
		{
			minLine[j].y = minLine[j - 1].y;
			//minLine[j].y = float(minLine[j - 3].y + minLine[j + 3].y) * 8 / 20 + float(minLine[j -2].y + minLine[j +2].y) * 2 / 20;
		}
		if (abs(maxLine[j - 1].y - maxLine[j].y) > 10&&abs(maxLine[j + 1].y - maxLine[j].y)  > 10)
		{
			maxLine[j].y = maxLine[j - 1].y;
		}
	}

	//修补山连着上边界的情况
	for (int j = 0; j < minLine.size() - 1; j++)
	{
		if (minLine[j].y < 10)
		{
			if (abs(minLine[j - 1].y - minLine[j].y) > 10)
			{
				for (int x = j - 1; x > 0; x--)
				{
					if (abs(minLine[x].y - minLine[x - 1].y) <100)
					{
						minLine[x].y = 1;
					}
				}
			}

		}
	}

	for (int j = minLine.size() - 1; j > 20; j--)
	{
		if (minLine[j].y < 10)
		{
			if (abs(minLine[j + 1].y - minLine[j].y) > 20)
			{
				for (int x = j + 1; x < minLine.size() - 1; x++)
				{
					if (abs(minLine[x].y - minLine[x + 1].y) <100)
					{
						minLine[x].y = 1;
					}
				}
			}

		}
	}

	//去除单点噪声
	for (int j = 2; j < minLine.size() - 2; j++)
	{
		if (abs(minLine[j - 1].y - minLine[j].y) >20)
		{
			minLine[j].y = minLine[j - 1].y;
		}
		if (abs(maxLine[j - 1].y - maxLine[j].y) > 20)
		{
			maxLine[j].y = maxLine[j - 1].y;
		}
	}




		for (int j = 0; j < maxLine.size()-1; j++)
		{
	
			line(dst, maxLine[j], maxLine[j+1], Scalar(255, 255, 255), 1, CV_AA);
			line(dst, minLine[j], minLine[j + 1], Scalar(0,0, 0), 1, CV_AA);

		}
	//	imshow("hah", dst);

		for (int x = 1; x < dst.cols - 1; x ++)
		{
			for (int y = 1; y < dst.rows - 1; y++)
			{
				if (dst.at<uchar>(y, x)>200)
				{
					dst.at<uchar>(y, x) = 255;
				}
				else if (dst.at<uchar>(y, x)<50)
				{
					dst.at<uchar>(y, x) = 2;
				}
	
			}
		}


		}

void coloraverLab(Mat &waterline,Mat & image)
{
	cvtColor(image, image, CV_BGR2Lab);
	imshow("lab", image);
	//山体

				for (int x = 10; x < image.cols-10; x++)
				{
					int ymax = 0,ymin=0;
					for (int yin = 10; yin < image.rows - 10; yin++)
					{
						if (waterline.at<uchar>(yin, x) ==255)
						{
							ymax = yin;
						}
						else if(waterline.at<uchar>(yin, x) == 0)
						{
							ymin = yin;
						}
					}
					for (int y = 10; y < image.rows-10; y++)
					{
					
						/*image.at<Vec3b>(i, j)[0] = image.at<Vec3b>(i, j)[0];
						image.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1];
						image.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2];*/
	
				}
			}
}

void mouAndSky(Mat &waterline, Mat & image)
{
	//cvtColor(image, image, CV_BGR2Lab);
	//imshow("lab", image);
	//山体
	imshow("waterline", waterline);
		for (int x = 10; x < image.cols - 10; x++)
		{
			int ymax = 0, ymin = 0;
			for (int yin = 10; yin < image.rows - 10; yin++)
			{
				if (waterline.at<uchar>(yin, x) == 2)
				{
					ymin = yin;
				}
				if (waterline.at<uchar>(yin, x) == 255)
				{
					ymax = yin;
				}
		
			}
			cout << "x=" << x << "ymin=" << ymin << endl;
			for (int y = 10; y < image.rows - 10; y++)
			{
				
				if (y<ymax&&y>ymin)
				{
					image.at<Vec3b>(y, x)[0] = 255;
					image.at<Vec3b>(y, x)[1] = 0;
					image.at<Vec3b>(y, x)[2] = 0;
				}
				else if (y<ymin)
				{
					image.at<Vec3b>(y, x)[0] = 0;
					image.at<Vec3b>(y, x)[1] = 255;
					image.at<Vec3b>(y, x)[2] = 0;
				}
				else
				{
					image.at<Vec3b>(y, x)[0] = 0;
					image.at<Vec3b>(y, x)[1] = 0;
					image.at<Vec3b>(y, x)[2] = 255;
				}

			}
		}
		imshow("mousky", image);
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

void getHC(const Mat &src, const Mat &waterline, Mat &dst)
{

	Mat labImage;
	Mat srccopy,srclab;
	src.copyTo(srccopy);
	colorReduce(srccopy,12);
	src.copyTo(srclab);
	src.copyTo(dst);
	cvtColor(dst, dst, CV_BGR2GRAY);
	cvtColor(srclab, labImage, CV_BGR2Lab);
	//colorboxInfo firstpixel = { 1,srccopy.at<Vec3b>(1, 1)[2],srccopy.at<Vec3b>(1, 1)[1],srccopy.at<Vec3b>(1, 1)[0] ,
	//	labImage.at<Vec3b>(1, 1)[0],	labImage.at<Vec3b>(1, 1)[1],	labImage.at<Vec3b>(1, 1)[2],0};
	colorboxInfo firstpixel = { 1,srccopy.at<Vec3b>(10, 10)[2],srccopy.at<Vec3b>(10, 10)[1],srccopy.at<Vec3b>(10, 10)[0] ,
		labImage.at<Vec3b>(10, 10)[0],	labImage.at<Vec3b>(10, 10)[1],	labImage.at<Vec3b>(10, 10)[2],0 };
	colorboxInfo cherry = { 5,6,7,8 };
	colorboxInfoVec coloboxinfovec;
    coloboxinfovec.push_back(firstpixel);
	int pixelnum=0;
	for (int x = 10; x < src.cols - 10; x++)
	{

		int waterpointy = 0;
		int watergary = 0;
		for (int yin = 10; yin < src.rows - 10; yin++)
		{
			if (watergary<waterline.at<uchar>(yin, x) )
			{
				watergary = waterline.at<uchar>(yin, x);
				waterpointy = yin;
			}
		}
		for (int y = 10; y < src.rows - 10; y++)
		{
		
			if (x==10&y==10)
			{
				continue;
			}

				if (y<=waterpointy+2)
				{
					continue;
				}
				int i = 0;
				colorboxInfo changepixel = { 1,srccopy.at<Vec3b>(y, x)[2],srccopy.at<Vec3b>(y, x)[1],srccopy.at<Vec3b>(y, x)[0],
					labImage.at<Vec3b>(y, x)[0],	labImage.at<Vec3b>(y, x)[1],	labImage.at<Vec3b>(y, x)[2],0 };
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
				pixelnum++;
		}
	}


	//int pixelnum = (src.rows - 10)*(src.cols - 10);
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
	//print(&coloboxinfovec);
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
	for (int x = 10; x < src.cols - 10; x++)
	{
		int waterpointy = 0;
		int watergary = 0;
		for (int yin = 10; yin < src.rows - 10; yin++)
		{
			if (watergary < waterline.at<uchar>(yin, x))
			{
				watergary = waterline.at<uchar>(yin, x);
				waterpointy = yin;
			}
		}
		for (int y = 10; y < src.rows - 10; y++)
		{
			if (y <= waterpointy+2)
			{
				dst.at<uchar>(y, x) = 0;
				continue;
			}
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

	//imshow("dst", dst);
//	imshow("lab", labImage);



}


void foundmax(const Mat &src, const Mat &waterline)
{


	int x =20;
	int waterpointy = 0;
	for (int yin = 10; yin < src.rows - 10; yin++)
	{
		int aaa = waterline.at<uchar>(yin, x);
		if (waterline.at<uchar>(yin, x) != 125)
		{
			waterpointy = yin;
			cout << "yin=" << yin << "x=" << x << endl;
			cout << "aaa=" << aaa << endl;
		}
	}
	//for (int x = 10; x < src.cols - 10; x++)
	//{
	//	for (int y = 10; y < src.rows - 10; y++)
	//	{

	//		if (x == 10 & y == 10)
	//		{
	//			continue;
	//		}

	//		int waterpointy = 0;
	//		for (int yin = 10; yin < src.rows - 10; yin++)
	//		{
	//			int aaa = waterline.at<uchar>(yin, x);
	//			if (waterline.at<uchar>(yin, x) !=125)
	//			{
	//				waterpointy = yin;
	//				cout << "yin=" << yin << "x=" << x << endl;
	//			}
	//		}

	//		/*	if (y <= waterpointy)
	//			{

	//			}*/
	//	}
	//}
}

void getBoat(Mat &src,Mat &dst)
{
	//src.copyTo(dst);
	for (int x =0; x < src.cols ; x++)
	{
		for (int y = 0; y < src.rows ; y++)
		{
			if ((x>=0&&x<=10)||(x>=src.cols-10&&x<=src.cols-1)|| (y >= 0 && y <= 10) || (y >= src.rows- 10 && y<= src.rows- 1))
			{
				src.at<uchar>(y, x) = 0;
				continue;
			}
			if (src.at<uchar>(y , x )>=120)
			{
				src.at<uchar>(y, x) = 255;
			}
			else 
			{
				src.at<uchar>(y, x) = 0;
			}
		}
	}
	Point lefttop, righttop ,leftbottom, rightbottom;
	int left = src.cols;
	int right = 0;
	int bottom = 0;
	int top = src.rows;
	for (int x = 0; x < src.cols; x++)
	{
		for (int y = 0; y < src.rows; y++)
		{
			if (src.at<uchar>(y, x) == 255)
			{
				if (x<left)
				{
					left = x;
				}
				if (x>right)
				{
					right = x;
				}
				if (y<top)
				{
					top = y;
				}
				if (y>bottom)
				{
					bottom = y;
				}
			}
		}
	}
	lefttop.x = left;
	lefttop.y = top;
	leftbottom.x =left ;
	leftbottom.y = bottom;
	righttop.x =right ;
	righttop.y = top;
	rightbottom.x =right ;
	rightbottom.y = bottom;
	line(dst, lefttop, leftbottom, Scalar(0, 0, 255), 1, CV_AA);
	line(dst, lefttop, righttop, Scalar(0, 0, 255), 1, CV_AA);
	line(dst, leftbottom, rightbottom, Scalar(0, 0, 255), 1, CV_AA);
	line(dst, righttop, rightbottom, Scalar(0, 0, 255), 1, CV_AA);

	imshow("boat", dst);

}


int main()
{
	//读入图像，RGB三通道    
	Mat  srcImage = imread("12.jpg");

	Mat out1;
 	rmHighlight(srcImage,out1);
	


	Mat  outMeanshift;

	meanShiftMy(out1, outMeanshift);
	Mat  sobelimg;
	outMeanshift.copyTo(sobelimg);
	mySobel(sobelimg);

	cvtColor(sobelimg, sobelimg, CV_BGR2GRAY);
	Mat waterline;
	srcImage.copyTo(waterline);
	cvtColor(waterline, waterline, CV_BGR2GRAY);
	colorgraychange(waterline,125);
	myWaterLine(sobelimg,waterline);
	imshow("water",waterline);
	Mat  colorlabsrc;
	out1.copyTo(colorlabsrc);
	mouAndSky(waterline,colorlabsrc);

	Mat hcImage;
	getHC(out1,waterline, hcImage);
	imshow("hc", hcImage);

	Mat boatimage;
	srcImage.copyTo(boatimage);
	getBoat(hcImage,boatimage);
	imshow("boat", boatimage);

	waitKey(0);
    return 0;
}

