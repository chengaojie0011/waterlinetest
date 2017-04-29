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


void mySobel(Mat &image)
{

	Mat grayImage;
	Mat grayImagecopy;
	cvtColor(image, grayImage, CV_RGB2GRAY);
	grayImage.copyTo(grayImagecopy);
	imshow("gray", grayImage);
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
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//进行闭运算操作  
	//morphologyEx(out11, out111, MORPH_CLOSE ,element);

	medianBlur(src, out1, 5);//中值滤波

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
int Quantize(const Mat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio)
{
	static const int clrNums[3] = { 12, 12, 12 };
	static const float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	static const int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

	CV_Assert(img3f.data != NULL);
	idx1i = Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;
	if (img3f.isContinuous() && idx1i.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}

	// Build color pallet
	map<int, int> pallet;
	for (int y = 0; y < img3f.cols; y++)
	{
		const float* imgData = img3f.ptr<float>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < img3f.cols; x++, imgData += 3)
		{
			idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
			pallet[idx[x]] ++;   // (color, num) pairs in pallet
		}
	}

	// Fine significant colors
	int maxNum = 0;
	{
		int count = 0;
		vector<pair<int, int>> num; //
		num.reserve(pallet.size());
		for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(pair<int, int>(it->second, it->first)); // (num, color) pairs in num
		sort(num.begin(), num.end(), std::greater<pair<int, int>>());

		//maxNum 表示直方图中的bin数目
		maxNum = (int)num.size();
		int maxDropNum = cvRound(rows * cols * (1 - ratio));
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum < 10)
			maxNum = min((int)pallet.size(), 100);
		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i;

		vector<Vec3i> color3i(num.size());
		for (unsigned int i = 0; i < num.size(); i++)
		{
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}
		//将少于（1-ratio）的像素所占的颜色被直方图中距离最近的颜色所替代。
		for (unsigned int i = maxNum; i < num.size(); i++)
		{
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++)
			{
				int d_ij = vecSqrDist3(color3i[i], color3i[j]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}

	_color3f = Mat::zeros(1, maxNum, CV_32FC3);
	_colorNum = Mat::zeros(_color3f.size(), CV_32S);

	Vec3f* color = (Vec3f*)(_color3f.data);
	int* colorNum = (int*)(_colorNum.data);
	for (int y = 0; y < rows; y++)
	{
		const Vec3f* imgData = img3f.ptr<Vec3f>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			idx[x] = pallet[idx[x]];             //pallet(color,index)
			color[idx[x]] += imgData[x];
			colorNum[idx[x]] ++;
		}
	}
	for (int i = 0; i < _color3f.cols; i++)
		color[i] /= colorNum[i];

	return _color3f.cols;
}

void SmoothSaliency(const Mat &binColor3f, Mat &sal1d, float delta, const vector<vector<CostfIdx>> &similar)
{
	if (sal1d.cols < 2)
		return;
	CV_Assert(binColor3f.size() == sal1d.size() && sal1d.rows == 1);
	int binN = binColor3f.cols;
	Vec3f* color = (Vec3f*)(binColor3f.data);
	Mat tmpSal;
	sal1d.copyTo(tmpSal);
	float *sal = (float*)(tmpSal.data);
	float *nSal = (float*)(sal1d.data);

	//* Distance based smooth
	int n = max(cvRound(binN / delta), 2);
	vecF dist(n, 0), val(n);
	for (int i = 0; i < binN; i++)
	{
		const vector<CostfIdx> &similari = similar[i];
		float totalDist = 0;

		val[0] = sal[i];
		for (int j = 1; j < n; j++)
		{
			int ithIdx = similari[j].second;
			dist[j] = similari[j].first;
			val[j] = sal[ithIdx];
			totalDist += dist[j];
		}
		float valCrnt = 0;
		for (int j = 0; j < n; j++)
			valCrnt += val[j] * (totalDist - dist[j]);

		nSal[i] = valCrnt / ((n - 1) * totalDist);
	}
	//*/

	/* Gaussian smooth
	const float guassCoeff = -0.5f/(delta*delta);
	for (int i = 0; i < binN; i++)
	{
	const vector<CostfIdx> &similari = similar[i];
	float saliencyI = sal[i], totalW = 1;

	for (int j = 1; j < binN; j++)
	{
	float w = expf(sqr(similari[j].first)*guassCoeff);
	if (w < 1e-8f)
	break;
	saliencyI += w * sal[similari[j].second];
	totalW += w;
	}
	nSal[i] = saliencyI / totalW;
	}
	//*/
}
void GetHCIN(const Mat &binColor3f, const Mat &weight1f, Mat &_colorSal)
{
	int binN = binColor3f.cols;
	_colorSal = Mat::zeros(1, binN, CV_32F);
	float* colorSal = (float*)(_colorSal.data);
	vector<vector<CostfIdx>> similar(binN); // Similar color: how similar and their index
	Vec3f* color = (Vec3f*)(binColor3f.data);
	float *w = (float*)(weight1f.data);
	for (int i = 0; i < binN; i++)
	{
		vector<CostfIdx> &similari = similar[i];
		similari.push_back(make_pair(0.f, i));
		for (int j = 0; j < binN; j++)
		{
			if (i == j)
				continue;
			float dij = vecDist3<float>(color[i], color[j]);
			similari.push_back(make_pair(dij, j));
			colorSal[i] += w[j] * dij;
		}
		sort(similari.begin(), similari.end());
	}

	SmoothSaliency(binColor3f, _colorSal, 4.0f, similar);
}

Mat  GetHC(const Mat &img3f)
{
	// Quantize colors and
	Mat idx1i, binColor3f, colorNums1i, weight1f, _colorSal;
	Quantize(img3f, idx1i, binColor3f, colorNums1i,0);
	cvtColor(binColor3f, binColor3f, CV_BGR2Lab);

	normalize(colorNums1i, weight1f, 1, 0, NORM_L1, CV_32F);
	GetHCIN(binColor3f, weight1f, _colorSal);
	float* colorSal = (float*)(_colorSal.data);
	Mat salHC1f(img3f.size(), CV_32F);
	for (int r = 0; r < img3f.rows; r++)
	{
		float* salV = salHC1f.ptr<float>(r);
		int* _idx = idx1i.ptr<int>(r);
		for (int c = 0; c < img3f.cols; c++)
			salV[c] = colorSal[_idx[c]];
	}
	GaussianBlur(salHC1f, salHC1f, Size(3, 3), 0);
	normalize(salHC1f, salHC1f, 0, 1, NORM_MINMAX);
	return salHC1f;
}


int main()
{
	//读入图像，RGB三通道    
	Mat  srcImage = imread("4.jpg");
	imshow("src", srcImage);
//	colorReduce(srcImage,254);

	Mat  test;
	test = GetHC(srcImage);
	imshow("HC", test);
	Mat out1;
 	rmHighlight(srcImage,out1);

	//Mat  outMeanshift;

//	meanShiftMy(out1, outMeanshift);
	//imshow("mean_shift", outMeanshift);

	mySobel(out1);
	imshow("out", out1);
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

