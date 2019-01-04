#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include "cv.h"
#include "highgui.h"
#include<opencv2\opencv.hpp>
#include<iostream>
//#include<Windows.h>
//#include <math.h>


IplImage*  EliminateNoise(IplImage *green);
IplImage* process_rgb(IplImage*img);

using namespace cv;

using namespace std;


const char* IMages = ".jpg"; //保存图片的文件夹路径

const char* image_out = "out_1.jpg";
const char* image_out1 = "out_2.jpg";



//OSTU算法

int HistogramBins = 256;

float HistogramRange1[2] = { 0, 255 };

float *HistogramRange[1] = { &HistogramRange1[0] };

typedef enum { back, object } entropy_state;



double caculateCurrentEntropy(CvHistogram * Histogram1, int cur_threshold, entropy_state state)

{

    int start, end;

    if (state == back)

    {

        start = 0;

        end = cur_threshold;

    }

    else

    {

        start = cur_threshold;

        end = 256;

    }

    int  total = 0;

    for (int i = start; i < end; i++)

    {

        total += (int)cvQueryHistValue_1D(Histogram1, i);

    }

    double cur_entropy = 0.0;

    for (int i = start; i < end; i++)

    {

        if ((int)cvQueryHistValue_1D(Histogram1, i) == 0)

            continue;

        double percentage = cvQueryHistValue_1D(Histogram1, i) / total;

        cur_entropy += -percentage * logf(percentage);

    }

    return cur_entropy;

}



IplImage* MaxEntropy(IplImage *src, IplImage *dst)

{

    assert(src != NULL);

    assert(src->depth == 8 && dst->depth == 8);

    assert(src->nChannels == 1);

    CvHistogram * hist = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY, HistogramRange);

    cvCalcHist(&src, hist);

    double maxentropy = -1.0;

    int max_index = -1;

    for (int i = 0; i < HistogramBins; i++)

    {

        double cur_entropy = caculateCurrentEntropy(hist, i, object) + caculateCurrentEntropy(hist, i, back);

        if (cur_entropy > maxentropy)

        {

            maxentropy = cur_entropy;

            max_index = i;

        }

    }

    printf("%f", max_index);

    cvThreshold(src, dst, (double)max_index, 255, CV_THRESH_BINARY);

    cvReleaseHist(&hist);

    return dst;

}










//颜色分割算法





int main(int argc, char* argv[])

{

    int i=1;
    //for (i = 48; i < 97; i++)
    //{

    char IMAGEpath[100];
    sprintf(IMAGEpath, "%d%s", i, IMages);
    //char IMAGEpath[100];
    //sprintf(IMAGEpath, "%s%s", IMAGEpath11, IMages);

    IplImage *img = cvLoadImage(IMAGEpath);
    IplImage *OtsuImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *OtsuImg1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *dst = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
    Mat dst1;

     for (int i = 0; i < dst->height; i++)
     {
         for (int j = 0; j < dst->width; j++)
         {
             ((uchar *)(dst->imageData + i*dst->widthStep))[j*dst->nChannels + 0] = 0; // B
             ((uchar *)(dst->imageData + i*dst->widthStep))[j*dst->nChannels + 1] = 0; // G
             ((uchar *)(dst->imageData + i*dst->widthStep))[j*dst->nChannels + 2] = 0; // R
         }
    }




    CvSeq * contour = 0;

    CvMemStorage * storage = cvCreateMemStorage();

    IplImage * tempdst1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);



    clock_t start, finish;

    double duration;

 //   cvNamedWindow("tempdst1", 1);

 //  cvNamedWindow("img", 1);

 //   cvNamedWindow("OstuImg", 1);

//    cvNamedWindow("OstuImg1", 1);

//    cvNamedWindow("dst1", 1);
//    cvShowImage("dst1", dst);
//    cvNamedWindow("dst", 1);







    OtsuImg = process_rgb(img);

//    cvNamedWindow("process_rgb", 1);
//    cvShowImage("process_rgb", OtsuImg);

    OtsuImg1 = MaxEntropy(OtsuImg, OtsuImg1);

 //   cvNamedWindow("MaxEntropy", 1);
 //   cvShowImage("MaxEntropy", OtsuImg1);
    cvSaveImage("Ostu.jpg", OtsuImg1);


    cvFindContours(OtsuImg1, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);



    cvZero(tempdst1);

    //	cvZero(temp_iamge);

    for (; contour != 0; contour = contour->h_next)

    {   //应用函数 fabs() 得到面积的绝对值。 

        double area = cvContourArea(contour, CV_WHOLE_SEQ);

        //计算整个轮廓或部分轮廓的面积

        if (fabs(area) < 10)

        {

            continue;

        }

        CvPoint *point = new CvPoint[contour->total];

        CvPoint *Point;

        for (int i = 0; i < contour->total; i++)

        {

            Point = (CvPoint*)cvGetSeqElem(contour, i);

            point[i].x = Point->x;

            point[i].y = Point->y;

        }

        int pts[1] = { contour->total };

        cvFillPoly(tempdst1, &point, pts, 1, CV_RGB(255, 255, 255));//填充多边形内部 



    }
 //   tempdst1 = EliminateNoise(tempdst1);
  //  cvShowImage("tempdst1", tempdst1);

  
  //  cvShowImage("img", img);
    //cvShowImage("OstuImg", OtsuImg);
    //cvShowImage("OstuImg1", OtsuImg1);
  //  cvShowImage("tian", tempdst1);
    //EliminateNoise(tempdst1);
    //cvShowImage("dst", dst);

    //IplImage* temp = cvCreateImage( //创建一个size为image,三通道8位的彩色图
    //    cvGetSize(tempdst1),
    //    IPL_DEPTH_8U,
    //    1
    //    );

    //

    //腐蚀膨胀
    //IplConvKernel* kernel = cvCreateStructuringElementEx(5, 3, 2, 1, CV_SHAPE_ELLIPSE);
    //cvDilate(tempdst1, tempdst1, kernel, 4);
    //cvErode(tempdst1, tempdst1, kernel, 1);



    cvCopy(img, dst, tempdst1);


    char IMAGEpath1[100];
    sprintf(IMAGEpath1, "%d%s",i,image_out);
    char IMAGEpath2[100];
    sprintf(IMAGEpath2, "%d%s",i,image_out1);


    cvSaveImage(IMAGEpath1, tempdst1);
    cvSaveImage(IMAGEpath2, dst);



   //while (1){

   //    if (cvWaitKey(100) == 27)

   //         break;
   // }

    cvReleaseImage(&img);
    cvReleaseImage(&dst);
    cvReleaseImage(&OtsuImg);
    cvReleaseImage(&OtsuImg1);
    cvReleaseImage(&tempdst1);
    cvReleaseMemStorage(&storage);


     //}   


    return 0;



}

IplImage*  EliminateNoise(IplImage *green)
{

          int color = 254;// 从254开始，因此连通域不能多于253个

          CvSize sz = cvGetSize(green);

          int w;

          int h;

          for (w = 0; w<sz.width; w++)

          {

              for (h = 0; h<sz.height; h++)

              {

                  if (color > 0)

                  {

                      if (CV_IMAGE_ELEM(green, unsigned char, h, w) == 255)

                      {

                          cvFloodFill(green, cvPoint(w, h), CV_RGB(color, color, color));//把各连通域标记上颜色

                          color--;

                      }

                  }

              }

          }

//          cvNamedWindow("labeled");

//          cvShowImage("labeled", green);//显示标记后的图像

          int colorsum[255] = { 0 };

          for (w = 0; w<sz.width; w++)

          {

              for (h = 0; h<sz.height; h++)

              {

                  if (CV_IMAGE_ELEM(green, unsigned char, h, w) > 0)//不对0值计数，不可能为255

                  {

                      colorsum[CV_IMAGE_ELEM(green, unsigned char, h, w)]++;//统计每种颜色的数量

                  }

              }

          }

          vector<int> v1(colorsum, colorsum + 255);//用数组初始化vector

          int maxcolorsum = max_element(v1.begin(), v1.end()) - v1.begin();//求出最多数量的颜色

          for (w = 0; w<sz.width; w++)

          {

              for (h = 0; h<sz.height; h++)

              {

                  if (CV_IMAGE_ELEM(green, unsigned char, h, w) == maxcolorsum)

                  {

                      CV_IMAGE_ELEM(green, unsigned char, h, w) = 255;//只把最多数量的颜色标为255

                  }

                  else

                  {

                      CV_IMAGE_ELEM(green, unsigned char, h, w) = 0;//其他标为0

                  }

              }

          }

//          cvNamedWindow("最大连通域");

//          cvShowImage("最大连通域", green);//显示最大连通域

          return green;
}


IplImage* process_rgb(IplImage*img){

   

    IplImage *b = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    IplImage *g = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    IplImage *temp1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    IplImage *temp2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *temp3 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *temp4 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *temp5 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    IplImage *r = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    IplImage *dst1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    cvSplit(img, b, g, r, NULL);
    //亮度

    //cvAddWeighted(b, 1.0 / 3.0, g, 1.0 / 3.0, 0.0, temp4);
    //cvAddWeighted(r, 1.0 / 3.0, temp4, 1.0, 0.0, temp4);
    //cvInRangeS(temp4, cvScalar(180.0, 0.0, 0.0), cvScalar(200.0, 0.0, 0.0), temp1);
    //cvNamedWindow("1111", 1);
    //cvShowImage("1111", temp1);

    ////红色分量
    //cvInRangeS(r, cvScalar(220.0, 0.0, 0.0), cvScalar(255.0, 0.0, 0.0), temp2);
    //cvNamedWindow("2222", 1);
    //cvShowImage("2222", temp2);

    ////亮度与红色分量加权
    //cvAddWeighted(temp2, 2.0 / 3.0, temp1, 1.0/3.0, 0.0, temp1);
    //cvNamedWindow("3333", 1);
    //cvShowImage("3333", temp1);




    //RGB HSI
    CvSize size = cvGetSize(img);
    IplImage * pImgFire = cvCreateImage(size, 8, 3);
    IplImage *dst111 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

    cvSet(pImgFire, cvScalar(0, 0, 0));

    int RedThreshold = 100;  //115~135 
    int SaturationThreshold = 45;  //55~65

    for (int j = 0; j < img->height; j++){
        for (int i = 0; i < img->widthStep; i += 3){
            uchar B = (uchar)img->imageData[j*img->widthStep + i + 0];
            uchar G = (uchar)img->imageData[j*img->widthStep + i + 1];
            uchar R = (uchar)img->imageData[j*img->widthStep + i + 2];
            uchar maxv = max(max(R, G), B);
            uchar minv = min(min(R, G), B);
            double S = (1 - 3.0*minv / (R + G + B));
            double I = (R + G + B) / 3;
            double H = acos(((0.5*((R - G) + R - B))) / sqrt((R - G)^2 + (R - B)*(G - B)));

            //生成二值图
            if (I>108 &&S > 0.05){
                pImgFire->imageData[j*img->widthStep + i + 0] = 255;
                pImgFire->imageData[j*img->widthStep + i + 1] = 255;
                pImgFire->imageData[j*img->widthStep + i + 2] = 255;
            }
            else{
                pImgFire->imageData[j*img->widthStep + i + 0] = 0;
                pImgFire->imageData[j*img->widthStep + i + 1] = 0;
                pImgFire->imageData[j*img->widthStep + i + 2] = 0;
            }
        }
    }

    cvCvtColor(pImgFire, dst111, CV_BGR2GRAY);
    cvReleaseImage(&pImgFire);

//    cvNamedWindow("th", 1);
//    cvShowImage("th", dst111);
    cvSaveImage("HSI.jpg", dst111);




    //R-G分量
    cvSub(r, g, temp3);
    cvAddWeighted(temp3, 1.0 / 2.0, temp3, 0.0, 0.0, temp3);
    //cvNamedWindow("rg", 1);
    //cvShowImage("rg", temp3);


    //cvSub(b, g, temp4);
    //cvAddWeighted(temp4, 1.0 / 2.0, temp4, 0.0, 0.0, temp4);
    //cvNamedWindow("bg", 1);
    //cvShowImage("bg", temp4);

    //R-B分量
    cvSub(r, b, g);	
    cvAddWeighted(g, 1.0 / 2.0, r, 0.0, 0.0, r);  	
    //cvNamedWindow("rb", 1);
    //cvShowImage("rb", r);

    //R-G与R-B分量加权
    cvAddWeighted(r, 1.0 / 2.0, temp3, 1.0 / 2.0, 0.0, temp3);
//    cvNamedWindow("rg+rb", 1);
//    cvShowImage("rg+rb", temp3);
    cvSaveImage("RGB.jpg", temp3);

    cvAnd(temp3, dst111, r);
//    cvNamedWindow("and", 1);
//    cvShowImage("and", r);


    cvSmooth(r, r, CV_GAUSSIAN, 3, 0, 0, 0);
//    cvNamedWindow("mohu", 1);
//    cvShowImage("mohu", r);
    cvSaveImage("RGBandHSIsmooth.jpg", r);



    return r;


    cvReleaseImage(&r);
    cvReleaseImage(&g);
    cvReleaseImage(&b);
    cvReleaseImage(&temp1);
    cvReleaseImage(&temp2);

}

