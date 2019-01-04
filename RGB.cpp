
//本程序为在HSV颜色空间，利用圆形度信息识别图片中的火焰
//需要配置电脑环境变量，配置VS包含目录、库目录和链接器中的附加依赖项
#include<time.h>//计算程序运行时间
#include<iostream>
#include<opencv2/highgui/highgui.hpp> 
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

RNG rng(time(0));

int main(void)
{
    //读取待测图片
    Mat srcImage = imread("F:\\图片处理\\Picture\\干扰\\4\\1.jpg");//出自highgui.hpp;
    imshow("srcImage", srcImage);

    while (1)
    {
        Mat rgbPicture = srcImage.clone();
        Mat hsvPicture = srcImage.clone();
        Mat fireExtractPicture = srcImage.clone();//深拷贝

        clock_t start = clock();//开始时间
        /*提取火焰图片的颜色特征*/
        //在HSV颜色空间进行筛选	
        cvtColor(rgbPicture, hsvPicture, CV_BGR2HSV);
        //imshow("hsvPicture", hsvPicture);
        for (int i = 0; i < hsvPicture.rows; i++)
        {
            for (int j = 0; j < hsvPicture.cols; j++)
            {
                if ((hsvPicture.ptr<Vec3b>(i)[j][0] >= 0 && hsvPicture.ptr<Vec3b>(i)[j][0] <= 60) && (hsvPicture.ptr<Vec3b>(i)[j][1] >= 90 && hsvPicture.ptr<Vec3b>(i)[j][1] <= 255) && (hsvPicture.ptr<Vec3b>(i)[j][2] >= 180 && hsvPicture.ptr<Vec3b>(i)[j][2] <= 255))//HSV的取值范围
                {
                    fireExtractPicture.ptr<Vec3b>(i)[j][0] = 255;//H
                    fireExtractPicture.ptr<Vec3b>(i)[j][1] = 255;//S
                    fireExtractPicture.ptr<Vec3b>(i)[j][2] = 255;//V
                }
                else
                {
                    fireExtractPicture.ptr<Vec3b>(i)[j][0] = 0;
                    fireExtractPicture.ptr<Vec3b>(i)[j][1] = 0;
                    fireExtractPicture.ptr<Vec3b>(i)[j][2] = 0;
                }
                //cout << fireExtractPicture.ptr<Vec3b>(i)[j][0] << endl;
            }
        }
        //imshow("endPicture", fireExtractPicture);
        //开运算，去除图像中小的白点
        morphologyEx(fireExtractPicture, fireExtractPicture, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
        //imshow("MORPH_OPENPicture", fireExtractPicture);
        //闭运算，还原图像
        morphologyEx(fireExtractPicture, fireExtractPicture, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
        //imshow("MORPH_CLOSEPicture", fireExtractPicture);
        Mat circleCalculatePicture = fireExtractPicture.clone();//深拷贝

        /*火焰圆形度计算*/
        //提取轮廓，计算周长和面积，圆形度
        Canny(fireExtractPicture, circleCalculatePicture, 200, 100, 3);
        //imshow("cannyPicture", circleCalculatePicture);

        ////cvtColor(circleCalculatePicture, circleCalculatePicture,CV_BGR2GRAY);
        vector<vector<Point>> contours;//轮廓数组
        findContours(circleCalculatePicture, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//寻找图片中的轮廓
        vector<Rect> boundRect(contours.size());//轮廓矩形框

        drawContours(srcImage, contours, -1, Scalar(255, 0, 0), 1);//在原图上画出轮廓
        for (unsigned int i = 0; i < contours.size(); i++)
        {
            boundRect[i] = boundingRect(Mat(contours[i]));

            double length = arcLength(contours[i], true);
            double area = contourArea(contours[i], false);
            if (area > 0)
            {
                if ((length*length) / (12.57*area) > 2)//计算圆度C=L*L/4*Pi*A
                {
                    //drawContours(srcImage, Mat(contours[i]), -1, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1);
                    rectangle(srcImage, boundRect[i].tl(), boundRect[i].br(), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, 0);//在原图上画出疑似火源的外框
                }
            }
        }

        clock_t finish = clock();//结束时间
        namedWindow("圆形度提取结果", WINDOW_AUTOSIZE);
        imshow("圆形度提取结果", srcImage);

        double totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
        cout << totaltime << endl;

        waitKey(0);
    }
    return 0;
}




//
//
//#include<opencv2\opencv.hpp>
//#include<iostream>
//    using namespace std;
//    using namespace cv;
//
//    class HistogramND{
//    private:
//        Mat image;//源图像
//        int hisSize[1], hisWidth, hisHeight;//直方图的大小,宽度和高度
//        float range[2];//直方图取值范围
//        const float *ranges;
//        Mat channelsRGB[3];//分离的BGR通道
//        MatND outputRGB[3];//输出直方图分量
//    public:
//        HistogramND(){
//            hisSize[0] = 256;
//            hisWidth = 400;
//            hisHeight = 400;
//            range[0] = 5.0;
//            range[1] = 255.0;
//            ranges = &range[0];
//        }
//
//        //导入图片
//        bool importImage(String path){
//            image = imread(path);
//            if (!image.data)
//                return false;
//            return true;
//        }
//
//        //分离通道
//        void splitChannels(){
//            split(image, channelsRGB);
//        }
//
//        //计算直方图
//        void getHistogram(){
//            calcHist(&channelsRGB[0], 1, 0, Mat(), outputRGB[0], 1, hisSize, &ranges);
//            calcHist(&channelsRGB[1], 1, 0, Mat(), outputRGB[1], 1, hisSize, &ranges);
//            calcHist(&channelsRGB[2], 1, 0, Mat(), outputRGB[2], 1, hisSize, &ranges);
//
//            //输出各个bin的值
//            for (int i = 0; i < hisSize[0]; ++i){
//                cout << i << "   B:" << outputRGB[0].at<float>(i);
//                cout << "   G:" << outputRGB[1].at<float>(i);
//                cout << "   R:" << outputRGB[2].at<float>(i) << endl;
//            }
//        }
//
//        //显示直方图
//        void displayHisttogram(){
//            Mat rgbHist[3];
//            for (int i = 0; i < 3; i++)
//            {
//                rgbHist[i] = Mat(hisWidth, hisHeight, CV_8UC3, Scalar::all(0));
//            }
//            normalize(outputRGB[0], outputRGB[0], 0, hisWidth - 20, NORM_MINMAX);
//            normalize(outputRGB[1], outputRGB[1], 0, hisWidth - 20, NORM_MINMAX);
//            normalize(outputRGB[2], outputRGB[2], 0, hisWidth - 20, NORM_MINMAX);
//            for (int i = 0; i < hisSize[0]; i++)
//            {
//                int val = saturate_cast<int>(outputRGB[0].at<float>(i));
//                rectangle(rgbHist[0], Point(i * 2 + 10, rgbHist[0].rows), Point((i + 1) * 2 + 10, rgbHist[0].rows - val), Scalar(0, 0, 255), 1, 8);
//                val = saturate_cast<int>(outputRGB[1].at<float>(i));
//                rectangle(rgbHist[1], Point(i * 2 + 10, rgbHist[1].rows), Point((i + 1) * 2 + 10, rgbHist[1].rows - val), Scalar(0, 255, 0), 1, 8);
//                val = saturate_cast<int>(outputRGB[2].at<float>(i));
//                rectangle(rgbHist[2], Point(i * 2 + 10, rgbHist[2].rows), Point((i + 1) * 2 + 10, rgbHist[2].rows - val), Scalar(255, 0, 0), 1, 8);
//            }
//
//            cv::imshow("R", rgbHist[0]);
//            imshow("G", rgbHist[1]);
//            imshow("B", rgbHist[2]);
//            imshow("image", image);
//        }
//    };
//
//
//    int main(){
//        string path = "F:\\图片处理\\Picture\\火\\25\\1.jpg";
//        HistogramND hist;
//        if (!hist.importImage(path)){
//            cout << "Import Error!" << endl;
//            return -1;
//        }
//        hist.splitChannels();
//        hist.getHistogram();
//        hist.displayHisttogram();
//        waitKey(0);
//        return 0;
//    }
//
//
////#include <opencv2/opencv.hpp>
////#include <iostream>
////
////using namespace std;
////using namespace cv;
////
/////*  全局变量的声明及初始化    */
////Mat srcImage; //读入的图片矩阵
////Mat dstImage; //读入的图片矩阵
////MatND dstHist; //直方图矩阵，对应老版本中的cvCreateHist（）
////int g_hdims = 50;     // 划分HIST的初始个数，越高越精确
////
/////* 回调函数声明 */
////void on_HIST(int t, void *);
////
////
/////*   主函数   */
////int main(int argc, char** argv)
////{
////
////    srcImage = imread("F:\\图片处理\\Picture\\干扰\\JPEG\\16\\1.jpg", 0);//"0"表示读入灰度图像
////    namedWindow("原图", 1);//对应老版本中的cvNamedWindow( )
////    imshow("原图", srcImage);//对应老版本中的 cvShowImage（）
////
////    createTrackbar("hdims", "原图", &g_hdims, 256, on_HIST);//对应旧版本中的cvCreateTrackbar( );
////    on_HIST(0, 0);//调用滚动条回调函数
////    cvWaitKey(0);
////    return 0;
////}
////
////
/////*      滚动条回调函数       */
////void on_HIST(int t, void *)
////{
////    dstImage = Mat::zeros(512, 800, CV_8UC3);//每次都要初始化
////    float hranges[] = { 0, 255 }; //灰度范围
////    const float *ranges[] = { hranges };//灰度范围的指针
////
////    if (g_hdims == 0)
////    {
////        printf("直方图条数不能为零！\n");
////    }
////    else
////    {
////        /*
////        srcImage:读入的矩阵
////        1:数组的个数为1
////        0：因为灰度图像就一个通道，所以选0号通道
////        Mat（）：表示不使用掩膜
////        dstHist:输出的目标直方图
////        1：需要计算的直方图的维度为1
////        g_hdims:划分HIST的个数
////        ranges:表示每一维度的数值范围
////        */
////        //int channels=0;
////        calcHist(&srcImage, 1, 0, Mat(), dstHist, 1, &g_hdims, ranges); // 计算直方图对应老版本的cvCalcHist
////
////        /* 获取最大最小值 */
////        double max = 0;
////        minMaxLoc(dstHist, NULL, &max, 0, 0);// 寻找最大值及其位置，对应旧版本的cvGetMinMaxHistValue();
////
////        /*  绘出直方图    */
////
////        double bin_w = (double)dstImage.cols / g_hdims;  // hdims: 条的个数，则 bin_w 为条的宽度
////        double bin_u = (double)dstImage.rows / max;  //// max: 最高条的像素个数，则 bin_u 为单个像素的高度
////
////        // 画直方图
////        for (int i = 0; i<g_hdims; i++)
////        {
////            Point p0 = Point(i*bin_w, dstImage.rows);//对应旧版本中的cvPoint（）
////
////            int val = dstHist.at<float>(i);//注意一点要用float类型，对应旧版本中的 cvGetReal1D(hist->bins,i);
////            Point p1 = Point((i + 1)*bin_w, dstImage.rows - val*bin_u);
////            rectangle(dstImage, p0, p1, cvScalar(0, 255), 1, 8, 0);//对应旧版中的cvRectangle();
////        }
////
////        /*   画刻度   */
////        char string[12];//存放转换后十进制数，转化成十进制后的位数不超过12位，这个根据情况自己设定
////        //画纵坐标刻度（像素个数）
////        int kedu = 0;
////        for (int i = 1; kedu<max; i++)
////        {
////            kedu = i*max / 10;//此处选择10个刻度
////            itoa(kedu, string, 10);//把一个整数转换为字符串，这个当中的10指十进制
////            //在图像中显示文本字符串
////            putText(dstImage, string, Point(0, dstImage.rows - kedu*bin_u), 1, 1, Scalar(0, 255, 255));//对应旧版中的cvPutText（）
////        }
////        //画横坐标刻度（像素灰度值）
////        kedu = 0;
////        for (int i = 1; kedu<256; i++)
////        {
////            kedu = i * 20;//此处选择间隔为20
////            itoa(kedu, string, 10);//把一个整数转换为字符串
////            //在图像中显示文本字符串
////            putText(dstImage, string, cvPoint(kedu*(dstImage.cols / 256), dstImage.rows), 1, 1, Scalar(0, 255, 255));
////        }
////        namedWindow("Histogram", 1);
////        imshow("Histogram", dstImage);
////    }
////
////}
//
