
//������Ϊ��HSV��ɫ�ռ䣬����Բ�ζ���Ϣʶ��ͼƬ�еĻ���
//��Ҫ���õ��Ի�������������VS����Ŀ¼����Ŀ¼���������еĸ���������
#include<time.h>//�����������ʱ��
#include<iostream>
#include<opencv2/highgui/highgui.hpp> 
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

RNG rng(time(0));

int main(void)
{
    //��ȡ����ͼƬ
    Mat srcImage = imread("F:\\ͼƬ����\\Picture\\����\\4\\1.jpg");//����highgui.hpp;
    imshow("srcImage", srcImage);

    while (1)
    {
        Mat rgbPicture = srcImage.clone();
        Mat hsvPicture = srcImage.clone();
        Mat fireExtractPicture = srcImage.clone();//���

        clock_t start = clock();//��ʼʱ��
        /*��ȡ����ͼƬ����ɫ����*/
        //��HSV��ɫ�ռ����ɸѡ	
        cvtColor(rgbPicture, hsvPicture, CV_BGR2HSV);
        //imshow("hsvPicture", hsvPicture);
        for (int i = 0; i < hsvPicture.rows; i++)
        {
            for (int j = 0; j < hsvPicture.cols; j++)
            {
                if ((hsvPicture.ptr<Vec3b>(i)[j][0] >= 0 && hsvPicture.ptr<Vec3b>(i)[j][0] <= 60) && (hsvPicture.ptr<Vec3b>(i)[j][1] >= 90 && hsvPicture.ptr<Vec3b>(i)[j][1] <= 255) && (hsvPicture.ptr<Vec3b>(i)[j][2] >= 180 && hsvPicture.ptr<Vec3b>(i)[j][2] <= 255))//HSV��ȡֵ��Χ
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
        //�����㣬ȥ��ͼ����С�İ׵�
        morphologyEx(fireExtractPicture, fireExtractPicture, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
        //imshow("MORPH_OPENPicture", fireExtractPicture);
        //�����㣬��ԭͼ��
        morphologyEx(fireExtractPicture, fireExtractPicture, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
        //imshow("MORPH_CLOSEPicture", fireExtractPicture);
        Mat circleCalculatePicture = fireExtractPicture.clone();//���

        /*����Բ�ζȼ���*/
        //��ȡ�����������ܳ��������Բ�ζ�
        Canny(fireExtractPicture, circleCalculatePicture, 200, 100, 3);
        //imshow("cannyPicture", circleCalculatePicture);

        ////cvtColor(circleCalculatePicture, circleCalculatePicture,CV_BGR2GRAY);
        vector<vector<Point>> contours;//��������
        findContours(circleCalculatePicture, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//Ѱ��ͼƬ�е�����
        vector<Rect> boundRect(contours.size());//�������ο�

        drawContours(srcImage, contours, -1, Scalar(255, 0, 0), 1);//��ԭͼ�ϻ�������
        for (unsigned int i = 0; i < contours.size(); i++)
        {
            boundRect[i] = boundingRect(Mat(contours[i]));

            double length = arcLength(contours[i], true);
            double area = contourArea(contours[i], false);
            if (area > 0)
            {
                if ((length*length) / (12.57*area) > 2)//����Բ��C=L*L/4*Pi*A
                {
                    //drawContours(srcImage, Mat(contours[i]), -1, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1);
                    rectangle(srcImage, boundRect[i].tl(), boundRect[i].br(), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, 0);//��ԭͼ�ϻ������ƻ�Դ�����
                }
            }
        }

        clock_t finish = clock();//����ʱ��
        namedWindow("Բ�ζ���ȡ���", WINDOW_AUTOSIZE);
        imshow("Բ�ζ���ȡ���", srcImage);

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
//        Mat image;//Դͼ��
//        int hisSize[1], hisWidth, hisHeight;//ֱ��ͼ�Ĵ�С,��Ⱥ͸߶�
//        float range[2];//ֱ��ͼȡֵ��Χ
//        const float *ranges;
//        Mat channelsRGB[3];//�����BGRͨ��
//        MatND outputRGB[3];//���ֱ��ͼ����
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
//        //����ͼƬ
//        bool importImage(String path){
//            image = imread(path);
//            if (!image.data)
//                return false;
//            return true;
//        }
//
//        //����ͨ��
//        void splitChannels(){
//            split(image, channelsRGB);
//        }
//
//        //����ֱ��ͼ
//        void getHistogram(){
//            calcHist(&channelsRGB[0], 1, 0, Mat(), outputRGB[0], 1, hisSize, &ranges);
//            calcHist(&channelsRGB[1], 1, 0, Mat(), outputRGB[1], 1, hisSize, &ranges);
//            calcHist(&channelsRGB[2], 1, 0, Mat(), outputRGB[2], 1, hisSize, &ranges);
//
//            //�������bin��ֵ
//            for (int i = 0; i < hisSize[0]; ++i){
//                cout << i << "   B:" << outputRGB[0].at<float>(i);
//                cout << "   G:" << outputRGB[1].at<float>(i);
//                cout << "   R:" << outputRGB[2].at<float>(i) << endl;
//            }
//        }
//
//        //��ʾֱ��ͼ
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
//        string path = "F:\\ͼƬ����\\Picture\\��\\25\\1.jpg";
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
/////*  ȫ�ֱ�������������ʼ��    */
////Mat srcImage; //�����ͼƬ����
////Mat dstImage; //�����ͼƬ����
////MatND dstHist; //ֱ��ͼ���󣬶�Ӧ�ϰ汾�е�cvCreateHist����
////int g_hdims = 50;     // ����HIST�ĳ�ʼ������Խ��Խ��ȷ
////
/////* �ص��������� */
////void on_HIST(int t, void *);
////
////
/////*   ������   */
////int main(int argc, char** argv)
////{
////
////    srcImage = imread("F:\\ͼƬ����\\Picture\\����\\JPEG\\16\\1.jpg", 0);//"0"��ʾ����Ҷ�ͼ��
////    namedWindow("ԭͼ", 1);//��Ӧ�ϰ汾�е�cvNamedWindow( )
////    imshow("ԭͼ", srcImage);//��Ӧ�ϰ汾�е� cvShowImage����
////
////    createTrackbar("hdims", "ԭͼ", &g_hdims, 256, on_HIST);//��Ӧ�ɰ汾�е�cvCreateTrackbar( );
////    on_HIST(0, 0);//���ù������ص�����
////    cvWaitKey(0);
////    return 0;
////}
////
////
/////*      �������ص�����       */
////void on_HIST(int t, void *)
////{
////    dstImage = Mat::zeros(512, 800, CV_8UC3);//ÿ�ζ�Ҫ��ʼ��
////    float hranges[] = { 0, 255 }; //�Ҷȷ�Χ
////    const float *ranges[] = { hranges };//�Ҷȷ�Χ��ָ��
////
////    if (g_hdims == 0)
////    {
////        printf("ֱ��ͼ��������Ϊ�㣡\n");
////    }
////    else
////    {
////        /*
////        srcImage:����ľ���
////        1:����ĸ���Ϊ1
////        0����Ϊ�Ҷ�ͼ���һ��ͨ��������ѡ0��ͨ��
////        Mat��������ʾ��ʹ����Ĥ
////        dstHist:�����Ŀ��ֱ��ͼ
////        1����Ҫ�����ֱ��ͼ��ά��Ϊ1
////        g_hdims:����HIST�ĸ���
////        ranges:��ʾÿһά�ȵ���ֵ��Χ
////        */
////        //int channels=0;
////        calcHist(&srcImage, 1, 0, Mat(), dstHist, 1, &g_hdims, ranges); // ����ֱ��ͼ��Ӧ�ϰ汾��cvCalcHist
////
////        /* ��ȡ�����Сֵ */
////        double max = 0;
////        minMaxLoc(dstHist, NULL, &max, 0, 0);// Ѱ�����ֵ����λ�ã���Ӧ�ɰ汾��cvGetMinMaxHistValue();
////
////        /*  ���ֱ��ͼ    */
////
////        double bin_w = (double)dstImage.cols / g_hdims;  // hdims: ���ĸ������� bin_w Ϊ���Ŀ��
////        double bin_u = (double)dstImage.rows / max;  //// max: ����������ظ������� bin_u Ϊ�������صĸ߶�
////
////        // ��ֱ��ͼ
////        for (int i = 0; i<g_hdims; i++)
////        {
////            Point p0 = Point(i*bin_w, dstImage.rows);//��Ӧ�ɰ汾�е�cvPoint����
////
////            int val = dstHist.at<float>(i);//ע��һ��Ҫ��float���ͣ���Ӧ�ɰ汾�е� cvGetReal1D(hist->bins,i);
////            Point p1 = Point((i + 1)*bin_w, dstImage.rows - val*bin_u);
////            rectangle(dstImage, p0, p1, cvScalar(0, 255), 1, 8, 0);//��Ӧ�ɰ��е�cvRectangle();
////        }
////
////        /*   ���̶�   */
////        char string[12];//���ת����ʮ��������ת����ʮ���ƺ��λ��������12λ�������������Լ��趨
////        //��������̶ȣ����ظ�����
////        int kedu = 0;
////        for (int i = 1; kedu<max; i++)
////        {
////            kedu = i*max / 10;//�˴�ѡ��10���̶�
////            itoa(kedu, string, 10);//��һ������ת��Ϊ�ַ�����������е�10ָʮ����
////            //��ͼ������ʾ�ı��ַ���
////            putText(dstImage, string, Point(0, dstImage.rows - kedu*bin_u), 1, 1, Scalar(0, 255, 255));//��Ӧ�ɰ��е�cvPutText����
////        }
////        //��������̶ȣ����ػҶ�ֵ��
////        kedu = 0;
////        for (int i = 1; kedu<256; i++)
////        {
////            kedu = i * 20;//�˴�ѡ����Ϊ20
////            itoa(kedu, string, 10);//��һ������ת��Ϊ�ַ���
////            //��ͼ������ʾ�ı��ַ���
////            putText(dstImage, string, cvPoint(kedu*(dstImage.cols / 256), dstImage.rows), 1, 1, Scalar(0, 255, 255));
////        }
////        namedWindow("Histogram", 1);
////        imshow("Histogram", dstImage);
////    }
////
////}
//
