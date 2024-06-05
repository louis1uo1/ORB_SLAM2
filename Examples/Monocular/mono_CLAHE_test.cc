#include <iostream>
#include <fstream>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
using namespace ORB_SLAM2;

int main(int argc, char **argv)
{
    // 读取配置文件
    string strSettingPath = "underwate_cave.yaml"; // 配置文件路径
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // 读取图片
    Mat src_img = imread("frame_ud_009267.png");
    if (src_img.empty())
        cout << "the image is empty!" << endl;

    // 通道转换BGR2YUV
    Mat src_yuv_img;
    cvtColor(src_img, src_yuv_img, COLOR_BGR2YUV);
    // YUV通道分离
    vector<Mat> channels;
    Mat color_y, color_u, color_v;
    split(src_yuv_img, channels);
    color_y = channels.at(0);
    color_u = channels.at(1);
    color_v = channels.at(2);

    // 对y（灰度）通道进行直方图均衡化处理
    Mat AHE_y_img;
    equalizeHist(color_y, AHE_y_img);

    // 对y（灰度）通道进行限制对比度的自适应直方图均衡化处理
    Ptr<CLAHE> clahe = createCLAHE(10, Size(8, 8));
    Mat CLAHE_y_img;
    clahe->apply(color_y, CLAHE_y_img);

    // 通道合成 转化为bgr
    Mat AHE_img;
    Mat CLAHE_img;
    vector<Mat> channels_AHE;
    vector<Mat> channels_CLAHE;

    channels_AHE.push_back(AHE_y_img);
    channels_AHE.push_back(color_u);
    channels_AHE.push_back(color_v);

    channels_CLAHE.push_back(CLAHE_y_img);
    channels_CLAHE.push_back(color_u);
    channels_CLAHE.push_back(color_v);

    merge(channels_AHE, AHE_img);
    merge(channels_CLAHE, CLAHE_img);
    cvtColor(AHE_img, AHE_img, COLOR_YUV2BGR);
    cvtColor(CLAHE_img, CLAHE_img, COLOR_YUV2BGR);

    // 处理后图片对比
    namedWindow("原图", CV_WINDOW_NORMAL);
    imshow("原图", src_img);
    imwrite("./test_result/原图.png", src_img);

    namedWindow("AHE", CV_WINDOW_NORMAL);
    imshow("AHE", AHE_img);
    imwrite("./test_result/AHE.png", AHE_img);

    namedWindow("CLAHE", CV_WINDOW_NORMAL);
    imshow("CLAHE", CLAHE_img);
    imwrite("./test_result/CLAHE.png", CLAHE_img);

    // 绘制直方图
    Mat hist_src, hist_AHE, hist_CLAHE;
    int histSize = 255;
    float range[] = {0, 255};
    const float *histRange = {range};

    calcHist(&color_y, 1, 0, Mat(), hist_src, 1, &histSize, &histRange, true, false);
    calcHist(&AHE_y_img, 1, 0, Mat(), hist_AHE, 1, &histSize, &histRange, true, false);
    calcHist(&CLAHE_y_img, 1, 0, Mat(), hist_CLAHE, 1, &histSize, &histRange, true, false);

    int hist_w = 400;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    // 创建直方图画布
    Mat histImage_src(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
    Mat histImage_AHE(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
    Mat histImage_CLAHE(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

    normalize(hist_src, hist_src, 0, histImage_src.rows, NORM_MINMAX, -1, Mat());
    normalize(hist_AHE, hist_AHE, 0, histImage_AHE.rows, NORM_MINMAX, -1, Mat());
    normalize(hist_CLAHE, hist_CLAHE, 0, histImage_CLAHE.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage_src, Point(bin_w * (i - 1), hist_h - cvRound(hist_src.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist_src.at<float>(i))),
             Scalar(255, 255, 255), 2, 8, 0);
        line(histImage_AHE, Point(bin_w * (i - 1), hist_h - cvRound(hist_AHE.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist_AHE.at<float>(i))),
             Scalar(255, 255, 255), 2, 8, 0);
        line(histImage_CLAHE, Point(bin_w * (i - 1), hist_h - cvRound(hist_CLAHE.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist_CLAHE.at<float>(i))),
             Scalar(255, 255, 255), 2, 8, 0);
    }

    namedWindow("直方图_原图", CV_WINDOW_NORMAL);
    imshow("直方图_原图", histImage_src);
    imwrite("./test_result/直方图_原图.png", histImage_src);

    namedWindow("直方图_AHE", CV_WINDOW_NORMAL);
    imshow("直方图_AHE", histImage_AHE);
    imwrite("./test_result/直方图_AHE.png", histImage_AHE);

    namedWindow("直方图_CLAHE", CV_WINDOW_NORMAL);
    imshow("直方图_CLAHE", histImage_CLAHE);
    imwrite("./test_result/直方图_CLAHE.png", histImage_CLAHE);

    // 提取特征点
    std::vector<cv::KeyPoint> mvKeys, mvKeys1, mvKeys2;
    cv::Mat mDescriptors, mDescriptors1, mDescriptors2;
    ORBextractor *IniORBextractor;
    IniORBextractor = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    (*IniORBextractor)(src_img, cv::Mat(), mvKeys, mDescriptors);
    (*IniORBextractor)(AHE_img, cv::Mat(), mvKeys1, mDescriptors1);
    (*IniORBextractor)(CLAHE_img, cv::Mat(), mvKeys2, mDescriptors2);

    string name_src="原图_特征点_"+to_string(mvKeys.size());
    string name_AHE="AHE_特征点_"+to_string(mvKeys1.size());
    string name_CLAHE="CLAHE_特征点_"+to_string(mvKeys2.size());

    cout << mvKeys.size() << endl;
    drawKeypoints(src_img, mvKeys, src_img);
    namedWindow(name_src, CV_WINDOW_NORMAL);
    imshow(name_src, src_img);
    imwrite(("./test_result/"+name_src+".png"), src_img);

    cout << mvKeys1.size() << endl;
    drawKeypoints(AHE_img, mvKeys1, AHE_img);
    namedWindow(name_AHE, CV_WINDOW_NORMAL);
    imshow(name_AHE, AHE_img);
    imwrite(("./test_result/"+name_AHE+".png"), AHE_img);

    cout << mvKeys2.size() << endl;
    drawKeypoints(CLAHE_img, mvKeys2, CLAHE_img);
    namedWindow(name_CLAHE, CV_WINDOW_NORMAL);
    imshow(name_CLAHE, CLAHE_img);
    imwrite(("./test_result/"+name_CLAHE+".png"), CLAHE_img);
    
    waitKey(0);
}
