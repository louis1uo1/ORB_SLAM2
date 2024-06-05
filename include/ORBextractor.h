/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


namespace ORB_SLAM2
{
// 分配四叉树时用到的结点类型
class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    
    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    void operator()( cv::InputArray image, cv::InputArray mask,
      std::vector<cv::KeyPoint>& keypoints,
      cv::OutputArray descriptors);
      
    /**
     * @brief 获取图像金字塔的层数
     * @return int 图像金字塔的层数
     */
    int inline GetLevels(){
        return nlevels;}

    /**
     * @brief 获取当前提取器所在的图像的缩放因子，这个不带s的因子表示是相临近层之间的
     * @return float 当前提取器所在的图像的缩放因子，相邻层之间
     */
    float inline GetScaleFactor(){
        return scaleFactor;}

    /**
     * @brief 获取图像金字塔中每个图层相对于底层图像的缩放因子
     * @return std::vector<float> 图像金字塔中每个图层相对于底层图像的缩放因子
     */
    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    /**
     * @brief 获取上面的那个缩放因子s的倒数
     * @return std::vector<float> 倒数
     */
    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    /**
     * @brief 获取sigma^2，就是每层图像相对于初始图像缩放因子的平方，参考cpp文件中类构造函数的操作
     * @return std::vector<float> sigma^2
     */
    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    /**
     * @brief 获取上面sigma平方的倒数
     * @return std::vector<float> 
     */
    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;//存储图像金字塔的容器，一个矩阵存储一层图像

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    
    std::vector<cv::Point> pattern;//用于计算描述子的随机采样点集合

    int nfeatures;//在图像金字塔所有层级提取到的特征点数之和，从yaml配置文件中读取
    double scaleFactor;//图像金字塔相邻层级之间的缩放系数，从yaml配置文件中读取
    int nlevels;//金字塔层级数 从yaml配置文件中读取
    int iniThFAST;//初始的FAST角点检测阈值 从yaml配置文件中读取
    int minThFAST;//最小的FAST角点检测阈值 从yaml配置文件中读取

    std::vector<int> mnFeaturesPerLevel;//每层金字塔中提取的特征点数（正比于图层边长，总和为nfeatures）

    std::vector<int> umax;//计算特征点方向的时候，有个圆形的图像区域，这个vector中存储了每行u轴的边界（四分之一，其他部分通过对称获得）

    std::vector<float> mvScaleFactor;//各层金字塔的缩放系数
    std::vector<float> mvInvScaleFactor;// 各层金字塔的缩放系数的倒数
    std::vector<float> mvLevelSigma2;//各层金字塔的缩放系数的平方
    std::vector<float> mvInvLevelSigma2;//各层金字塔的缩放系数的平方的倒数
};

} //namespace ORB_SLAM

#endif

