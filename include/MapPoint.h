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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;


class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;//第一次观测到该地图点的关键帧的id
    long int mnFirstFrame;//?和mnFirstKFid的区别在哪?
    int nObs;//记录了当前地图点被多少个关键帧相机观测到了(单目关键帧每次观测算1个相机,双目/RGBD帧每次观测算2个相机).

    //!下面几个变量用于tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;//?暂时不清楚具体含义
    long unsigned int mnLastFrameSeen;//该地图点最后一次被观测到的帧的ID

    //!下面两个变量用于local mapping
    long unsigned int mnBALocalForKF;//该地图点在局部BA优化中关键帧的id
    /*一般来说，在SLAM系统中，当一个新的关键帧被加入时，会尝试将新的观测与地图点进行匹配。
    如果某个地图点在这个新的关键帧中没有被观测到，但是与该关键帧周围的帧有着较好的匹配，那么它可能被标记为融合候选关键帧。
    这种情况下，mnFuseCandidateForKF 将被设置为该关键帧的ID，以指示这个地图点是一个融合候选。*/
    long unsigned int mnFuseCandidateForKF;

    //!下面几个变量用于loop closing
    /*标记地图点象是否被用作回环检测中的候选点；
    可以帮助系统在回环检测过程中进行快速的候选点匹配，并有助于提高回环检测的准确性和效率。
    通过标记地图点是否为回环候选点，系统可以更有效地管理和利用地图点的信息，以支持更可靠的回环检测。*/
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;//记录地图点是否已经被某个关键帧（KeyFrame）进行过位置校正
    long unsigned int mnCorrectedReference;//记录地图点被校正时的参考关键帧id    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;//记录地图点在全局BA优化中的关键帧（KeyFrame）的ID。


    static std::mutex mGlobalMutex;//当前地图点的锁

protected:    
     cv::Mat mWorldPos;// 地图点的世界坐标

     std::map<KeyFrame*,size_t> mObservations;// 当前地图点在某关键帧keyframe中的索引

     cv::Mat mNormalVector;//平均观测方向

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     //通常情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;//?地图点的可视次数??
     int mnFound;//?地图点的可找到次数??

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;//坏点标记
     MapPoint* mpReplaced;//用来替换当前地图点的新点

     // Scale invariance distances
     float mfMinDistance;//最小平均观测距离
     float mfMaxDistance;//最大平均观测距离

     Map* mpMap;//声明一个Map类的变量，作用：将地图点与其所属的地图实体联系起来，以便在需要时可以方便地访问地图的其他部分或执行地图相关的操作。

     std::mutex mMutexPos;//当前地图点位姿的锁
     std::mutex mMutexFeatures;//当前地图点的特征信息的锁
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
