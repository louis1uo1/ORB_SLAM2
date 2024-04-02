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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

/**
 * @brief 构造函数
 * @param Pos   地图点的世界坐标
 * @param pRefKF 该地图点所在的关键帧
 * @param pMap  地图点所在的地图
 */
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);//地图信息（深）拷贝给mWorldPos
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);//初始化平均观测方向为0

    //确保在创建地图点时对地图对象的操作是线程安全的。
    //将地图点的ID递增并分配给mnId，以保证地图点具有唯一的标识符。
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

/**
 * @brief 根据传入的帧信息初始化地图点对象，并计算出地图点的法向量、最小和最大观测距离等属性。
 * @param Pos  地图点的世界坐标
 * @param pMap 地图点所在的地图
 * @param pFrame 帧信息
 * @param idxF  地图点在创建时所属的帧中的特征点的索引
 * @note  之前的构造函数相比，这个构造函数除了接受位置信息和地图指针外，还接受了帧（Frame）指针和一个索引作为参数。
 */
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();//相机中心的世界坐标
    mNormalVector = mWorldPos - Ow;//世界坐标系下相机到3D点的向量 (当前关键帧的观测方向)
    mNormalVector = mNormalVector/cv::norm(mNormalVector);//归一化

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);//获取这个地图点的描述子到mDescriptor

    //确保在创建地图点时对地图对象的操作是线程安全的。
    //将地图点的ID递增并分配给mnId，以保证地图点具有唯一的标识符。
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

/**
 * @brief 设置地图点在世界坐标系下的坐标
 * @param Pos 世界坐标系下地图点的位姿
 */
void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

/**
 * @brief 获取当前地图点的世界坐标
 * @return 当前地图点的世界坐标
 */
cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

/**
 * @brief //世界坐标系下地图点被多个相机观测的平均观测方向
 * @return mNormalVector平均观测方向
 */
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

/**
 * @brief 获取地图点的参考关键帧
 * @return mpRefKF参考关键帧
 */
KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}


/**
 * @brief 添加地图点的新观测，并根据观测的情况更新地图点的观测次数
 * @param pKF 关键帧
 * @param idx 地图点在关键帧的索引
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    //如果已经添加了观测，则返回
    if(mObservations.count(pKF))
        return;
    //如果没有添加过观测，将关键帧pKF和索引idx存储到mObservations中
    mObservations[pKF]=idx;

    //更新地图点的观测次数
    if(pKF->mvuRight[idx]>=0)
        nObs+=2;//双目 or rgb-d 观测次数+2
    else
        nObs++;//单目 观测次数+1
}


/**
 * @brief 删除某个关键帧对当前地图点的观测
 * @param pKF 关键帧
 */
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        //检查地图点是否在该关键帧pKF中被观测
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];//获取地图点在关键帧中的索引idx。
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;//双目 rgb-d 观测次数-2
            else
                nObs--;//单目 观测次数-1

            mObservations.erase(pKF);//删除该关键帧
            //如果pKF是参考关键帧，则该帧被删除后重新指定新的参考关键帧RefFrame
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            //当观测到该点的相机数目少于2时，标记该点为坏点(至少需要两个观测才能三角化)
            if(nObs<=2)
                bBad=true;
        }
    }
    //如果被标记为坏点，则更新相关观测帧的信息
    if(bBad)
    SetBadFlag();
}

/**
 * @brief 获取成员变量mObservations
 * @return mObservation当前地图点在某关键帧keyframe中的索引
 */
map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}


/**
 * @brief 获取地图点的被观测次数
 * @return nObs地图点的被观测次数
 */
int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

/**
 * @brief 将地图点标记为坏点，删除该地图点并更新相关观测帧的信息。
 * @note 理解：删除该地提点并告知可以观测到该MapPoint的Frame：该MapPoint已被删除；
 *     删除关键点时利用线程锁采取先标记再清除的方式,
 */
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;//创建一个变量来存储地图点的观测信息
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;//标记坏点
        obs = mObservations;
        mObservations.clear();
    }

    //遍历obs中存储的关键帧和特征点索引信息，将关键帧与地图点的匹配信息移除，以确保关键帧不再持有该地图点。
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }
    //通知地图对象将该地图点从地图中移除
    mpMap->EraseMapPoint(this);
}

/**
 * @brief 获取用于替换的新点
 * @return mpReplaced
 */
MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

/**
 * @brief 替换地图点，并更新观测关系
 * @param pMP 用于替换的新点
 */
void MapPoint::Replace(MapPoint* pMP)
{
    //如果新点和被替换点是同一个点，返回
    if(pMP->mnId==this->mnId)
        return;
    // step1.将当地图点的数据叠加到新地图点上；逻辑上删除当前地图点
    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        //清除当前地图点的观测信息
        mObservations.clear();
        //相当于逻辑上删除
        mbBad=true;
        //暂存当前地图点的可视次数和被找到的次数
        nvisible = mnVisible;
        nfound = mnFound;
        //替换
        mpReplaced = pMP;
    }

    //step2.；遍历所有关键帧，将观测到当前地图点的关键帧的信息进行更新
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        //如果新点pMP不在该关键帧中，将当前地图点替换为目标地图点，并更新观测帧中的地图点信息
        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        //否则，直接删除当前地图点在该关键帧中的观测信息
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    //将当前地图点的观测数据等其他数据都"叠加"到新的地图点上
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    //更新pMP的特征描述子
    pMP->ComputeDistinctiveDescriptors();

    // step3.删除被替换的地图点
    mpMap->EraseMapPoint(this);
}

/**
 * @brief 获取坏点标记
 * @return mbBad
 */
bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

//??添加地图点的可视次数？？暂时不理解
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

//??添加能找到该地图点的帧数？？暂时不理解
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

//??计算被找到的比例??
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}



/**
 * @brief 计算地图点的特征描述子
 * @note 由于一个地图点在不同关键帧中对应不同的特征点和描述子，其特征描述子mDescriptor是其在所有观测关键帧中描述子的中位数
 *    (准确地说,该描述子与其他所有描述子的中值距离最小). 
 *    在函数ORBmatcher::SearchByProjection()和ORBmatcher::Fuse()中,
 *  通过比较地图点的特征描述子与图片特征点描述子,实现将地图点与图像特征点的匹配(3D-2D匹配).
 */
void MapPoint::ComputeDistinctiveDescriptors()
{
    vector<cv::Mat> vDescriptors;
    map<KeyFrame*,size_t> observations;
     //Step1获取该地图点所有有效的观测关键帧信息
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        //如果是坏点，返回
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // Step2遍历观测到该地图点的所有关键帧，将对应的描述子放到向量vDescriptors中
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    
    const size_t N = vDescriptors.size();
    float Distances[N][N];
    //Step3计算这些描述子两两之间的距离
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    int BestMedian = INT_MAX;
    int BestIdx = 0;
    // Step4选择特征描述子，它与其他描述子应该具有最小中位数距离
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

/**
 * @brief 获取描述子
 * @return mDescriptor描述子
 */
cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

/**
 * @brief 获取地图点在某个关键帧中索引
 * @param pKF 关键帧
 * @return 找到返回mObservations[pKF]
 *      没找到返回-1
 */
int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

/**
 * @brief 检查该地图点是否在关键帧中
 * @param pKF 关键帧
 * @return 存在返回true
 *      不存在返回false
 */
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}


/**
 * @brief 更新地图点的平均观测方向和平均观测距离
 * @note 只要地图点本身或关键帧对该地图点的观测发生变化,就应该调用函数UpdateNormalAndDepth()更新其观测尺度和方向信息
 */
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    //Step1获得观测到该地图点的所有关键帧、坐标等信息
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    //Step2计算该地图点的平均观测方向
    //遍历存储在observations中的关键帧和特征点索引信息，
    //对每个关键帧pKF，计算关键帧相机光心与地图点的连线，然后将该连线的单位向量加到normal中，并递增n。
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);
        n++;
    }
    //计算参考关键帧与地图点的距离，并根据参考关键帧的尺度信息计算地图点的最大和最小深度范围。
    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;
    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n;
    }
}

/**
 * @brief 获取地图点的最小平均观测距离
 * @return 0.8f*mfMinDistance
 */
float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

/**
 * @brief 获取地图点的最大平均观测距离
 * @return 1.2f*mfMaxDistance
 * @note    图解：
 * picture\平均观测距离.jpg
 */
float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

/**
 * @brief 预测地图点对应特征点所在的图像金字塔层级
 * @param currentDist 相机光心距离地图点距离
 * @param pKF 关键帧
 * @return nScale金字塔层级
 */
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);//取对数
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

/**
 * @brief 预测地图点对应特征点所在的图像金字塔层级
 * @param currentDist 相机光心距离地图点距离
 * @param pF 普通帧
 * @return nScale金字塔层级
 * @note 和另外一个和重载不同在于参数为pF普通帧
 */
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
