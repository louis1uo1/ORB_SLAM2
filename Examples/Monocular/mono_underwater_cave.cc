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

#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include "System.h"
using namespace std;
using namespace ORB_SLAM2;
using namespace cv;


void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void img_CLAHE(Mat &src_img);

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;

    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // Main loop
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni]);
        double tframe = vTimestamps[ni];
        
        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        img_CLAHE(im);
        SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = "/undistorted_frames_timestamps.txt";
    strPathTimeFile = strPathToSequence + strPathTimeFile;
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string f;
            ss >> f;
            ss >> t;
            vstrImageFilenames.push_back(f);
            vTimestamps.push_back(t);
        }
        //if (vTimestamps.size() > 1000)
           // break;
    }
    for (int i = 0; i < vstrImageFilenames.size(); i++)
    {
        vstrImageFilenames[i] = strPathToSequence + "/undistorted_frames/" + vstrImageFilenames[i];

    }
}

void img_CLAHE(Mat &src_img)
{

    // 通道转换BGR2YUV
    Mat src_yuv_img;
    cvtColor(src_img, src_yuv_img, COLOR_RGB2YUV);
    // YUV通道分离
    vector<Mat> channels;
    Mat color_y, color_u, color_v;
    split(src_yuv_img, channels);
    color_y = channels.at(0);
    color_u = channels.at(1);
    color_v = channels.at(2);

    // 对y（灰度）通道进行限制对比度的自适应直方图均衡化处理
    Ptr<CLAHE> clahe = createCLAHE(4, Size(8, 8));
    Mat CLAHE_y_img;
    clahe->apply(color_y, CLAHE_y_img);

    // 通道合成 转化为bgr
    Mat AHE_img;
    Mat CLAHE_img;
    vector<Mat> channels_CLAHE;

    channels_CLAHE.push_back(CLAHE_y_img);
    channels_CLAHE.push_back(color_u);
    channels_CLAHE.push_back(color_v);

    merge(channels_CLAHE, src_img);
    cvtColor(src_img, src_img, COLOR_YUV2RGB);

}
