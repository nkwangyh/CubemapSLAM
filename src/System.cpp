/**
* This file is part of CubemapSLAM.
*
* Copyright (C) 2017-2019 Yahui Wang <nkwangyh at mail dot nankai dot edu dot cn> (Nankai University)
* For more information see <https://github.com/nkwangyh/CubemapSLAM>
*
* CubemapSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* CubemapSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with CubemapSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

/*
* CubemapSLAM is based on ORB-SLAM2 and Multicol-SLAM which were also released under GPLv3
* For more information see <https://github.com/raulmur/ORB_SLAM2>
* Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* and <https://github.com/urbste/MultiCol-SLAM>
* Steffen Urban <urbste at googlemail.com>
*/

#include "System.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <iostream>

using namespace std;

System::System(const string &strVocFile, const string &strSettingsFile, const bool bUseViewer):
   mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)

{
    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(EXIT_FAILURE);
    }

    //Load ORB Vocabulary
    std::cout << std::endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        std::cerr << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Falied to open at: " << strVocFile << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Vocabulary loaded!" << std::endl << std::endl;

    //read in params
    int nrpol = fsSettings["Camera.nrpol"];
    int nrinvpol = fsSettings["Camera.nrinvpol"];

    cv::Mat_<double> poly = cv::Mat::zeros(5, 1, CV_64F);
    for (int i = 0; i < nrpol; ++i)
        poly.at<double>(i, 0) = fsSettings["Camera.a" + std::to_string(i)];
    cv::Mat_<double>  invpoly = cv::Mat::zeros(12, 1, CV_64F);
    for (int i = 0; i < nrinvpol; ++i)
        invpoly.at<double>(i, 0) = fsSettings["Camera.pol" + std::to_string(i)];

    int Iw = (int)fsSettings["Camera.Iw"];
    int Ih = (int)fsSettings["Camera.Ih"];

    double cdeu0v0[5] = { fsSettings["Camera.c"], fsSettings["Camera.d"], fsSettings["Camera.e"],
        fsSettings["Camera.u0"], fsSettings["Camera.v0"] };
    
    //cubemap params
    int nFaceH = fsSettings["CubeFace.h"];
    int nFaceW = fsSettings["CubeFace.w"];
    double fx = static_cast<double>(nFaceW)/2, fy = static_cast<double>(nFaceH)/2;
    double cx = static_cast<double>(nFaceW)/2, cy = static_cast<double>(nFaceH)/2;

    double camFov = fsSettings["Camera.fov"];

    //Set camera model
    CamModelGeneral::GetCamera()->SetCamParams(cdeu0v0, poly, invpoly, Iw, Ih, fx, fy, cx, cy, nFaceW, nFaceH, camFov);
    std::cout << "finish creating general camera model" << std::endl;
    mnWithFisheyeMask = fsSettings["Camera.withFisheyeMask"];

    //Create mapping from cubemap to fisheye
    CreateUndistortRectifyMap();

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();
    
    //Create Drawers. These are used by the Viewer
    int nMarginH = fsSettings["FrameDrawer.MarginX"];
    int nMarginV = fsSettings["FrameDrawer.MarginY"];
    mpFrameDrawer = new FrameDrawer(mpMap,nFaceH,nFaceW,nMarginH,nMarginV);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
           mpMap, mpKeyFrameDatabase, strSettingsFile);
    
    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap);
    //Don't run localmapping as a thread until it's fine tuned
    mptLocalMapping = new thread(&LocalMapping::Run, mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, false);
    mptLoopClosing = new thread(&LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }
    
    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackCubemap(const cv::Mat &im, const cv::Mat &mask, const double &timestamp)
{
    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }
    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageCubemap(im, mask, timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    //output tracking summary
    mpViewer->OutputTrackingSummary();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("Map Viewer");
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

//convert fisheye image to cubemap
void System::CvtFisheyeToCubeMap_reverseQuery(cv::Mat &cubemapImg, const cv::Mat &fisheyeImg)
{
    //clear rectified image
    cubemapImg.setTo(cv::Scalar::all(0));
    int width3 = CamModelGeneral::GetCamera()->GetCubeFaceWidth() * 3, height3 = CamModelGeneral::GetCamera()->GetCubeFaceHeight() * 3;
    int Iw = CamModelGeneral::GetCamera()->GetFisheyeWidth(), Ih = CamModelGeneral::GetCamera()->GetFisheyeHeight();
    for(int i = 0; i < width3; ++i)
    {
        for(int j = 0; j < height3; ++j)
        {
            double u, v;
            CamModelGeneral::GetCamera()->CubemapToFisheye(u, v, static_cast<double>(i), static_cast<double>(j));
            //int ui = cvRound(u), vi = cvRound(v);
            int ui = std::floor(u), vi = std::floor(v);
            // on some face but doesn't map to a fisheye valid region
            if(ui < 0 || vi < 0 || ui >= Iw || vi >= Ih)
                continue;

            uchar intensity = fisheyeImg.at<uchar>(vi, ui);
            cubemapImg.at<uchar>(j, i) = intensity;
        }
    }
}

void System::CreateUndistortRectifyMap()
{
    int width3 = CamModelGeneral::GetCamera()->GetCubeFaceWidth() * 3, height3 = CamModelGeneral::GetCamera()->GetCubeFaceHeight() * 3;
    //create map for u(mMap1) and v(mMap2)
    mMap1.create(height3, width3, CV_32F);
    mMap2.create(height3, width3, CV_32F);
    mMap1.setTo(cv::Scalar::all(0));
    mMap2.setTo(cv::Scalar::all(0));

    int Iw = CamModelGeneral::GetCamera()->GetFisheyeWidth(), Ih = CamModelGeneral::GetCamera()->GetFisheyeHeight();
    for(int y = 0; y < height3; ++y)
    {
        for(int x = 0; x < width3; ++x)
        {
            double u, v;
            CamModelGeneral::GetCamera()->CubemapToFisheye(u, v, static_cast<double>(x), static_cast<double>(y));
            // on some face but doesn't map to a fisheye valid region
            if(u < 0 || v < 0 || u >= Iw || v >= Ih)
                continue;
            mMap1.at<float>(y, x) = static_cast<float>(u);
            mMap2.at<float>(y, x) = static_cast<float>(v);
        }
    }
}

//convert fisheye image to cubemap
void System::CvtFisheyeToCubeMap_reverseQuery_withInterpolation(cv::Mat &cubemapImg, const cv::Mat &fisheyeImg, 
                int interpolation, int borderType, const cv::Scalar& borderValue)
{
    const int offset = 0;
    const int width = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), height = CamModelGeneral::GetCamera()->GetCubeFaceHeight();
    cv::Mat cubemapImg_front = cubemapImg.rowRange(height, 2 * height).colRange(width, 2 * width);
    cv::Mat cubemapImg_left = cubemapImg.rowRange(height, 2 * height).colRange(0+offset, width+offset);
    cv::Mat cubemapImg_right = cubemapImg.rowRange(height, 2 * height).colRange(2 * width-offset, 3 * width-offset);
    cv::Mat cubemapImg_upper = cubemapImg.rowRange(0+offset, height+offset).colRange(width, 2 * width);
    cv::Mat cubemapImg_lower = cubemapImg.rowRange(2 * height-offset, 3 * height-offset).colRange(width, 2 * width);

    cv::Mat mMap1_front = mMap1.rowRange(height, 2 * height).colRange(width, 2 * width);
    cv::Mat mMap1_left = mMap1.rowRange(height, 2 * height).colRange(0+offset, width+offset);
    cv::Mat mMap1_right = mMap1.rowRange(height, 2 * height).colRange(2 * width-offset, 3 * width-offset);
    cv::Mat mMap1_upper = mMap1.rowRange(0+offset, height+offset).colRange(width, 2 * width);
    cv::Mat mMap1_lower = mMap1.rowRange(2 * height-offset, 3 * height-offset).colRange(width, 2 * width);

    cv::Mat mMap2_front = mMap2.rowRange(height, 2 * height).colRange(width, 2 * width);
    cv::Mat mMap2_left = mMap2.rowRange(height, 2 * height).colRange(0+offset, width+offset);
    cv::Mat mMap2_right = mMap2.rowRange(height, 2 * height).colRange(2 * width-offset, 3 * width-offset);
    cv::Mat mMap2_upper = mMap2.rowRange(0+offset, height+offset).colRange(width, 2 * width);
    cv::Mat mMap2_lower = mMap2.rowRange(2 * height-offset, 3 * height-offset).colRange(width, 2 * width);
    //interpolation with cv::remap for each face
    cv::remap(fisheyeImg, cubemapImg_front, mMap1_front, mMap2_front, interpolation, borderType, borderValue);
    cv::remap(fisheyeImg, cubemapImg_left, mMap1_left, mMap2_left, interpolation, borderType, borderValue);
    cv::remap(fisheyeImg, cubemapImg_right, mMap1_right, mMap2_right, interpolation, borderType, borderValue);
    cv::remap(fisheyeImg, cubemapImg_upper, mMap1_upper, mMap2_upper, interpolation, borderType, borderValue);
    cv::remap(fisheyeImg, cubemapImg_lower, mMap1_lower, mMap2_lower, interpolation, borderType, borderValue);
}

//convert fisheye to cubemap with pCamModel->FisheyeToCubemap
void System::CvtFisheyeToCubeMap(cv::Mat &cubemapImg, const cv::Mat &fisheyeImg)
{
    cubemapImg.setTo(cv::Scalar::all(0));
    //int width = CamModelGeneral::GetCamera()->mWCubeFace * 3, height = CamModelGeneral::GetCamera()->mHCubeFace * 3;
    int Iw = CamModelGeneral::GetCamera()->GetFisheyeWidth(), Ih = CamModelGeneral::GetCamera()->GetFisheyeHeight();

    for(int i = 0; i < Iw; ++i)
    {
        for(int j = 0; j < Ih; ++j) 
        {
            float ui, vi;
            uchar intensity = fisheyeImg.at<uchar>(j, i);
            CamModelGeneral::eFace face = CamModelGeneral::GetCamera()->FisheyeToCubemap(static_cast<float>(i), static_cast<float>(j), ui, vi);
            int u = cvRound(ui), v = cvRound(vi);
            if(face == CamModelGeneral::UNKNOWN_FACE || u < 0 || v < 0)
                continue;
            cubemapImg.at<uchar>(v, u) = intensity;
        }
    }
    cv::imshow("cubemap_from_fisheye", cubemapImg);
}
