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

#include "Frame.h"

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementLengthInv;
float Frame::mfGridElementLength;

static void AddCells(const CamModelGeneral::eFace face, const int &nMinCellX, const int &nMaxCellX, const int &nMinCellY, const int &nMaxCellY, 
        vector<size_t> &vIndices, const vector<size_t> mGrid[][CUBEFACE_GRID_COLS][CUBEFACE_GRID_ROWS], const vector<cv::KeyPoint> &mvKeys, 
        const float &x, const float &y, const float &r, const int &minLevel, const int &maxLevel, const bool &bCheckLevels = true)
{
    const int minCellX = max(0, nMinCellX), maxCellX = min(CUBEFACE_GRID_COLS-1, nMaxCellX);
    const int minCellY = max(0, nMinCellY), maxCellY = min(CUBEFACE_GRID_ROWS-1, nMaxCellY);

    for(int ix = minCellX; ix<=maxCellX; ix++)
    {
        for(int iy = minCellY; iy<=maxCellY; iy++)
        {
            const vector<size_t> &vCell = mGrid[face][ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kp = mvKeys[vCell[j]];
                if(bCheckLevels)
                {
                    if(kp.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kp.octave>maxLevel)
                            continue;
                }

                //This should be removed since the image plane coordinate doesn't coincide with cube faces
                const float distx = kp.pt.x-x;
                const float disty = kp.pt.y-y;
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
}

Frame::Frame()
{}

Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractor(frame.mpORBextractor),
     mTimeStamp(frame.mTimeStamp), N(frame.N), mvKeys(frame.mvKeys), mvKeyRays(frame.mvKeyRays),
     mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec), mDescriptors(frame.mDescriptors.clone()), mvpMapPoints(frame.mvpMapPoints), 
     mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),mpReferenceKF(frame.mpReferenceKF),mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mK(frame.mK)
{
    for(int i = 0; i < CUBEMAP_FACES; i++)
        for(int j = 0; j < CUBEFACE_GRID_COLS; j++)
            for(int k = 0; k < CUBEFACE_GRID_ROWS; k++)
                mGrid[i][j][k] = frame.mGrid[i][j][k];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

inline void DrawKeyPoints(const cv::Mat &im, const std::vector<cv::KeyPoint> &kpt)
{
    cv::Mat imColor;
    cv::cvtColor(im, imColor, CV_GRAY2BGR);
    cv::drawKeypoints(imColor, kpt, imColor);
    cv::imshow("kpts", imColor);
    cv::waitKey(0);
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &mask, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc)
    :mpORBvocabulary(voc), mpORBextractor(extractor), mTimeStamp(timeStamp), mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractor->GetLevels();
    mfScaleFactor = mpORBextractor->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractor->GetScaleFactors();
    mvInvScaleFactors = mpORBextractor->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractor->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractor->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(imGray, mask);

    mK = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0,0) = CamModelGeneral::GetCamera()->Get_fx();
    mK.at<float>(1,1) = CamModelGeneral::GetCamera()->Get_fy();
    mK.at<float>(0,2) = CamModelGeneral::GetCamera()->Get_cx();
    mK.at<float>(1,2) = CamModelGeneral::GetCamera()->Get_cy();
    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    ////for debug
    //DrawKeyPoints(imGray, mvKeys);

    ComputeKeyPointRays();

    //The MapPoints are corresponding to extracted features
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        mnMinX = 0.0f;
        mnMaxX = imGray.cols;
        mnMinY = 0.0f;
        mnMaxY = imGray.rows;

        mfGridElementLengthInv=static_cast<float>(3*CUBEFACE_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementLength = static_cast<float>(mnMaxX-mnMinX) / static_cast<float>(3*CUBEFACE_GRID_COLS);

        mbInitialComputations=false;
    }

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(CUBEMAP_FACES * CUBEFACE_GRID_ROWS * CUBEFACE_GRID_COLS);
    for(unsigned int i = 0; i < CUBEMAP_FACES; i++)
        for (unsigned int j = 0; j < CUBEFACE_GRID_COLS; j++)
            for(unsigned int k = 0; k < CUBEFACE_GRID_ROWS; k++)
                mGrid[i][j][k].reserve(nReserve);
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];

        CamModelGeneral::eFace face;
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp, face, nGridPosX, nGridPosY))
        {
            mGrid[face][nGridPosX][nGridPosY].push_back(i);
        }
    }
}

void Frame::ExtractORB(const cv::Mat &im, const cv::Mat &mask)
{
    (*mpORBextractor)(im, mask, mvKeys, mDescriptors);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Project in image and check it is not outside
    float u, v;
    CamModelGeneral::eFace face = CamModelGeneral::GetCamera()->TransformRaysToCubemap(u, v, PcX, PcY, PcZ);
    if(face == CamModelGeneral::UNKNOWN_FACE)
        return false;
    
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // A lookup table for neighbour cube faces
    // The searching center should be reside in some cube face
    CamModelGeneral::eFace face = CamModelGeneral::GetCamera()->FaceInCubemap<float>(x, y);
    if(face == CamModelGeneral::UNKNOWN_FACE)
        return vIndices;

    // Assume search radius should not large than half of grid width or height
    // so that the number of faces covered by search window will not larger than 3
    // And the face coordinates should coincide with the whole cubemap
    int nFaceW = CamModelGeneral::GetCamera()->GetCubeFaceWidth(), nFaceH = CamModelGeneral::GetCamera()->GetCubeFaceHeight();
    int nCornerX = static_cast<int>(x)/nFaceW * nFaceW;
    int nCornerY = static_cast<int>(y)/nFaceH * nFaceH;
    CamModelGeneral::eFace gridCornerFace = CamModelGeneral::GetCamera()->FaceInCubemap<float>(nCornerX, nCornerY);
    assert(face == gridCornerFace && "warning: face of feature should be cosistent with grid corner");

    float xInCurFace = x - nCornerX;
    float yInCurFace = y - nCornerY;
    //search window [-r, r]*[-r, r]
    float xStart = xInCurFace - r, xEnd = xInCurFace + r;
    float yStart = yInCurFace - r, yEnd = yInCurFace + r;
    //when x or y less than 0, set underflow as true; Otherwise set overflow as true;
    const bool bXUnderflow = (xStart < 0), bXOverflow = (xEnd > nFaceW - 1);
    const bool bYUnderflow = (yStart < 0), bYOverflow = (yEnd > nFaceH - 1);
    const bool bXInFace = (!bXOverflow && !bXUnderflow);
    const bool bYInFace = (!bYOverflow && !bYUnderflow);

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);
    //searching window within current cube face
    if(bXInFace && bYInFace)
    {
        int nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
        int nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
        int nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
        int nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);

        AddCells(face, nMinCellX, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
        return vIndices;
    }
    else if(bXInFace && !bYInFace) // X direction within face only
    {
        int nMinCellX, nMaxCellX, nMinCellY, nMaxCellY;
        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
        switch(face) {
            case CamModelGeneral::FRONT_FACE:
                {
                    //cover lower face
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover upper face
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::LEFT_FACE:
                {
                    //cover lower face
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        //a reverse of X and Y would occur since coordinate doesn't consistent
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, vIndices, 
                                mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    } else {// cover upper face
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMinCellY, nMinCellX, nMaxCellX, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }
            case CamModelGeneral::RIGHT_FACE:
                {
                    //cover lower face
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        //a reverse of X and Y would occur since coordinate doesn't consistent
                        AddCells(CamModelGeneral::LOWER_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - 1, nMinCellX, nMaxCellX, vIndices, mGrid, 
                                mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    } else {// cover upper face
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellY, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }
            case CamModelGeneral::UPPER_FACE:
                {
                    //cover front face
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover back face
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::LOWER_FACE:
                {
                    //cover back face, should be discarded
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover front face
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            default:
                return vIndices;
        }
        return vIndices;
    }
    else if(!bXInFace && bYInFace) // Y direction within face only
    {
        int nMinCellX, nMaxCellX, nMinCellY, nMaxCellY;
        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
        switch(face) {
            case CamModelGeneral::FRONT_FACE:
                {
                    //cover right face
                    if(bXOverflow) {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover left face
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::LEFT_FACE:
                {
                    //cover front face
                    if(bXOverflow) {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover back face
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::RIGHT_FACE:
                {
                    //cover back face
                    if(bXOverflow) {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover front face
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::UPPER_FACE:
                {
                    //cover right face
                    if(bXOverflow) {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::RIGHT_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_ROWS - nMinCellY - 1, 0, nMaxCellX, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover left face
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellY, nMaxCellY, 0, nMinCellX, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::LOWER_FACE:
                {
                    //cover right face
                    if(bXOverflow) {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellY, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    } else { //cover left face
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - nMinCellY - 1, nMinCellX, 
                                CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        return vIndices;
                    }
                    break; 
                }
            default:
                return vIndices;
        }
        return vIndices;
    }
    else //neither X nor Y with current face
    {
        int nMinCellX, nMaxCellX, nMinCellY, nMaxCellY;
        switch(face) {
            case CamModelGeneral::FRONT_FACE:
                {
                    //four corners
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }
            case CamModelGeneral::LEFT_FACE:
                {
                    //four corners
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellY, 0, CUBEFACE_GRID_ROWS - nMinCellX - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMinCellY, nMinCellX, CUBEFACE_GRID_COLS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMinCellY, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }
            case CamModelGeneral::RIGHT_FACE:
                {
                    //four corners
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - 1, nMinCellX, CUBEFACE_GRID_COLS - 1, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1,  nMinCellY, CUBEFACE_GRID_ROWS - 1, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1,  CUBEFACE_GRID_COLS - 1, 0, nMaxCellX, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, CUBEFACE_GRID_COLS - nMinCellY - 1, CUBEFACE_GRID_COLS - 1, 0, CUBEFACE_GRID_ROWS - nMinCellX - 1, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, CUBEFACE_GRID_COLS - nMinCellY - 1, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMaxCellX - 1, 
                                CUBEFACE_GRID_ROWS - 1 , vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }
            case CamModelGeneral::UPPER_FACE:
                {
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, CUBEFACE_GRID_ROWS - nMinCellY - 1, 0, nMaxCellX, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellY, CUBEFACE_GRID_ROWS - 1, 0, nMinCellX, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellY, 0, nMinCellX, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }
            case CamModelGeneral::LOWER_FACE:
                {
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - 1, 
                                vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, CUBEFACE_GRID_COLS - nMinCellY - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, 
                                CUBEFACE_GRID_ROWS, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r, minLevel, maxLevel, bCheckLevels);
                    }
                    break; 
                }

            default:
                return vIndices;
        }
        return vIndices;
    }
    return vIndices;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, CamModelGeneral::eFace &face, int &posX, int &posY)
{
    face = CamModelGeneral::GetCamera()->FaceInCubemap(kp.pt);
    if(face == CamModelGeneral::UNKNOWN_FACE)
        return false;

    posX = static_cast<int>((kp.pt.x-mnMinX)*mfGridElementLengthInv);
    posY = static_cast<int>((kp.pt.y-mnMinY)*mfGridElementLengthInv);

    //left-upper corner of current grid; face of the corner should be same as original feature
    CamModelGeneral::eFace gridCornerFace = CamModelGeneral::GetCamera()->FaceInCubemap<float>(static_cast<float>(posX*mfGridElementLength), static_cast<float>(posY*mfGridElementLength));
    assert(face == gridCornerFace && "warning: face of feature should be cosistent with grid corner");

    posX = posX % CUBEFACE_GRID_COLS;
    posY = posY % CUBEFACE_GRID_ROWS;
    return true;
}

void Frame::ComputeKeyPointRays()
{
    int keySize = mvKeys.size();
    mvKeyRays.resize(keySize);
    for(int i = 0; i < keySize; ++i)
    {
        //from cubemap to bearing vectors
        CamModelGeneral::eFace face = CamModelGeneral::GetCamera()->TransformCubemapToRays(mvKeyRays[i], mvKeys[i].pt);
        if(face == CamModelGeneral::UNKNOWN_FACE)
        {
            std::cout << "warning: feature from uknown face " << mvKeys[i].pt << std::endl;
            exit(-1);
        }
    }
}
