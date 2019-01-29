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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBMatcher.h"
#include <mutex>

using namespace std;

long unsigned int KeyFrame::nNextId=0;

static void AddCells(const CamModelGeneral::eFace face, const int &nMinCellX, const int &nMaxCellX, const int &nMinCellY, const int &nMaxCellY, 
        vector<size_t> &vIndices, const vector< vector< vector <vector<size_t> > > > &mGrid, const vector<cv::KeyPoint> &mvKeys, 
        const float &x, const float &y, const float &r)
{
    //Border check. It's necessary here since the grid range caller given may out of bdry
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
                const float distx = kp.pt.x-x;
                const float disty = kp.pt.y-y;
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
}

KeyFrame::KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnCubeFaces(CUBEMAP_FACES), mnCubeFaceGridCols(CUBEFACE_GRID_COLS), 
    mnCubeFaceGridRows(CUBEFACE_GRID_ROWS), mfGridElementLengthInv(F.mfGridElementLengthInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    N(F.N), mvKeys(F.mvKeys), mvKeyRays(F.mvKeyRays), mDescriptors(F.mDescriptors.clone()), mBowVec(F.mBowVec), mFeatVec(F.mFeatVec),
    mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor), mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), 
    mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), mK(F.mK),
    mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB), mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnCubeFaces);
    for(int i = 0; i < mnCubeFaces; i++)
    {
        mGrid[i].resize(mnCubeFaceGridCols);
        for(int j = 0; j < mnCubeFaceGridCols; j++)
        {
            mGrid[i][j].resize(mnCubeFaceGridRows);
            for(int k = 0; k < mnCubeFaceGridRows; k++)
                mGrid[i][j][k] = F.mGrid[i][j][k];
        }
    }

    SetPose(F.mTcw);    
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float  &y, const float  &r) const
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

    //searching window within current cube face
    if(bXInFace && bYInFace)
    {
        int nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
        int nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
        int nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
        int nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);

        AddCells(face, nMinCellX, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover upper face
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        //a reverse of X and Y would occur since coordinate doesn't consistent
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, vIndices, 
                                mGrid, mvKeys, x, y, r);
                    } else {// cover upper face
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMinCellY, nMinCellX, nMaxCellX, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    break; 
                }
            case CamModelGeneral::RIGHT_FACE:
                {
                    //cover lower face
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        //a reverse of X and Y would occur since coordinate doesn't consistent
                        AddCells(CamModelGeneral::LOWER_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - 1, nMinCellX, nMaxCellX, vIndices, mGrid, 
                                mvKeys, x, y, r);
                    } else {// cover upper face
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellY, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, 
                                vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    break; 
                }
            case CamModelGeneral::UPPER_FACE:
                {
                    //cover front face
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover back face
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::LOWER_FACE:
                {
                    //cover back face, should be discarded
                    if(bYOverflow) {
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover front face
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover left face
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover back face
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    }
                    break; 
                }
            case CamModelGeneral::RIGHT_FACE:
                {
                    //cover back face
                    if(bXOverflow) {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover front face
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::RIGHT_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_ROWS - nMinCellY - 1, 0, nMaxCellX, 
                                vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover left face
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellY, nMaxCellY, 0, nMinCellX, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellY, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS, vIndices, mGrid, mvKeys, x, y, r);
                        return vIndices;
                    } else { //cover left face
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - nMinCellY - 1, nMinCellX, 
                                CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, nMinCellY, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((yStart+nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellY, 0, CUBEFACE_GRID_ROWS - nMinCellX - 1, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMinCellY, nMinCellX, CUBEFACE_GRID_COLS - 1, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMinCellY, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
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
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - 1, nMinCellX, CUBEFACE_GRID_COLS - 1, 
                                vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1,  nMinCellY, CUBEFACE_GRID_ROWS - 1, 
                                vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1,  CUBEFACE_GRID_COLS - 1, 0, nMaxCellX, 
                                vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, CUBEFACE_GRID_COLS - nMinCellY - 1, CUBEFACE_GRID_COLS - 1, 0, CUBEFACE_GRID_ROWS - nMinCellX - 1, 
                                vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, CUBEFACE_GRID_COLS - nMinCellY - 1, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMaxCellX - 1, 
                                CUBEFACE_GRID_ROWS - 1 , vIndices, mGrid, mvKeys, x, y, r);
                    }
                    break; 
                }
            case CamModelGeneral::UPPER_FACE:
                {
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, CUBEFACE_GRID_ROWS - nMinCellY - 1, 0, nMaxCellX, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, nMinCellY, CUBEFACE_GRID_ROWS - 1, 0, nMinCellX, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellY = (int)floor((yEnd-nFaceH) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, CUBEFACE_GRID_COLS - nMaxCellY - 1, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::UPPER_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, nMaxCellY, 0, nMinCellX, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    break; 
                }
            case CamModelGeneral::LOWER_FACE:
                {
                    if(bXOverflow && bYOverflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS - 1, 
                                vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYOverflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMinCellY = (int)floor(yStart * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, nMinCellY, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((xStart+nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, 0, CUBEFACE_GRID_COLS - nMinCellY - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXOverflow && bYUnderflow)
                    {
                        nMinCellX = (int)floor(xStart * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMaxCellX = (int)floor((xEnd-nFaceW) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::RIGHT_FACE, 0, nMaxCellY, CUBEFACE_GRID_ROWS - nMaxCellX - 1, CUBEFACE_GRID_ROWS, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, nMinCellX, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
                    }
                    else if(bXUnderflow && bYUnderflow)
                    {
                        nMaxCellX = (int)floor(xEnd * mfGridElementLengthInv);
                        nMaxCellY = (int)floor(yEnd * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LOWER_FACE, 0, nMaxCellX, 0, nMaxCellY, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellX = (int)floor((-xStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::LEFT_FACE, CUBEFACE_GRID_COLS - nMinCellX - 1, CUBEFACE_GRID_COLS - 1, CUBEFACE_GRID_ROWS - nMinCellX - 1, 
                                CUBEFACE_GRID_ROWS, vIndices, mGrid, mvKeys, x, y, r);
                        nMinCellY = (int)floor((-yStart) * mfGridElementLengthInv);
                        AddCells(CamModelGeneral::FRONT_FACE, 0, nMaxCellX, CUBEFACE_GRID_ROWS - nMinCellY - 1, CUBEFACE_GRID_ROWS - 1, vIndices, mGrid, mvKeys, x, y, r);
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

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}
