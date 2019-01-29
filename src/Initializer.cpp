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

#include<thread>

#include "ThirdParty/DBoW2/DUtils/Random.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "ORBMatcher.h"

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mvKeys1 = ReferenceFrame.mvKeys;
    mvKeyRays1 = ReferenceFrame.mvKeyRays;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;

    //for orb-based initialization
    mK = cv::Mat::eye(3, 3, CV_32F);
    mK.at<float>(0,0) = CamModelGeneral::GetCamera()->Get_fx();
    mK.at<float>(1,1) = CamModelGeneral::GetCamera()->Get_fy();
    mK.at<float>(0,2) = CamModelGeneral::GetCamera()->Get_cx();
    mK.at<float>(1,2) = CamModelGeneral::GetCamera()->Get_cy();
}

bool Initializer::InitializeWithRays(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeys;
    mvKeyRays2 = CurrentFrame.mvKeyRays;

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
    // Consider only Essential Matrix
    vector<bool> vbMatchesInliers;
    float score;
    cv::Mat E;

    FindEssential(vbMatchesInliers, score, E);

    return ReconstructE(vbMatchesInliers,E,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
}

void Initializer::FindEssential(vector<bool> &vbMatchesInliers, float &score, cv::Mat &E21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Vec3f> vRay1i(8);
    vector<cv::Vec3f> vRay2i(8);
    cv::Mat E21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];
            vRay1i[j] = mvKeyRays1[mvMatches12[idx].first];
            vRay2i[j] = mvKeyRays2[mvMatches12[idx].second];
        }

        cv::Mat E21i = ComputeE21(vRay1i,vRay2i);

        currentScore = CheckEssiential(E21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            E21 = E21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

cv::Mat Initializer::ComputeE21(const vector<cv::Vec3f> &vRay1,const vector<cv::Vec3f> &vRay2)
{
    const int N = vRay1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float x1 = vRay1[i](0);
        const float y1 = vRay1[i](1);
        const float z1 = vRay1[i](2);
        const float x2 = vRay2[i](0);
        const float y2 = vRay2[i](1);
        const float z2 = vRay2[i](2);

        A.at<float>(i,0) = x2*x1;
        A.at<float>(i,1) = x2*y1;
        A.at<float>(i,2) = x2*z1;
        A.at<float>(i,3) = y2*x1;
        A.at<float>(i,4) = y2*y1;
        A.at<float>(i,5) = y2*z1;
        A.at<float>(i,6) = z2*x1;
        A.at<float>(i,7) = z2*y1;
        A.at<float>(i,8) = z2*z1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Epre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Epre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

float Initializer::CheckEssiential(const cv::Mat &E21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float e11 = E21.at<float>(0,0);
    const float e12 = E21.at<float>(0,1);
    const float e13 = E21.at<float>(0,2);
    const float e21 = E21.at<float>(1,0);
    const float e22 = E21.at<float>(1,1);
    const float e23 = E21.at<float>(1,2);
    const float e31 = E21.at<float>(2,0);
    const float e32 = E21.at<float>(2,1);
    const float e33 = E21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::Vec3f &kpRay1 = mvKeyRays1[mvMatches12[i].first];
        const cv::Vec3f &kpRay2 = mvKeyRays2[mvMatches12[i].second];

        const float x1 = kpRay1(0);
        const float y1 = kpRay1(1);
        const float z1 = kpRay1(2);

        const float x2 = kpRay2(0);
        const float y2 = kpRay2(1);
        const float z2 = kpRay2(2);

        const float a2 = e11*x1+e12*y1+e13*z1;
        const float b2 = e21*x1+e22*y1+e23*z1;
        const float c2 = e31*x1+e32*y1+e33*z1;

        const float num2 = a2*x2+b2*y2+c2*z2;

        //(a2,b2,c2) may not be unit vector but (x2,y2,z2) is unit vector
        const float squareDist1 = num2*num2/(a2*a2+b2*b2+c2*c2);

        float unitVectorSigma = sigma * CamModelGeneral::GetCamera()->GetVectorSigma(mvKeys2[mvMatches12[i].second], cv::Vec3f(a2,b2,c2));
        float invSigmaSquare = 1.0f/(unitVectorSigma*unitVectorSigma);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        const float a1 = e11*x2+e21*y2+e31*z2;
        const float b1 = e12*x2+e22*y2+e32*z2;
        const float c1 = e13*x2+e23*y2+e33*z2;

        const float num1 = a1*x1+b1*y1+c1*z1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1+c1*c1);

        unitVectorSigma = sigma * CamModelGeneral::GetCamera()->GetVectorSigma(mvKeys1[mvMatches12[i].first], cv::Vec3f(a1,b1,c1));
        invSigmaSquare = 1.0f/(unitVectorSigma*unitVectorSigma);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool Initializer::ReconstructE(vector<bool> &vbMatchesInliers, cv::Mat &E21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, 
        vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvKeyRays1,mvKeyRays2,mvMatches12,vbMatchesInliers, K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvKeyRays1,mvKeyRays2,mvMatches12,vbMatchesInliers, K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvKeyRays1,mvKeyRays2,mvMatches12,vbMatchesInliers, K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvKeyRays1,mvKeyRays2,mvMatches12,vbMatchesInliers, K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

void Initializer::Triangulate(const cv::Vec3f &ray1, const cv::Vec3f &ray2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    //Adapted vector-based triangulation method 
    cv::Mat A(4,4,CV_32F);
    const float &x1 = ray1(0), &y1 = ray1(1), &z1 = ray1(2);
    const float &x2 = ray2(0), &y2 = ray2(1), &z2 = ray2(2);
    A.row(0) = x1*(P1.row(1)+P1.row(2)) - (y1+z1)*P1.row(0);
    A.row(1) = y1*(P1.row(0)+P1.row(2)) - (x1+z1)*P1.row(1);
    A.row(2) = x2*(P2.row(1)+P2.row(2)) - (y2+z2)*P2.row(0);
    A.row(3) = y2*(P2.row(0)+P2.row(2)) - (x2+z2)*P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<cv::Vec3f> &vKeyRays1, const vector<cv::Vec3f> &vKeyRays2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers, cv::Mat &K, vector<cv::Point3f> &vP3D,
                       float th2, vector<bool> &vbGood, float &parallax)
{
    vbGood = vector<bool>(vKeyRays1.size(),false);
    vP3D.resize(vKeyRays1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeyRays1.size());

    // Camera 1 Projection Matrix K[I|0]
    // [I|0] K is not needed
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    cv::Mat Id = cv::Mat::eye(3,3,CV_32F);
    Id.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // [R|t] K is not needed
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::Vec3f &ray1 = vKeyRays1[vMatches12[i].first];
        const cv::Vec3f &ray2 = vKeyRays2[vMatches12[i].second];

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];

        cv::Mat p3dC1;

        Triangulate(ray1,ray2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if((p3dC1.at<float>(2)/dist1)<=CamModelGeneral::GetCamera()->GetCosFovTh() && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if((p3dC2.at<float>(2)/dist2)<=CamModelGeneral::GetCamera()->GetCosFovTh() && cosParallax<0.99998)
            continue;

        float im1x, im1y;
        CamModelGeneral::GetCamera()->TransformRaysToCubemap(im1x, im1y, p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        float im2x, im2y;
        CamModelGeneral::GetCamera()->TransformRaysToCubemap(im2x, im2y, p3dC2.at<float>(0), p3dC2.at<float>(1), p3dC2.at<float>(2));

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}
