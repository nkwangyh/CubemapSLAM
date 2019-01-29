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

#include "CamModelGeneral.h"

CamModelGeneral* CamModelGeneral::mpCamModel = NULL;

CamModelGeneral* CamModelGeneral::GetCamera()
{
    if(!mpCamModel)
        mpCamModel = new CamModelGeneral();
    return mpCamModel;
}

void CamModelGeneral::SetCamParams()
{
    c=1;
    d=0;
    e=0;
    u0=0;
    v0=0;
    p=(cv::Mat_<double>(1, 1) << 1);
    invP=(cv::Mat_<double>(1, 1) << 1);
    mWFisheye=0;
    mHFisheye=0;
    p_deg=1;
    invP_deg=1;
    p1=1;
}

void CamModelGeneral::SetCamParams(double cdeu0v0[], cv::Mat_<double> p_, cv::Mat_<double> invP_)
{
    c=cdeu0v0[0];
    d=cdeu0v0[1];
    e=cdeu0v0[2];
    u0=cdeu0v0[3];
    v0=cdeu0v0[4];
    p=p_;
    invP=invP_;

    // initialize degree of polynomials
    p_deg = (p_.rows > 1) ? p_.rows : p_.cols;
    invP_deg = (p_.rows > 1) ?  invP_.rows : invP_.cols;

    cde1 = (cv::Mat_<double>(2, 2) << c, d, e, 1.0);
    p1 = p.at<double>(0);
    invAffine = c - d*e;
}

void CamModelGeneral::SetCamParams(double cdeu0v0[],
        cv::Mat_<double> p_, cv::Mat_<double> invP_, double Iw_, double Ih_, 
        double fx_, double fy_, double cx_, double cy_, double width_, double height_, double camFov_)
{
    c=cdeu0v0[0]; d=cdeu0v0[1]; e=cdeu0v0[2];
    u0=cdeu0v0[3]; v0=cdeu0v0[4];
    p=p_; invP=invP_; mWFisheye=Iw_; mHFisheye=Ih_;
    fx=fx_; fy=fy_; cx=cx_; cy=cy_; mWCubeFace=width_; mHCubeFace = height_;

    // initialize degree of polynomials
    p_deg = (p_.rows > 1) ? p_.rows : p_.cols;
    invP_deg = (p_.rows > 1) ? invP_.rows : invP_.cols;

    cde1 = (cv::Mat_<double>(2, 2) << c, d, e, 1.0);
    p1 = p.at<double>(0);
    invAffine = c - d*e;

    SetCosFovTh(static_cast<float>(camFov_));
}

CamModelGeneral::eFace CamModelGeneral::TransformRaysToCubemap(float &up, float &vp, const cv::Vec3f &rigPt)
{
    const float &x = rigPt(0), &y = rigPt(1), &z = rigPt(2);
    cv::Vec3f localPt;
    float &_x = localPt(0), &_y = localPt(1), &_z = localPt(2);

    //choose different face according to (x, y, z)
    if(z > 0 && x/z <= 1 && x/z >= -1 && y/z <=1 && y/z >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, FRONT_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        up += mWCubeFace; vp += mHCubeFace;
        return FRONT_FACE;
    }
    else if(x > 0 && y/x <= 1 && y/x >= -1 && z/x <=1 && z/x >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, RIGHT_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        up += 2*mWCubeFace; vp += mHCubeFace;
        return RIGHT_FACE;
    }
    else if(x < 0 && y/(-x) <= 1 && y/(-x) >= -1 && z/(-x) <=1 && z/(-x) >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, LEFT_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        vp += mHCubeFace;
        return LEFT_FACE;
    }
    else if(y > 0 && x/y <= 1 && x/y >= -1 && z/y <=1 && z/y >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, LOWER_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        up += mWCubeFace; vp += 2*mHCubeFace;
        return LOWER_FACE;
    }
    else if(y < 0 && x/(-y) <= 1 && x/(-y) >= -1 && z/(-y) <=1 && z/(-y) >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, UPPER_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        up += mWCubeFace;
        return UPPER_FACE;
    }
    up = -1; vp = -1;
    return UNKNOWN_FACE;
}

CamModelGeneral::eFace CamModelGeneral::TransformRaysToCubemap(cv::Point2f &pixel, const cv::Vec3f &rigPt)
{
    float up, vp;
    CamModelGeneral::eFace face = TransformRaysToCubemap(up, vp, rigPt);
    pixel.x = up; pixel.y = vp;
    return face;
}

CamModelGeneral::eFace CamModelGeneral::TransformRaysToCubemap(cv::Vec2f &pixel, const cv::Vec3f &rigPt)
{
    float up, vp;
    CamModelGeneral::eFace face = TransformRaysToCubemap(up, vp, rigPt);
    pixel(0) = up; pixel(1) = vp;
    return face;
}

CamModelGeneral::eFace CamModelGeneral::TransformRaysToCubemapFace(float &up, float &vp, const cv::Vec3f &rigPt)
{
    const float &x = rigPt(0), &y = rigPt(1), &z = rigPt(2);
    cv::Vec3f localPt;
    float &_x = localPt(0), &_y = localPt(1), &_z = localPt(2);

    //choose different face according to (x, y, z)
    if(z > 0 && x/z <= 1 && x/z >= -1 && y/z <=1 && y/z >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, FRONT_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        return FRONT_FACE;
    }
    else if(x > 0 && y/x <= 1 && y/x >= -1 && z/x <=1 && z/x >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, RIGHT_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        return RIGHT_FACE;
    }
    else if(x < 0 && y/(-x) <= 1 && y/(-x) >= -1 && z/(-x) <=1 && z/(-x) >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, LEFT_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        return LEFT_FACE;
    }
    else if(y > 0 && x/y <= 1 && x/y >= -1 && z/y <=1 && z/y >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, LOWER_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        return LOWER_FACE;
    }
    else if(y < 0 && x/(-y) <= 1 && x/(-y) >= -1 && z/(-y) <=1 && z/(-y) >= -1)
    {
        cvtRigToFaces<float>(localPt, rigPt, UPPER_FACE);
        up = _x * fx / _z + cx; 
        vp = _y * fy / _z + cy; 
        if(up < 0 || up >= mWCubeFace || vp < 0  || vp >= mHCubeFace)
            return UNKNOWN_FACE;
        return UPPER_FACE;
    }
    up = -1; vp = -1;
    return UNKNOWN_FACE;
}

void CamModelGeneral::TransformRaysToTargetFace(float &up, float &vp, const cv::Vec3f &rigPt, const eFace face)
{
    cv::Vec3f localPt;
    float &_x = localPt(0), &_y = localPt(1), &_z = localPt(2);

    //choose different face according to (x, y, z)
    switch(face) {
        case FRONT_FACE:
            cvtRigToFaces<float>(localPt, rigPt, FRONT_FACE);
            up = _x * fx / _z + cx; 
            vp = _y * fy / _z + cy; 
            break;
        case RIGHT_FACE:
            cvtRigToFaces<float>(localPt, rigPt, RIGHT_FACE);
            up = _x * fx / _z + cx; 
            vp = _y * fy / _z + cy; 
            break;
        case LEFT_FACE:
            cvtRigToFaces<float>(localPt, rigPt, LEFT_FACE);
            up = _x * fx / _z + cx; 
            vp = _y * fy / _z + cy; 
            break;
        case LOWER_FACE:
            cvtRigToFaces<float>(localPt, rigPt, LOWER_FACE);
            up = _x * fx / _z + cx; 
            vp = _y * fy / _z + cy; 
            break;
        case UPPER_FACE:
            cvtRigToFaces<float>(localPt, rigPt, UPPER_FACE);
            up = _x * fx / _z + cx; 
            vp = _y * fy / _z + cy; 
            break;
        default:
            up = -1.0; vp = -1.0;
    }
}

void CamModelGeneral::CubemapToFisheye(double &uf, double &vf, const double &up, const double &vp)
{
    //cvt (up, vp) to (i, j)
    float i = up, j = vp; 
    uf = -1; vf = -1;
    eFace face = FaceInCubemap<float>(i, j);
    if(face == UNKNOWN_FACE)
        return;

    double x, y, z = 1.0;
    i = i - static_cast<int>(i / mWCubeFace) * mWCubeFace; j = j - static_cast<int>(j / mHCubeFace) * mHCubeFace;

    x = (i - cx) * z / fx;
    y = (j - cy) * z / fy;
    const cv::Vec3d localPt(x, y, z);
    cv::Vec3d rigPt;
    double &_x = rigPt(0), &_y = rigPt(1), &_z = rigPt(2);

    cvtFacesToRig<double>(rigPt, localPt, face);
    WorldToImg(_x, _y, _z, uf, vf);
    if(uf < 0 || uf >= mWFisheye || vf < 0 || vf >= mHFisheye)
    {
        uf = -1;
        vf = -1;
    }
}

float CamModelGeneral::GetVectorSigma(const cv::KeyPoint &key, const float &sigmaInPixel)
{
    const float radius = GetEpipolarRadius(key);
    float sigma = sigmaInPixel*fx/(fx*fx+radius*(radius+sigmaInPixel));
    return sigma;
}

float CamModelGeneral::GetVectorSigma(const cv::KeyPoint &key, const float &coef, const float &sigmaInPixel)
{
    const float radius = GetEpipolarRadius(key);
    float sigma = sigmaInPixel*fx/(fx*fx+radius*(radius+sigmaInPixel));
    return sigma*coef;
}

//normal in body frame
float CamModelGeneral::GetVectorSigma(const cv::KeyPoint &key, const cv::Vec3f &normalRig, const float &sigmaInPixel)
{
    //transform normal into local camera frame
    cv::Vec3f normalCam;
    eFace face = FaceInCubemap(key.pt);
    cvtRigToFaces<float>(normalCam, normalRig, face);
    cv::Vec3f epipolar(normalCam(1), -normalCam(0), 0.0f);
    //perpendicular to epipolar
    cv::Vec3f vertical(normalCam(0), normalCam(1), 0.0f);
    float u, v;
    GetPosInFace<float>(u, v, key.pt.x, key.pt.y);
    cv::Vec3f OP(u-cx, v-cy, 0.0f);
    //OO1 and CO1 length in pixels
    float OO1 = OP.dot(epipolar) / cv::norm(epipolar); if(OO1 < 0) OO1 = -OO1;
    const float CO1 = std::sqrt(OO1*OO1 + fx*fx);
    float PO1 = OP.dot(vertical) / cv::norm(vertical); if(PO1 < 0) PO1 = -PO1;
    //tan(\phi)
    const float tan1 = PO1/CO1;
    //tan(\theta+\phi)
    const float tan2 = (PO1+sigmaInPixel)/CO1;
    //tan(\theta) = tan(\theta+\phi - \phi)
    const float tan3 = (tan2-tan1) / (1+tan1*tan2);
    //cos(\theta+\pi/2) = sin(\theta)
    const float sin_theta = 1.0f / std::sqrt(1.0f/(tan3*tan3)+1);

    return sin_theta;
}
