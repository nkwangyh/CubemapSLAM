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

#ifndef CAMMODELGENERAL_H
#define CAMMODELGENERAL_H

#include <opencv2/opencv.hpp>
#include <vector>

#ifndef M_PI
#define M_PI   3.1415926535897932384626433832795028841971693993
#endif

#ifndef M_PIf
#define M_PIf    3.1415926535897932384626f
#endif

inline double horner(
    const double* coeffs, const int& s, const double& x)
{
    double res = 0.0;
    for (int i = s - 1; i >= 0; i--)
        res = res * x + coeffs[i];
    return res;
}

class CamModelGeneral 
{
public:
    enum eFace {
        UNKNOWN_FACE = -1,
        FRONT_FACE = 0,
        LEFT_FACE = 1,
        RIGHT_FACE = 2,
        UPPER_FACE = 3,
        LOWER_FACE = 4 
    };
protected:
    // affin
    double c;
    double d;
    double e;
    double invAffine;
    cv::Mat_<double> cde1;
    // principal
    double u0;
    double v0;
    // polynomial
    cv::Mat_<double> p;
    // inverse polynomial
    cv::Mat_<double> invP;

    // image width and height
    int mWFisheye;
    int mHFisheye;

    // for cubemap face
    double fx;
    double fy;
    double cx;
    double cy;
    // perspective image(face) size
    int mWCubeFace;
    int mHCubeFace;

    // polynomial degrees
    int p_deg;
    int invP_deg;
    double p1;

    float mCosFovTh;

public:
    //Get camera model
    static CamModelGeneral *GetCamera();

    //Set Camera parameters
    void SetCamParams();

    void SetCamParams(double cdeu0v0[], cv::Mat_<double> p_, cv::Mat_<double> invP_);

    void SetCamParams(double cdeu0v0[],
        cv::Mat_<double> p_, cv::Mat_<double> invP_, double Iw_, double Ih_, 
        double fx_, double fy_, double cx_, double cy_, double width_, double height_, double camFov_);

    ~CamModelGeneral(){}
    // get functions
    double Get_c() { return c; }
    double Get_d() { return d; }
    double Get_e() { return e; }

    double Get_u0() { return u0; }
    double Get_v0() { return v0; }

    int GetInvDeg() { return invP_deg; }
    int GetPolDeg() { return p_deg; }

    cv::Mat_<double> Get_invP() { return invP; }
    cv::Mat_<double> Get_P() { return p; }

    int GetFisheyeWidth() { return mWFisheye; }
    int GetFisheyeHeight() { return mHFisheye; }
     
    double Get_fx() { return fx; }
    double Get_fy() { return fy; }
    double Get_cx() { return cx; }
    double Get_cy() { return cy; }
    double Get_invfx() { return 1.0/fx; }
    double Get_invfy() { return 1.0/fy; }

    int GetCubeFaceWidth() { return mWCubeFace; }
    int GetCubeFaceHeight() { return mHCubeFace; }

    float GetCosFovTh() { return mCosFovTh; }


    //fisheye 2D image point and 3D scene point transformation
	void ImgToWorld(cv::Point3_<double>& X,						        // 3D scene point
		const cv::Point_<double>& m); 			                        // 2D image point

	void ImgToWorld(double& x, double& y, double& z,					// 3D scene point
		const double& u, const double& v); 			                    // 2D image point

	void ImgToWorld(cv::Vec3f& X,						                // 3D scene point
		const cv::Point2f& m); 			                                // 2D image point


	void WorldToImg(const cv::Point3_<double>& X,			            // 3D scene point
		cv::Point_<double>& m);			                                // 2D image point

	void WorldToImg(const cv::Vec3d& X,			                        // 3D scene point
		cv::Vec2d& m);			                                        // 2D image point

	void WorldToImg(const cv::Vec3d& X,			                        // 3D scene point
		cv::Vec2f& m);			                                        // 2D image point

	void WorldToImg(const double& x, const double& y, const double& z,  // 3D scene point
		double& u, double& v) const;							        // 2D image point

    //transform to rig space
    template<class T>
    void cvtFacesToRig(cv::Vec<T, 3> &rigPt, const cv::Vec<T, 3> &localPt, const eFace &face);

    //transform to local face spaces
    template<class T>
    void cvtRigToFaces(cv::Vec<T, 3> &localPt, const cv::Vec<T, 3> &rigPt, const eFace &face);
    
    //find which face the pixel reside on
    eFace FaceInCubemap(const cv::Point2f &pixel);

    template<class T>
    eFace FaceInCubemap(const T &x, const T &y);

    template<class T>
    eFace FaceInCubemap(const T &x, const T &y, const T &z);

    //transform between rays and cubemap pixels
    eFace TransformCubemapToRays(cv::Vec3f &point, const cv::Point2f &pixel);

    eFace TransformRaysToCubemap(float &up, float &vp, const float &x, const float &y, const float &z);

    eFace TransformRaysToCubemap(float &up, float &vp, const cv::Vec3f &rigPt);

    eFace TransformRaysToCubemap(cv::Point2f &pixel, const cv::Vec3f &rigPt);

    eFace TransformRaysToCubemap(cv::Vec2f &pixel, const cv::Vec3f &rigPt);

    eFace TransformRaysToCubemapFace(float &up, float &vp, const cv::Vec3f &rigPt);

    //transform rays to a given face
    void TransformRaysToTargetFace(float &up, float &vp, const cv::Vec3f &rigPt, const eFace face);

    //Fisheye image and cubemap image transformation
    //assume pinhole faces share the same focal length and image size
    eFace FisheyeToCubemap(const float &uf, const float &vf, float &up, float &vp);

    void CubemapToFisheye(double &uf, double &vf, const double &up, const double &vp);

    template<class T>
    void GetPosInFace(T &u, T &v, const T &uCubemap, const T &vCubemap)
    {
        const int i = std::floor(uCubemap / mWCubeFace), j = std::floor(vCubemap / mHCubeFace);
        u = uCubemap - i*mWCubeFace, v = vCubemap - j*mHCubeFace;
    }

    float GetEpipolarRadius(const cv::KeyPoint &key)
    {
        float x, y;
        GetPosInFace(x,y,key.pt.x,key.pt.y);
        return std::sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy));
    }

    float GetVectorSigma(const cv::KeyPoint &key, const float &sigmaInPixel=1.0f);
    float GetVectorSigma(const cv::KeyPoint &key, const float &coef, const float &sigmaInPixel);
    //normal in body frame
    float GetVectorSigma(const cv::KeyPoint &key, const cv::Vec3f &normalRig, const float &sigmaInPixel=1.0f);

protected:
    void SetCosFovTh(const float &fovInDegree)
    {
        mCosFovTh = cos(fovInDegree/2*(M_PIf/180));
        std::cout << "CamFov is set to: " << fovInDegree << std::endl;
        std::cout << "Cos Fov threshold is: " << mCosFovTh << std::endl;
    }

private:
    // construtors
    CamModelGeneral(){};
    CamModelGeneral(CamModelGeneral const&){};
    CamModelGeneral& operator=(CamModelGeneral const&);
    static CamModelGeneral *mpCamModel;
};

////////////////////////////////////////
//Impl of inline function members
///////////////////////////////////////
inline void CamModelGeneral::ImgToWorld(cv::Point3_<double>& X,						// 3D scene point
    const cv::Point_<double>& m) 			            // 2D image point
{
    //double invAff = c - d*e;
    const double u_t = m.x - u0;
    const double v_t = m.y - v0;
    // inverse affine matrix image to sensor plane conversion
    X.x = (1 * u_t - d * v_t) / this->invAffine;
    X.y = (-e * u_t + c * v_t) / this->invAffine;
    const double X2 = X.x * X.x;
    const double Y2 = X.y * X.y;
    X.z = -horner((double*)p.data, p_deg, sqrt(X2 + Y2));

    // normalize vectors spherically
    const double norm = sqrt(X2 + Y2 + X.z*X.z);
    X.x /= norm;
    X.y /= norm;
    X.z /= norm;
}

inline void CamModelGeneral::ImgToWorld(double& x, double& y, double& z,						// 3D scene point
    const double& u, const double& v) 			    // 2D image point
{
    //double invAff = c - d*e;
    const double u_t = u - u0;
    const double v_t = v - v0;
    // inverse affine matrix image to sensor plane conversion
    x = (u_t - d * v_t) / this->invAffine;
    y = (-e * u_t + c * v_t) / this->invAffine;
    const double X2 = x * x;
    const double Y2 = y * y;
    z = -horner((double*)p.data, p_deg, sqrt(X2 + Y2));

    // normalize vectors spherically
    double norm = sqrt(X2 + Y2 + z*z);
    x /= norm;
    y /= norm;
    z /= norm;
}

inline void CamModelGeneral::ImgToWorld(cv::Vec3f& X,						// 3D scene point
    const cv::Point2f& m) 			            // 2D image point
{
    //double invAff = c - d*e;
    const double u_t = m.x - u0;
    const double v_t = m.y - v0;
    // inverse affine matrix image to sensor plane conversion
    X(0) = (u_t - d * v_t) / this->invAffine;
    X(1) = (-e * u_t + c * v_t) / this->invAffine;
    const double X2 = X(0) * X(0);
    const double Y2 = X(1) * X(1);
    X(2) = -horner((double*)p.data, p_deg, sqrt(X2 + Y2));

    // normalize vectors spherically
    double norm = sqrt(X2 + Y2 + X(2)*X(2));
    X(0) /= norm;
    X(1) /= norm;
    X(2) /= norm;
}


inline void CamModelGeneral::WorldToImg(const cv::Point3_<double>& X,			// 3D scene point
    cv::Point_<double>& m)			// 2D image point
{
    double norm = sqrt(X.x*X.x + X.y*X.y);

    if (norm == 0.0)
        norm = 1e-14;

    const double theta = atan(-X.z / norm);
    const double rho = horner((double*)invP.data, invP_deg, theta);

    const double uu = X.x / norm * rho;
    const double vv = X.y / norm * rho;

    m.x = uu*c + vv*d + u0;
    m.y = uu*e + vv + v0;
}

inline void CamModelGeneral::WorldToImg(const cv::Vec3d& X,			// 3D scene point
    cv::Vec2d& m)			// 2D image point
{

    double norm = cv::sqrt(X(0)*X(0) + X(1)*X(1));

    if (norm == 0.0)
        norm = 1e-14;

    const double theta = atan(-X(2) / norm);
    const double rho = horner((double*)invP.data, invP_deg, theta);

    const double uu = X(0) / norm * rho;
    const double vv = X(1) / norm * rho;

    m(0) = uu*c + vv*d + u0;
    m(1) = uu*e + vv + v0;
}

inline void CamModelGeneral::WorldToImg(const cv::Vec3d& X,			// 3D scene point
    cv::Vec2f& m)			// 2D image point
{
    double norm = cv::sqrt(X(0)*X(0) + X(1)*X(1));

    if (norm == 0.0)
        norm = 1e-14;

    const double theta = atan(-X(2) / norm);

    const double rho = horner((double*)invP.data, invP_deg, theta);

    const double uu = X(0) / norm * rho;
    const double vv = X(1) / norm * rho;

    m(0) = uu*c + vv*d + u0;
    m(1) = uu*e + vv + v0;
}

inline void CamModelGeneral::WorldToImg(const double& x, const double& y, const double& z,    // 3D scene point
    double& u, double& v) const							 // 2D image point
{
    double norm = sqrt(x*x + y*y);
    if (norm == 0.0)
        norm = 1e-14;

    const double theta = atan(-z / norm);
    const double rho = horner((double*)invP.data, invP_deg, theta);

    const double uu = x / norm * rho;
    const double vv = y / norm * rho;

    u = uu*c + vv*d + u0;
    v = uu*e + vv + v0;
}

//assume pinhole faces share the same focal length and image size
inline CamModelGeneral::eFace CamModelGeneral::FisheyeToCubemap(const float &uf, const float &vf, float &up, float &vp)
{
    double x, y, z; 
    ImgToWorld(x, y, z, uf, vf);
    cv::Vec3f rigPt(x, y, z);
    eFace face = TransformRaysToCubemap(up, vp, rigPt);
    
    return face;
}

//transform to rig space
template<class T>
void CamModelGeneral::cvtFacesToRig(cv::Vec<T, 3> &rigPt, const cv::Vec<T, 3> &localPt, const eFace &face)
{
    const T &x = localPt(0), &y = localPt(1), &z = localPt(2);
    T &_x = rigPt(0), &_y = rigPt(1), &_z = rigPt(2);

    switch(face) {
        case FRONT_FACE:
            _x = x; _y = y; _z = z;
            break;
        case LEFT_FACE:
            _x = -z; _y = y; _z = x;
            break;
        case RIGHT_FACE:
            _x = z; _y = y; _z = -x;
            break;
        case LOWER_FACE:
            _x = x; _y = z; _z = -y;
            break;
        case UPPER_FACE:
            _x = x; _y = -z; _z = y;
            break;
        default:
            _x = 0; _y = 0; _z = 0;
            return;
    }
}

//transform to local face spaces
template<class T>
void CamModelGeneral::cvtRigToFaces(cv::Vec<T, 3> &localPt, const cv::Vec<T, 3> &rigPt, const eFace &face)
{
    const T &x = rigPt(0), &y = rigPt(1), &z = rigPt(2);
    T &_x = localPt(0), &_y = localPt(1), &_z = localPt(2);

    switch(face) {
        case FRONT_FACE:
            _x = x; _y = y; _z = z;
            break;
        case LEFT_FACE:
            _x = z; _y = y; _z = -x;
            break;
        case RIGHT_FACE:
            _x = -z; _y = y; _z = x;
            break;
        case LOWER_FACE:
            _x = x; _y = -z; _z = y;
            break;
        case UPPER_FACE:
            _x = x; _y = z; _z = -y;
            break;
        default:
            _x = 0; _y = 0; _z = 0;
            return;
    }
}

inline CamModelGeneral::eFace CamModelGeneral::FaceInCubemap(const cv::Point2f &pixel)
{
    eFace face = UNKNOWN_FACE; 
    double i = pixel.x / mWCubeFace, j = pixel.y / mHCubeFace;

    if(i >= 0 && i < 1 && j >= 1 && j < 2) face = LEFT_FACE;
    else if(i >= 1 && i < 2 && j >= 0 && j < 1) face = UPPER_FACE;
    else if(i >= 1 && i < 2 && j >= 1 && j < 2) face = FRONT_FACE;
    else if(i >= 1 && i < 2 && j >= 2 && j < 3) face = LOWER_FACE;
    else if(i >= 2 && i < 3 && j >= 1 && j < 2) face = RIGHT_FACE;
    return face;
}

template<class T>
inline CamModelGeneral::eFace CamModelGeneral::FaceInCubemap(const T &x, const T &y)
{
    eFace face = UNKNOWN_FACE; 
    T i = x / mWCubeFace, j = y / mHCubeFace;

    if(i >= 0 && i < 1 && j >= 1 && j < 2) face = LEFT_FACE;
    else if(i >= 1 && i < 2 && j >= 0 && j < 1) face = UPPER_FACE;
    else if(i >= 1 && i < 2 && j >= 1 && j < 2) face = FRONT_FACE;
    else if(i >= 1 && i < 2 && j >= 2 && j < 3) face = LOWER_FACE;
    else if(i >= 2 && i < 3 && j >= 1 && j < 2) face = RIGHT_FACE;
    return face;
}

template<class T>
inline CamModelGeneral::eFace CamModelGeneral::FaceInCubemap(const T &x, const T &y, const T &z)
{
    //choose different face according to (x, y, z)
    if(z > 0 && x/z <= 1 && x/z >= -1 && y/z <=1 && y/z >= -1)
        return FRONT_FACE;

    if(x > 0 && y/x <= 1 && y/x >= -1 && z/x <=1 && z/x >= -1)
        return RIGHT_FACE;
    
    if(x < 0 && y/(-x) <= 1 && y/(-x) >= -1 && z/(-x) <=1 && z/(-x) >= -1)
        return LEFT_FACE;

    if(y > 0 && x/y <= 1 && x/y >= -1 && z/y <=1 && z/y >= -1)
        return LOWER_FACE;

    if(y < 0 && x/(-y) <= 1 && x/(-y) >= -1 && z/(-y) <=1 && z/(-y) >= -1)
        return UPPER_FACE;

    return UNKNOWN_FACE;
}

inline CamModelGeneral::eFace CamModelGeneral::TransformCubemapToRays(cv::Vec3f &point, const cv::Point2f &pixel)
{
    eFace face = FaceInCubemap(pixel);
    if(face == UNKNOWN_FACE)
    {
        return face;
    }

    double x, y, z = 1.0;
    double i = pixel.x, j= pixel.y;
    i = i - static_cast<int>(i / mWCubeFace) * mWCubeFace; j = j - static_cast<int>(j / mHCubeFace) * mHCubeFace;
    x = (i - cx) * z / fx;
    y = (j - cy) * z / fy;
    cvtFacesToRig<float>(point, cv::Vec3f(x, y, z), face);
    //normalize
    double normal = cv::norm(point);
    point = point * (normal > 0 ? 1./normal : 0.);

    return face;
}

inline CamModelGeneral::eFace CamModelGeneral::TransformRaysToCubemap(float &up, float &vp, const float &x, const float &y, const float &z)
{
    cv::Vec3f rigPt(x, y, z);
    return TransformRaysToCubemap(up, vp, rigPt);
}

#endif //CAMMODELGENERAL_H
