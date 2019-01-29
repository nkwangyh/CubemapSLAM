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

#include "g2o_cubemap_vertices_edges.h"

namespace g2o {

//Multi-pinhole Only Pose
bool EdgeSE3ProjectXYZMultiPinholeOnlyPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZMultiPinholeOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZMultiPinholeOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

    //get cube face xyz_trans reside on
    Eigen::Matrix<double,3,3> R_local;
    switch(_face) {
        case CamModelGeneral::FRONT_FACE: 
            R_local << 1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;
            break;
        case CamModelGeneral::LEFT_FACE: 
            R_local << 0, 0, 1,
                    0, 1, 0,
                    -1, 0, 0;
            break;
        case CamModelGeneral::RIGHT_FACE: 
            R_local << 0, 0, -1,
                    0, 1, 0,
                    1, 0, 0;
            break;
        case CamModelGeneral::LOWER_FACE: 
            R_local << 1, 0, 0,
                    0, 0, -1, 
                    0, 1, 0;
            break;
        case CamModelGeneral::UPPER_FACE: 
            R_local << 1, 0, 0,
                    0, 0, 1,
                    0, -1, 0;
            break;
        default:
            R_local << 0, 0, 0,
                    0, 0, 0,
                    0, 0, 0;
            std::cout << "@func: PoseOptimization linearizeOplus() point from unknown face should be culled before this function(by deactivating the edge)" << std::endl;
            exit(EXIT_FAILURE);
    }

    Eigen::Matrix<double,3,3> negSkewOfRigPt;
    negSkewOfRigPt << 0, xyz_trans[2], -xyz_trans[1],
                    -xyz_trans[2], 0, xyz_trans[0],
                    xyz_trans[1], -xyz_trans[0], 0;
    Eigen::Matrix<double,3,6> dRigPt_dXi;
    dRigPt_dXi << negSkewOfRigPt, Eigen::Matrix<double,3,3>::Identity();

    Vector3d localPt = R_local * xyz_trans;
    Eigen::Matrix<double,2,3> dudLocalPt;
    dudLocalPt << fx/localPt[2], 0, -fx*localPt[0]/(localPt[2]*localPt[2]),
                0, fy/localPt[2], -fy*localPt[1]/(localPt[2]*localPt[2]);

    _jacobianOplusXi = -1.0 * dudLocalPt * R_local * dRigPt_dXi;
}

Vector2d EdgeSE3ProjectXYZMultiPinholeOnlyPose::multipinhole_project(const Vector3d & trans_xyz) const{
    float u,v;
    cv::Vec3f rigPt(trans_xyz[0], trans_xyz[1], trans_xyz[2]);
    CamModelGeneral::GetCamera()->TransformRaysToTargetFace(u,v,rigPt,_face);
    Vector2d res;
    res[0] = u;
    res[1] = v;
    return res;
}

Vector2d EdgeSE3ProjectXYZMultiPinholeOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

//Multi-pinhole localBA
EdgeSE3ProjectXYZMultiPinhole::EdgeSE3ProjectXYZMultiPinhole() : BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>() {}

bool EdgeSE3ProjectXYZMultiPinhole::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZMultiPinhole::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}


void EdgeSE3ProjectXYZMultiPinhole::linearizeOplus() {
    VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
    SE3Quat T(vj->estimate());
    VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d xyz = vi->estimate();
    Vector3d xyz_trans = T.map(xyz);

    //get cube face xyz_trans reside on
    Eigen::Matrix<double,3,3> R_local;
    switch(_face) {
        case CamModelGeneral::FRONT_FACE: 
            R_local << 1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;
            break;
        case CamModelGeneral::LEFT_FACE: 
            R_local << 0, 0, 1,
                    0, 1, 0,
                    -1, 0, 0;
            break;
        case CamModelGeneral::RIGHT_FACE: 
            R_local << 0, 0, -1,
                    0, 1, 0,
                    1, 0, 0;
            break;
        case CamModelGeneral::LOWER_FACE: 
            R_local << 1, 0, 0,
                    0, 0, -1, 
                    0, 1, 0;
            break;
        case CamModelGeneral::UPPER_FACE: 
            R_local << 1, 0, 0,
                    0, 0, 1,
                    0, -1, 0;
            break;
        default:
            R_local << 0, 0, 0,
                    0, 0, 0,
                    0, 0, 0;
            std::cout << "@func: BundleAdjustment linearizeOplus() point from unknown face should be culled before this function(by deactivating the edge)" << std::endl;
            exit(EXIT_FAILURE);
    }

    Eigen::Matrix<double,3,3> negSkewOfRigPt;
    negSkewOfRigPt << 0, xyz_trans[2], -xyz_trans[1],
                    -xyz_trans[2], 0, xyz_trans[0],
                    xyz_trans[1], -xyz_trans[0], 0;
    Eigen::Matrix<double,3,6> dRigPt_dXj;
    dRigPt_dXj << negSkewOfRigPt, Eigen::Matrix<double,3,3>::Identity();

    Vector3d localPt = R_local * xyz_trans;
    Eigen::Matrix<double,2,3> dudLocalPt;
    dudLocalPt << fx/localPt[2], 0, -fx*localPt[0]/(localPt[2]*localPt[2]),
                0, fy/localPt[2], -fy*localPt[1]/(localPt[2]*localPt[2]);

    Eigen::Matrix<double,2,3> dudLocalPtRlocal = -1.0 * dudLocalPt * R_local;

    _jacobianOplusXj = dudLocalPtRlocal * dRigPt_dXj;
    _jacobianOplusXi = dudLocalPtRlocal * T.rotation().toRotationMatrix();
}

Vector2d EdgeSE3ProjectXYZMultiPinhole::multipinhole_project(const Vector3d & trans_xyz) const{
    float u,v;
    cv::Vec3f rigPt(trans_xyz[0], trans_xyz[1], trans_xyz[2]);
    CamModelGeneral::GetCamera()->TransformRaysToTargetFace(u,v,rigPt,_face);
    Vector2d res;
    res[0] = u;
    res[1] = v;
    return res;
}

Vector2d EdgeSE3ProjectXYZMultiPinhole::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

/**Sim3ProjectXYZ*/

EdgeSim3ProjectXYZMultiPinhole::EdgeSim3ProjectXYZMultiPinhole() :
BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSim3Expmap>()
{
}

bool EdgeSim3ProjectXYZMultiPinhole::read(std::istream& is)
{
    for (int i=0; i<2; i++)
    {
      is >> _measurement[i];
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++) {
  is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
    return true;
}

bool EdgeSim3ProjectXYZMultiPinhole::write(std::ostream& os) const
{
    for (int i=0; i<2; i++){
      os  << _measurement[i] << " ";
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++){
  os << " " <<  information()(i,j);
    }
    return os.good();
}

Vector2d EdgeSim3ProjectXYZMultiPinhole::multipinhole_project(const Vector3d & trans_xyz) const{
    float u,v;
    cv::Vec3f rigPt(trans_xyz[0], trans_xyz[1], trans_xyz[2]);
    CamModelGeneral::GetCamera()->TransformRaysToTargetFace(u,v,rigPt,_face);
    Vector2d res;
    res[0] = u;
    res[1] = v;
    return res;
}

/**InverseSim3ProjectXYZ*/

EdgeInverseSim3ProjectXYZMultiPinhole::EdgeInverseSim3ProjectXYZMultiPinhole() :
BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSim3Expmap>()
{
}

bool EdgeInverseSim3ProjectXYZMultiPinhole::read(std::istream& is)
{
    for (int i=0; i<2; i++)
    {
      is >> _measurement[i];
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++) {
  is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
    return true;
}

bool EdgeInverseSim3ProjectXYZMultiPinhole::write(std::ostream& os) const
{
    for (int i=0; i<2; i++){
      os  << _measurement[i] << " ";
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++){
  os << " " <<  information()(i,j);
    }
    return os.good();
}

Vector2d EdgeInverseSim3ProjectXYZMultiPinhole::multipinhole_project(const Vector3d & trans_xyz) const{
    float u,v;
    cv::Vec3f rigPt(trans_xyz[0], trans_xyz[1], trans_xyz[2]);
    CamModelGeneral::GetCamera()->TransformRaysToTargetFace(u,v,rigPt,_face);
    Vector2d res;
    res[0] = u;
    res[1] = v;
    return res;
}

} //NAMESPACE g2o
