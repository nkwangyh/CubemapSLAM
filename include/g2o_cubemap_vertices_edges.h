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

#include "ThirdParty/g2o/g2o/core/base_vertex.h"
#include "ThirdParty/g2o/g2o/core/base_unary_edge.h"
#include "ThirdParty/g2o/g2o/core/eigen_types.h"
#include "ThirdParty/g2o/g2o/types/types_six_dof_expmap.h"
#include "ThirdParty/g2o/g2o/types/types_seven_dof_expmap.h"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "CamModelGeneral.h"

namespace g2o {

Vector2d project2d(const Vector3d& v);

// Multi-pinhole minimization
class  EdgeSE3ProjectXYZMultiPinholeOnlyPose: public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZMultiPinholeOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    if(_face == CamModelGeneral::UNKNOWN_FACE)
    {
        _error = Vector2d(0.0, 0.0);
        std::cout << "@func: PoseOptimization computeError() error: find unknown face feature in computeError(). The function should not reach here" << std::endl;
        exit(EXIT_FAILURE);
        return;
    }
        
    _error = _measurementInFace-multipinhole_project(v1->estimate().map(Xw));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }


  virtual void linearizeOplus();

  //Vector2d multipinhole_project(const Vector3d & trans_xyz) const;
  Vector2d multipinhole_project(const Vector3d & trans_xyz) const;
  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw;
  double fx, fy, cx, cy;

  void setFace(const CamModelGeneral::eFace& face) { _face = face; }
  void setMeasurementInFace(const Measurement& mInFace) { _measurementInFace = mInFace;}

private:
  CamModelGeneral::eFace _face;
  Measurement _measurementInFace;
};

//Multi-pinhole localBA
class  EdgeSE3ProjectXYZMultiPinhole: public  BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZMultiPinhole();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    if(_face == CamModelGeneral::UNKNOWN_FACE)
    {
        _error = Vector2d(0.0, 0.0);
        std::cout << "@func: BundleAdjustment computeError() error: find unknown face feature in computeError(). The function should not reach here" << std::endl;
        exit(EXIT_FAILURE);
        return;
    }
        
    _error = _measurementInFace-multipinhole_project(v1->estimate().map(v2->estimate()));
  }

  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }
    

  virtual void linearizeOplus();

  Vector2d multipinhole_project(const Vector3d & trans_xyz) const;
  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;

  void setFace(const CamModelGeneral::eFace& face) { _face = face; }
  void setMeasurementInFace(const Measurement& mInFace) { _measurementInFace = mInFace;}

private:
  CamModelGeneral::eFace _face;
  Measurement _measurementInFace;
};

//Loop closing
class EdgeSim3ProjectXYZMultiPinhole : public  BaseBinaryEdge<2, Vector2d,  VertexSBAPointXYZ, VertexSim3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3ProjectXYZMultiPinhole();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    void computeError()  {

        const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[1]);
        const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);

        //_measurement is from cubemap image frame
        if(_face == CamModelGeneral::UNKNOWN_FACE)
        {
            _error = Vector2d(0.0, 0.0);
            std::cout << "@func: BundleAdjustment computeError() error: find unknown face feature in computeError(). The function should not reach here" << std::endl;
            exit(EXIT_FAILURE);
            return;
        }
            
        _error = _measurementInFace-multipinhole_project(v1->estimate().map(v2->estimate()));
    }

   // virtual void linearizeOplus();
  Vector2d multipinhole_project(const Vector3d & trans_xyz) const;
   
  void setFace(const CamModelGeneral::eFace& face) { _face = face; }
  void setMeasurementInFace(const Measurement& mInFace) { _measurementInFace = mInFace;}

private:
  CamModelGeneral::eFace _face;
  Measurement _measurementInFace;

};

/**/
class EdgeInverseSim3ProjectXYZMultiPinhole : public  BaseBinaryEdge<2, Vector2d,  VertexSBAPointXYZ, VertexSim3Expmap>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseSim3ProjectXYZMultiPinhole();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    void computeError()  {

        const VertexSim3Expmap* v1 = static_cast<const VertexSim3Expmap*>(_vertices[1]);
        const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);

        //_measurement is from cubemap image frame
        if(_face == CamModelGeneral::UNKNOWN_FACE)
        {
            _error = Vector2d(0.0, 0.0);
            std::cout << "@func: BundleAdjustment computeError() error: find unknown face feature in computeError(). The function should not reach here" << std::endl;
            exit(EXIT_FAILURE);
            return;
        }
            
        _error = _measurementInFace-multipinhole_project(v1->estimate().inverse().map(v2->estimate()));
    }

   // virtual void linearizeOplus();
  Vector2d multipinhole_project(const Vector3d & trans_xyz) const;

  void setFace(const CamModelGeneral::eFace& face) { _face = face; }
  void setMeasurementInFace(const Measurement& mInFace) { _measurementInFace = mInFace;}

private:
  CamModelGeneral::eFace _face;
  Measurement _measurementInFace;
};

} //NAMESPACE g2o
