# CubemapSLAM

CubemapSLAM is a real-time SLAM system for monocular fisheye cameras. The system incorporates the cubemap model into the state-of-the-art feature based SLAM system [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2) to utilize the large FoV of the fisheye camera without introducing the distortion. The system is able to compute the camera trajectory and recover a sparse structure of the environment. It is also able to detect loops and relocalize the camera in real time. We provide examples on the [Lafida dataset](https://www.ipf.kit.edu/lafida_datasets.php) and the self-collected data
sequences. Like ORB-SLAM, we also provide a GUI to change between a *SLAM Mode* and *Localization Mode*, please refer to the document for details.

[![CubemapSLAM](http://img.youtube.com/vi/QHmIGAFfXe0/0.jpg)](http://www.youtube.com/watch?v=QHmIGAFfXe0 "CubemapSLAM")

# 1. License

CubemapSLAM is released under a [GPLv3 license](https://github.com/nkwangyh/CubemapSLAM/blob/master/License-gpl.txt). For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/nkwangyh/CubemapSLAM/blob/master/Dependencies.md).

If you use CubemapSLAM in your work, please consider citing:

    @article{wang2018cubemapslam,
      title={CubemapSLAM: A Piecewise-Pinhole Monocular Fisheye SLAM System},
      author={Wang, Yahui and Cai, Shaojun and Li, Shi-Jie and Liu, Yun and Guo, Yangyan and Li, Tao and Cheng, Ming-Ming},
      journal={arXiv preprint arXiv:1811.12633},
      year={2018}
    }

    @article{murTRO2015,
      title={{ORB-SLAM}: a Versatile and Accurate Monocular {SLAM} System},
      author={Mur-Artal, Ra\'ul, Montiel, J. M. M. and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={31},
      number={5},
      pages={1147--1163},
      doi = {10.1109/TRO.2015.2463671},
      year={2015}
    }

    @article{murORB2,
      title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
      author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={33},
      number={5},
      pages={1255--1262},
      doi = {10.1109/TRO.2017.2705103},
      year={2017}
    }

    @article{urban2016multicol,
      title={MultiCol-SLAM-a modular real-time multi-camera slam system},
      author={Urban, Steffen and Hinz, Stefan},
      journal={arXiv preprint arXiv:1610.07336},
      year={2016}
    }

# 2. Prerequisites
The system has been tested in Ubuntu 16.04, but it should be easy to compile in other platforms. A powerful computer will ensure real-time performance and provide more stable and accurate results. Since we use the new thread and chrono functionalities, it should be compiled with C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **We have tested our system with OpenCV 2.4.11 and OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use the modified versions from [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2) of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

# 3. Building CubemapSLAM

Clone the repository:
```
git clone https://github.com/nkwangyh/CubemapSLAM
```

We provide a script `build.sh` to build the *Thirdparty* libraries and *CubemapSLAM*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd CubemapSLAM
chmod +x build.sh
./build.sh
```

This will create **libCubemapSLAM.so**  at *lib* folder and the executables **cubemap_fangshan**, **cubemap_lafida** in *bin* folder.

# 4. Examples

## Lafida Dataset

1. Download the sequences from https://www.ipf.kit.edu/lafida_datasets.php

2. Substitude the data paths of the scripts in the _Scripts_ folder and run the script.

```
    cd Scripts
    sudo chmod +x ./runCubemapLafida.sh
    ./runCubemapLafida.sh
```

## The Self-collected Dataset  

1. We currently release the _loop2_ and the _parkinglot_ sequences. Download the sequences from Google drive: [loop2_front](https://drive.google.com/open?id=19P7teqQk45EJMYPp9WjJNi5hCJWqudXH), [loop2_left](https://drive.google.com/open?id=1dbUyXD11hzh0OF-ukFOVQJlMRb5b4i4R), [parkinglot_front](https://drive.google.com/open?id=1Its2fOEIxEUzY1Tpkh0-r3_0xwxaAIJe) and [parkinglot_left](https://drive.google.com/open?id=1PDBgrbIIZnmzDkB4U6pxqHpy1zfGAR3S). 

2. Substitude the data paths of the scripts in the _Scripts_ folder and run the scripts, for example,

```
    cd Scripts
    sudo chmod +x ./runCubemapParkinglotFront.sh
    ./runCubemapParkinglotFront.sh
```

# 5. Processing your own sequences
To process your own sequences, you should first create your own setting file like the others in the _Config_ folder. To do that, you should calibrate your fisheye camera with the toolbox from [Prof. Davide Scaramuzza](https://sites.google.com/site/scarabotix/ocamcalib-toolbox) and fill in the intrinsic parameters as well as the image resolution and camera FoV. You can also change the face resolution of the cubemap. After that, you need to create an image name list to feed the system. Besides, a mask is needed in the feature extraction process to guarantee the performance.

# 6. SLAM and Localization Modes
You can change between the *SLAM* and *Localization mode* using the GUI of the map viewer.

### SLAM Mode
This is the default mode. The system runs in parallal three threads: Tracking, Local Mapping and Loop Closing. The system localizes the camera, builds new map and tries to close loops.

### Localization Mode
This mode can be used when you have a good map of your working area. In this mode the Local Mapping and Loop Closing are deactivated. The system localizes the camera in the map (which is no longer updated), using relocalization if needed. 

