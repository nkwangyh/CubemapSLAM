cd ThirdParty/g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../

cd DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../

cd ../build
cmake ..
make -j4
