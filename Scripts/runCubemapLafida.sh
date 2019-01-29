CUBE_MASK="../Masks/gray_lafida_cubemap_mask_650.png"
#CUBE_MASK="../Masks/gray_lafida_cubemap_mask_550.png"
#CUBE_MASK="../Masks/gray_lafida_cubemap_mask_450.png"

#lafida indoor dynamic sequence
DATASET_PATH="/home/wangyh/DataSets/indoor_dynamic/imgs/cam0"
IMG_TS_PATH="/home/wangyh/DataSets/indoor_dynamic/images_and_timestamps.txt"

#lafida indoor dynamic
../bin/cubemap_lafida ../Vocabulary/ORBvoc.txt \
    ../Config/lafida_cam0_params.yaml \
    "$DATASET_PATH" \
    "$IMG_TS_PATH" \
    "$CUBE_MASK" \
    ../bin/trajs/lafida_indoor_dynamic.txt \
    ../bin/trajs/lafida_indoor_dynamic_perf.txt

##lafida indoor static
#../bin/cubemap_lafida ../Vocabulary/ORBvoc.txt \
    #../Config/lafida_cam0_params.yaml \
    #"$DATASET_PATH" \
    #"$IMG_TS_PATH" \
    #"$CUBE_MASK" \
    #../bin/trajs/lafida_indoor_static.txt \
    #../bin/trajs/lafida_indoor_static_perf.txt

##lafida outdoor static
#../bin/cubemap_lafida ../Vocabulary/ORBvoc.txt \
    #../Config/lafida_cam0_params.yaml \
    #"$DATASET_PATH" \
    #"$IMG_TS_PATH" \
    #"$CUBE_MASK" \
    #../bin/trajs/lafida_outdoor_static.txt \
    #../bin/trajs/lafida_outdoor_static_perf.txt

##lafida outdoor static2
#../bin/cubemap_lafida ../Vocabulary/ORBvoc.txt \
    #../Config/lafida_cam0_params.yaml \
    #"$DATASET_PATH" \
    #"$IMG_TS_PATH" \
    #"$CUBE_MASK" \
    #../bin/trajs/lafida_outdoor_static2.txt \
    #../bin/trajs/lafida_outdoor_static2_perf.txt

##lafida outdoor rotation
#../bin/cubemap_lafida ../Vocabulary/ORBvoc.txt \
    #../Config/lafida_cam0_params.yaml \
    #"$DATASET_PATH" \
    #"$IMG_TS_PATH" \
    #"$CUBE_MASK" \
    #../bin/trajs/lafida_outdoor_rotation.txt \
    #../bin/trajs/lafida_outdoor_rotation_perf.txt

##lafida outdoor large loop
#../bin/cubemap_lafida ../Vocabulary/ORBvoc.txt \
    #../Config/lafida_cam0_params.yaml \
    #"$DATASET_PATH" \
    #"$IMG_TS_PATH" \
    #"$CUBE_MASK" \
    #../bin/trajs/lafida_outdoor_large_loop.txt \
    #../bin/trajs/lafida_outdoor_large_loop_perf.txt
