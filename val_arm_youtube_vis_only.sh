CUDA_VISIBLE_DEVICES=0 python train_arm.py \
-a hg --stacks 2 --blocks 1 --num-classes 17 \
-e -f \
--checkpoint ./checkpoint/arm/20181106 \
--resume ./checkpoint/arm/20181106/checkpoint.pth.tar \
--data-dir ./data/youtube_20181105 \
--meta-dir ./data/meta/17_vertex \
--save-result-dir ./visualization/20181106/youtube_20181105 \
--anno-type 2d \
--training-set-percentage 0 --camera-type video_shifted --ignore-invis-pts