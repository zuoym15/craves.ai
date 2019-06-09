CUDA_VISIBLE_DEVICES=0 python train_arm.py \
-a hg --stacks 2 --blocks 1 --num-classes 17 \
-e -f \
--checkpoint ./checkpoint/arm/20181109 \
--resume ./checkpoint/arm/20181106/checkpoint.pth.tar \
--data-dir ./data/20181107 \
--meta-dir ./data/meta/17_vertex \
--sample-img-dir ./visualization/20181109/20181108 \
--camera-type synthetic \
--anno-type 3D