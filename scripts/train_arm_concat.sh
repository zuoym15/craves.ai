CUDA_VISIBLE_DEVICES=0 python ../train_arm.py \
-a hg --stacks 2 --blocks 1 --num-classes 17 \
--checkpoint ../checkpoint/train_concat \
--data-dir ../data/20181107/ ../data/ft_20181105 \
--meta-dir ../data/meta/17_vertex \
--epoch 30 --schedule 20 --train-batch 6 --test-batch 6 \
--anno-type 3D 2D --ratio 6 4 --training-set-percentage 0.9 1.0 \
--replace-bg --random-bg-dir ../background_img 