export CUDA_VISIBLE_DEVICES=5
python train.py --name train_subnet_generator_only --use_reconstruction_loss --use_contextual_loss --use_smoothness_loss --pair_file pair_img/train.txt --no_fid --dataroot /data1/imagenet/ --niter 100 --niter_decay 100 --batchSize 1 --ref_type l --train_subnet_only --use_wandb --flip_to_target --crop_to_target --crop_to_ref