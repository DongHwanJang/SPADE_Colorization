"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import os
import wandb
import json
import torch

# parse options
opt = TrainOptions().parse()
if opt.use_wandb:
    json_path = "/home/minds/.wandb_api_keys.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as j:
            json_file = json.loads(j.read())
            os.environ["WANDB_API_KEY"] = json_file[opt.wandb_user_name]
    wandb.init(entity="eccv2020_best_paper", project="SPADE Colorization", name=opt.name,
               resume=opt.continue_train, magic=True)
    wandb.config.update(opt)
    opt.wandb = wandb

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        if not opt.no_fid and i % opt.fid_period == 0:
            data_i["get_fid"] = True

        if opt.no_fid or i % opt.fid_period != 0:
            data_i["get_fid"] = False

        if opt.train_subnet_only or (opt.train_subnet and i % opt.train_subnet_period == 0):
            data_i["is_training_subnet"] = True

        if not (opt.train_subnet_only or (opt.train_subnet and i % opt.train_subnet_period == 0)):
            data_i["is_training_subnet"] = False

        # to run reconstruction loss, set reference_LAB = target_LAB
        if opt.use_reconstruction_loss and i % opt.reconstruction_period == 0:
            data_i["reference_LAB"] = data_i["target_LAB"].clone().detach()
            data_i["is_reconstructing"] = True

        if not opt.use_reconstruction_loss or i % opt.reconstruction_period != 0:
            data_i["is_reconstructing"] = False

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            if opt.train_subnet_only or data_i["is_training_subnet"]:
                with torch.autograd.set_detect_anomaly(True):
                    trainer.run_subnet_generator_one_step(data_i)
            else:
                trainer.run_generator_one_step(data_i)


        # train discriminator
        if opt.train_subnet_only or data_i["is_training_subnet"]:
            trainer.run_subnet_discriminator_one_step(data_i)
        else:
            trainer.run_discriminator_one_step(data_i)


        # Visualizations
        losses = {}
        subnet_losses = {}
        if opt.train_subnet_only or data_i["is_training_subnet"]:
            subnet_losses = trainer.get_subnet_latest_losses()

        else:
            losses = trainer.get_latest_losses()

        total_losses = {**losses, **subnet_losses}

        visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                        total_losses, iter_counter.time_per_iter)
        visualizer.plot_current_errors(total_losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visual_list = [('input_label', data_i['label'])]
            if opt.train_subnet_only or data_i["is_training_subnet"]:
                visual_list += [
                               ('subnet_warped_LAB_gt_resized', data_i['subnet_warped_LAB_gt_resized']),
                               ('subnet_index_gt_resized', data_i['subnet_index_gt_resized']
                                .unsqueeze(1).repeat(1, 3, 1, 1)),
                               ('subnet_synthesized_image', trainer.get_subnet_latest_generated()),
                               ('subnet_synthesized_index', trainer.get_subnet_latest_index()),
                               ('subnet_target_L_gray_image', data_i['subnet_target_L_gray_image']),
                               ('subnet_target_LAB', data_i['subnet_target_LAB']),
                               ('subnet_ref_LAB', data_i['subnet_ref_LAB']),
                               ]

            else:
                visual_list += [
                               ('conf_map', trainer.get_latest_conf_map()),
                               ('attention_map', trainer.get_latest_attention()),
                               ('warped_img_LAB', trainer.get_latest_warped_ref_img()),
                               ('synthesized_image', trainer.get_latest_generated()),
                               ('target_image', data_i['target_image']),
                               ('reference_image', data_i['reference_image']),
                               ('target_L_gray_image', data_i['target_L_gray_image']),
                               ('target_LAB', data_i['target_LAB']),
                               ('reference_LAB', data_i['reference_LAB']),
                               ]

            visuals = OrderedDict(visual_list)

            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if data_i["get_fid"]:
            if opt.train_subnet_only or data_i["is_training_subnet"]:
                visualizer.display_value("subnet_fid", trainer.get_subnet_latest_fid(), iter_counter.total_steps_so_far)

            else:
                visualizer.display_value("fid", trainer.get_latest_fid(), iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
            epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

if opt.use_wandb:
    wandb.save(opt.name + ".h5")
print('Training was successfully finished.')
