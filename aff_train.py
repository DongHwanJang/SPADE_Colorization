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
if opt.val_freq != -1:
    train_dataloader, val_dataloader = data.create_dataloader(opt)
else:
    train_dataloader = data.create_dataloader(opt)

# create trainer for our model
# trainer = Pix2PixTrainer(opt)
trainer = AffinTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(train_dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)

    if opt.val_freq != -1 and epoch % opt.val_freq == 0:
        print("Validation at epoch: " + str(epoch))
        for i, data_i in zip(range(1, len(val_dataloader) + 1), val_dataloader):
            # TODO
            trainer.val_generator_one_step(data_i)

            # get losses and rename the keys to include "VAL_"
            losses = trainer.get_latest_losses(get_D_losses=False)
            for key in losses:
                new_key = "VAL" + str(epoch) + "_" + key
                losses[new_key] = losses[key]
                del losses[key]

            # visualizer.plot_current_errors(losses, i)

            if i % opt.val_display_freq == 0:
                visual_list = [
                    ('VAL' + str(epoch) + '_conf_map', trainer.get_latest_conf_map()),
                    ('VAL' + str(epoch) + '_attention_map', trainer.get_latest_attention()),
                    ('VAL' + str(epoch) + '_warped_img_LAB', trainer.get_latest_warped_ref_img()),
                    ('VAL' + str(epoch) + '_synthesized_image', trainer.get_latest_generated()),
                    ('VAL' + str(epoch) + '_target_image', data_i['target_image']),
                    ('VAL' + str(epoch) + '_reference_image', data_i['reference_image']),
                    ('VAL' + str(epoch) + '_target_L_gray_image', data_i['target_L_gray_image']),
                    ('VAL' + str(epoch) + '_target_LAB', data_i['target_LAB']),
                    ('VAL' + str(epoch) + '_reference_LAB', data_i['reference_LAB']),
                ]

                visuals = OrderedDict(visual_list)

                visualizer.display_current_results(visuals, epoch, i)
                if data_i["get_fid"]:
                    visualizer.display_value("VAL" + str(epoch) + "_fid", trainer.get_latest_fid(), i)


    for i, data_i in enumerate(train_dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # if i % opt.D_steps_per_G == 0:
        losses = {}
        subnet_losses = {}
        affinnet_losses = {}

        ## This part is added for affinityNet training
        ## TODO
        trainer.run_affinnet_generator(data_i)
        affinnet_losses = trainer.get_affinnet_latest_losses()

        total_losses = {**losses}

        visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                        total_losses, iter_counter.time_per_iter)
        visualizer.plot_current_errors(total_losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visual_list = [('input_label', data_i['label'])]

            if not opt.train_subnet_only:
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
