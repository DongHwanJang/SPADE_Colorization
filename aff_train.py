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
from trainers.affin_trainer import AffinTrainer
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
            trainer.val_affinnet_one_step(data_i)

            # get losses and rename the keys to include "VAL_"
            loss = trainer.get_latest_losses()

            if i % opt.val_display_freq == 0:
                visual_list = [
                    ('VAL' + str(epoch) + '_attention_map', trainer.get_latest_attention()),
                    ('VAL' + str(epoch) + '_target_LAB', data_i['target_LAB']),
                ]

                visuals = OrderedDict(visual_list)

                visualizer.display_current_results(visuals, epoch, i)


    for i, data_i in enumerate(train_dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # if i % opt.D_steps_per_G == 0:
        losses = {}
        subnet_losses = {}
        affinnet_losses = {}

        trainer.run_affinnet_one_step(data_i)

        total_loss = {'loss': trainer.get_latest_losses()}

        visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                        total_loss, iter_counter.time_per_iter)
        visualizer.plot_current_errors(total_loss, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():

            visual_list = [('attention_map', trainer.get_latest_attention()),
                           ('target_LAB', data_i['target_LAB'])
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
