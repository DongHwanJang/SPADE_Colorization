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

# parse options
opt = TrainOptions().parse()
if opt.use_wandb:
    wandb.init(project="SPADE Colorization", name=opt.name, resume=opt.continue_train, magic=True)
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

        # to run reconstruction loss, set reference_LAB = target_LAB
        if opt.use_reconstruction_loss and i % opt.reconstruction_period == 0:
            data_i["reference_LAB"] = data_i["target_LAB"].clone().detach()
            data_i["is_reconstructing"] = True

        if not opt.use_reconstruction_loss or i % opt.reconstruction_period != 0:
            data_i["is_reconstructing"] = False

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        losses = trainer.get_latest_losses()
        visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                        losses, iter_counter.time_per_iter)
        visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        # if iter_counter.needs_displaying():
        if True:
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('conf_map', trainer.get_latest_conf_map()),
                                   ('attention_map', trainer.get_latest_attention()),
                                   ('warped_img_LAB', trainer.get_latest_warped_ref_img()),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('target_image', data_i['target_image']),
                                   ('reference_image', data_i['reference_image']),
                                   ('target_L_gray_image', data_i['target_L_gray_image']),
                                   ('target_LAB', data_i['target_LAB']),
                                   ('reference_LAB', data_i['reference_LAB']),
                                   ])
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
