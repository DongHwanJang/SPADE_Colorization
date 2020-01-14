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
import PIL
from skimage import io, color
from skimage.transform import downscale_local_mean

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, 1)

# create tool for visualization
visualizer = Visualizer(opt)

iteration = 100

target_path = "/DATA1/hksong/imagenet/n04428191/n04428191_39506.JPEG"
reference_path ="/DATA1/hksong/imagenet/n04428191/n04428191_39564.JPEG"

target_LAB = PIL.Image(target_path).convert("LAB")

reference_LAB = PIL.Image(reference_path).convert("LAB")

data = {
    "target_LAB":target_LAB,
    "reference_LAB":reference_LAB
}

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i in range(iteration):

        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data)

        # train discriminator
        trainer.run_discriminator_one_step(data)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data['target_LAB'][0]),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data['reference_LAB'])])
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

print('Training was successfully finished.')
