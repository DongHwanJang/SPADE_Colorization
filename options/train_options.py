"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import wandb

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--use_html', action='store_true', help='save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument("--pair_file", type=str)
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument("--D_in_channel", type=int, default=3, help="D input channel. 2 or 3")

        parser.add_argument("--crop_to_ref", action='store_true')
        parser.add_argument('--crop_to_ref_size', type=int, default=256)
        parser.add_argument("--crop_to_target", action='store_true')
        parser.add_argument('--crop_to_target_size', type=int, default=180) # ~70% of 256x256
        parser.add_argument("--flip_to_target", action='store_true')

        parser.add_argument('--use_gamma', action='store_true', help='parameterize how much attention will be applied')
        parser.add_argument('--use_smoothness_loss', action='store_true', help='Use smoothness loss')
        parser.add_argument('--use_reconstruction_loss', action='store_true', help='Use reconstruction loss')
        parser.add_argument('--use_contextual_loss', action='store_true', help='Use contextual loss')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases')
        parser.add_argument('--wandb_user_name', default="isaac", help='key used to find pai key in ".wandb_api_keys.json"')
        parser.add_argument('--train_subnet_only', action='store_true',
                            help='whether to train only with correspondence subnet')
        parser.add_argument('--train_subnet', action='store_true',
                            help='whether to train full network with supplementary training')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1,
                            help='number of discriminator iterations per generator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=1, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=1, help='weight for feature loss')
        parser.add_argument('--lambda_smooth', type=float, default=0.00001, help='weight for smoothness loss')
        parser.add_argument('--lambda_recon', type=float, default=0.1, help='weight for reconstruction loss')
        parser.add_argument('--lambda_context', type=float, default=0.25, help='weight for contextual loss')

        parser.add_argument('--subnet_reconstruction_period', type=float, default=3, help='weight for feature matching loss')
        parser.add_argument('--lambda_subnet_feat', type=float, default=1, help='weight for feature matching loss')
        parser.add_argument('--lambda_subnet_vgg', type=float, default=1, help='weight for feature loss')
        parser.add_argument('--lambda_subnet_smooth', type=float, default=0.000001, help='weight for smoothness loss')
        parser.add_argument('--lambda_subnet_softmax', type=float, default=0.1, help='weight for cross entropy loss')
        parser.add_argument('--lambda_subnet_l1', type=float, default=0.01, help='weight for l1 loss')

        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image|sagan)')
        parser.add_argument('--lambda_kld', type=float, default=0.05)
        parser.add_argument("--use_attention_convs", action="store_true")

        parser.add_argument('--reconstruction_period', type=int, default=10)
        parser.add_argument('--train_subnet_period', type=int, default=10)
        parser.add_argument('--subnet_load_size', type=int, default=256,
                            help='Scale images to this size (for subnet only)')
        parser.add_argument('--subnet_crop_size', type=int, default=64,
                            help='Crop to the width of crop_size (for subnet only)')

        parser.add_argument('--val_freq', type=int, default=10,
                            help='Run validation after this many epochs')
        parser.add_argument('--val_display_freq', type=int, default=8,
                            help='Run validation after this many epochs')

        self.isTrain = True

        return parser
