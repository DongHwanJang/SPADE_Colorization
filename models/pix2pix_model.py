"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
from util.img_loader import lab_deloader
from torchvision.transforms import functional as F
from util import img_loader

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        if opt.use_wandb:
            opt.wandb.watch(self.netG, log="all")
            opt.wandb.watch(self.netD, log="all")

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                # self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids, vgg=self.netG.corr_subnet.vgg)
                # set vgg=None because the version for vgg in perceptual loss may be different
                # with that in using corr_feat
                self.criterionVGG = networks.VGGLoss(self.opt, self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.use_smoothness_loss:
                self.smoothnessLoss = networks.SmoothnessLoss()
            if opt.use_reconstruction_loss:
                self.reconstructionLoss = networks.ReconstructionLoss()
            if opt.use_contextual_loss:
                self.contextualLoss = networks.ContextualLoss()

    # parses LAB into L and AB.
    # This shouldn't make new copies. It should also handle batch cases
    def parse_LAB(self, image_LAB):
        if len(image_LAB.size()) == 4:
            image_L = image_LAB[:, 0, :, :].unsqueeze(1)
            image_A = image_LAB[:, 1, :, :].unsqueeze(1)
            image_B = image_LAB[:, 2, :, :].unsqueeze(1)
            image_AB = torch.cat([image_A, image_B], 1)
            return image_L, image_AB

        elif len(image_LAB.size()) == 3:
            # It would be a tensor whose batch size = 1.
            # Then, unsqueeze to be 4D tensor?
            return self.parse_LAB(image_LAB.unsqueeze(0))

        else:
            raise ("Pass 3D or 4D tensor")



    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        reference_LAB = data["reference_LAB"]
        target_LAB = data["target_LAB"]
        target_RGB = data["target_image"]
        target_L, target_AB = self.parse_LAB(target_LAB)

        if mode == 'generator':
            g_loss, generated, attention, conf_map = self.compute_generator_loss(
                target_L, target_LAB, target_RGB, reference_LAB, is_reconstructing=data["is_reconstructing"])
            return g_loss, generated, attention, conf_map
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(target_L, target_LAB, reference_LAB)
            return d_loss
        # elif mode == 'encode_only':
        #     z, mu, logvar = self.encode_z(real_image)
        #     return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_AB, _, attention, conf_map = self.generate_fake(target_L, reference_LAB)
                fake_LAB = torch.cat([target_L, fake_AB], dim=1)
            return fake_LAB
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        # label_map = data['label']
        # bs, _, h, w = label_map.size()
        # nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
        #     else self.opt.label_nc
        # input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        # if not self.opt.no_instance:
        #     inst_map = data['instance']
        #     instance_edge_map = self.get_edges(inst_map)
        #     input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return data['label'], data['image']

    def compute_generator_loss(self, target_L, target_LAB, target_RGB, reference_LAB, is_reconstructing=False):
        G_losses = {}


        # if not using VAE, this is just a forward pass of G
        fake_AB, _, attention, conf_map = self.generate_fake(target_L, reference_LAB)
        # FIXME: where is the best place(=line) that concat gt luminance to generated_AB
        fake_LAB = torch.cat([target_L, fake_AB], dim=1)

        fake_RGB = img_loader.torch_lab2rgb(fake_LAB, normalize=True)

        # if self.opt.use_vae:
        #     G_losses['KLD'] = KLD_loss

        # We let discriminator compare fake_LAB and target_LAB.
        pred_fake, pred_real = self.discriminate(target_L, fake_LAB, target_LAB)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        # calculate feature matching loss with L1 distance
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss * self.opt.lambda_feat

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_RGB, target_RGB) * self.opt.lambda_vgg
            # pass

        if self.opt.use_smoothness_loss:
            G_losses["smoothness"] = self.smoothnessLoss.forward(fake_LAB[:, 1:, :, :]) * self.opt.lambda_smooth # put fake_AB

        # TODO is_reconstructing.size() is not 1 for the batch input case
        if is_reconstructing:
            G_losses["reconstruction"] = self.reconstructionLoss(fake_LAB, reference_LAB) * self.opt.lambda_recon

        if self.opt.use_contextual_loss:
            G_losses["contextual"] = self.contextualLoss(fake_LAB, reference_LAB) * self.opt.lambda_context

        return G_losses, fake_LAB, attention, conf_map

    def compute_discriminator_loss(self, target_L, target_LAB, reference_LAB):
        D_losses = {}
        with torch.no_grad():
            fake_AB, _, _, _ = self.generate_fake(target_L, reference_LAB)
            fake_AB = fake_AB.detach()
            fake_AB.requires_grad_()
            fake_LAB = torch.cat([target_L, fake_AB], dim=1)

        pred_fake, pred_real = self.discriminate(
            target_L, fake_LAB, target_LAB)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    # def encode_z(self, real_image):
    #     mu, logvar = self.netE(real_image)
    #     z = self.reparameterize(mu, logvar)
    #     return z, mu, logvar

    def generate_fake(self, target_L, reference_LAB, compute_kld_loss=False):
        z = None
        KLD_loss = None
        # if self.opt.use_vae:
        #     z, mu, logvar = self.encode_z(real_image)
        #     if compute_kld_loss:
        #         KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        # G forward during training
        fake_image, attention, conf_map = self.netG(target_L, reference_LAB, z=z)
        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss, attention, conf_map

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.
    # channel-wise concatenates input (of G) and fake, input (of G) and real, then
    # concatenates the two again channelwise
    def discriminate(self, target_L, fake_LAB, target_LAB):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_LAB, target_LAB], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    # pred is a list of list of feature maps
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []

            # iterate through each discriminator's output
            for p in pred:

                # iterate through each feature map in the discriminator's output
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        # fake, real are list of list of tensors
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
