from net.brigsc_encoder import *
from net.brigsc_decoder import *
from loss.distortion import Distortion
from net.channel import Channel
from random import choice
import torch.nn as nn
import torch


class BriGSC(nn.Module):
    def __init__(self, args, config):
        super(BriGSC, self).__init__()
        self.config = config
        encoder_kwargs = config.brigsc_encoder_kwargs
        decoder_kwargs = config.brigsc_decoder_kwargs
        self.encoder = create_BriGSCEncoder(**encoder_kwargs)  # BriGSC_Encoder
        self.decoder = create_BriGSCDecoder(**decoder_kwargs)  # BriGSC_Decoder

        if self.config.vit_model_path is not None:
            self.encoder.load_pretrained(self.config.vit_model_path)

        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)

        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, snr=None):
        B, C, H, W = input_image.shape

        if snr is None:
            snr = choice(self.multiple_snr)

        # BriGSC_Encoder
        feature, ori_vit_embeddings = self.encoder(input_image, snr, compress=self.config.compress)
        # feature = feature.type(torch.float16)

        CBR = feature.numel() / 2 / input_image.numel()

        # Feature pass channel
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, snr)
        else:
            noisy_feature = feature

        # BriGSC_Decoder
        recon_image = self.decoder(noisy_feature, snr)

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))

        return recon_image, CBR, snr, mse.mean(), loss_G.mean()
