import argparse
import torch.nn as nn
from torchvision import transforms
from net.network import BriGSC
from net.blip_encoder import BLIP
from loss.distortion import *
from utils import *
from PIL import Image
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='BriGSC')
parser.add_argument('--datasets', type=str, default='datasets/kodak', help='dataset path')
parser.add_argument('--model', type=str, default='BriGSC', choices=['BriGSC', 'BriGSC_W/O'], help='BriGSC model or BriGSC without Channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], help='wireless channel model, awgn or rayleigh')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='evaluation metrics')
parser.add_argument('--comp_size', type=int, default=1024, help='compress size')
parser.add_argument('--multiple-snr', type=str, default='10', help='random or fixed snr')
args = parser.parse_args()


class Config:
    seed = 1025
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    model_path = f"ckpt/brigsc.model"
    vit_model_path = None
    logger = None
    compress = True

    image_dims = (3, 256, 256)
    downsample = 4
    vit_embed_dims = 768

    brigsc_encoder_kwargs = dict(
        img_size=image_dims[1], patch_size=16, in_chans=3, embed_dims=vit_embed_dims,
        depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        comp_size=args.comp_size, model=args.model
    )
    brigsc_decoder_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
        C=vit_embed_dims, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True, downsample=downsample, model=args.model
    )


def load_weights(net, model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=False)
    del pretrained


CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


if __name__ == '__main__':
    seed_torch()
    torch.manual_seed(seed=Config.seed)

    brigsc = BriGSC(args, Config)
    brigsc = brigsc.cuda()
    brigsc.eval()
    load_weights(brigsc, Config.model_path)

    text_extractor = BLIP()
    text_extractor = text_extractor.cuda()
    text_extractor.eval()

    img_list = os.listdir(args.datasets)

    psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(4)]
    metrics = [psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))

    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            for idx, img_name in enumerate(img_list):
                input = Image.open(os.path.join(args.datasets, img_name)).convert('RGB')

                input = transforms.ToTensor()(input).unsqueeze(0)
                input = transforms.Resize((Config.image_dims[1], Config.image_dims[2]))(input)
                input = input.cuda()

                recon_image, CBR, SNR, mse, loss_G = brigsc(input, SNR)
                caption, image_embeds = text_extractor(input)

                img = transforms.ToPILImage()(recon_image.squeeze().cpu())
                img.save(f"./output/{idx}-{caption[0]}.jpg")

                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                print(f"[Finish test {img_name}] SNR: {SNR} | PSNR: {psnr} | MS-SSIM: {msssim} | Caption: {caption[0]}")

        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    print("AVG SNR: {}".format(results_snr.tolist()))
    print("AVG CBR: {}".format(results_cbr.tolist()))
    print("AVG PSNR: {}".format(results_psnr.tolist()))
    print("AVG MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

