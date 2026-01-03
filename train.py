import time
import argparse
import torch.nn as nn
import torch.optim as optim
from net.network import BriGSC
from data.datasets import get_loader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from utils import *
from datetime import datetime
from loss.distortion import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='BriGSC')
parser.add_argument('--training', type=bool, default=True, help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K', help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak', choices=['kodak', 'ori'], help='specify the testset for HR models')
parser.add_argument('--model', type=str, default='BriGSC_W/O', choices=['BriGSC', 'BriGSC_W/O'], help='BriGSC model or BriGSC without Channel ModNet')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='evaluation metrics')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], help='wireless channel model, awgn or rayleigh')
parser.add_argument('--comp_size', type=int, default=512, help='compress size')
parser.add_argument('--multiple-snr', type=str, default='1', help='random or fixed snr')
parser.add_argument('--first_stage_ckpt', type=str, default='./history/512-1/models/train_best.model', help='the checkpoint of first stage')
args = parser.parse_args()


class Config:
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False

    vit_model_path = None
    if args.model is not "BriGSC" and args.training is True:
        vit_model_path = "./ckpt/vit_model_base.pth"

    # logger
    filename = datetime.now().strftime("%Y-%m-%d").__str__()
    workdir = f'./history/{filename}'
    log = workdir + f'/Log_{filename}.log'
    train_tensorboard_dir = workdir + '/logs/train'
    test_tensorboard_dir = workdir + '/logs/test'
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None
    train_tensorboard_writer = SummaryWriter(train_tensorboard_dir)
    test_tensorboard_writer = SummaryWriter(test_tensorboard_dir)

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000
    compress = True

    save_model_freq = 10
    image_dims = (3, 256, 256)
    batch_size = 16
    downsample = 4
    vit_embed_dims = 768

    train_data_dir = ["./datasets/DIV2K_train_HR/"]
    if args.testset == 'kodak':
        test_data_dir = ["./datasets/kodak/"]
    elif args.testset == 'ori':
        test_data_dir = ["./datasets/ori/"]

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


CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def load_weights(net, model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=False)
    del pretrained


def train_one_epoch(epoch):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step

    for batch_idx, input in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input = input.cuda()
        recon_image, CBR, SNR, mse, loss_G = net(input)
        loss = loss_G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
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

    Config.train_tensorboard_writer.add_scalar("Loss", losses.avg, epoch)
    Config.train_tensorboard_writer.add_scalar("PSNR", psnrs.avg, epoch)
    Config.train_tensorboard_writer.add_scalar("MS-SSIM", msssims.avg, epoch)
    mean_psnr = psnrs.avg

    process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
    log = (' | '.join([
        f'Epoch {epoch}',
        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
        f'Time {elapsed.val:.3f}',
        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
        f'MS-SSIM {msssims.val:.3f} ({msssims.avg:.3f})',
        f'Lr {cur_lr}',
    ]))
    logger.info(log)
    for i in metrics:
        i.clear()

    return mean_psnr


def evaluation(epoch=None):
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")

    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))

    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            for batch_idx, input in enumerate(test_loader):
                input = transforms.Resize((Config.image_dims[1], Config.image_dims[2]))(input)
                start_time = time.time()

                input = input.cuda()
                recon_image, CBR, SNR, mse, loss_G = net(input, SNR)

                img = transforms.ToPILImage()(recon_image.squeeze().cpu())
                img.save(f"./output/{SNR}_{batch_idx}.png")

                elapsed.update(time.time() - start_time)
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

                log = (' | '.join([
                    f'Time {elapsed.val:.3f}',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f}',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)

        if epoch is not None:
            Config.test_tensorboard_writer.add_scalar("PSNR", psnrs.avg, epoch)
            Config.test_tensorboard_writer.add_scalar("MS-SSIM", msssims.avg, epoch)

        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    results_mean_psnr = np.mean(results_psnr)
    results_mean_msssim = np.mean(results_msssim)

    if epoch is not None:
        Config.test_tensorboard_writer.add_scalar("PSNR", results_mean_psnr, epoch)
        Config.test_tensorboard_writer.add_scalar("MS-SSIM", results_mean_msssim, epoch)

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MEAN PSNR: {}".format(results_mean_psnr))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

    return results_mean_psnr


if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(Config, save_log=args.training)
    logger.info(Config.__dict__)

    torch.manual_seed(seed=Config.seed)

    net = BriGSC(args, Config)
    net = net.cuda()

    cur_lr = Config.learning_rate
    model_params = [{'params': net.parameters(), 'lr': cur_lr}]
    train_loader, test_loader = get_loader(args, Config)
    optimizer = optim.Adam(model_params, lr=cur_lr)

    global_step = 0

    if not args.training or args.model == "BriGSC":
        load_weights(net, args.first_stage_ckpt)

    if args.training:
        steps_epoch = global_step // train_loader.__len__()
        train_max_psnr = 0
        test_max_psnr = 0
        for epoch in range(steps_epoch, Config.tot_epoch):
            train_psnr = train_one_epoch(epoch)
            if train_psnr > train_max_psnr:
                train_max_psnr = train_psnr
                save_model(net, save_path=Config.models + '/train_best.model')
            if (epoch + 1) % Config.save_model_freq == 0:
                save_model(net, save_path=Config.models + '/{}_EP{}.model'.format(Config.filename, epoch + 1))
                test_mean_psnr = evaluation(epoch)
                if (test_mean_psnr > test_max_psnr):
                    test_max_psnr = test_mean_psnr
                    save_model(net, save_path=Config.models + '/test_best.model')
    else:
        evaluation()
