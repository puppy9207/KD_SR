import os
import hydra
import wandb
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from models import Generator, MultiscaleDiscriminator, Discriminator_PatchGAN
from loss import PerceptualLoss, GANLoss
from utils import reduce_loss_dict, requires_grad, sample_data, yuv2rgb
from dataset import Dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from collections import OrderedDict

def setup(rank, world_size):
    """DDP 디바이스 설정"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)

def cleanup():
    """Kill DDP process group"""
    dist.destroy_process_group()

def gan_trainer(cfg, gpu, loader, generator, discriminator):
    if not os.path.exists(cfg.training.GAN.ckpt_dir):
        os.makedirs(cfg.training.GAN.ckpt_dir)

    generator.train()
    discriminator.train()

    loss_dict = {}
    start_iters = 0
    best_lpips = 0
    best_psnr = 0

    psnr_criterion = nn.MSELoss().to(gpu)
    pixel_criterion = nn.L1Loss().to(gpu)
    content_criterion = PerceptualLoss(cfg.training.GAN.perceptual_opt).to(gpu)
    adversarial_criterion = GANLoss(cfg.training.GAN.gan_opt).to(gpu)
    # identity_criterion = IdentityLoss(cfg.training.GAN.identity_opt).to(gpu)

    # g_optim = torch.optim.Adam(generator.parameters(), cfg.training.GAN.lr, (0.9, 0.999))
    # d_optim = torch.optim.Adam(discriminator.parameters(), cfg.training.GAN.lr, (0.9, 0.999))
    d_optim = torch.optim.RAdam(discriminator.parameters(), cfg.training.GAN.lr, (0.9, 0.999))
    g_optim = torch.optim.RAdam(generator.parameters(), cfg.training.GAN.lr, (0.9, 0.999))

    if os.path.exists(cfg.training.GAN.resume):
        ckpt = torch.load(cfg.training.GAN.resume, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"])
        start_iters = ckpt["iteration"] + 1
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        best_psnr = ckpt["best_psnr"]
        best_lpips = ckpt["best_lpips"]
        
    if start_iters > 0 :
        g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optim, [200000, 400000, 800000, 900000], 0.5, last_epoch=start_iters)
        d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optim, [200000,400000, 800000, 900000], 0.5, last_epoch=start_iters)
    else:
        g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optim, [200000, 400000, 800000, 900000], 0.5)
        d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optim, [200000, 400000, 800000, 900000], 0.5)

    for i in range(start_iters, cfg.training.GAN.n_iters):
        lr, hr = next(loader)
        lr = lr.to(gpu)
        hr = hr.to(gpu)
        
        """ Discriminator 학습 """
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        
        preds = generator(lr)
        fake_pred = discriminator(preds)
        real_pred = discriminator(hr)

        d_loss = 0
        for fake, real in zip(fake_pred, real_pred):
            d_loss += adversarial_criterion(real, True)
            d_loss += adversarial_criterion(fake, False)
        d_loss /= 2 * len(real_pred)

        loss_dict["d_loss"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()
        d_scheduler.step()

        """ Generator 학습 """
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        preds = generator(lr)
        fake_pred = discriminator(preds)
        
        adversarial_loss = 0
        content_loss = 0
        pixel_loss = pixel_criterion(preds, hr.detach())
        content_loss = content_criterion(preds, hr.detach())
        # identity_loss = identity_criterion(preds,hr.detach())
        
        for fake in fake_pred:
            adversarial_loss += adversarial_criterion(fake, True)
        adversarial_loss /= len(real_pred)

        g_loss = pixel_loss + content_loss + adversarial_loss
        
        loss_dict["pixel_loss"] = pixel_loss
        loss_dict["content_loss"] = content_loss
        loss_dict["adversarial_loss"] = adversarial_loss
        loss_dict["g_loss"] = g_loss
        
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
        g_scheduler.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss = loss_reduced["d_loss"].mean().item()
        pixel_loss = loss_reduced["pixel_loss"].mean().item()
        content_loss = loss_reduced["content_loss"].mean().item()
        adversarial_loss = loss_reduced["adversarial_loss"].mean().item()
        g_loss = loss_reduced["g_loss"].mean().item()

        if gpu == 0:
            results = torch.cat((hr.detach(), F.interpolate(lr, scale_factor=cfg.training.dataset.scale, mode="nearest").detach(), preds.detach()), 2)
            vutils.save_image(results, os.path.join(cfg.training.GAN.ckpt_dir, f"preds.jpg"))

            if wandb and cfg.training.common.use_wandb:
                print("hellllll")
                wandb.log(
                    {
                        "d_loss": d_loss,
                        "pixel_loss": pixel_loss,
                        "content_loss": content_loss,
                        "adversarial_loss": adversarial_loss,
                        "g_loss": g_loss,
                        "GT": [wandb.Image(hr, caption="step{}_Label".format(i))],
                        "Preds": [wandb.Image(preds, caption="step{}_Label".format(i))],
                        "LQ":[wandb.Image(lr, caption="step{}_Label".format(i))]
                    }
                )

            if i % 10000 == 0:
                if cfg.training.ddp.distributed:
                    torch.save(
                        {
                            "g": generator.module.state_dict(),
                            "d": discriminator.module.state_dict(),
                            "g_module": generator.state_dict(),
                            "d_module": discriminator.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "iteration": i,
                            "best_psnr": best_psnr,
                            "best_lpips": best_lpips
                        },
                        f"{cfg.training.GAN.ckpt_dir}/{str(i).zfill(6)}.pth",
                    )
                else:
                    torch.save(
                        {
                            "g": generator.state_dict(),
                            "d": discriminator.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "iteration": i,
                            "best_psnr": best_psnr,
                            "best_lpips": best_lpips
                        },
                        f"{cfg.training.GAN.ckpt_dir}/{str(i).zfill(6)}.pth",
                    )

def psnr_trainer(cfg, gpu, loader, generator):
    """ weights를 저장 할 경로 설정 """
    if not os.path.exists(cfg.training.PSNR.ckpt_dir):
        os.makedirs(cfg.training.PSNR.ckpt_dir)

    generator.train()

    loss_dict = {}
    start_iters = 0
    best_psnr = 0

    psnr_loss = nn.L1Loss().to(gpu)
    psnr_optim = torch.optim.Adam(generator.parameters(), cfg.training.PSNR.lr, (0.9, 0.99))

    if os.path.exists(cfg.training.PSNR.resume):
        ckpt = torch.load(cfg.training.PSNR.resume, map_location=lambda storage, loc: storage)
        if cfg.training.ddp.distributed:
            generator.load_state_dict(ckpt["g_module"])
        else:
            generator.load_state_dict(ckpt["g"])
        start_iters = ckpt["iteration"] + 1
        psnr_optim.load_state_dict(ckpt["g_optim"])
        best_psnr = ckpt["best_psnr"]

    if start_iters > 0:
        psnr_scheduler = torch.optim.lr_scheduler.MultiStepLR(psnr_optim, [400000, 800000, 900000], 0.5, last_epoch=start_iters)
    else:
        psnr_scheduler = torch.optim.lr_scheduler.MultiStepLR(psnr_optim, [400000, 800000, 900000], 0.5)

    for i in range(start_iters, cfg.training.PSNR.n_iters):        
        lr, hr = next(loader)
        lr = lr.to(gpu)
        hr = hr.to(gpu)

        preds = generator(lr)
        loss = psnr_loss(preds, hr)
        loss_dict["l1loss"] = loss
        
        generator.zero_grad()
        loss.backward()
        psnr_optim.step()
        psnr_scheduler.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        l1loss = loss_reduced["l1loss"].mean().item()

        if gpu == 0:
            if cfg.training.dataset.yuv:
                hr = yuv2rgb(hr.detach(), gpu)
                lr = yuv2rgb(F.interpolate(lr, scale_factor=cfg.training.dataset.scale, mode="nearest").detach(), gpu)
                preds = yuv2rgb(preds.detach(), gpu)
            else:
                hr = hr.detach()
                lr = F.interpolate(lr, scale_factor=cfg.training.dataset.scale, mode="nearest").detach()
                preds = preds.detach()

            results = torch.cat((hr, lr, preds), 2)
            vutils.save_image(results, os.path.join(cfg.training.PSNR.ckpt_dir, f"preds.jpg"))

            if wandb and cfg.training.common.use_wandb:
                print("hellllll")
                wandb.log(
                    {
                        "d_loss": d_loss,
                        "pixel_loss": pixel_loss,
                        "content_loss": content_loss,
                        "adversarial_loss": adversarial_loss,
                        "g_loss": g_loss,
                        "GT": [wandb.Image(hr, caption="step{}_Label".format(i))],
                        "Preds": [wandb.Image(preds, caption="step{}_Label".format(i))],
                        "LQ":[wandb.Image(lr, caption="step{}_Label".format(i))]
                    }
                )

            if i % 20000 == 0:
                if cfg.training.ddp.distributed:
                    torch.save(
                        {
                            "g": generator.module.state_dict(),
                            "g_module": generator.state_dict(),
                            "g_optim": psnr_optim.state_dict(),
                            "iteration": i,
                            "best_psnr": best_psnr
                        },
                        f"{cfg.training.PSNR.ckpt_dir}/{str(i).zfill(6)}.pth",
                    )
                else:
                    torch.save(
                        {
                            "g": generator.state_dict(),
                            "g_optim": psnr_optim.state_dict(),
                            "iteration": i,
                            "best_psnr": best_psnr
                        },
                        f"{cfg.training.PSNR.ckpt_dir}/{str(i).zfill(6)}.pth",
                    )

def workers(gpu, cfg):
    if gpu == 0:
        """ Wandb 사용여부 설정 """
        if cfg.training.common.use_wandb:
            wandb.init(project=cfg.training.common.project)
            wandb.config.update(cfg)

    if cfg.training.ddp.distributed:
        cfg.training.ddp.rank = cfg.training.ddp.nr * cfg.training.ddp.gpus + gpu
        setup(cfg.training.ddp.rank, cfg.training.ddp.world_size)
        
    dataset = Dataset(cfg.training.dataset)
    train_sampler = None

    generator = Generator(cfg.training.models.generator).to(gpu)
    if cfg.training.GAN.discriminator == "PatchGAN":
        discriminator = Discriminator_PatchGAN(cfg.training.models.patchgan_discriminator).to(gpu)
    elif cfg.training.GAN.discriminator == "Unet":    
        discriminator = MultiscaleDiscriminator(cfg.training.models.unet_discriminator).to(gpu)
    else:
        raise ValueError("Wrong discriminator setting in configs")

    """ DDP setting """
    if cfg.training.ddp.distributed:
        generator = DDP(generator, device_ids=[gpu], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[gpu], find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=cfg.training.ddp.world_size, rank=cfg.training.ddp.rank
        )
    dataloader = DataLoader(
                dataset=dataset,
                batch_size=cfg.training.dataset.batch_size,
                shuffle=(train_sampler is None),
                num_workers=cfg.training.common.num_workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True
            )
    loader = sample_data(dataloader)
    psnr_trainer(cfg, gpu, loader, generator)
    gan_trainer(cfg, gpu, loader, generator, discriminator)
    
    cleanup()

# NET : GAN 767667 GPU 6

@hydra.main(config_path="./configs", config_name="train.yaml")
def main(cfg):
    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    cudnn.deterministic = True

    """ Torch Seed 설정 """
    torch.manual_seed(cfg.training.common.seed)
 
    """ CPU worker 설정 """
    cfg.training.common.num_workers = 4 * torch.cuda.device_count()

    if torch.cuda.device_count() > 1:
        print("Train with multiple GPUs")
        cfg.training.ddp.distributed = True
        gpus = torch.cuda.device_count()
        cfg.training.common.num_workers = gpus * 4
        cfg.training.ddp.world_size = gpus * cfg.training.ddp.nodes
        mp.spawn(workers, nprocs=gpus, args=(cfg,))
    else:
        print("Train with single GPUs")
        cfg.training.ddp.distributed = False
        workers(0, cfg)


if __name__ == "__main__":
    main()