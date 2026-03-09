# src/train.py
"""
Baseline training, sampling, evaluation helpers, and baseline metrics stage.

DVC stage entry point: trains (if not pretrained), evaluates baseline,
saves baseline_metrics.json and baseline_model.pth to outputs/.
"""

import json
import os
import time
import tempfile

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, DDIMScheduler
from cleanfid import fid
from torchmetrics.image.inception import InceptionScore

from src.config import config, device, USE_PRETRAINED, OUTPUTS_DIR, DATA_DIR
from src.data import get_dataloader, get_dataset
from src.model import create_model


# ===================== Schedulers =====================

scheduler_ddpm = DDPMScheduler(
    num_train_timesteps=config['train_timesteps'],
    beta_schedule='linear'
)
scheduler_ddim = DDIMScheduler.from_config(scheduler_ddpm.config)


# ===================== Training =====================

def train_one_epoch(model, optimizer, dataloader, epoch):
    model.train()
    total_loss = 0
    for x0, _ in tqdm(dataloader, leave=False):
        x0 = x0.to(device)
        bs = x0.shape[0]
        t = torch.randint(0, config['train_timesteps'], (bs,), device=device).long()
        noise = torch.randn_like(x0)
        xt = scheduler_ddpm.add_noise(x0, noise, t)
        pred = model(xt, t).sample
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ===================== Sampling =====================

def sample_images(model, num_images=64):
    model.eval()
    scheduler = scheduler_ddim
    scheduler.set_timesteps(config['sampling_timesteps'])
    noise = torch.randn(
        num_images, config['channels'],
        config['image_size'], config['image_size']
    ).to(device)
    xt = noise
    for t_val in scheduler.timesteps:
        with torch.no_grad():
            t_on_device = t_val.repeat(num_images).long().to(device)
            pred = model(xt, t_on_device).sample
            xt = scheduler.step(pred, t_val, xt).prev_sample
    return (xt * 0.5 + 0.5).cpu()


def ddim_sampler_for_eval(model, initial_xt):
    model.eval()
    scheduler = scheduler_ddim
    scheduler.set_timesteps(config['sampling_timesteps'])
    xt = initial_xt
    for t_val in scheduler.timesteps:
        with torch.no_grad():
            t_on_device = t_val.repeat(xt.shape[0]).long().to(device)
            pred = model(xt, t_on_device).sample
            xt = scheduler.step(pred, t_val, xt).prev_sample
    return (xt * 0.5 + 0.5).cpu()


# ===================== Evaluation Helpers =====================

def measure_latency_vram(model, sampler_func, num_images=32, batch_size=8):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            bs = min(batch_size, num_images - i)
            noise = torch.randn(
                bs, config['channels'],
                config['image_size'], config['image_size']
            ).to(device)
            start = time.perf_counter()
            _ = sampler_func(noise)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
    total_time = sum(timings)
    avg_ms_per_img = total_time / num_images * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
    return avg_ms_per_img, peak_mb


def compute_fid_cifar10(sampler_func, num_samples=5000, batch_size=64, use_temp=True):
    """Compute FID using clean-fid between real CIFAR-10 and generated images."""
    if use_temp:
        gen_folder = tempfile.TemporaryDirectory()
        gen_path = gen_folder.name
    else:
        gen_path = './fid_temp'
        os.makedirs(gen_path, exist_ok=True)

    generated = 0
    pbar = tqdm(total=num_samples, desc="Generating for FID")
    try:
        while generated < num_samples:
            bs = min(batch_size, num_samples - generated)
            images = sampler_func(bs)
            if isinstance(images, torch.Tensor):
                images = images.cpu()
            else:
                images = torch.tensor(images)
            if images.dim() == 3:
                images = images.unsqueeze(0)
            for i, img in enumerate(images):
                idx = generated + i
                save_path = os.path.join(gen_path, f"{idx:05d}.png")
                torchvision.utils.save_image(img, save_path)
            generated += bs
            pbar.update(bs)
        pbar.close()

        saved_files = os.listdir(gen_path)
        print(f"Saved {len(saved_files)} images in {gen_path}")
        if len(saved_files) == 0:
            raise RuntimeError("No images were saved – check sampler_func output.")

        # Prepare real CIFAR-10 images folder
        real_path = os.path.join(DATA_DIR, 'fid_real_cifar10')
        if not os.path.exists(real_path) or len(os.listdir(real_path)) < 50000:
            os.makedirs(real_path, exist_ok=True)
            print("Preparing real CIFAR-10 images...")
            real_ds = torchvision.datasets.CIFAR10(
                root=DATA_DIR, train=True, download=True,
                transform=transforms.ToTensor()
            )
            for idx, (img, _) in enumerate(tqdm(real_ds, desc="Saving real images")):
                torchvision.utils.save_image(
                    img, os.path.join(real_path, f"{idx:05d}.png")
                )

        score = fid.compute_fid(fdir1=real_path, fdir2=gen_path, mode="clean")
    finally:
        if use_temp:
            gen_folder.cleanup()

    return score


def compute_is_cifar10(sampler_func, num_samples=5000, batch_size=64):
    inception = InceptionScore().to(device)
    generated = 0
    pbar = tqdm(total=num_samples, desc="Computing IS")
    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        images = sampler_func(bs)
        images_uint8 = (images * 255).byte().to(device)
        inception.update(images_uint8)
        generated += bs
        pbar.update(bs)
    pbar.close()
    mean_is, std_is = inception.compute()
    return mean_is.item(), std_is.item()


# ===================== Stage Entry Point =====================

if __name__ == '__main__':
    print(f"Using device: {device}")

    # Pre-cache clean-fid reference statistics
    _ = fid.get_reference_statistics(name='cifar10', res=32, mode='clean')

    model = create_model()

    if not USE_PRETRAINED:
        dataloader = get_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
        start_epoch = 0
        checkpoint_path = os.path.join(OUTPUTS_DIR, 'baseline_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_lr.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, config['epochs']):
            loss = train_one_epoch(model, optimizer, dataloader, epoch)
            print(f"Epoch {epoch + 1}/{config['epochs']}, loss: {loss:.4f}")
            scheduler_lr.step()
            if (epoch + 1) % 50 == 0:
                samples = sample_images(model, num_images=16)
                grid = torchvision.utils.make_grid(samples, nrow=4)
                plt.imshow(grid.permute(1, 2, 0).squeeze())
                plt.axis('off')
                plt.title(f"Baseline Epoch {epoch + 1}")
                plt.savefig(os.path.join(OUTPUTS_DIR, f'baseline_epoch_{epoch + 1}.png'))
                plt.close()
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler_lr.state_dict(),
            }, checkpoint_path)

    # Save baseline model
    model_path = os.path.join(OUTPUTS_DIR, 'baseline_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Baseline model saved to {model_path}")

    # Baseline evaluation
    print("Evaluating baseline...")
    sampler_wrapper = lambda noise: ddim_sampler_for_eval(model, noise)
    avg_ms, peak_mb = measure_latency_vram(model, sampler_wrapper)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2

    print("Computing FID for baseline...")
    fid_score = compute_fid_cifar10(
        sampler_func=lambda bs: sample_images(model, bs),
        num_samples=config['fid_num_samples'],
        batch_size=config['fid_batch_size']
    )
    print(f"Baseline FID: {fid_score:.2f}")

    baseline_metrics = {
        'fid': fid_score,
        'ms_per_img': avg_ms,
        'vram_mb': peak_mb,
        'model_size_mb': model_size,
    }
    print(f"Baseline - Speed: {avg_ms:.2f} ms/img, VRAM: {peak_mb:.2f} MB, "
          f"Model size: {model_size:.2f} MB")

    metrics_path = os.path.join(OUTPUTS_DIR, 'baseline_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
