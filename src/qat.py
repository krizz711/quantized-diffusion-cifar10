# src/qat.py
"""
Quantization-Aware Training (QAT) experiment stage.

Loads the baseline model, applies QAT with 1-bit weights,
fine-tunes, evaluates, and saves results to outputs/.
"""

import gc
import json
import os
from copy import deepcopy

import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import config, device, OUTPUTS_DIR
from src.model import create_model
from src.quantization import apply_qat, condition_quant
from src.train import (
    sample_images, ddim_sampler_for_eval,
    measure_latency_vram, compute_fid_cifar10,
)


if __name__ == '__main__':
    print("\n=== QAT for 1-bit weights ===")

    # Load baseline
    baseline_path = os.path.join(OUTPUTS_DIR, 'baseline_model.pth')
    model_qat = create_model()
    model_qat.load_state_dict(torch.load(baseline_path, map_location='cpu'))
    model_qat = model_qat.to(device)

    # Apply QAT
    model_qat = apply_qat(
        model_qat,
        weight_bits=1,
        act_bits=32,
        condition=condition_quant,
        epochs=config['qat_epochs'],
        lr=config['qat_lr'],
    )

    # Visual check
    samples = sample_images(model_qat, 16)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.title("QAT 1-bit weights")
    plt.savefig(os.path.join(OUTPUTS_DIR, 'qat_1bit_samples.png'))
    plt.close()

    # Efficiency metrics
    sampler_wrapper_qat = lambda noise: ddim_sampler_for_eval(model_qat, noise)
    avg_ms, peak_mb = measure_latency_vram(model_qat, sampler_wrapper_qat)
    model_size = sum(p.numel() for p in model_qat.parameters()) * (1 / 32) / 1024 ** 2

    # FID
    print("Computing FID for QAT 1-bit...")
    fid_val = compute_fid_cifar10(
        sampler_func=lambda bs: sample_images(model_qat, bs),
        num_samples=config['fid_num_samples'],
        batch_size=config['fid_batch_size']
    )
    print(f"QAT 1-bit FID: {fid_val:.2f}")

    qat_results = {
        '1bit_qat': {
            'fid': fid_val,
            'ms_per_img': avg_ms,
            'vram_mb': peak_mb,
            'model_size_mb': model_size,
        }
    }
    print(f"QAT 1-bit: {avg_ms:.2f} ms/img, {peak_mb:.2f} MB VRAM, "
          f"{model_size:.2f} MB model")

    torch.save({
        'model_state_dict': model_qat.state_dict(),
        'weight_bits': 1,
        'act_bits': 32,
        'quant_layers': config['quant_layers'],
        'epochs': config['qat_epochs'],
    }, os.path.join(OUTPUTS_DIR, 'qat_1bit.pth'))

    # Save results
    results_path = os.path.join(OUTPUTS_DIR, 'qat_results.json')
    with open(results_path, 'w') as f:
        json.dump(qat_results, f, indent=2)
    print(f"QAT results saved to {results_path}")

    # Cleanup
    del model_qat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
