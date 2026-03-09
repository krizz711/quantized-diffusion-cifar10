# src/ptq.py
"""
Post-Training Quantization (PTQ) experiment stage.

Loads the baseline model, applies PTQ at 8/4/1 bit widths,
evaluates each, and saves results to outputs/.
"""

import json
import os

import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import config, device, OUTPUTS_DIR
from src.model import create_model
from src.quantization import apply_ptq, condition_quant
from src.train import (
    sample_images, ddim_sampler_for_eval,
    measure_latency_vram, compute_fid_cifar10,
)


if __name__ == '__main__':
    # Load baseline model
    baseline_path = os.path.join(OUTPUTS_DIR, 'baseline_model.pth')
    base_model = create_model()
    base_model.load_state_dict(torch.load(baseline_path, map_location=device))
    base_model.eval()
    print("Loaded baseline model.")

    ptq_results = {}

    for bits in [8, 4, 1]:
        print(f"\n=== PTQ with {bits}-bit weights (activation FP32) ===")
        model_ptq = create_model()
        model_ptq.load_state_dict(base_model.state_dict())
        model_ptq = apply_ptq(
            model_ptq, weight_bits=bits, act_bits=32, condition=condition_quant
        )

        # Visual check
        samples = sample_images(model_ptq, 16)
        grid = torchvision.utils.make_grid(samples, nrow=4)
        plt.imshow(grid.permute(1, 2, 0).squeeze())
        plt.axis('off')
        plt.title(f"PTQ {bits}-bit weights")
        plt.savefig(os.path.join(OUTPUTS_DIR, f'ptq_{bits}bit_samples.png'))
        plt.close()

        # Efficiency metrics
        sampler_wrapper_ptq = lambda noise, m=model_ptq: ddim_sampler_for_eval(m, noise)
        avg_ms, peak_mb = measure_latency_vram(model_ptq, sampler_wrapper_ptq)

        # Load baseline metrics for model_size_mb reference
        with open(os.path.join(OUTPUTS_DIR, 'baseline_metrics.json'), 'r') as f:
            baseline_metrics = json.load(f)
        model_size = sum(p.numel() for p in model_ptq.parameters()) * (bits / 32) / 1024 ** 2

        # FID
        print(f"Computing FID for PTQ {bits}-bit...")
        fid_val = compute_fid_cifar10(
            sampler_func=lambda bs, m=model_ptq: sample_images(m, bs),
            num_samples=config['fid_num_samples'],
            batch_size=config['fid_batch_size']
        )
        print(f"PTQ {bits}-bit FID: {fid_val:.2f}")

        ptq_results[f'{bits}bit'] = {
            'fid': fid_val,
            'ms_per_img': avg_ms,
            'vram_mb': peak_mb,
            'model_size_mb': model_size,
        }
        print(f"PTQ {bits}-bit: {avg_ms:.2f} ms/img, {peak_mb:.2f} MB VRAM, "
              f"{model_size:.2f} MB model")

        torch.save({
            'model_state_dict': model_ptq.state_dict(),
            'weight_bits': bits,
            'act_bits': 32,
            'quant_layers': config['quant_layers'],
        }, os.path.join(OUTPUTS_DIR, f'ptq_{bits}bit.pth'))

    # Save results
    results_path = os.path.join(OUTPUTS_DIR, 'ptq_results.json')
    with open(results_path, 'w') as f:
        json.dump(ptq_results, f, indent=2)
    print(f"\nPTQ results saved to {results_path}")
