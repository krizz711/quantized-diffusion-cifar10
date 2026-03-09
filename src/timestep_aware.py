# src/timestep_aware.py
"""
Timestep-Aware Mixed Precision experiment stage.

Loads PTQ checkpoints, applies timestep-aware quantization with
different layer strategies, evaluates, and saves results.
"""

import gc
import json
import os
from copy import deepcopy

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from src.config import config, device, OUTPUTS_DIR
from src.model import create_model
from src.quantization import apply_timestep_aware
from src.train import (
    measure_latency_vram, compute_fid_cifar10,
    scheduler_ddim,
)


def ddim_sampler_ta(initial_xt, model):
    """DDIM sampler for timestep-aware models."""
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


if __name__ == '__main__':
    print("\n=== Timestep-Aware Mixed Precision Experiments ===\n")

    # Load baseline metrics for model size reference
    with open(os.path.join(OUTPUTS_DIR, 'baseline_metrics.json'), 'r') as f:
        baseline_metrics = json.load(f)

    # Layer-wise application strategies
    layer_conditions = {
        'all': lambda name, module: True,
        'mid_only': lambda name, module: 'mid' in name,
        'down_only': lambda name, module: 'down' in name,
    }

    # Base PTQ model configs
    base_configs = [
        {'name': 'PTQ_8bit', 'file': 'ptq_8bit.pth', 'bits': 8},
        {'name': 'PTQ_4bit', 'file': 'ptq_4bit.pth', 'bits': 4},
        {'name': 'PTQ_1bit', 'file': 'ptq_1bit.pth', 'bits': 1},
    ]

    ta_results = []

    for base in base_configs:
        model_file = os.path.join(OUTPUTS_DIR, base['file'])
        if not os.path.exists(model_file):
            print(f"⚠️  {base['file']} not found – skipping {base['name']}")
            continue

        print(f"\n--- Base model: {base['name']} ---")
        checkpoint = torch.load(model_file, map_location='cpu')
        model_base = create_model()
        model_base.load_state_dict(checkpoint['model_state_dict'])
        model_base = model_base.to(device)
        model_base.eval()

        for cond_name, cond_func in layer_conditions.items():
            print(f"   Applying timestep-aware to layers: {cond_name}")
            model_ta = deepcopy(model_base)
            model_ta = apply_timestep_aware(model_ta, condition=cond_func)

            # Measure latency and VRAM
            sampler_fn = lambda noise, m=model_ta: ddim_sampler_ta(noise, m)
            avg_ms, peak_mb = measure_latency_vram(model_ta, sampler_fn)

            # Compute FID
            print("      Computing FID...")
            fid_val = compute_fid_cifar10(
                sampler_func=lambda bs, m=model_ta: ddim_sampler_ta(
                    torch.randn(
                        bs, config['channels'],
                        config['image_size'], config['image_size']
                    ).to(device),
                    m
                ),
                num_samples=config['fid_num_samples'],
                batch_size=config['fid_batch_size']
            )

            base_size = base['bits'] / 32 * baseline_metrics['model_size_mb']

            ta_results.append({
                'base_model': base['name'],
                'layers': cond_name,
                'fid': fid_val,
                'ms_per_img': avg_ms,
                'vram_mb': peak_mb,
                'model_size_mb': base_size,
            })

            print(f"      FID: {fid_val:.2f}, Speed: {avg_ms:.2f} ms/img, "
                  f"VRAM: {peak_mb:.2f} MB")

            del model_ta
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        del model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save results
    if ta_results:
        print("\n=== Timestep-Aware Experiment Summary ===")
        df_ta = pd.DataFrame(ta_results)
        print(df_ta.to_string(index=False))
        csv_path = os.path.join(OUTPUTS_DIR, 'timestep_aware_results.csv')
        df_ta.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        json_path = os.path.join(OUTPUTS_DIR, 'timestep_aware_results.json')
        with open(json_path, 'w') as f:
            json.dump(ta_results, f, indent=2)
    else:
        print("No timestep-aware experiments completed.")
