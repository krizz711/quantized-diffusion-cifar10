# src/evaluate.py
"""
Final evaluation stage: aggregate all experiment results into a comparison table.

Reads baseline_metrics.json, ptq_results.json, qat_results.json, and
timestep_aware_results.json, merges them, and saves the final CSV.
"""

import json
import os

import pandas as pd

from src.config import OUTPUTS_DIR


def load_json(filename):
    """Load a JSON file from outputs/, return None if not found."""
    path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    print(f"⚠️  {filename} not found – skipping.")
    return None


if __name__ == '__main__':
    print("\n=== Aggregating Final Results ===\n")

    results = {}

    # Baseline
    baseline = load_json('baseline_metrics.json')
    if baseline:
        results['Baseline (FP32)'] = baseline

    # PTQ
    ptq = load_json('ptq_results.json')
    if ptq:
        for k, v in ptq.items():
            results[f'PTQ {k}'] = v

    # QAT
    qat = load_json('qat_results.json')
    if qat:
        for k, v in qat.items():
            results[f'QAT {k}'] = v

    # Timestep-Aware (take the best per base model)
    ta = load_json('timestep_aware_results.json')
    if ta:
        for entry in ta:
            key = f"TA {entry['base_model']}_{entry['layers']}"
            results[key] = {
                'fid': entry['fid'],
                'ms_per_img': entry['ms_per_img'],
                'vram_mb': entry['vram_mb'],
                'model_size_mb': entry['model_size_mb'],
            }

    if results:
        df = pd.DataFrame(results).T
        cols = ['fid', 'ms_per_img', 'vram_mb', 'model_size_mb']
        available_cols = [c for c in cols if c in df.columns]
        print("=== Final Comparison Table ===")
        print(df[available_cols].round(2).to_string())

        csv_path = os.path.join(OUTPUTS_DIR, 'quantization_final_results.csv')
        df.to_csv(csv_path)
        print(f"\nResults saved to {csv_path}")
    else:
        print("No results found to aggregate.")
